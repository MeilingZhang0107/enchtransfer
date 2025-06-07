import json
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, MutableSequence, Optional, Sequence, Tuple

import h5py
import jsonlines
import nltk
import numpy as np
from sklearn.model_selection import StratifiedKFold

import config

CLS_TOKEN_INDEX = 0


def pickle_loader(filename: str) -> Any:
    with open(filename, "rb") as file:
        return pickle.load(file, encoding="latin1")


class DataLoader:
    DATA_PATH = "/scratch/s5511038/mustard_reproduce/MUStARD/data/joint/sarcasm_data.json"
    AUDIO_PICKLE = "/scratch/s5511038/mustard_reproduce/MUStARD/data/joint/joint_audio_features.p"
    INDICES_FILE = "/scratch/s5511038/mustard_reproduce/MUStARD/data/split_indices.p"
    GLOVE_DICT = "/scratch/s5511038/mustard_reproduce/MUStARD/data/glove_full_dict.p"
    BERT_TARGET_EMBEDDINGS = "/scratch/s5511038/mustard_reproduce/MUStARD/data/joint/joint_bert_output.jsonl"
    BERT_CONTEXT_EMBEDDINGS = "/scratch/s5511038/mustard_reproduce/MUStARD/data/joint/joint_bert_output.jsonl"
    UTT_ID = 0
    CONTEXT_ID = 2
    SHOW_ID = 9
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    def __init__(self, config: config.Config) -> None:
        self.config = config

        with open(self.DATA_PATH) as file:
            dataset_dict = json.load(file)

        if config.use_bert and config.use_target_text:
            text_bert_embeddings = []
            with jsonlines.open(self.BERT_TARGET_EMBEDDINGS) as utterances:
                for utterance in utterances:
                    features = utterance["features"][CLS_TOKEN_INDEX]
                    bert_embedding_target = np.mean([np.array(features["layers"][layer]["values"])
                                                     for layer in range(4)], axis=0)
                    text_bert_embeddings.append(np.copy(bert_embedding_target))
        else:
            text_bert_embeddings = None

        context_bert_embeddings = self.load_context_bert(dataset_dict) if config.use_context else None
        audio_features = pickle_loader(self.AUDIO_PICKLE) if config.use_target_audio else None

        if config.use_target_video:
            video_features_file = h5py.File("/scratch/s5511038/mustard_reproduce/MUStARD/data/joint/joint_resnet_pool5.hdf5")
            context_video_features_file = h5py.File("/scratch/s5511038/mustard_reproduce/MUStARD/data/joint/joint_resnet_pool5.hdf5")
        else:
            video_features_file = None
            context_video_features_file = None

        self.data_input = []
        self.data_output = []

        self.word_emb_dict = {}

        self.train_ind_si = []
        self.test_ind_si = []

        self.parse_data(dataset_dict, audio_features, video_features_file, context_video_features_file,
                        text_bert_embeddings, context_bert_embeddings)

        if config.use_target_video:
            video_features_file.close()
            context_video_features_file.close()

        self.split_indices = None
        self.stratified_k_fold()

        self.setup_glove_dict()

        self.speaker_independent_split()

    def parse_data(self, dataset_dict: Mapping[str, Mapping[str, Any]],
                   audio_features: Optional[Mapping[str, Any]] = None,
                   video_features_file: Optional[Mapping[str, Any]] = None,
                   context_video_features_file: Optional[Mapping[str, Any]] = None,
                   text_embeddings: Optional[Mapping[int, Any]] = None,
                   context_embeddings: Optional[Mapping[int, Any]] = None) -> None:
        """Converts the dictionary data into lists."""
        for idx, id_ in enumerate(dataset_dict):
            self.data_input.append((dataset_dict[id_]["utterance"], dataset_dict[id_]["speaker"],
                                    dataset_dict[id_]["context"], dataset_dict[id_]["context_speakers"],
                                    self._fix_audio(audio_features.get(id_)) if audio_features else None,
                                    video_features_file[id_][()] if video_features_file else None,
                                    context_video_features_file[id_][()] if context_video_features_file else None,
                                    text_embeddings[idx] if text_embeddings else None,
                                    context_embeddings[idx] if context_embeddings else None,
                                    dataset_dict[id_]["show"]))
            self.data_output.append(int(dataset_dict[id_]["sarcasm"]))

    def load_context_bert(self, dataset: Mapping[str, Mapping[str, Any]]) -> Iterable[Iterable[np.ndarray]]:
        # Get the number of context utterances for each entry
        length = [len(dataset[id_]["context"]) for id_ in dataset]

        with jsonlines.open(self.BERT_CONTEXT_EMBEDDINGS) as utterances:
            context_utterance_embeddings = []
            for utterance in utterances:
                features = utterance["features"][CLS_TOKEN_INDEX]
                bert_embedding_target = np.mean([np.array(features["layers"][layer]["values"])
                                                 for layer in [0, 1, 2, 3]], axis=0)
                context_utterance_embeddings.append(np.copy(bert_embedding_target))

        assert len(context_utterance_embeddings) == sum(length)

        cumulative_length = [length[0]]
        cumulative_value = length[0]
        for val in length[1:]:
            cumulative_value += val
            cumulative_length.append(cumulative_value)

        assert len(length) == len(cumulative_length)

        end_index = cumulative_length
        start_index = [0] + cumulative_length[:-1]

        return [[context_utterance_embeddings[idx] for idx in range(start, end)]
                for start, end in zip(start_index, end_index)]

    @staticmethod
    def _fix_audio(feat):
        # Always return a 128-dim audio feature
        return feat if isinstance(feat, np.ndarray) and feat.shape == (128,) \
           else np.zeros(128, dtype=np.float32)

    def stratified_k_fold(self, splits: int = 5) -> None:
        """Prepares or loads splits for K-fold cross-validation."""
        cross_validator = StratifiedKFold(n_splits=splits, shuffle=True)
        split_indices = [(train_index, test_index)
                         for train_index, test_index in cross_validator.split(self.data_input, self.data_output)]

        if not os.path.exists(self.INDICES_FILE):
            with open(self.INDICES_FILE, "wb") as file:
                pickle.dump(split_indices, file, protocol=2)

    def get_stratified_k_fold(self) -> Iterable[Tuple[Iterable[int], Iterable[int]]]:
        """Returns train/test indices for k-fold cross-validation."""
        self.split_indices = pickle_loader(self.INDICES_FILE)
        return self.split_indices

    def speaker_independent_split(self) -> None:
        for i, data in enumerate(self.data_input):
            if data[self.SHOW_ID] == "FRIENDS":
                self.test_ind_si.append(i)
            else:
                self.train_ind_si.append(i)

    def get_speaker_independent(self) -> Tuple[Iterable[int], Iterable[int]]:
        """Returns the train/test indices of the speaker-independent setting."""
        return self.train_ind_si, self.test_ind_si

    def get_split(self, indices: Iterable[int]) -> Tuple[Sequence[Tuple[Any, ...]], Sequence[int]]:
        """Returns the split based on indices."""
        data_input = [self.data_input[i] for i in indices]
        data_output = [self.data_output[i] for i in indices]
        return data_input, data_output

    def full_dataset_vocab(self) -> Mapping[str, int]:
        """Returns the vocabulary for the whole dataset for GloVe filtering."""
        vocab = defaultdict(int)
        utterances = [instance[self.UTT_ID] for instance in self.data_input]
        contexts = [instance[self.CONTEXT_ID] for instance in self.data_input]

        for utterance in utterances:
            clean_utt = DataHelper.clean_str(utterance)
            utt_words = nltk.word_tokenize(clean_utt)
            for word in utt_words:
                vocab[word.lower()] += 1

        for context in contexts:
            for c_utt in context:
                clean_utt = DataHelper.clean_str(c_utt)
                utt_words = nltk.word_tokenize(clean_utt)
                for word in utt_words:
                    vocab[word.lower()] += 1

        return vocab

    def setup_glove_dict(self) -> None:
        """
        Cache the GloVe dictionary based on the dataset vocabulary.
        """
        assert self.config.word_embedding_path is not None

        vocab = self.full_dataset_vocab()

        if os.path.exists(self.GLOVE_DICT):
            self.word_emb_dict = pickle_loader(self.GLOVE_DICT)
        else:
            self.word_emb_dict = {}
            with open(self.config.word_embedding_path) as file:
                for line in file:
                    split_line = line.split()
                    word = split_line[0]
                    try:
                        embedding = np.array([float(val) for val in split_line[1:]])
                        if word.lower() in vocab:
                            self.word_emb_dict[word.lower()] = embedding
                    except ValueError:
                        print("Error word in glove file (skipped): ", word)
                        continue
            self.word_emb_dict[self.PAD_TOKEN] = np.zeros(self.config.embedding_dim)
            self.word_emb_dict[self.UNK_TOKEN] = np.random.uniform(-0.25, 0.25, self.config.embedding_dim)

            with open(self.GLOVE_DICT, "wb") as file:
                pickle.dump(self.word_emb_dict, file)


class DataHelper:
    UTT_ID = 0
    SPEAKER_ID = 1
    CONTEXT_ID = 2
    CONTEXT_SPEAKERS_ID = 3
    TARGET_AUDIO_ID = 4
    TARGET_VIDEO_ID = 5
    CONTEXT_VIDEO_ID = 6
    TEXT_BERT_ID = 7
    CONTEXT_BERT_ID = 8

    PAD_ID = 0
    UNK_ID = 1
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    GLOVE_MODELS = "data/temp/glove_dict_{}.p"
    GLOVE_MODELS_CONTEXT = "data/temp/glove_dict_context_{}.p"

    def __init__(self, train_input: Sequence[Tuple[Any, ...]], train_output: Sequence[int],
                 test_input: Sequence[Tuple[Any, ...]], test_output: Sequence[int], config: config.Config,
                 data_loader: DataLoader) -> None:
        self.data_loader = data_loader
        self.config = config
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output

        self.author_to_index = None
        self.UNK_AUTHOR_ID = None
        self.audio_max_length = None

        self.vocab = defaultdict(int)
        self.create_vocab(config.use_context)
        print("Vocab size:", len(self.vocab))

        self.model = {}
        self.embed_dim = -1
        self.load_glove_model_for_current_split(config.use_context)

        self.word_idx_map = {}
        self.W = np.empty(0)
        self.create_embedding_matrix()

    @staticmethod
    def clean_str(s: str) -> str:
        """Clean a string for tokenization."""
        s = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", s)
        s = re.sub(r"\'s", " \'s", s)
        s = re.sub(r"\'ve", " \'ve", s)
        s = re.sub(r"n\'t", " n\'t", s)
        s = re.sub(r"\'re", " \'re", s)
        s = re.sub(r"\'d", " \'d", s)
        s = re.sub(r"\'ll", " \'ll", s)
        s = re.sub(r",", " , ", s)
        s = re.sub(r"!", " ! ", s)
        s = re.sub(r"\"", " \" ", s)
        s = re.sub(r"\(", " ( ", s)
        s = re.sub(r"\)", " ) ", s)
        s = re.sub(r"\?", " ? ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"\.", " . ", s)
        s = re.sub(r"., ", " , ", s)
        s = re.sub(r"\\n", " ", s)
        return s.strip().lower()

    def get_data(self, id_: int, mode: str) -> MutableSequence[Any]:
        if mode == "train":
            return [instance[id_] for instance in self.train_input]
        elif mode == "test":
            return [instance[id_] for instance in self.test_input]
        else:
            raise ValueError(f"Unrecognized mode: {mode}")

    def create_vocab(self, use_context: bool = False) -> None:
        utterances = self.get_data(self.UTT_ID, mode="train")
        for utterance in utterances:
            clean_utt = self.clean_str(utterance)
            utt_words = nltk.word_tokenize(clean_utt)
            for word in utt_words:
                self.vocab[word.lower()] += 1

        if use_context:
            context_utterances = self.get_data(self.CONTEXT_ID, mode="train")
            for context in context_utterances:
                for c_utt in context:
                    clean_utt = self.clean_str(c_utt)
                    utt_words = nltk.word_tokenize(clean_utt)
                    for word in utt_words:
                        self.vocab[word.lower()] += 1

    def load_glove_model_for_current_split(self, use_context: bool = False) -> None:
        """Load the GloVe pre-trained model for the current split."""
        print("Loading the GloVe model…")
        filename = self.GLOVE_MODELS_CONTEXT if use_context else self.GLOVE_MODELS
        filename = filename.format(self.config.fold)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if os.path.exists(filename):
            self.model = pickle_loader(filename)
            self.embed_dim = len(self.data_loader.word_emb_dict[self.PAD_TOKEN])
        else:
            self.model = {}
            self.embed_dim = 0
            for word, embedding in self.data_loader.word_emb_dict.items():
                if word in self.vocab:
                    self.model[word.lower()] = embedding
                self.embed_dim = len(embedding)
            with open(filename, "wb") as file:
                pickle.dump(self.model, file, protocol=2)

    def create_embedding_matrix(self) -> None:
        """Create the embedding matrix and word-to-index mapping."""
        vocab_size = len(self.model)
        self.W = np.zeros(shape=(vocab_size + 2, self.embed_dim), dtype=np.float32)
        self.W[self.PAD_ID] = self.data_loader.word_emb_dict[self.PAD_TOKEN]
        self.W[self.UNK_ID] = self.data_loader.word_emb_dict[self.UNK_TOKEN]
        self.word_idx_map[self.PAD_TOKEN] = self.PAD_ID
        self.word_idx_map[self.UNK_TOKEN] = self.UNK_ID
        i = 2
        for word in self.model:
            if word not in {self.PAD_TOKEN, self.UNK_TOKEN}:
                self.W[i] = np.copy(self.model[word])
                self.word_idx_map[word] = i
                i += 1
        for word in self.vocab:
            if word not in self.model:
                self.word_idx_map[word] = self.UNK_ID

    def get_embedding_matrix(self) -> np.ndarray:
        return self.W

    def word_to_index(self, utterance: str) -> Iterable[int]:
        word_indices = [self.word_idx_map.get(word, self.UNK_ID) for word in
                        nltk.word_tokenize(self.clean_str(utterance))]
        word_indices = word_indices[:self.config.max_sent_length]
        word_indices = word_indices + [self.PAD_ID] * (self.config.max_sent_length - len(word_indices))
        assert len(word_indices) == self.config.max_sent_length
        return word_indices

    def get_target_bert_feature(self, mode: str) -> MutableSequence[np.ndarray]:
        return self.get_data(self.TEXT_BERT_ID, mode)

    def get_context_bert_features(self, mode: str) -> np.ndarray:
        utterances = self.get_data(self.CONTEXT_BERT_ID, mode)
        return np.array([np.mean(utterance, axis=0) for utterance in utterances])

    def vectorize_utterance(self, mode: str) -> Sequence[Iterable[int]]:
        utterances = self.get_data(self.UTT_ID, mode)
        return [self.word_to_index(utterance) for utterance in utterances]

    def get_author(self, mode: str) -> np.ndarray:
        authors = self.get_data(self.SPEAKER_ID, mode)
        if mode == "train":
            author_set = {"PERSON"}
            for author in authors:
                author = author.strip()
                if "PERSON" not in author:
                    author_set.add(author)
            self.author_to_index = {author: i for i, author in enumerate(author_set)}
            self.UNK_AUTHOR_ID = self.author_to_index["PERSON"]
            self.config.num_authors = len(self.author_to_index)
        authors = [self.author_to_index.get(author.strip(), self.UNK_AUTHOR_ID) for author in authors]
        return self.to_one_hot(authors, len(self.author_to_index))

    def vectorize_context(self, mode: str) -> np.ndarray:
        dummy_sent = [self.PAD_ID] * self.config.max_sent_length
        contexts = self.get_data(self.CONTEXT_ID, mode)
        vector_context = []
        for context in contexts:
            local_context = []
            for utterance in context[-self.config.max_context_length:]:
                word_indices = self.word_to_index(utterance)
                local_context.append(word_indices)
            for _ in range(self.config.max_context_length - len(local_context)):
                local_context.append(dummy_sent[:])
            local_context = np.array(local_context)
            vector_context.append(local_context)
        return np.array(vector_context)

    def pool_text(self, data: Iterable[int]) -> np.ndarray:
        return np.mean([self.W[i] for i in data if i != 0], axis=0)

    def get_context_pool(self, mode: str) -> np.ndarray:
        contexts = self.get_data(self.CONTEXT_ID, mode)
        vector_context = []
        for context in contexts:
            local_context = []
            for utterance in context[-self.config.max_context_length:]:
                if utterance == "":
