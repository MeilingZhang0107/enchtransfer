## Raw Data

This project investigates cross-lingual model transfer in multimodal sarcasm detection using the following datasets:

### MUStARD

- **Description:**  
  MUStARD (Multimodal Sarcasm Detection Dataset) is an English-language dataset containing labeled utterances from dialogues, with aligned text, audio, and video modalities, plus conversational context.
- **Repository:**  
  [https://github.com/soujanyaporia/MUStARD](https://github.com/soujanyaporia/MUStARD)

### MCSD

- **Description:**  
  MCSD (Mandarin Chinese Sarcasm Dataset) is a multimodal sarcasm detection dataset for Mandarin Chinese, with similar structure to MUStARD, including text, audio, and video.
- **Direct Download:**  
  [Google Drive Link](https://drive.google.com/drive/folders/1uQrdBMxYhA4nOEAn_AtZgfCBjzi73G7R?usp=drive_link)

### Usage Notice

- The datasets are for academic, non-commercial research only.
- This repository does **not** include any raw data files. Please download datasets directly from the official links above.
- Please follow the original dataset license and citation requirements if you use these data in your work.

## Features

For efficient experimentation and reproducibility, this project provides pre-extracted feature representations for both datasets.

### MUStARD

- **Video:**  
  The MUStARD official repository already provides pre-extracted video features using ResNet-152. https://github.com/soujanyaporia/MUStARD
- **Audio:**  
  We provide audio features extracted using vggish or extract the audio features yourself(https://github.com/harritaylor/torchvggish).
- **Text:**  
  We provide text features extracted by BERT-base-multilingual-cased or extract the text features yourself(https://huggingface.co/bert-base-multilingual-cased).    

**Download links:**  
- [VGGish audio features for MUStARD](#)  
- [BERT-base-multilingual-cased text features for MUStARD](#)

### MCSD

- **Video:**  
  Video features are extracted using ResNet-152.
- **Audio:**  
  Audio features are extracted using VGGish.
- **Text:**  
  Text features are extracted using BERT-base-multilingual-cased.

**Download links:**  
----（）

## Run the Code

### Zero-Shot Inference Example

Suppose you want to test MUStARD.   
After switching to the `mustard` directory and updating your dataloader and config as needed, run:

For example:
    python zeroshot_inference.py \
    --model <path_to_mcsd_trained_model> \
    --config-key tav
    
### Few-Shot Inference Example

To perform few-shot cross-lingual transfer (e.g., using 50 labeled target samples), run:

python fewshot.py \
    --src-config-key tav \
    --tgt-config-key tav \
    --few-shot 50




