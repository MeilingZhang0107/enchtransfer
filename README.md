## Reference Pipeline

The core experimental pipeline of this project is based on the implementation from the official [MUStARD repository](https://github.com/soujanyaporia/MUStARD).  
We have adapted and extended their workflow for our cross-lingual multimodal sarcasm detection experiments, including feature extraction, data processing, and baseline modeling.

## Environment
The main experiments were conducted using the following Python packages:
 ```
h5py==3.13.0
jsonlines==4.0.0
numpy==2.1.3
scikit-learn==1.6.1
torch==2.2.2
torchaudio==2.5.1+cu121
torchvggish==0.1
torchvision==0.17.2
 ```
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
- [Google Drive: MUStARD Data](https://drive.google.com/drive/folders/16W0FcQTtyF6nR0m9LPWzgWr8bCH3koRt?usp=drive_link)

### MCSD

  Video features are extracted using ResNet-152. Audio features are extracted using VGGish.Text features are extracted using BERT-base-multilingual-cased.

**Download links:**   
- [Google Drive: MCSD Data](https://drive.google.com/drive/folders/1wCZ-SgmzzClbDvyKXI6VdC6CEz5dOf6Y?usp=drive_link)

### Joint Training
**Download links:**   
- [Google Drive: Joint Data](https://drive.google.com/drive/folders/1fO4L3QfqXRjVq3WQ9n_m--SiQmSyhG8B?usp=drive_link)

## Pretrained Models

You can download pretrained model checkpoints used in this project from the following link:

- [Google Drive: Pretrained Models](https://drive.google.com/drive/folders/14DASVEo7lSodRQlBqzSURSDpAlzoHPvg?usp=drive_link)


## Run the Code

### Zero-Shot Inference Example

Suppose you want to test MUStARD.   
After switching to the `mustard` directory and updating your dataloader and config as needed, run:

For example:
```
    python zeroshot_inference.py \
    --model <path_to_mcsd_trained_model> \
    --config-key tav
 ```
    
### Few-Shot Inference Example

To perform few-shot cross-lingual transfer (e.g., using 50 labeled target samples), run:
```
python fewshot.py \
    --src-config-key tav \
    --tgt-config-key tav \
    --few-shot 50
 ```


## Config

Below are example configurations for different experimental modalities, all with context enabled.
Copy the relevant block into your configuration file as needed.

<details>
<summary><strong>Audio-only (with context)</strong></summary>

```python
use_context = True
use_author = False

use_bert = True

use_target_text = False
use_target_audio = True
use_target_video = False

speaker_independent = False
```

</details>

<details>
<summary><strong>Text-only (with context)</strong></summary>

```python
use_context = True
use_author = False

use_bert = True

use_target_text = True
use_target_audio = False
use_target_video = False

speaker_independent = False
```

</details>

<details>
<summary><strong>Video-only (with context)</strong></summary>

```python
use_context = True
use_author = False

use_bert = True

use_target_text = False
use_target_audio = False
use_target_video = True

speaker_independent = False
```

</details>

<details>
<summary><strong>Text + Audio (with context)</strong></summary>

```python
use_context = True
use_author = False

use_bert = True

use_target_text = True
use_target_audio = True
use_target_video = False

speaker_independent = False
```

</details>

<details>
<summary><strong>Text + Audio + Video (with context)</strong></summary>

```python
use_context = True
use_author = False

use_bert = True

use_target_text = True
use_target_audio = True
use_target_video = True

speaker_independent = False
```

</details>





