# Real-Time Voice Cloning
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. This was my [master's thesis](https://matheo.uliege.be/handle/2268.2/6801).

SV2TTS is a deep learning framework in three stages. In the first stage, one creates a digital representation of a voice from a few seconds of audio. In the second and third stages, this representation is used as reference to generate speech given arbitrary text.

**Video demonstration** (click the picture):

[![Toolbox demo](https://i.imgur.com/8lFUlgz.png)](https://www.youtube.com/watch?v=-O_hYhToKoA)



### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1609.03499](https://arxiv.org/pdf/1609.03499.pdf) | WaveNet (vocoder) | A Generative Model for Raw Audio | [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder) |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[2010.05646](https://arxiv.org/pdf/2010.05646.pdf) | HiFi-GAN (vocoder) | Generative Adversarial Networks for Efficient and High Fidelity | [jik876/hifi-gan](https://github.com/jik876/hifi-gan) |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron (synthesizer) | Tacotron: Towards End-to-End Speech Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |


## Setup

### 1. Install Requirements
1. Both Windows and Linux are supported. A GPU is recommended for training and for inference speed, but is not mandatory.
2. Python 3.7 is recommended. Python 3.5 or greater should work, but you'll probably have to tweak the dependencies' versions. I recommend setting up a virtual environment using `venv`, but this is optional.
3. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.
4. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version, your operating system, your package manager (pip by default) and finally pick any of the proposed CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.
5. Install the remaining requirements with `pip install -r requirements.txt`

### 2. (Optional) Download Pretrained Models
Pretrained models are now downloaded automatically. If this doesn't work for you, you can manually download them [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).

### 3. (Optional) Test Configuration
Before you download any dataset, you can begin by testing your configuration with:

`python demo_cli.py`

If all tests pass, you're good to go.


## Training

This is a step-by-step guide for reproducing the training.

### Datasets
This experiments uses **[VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)** (used in the SV2TTS paper) and **[UncommonVoice](https://merriekay.com/uncommonvoice/)** (Dysphonia dataset audio).

**[VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)** data is used in Encoders, Synthesizers, and Vocoders. **[UncommonVoice](https://merriekay.com/uncommonvoice/)** used in Vocoders only.


### Preprocessing and training
Here's the great thing about this repo: you're expected to run all python scripts in their alphabetical order. You likely started with the demo scripts, now you can run the remaining ones (pass `-h` to get argument infos for any script): 

`python encoder_preprocess.py <datasets_root>`

For training, the encoder uses visdom. You can disable it with `--no_visdom`, but it's nice to have. Run "visdom" in a separate CLI/process to start your visdom server. Then run:

`python encoder_train.py my_run <datasets_root>/SV2TTS/encoder`

Then you have two separate scripts to generate the data of the synthesizer. This is convenient in case you want to retrain the encoder, you will then have to regenerate embeddings for the synthesizer.

Begin with the audios and the mel spectrograms:

`python synthesizer_preprocess_audio.py <datasets_root>`

Then the embeddings:
 
`python synthesizer_preprocess_embeds.py <datasets_root>/SV2TTS/synthesizer`

You can then train the synthesizer:

`python synthesizer_train.py my_run <datasets_root>/SV2TTS/synthesizer`

The synthesizer will output generated audios and spectrograms to its model directory when training. 

Use the synthesizer to generate training data for the vocoder:

`python vocoder_preprocess.py <datasets_root>`

And finally, train the vocoder:

`python vocoder_train.py my_run <datasets_root>`

The vocoder also outputs ground truth/generated audios to its model directory.
 

## Inference

To run the trained model and cloning voice, you can run cli in terminal with:

`python demo_cli.py --no_sound`

And for Speaker Conditional model, you can run:

`python demo_cli_sc.py --no_sound`


## NOTES

A few things to concern

### 1. Change Vocoder

The default vocoder in this repository is WaveRNN.

To use another vocoder (WaveNet, HiFi-GAN), please refer to the `repositories` folder in my root directory.

### 2. Use Speaker Conditional

To use Speaker Conditional for the vocoder, please retrain and use file with suffix `xx_sc.py` inside the vocoder directory.

### 3. HiFI-GAN

To use HiFI-GAN, you will need to retrain the synthesizer model and preprocess data again.

Replace `synthesizer import audio` with `synthesizer import audio_hifigan`, in files `synthesizer/preprocess.py`, `synthesizer/train.py`, and `synthesizer/inference.py`.
