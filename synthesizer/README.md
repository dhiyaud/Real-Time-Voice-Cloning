# Synthesizer

This is directory for Synthesizer model.

## Please note:

To use HiFI-GAN, you will need to retrain the synthesizer model and preprocess data again.

Replace `synthesizer import audio` with `synthesizer import audio_hifigan`, in files `synthesizer/preprocess.py`, `synthesizer/train.py`, and `synthesizer/inference.py`.