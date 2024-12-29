# -*- coding: utf-8 -*-
import os


root_dir = "/Users/zfwang/project/machinelearning"
project_dir = os.path.join(root_dir, "computer_vision")
# data_dir = "https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download"
# DATASET_ROOT = os.path.join(os.path.expanduser("~", "Downloads/16000_pcm_speeches"))
data_dir = os.path.join(project_dir, "speaker_recognition/data/16000_pcm_speeches")
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"
DATASET_AUDIO_PATH = os.path.join(data_dir, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(data_dir, NOISE_SUBFOLDER)
