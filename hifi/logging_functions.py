import os
import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import display


import math
import wandb

def add_audio_to_wandb(wav, transcript='', train=True, real=False):
    if train:
        key = 'train'
    else:
        key = 'val'
    if real:
      key = 'real_' + key
    else:
      key = 'fake_' + key
    path = 'audio_log_file.wav'
    with open(path, "wb") as f:
        f.write(wav.data)
    #torch.save(wav, path)    
    wandb.log(
          {key + '_audio': wandb.Audio(path, sample_rate=22050),
           #key + '_transcript' : wandb.Html(transcript)
           })
    os.remove(path)