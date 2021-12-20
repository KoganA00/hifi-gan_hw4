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

import math

def feature_map_loss(real_feature_map, fake_feature_map):
  loss = 0
  for i in range(len(real_feature_map)):
    for j in range(len(real_feature_map[i])):
      loss += torch.mean(torch.abs(real_feature_map[i][j] - fake_feature_map[i][j]))
  return loss * 2


def discriminator_loss(real_output, fake_output):
    loss = 0
    real_losses = []
    fake_losses = []
    for i in range(len(real_output)):

        real_loss = torch.mean((1 - real_output[i]) ** 2)
        real_losses.append(real_loss.item())

        fake_loss = torch.mean(fake_output[i] ** 2)
        fake_losses.append(fake_loss.item())

        loss += (real_loss + fake_loss)
        
    return loss, real_losses, fake_losses


def generator_loss(output):
    loss = 0
    losses = []
    for i in range(len(output)):
        loss_ = torch.mean((1 - output[i])**2)
        losses.append(loss_)
        loss += loss_

    return loss, losses