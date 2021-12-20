import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn.modules.activation import LeakyReLU

from hifi.generator import *


#Это у нас sub-discriminators для MPD
class MPDSubBlock(nn.Module):
    #Я не знаю, как красиво подать все
    #Поэтому хардкодю
    def __init__(self, 
                 p,
                 kernel_size=5,
                 stride=3,
                 slope=0.1
                 ):
      super(MPDSubBlock, self).__init__()

      self.p = p
       
      #Не забываю про ModuleList 
      self.blocks = nn.ModuleList([
            nn.Sequential(*[nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
                           nn.LeakyReLU(slope)]),
            #В гитхабэ это выносят отдеьно, но т.к. я засунула релу внутрь, то можно засунуть сюда.               
            nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))               
        ])
      
    def forward(self, x):
      batch_size, input_ch, T = x.shape

      #Вроде бы можно вот так без if
      #Будет глупо, если ошиблась:)
      pad = (self.p - (T % self.p)) % self.p

      x = nn.functional.pad(x, (0, pad), 'reflect')
      
      #TODO понять, откуда -1 (без него не работает)
      x = x.view(batch_size, input_ch, (T + self.p - 1) // self.p, self.p)

      feature_map = []

      for block in self.blocks:
        x = block(x)
        feature_map.append(x)
      
     
      return  torch.flatten(x, 1, -1), feature_map


class MPDBlock(nn.Module):
  def __init__(self):

    super(MPDBlock, self).__init__()
    ##Только хардкод
    self.blocks = nn.ModuleList([
            MPDSubBlock(2),
            MPDSubBlock(3),
            MPDSubBlock(5),
            MPDSubBlock(7),
            MPDSubBlock(11)
        ])
  def forward(self, real, fake):
    real_res, real_feature_map = [], []
    fake_res, fake_feature_map = [], []

    for block in self.blocks:
      real_res_b, real_feature_map_b = block(real)
      fake_res_b, fake_feature_map_b = block(fake)

      real_res.append(real_res_b)
      real_feature_map.append(real_feature_map_b)
      fake_res.append(fake_res_b)
      fake_feature_map.append(fake_feature_map_b)
    #Проверю, что все аппенднулось правильно
    assert (len(fake_feature_map) == len(real_feature_map) and 
            len(fake_res) == len(real_feature_map) and 
            len(fake_res) == len(real_res))
    return real_res, real_feature_map, fake_res, fake_feature_map

class MSDSubBlock(nn.Module):
    def __init__(self, 
                 slope=0.1
                 ):
      super(MSDSubBlock, self).__init__()

       
      #Не забываю про ModuleList 
      #Я даже пытаться не буду красиво это подсть
      #Я просто скопипасщу
      self.blocks = nn.ModuleList([
            nn.Sequential(*[nn.Conv1d(1, 128, 15, 1, padding=7),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
                           nn.LeakyReLU(slope)]),
            nn.Sequential(*[nn.Conv1d(1024, 1024, 5, 1, padding=2),
                           nn.LeakyReLU(slope)]),               
            #В гитхабэ это выносят отдеьно, но т.к. я засунула релу внутрь, то можно засунуть сюда.               
            nn.Conv1d(1024, 1, 3, 1, padding=1)               
        ])
    #forward вроде тоно такойже
    #TODO: может, можно объединить как-то в один класс? 
    def forward(self, x):
    
      feature_map = []

      for block in self.blocks:
        x = block(x)
        feature_map.append(x)
      
      return  torch.flatten(x, 1, -1), feature_map


class MSDBlock(nn.Module):
  def __init__(self):

    super(MSDBlock, self).__init__()
    ##Только хардкод
    self.blocks = nn.ModuleList([
            MSDSubBlock(),
            nn.Sequential(*[
                           nn.AvgPool1d(4, 2, padding=2),
                           MSDSubBlock()]),
            nn.Sequential(*[
                           nn.AvgPool1d(4, 2, padding=2),
                           MSDSubBlock()]),
        ])
  #Такая же ситуация
  def forward(self, real, fake):
    real_res, real_feature_map = [], []
    fake_res, fake_feature_map = [], []

    for block in self.blocks:
      real_res_b, real_feature_map_b = block(real)
      fake_res_b, fake_feature_map_b = block(fake)

      real_res.append(real_res_b)
      real_feature_map.append(real_feature_map_b)
      fake_res.append(fake_res_b)
      fake_feature_map.append(fake_feature_map_b)
    #Проверю, что все аппенднулось правильно
    assert (len(fake_feature_map) == len(real_feature_map) and 
            len(fake_res) == len(real_feature_map) and 
            len(fake_res) == len(real_res))
    return real_res, real_feature_map, fake_res, fake_feature_map

