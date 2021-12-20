import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class ResBlock(nn.Module):
  def __init__(self, 
               D_r, # Матрица dilation rates
               k_r, # Вектор kernel sizes
               channels,
               slope=0.1
               ):
    super(ResBlock, self).__init__()

    assert len(D_r.shape) == 2, 'D_r shape is ' + str(D_r.shape)
    #(VERSION 3)Вот это нифига не правильно
    #assert len(k_r.shape) == 1, 'k_r shape is ' + str(k_r.shape)

    net = []
    for m in range(len(D_r)):
      for l in range(len(D_r[m])):

        net.append(nn.Sequential(*[
                                nn.LeakyReLU(slope),
                                nn.Conv1d(channels, 
                                          channels,
                                          kernel_size=k_r,
                                          dilation=D_r[m][l],
                                          padding=get_padding(k_r, D_r[m][l])
                                          )
                               ])
                         )
    #Это у нас к-во "подблоков", созданных в внутреннем цикле    
    self.num_blocks = len(D_r)
    #Это длина одного "подблока". 
    #По-моему, это эквивалентно D_r.shape[1]
    #Но раз мы идем по массиву net, то посчитаю из соображений его длины
    self.len_block = len(net) // len(D_r)
    #Я помню эту ошибку из предыдущей домашки
    self.net = nn.ModuleList(net)

  #Вроде надо последовательно подавать в блоки, не забывая прибавлять skip-connection
  #TODO понять, так ли это  
  #(VERSION 2)Не так, блин
  #Сквозь подблоки - подряд
  #Между подблоками со skip-connection
  def forward(self, x):
    count = 0
    for i in range(self.num_blocks):
      x_new = x
      #Идем сквозь
      for j in range(self.len_block):
        x_new = self.net[count](x_new)
        count += 1
      #skip-connection
      x = x + x_new
    return x


class MRFBlock(nn.Module):

  def __init__(self, 
               channels,
               D_r, # 3D
               k_r # 1D
               ):
    super(MRFBlock, self).__init__()

    #Там будет деление на эту штуку
    #self.num_kernels = len(h.resblock_kernel_sizes)
    #Вроде это оно
    self.num_kernels = len(k_r)



    assert len(D_r.shape) == 3, 'D_r shape is ' + str(D_r.shape)
    assert len(k_r.shape) == 1, 'k_r shape is ' + str(k_r.shape)
    
    net = []
    for n in range(self.num_kernels):
      net.append(ResBlock(D_r[n],
                          k_r[n],
                          channels))
    #Я помню эту ошибку из предыдущей домашки
    self.net = nn.ModuleList(net)

  
  
  def forward(self, x):
    #Надо прогнать сквозь блоки, сложить и разделить
    out = 0
    for block in self.net:
      out += block(x)
    return out / self.num_kernels

class GeneratorModel(nn.Module):
  def __init__(self,
               h_u, #hidden dimension
               k_u, #kernel size for conv transposed
               D_r,
               k_r,
               slope=0.1
               ):
    super(GeneratorModel, self).__init__()

    #Это у нас первый блок. 
    #Захардкодила чиселки (
    self.first = nn.Conv1d(80, h_u, 7, 1, padding=3)  
    
    net = []
    in_channels = h_u

    for l in range(len(k_u)):
      out_channels = in_channels // 2

      net.append(nn.Sequential(*[
                                 nn.LeakyReLU(slope),
                                 #В мдз5 по глубинному обучению я использовала Upsample
                                 #Это было ошибкой
                                 nn.ConvTranspose1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=k_u[l],
                                                    stride = k_u[l] // 2,
                                                    padding = (k_u[l] - k_u[l] // 2) // 2
                                                    ),
                                 MRFBlock(out_channels, D_r, k_r)
                                 ])
                 )
      in_channels = out_channels
    
    self.net = nn.Sequential(*net)

    self.last = nn.Sequential(*[
                                nn.LeakyReLU(slope),
                                nn.Conv1d(in_channels, 1, 7, 1, padding=3),
                                nn.Tanh()
                                ])
    
  def forward(self, x):
    return self.last(self.net(self.first(x)))
    



