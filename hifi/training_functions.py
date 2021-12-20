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
import wandb

from tqdm import tqdm

from hifi.logging_functions import *
from hifi.losses import *
from hifi.dataset import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_g(model,
          loss_func,
          n_epochs,
          opt,
          train_loader,
          num_model=0,
          logging=False
          ):
    
    for epoch in range(n_epochs):
        train_loss  = train_epoch_g(model,
                    loss_func,
                    opt,
                    train_loader,
                    num_model,
                    logging)
        print('\nepoch ' + str(epoch) + '/' + str(n_epochs) + ' train loss = ' + str(train_loss))
        torch.save(model.state_dict(), 'model_' + str(num_model) +'.pt')
        torch.save(model.state_dict(), '/content/drive/MyDrive/hifi/model_' + str(num_model) +'.pt')
        if logging:
            wandb.log({
              'train_loss_epoch' : train_loss
            })
        
        
        
def train_epoch_g(model,
                loss_func,
                opt,
                train_loader,
                num_model=0,
                logging=False):
    model.train()
    model = model.to(DEVICE)

    torch.autograd.set_detect_anomaly(True)

    loss_sum = 0

    n_fft =1024
    num_mels = 80
    hop_size = 256
    win_size = 1024
    sampling_rate = 22050
    fmin = 0
    fmax = 8000 
    
    for mel, audio, filename, mel_loss in train_loader:

        mel = mel.to(DEVICE)
        audio = audio.to(DEVICE)

        opt.zero_grad()

        preds = model(mel)
        mel_preds = mel_spectrogram(preds.squeeze(1), 
                                    n_fft=n_fft,
                                    num_mels=num_mels,
                                    hop_size=hop_size,
                                    win_size=win_size,
                                    sampling_rate=sampling_rate,
                                    fmin = fmin,
                                    fmax=fmax).to(DEVICE)

        loss = loss_func(mel, mel_preds) * 45
        loss.backward()

        opt.step()

        
        loss_sum += loss.item()

        if logging:
            wandb.log({
              'train_loss_step' : loss.item(),
            })

        
    
    return loss_sum / len(train_loader)


from tqdm import tqdm
def train(g,
          mpd,
          msd,
          
          n_epochs,
          g_opt,
          d_opt,
          train_loader,
          num_model=0,
          logging=False
          ):
    
    for epoch in range(n_epochs):
        train_loss_g, train_loss_d  = train_epoch(g,
                                  mpd,
                                  msd,
                                 
                                  g_opt,
                                  d_opt,
                                  train_loader,
                                  num_model,
                                  logging)
        print('\nepoch ' + str(epoch) + '/' + str(n_epochs) + 
              ' train loss g = ' + str(train_loss_g) + 
              ' train loss d = ' + str(train_loss_d))
        
        torch.save(g.state_dict(), '/content/drive/MyDrive/hifi/g_' + str(num_model) +'.pt')
        torch.save(msd.state_dict(), '/content/drive/MyDrive/hifi/msd_' + str(num_model) +'.pt')
        torch.save(mpd.state_dict(), '/content/drive/MyDrive/hifi/mpd_' + str(num_model) +'.pt')
        if logging:
            wandb.log({
              'train_loss_g_epoch' : train_loss_g,
              'train_loss_d_epoch' : train_loss_d
            })
        
        
        
def train_epoch(g,
                mpd,
                msd,
                
                g_opt,
                d_opt,
                train_loader,
                num_model=0,
                logging=False):
    g.train()
    g = g.to(DEVICE)

    mpd.train()
    mpd = mpd.to(DEVICE)

    msd.train()
    msd = msd.to(DEVICE)


    #torch.autograd.set_detect_anomaly(True)

    loss_sum_d, loss_sum_g = 0, 0

    n_fft =1024
    num_mels = 80
    hop_size = 256
    win_size = 1024
    sampling_rate = 22050
    fmin = 0
    fmax = 8000 
    
    for ii, batch in tqdm(enumerate(train_loader), total = len(train_loader)):

        mel, audio, filename, mel_loss = batch

        mel = mel.to(DEVICE)
        audio = audio.to(DEVICE)
        audio = audio.unsqueeze(1)


        #print('input ', mel.shape)


        preds = g(mel)

        #print('preds ', preds.shape)
        mel_preds = mel_spectrogram(preds.squeeze(1), 
                                    n_fft=n_fft,
                                    num_mels=num_mels,
                                    hop_size=hop_size,
                                    win_size=win_size,
                                    sampling_rate=sampling_rate,
                                    fmin = fmin,
                                    fmax=fmax).to(DEVICE)

        #print('mel preds ', mel_preds.shape)                           
        ###Обучаем дискриминатор

        d_opt.zero_grad()

        # MPD

        #Не забываю сделать detach, так как генератор не учим
        real_res_p, _, fake_res_p, _ = mpd(audio, preds.detach())
        #real и fake - это просто массивы, их можно залоггировать
        loss_p, real_losses_p, fake_losses_p = discriminator_loss(real_res_p, fake_res_p)



        # MSD

        #Не забываю сделать detach, так как генератор не учим
        real_res_s, _, fake_res_s, _ = msd(audio, preds.detach())
        #real и fake - это просто массивы, их можно залоггировать
        loss_s, real_losses_s, fake_losses_s = discriminator_loss(real_res_s, fake_res_s)

        loss = loss_p + loss_s
      
        loss_sum_d += loss.item()

        loss.backward()

        d_opt.step()

      
        if (ii+1) % 90 == 0 and logging:
            the_dict = {
              'd_train_loss_step' : loss.item(),
            }
            for i in range(len(real_losses_p)):
              the_dict['d_real_losses_p_'+str(i)] = real_losses_p[i]
              the_dict['d_fake_losses_p_'+str(i)] = fake_losses_p[i]
            for i in range(len(real_losses_s)):
              the_dict['d_real_losses_s_'+str(i)] = real_losses_s[i]
              the_dict['d_fake_losses_s_'+str(i)] = fake_losses_s[i]

            wandb.log(the_dict)

            
  
        ### Обучаем генератор 
        g_opt.zero_grad()

        loss_mel = F.l1_loss(mel, mel_preds) * 45

        l1_loss = loss_mel.item()


        real_res_p, real_feature_map_p, fake_res_p, fake_feature_map_p = mpd(audio, preds)
        real_res_s, real_feature_map_s, fake_res_s, fake_feature_map_s = msd(audio, preds)

        f_loss_p = feature_map_loss(real_feature_map_p, fake_feature_map_p)
        f_loss_s = feature_map_loss(real_feature_map_s, fake_feature_map_s)

        gen_loss_p, g_losses_p = generator_loss(fake_res_p)
        gen_loss_s, g_losses_s = generator_loss(fake_res_s)

        loss = f_loss_p + f_loss_s + gen_loss_p + gen_loss_s + loss_mel

        loss_sum_g += loss.item()

        loss.backward()

        g_opt.step()


        if (ii+1) % 90 == 0 and logging:
            the_dict = {
              'g_train_loss_step' : loss.item(),
              'l1_loss_step'  : l1_loss
            }
            for i in range(len(real_losses_p)):
              the_dict['g_real_losses_p_'+str(i)] = real_losses_p[i]
              the_dict['g_fake_losses_p_'+str(i)] = fake_losses_p[i]
            for i in range(len(real_losses_s)):
              the_dict['g_real_losses_s_'+str(i)] = real_losses_s[i]
              the_dict['g_fake_losses_s_'+str(i)] = fake_losses_s[i]
            for i in range(len(g_losses_p)):
              the_dict['g_losses_p_'+str(i)] = g_losses_p[i]
            for i in range(len(g_losses_s)):
              the_dict['g_losses_s_'+str(i)] = g_losses_s[i]

            wandb.log(the_dict)
      
        if (ii+1) % 90 == 0 and logging:     
            wav = preds[0].detach().squeeze(0).cpu()
            wav = display.Audio(wav, rate=22050)
            add_audio_to_wandb(wav, real=False)


            wav = audio[0].detach().squeeze(0).cpu()
            wav = display.Audio(wav, rate=22050)
            add_audio_to_wandb(wav, real=True)

    return loss_sum_g / len(train_loader), loss_sum_d / len(train_loader)



     


     