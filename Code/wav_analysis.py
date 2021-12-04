import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from tqdm import tqdm
import librosa
import librosa.display
import glob
import os


audio = '/home/ubuntu/ML2/Final_Project/Code/Data/urbansound8k/fold4/30832-3-5-1.wav'


# load audio file
def load_file(audio_file):
    waveform, sampling_rate = torchaudio.load(audio_file)

    return waveform, sampling_rate


plt.figure()
waveform, sr = librosa.load(audio)
librosa.display.waveplot(waveform, sr=sr)
plt.show()


# convert all audio files with 1 audio channel to 2 channels (majority have 2 channels)
def convert_channels(audio, num_channel):
    waveform, sampling_rate = audio

    if waveform.shape[0] == num_channel:
        return audio

    if num_channel == 1:
        new_waveform = waveform[:1, :]
    else:
        #new_waveform = torch.cat([waveform, waveform])
        new_waveform = np.concatenate([waveform, waveform])

    return new_waveform, sampling_rate


print('Original Waveform shape: ', waveform.shape)
print('Original Sample Rate: ', sr)

waveform, sr = convert_channels((waveform, sr), 2)
print('\nWaveform shape (after converting to 2 channels): ', waveform.shape)
print('Sample Rate (after converting to 2 channels): ', sr)


# standardize the sampling rate of each audio file
def standardize_audio(audio, new_sample_rate):
    new_waveform, sampling_rate = audio

    if sampling_rate == new_sample_rate:
        return audio

    # get number of channels
    num_channel = new_waveform.shape[0]

    # standardize (resample) the first channel
    waveform_1 = torchaudio.transforms.Resample(sampling_rate, new_sample_rate)(new_waveform[:1, :])

    # if number of channels > 1, resample second channel
    if num_channel > 1:
        waveform_2 = torchaudio.transforms.Resample(sampling_rate, new_sample_rate)(new_waveform[1:, :])

        # merge both channels
        new_waveform = torch.cat([waveform_1, waveform_2])

    return new_waveform, new_sample_rate


waveform, sr = standardize_audio(load_file(audio), 44100)
print('\nWaveform shape (after resampling audio to 44100 Hz): ', waveform.shape)
print('Sample Rate (after resampling audio to 44100 Hz): ', sr)


# apply a random time shift to shift the audio left or right by a random amount
def random_time_shift(audio, shift_limit):
    waveform, sample_rate = audio
    _, wave_len = waveform.shape
    shift_amount = int(random.random() * shift_limit * wave_len)

    return waveform.roll(shift_amount), sample_rate



# example of mel spectorgram
"""spect = mel_spectrogram((wav, sr))
new_spect = data_augmentation(spect)
test = librosa.power_to_db(new_spect)
spect = test.reshape(test.shape[1], test.shape[2])
fig, axs = plt.subplots(1, 1)
im = axs.imshow(spect, origin='lower')
fig.colorbar(im, ax=axs)
plt.show()"""

