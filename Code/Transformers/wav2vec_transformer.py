"""
Tristin Johnson
Final Project - UrbanSound8K
DATS 6203 - Machine Learning II
December 6, 2021
"""
import random
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from tqdm import tqdm
from torchaudio import models
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
import librosa


tokenizer = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')

model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

waveform, sr = librosa.load('/home/ubuntu/ML2/Final_Project/Code/Data/urbansound8k/fold4/30832-3-5-1.wav', sr=44100)

input = tokenizer()

logits = model(input).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)

# Print the output
print(transcription)



