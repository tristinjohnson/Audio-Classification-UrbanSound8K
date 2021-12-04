import pandas as pd
import os
import struct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import librosa
import librosa.display
import random
import glob
from train_validate_model import load_file, convert_channels, standardize_audio, pad_audio_files, random_time_shift
import warnings
warnings.filterwarnings('ignore')


# load metadata on urbansounds
metadata = pd.read_csv('Data/urbansound8k/UrbanSound8K.csv')

# output classes
classes = metadata['class'].unique()
print('\nUnique Classes: ', classes, '\n')

# randomly get one row of each class and put into a df
unique_audio = metadata.groupby(['class']).apply(lambda x: x.sample()).reset_index(drop=True)
print('One row of each class from metadata: \n', unique_audio, '\n')

# look at different sampling rates for the same audio file
audios = glob.glob(os.path.join("Data/urbansound8k/*/*.wav"), recursive=True)
audio = random.choice(audios)

# look at number of instances for each class and plot
instances = metadata['class'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=instances.values, y=instances.index, orient='h')
plt.title('Total Number of Audio Files per Class')
plt.xlabel('Count')
plt.show()

# plot an audio file with different frequencies
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Sampling of Audio File with Different Sampling Rates')
axs = np.reshape(axs, -1)
diff_sr = [100, 2500, 20000, 44100]
for ax, sr in zip(axs, diff_sr):
    data, sample_rate = librosa.load(audio, sr=sr)
    librosa.display.waveplot(data, sr=sample_rate, ax=ax)
    ax.set_title(f'Sampling Rate of {sr} Hz')

plt.show()

# plot each row as a waveplot using librosa
fig, axs = plt.subplots(5, 2, figsize=(15, 8), constrained_layout=True)
axs = np.reshape(axs, -1)

for (idx, row), ax in zip(unique_audio.iterrows(), axs):
    ax.set_title(row.values[-1])
    data, sr = librosa.load(f'Data/urbansound8k/fold{row.values[-3]}/' + row.values[0])
    _ = librosa.display.waveplot(data, ax=ax)

plt.show()


# get path of file
def get_path(file_name):
    name = metadata[metadata['slice_file_name'] == file_name]
    path = os.path.join('Data/urbansound8k', 'fold' + str(name.fold.values[0]), file_name)

    return path, name['class'].values[0]


# get the number of channels and sample rate for each audio file
def get_more_info(file_name):
    path, _ = get_path(file_name)
    wave_file = open(path, 'rb')
    format = wave_file.read(36)

    num_channels = format[22:24]
    num_channels = struct.unpack('H', num_channels)[0]

    sample_rate = format[24:28]
    sample_rate = struct.unpack('I', sample_rate)[0]

    return num_channels, sample_rate


# add num_channels and sampling_rate to dataframe
extra_data = [get_more_info(i) for i in metadata.slice_file_name]
metadata[['num_channels', 'sampling_rate']] = pd.DataFrame(extra_data)

# plot how many audio files have a certain sample rate
sample_rate_totals = metadata['sampling_rate'].value_counts()

plt.figure(figsize=(9, 5))
sns.barplot(x=sample_rate_totals.index, y=sample_rate_totals.values)
plt.title('Total Number of Audio Files Grouped By Sample Rate')
plt.xlabel('Sample Rate')
plt.ylabel('Count')
plt.show()

# plot how many audio files have a certain number of channels
num_channels_total = metadata['num_channels'].value_counts()

plt.figure(figsize=(9, 5))
sns.barplot(x=num_channels_total.index, y=num_channels_total.values)
plt.title('Total Number of Audio Files Grouped by Channels')
plt.xlabel('Num Channels')
plt.ylabel('Count')
plt.show()

# get new waveform and sample rates using functions from 'train_validate_model.py'
waveform, sr = load_file(audio)
waveform, sr = convert_channels((waveform, sr), 2)
waveform, sr = standardize_audio((waveform, sr), 44100)
waveform, sr = pad_audio_files((waveform, sr), 4000)

# plot before random time shift
plt.figure(figsize=(9, 5))
librosa.display.waveplot(waveform.numpy(), sr)
plt.title(f'Wave Plot Before Random Time Shift')
plt.xlabel('Time')
plt.ylabel('Waveform')
plt.show()

# add random time shift
waveform, sr = random_time_shift((waveform, sr), 0.5)

# plt after random time shift
plt.figure(figsize=(9, 5))
librosa.display.waveplot(waveform.numpy(), sr)
plt.title('Wave Plot After Random Time Shift')
plt.xlabel('Time')
plt.ylabel('Waveform')
plt.show()

# plot Mel Spectrogram (input to network)
wv, sample_rate = librosa.load(audio)
fig, ax = plt.subplots()
s = librosa.feature.melspectrogram(y=wv, sr=sample_rate)
img = librosa.display.specshow(librosa.power_to_db(s, ref=np.max), x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Mel Spectrogram')
fig.colorbar(img, ax=ax, format='%+2.f dB')
plt.show()
