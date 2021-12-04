import pandas as pd
import os
import struct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import librosa
import librosa.display


# load metadata on urbansounds
metadata = pd.read_csv('../Data/UrbanSound8K.csv')

# output features and head of metadata
print('Urban Sounds metadata df: \n', metadata.head())
print('Features: ', metadata.columns)

# output classes
classes = metadata['class'].unique()
print('\nUnique Classes: ', classes)


# get path of audio file from metadata
def audio_path(file_name):
    # get file name from urbansounds8k.csv
    file = metadata[metadata['slice_file_name'] == file_name]

    # get path to all .wav files from all directories (fold1 - fold10)
    path = os.path.join('../Data', 'fold'+str(file.fold.values[0]), file_name)

    return path, file['class'].values[0]


# get additional info from .wav files (num of channels, sampling rate, bit depth)
def wav_file_info(file_name):
    # get path using audio_path
    path, _ = audio_path(file_name)

    # open .wav file
    wave_file = open(path, "rb")
    riff_fmt = wave_file.read(36)

    # get number of channels from .wav files
    n_channels_string = riff_fmt[22:24]
    num_channels = struct.unpack("H", n_channels_string)[0]

    # get the sampling rate of each audio file
    sampling_rate_string = riff_fmt[24:28]
    sampling_rate = struct.unpack("I", sampling_rate_string)[0]

    # get the bit depth of each audio file
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H", bit_depth_string)[0]

    return num_channels, sampling_rate, bit_depth


# randomley get one row of each class and put into a df
unique_audio = metadata.groupby(['class']).apply(lambda x: x.sample()).reset_index(drop=True)

# plot each row as a waveplot using librosa
fig, axs = plt.subplots(5, 2, figsize=(15, 8), constrained_layout=True)
axs = np.reshape(axs, -1)

for (idx, row), ax in zip(unique_audio.iterrows(), axs):
    ax.set_title(row.values[-1])
    data, sr = librosa.load(f'../Data/fold{row.values[-3]}/' + row.values[0])
    _ = librosa.display.waveplot(data, ax=ax)

plt.show()

# look at number of instances for each class and plot
instances = metadata['class'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=instances.values, y=instances.index, orient='h')
plt.title('Total Number of Audio Files per Class')
plt.xlabel('Count')
plt.show()


# list comprehension to get additional info about wav files
#wav_info = [wav_file_info(i) for i in df.slice_file_name]

# join additional info to original dataframe
#df[['num_channels', 'sampling_rate', 'bit_depth']] = pd.DataFrame(wav_info)
#print(df.head())
