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


# load metadata on urbansounds
metadata = pd.read_csv('Data/urbansound8k/UrbanSound8K.csv')

# output classes
classes = metadata['class'].unique()
print('\nUnique Classes: ', classes)

# randomly get one row of each class and put into a df
unique_audio = metadata.groupby(['class']).apply(lambda x: x.sample()).reset_index(drop=True)
print(unique_audio)

# plot each row as a waveplot using librosa
fig, axs = plt.subplots(5, 2, figsize=(15, 8), constrained_layout=True)
axs = np.reshape(axs, -1)

for (idx, row), ax in zip(unique_audio.iterrows(), axs):
    ax.set_title(row.values[-1])
    data, sr = librosa.load(f'Data/urbansound8k/fold{row.values[-3]}/' + row.values[0])
    _ = librosa.display.waveplot(data, ax=ax)

plt.show()

# look at different sampling rates for the same audio file
audios = glob.glob(os.path.join("Data/urbansound8k/*/*.wav"), recursive=True)

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Sampling of Audio File with Different Sampling Rates')
axs = np.reshape(axs, -1)
audio = random.choice(audios)
diff_sr = [100, 2500, 20000, 44100]
for ax, sr in zip(axs, diff_sr):
    data, sample_rate = librosa.load(random.choice(audios), sr=sr)
    librosa.display.waveplot(data, sr=sample_rate, ax=ax)
    ax.set_title(f'Sampling Rate of {sr} Hz')

plt.show()

# look at number of instances for each class and plot
instances = metadata['class'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=instances.values, y=instances.index, orient='h')
plt.title('Total Number of Audio Files per Class')
plt.xlabel('Count')
plt.show()

