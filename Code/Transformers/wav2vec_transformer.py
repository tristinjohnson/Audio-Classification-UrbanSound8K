import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)

# read excel file for more information
metadata = pd.read_csv('../Data/urbansound8k/UrbanSound8K.csv')
metadata['file_path'] = '/fold' + metadata['fold'].astype(str) + '/' + metadata['slice_file_name'].astype(str)

# get only file_path and classID
data = metadata[['file_path', 'classID']]


# load audio file
def load_file(audio_file):
    waveform, sampling_rate = torchaudio.load(audio_file)

    return waveform, sampling_rate


# convert all audio files with 1 audio channel to 2 channels (majority have 2 channels)
def convert_channels(audio, num_channel):
    waveform, sampling_rate = audio

    if waveform.shape[0] == num_channel:
        return audio

    if num_channel == 1:
        new_waveform = waveform[:1, :]
    else:
        new_waveform = torch.cat([waveform, waveform])

    return new_waveform, sampling_rate


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


# pad the waveform of all audio files to a fixed length in ms (milliseconds)
def pad_audio_files(audio, max_ms):
    waveform, sampling_rate = audio
    rows, wave_len = waveform.shape
    max_len = sampling_rate//1000 * max_ms

    # pad the waveform to the max length
    if wave_len > max_len:
        waveform = waveform[:, :max_len]

    # add padding to beginning and end of the waveform
    elif wave_len < max_len:
        padding_front_len = random.randint(0, max_len - wave_len)
        padding_end_len = max_len - wave_len - padding_front_len

        # pad the waveforms with 0
        padding_front = torch.zeros(rows, padding_front_len)
        padding_end = torch.zeros(rows, padding_end_len)

        # concat all padded Tensors
        waveform = torch.cat((padding_front, waveform, padding_end), 1)

    return waveform, sampling_rate


# apply a random time shift to shift the audio left or right by a random amount
def random_time_shift(audio, shift_limit):
    waveform, sample_rate = audio
    _, wave_len = waveform.shape
    shift_amount = int(random.random() * shift_limit * wave_len)

    return waveform.roll(shift_amount), sample_rate


# get a Mel Spectrogram from audio files
# n_fft = number of Fast Fourier Transform - look this up
# n_mel = number of mel filterbanks - look this up
# hop_length = length of hop between STFT windows - look this up
def mel_spectrogram(audio, num_mel=64, num_fft=1024, hop_len=None):
    waveform, sampling_rate = audio
    top_decibel = 80  # min negative cut-off in decibels (default is 80)

    # fit audio to a mel spectrogram
    spectrogram = torchaudio.transforms.MelSpectrogram(sampling_rate,
                                                       n_fft=num_fft,
                                                       hop_length=hop_len,
                                                       n_mels=num_mel)(waveform)

    # convert spectrogram to decibels
    spectrogram = torchaudio.transforms.AmplitudeToDB(top_db=top_decibel)(spectrogram)

    return spectrogram


# data augmentation on audio files
# 1. frequency mask --> randomly mask out a range of consecutive frequencies (horizontal bars)
# 2. time mask --> randomly block out ranges of time from spectrogram (vertical bars)
def data_augmentation(spectrogram, max_mask_pct=0.1, num_freq_masks=1, num_time_masks=1):
    # get channels, number of mels, and number of steps from spectrogram
    channels, num_mels, num_steps = spectrogram.shape

    # get the mask value from spectrogram (the mean)
    mask_value = spectrogram.mean()

    # spectrogram augmentation
    augmented_spectrogram = spectrogram

    # apply number of frequency masks to audio file
    freq_mask_params = max_mask_pct * num_mels
    for _ in range(num_freq_masks):
        augmented_spectrogram = torchaudio.transforms.FrequencyMasking(freq_mask_params)(augmented_spectrogram,
                                                                                         mask_value)

    # apply number of time masks to audio file
    time_mask_params = max_mask_pct * num_steps
    for _ in range(num_time_masks):
        augmented_spectrogram = torchaudio.transforms.TimeMasking(time_mask_params)(augmented_spectrogram, mask_value)

    return augmented_spectrogram


class UrbanSounds(Dataset):
    def __init__(self, data, data_path):
        self.data = data
        self.data_path = data_path
        self.duration = audio_duration
        self.sample_rate = sample_rate
        self.channels = num_channels
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio_file = self.data_path + self.data.loc[index, 'file_path']

        class_id = self.data.loc[index, 'classID']

        audio = load_file(audio_file)

        resample = standardize_audio(audio, self.sample_rate)

        rechannel = convert_channels(resample, self.channels)

        pad_audio = pad_audio_files(rechannel, self.duration)
        # randomize time shift
        shift_audio = random_time_shift(pad_audio, self.shift_pct)

        # get mel spectrogram
        spectrogram = mel_spectrogram(shift_audio)

        # augment the spectrogram
        augment_spectrogram = data_augmentation(spectrogram, num_freq_masks=2, num_time_masks=2)

        return augment_spectrogram, class_id


def model_definition():
    model = bundle.get_model().to(device)

    # define optimizer, scheduler, criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0, verbose=True)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion

# define training and validation loop
def training_and_validation(train_ds):
    model, optimizer, scheduler, criterion = model_definition()

    model_best_acc = 0

    for epoch in range(num_epochs):
        train_loss, steps_train, corr_pred_train, total_pred_train = 0, 0, 0, 0

        model.train()

        pred_labels, real_labels = [], []

        # train the model
        with tqdm(total=len(train_ds), desc=f'Epoch {epoch}') as pbar:

            for x_data, x_target in train_ds:

                x_data, x_target = x_data.to(device), x_target.to(device)

                # normalize inputs
                x_data_mean, x_data_std = x_data.mean(), x_data.std()
                x_data = (x_data - x_data_mean) / x_data_std

                optimizer.zero_grad()
                output = model(x_data)
                loss = criterion(output, x_target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                steps_train += 1

                _, prediction = torch.max(output, 1)

                corr_pred_train += (prediction == x_target).sum().item()
                total_pred_train += prediction.shape[0]

                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {train_loss / steps_train:0.5f} "
                                     f"--> Acc: {corr_pred_train / total_pred_train:0.5f}")

                pred_labels.append(prediction.detach().cpu().numpy())
                real_labels.append(x_target.detach().cpu().numpy())

        # output training metrics
        avg_loss_train = train_loss / len(train_ds)
        acc_train = corr_pred_train / total_pred_train
        print(f'Training: Epoch {epoch} --> Loss: {avg_loss_train:0.5f} --> Accuracy: {acc_train:0.5f}')


wave, sr = load_file('/home/ubuntu/ML2/Final_Project/Code/Data/urbansound8k/fold1/99180-9-0-7.wav')

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

model = bundle.get_model().to(device)
print(model.__class__)

print('original wave: ', wave)

wave = wave.to(device)

if sr != bundle.sample_rate:
    wave = torchaudio.functional.resample(wave, sr, bundle.sample_rate)

print('new wave: ', wave)

with torch.inference_mode():
    features, _ = model.extract_features(wave)

"""fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu())
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")
plt.tight_layout()
plt.show()"""

with torch.inference_mode():
    emission, _ = model(wave)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, ignore):
        super().__init__()
        self.labels = labels
        self.ignore = ignore

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i not in self.ignore]
        return ''.join([self.labels[i] for i in indices])


decoder = GreedyCTCDecoder(
    labels=bundle.get_labels(),
    ignore=(0, 1, 2, 3),
)
transcript = decoder(emission[0])

print(transcript)

"""# define audio variables
audio_duration = 4000  # 4000 ms = 4 s
sample_rate = bundle.sample_rate  # standard sampling rate for audio files
num_channels = 2  # define 2 channels

# define variables
num_epochs = 20
batch_size = 16
learning_rate = 0.001
num_outputs = 10

dataset = UrbanSounds(data, '../Data/urbansound8k')

# get length of data to get training and validation data
data_len = len(dataset)
train_len = round(data_len * 0.70)
val_len = round(data_len * 0.15)
test_len = data_len - train_len - val_len

# split training and validation data
train, validation, test = random_split(dataset, [train_len, val_len, test_len])

# input training and validation data into Dataloaders
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
#validation_dataloader = DataLoader(validation, batch_size=batch_size, shuffle=False)

training_and_validation(train_dataloader)
"""
