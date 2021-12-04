"""
Tristin Johnson
Final Project - UrbanSound8K
DATS 6203 - Machine Learning II
December 6, 2021
"""
import random
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# define variables
num_epochs = 2
batch_size = 16
learning_rate = 0.001
num_outputs = 10

# define audio variables
audio_duration = 4000  # 4000 ms = 4 s
sample_rate = 44100  # standard sampling rate for audio files
num_channels = 2  # define 2 channels


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


# define custom variables for UrbanSounds DataSet
class UrbanSoundsDS(Dataset):
    def __init__(self, data, data_path):
        self.data = data
        self.data_path = data_path
        self.duration = audio_duration
        self.sampling_rate = sample_rate
        self.channel = num_channels
        self.shift_pct = 0.4

    # total number of items in dataset
    def __len__(self):
        return len(self.data)

    # get the i'th item in dataset
    def __getitem__(self, index):
        # get path of audio file
        audio_file = self.data_path + self.data.loc[index, 'file_path']

        # get class id from audio file
        class_id = self.data.loc[index, 'classID']

        # load the audio file
        audio = load_file(audio_file)

        # standardize all audio files
        resample_audio = standardize_audio(audio, self.sampling_rate)

        # make all audio files have same number of channels
        rechannel = convert_channels(resample_audio, self.channel)

        # add padding
        pad_audio = pad_audio_files(rechannel, self.duration)

        # random time shift
        time_shift = random_time_shift(pad_audio, self.shift_pct)

        # get mel spectrogram from audio file
        spectrogram = mel_spectrogram(time_shift)

        # augment the spectrogram
        augment_spectrogram = data_augmentation(spectrogram, num_freq_masks=2, num_time_masks=2)

        return augment_spectrogram, class_id, audio_file


# define the CNN architecture
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        # activation function
        self.act = nn.ReLU()

        # first convolution layer
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.batch1 = nn.BatchNorm2d(8)
        layers += [self.conv1, self.act, self.batch1]

        # second convolution layer
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch2 = nn.BatchNorm2d(32)
        layers += [self.conv2, self.act, self.batch2]

        # third convolution layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch3 = nn.BatchNorm2d(64)
        layers += [self.conv3, self.act, self.batch3]

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(128)
        layers += [self.conv4, self.act, self.batch4]

        # linear layer and adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=128, out_features=10)

        self.convolution = nn.Sequential(*layers)

    # forward propogation
    def forward(self, x):
        x = self.convolution(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)

        return x


# load the model from training, and define criterion
def model_definition():
    # load the model from training
    model = AudioClassifier()
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model = model.to(device)

    # define criterion
    criterion = nn.CrossEntropyLoss()

    # output model summary
    print(model, file=open('model_summary.txt', 'w'))

    return model, criterion


# test the model
def testing_model(test_ds):
    # load model and criterion
    model, criterion = model_definition()
    model.load_state_dict(torch.load('Best_Model/best_model.pt', map_location=device))

    test_loss, steps_test, corr_pred_test, total_pred_test = 0, 0, 0, 0
    test_real_labels, test_pred, file_paths = [], [], []

    with torch.no_grad():
        with tqdm(total=len(test_ds), desc="Test Set -> ") as pbar:

            for x_data, x_target, idx in test_ds:
                x_data, x_target = x_data.to(device), x_target.to(device)

                output = model(x_data)
                loss = criterion(output, x_target)

                test_loss += loss.item()
                steps_test += 1

                _, prediction = torch.max(output, 1)

                corr_pred_test += (prediction == x_target).sum().item()
                total_pred_test += prediction.shape[0]

                # append real_labels, predictions, and file_paths for results.xlsx
                test_real_labels.append(x_target.detach().cpu().numpy())
                test_pred.append(prediction.detach().cpu().numpy())
                file_paths.append(list(idx))

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {test_loss / steps_test:0.5f} '
                                     f'--> Acc: {corr_pred_test / total_pred_test:0.5f}')

    # calculate test loss and test accuracy
    avg_loss_test = test_loss / len(test_ds)
    acc_test = corr_pred_test / total_pred_test
    print(f'Testing: Loss: {avg_loss_test:0.5f} --> Accuracy: {acc_test:0.5f}\n')

    # write all predictions and true results to excel file
    file_names = np.concatenate(file_paths)
    results_df = pd.DataFrame(file_names, columns=['file_names'])
    real_labels, pred = np.concatenate(test_real_labels), np.concatenate(test_pred)
    results_df['real_labels'], results_df['predictions'] = real_labels, pred
    results_df['file_names'] = results_df['file_names'].map(lambda x: x.lstrip('Data/urbansound8k/'))

    results_df.to_excel('Results/model_results.xlsx', index=False)

    # define more metrics
    acc_metric = accuracy_score(real_labels, pred)
    f1_metric = f1_score(real_labels, pred, average='weighted')

    print(f'Final Accuracy: {acc_metric:0.6f}')
    print(f'F1 Score (weighted): {f1_metric:0.6f}')


# main
if __name__ == '__main__':
    # use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    # read in excel file
    metadata = pd.read_csv('Data/urbansound8k/UrbanSound8K.csv')
    metadata['file_path'] = '/fold' + metadata['fold'].astype(str) + '/' + metadata['slice_file_name'].astype(str)

    # get the file path and class ID
    data = metadata[['file_path', 'classID']]

    # get the data
    dataset = UrbanSoundsDS(data, 'Data/urbansound8k')

    # get length of data for training, validation, testing
    data_len = len(dataset)
    train_len = round(data_len * 0.70)
    val_len = round(data_len * 0.15)
    test_len = data_len - train_len - val_len

    # split training, validation and testing data
    train, validation, test = random_split(dataset, [train_len, val_len, test_len])

    # input testing data into DataLoader
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    # test the model
    testing_model(test_dataloader)

# test acc using best model thus far (in Best_Model dir) -->
