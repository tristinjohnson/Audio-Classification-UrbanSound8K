"""
Tristin Johnson
Final Project - UrbanSound8K
DATS 6203 - Machine Learning II
December 6, 2021
"""
# import required packages
import random
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from tqdm import tqdm

##################################################################################################
#
# This is the script to train, validate, and save your best model. Before running, make sure you
# have downloaded the data using the 'download_data.py' script inside of the Data directory.
# Furthermore, do not move the data outside of the Data directory, or this script will fail.
# After running this file, your best model will be save as 'model_urbansounds8k.pt' in this
# directory, along with a 'model_summary.txt', which shows the model architecture. The model
# will only be overwritten if your validation accuracy score beat the previous epoch's
# validation score. Other than that, have fun!
#
##################################################################################################


# define variables
num_epochs = 20
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

        # randomize time shift
        shift_audio = random_time_shift(pad_audio, self.shift_pct)

        # get mel spectrogram
        spectrogram = mel_spectrogram(shift_audio)

        # augment the spectrogram
        augment_spectrogram = data_augmentation(spectrogram, num_freq_masks=2, num_time_masks=2)

        return augment_spectrogram, class_id


# define the CNN architecture
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # first convolution layer
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.batch1 = nn.BatchNorm2d(8)
        self.pad1 = nn.ZeroPad2d(2)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2)

        # second convolution layer
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch2 = nn.BatchNorm2d(32)
        self.pad2 = nn.ZeroPad2d(2)

        # third convolution layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch3 = nn.BatchNorm2d(64)

        # fourth convolution layer
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(128)

        # activation func, linear layer, adaptive pooling
        self.act = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear1 = nn.Linear(in_features=128, out_features=10)

    # forward propogation
    def forward(self, x):
        x = self.pool1(self.pad1(self.batch1(self.act(self.conv1(x)))))
        x = self.pad2(self.batch2(self.act(self.conv2(x))))
        x = self.batch3(self.act(self.conv3(x)))
        x = self.batch4(self.act(self.conv4(x)))

        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)

        return x


# save the model
def save_best_model(model):
    # save the best model in training
    print(model, file=open('model_summary.txt', 'w'))


# create the model definition: model, optimizer, scheduler, criterion
def model_definition():

    # define model
    model = AudioClassifier()
    model = model.to(device)

    # define optimizer, scheduler, criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # save a summary of the model architecture
    save_best_model(model)

    return model, optimizer, scheduler, criterion


# define training and validation loop
def training_and_validation(train_ds, validation_ds):
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

        # evaluate the model using validation set
        model.eval()

        val_loss, steps_val, corr_pred_val, total_pred_val = 0, 0, 0, 0

        # validating the model
        with torch.no_grad():
            with tqdm(total=len(validation_ds), desc=f'Epoch {epoch}') as pbar:

                for x_data, x_target in validation_ds:
                    x_data, x_target = x_data.to(device), x_target.to(device)

                    # normalize inputs
                    x_data_mean, x_data_std = x_data.mean(), x_data.std()
                    x_data = (x_data - x_data_mean) / x_data_std

                    optimizer.zero_grad()
                    output = model(x_data)
                    loss = criterion(output, x_target)

                    val_loss += loss.item()
                    steps_val += 1

                    _, prediction = torch.max(output, 1)

                    corr_pred_val += (prediction == x_target).sum().item()
                    total_pred_val += prediction.shape[0]

                    pbar.update(1)
                    pbar.set_postfix_str(f"Loss: {val_loss / steps_val:0.5f} "
                                         f"--> Acc: {corr_pred_val / total_pred_val:0.5f}")

        # output training metrics
        avg_loss_train = train_loss / len(train_ds)
        acc_train = corr_pred_train / total_pred_train
        print(f'Training: Epoch {epoch} --> Loss: {avg_loss_train:0.5f} --> Accuracy: {acc_train:0.5f}')

        # output validation metrics
        avg_loss_val = val_loss / len(validation_ds)
        acc_val = corr_pred_val / total_pred_val
        print(f'Validation: Epoch {epoch} --> Loss: {avg_loss_val:0.5f} --> Accuracy: {acc_val:0.5f}\n')

        # define model accuracy
        model_acc = acc_val

        # if epoch validation accuracy is better than previous, save the model
        if model_acc > model_best_acc:
            torch.save(model.state_dict(), 'model_urbansounds8k.pt')

            if epoch == 0:
                print('This model has been saved!\n')
            else:
                print('This model out-performed previous models and has been saved!\n')

            model_best_acc = model_acc

    print('Training and Validation complete!')


# main
if __name__ == '__main__':
    # use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Device: ', device)

    # read excel file for more information
    metadata = pd.read_csv('Data/urbansound8k/UrbanSound8K.csv')
    metadata['file_path'] = '/fold' + metadata['fold'].astype(str) + '/' + metadata['slice_file_name'].astype(str)

    # get only file_path and classID
    data = metadata[['file_path', 'classID']]

    # get the data (with df and path to data folder)
    dataset = UrbanSoundsDS(data, 'Data/urbansound8k')

    # get length of data to get training and validation data
    data_len = len(dataset)
    train_len = round(data_len * 0.70)
    val_len = round(data_len * 0.15)
    test_len = data_len - train_len - val_len

    # split training and validation data
    train, validation, test = random_split(dataset, [train_len, val_len, test_len])

    # input training and validation data into Dataloaders
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation, batch_size=batch_size, shuffle=False)

    # train
    training_and_validation(train_dataloader, validation_dataloader)
