import os

os.system("echo =================== Installing OpenDatasets for UrbanSound8K Data Download ===================")
os.system("pip install opendatasets")

os.system("echo =================== Installing TorchAudio for model execution ===================")
os.system("pip install torchaudio")

os.system("echo =================== Installing Librosa to load audio files ===================")
os.system("pip install librosa")


print('\n********************* DOWNLOADING urbansounds8k IN THIS DIRECTORY *********************\n')
print('This Dataset is being directly downloaded from Kaggle.com. Please make sure you have a '
      'Kaggle account with your Kaggle username and Kaggle API Key available in order to download the dataset\n')
print('Run the following command to download the dataset and enter your Kaggle username and API key: '
      'python3 download_data.py\n')


import opendatasets as od

# downloads the dataset from kaggle into directory
# this way is much more efficient that directly downloading data from UrbanSounds8k.com
data_url = 'https://www.kaggle.com/chrisfilo/urbansound8k'
od.download(data_url)
