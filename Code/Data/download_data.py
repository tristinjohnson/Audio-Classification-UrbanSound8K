import opendatasets as od

print('\n********************* DOWNLOADING urbansounds8k IN THIS DIRECTORY *********************\n')
print('This Dataset is being directly downloaded from Kaggle.com. Please make sure you have a '
      'Kaggle account with your Kaggle username and Kaggle API Key available in order to download the dataset\n')
print('Run the following command to download the dataset and enter your Kaggle username and API key: '
      'python3 download_data.py\n')

# downloads the dataset from kaggle into directory
# this way is much more efficient that directly downloading data from UrbanSounds8k.com
data_url = 'https://www.kaggle.com/chrisfilo/urbansound8k'
od.download(data_url)

# need to add 'pip install opendatasets
# need to add 'pip install torchaudio'
# need to add 'pip install librosa'
