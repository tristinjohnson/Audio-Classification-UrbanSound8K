# Download UrbanSounds8K Dataset

In order to download the data into this directory, simply run the python script in this directory (preferably in your terminal):

    python3 download_data.py
    
    
To efficiently download this data, it was much easier and quicker to download straight from Kaggle.com. Therefore, you will need your Kaggle username and Kaggle API key to download the data.

Make sure to not reshuffle any of the data. The data comes with a predefined 10-fold cross validation. 

Furthermore, make sure you have Pytorch installed in your local IDE. The above script will also install 3 important packages needed to run all of the training and testing scripts:

    opendatasets, librosa, torchaudio
    
## ***** THIS SCRIPT MAY FAIL WHEN YOU FIRST RUN IT! *****

Since you are downloading a package that is directly used in the download file (opendatasets), you may have to run the script a second time in order to successfully download the data.
    
Once you have ran the script, you will see a folder called 'urbansound8k'. When you enter this directory, you will see the 10 fold cross validation (in folders labeled fold1, fold2, ..., fold10) containing all of the audio file (in .wav format), and a .csv file containing all of the metadata about the audio files 'UrbanSound8K.csv'. Now that the data has been downloaded, you can run the training script to train and validate the model.
