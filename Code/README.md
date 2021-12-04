# Instructions to Run the Code

## 1. Downlaod the Data

After you have downloaded this repository, go into the Data directory and run the 'download_data.py' script to download all of the data. For more information regarding this script, click [here](https://github.com/tristinjohnson/Final-Project-GroupX/tree/main/Code/Data).


## 2. Train, Validate, and Save the Model

Once the data has been downloaded, you now can run the training script to train the model. Simply type in the following command to run the script:

    python3 train_validate_model.py
    
This script will take around 30 - 35 minutes to run 20 epochs (on GPU). If you would like to make this shorter, simply change the number of epochs at the beginning of the script. Once the model has been trained and validated, you will notice two additional files in this directory. The first being 'model_summary.txt', which is the architecture of the model. The second file is 'model_urbansounds8k.pt'. This is loaded from the training script, in which the script will save the best model from training. Make sure to not delete the model as it is needed for testing the model. 


## 3. Test the Model

After training the model, now you can test the performance of the model. To do so, run the following command:

    python3 test_model.py
    
This will test the model that was generated from the training script above. Once you have tested the model, an excel file will be generated in the 'Results' folder that displays the results of the model. There are 3 columns 'file_names', 'real_labels', 'predictions'. Here you can see how well your model predicted certain classes of the audio files. 


# Extras for UrbanSounds8k Final Project (Optional)

## 1. Metadata Analysis

In order to analyze information about the dataset, you can run the 'wav_analysis.py' script. This will display multiple analysis regarding the data, such as, the class information, what the audio files look like (in terms of waveform and sample rate), all the shapes of the audio files after each transformation, a Mel Spectrogram, and moore.

## 2. Test Your Best Model vs. My Best Model

In the Best_Model directory, I have my best model from training. If you would like to use this model for testing, change the code in the 'train_model.py' on line 264 to:

    model.load_state_dict(torch.load('Best_Model/model_urbansounds8k.pt', map_location=device))
    
Feel free to change the architecture of the model in the training script and see if you can create a model that beats my accuracy score!
