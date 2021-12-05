# Instructions to Run the Code

## 1. Downlaod the Data

After you have downloaded this repository, go into the Data directory and run the 'download_data.py' script to download all of the data. For more information regarding this script, click [here](https://github.com/tristinjohnson/Final-Project-Group2/tree/main/Code/Data).


## 2. Train, Validate, and Save the Model

Once the data has been downloaded, you now can run the training script to train the model. Simply type in the following command to run the script:

    python3 train_validate_model.py
    
This script will take around 30 - 35 minutes to run 20 epochs (on GPU). If you would like to make this shorter, simply change the number of epochs at the beginning of the script. Once the model has been trained and validated, you will notice two additional files in this directory. The first being 'model_summary.txt', which is the architecture of the model. The second file is 'model_urbansounds8k.pt'. This is loaded from the training script, in which the script will save the best model from training. Make sure to not delete the model as it is needed for testing the model. 


## 3. Test the Model

After training the model, now you can test the performance of the model. To do so, run the following command:

    python3 test_model.py
    
There are two options when running the 'test_model.py' script. There is a '--model' flag where you can either use your generated model from the training script in step 2, or you can use my best and most accurate model I was able to achieve. If you don't use the '--model' flag, the script will default to your generated model from training. See below examples on how to use:

    # to use your model generated from training
    python3 test_model.py --model user_model
    
    # to use my best and most accurate model
    python3 test_model.py --model pretrained_model

This will test the model that was generated from the training script above. Once you have tested the model, an excel file will be generated in the 'Results' folder that displays the results of the model. There are 3 columns 'file_names', 'real_labels', 'predictions'. Here you can see how well your model predicted certain classes of the audio files. In regards to my model, feel free to change the architecture of the training script and see if you can get a better accuracy score than my model!

## 4. UrbanSounds8K Metadata Analysis

In order to analyze information about the dataset, you can run the 'urbansounds_metadata_analysis.py' script. This will display multiple analysis regarding the data, such as the class information, what the audio files look like (in terms of waveform and sample rate), what different sampling rates look like, and some transformations/augmentations that I used to train my model!

Make sure to keep the metadata analysis file in the same directory as the 'train_validate_model.py' file as the analysis uses multiple preprocessing functions from the training file. 

