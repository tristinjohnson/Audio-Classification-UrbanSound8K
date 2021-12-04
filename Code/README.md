# Instructions to Run the Code

## 1. Downlaod the Data

After you have downloaded this repository, go into the Data directory and run the 'download_data.py' script to download all of the data. For more information regarding this script, click [here](https://github.com/tristinjohnson/Final-Project-GroupX/tree/main/Code/Data).


### 2. Run the Training Script

Once the data has been downloaded, you now can run the training script to train the model. Simply type in the following command to run the script:

    python3 train_validate_model.py
    
This script will take around 30 - 35 minutes to run 20 epochs. If you would like to make this shorter, simply change the number of epochs at the beginning of the script. Once the model has been trained and validated, you will notice two additional files in this directory. The first being 'model_summary.txt', which is the architecture of this model. The second file is 'model_urbansounds8k.pt', which is the best model from training. Make sure to not delete the model as it is needed for testing the model. 
