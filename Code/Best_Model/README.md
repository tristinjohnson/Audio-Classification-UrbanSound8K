# My Best Model

Above is the best and most accurate model I was able to achieve while training the network. This is here in case the user wants to test the model using my model. In order to use this model, type in the following command:

    # use my model
    python3 test_model.py --model pretrained_model
    
The test script will default to the user generated model that was saved after running the training script. You do not need to specify the '--model' flag if you are using your model after running the training script. But in case you would like to, use the following command:

    # use user generated model
    python3 test_model.py --model user_model
