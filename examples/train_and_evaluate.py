
# Import the needed tools
import numpy as np
from sfc.models.Attention_RNN import Attention_RNN
from sfc.models.CNN_FF import CNN_FF
from sfc.models.FF import FF

# Define some constants
DATA_PATH = './dataset_sd/'
MAX_SEQ_LEN = 1
NUM_FEATURES = 640
LSTM_units = 32
Logs_DIR = './logs_sd/'


def TrainEvaluate(model, log_dir, data_path):
    My_Model_Logs_DIR = log_dir + model + '/'

    # Create the model
    if model is 'FF': 
        My_Model = FF(  Logs_DIR=My_Model_Logs_DIR,
                        num_features=NUM_FEATURES,
                        Print_Model_Summary=True)
    if model is 'CNN_FF': 
        My_Model = CNN_FF(   Logs_DIR=My_Model_Logs_DIR,
                                num_features=NUM_FEATURES,
                                Print_Model_Summary=True)
    if model is 'Attention_RNN' : 
        My_Model = Attention_RNN(   Logs_DIR=My_Model_Logs_DIR,
                                    LSTM_units=LSTM_units,
                                    seq_n_timesteps=MAX_SEQ_LEN,
                                    seq_n_features_in=NUM_FEATURES,
                                    Print_Model_Summary=True)

    # Create the dataset and load it into the model
    My_Model.x_train, My_Model.y_train = My_Model.Create_Dataset(data_path + 'train/')
    My_Model.x_test, My_Model.y_test = My_Model.Create_Dataset(data_path + 'test/')

    # Train the model
    My_Model.Fit()

    # Evaluate the model
    My_Model.Load_Model()
    My_Model.Evaluate()

    # # Predict using the model
    label = My_Model.Predict(My_Model.x_test)


if __name__ == '__main__':
    for model in ['FF', 'CNN_FF', 'Attention_RNN']:
        TrainEvaluate(model, Logs_DIR, DATA_PATH) # TODO: you should also pass the model_params