
# Import the needed tools
import numpy as np
from sfc.util import create_dataset
from sfc.models.Attention_RNN import Attention_RNN

# Define some constants
DATA_PATH = './dataset/'
NUM_FEATURES = 24
My_Model_Logs_DIR = './logs/'
My_Model_Weights = My_Model_Logs_DIR + 'Best_Attention_RNN_ckpt.h5'

# Create the dataset
features, labels = create_dataset(DATA_PATH, NUM_FEATURES)

# Create the model
My_Model = Attention_RNN(   Logs_DIR=My_Model_Logs_DIR,
                            LSTM_units=32,
                            seq_n_timesteps=4,
                            seq_n_features_in=24,
                            seq_n_features_out=1,
                            Print_Model_Summary=True)

# Load the data into the model
My_Model.x_train = features
My_Model.y_train = labels
My_Model.x_test = features
My_Model.y_test = labels

# Train the model
# My_Model.Fit()

# Evaluate the model
# My_Model.Load_Model(My_Model_Weights)
# My_Model.Evaluate()

# Predict using the model
label = My_Model.Predict(features[50])
print(label)
