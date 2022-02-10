
"""
The attention layer we used in this model was copied from:
https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/

"""

# !pip install tensorflow-gpu==1.14.0
# !pip install keras==2.2.5

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Flatten

from sfc.models.AttentionDecoder import AttentionDecoder
from sfc.models.BaseModel import BaseModel
from sfc.util import get_files_list

class Attention_RNN(BaseModel):
    def __init__(   self, 
                    Logs_DIR, 
                    LSTM_units=32,
                    seq_n_timesteps=4,
                    seq_n_features_in=24,
                    seq_n_features_out=1,
                    optimizer = "adam",
                    loss = "binary_crossentropy", 
                    metrics = ['accuracy'],
                    Print_Model_Summary=False):
        self.LSTM_units = LSTM_units
        self.seq_n_timesteps = seq_n_timesteps
        self.seq_n_features_in = seq_n_features_in
        self.seq_n_features_out = seq_n_features_out
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        super(Attention_RNN, self).__init__(Logs_DIR=Logs_DIR, Print_Model_Summary=Print_Model_Summary)  

    def load_from_csv(self, path, seq_n_timesteps=None, num_features=None):
        """Reads the data for one map from a .csv file

        Parameters
        ----------
        path : str
            path to the .csv file
        seq_n_timesteps : int
            the maximum number of timesteps in the input sequence
        num_features : int, optional
            the number of features for that describe the object at each timestep

        Returns
        -------
        numpy.ndarray
            a numpy array of the sequence of features [num_objects x num_timesteps x num_features]
        numpy.ndarray
            a numpy array of the labels [num_objects x 1 x 1]
        """
        if seq_n_timesteps is None:
            seq_n_timesteps = self.seq_n_timesteps
        if num_features is None:
            num_features = self.seq_n_features_in

        # Load the data
        data = pd.read_csv(path)

        # Calculate the number of timesteps for each map. neglect the first and the last columns then we divide by the number of features for each timestep (24 features)
        num_timesteps = int((len(data.columns) - 2) / num_features) 

        # Put the data into sequences
        features = []
        for i in range(num_timesteps):
            columns = [j for j in data.columns if j.startswith(f't{i}_')]
            arr = pd.DataFrame(data, columns=columns).to_numpy()
            arr = np.reshape(arr, (arr.shape[0], 1, arr.shape[1]))
            features.append(arr)

        features = np.hstack(features)

        # Get the labels
        labels =  pd.DataFrame(data, columns=['Label']).to_numpy()
        
        # Make sure to have the correct sequence length
        if seq_n_timesteps > num_timesteps:
            all_features = np.zeros((features.shape[0], seq_n_timesteps, features.shape[2]))
            all_features[:,features.shape[1]:,:] = features
        else:
            all_features = features

        return all_features, labels

    def _create_model(  self, 
                        seq_n_timesteps=None,
                        seq_n_features_in=None,
                        seq_n_features_out=None,
                        LSTM_units=None,
                        ):
        if seq_n_timesteps is None:
            seq_n_timesteps = self.seq_n_timesteps
        if seq_n_features_in is None:
            seq_n_features_in = self.seq_n_features_in
        if seq_n_features_out is None:
            seq_n_features_out = self.seq_n_features_out
        if LSTM_units is None:
            LSTM_units = self.LSTM_units

        model = Sequential()
        model.add(LSTM( LSTM_units, 
                        input_shape=(seq_n_timesteps, seq_n_features_in),
                        return_sequences=True))
        model.add(AttentionDecoder(LSTM_units, seq_n_features_in))
        model.add(Flatten())
        model.add(Dense(seq_n_features_out,activation='sigmoid',trainable=True))
        return model

    def Evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test

        y_probs = self.model.predict(x_test)[:, 0]
        y_hat = y_probs.round(0)
        y_test = y_test[:, 0]

        self._calculate_scores(y_test, y_probs, y_hat)
