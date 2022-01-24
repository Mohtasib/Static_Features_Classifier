
"""
The attention layer we used in this model was copied from:
https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/

"""

# !pip install tensorflow-gpu==1.14.0
# !pip install keras==2.2.5

import os
import numpy as np
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Flatten
from keras.layers import LSTM, Dense

from sfc.models.AttentionDecoder import AttentionDecoder

class Attention_RNN():
    def __init__(   self, 
                    Logs_DIR, 
                    LSTM_units=32,
                    seq_n_timesteps=4,
                    seq_n_features_in=24,
                    seq_n_features_out=1,
                    Print_Model_Summary=False):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.LSTM_units = LSTM_units
        self.seq_n_timesteps = seq_n_timesteps
        self.seq_n_features_in = seq_n_features_in
        self.seq_n_features_out = seq_n_features_out
        self.model = None
        self.Logs_DIR = Logs_DIR
        os.makedirs(self.Logs_DIR, exist_ok=True)
        self.Print_Model_Summary = Print_Model_Summary
        self.Weights = self.Logs_DIR + 'Attention_RNN_Weights.h5'
        self.Model_Summary = self.Logs_DIR + 'Attention_RNN_Summary.txt'
        self.model = self.Create_Model(Print_Model_Summary=self.Print_Model_Summary)   

    def Create_Model(self, Print_Model_Summary=False):
        model = Sequential()
        model.add(LSTM( self.LSTM_units, 
                        input_shape=(self.seq_n_timesteps, self.seq_n_features_in),
                        return_sequences=True))
        model.add(AttentionDecoder(self.LSTM_units, self.seq_n_features_in))
        # model.add(Dense(self.seq_n_timesteps,activation='sigmoid',trainable=True))
        model.add(Flatten())
        model.add(Dense(self.seq_n_features_out,activation='sigmoid',trainable=True))

        model.compile( optimizer = "adam",
                            loss = "binary_crossentropy", 
                            metrics = ['accuracy'])

        if Print_Model_Summary:
            print(model.summary())
        # Open the file
        with open(self.Model_Summary,'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        self.model = model
        return model

    def Fit(self, x_train=None, y_train=None,
            patience=15,
            batch_size=32,
            epochs=100,
            validation_split=0.2):
        if x_train is None:
            x_train = self.x_train
        if y_train is None:
            y_train = self.y_train
        # simple early stopping
        es = EarlyStopping( monitor='val_loss',
                            mode='min',
                            verbose=1,
                            patience=patience)

        mc = ModelCheckpoint(   filepath=self.Logs_DIR + "/Best_Attention_RNN_ckpt.h5",
                                save_best_only=True,
                                monitor='val_loss',
                                mode='min',
                                verbose=1)

        history = self.model.fit(   x_train,  y_train, 
                                    batch_size = batch_size, 
                                    epochs = epochs, 
                                    validation_split = validation_split,
                                    verbose = 1,
                                    callbacks=[es, mc])
        self.model.save(self.Weights)
        return history

    def Load_Model(self, MODEL_PATH):
        self.model.load_weights(MODEL_PATH)

    def Predict(self, data):
        # Check the data dimenisons, it should be three dimensions
        if len(data.shape) < 3:
            data = np.expand_dims(data, axis=0)
        return self.model.predict(data, verbose=0)[:, 0]

    def Evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test

        y_probs = self.model.predict(x_test)[:, 0]
        y_hat = y_probs.round(0)
        y_test = y_test[:, 0]

        model_acc = accuracy_score(y_test, y_hat)
        model_f1 = f1_score(y_test, y_hat)
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_probs)
        model_auc = auc(lr_recall, lr_precision)
        model_precision = precision_score(y_test, y_hat), 
        model_recall = recall_score(y_test, y_hat)
        model_cm = confusion_matrix(y_test, y_hat)
        
        print('=================================')
        print("Attention_RNN: ")
        print("Accuracy = %0.3f" %(model_acc))
        print("Precision = %0.3f" %(model_precision))
        print("Recall = %0.3f" %(model_recall))
        print("F1 = %0.3f" %(model_f1))
        print("AUC = %0.3f" %(model_auc))
        print("CM = ", model_cm)
        print('=================================')

    def version(self):
        return "0.1.0"
