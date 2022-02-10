
# !pip install tensorflow-gpu==1.14.0
# !pip install keras==2.2.5

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sfc.models.BaseModel import BaseModel
from sfc.util import get_files_list

class FF(BaseModel):
    def __init__(   self, 
                    Logs_DIR, 
                    num_classes=1, 
                    num_features=640, 
                    optimizer = "adam",
                    loss = "binary_crossentropy", 
                    metrics = ['accuracy'],
                    Print_Model_Summary=False):
        self.num_classes = num_classes
        self.num_features = num_features
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        super(FF, self).__init__(Logs_DIR=Logs_DIR, Print_Model_Summary=Print_Model_Summary)

    def load_from_csv(self, path, num_features=None):
        """Reads the data for one map from a .csv file

        Parameters
        ----------
        path : str
            path to the .csv file
        num_features : int, optional
            the number of features that describe the object

        Returns
        -------
        numpy.ndarray
            a numpy array of the features [num_objects x num_timesteps x num_features]
        numpy.ndarray
            a numpy array of the labels [num_objects x 1 x 1]
        """
        if num_features is None:
            num_features = self.num_features

        # Load the data
        data = pd.read_csv(path)

        # Get the features
        features = []
        columns = list(data.columns)[1:-1]
        features = pd.DataFrame(data, columns=columns).to_numpy()

        # Get the labels
        labels =  pd.DataFrame(data, columns=['Label']).to_numpy()
        if self.num_classes > 1:
            labels = to_categorical(labels, num_classes=self.num_classes)

        return features, labels


    def _create_model(  self,
                        num_features=None, 
                        num_classes=None, 
                        ):
        if num_features is None:
            num_features = self.num_features
        if num_classes is None:
            num_classes = self.num_classes

        model = Sequential()
        model.add(Dense(512, input_shape=(num_features,), activation='relu'))
        model.add(Dense(256, input_shape=(num_features,), activation='relu'))
        model.add(Dense(64, input_shape=(num_features,), activation='relu'))
        model.add(Dense(32, input_shape=(num_features,), activation='relu'))

        if self.num_classes > 1:
            model.add(Dense(num_classes, activation='softmax'))
        else:
            model.add(Dense(num_classes, activation='sigmoid'))
        return model