import os
import numpy as np
import pandas as pd
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

from sfc.util import get_files_list

class BaseModel():
    def __init__(self, Logs_DIR, Print_Model_Summary=False):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.Logs_DIR = Logs_DIR
        self.Print_Model_Summary = Print_Model_Summary
        os.makedirs(self.Logs_DIR, exist_ok=True)
        self.class_name = self.__class__.__name__
        self.save_path = self.Logs_DIR + self.class_name
        self.model = self.Create_Model(Print_Model_Summary=self.Print_Model_Summary)   

    def load_from_csv(self, path):
        """Reads the data for one map from a .csv file

        Parameters
        ----------
        path : str
            path to the .csv file

        Returns
        -------
        numpy.ndarray
            a numpy array of features
        numpy.ndarray
            a numpy array of the labels
        """
        raise NotImplementedError

    def Create_Dataset(self, data_path):
        """Creates a dataset from the .csv files in the 'data_path' directory

        Parameters
        ----------
        path : str
            path to the .csv files


        Returns
        -------
        numpy.ndarray
            a numpy array of the features
        numpy.ndarray
            a numpy array of the labels
        """
        # Get a list of the .csv file
        maps_data = get_files_list(data_path)

        # Load the data from the .csv files
        all_features, all_labels = [], []
        for map_data in maps_data:
            features, labels = self.load_from_csv(path=map_data)
            all_features.append(features)
            all_labels.append(labels)

        return np.vstack(all_features), np.vstack(all_labels)

    def _create_model(self):
        raise NotImplementedError

    def Create_Model(self, 
                    optimizer = None,
                    loss = None, 
                    metrics = None,
                    Print_Model_Summary=False,
                    ):
        if optimizer is None:
            optimizer = self.optimizer
        if loss is None:
            loss = self.loss
        if metrics is None:
            metrics = self.metrics

        model = self._create_model()

        model.compile(  optimizer = optimizer,
                        loss = loss, 
                        metrics = metrics)

        if Print_Model_Summary:
            print(model.summary())
        # Open the file
        with open(self.save_path + '_Summary.txt','w') as fh:
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

        mc = ModelCheckpoint(   filepath=self.save_path + "_Weights.h5",
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
        # self.model.save(self.save_path + '_Weights.h5')
        self._save_train_history(history)
        return history

    def _save_train_history(self, history):
        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 

        # save to csv: 
        hist_csv_file = self.save_path + '_Train_History.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

    def Load_Model(self, MODEL_PATH=None):
        if MODEL_PATH is None:
            MODEL_PATH = self.save_path + "_Weights.h5"
        self.model.load_weights(MODEL_PATH)

    def Predict(self, data):
        # Check the data dimenisons, it should be three dimensions
        if len(data.shape) < len(self.model.layers[0].input_shape):
            data = np.expand_dims(data, axis=0)
        return self.model.predict(data, verbose=0)[:, 0]

    def Evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test

        y_probs = self.model.predict(x_test)
        y_hat = y_probs.argmax(1)

        if self.num_classes > 1:
            y_test = y_test[:, 1]
            y_probs = y_probs[:, 1]

        self._calculate_scores(y_test, y_probs, y_hat)

    def _calculate_scores(self, y_test, y_probs, y_hat):
        model_acc = accuracy_score(y_test, y_hat)
        model_f1 = f1_score(y_test, y_hat)
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_probs)
        model_auc = auc(lr_recall, lr_precision)
        model_precision = precision_score(y_test, y_hat), 
        model_recall = recall_score(y_test, y_hat)
        model_cm = confusion_matrix(y_test, y_hat)
        
        results = []
        results.append('=================================')
        results.append(f"{self.class_name}: ")
        results.append("Accuracy = %0.3f" %(model_acc))
        results.append("Precision = %0.3f" %(model_precision))
        results.append("Recall = %0.3f" %(model_recall))
        results.append("F1 = %0.3f" %(model_f1))
        results.append("AUC = %0.3f" %(model_auc))
        results.append(f"CM = {np.array2string(model_cm)}")
        results.append('=================================')

        textfile = open(self.save_path + '_Evaluation_Results.txt', "w")
        for element in results:
            textfile.write(element + "\n")
            print(element)
        textfile.close()

    def version(self):
        return "0.1.0"