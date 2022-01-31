import os
import numpy as np
import pandas as pd


def get_files_list(path, file_type='.csv'):
    """Generates a list of files in a directory that has a specific type

    Parameters
    ----------
    path : str
        path to the directory of interest
    file_type : str, optional
        the type of file you interested in

    Returns
    -------
    list
        a list of files of type 'file_type' in the directory 'path'
    """
    files = os.listdir(path)
    csv_list = [os.path.join(path, j) for j in files if j.endswith(file_type)]
    return csv_list

def load_from_csv(path, max_num_timesteps, num_features=24):
    """Reads the data for one map from a .csv file

    Parameters
    ----------
    path : str
        path to the .csv file
    max_num_timesteps : int
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
    if max_num_timesteps > num_timesteps:
        all_features = np.zeros((features.shape[0], max_num_timesteps, features.shape[2]))
        all_features[:,features.shape[1]:,:] = features
    else:
        all_features = features

    return all_features, labels

def create_dataset(data_path, max_num_timesteps, num_features=24):
    """Creates a dataset from the .csv files in the 'data_path' directory

    Parameters
    ----------
    path : str
        path to the .csv files
    max_num_timesteps : int
        the maximum number of timesteps in the input sequence
    num_features : int, optional
        the number of features for that describe the object at each timestep

    Returns
    -------
    numpy.ndarray
        a numpy array of the sequence of features [num_objects_in_all_maps x num_timesteps x num_features]
    numpy.ndarray
        a numpy array of the labels [num_objects_in_all_maps x 1 x 1]
    """
    # Get a list of the .csv file
    maps_data = get_files_list(data_path)

    # Load the data from the .csv files
    all_features, all_labels = [], []
    for map_data in maps_data:
        features, labels = load_from_csv(path=map_data, max_num_timesteps=max_num_timesteps, num_features=num_features)
        all_features.append(features)
        all_labels.append(labels)

    return np.vstack(all_features), np.vstack(all_labels)

