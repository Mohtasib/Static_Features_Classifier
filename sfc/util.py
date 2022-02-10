import os


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
