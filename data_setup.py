import numpy as np
import pickle
import os

"""
Includes functions to load the RML2016 dataset and load it as numpt arrays and other data manipulation.
"""

# A function which loads the RML2016 data as bytes
def load_data(path:str) -> dict:
    """
    A function to load the RML2016 dataset as bytes
    Args:
        path : is a string where the file is located
    Returns:
        the dataset as a dictionary
    """
    with open(path , 'rb') as f:
        data = pickle.load(f, encoding= "bytes")

    return data



# Function to get the modulation names from the RML2016 dataset
def modulation_name_list(data : dict) -> list:
    """
    A funtion which recieves a dict and retunrs the the key namse as list of bytes!
    Args:
        data : its a dict, which is loaded using pickle.load
    Returns:
        A list of modulation names which are in byte not str
    """
    list_of_modulation  = []

    for key in data.keys():
        list_of_modulation.append(key[0]) # append the first element because the keys are in (mod , snr) format
    
    return list(set(list_of_modulation))




# Function to filter the data on desired modulations
def filter_on_modulation(data : dict , keys_to_exclude : list = None, keys_to_inculde : list = None) -> dict:
    """
    A function to exclude the modulation from the main RML data
    Args:
        data : which a dict
        keys_to_exclude : its a list of bytes, becasue the keys from the original data is loaded as bytes
    Returns:
        return another dict without the modulations specified 
        """
    if keys_to_inculde:
        filtered_data = {key: value for key , value in data.items() if key[0] in keys_to_inculde}
    else :
        filtered_data = {key: value for key , value in data.items() if key[0] not in keys_to_exclude}

    return filtered_data



def filter_on_snr(data:dict , snr_range:tuple):

    filtered_data = {key: value for key , value in data.items() if key[1] in range(snr_range[0],snr_range[1])}
    
    return filtered_data

def groupe_labels(labels : list, keys: list, target_label):
    labels_copy = np.copy(labels)
    for index in range(len(labels_copy)):
        if labels_copy[index,0] in keys:
            labels_copy[index,0] = target_label

    return labels_copy

# A function to create splits of train and test split for RML2016 dataset
def create_train_test_splits(data : dict , ratio : int) -> np.array:
    """
    This function recieves the RML dataset and splits each modulation with respect to raito into
    train and test numpy arrays
    Args:
        data : which is RML2016
        ratio : is an int which uses to split each modulation
    Returns:
        4 splits of np.array of train , test and their respective targets, ecah target is (mod_type , snr)
    """
    data_copy = data.copy()

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    train_test_ratio = ratio

    for items in data_copy:
        mod , snr = items
        
        num_train_samples = int(data_copy[items].shape[0]* (1 - train_test_ratio))
        num_test_samples = int(data_copy[items].shape[0] - num_train_samples )

        train_data.append(data_copy[items][:num_train_samples])
        train_labels.append([[mod, snr]] * num_train_samples)
        test_data.append(data_copy[items][num_train_samples:])
        test_labels.append([[mod, snr]] * num_test_samples)

    train_data_np = np.array(train_data).reshape((-1,2,128))
    test_data_np = np.array(test_data).reshape((-1,2,128))
    train_labels_np = np.array(train_labels).reshape((-1,2))
    test_labels_np = np.array(test_labels).reshape((-1,2))

    return train_data_np , train_labels_np , test_data_np , test_labels_np



# Function to load data as numpy arrays with their labels
def load_digital_mods(data: dict, ratio : int) -> np.array:
    """
    This function first filter outs the RML2016 data on analogue modulation
    then returns splits of train and test data with their labels as numpy arrays.
    Args:
        data : which is the loaded RML2016
        ratio : the ratio which to split train and test data
    Returns:
        train and test data and their lables and the label to one hot converter 
    """
    filtered_data = filter_on_modulation(data = data , keys_to_exclude = [b'AM-SSB', b'WBFM', b'AM-DSB'])
    filterd_mods = modulation_name_list(filtered_data)
    train_data , train_label , test_data , test_label = create_train_test_splits(filtered_data , ratio)

    return train_data , train_label , test_data , test_label, filterd_mods

