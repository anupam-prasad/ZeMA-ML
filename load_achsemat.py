import scipy.io
import numpy as np

"""
Load the .mat file with the training and validation data and reshape the dataset to a more managebale format
"""


def load_achsemat():
    try:
        achse_mat = scipy.io.loadmat('ZeMA_Matlab/Achse11_Szenario.mat')
    except FileNotFoundError:
        achse_mat = scipy.io.loadmat('/data/vedurm01/Achse11_Szenario.mat')

    trainData = achse_mat['trainData']
    train_data_reshaped = []
    for counter, val in enumerate(trainData):
        train_data_reshaped += [val[0]]
    train_data_reshaped = np.array(train_data_reshaped)
    train_data_reshaped = np.swapaxes(train_data_reshaped, 0, 1)

    train_target_reshaped = achse_mat['trainTarget'].reshape(-1)

    return train_data_reshaped, train_target_reshaped
