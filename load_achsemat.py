import scipy.io

"""
Load the .mat file with the training and validation data
"""


def load_achsemat():
    try:
        achse_mat = scipy.io.loadmat('ZeMA_Matlab/Achse11_Szenario.mat')
    except FileNotFoundError:
        achse_mat = scipy.io.loadmat('/data/vedurm01/Achse11_Szenario.mat')
    return achse_mat
