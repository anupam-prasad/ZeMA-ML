import scipy.io

"""
Load the .mat file with the training and validation data
"""
achse_mat = scipy.io.loadmat('ZeMA_Matlab/Achse11_Szenario.mat')

#extract the training data and target values. note the weird matlab formatting

trainData = achse_mat['trainData']