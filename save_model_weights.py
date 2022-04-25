"""
Loads the best parameters estimated using hyperopt and compiles and fits the model to the training data.
The model weights are saved/serialized in hdf5 format.

Unless specified, i.e. running the script without any arguments, the parameters for unscaled data are loaded.
"""
import pickle
from zema_generate_model import save_model
import sys


if __name__ == "__main__":

    try:
        arg1 = sys.argv[1]
        print('loading model for scaled data')
        with open('best_parameters_simple.dict', "rb") as f:
            params = pickle.load(f)
    except IndexError:
        print('loading model for unscaled data')
        with open('best_parameters_simple_noscaling.dict', "rb") as f:
            params = pickle.load(f)

    print(params)

    save_model(params)
