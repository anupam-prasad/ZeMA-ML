"""
Loads the best parameters estimated using hyperopt and compiles and fits the model to the training data.
The model weights are saved/serialized in hdf5 format.
"""
import pickle
from zema_generate_model import save_model_weights



if __name__ == "__main__":
    with open('best_parameters_simple.dict', "rb") as f:
        params = pickle.load(f)
    print(params)

    save_model_weights(params)