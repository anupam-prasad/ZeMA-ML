"""
Loads the best parameters estimated using hyperopt and compiles and fits the model to the training data.
The model weights are saved/serialized in hdf5 format.
"""
import pickle
import sys

from sklearn.preprocessing import StandardScaler

from load_achsemat import load_achsemat
from zema_generate_model import generate_model

def save_model_weights(NN_params):
    best_model = generate_model(NN_params)
    print(best_model.summary())

    best_model.compile(
        optimizer=NN_params['optimizer'], loss="mean_squared_error", metrics=["mse"]
    )

    trainData, trainTarget = load_achsemat()
    try:
        arg1 = sys.argv[1]
        print('scaling training data')
        for k in range(11):
            trainData[:, k, :] = StandardScaler().fit_transform(trainData[:, k, :])
    except IndexError:
        print('proceeding without scaling')

    best_model.fit(trainData, trainTarget, epochs=100, validation_split=0.1, verbose=0)
    best_model.save_weights("best_model_simple.h5")

if __name__ == "__main__":
    with open('best_parameters_simple.dict', "rb") as f:
        params = pickle.load(f)
    print(params)

    save_model_weights(params)