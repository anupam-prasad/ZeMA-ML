import os
import pickle

import keras.backend as kb
from hyperopt import fmin, hp, STATUS_OK, Trials, space_eval, tpe
from hyperopt.pyll import scope
from keras import Sequential
from keras.layers import Input, Dense, Dropout, Flatten

from load_achsemat import load_achsemat as load_axis_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def node_params(n_layers):
    """
    Function to dynamically define a search space for a given number of layers, where each layer can have a varying
    number of units and a given dropout rate
    :param n_layers:
    :return: dictionary where the key refers to the layer number and the value is a tuple consisting of the number of
             units in the layer and its corresponding dropout rate
    """
    params = {}
    for n in range(1, n_layers + 1):
        params['layer_{}'.format(n)] = {
            'n_units': scope.int(hp.quniform('n_nodes_{}_{}'.format(n_layers, n), 100, 1000, 50)),
            'dropout_rate'.format(n): hp.quniform('dropout_rate_{}_{}'.format(n_layers, n), .0, .75, .05),
            'activation': hp.choice('activation_{}_{}'.format(n_layers, n), ['relu', 'tanh', None])}
    return params


def generate_model(params):
    generated_model = Sequential()
    generated_model.add(Input(shape=(11,2000)))


    n_layers = len(params['layers'])
    for n_layer in range(1, n_layers + 1):
        generated_model.add(Dense(params['layers']['layer_{}'.format(n_layer)]['n_units'],
                                  activation=params['layers']['layer_{}'.format(n_layer)]['activation'],
                                  use_bias=params.get('use_bias', None)))
        generated_model.add(Dropout(rate=params['layers']['layer_{}'.format(n_layer)]['dropout_rate']))

    generated_model.add(Flatten())
    
    return generated_model


def lossfn_nn(params):
    print('Testing params: ', params)
    model = generate_model(params)

    model.compile(
        optimizer=params['optimizer'], loss="mean_squared_error", metrics=["mse"]
    )

    trainData, trainTarget = load_axis_data()

    history = model.fit(trainData, trainTarget, epochs=100, validation_split=.1, verbose=0)

    nneval_out = history.history['loss']

    return {'loss': nneval_out, 'status': STATUS_OK}


def generate_search_space(n_layers=6):
    # generates a search space dictionary

    # List of the number of layers you want to consider. We allow for up to 6 NN layers
    layer_options = list(range(1, n_layers))

    # Defines the search space in which hyperopt searches for the optimal configuration
    # Dynamically build the space based on the possible number of layers
    layer_space = hp.choice('layers', [node_params(n) for n in layer_options])

    search_space = {'layers': layer_space,
                    'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
                    'use_bias': hp.choice('use_bias', [True, False])
                    }

    return search_space

if __name__ == "__main__":
    space = generate_search_space()
    trials = Trials()
    best = fmin(lossfn_nn, space, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best: ', best)
    print(space_eval(space, best))
    with open('zema_ml_simple' + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)
   
    with open('best_parameters_simple.dict', "wb") as f:
        pickle.dump(space_eval(space, best), f)
