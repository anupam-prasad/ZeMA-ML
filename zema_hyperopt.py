import os
import pickle
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from hyperopt import fmin, hp, STATUS_OK, Trials, space_eval, tpe
from hyperopt.pyll import scope
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

tf.config.threading.set_inter_op_parallelism_threads(10)
from load_achsemat import load_achsemat as load_axis_data
from zema_generate_model import generate_model

trainData, trainTarget = load_axis_data()
# scale the individual time-series data
try:
    arg1 = sys.argv[1]
    print('scaling training data')
    for k in range(11):
        trainData[:, k, :] = StandardScaler().fit_transform(trainData[:, k, :])
except IndexError:
    print('proceeding without scaling')


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
            'dropout': hp.choice('dropout_{}_{}'.format(n_layers, n), [
                {'choice': False},
                {'choice': True, 'dropout_rate': hp.quniform('dropout_rate_{}_{}'.format(n_layers, n), .1, .8, .1)}]),
            'activation': hp.choice('activation_{}_{}'.format(n_layers, n), ['relu', 'tanh', None])}
    return params


def lossfn_nn(params):
    print('Testing params: ', params)
    model = generate_model(params)

    model.compile(
        optimizer=params['optimizer'], loss="mean_squared_error", metrics=["mse"]
    )

    history = model.fit(trainData, trainTarget, epochs=100, validation_split=.1, verbose=0)

    nneval_out = history.history['loss']

    return {'loss': nneval_out[-1], 'status': STATUS_OK}


def generate_search_space(n_layers=6):
    # generates a search space dictionary according to the hyperopt format

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


def save_model(nn_params):
    best_model = generate_model(nn_params)
    print(best_model.summary())

    best_model.compile(
        optimizer=nn_params['optimizer'], loss="mean_squared_error", metrics=["mse"]
    )

    fname = 'best_model_simple'
    try:
        arg2 = sys.argv[1]
        print('scaling training data')
    except IndexError:
        fname += '_noscaling'
        print('proceeding without scaling')

    best_model.fit(trainData, trainTarget, epochs=100, validation_split=0.1, verbose=0)
    best_model.save("best_model_simple_noscaling"+".h5")


if __name__ == "__main__":
    space = generate_search_space()
    trials = Trials()
    best = fmin(lossfn_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
    print('best: ', best)
    print(space_eval(space, best))
    with open('zema_ml_simple' + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)

    with open('best_parameters_simple.dict', "wb") as f:
        pickle.dump(space_eval(space, best), f)

    save_model(space_eval(space, best))
