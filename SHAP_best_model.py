import os
import pickle
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from zema_generate_model import generate_model
tf.config.threading.set_inter_op_parallelism_threads(10)
from load_achsemat import load_achsemat as load_axis_data

trainData, trainTarget = load_axis_data()
# scale the individual time-series data
try:
    arg1 = sys.argv[1]
    print('scaling training data')
    for k in range(11):
        trainData[:, k, :] = StandardScaler().fit_transform(trainData[:, k, :])
    print('loading model for scaled data')
    with open('best_parameters_simple.dict', "rb") as f:
        params = pickle.load(f)
except IndexError:
    print('proceeding without scaling')
    print('loading model for unscaled data')
    with open('best_parameters_simple_noscaling.dict', "rb") as f:
        params = pickle.load(f)


model = generate_model(params)

model.compile(
    optimizer=params['optimizer'], loss="mean_squared_error", metrics=["mse"]
)

model.fit(trainData, trainTarget, epochs=100, validation_split=0, verbose=0)


n_samples = 150
sampled_data = shap.sample(trainData, n_samples)
explainer = shap.KernelExplainer(model=model.predict, data=sampled_data, link="identity")
X_idx = 0
shap_values = explainer.shap_values(X=sampled_data, nsamples=n_samples)

with open('shapley_nn_bestmodel', "wb") as f:
    pickle.dump(shap_values, f)
