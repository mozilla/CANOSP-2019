import simulation
import random_data_gen
import pandas as pd
from simulation_util import client_update, server_update

from sklearn.model_selection import ParameterGrid, train_test_split
import numpy as np


def run_data_gen_blob(s_prms):
    return _run_gen_func(s_prms, random_data_gen.generate_blob_data)


def run_data_gen_rand(s_prms):
    return _run_gen_func(s_prms, random_data_gen.generate_rand_data)


# Blob and rand are called identically, so it makes sense to wrap this in a func
def _run_gen_func(s_prms, gen_func):
    g_prms = create_g_params_from_s_params(s_prms)
    rand_data = gen_func(g_prms)
    return random_data_gen.transform_data_for_simulator_format(rand_data, g_prms)


def read_data_from_file(s_prms):
    file_path = s_prms[simulation.P_KEY_DATA_FILE_PATH]
    df = pd.read_csv(file_path)

    g_prms = create_g_params_from_s_params(s_prms)
    return random_data_gen.transform_data_for_simulator_format(df, g_prms)


def create_g_params_from_s_params(s_prms):
    return random_data_gen.InputGenParams(
        s_prms[simulation.P_KEY_NUM_SAMPLES],
        s_prms[simulation.P_KEY_NUM_LABELS],
        s_prms[simulation.P_KEY_NUM_FEATURES],
        s_prms[simulation.P_KEY_NUM_USERS],
    )


def run_fed_learn_sim(s_prms, data):
    num_labels = s_prms[simulation.P_KEY_NUM_LABELS]
    num_features = s_prms[simulation.P_KEY_NUM_FEATURES]

    # Note: data is already transformed for sim format

    sim_labels, sim_features = data
    features = np.array(sim_features)
    labels = np.array(sim_labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.4, random_state=0
    )

    init_weights = np.zeros((num_labels, num_features), dtype=np.float64, order="C")
    init_intercept = np.zeros(num_labels, dtype=np.float64, order="C")

    # Find all the permutations of the parameters
    param_grid = {
        "client_fraction": [1, 0.1],
        "epoch": [1, 5],
        "batch_size": [40],  # TODO: need to implement an infinite batch size
        "init_weight": [[init_weights, init_intercept]],
        "num_rounds": [10],
    }

    # run training/testing over all parameter combinations to get the best combination
    for params in ParameterGrid(param_grid):
        print("Training...")
        print("Params: ", params)
        classifier = server_update(
            params["init_weight"],
            params["client_fraction"],
            params["num_rounds"],
            X_train,
            y_train,
            params["epoch"],
            params["batch_size"],
            False,
        )
        weights = [classifier.coef_, classifier.intercept_]

        # need to remove the client dimension from our data for testing
        # ex: [[[1, 1], [2, 2]], [[3, 3], [4, 4]]] needs to become [[1, 1], [2, 2], [3, 3], [4, 4]] for features
        # and [[1, 2], [3, 4]] needs to become [1, 2, 3, 4] for labels
        reshaped_X_test = np.reshape(
            X_test, (X_test.shape[0] * X_test.shape[1], X_test.shape[2])
        )
        reshaped_y_test = np.reshape(y_test, y_test.size)

        score = classifier.score(reshaped_X_test, reshaped_y_test)

        print("Weights: {}\nScore: {:f}\n\n".format(weights, score))


def run_fed_avg_with_dp(sim_params, data):
    print("TODO: Implement Federated Averaging with DP!")
