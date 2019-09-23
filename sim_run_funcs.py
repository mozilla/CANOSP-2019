import simulation
import random_data_gen
from simulation_util import client_update, server_update

from sklearn.model_selection import ParameterGrid, train_test_split
import numpy as np


def run_data_gen_blob(s_prms):
    g_prms = random_data_gen.InputGenParams(
        s_prms[simulation.P_KEY_NUM_SAMPLES],
        s_prms[simulation.P_KEY_NUM_LABELS],
        s_prms[simulation.P_KEY_NUM_FEATURES],
        s_prms[simulation.P_KEY_NUM_USERS],
    )
    rand_data = random_data_gen.generate_blob_data(g_prms)
    return random_data_gen.transform_data_for_simulator_format(rand_data)


def run_fed_learn_sim(s_prms, data):

    # Load the data
    num_client = s_prms[simulation.P_KEY_NUM_USERS]
    samples_per_client = s_prms[simulation.P_KEY_NUM_SAMPLES]
    num_features = s_prms[simulation.P_KEY_NUM_FEATURES]

    features = np.random.randint(
        10, size=(num_client, samples_per_client, num_features)
    )
    labels = np.random.randint(2, size=(num_client, samples_per_client))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.4, random_state=0
    )

    # Find all the permutations of the parameters
    param_grid = {
        "client_fraction": [1, 0.1],
        "epoch": [1, 5],
        "batch_size": [10, samples_per_client],
        "init_weight": [[0, 0, 0, 0]],
        "num_rounds": [10],
    }

    # run training/testing over all parameter combinations to get the best combination
    for params in ParameterGrid(param_grid):
        print("Training...")
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
        weights = np.append(classifier.coef_[0], classifier.intercept_)

        # need to remove the client dimension from our data for testing
        # ex: [[[1, 1], [2, 2]], [[3, 3], [4, 4]]] needs to become [[1, 1], [2, 2], [3, 3], [4, 4]] for features
        # and [[1, 2], [3, 4]] needs to become [1, 2, 3, 4] for labels
        reshaped_X_test = np.reshape(
            X_test, (X_test.shape[0] * X_test.shape[1], X_test.shape[2])
        )
        reshaped_y_test = np.reshape(y_test, y_test.size)

        score = classifier.score(reshaped_X_test, reshaped_y_test)

        print("Params: {}\nWeights: {}\nScore: {:f}\n\n".format(params, weights, score))


def run_fed_avg_with_dp(sim_params, data):
    print("TODO: Implement Federated Averaging with DP!")
