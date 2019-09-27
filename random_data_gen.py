# Simple set of functions to generate different types of random data
# To see how these functions are intended to be called, take a look through "Data Generation Prototype.ipynb".


import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import datasets

# Used to control the range of values for the random gen func (1.0 --> [-1.0, 1.0])
RAND_FEAT_RANGE = 1.0
USERS_MIN_UNIQUE_LABELS = 2


class InputGenParams:
    """
    This is just used to nicely pass around generation parameters between functions.
    samples --> Total number of data points to generate
    labels --> Number of labels (species) groups to be generated. Evenly distributed.
    features --> Number of features per entry
    users --> How many unique users are in the genereated data. Evenly distributed.
    """

    def __init__(self, num_samples, num_labels, num_features, num_users):
        self.num_samples = num_samples
        self.num_labels = num_labels
        self.num_features = num_features
        self.num_users = num_users


def generate_blob_data(g_prms):
    """
    Generates blob data, or data points that tend to cluster around a point for each label.
    Returns: A Pandas DataFrame containing the generated records (see notebook for specific format)
    """
    return _gen_data_and_add_user_data(_gen_blob_data, g_prms)


def generate_random_data(g_prms):
    """
    Generates random data where features should have no correlation.
    Returns: A Pandas DataFrame containing the generated records (see notebook for specific format)
    """
    return _gen_data_and_add_user_data(_gen_random_data, g_prms)


def transform_data_for_simulator_format(df, g_prms):
    """
    Transforms a Pandas DataFrame returned by the generation functions into a format that is usable by the simulator.
    See notebook for the specific format.
    """
    labels = []
    feats = []
    for i in range(g_prms.num_users):
        client_df = df[df.user_id == i].drop(df.columns[0], axis=1)
        labels.append(list(client_df.labels))
        client_feats = client_df.drop(columns=["user_id", "labels"]).to_records(
            index=False
        )
        
        # convert tuples to lists
        client_feats = [list(feat) for feat in client_feats]
        
        feats.append(client_feats)

    return (labels, feats)


def _gen_blob_data(g_prms):
    feat_arr, label_idxs = skl.datasets.make_blobs(
        n_samples=g_prms.num_samples,
        n_features=g_prms.num_features,
        centers=g_prms.num_labels,
    )

    df = pd.DataFrame(feat_arr)
    _evenly_add_labels_to_data(df, g_prms)
    return df


def _gen_random_data(g_prms):
    df = pd.DataFrame()

    for i in range(0, g_prms.num_features):
        feat_arr = np.random.ranf(g_prms.num_samples) * RAND_FEAT_RANGE
        df[i] = feat_arr

    _evenly_add_labels_to_data(df, g_prms)
    return df


def _evenly_add_labels_to_data(df, g_prms):
    _add_evenly_distributed_values_to_data(df, g_prms, g_prms.num_labels, "labels")


def _add_users_ids_to_data_and_shuffle(df, g_prms):
    df = skl.utils.shuffle(df)
    _add_evenly_distributed_values_to_data(df, g_prms, g_prms.num_users, "user_id")
    return df


def _all_users_have_at_least_n_unique_lables(data):
    nunique_labels = data.groupby("user_id", as_index=False).agg({"labels": "nunique"})
    ok = (nunique_labels["labels"] < USERS_MIN_UNIQUE_LABELS).sum() == 0

    if not ok:
        print(
            "Not all users have {} unique labels! Regenerating!".format(
                USERS_MIN_UNIQUE_LABELS
            )
        )
    return ok


def _gen_data_until_prereq_met(data_gen_func, prereq_func):
    prereq_met = False
    df = None

    while not prereq_met:
        df = data_gen_func()
        prereq_met = prereq_func(df)

    return df


def _gen_data_and_add_user_data(data_gen_func, g_prms):
    def gen_and_add_users_func():
        data = data_gen_func(g_prms)
        data = _add_users_ids_to_data_and_shuffle(data, g_prms)
        return data

    return _gen_data_until_prereq_met(
        gen_and_add_users_func, _all_users_have_at_least_n_unique_lables
    )


def _gen_data_and_add_user_data(data_gen_func, g_prms):
    def gen_and_add_users_func():
        data = data_gen_func(g_prms)
        data = _add_users_ids_to_data_and_shuffle(data, g_prms)
        return data

    return _gen_data_until_prereq_met(
        gen_and_add_users_func, _all_users_have_at_least_n_unique_lables
    )


def _add_users_ids_to_data_and_shuffle(df, g_prms):
    df = skl.utils.shuffle(df)
    _add_evenly_distributed_values_to_data(df, g_prms, g_prms.num_users, "user_id")
    return df


def _gen_data_until_prereq_met(data_gen_func, prereq_func):
    prereq_met = False
    df = None

    while not prereq_met:
        df = data_gen_func()
        prereq_met = prereq_func(df)

    return df


def _all_users_have_at_least_n_unique_lables(data):
    nunique_labels = data.groupby("user_id", as_index=False).agg({"labels": "nunique"})
    ok = (nunique_labels["labels"] < USERS_MIN_UNIQUE_LABELS).sum() == 0

    if not ok:
        print(
            "Not all users have {} unique labels! Regenerating!".format(
                USERS_MIN_UNIQUE_LABELS
            )
        )
    return ok


def _add_evenly_distributed_values_to_data(df, g_prms, num_items, field_name):
    new_field = [
        i % num_items for i in range(1, g_prms.num_samples + 1)
    ]  # Avoid div by 0
    df[field_name] = new_field
