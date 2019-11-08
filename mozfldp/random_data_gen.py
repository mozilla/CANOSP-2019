# Simple set of functions to generate different types of random data
# To see how these functions are intended to be called, take a look through "Data Generation Prototype.ipynb".


import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


class InputGenParams:
    """
    This is just used to nicely pass around generation parameters between functions.
    samples --> Total number of data points to generate
    labels --> Number of label groups to be generated. Evenly distributed.
    features --> Number of features per entry
    users --> How many unique users are in the genereated data. Evenly
    distributed but randomized.
    rand_range --> The range of values covered by the features (extends to
    [-rand_range, rand_range])
    min_unique_classes --> Minimum number of unique class labels each user
    must have.
    """

    def __init__(
        self,
        num_samples,
        num_labels,
        num_features,
        num_users,
        rand_range=1.0,
        min_unique_classes=2,
    ):
        self.num_samples = num_samples
        self.num_labels = num_labels
        self.num_features = num_features
        self.num_users = num_users
        self.rand_range = rand_range
        self.min_unique_classes = min_unique_classes


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
        client_df = df[df.user_id == i]  # .drop(df.columns[0], axis=1)
        labels.append(list(client_df.label))
        client_feats = client_df.drop(columns=["user_id", "label"]).to_records(
            index=False
        )

        # convert tuples to lists
        client_feats = [list(feat) for feat in client_feats]
        feats.append(client_feats)

    return (labels, feats)


def _gen_blob_data(g_prms):
    feat_arr, label_idxs = make_blobs(
        n_samples=g_prms.num_samples,
        n_features=g_prms.num_features,
        centers=g_prms.num_labels,
    )

    df = pd.DataFrame(feat_arr)
    df["label"] = label_idxs
    return df


def _gen_random_data(g_prms):
    # Create a uniform random array of shape (num_samples, num_features).
    feat_arr = np.random.ranf((g_prms.num_samples, g_prms.num_features))
    # Adjust the range to (-rand_range, rand_range).
    feat_arr = feat_arr * 2 * g_prms.rand_range - g_prms.rand_range

    df = pd.DataFrame(feat_arr)
    # Add dummy class labels.
    df["label"] = _generate_evenly_distributed_ids(
        g_prms.num_samples, g_prms.num_labels
    )
    return df


def _add_users_ids_to_data_and_shuffle(df, g_prms):
    user_ids = _generate_evenly_distributed_ids(g_prms.num_samples, g_prms.num_users)
    # Randomize which rows get assigned which user IDs.
    np.random.shuffle(user_ids)

    df["user_id"] = user_ids
    return df


def _all_users_have_at_least_n_unique_lables(data, g_prms):
    nunique_labels = data.groupby("user_id", as_index=False).agg({"label": "nunique"})
    ok = (nunique_labels["label"] < g_prms.min_unique_classes).sum() == 0

    if not ok:
        print(
            "Not all users have {} unique labels! Regenerating!".format(
                g_prms.min_unique_classes
            )
        )
    return ok


def _gen_data_until_prereq_met(data_gen_func, prereq_func, g_prms):
    prereq_met = False
    df = None

    while not prereq_met:
        df = data_gen_func()
        prereq_met = prereq_func(df, g_prms)

    return df


def _gen_data_and_add_user_data(data_gen_func, g_prms):
    def gen_and_add_users_func():
        data = data_gen_func(g_prms)
        data = _add_users_ids_to_data_and_shuffle(data, g_prms)
        return data

    return _gen_data_until_prereq_met(
        gen_and_add_users_func, _all_users_have_at_least_n_unique_lables, g_prms
    )


def _generate_evenly_distributed_ids(num_samples, num_items):
    # Repeat the range 0, 1, ..., num_items-1 enough times to cover the size of
    # the dataset.
    ids = np.tile(np.arange(num_items), num_samples // num_items + 1)
    # Restrict to the desired size.
    return ids[:num_samples]
