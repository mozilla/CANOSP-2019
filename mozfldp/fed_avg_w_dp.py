import random
import numpy as np
from sklearn.linear_model import SGDClassifier


class FedAvgWithDpParams:
    """
    Used just to hold parameters needed to run FedAvgWithDP
    """

    def __init__(
        self,
        num_users,
        num_features,
        num_labels,
        num_rounds,
        batch_size,
        num_epochs,
        user_weight_cap,
        user_sel_prob,
        sensitivity,
        noise_scale,
        rand_seed,
    ):
        self.num_users = num_users
        self.num_features = num_features
        self.num_labels = num_labels
        self.num_rounds = num_rounds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.user_weight_cap = user_weight_cap  # w_hat
        self.user_sel_prob = user_sel_prob
        self.sensitivity = sensitivity
        self.noise_scale = noise_scale
        self.rand_seed = rand_seed


def run_fed_avg_with_dp(prms, data):
    """
    Runs federated averaging with differential privacy

    prms: The parameters needed to run (FedAvgWithDpParams)
    data: Data to train on. In the format of: ([user_1_labels, user_2_labels, ...], [user_1_feats, user_2_feats, ...])
    """

    # Fix the seed for all rng calls
    np.random.seed(prms.rand_seed)
    random.seed(prms.rand_seed)

    user_weights, weight_sum = _init_user_weights_and_weight_sum(
        data, prms.num_users, prms.user_weight_cap
    )
    theta = _init_theta(prms.num_features, prms.num_labels)
    standard_dev = _calc_standard_dev(
        prms.noise_scale, prms.sensitivity, prms.user_sel_prob, weight_sum
    )
    user_updates_buf = []
    num_theta_elems = prms.num_features * prms.num_labels + prms.num_labels

    for round_t in range(prms.num_rounds):
        # Pick a random set of users to sample
        random_user_idxs_sample = _get_random_selection_of_user_idxs(
            prms.num_users, prms.user_sel_prob
        )

        # Query the selected users
        user_updates_buf.clear()
        for user_idx in random_user_idxs_sample:
            user_round_labels, user_round_feats = _get_data_for_user(data, user_idx)
            user_updates_buf.append(
                user_update_fed_avg(prms, user_round_feats, user_round_labels, theta)
            )

        # Merge (fc)
        merged_user_values = _merge_all_user_thetas(
            prms.user_sel_prob,
            weight_sum,
            user_updates_buf,
            user_weights,
            num_theta_elems,
        )  # Note that we start at t + 1

        # Note: Assuming for now that S is defined before we run
        rand_gauss_noise = _gen_gausian_rand_noise(
            standard_dev, len(merged_user_values)
        )

        theta += merged_user_values + rand_gauss_noise

        print("Theta ([coef, inter] for round {}: \n {}".format(round_t, theta))
        _moments_accountant_accum_priv_spending(prms.noise_scale)

    privacy_spent = _calc_privacy_spent()
    print("Total privacy spent from privacy budget: {}".format(privacy_spent))

    coef_slice, inter_slice = _get_coef_and_inter_slice_from_theta(
        theta, prms.num_features, prms.num_labels
    )

    return np.array(coef_slice), np.array(inter_slice)


# TODO: Give better function name...
def _merge_all_user_thetas(
    user_sel_prob, weight_sum, user_updates_buf, user_weights, num_theta_elems
):
    """
    Merge all user updates for a round into a single delta (vector).

    user_sel_prob: Probability of any given user being selected for a round
    weight_sum: The sum of all user weights
    user_updates_buf: The user updates (thetas) that we are merging.
    user_weights: The weights applied to each user. Users with more data have more weight.
    num_theta_elems: The number of elements in theta.
    """

    num_users_in_batch = len(user_updates_buf)
    merged_weights = np.zeros(num_theta_elems, dtype=np.float64, order="C")

    for i in range(num_users_in_batch):
        weighted_user_val = np.multiply(user_updates_buf[i], user_weights[i])
        merged_weights = np.add(merged_weights, weighted_user_val)

    merged_weights = np.divide(merged_weights, user_sel_prob * weight_sum)
    return merged_weights


def flat_clip(sensitivity, vec):
    """
    "Clips" a vector, in order to limit how "long" a vector can be.

    sensitivity: Affects how long a vector can be. Higher values --> longer vector.
    """
    norm = np.linalg.norm(vec)
    if norm > sensitivity:
        vec *= sensitivity / norm

    return vec


def user_update_fed_avg(prms, round_user_features, round_user_labels, theta_0):
    """
    Calculates an update to theta (coefs, intercepts) based on data sampled from a single user.
    Does not update theta directly.
    These updates from individual users are then merged into a single update before performing updating theta.

    round_user_features/labels: The data from the user that are using to calculate the update.
    theta_0: The current weights of theta from the previous round.
    Note that other user updates before this update do not affect theta_0!
    """

    batches_features = []
    batches_labels = []

    classifier = SGDClassifier(
        loss="hinge", penalty="l2", max_iter=1, random_state=prms.rand_seed
    )

    round_num_entries = len(round_user_features)
    num_batches = int(np.ceil(round_num_entries / prms.batch_size))
    theta = np.array(theta_0, copy=True)

    # Split the round data into seperate batches
    for i in range(0, round_num_entries, prms.batch_size):
        batches_features.append(round_user_features[i : i + prms.batch_size])
        batches_labels.append(round_user_labels[i : i + prms.batch_size])

    # Train with the batches
    for _ in range(prms.num_epochs):
        for j in range(num_batches):

            coef, inter = _get_coef_and_inter_slice_from_theta(
                theta, prms.num_features, prms.num_labels
            )

            classifier.fit(
                batches_features[j],
                batches_labels[j],
                coef_init=coef,
                intercept_init=inter,
            )

            _set_coef_and_inter_on_theta(theta, classifier.coef_, classifier.intercept_)
            theta = theta_0 + flat_clip(prms.sensitivity, theta - theta_0)

    return theta - theta_0


def _get_random_selection_of_user_idxs(num_users, user_sel_prob):
    """
    Selects a random set of user idxs based on the chances of picking any given user.

    num_users: The total number of users in the simulation.
    user_sel_prob: The chance of picking any given user
    """

    # This is definitely not the optimal way to do this, but it gets the job done
    # Very dumb brute force impl...

    sel_user_idxs = []

    for i in range(num_users):
        rand_val = random.random()
        if rand_val < user_sel_prob:
            sel_user_idxs.append(i)

    return sel_user_idxs


def _init_theta(num_features, num_labels):
    """
    Initializes theta.
    Currently this just means initializing all weights to 0.

    Returns: Inited theta
    """
    # Just a placeholder value for now

    # theta is internally a linear array. We will use a functon to get slices of the features/labels.
    return np.zeros(num_labels * num_features + num_labels, dtype=np.float64, order="C")


def _init_user_weights_and_weight_sum(data, num_users, user_weight_cap):
    """
    Inits all user weights and also returns the sum of all user weight.

    returns: A tuple containing the calculated weights for all users and the summed weight
    """

    user_weights = []
    weight_sum = 0

    for usr_idx in range(num_users):
        user_weight = _init_user_weight(data, usr_idx, user_weight_cap)
        user_weights.append(user_weight)
        weight_sum += user_weight

    return np.array(user_weights), weight_sum


def _init_user_weight(data, usr_idx, user_weight_cap):
    """
    Inits a user's weight based on the amount of entries they have and the weight cap param.

    data: The entire dataset passed into the sim
    usr_idx: The user idx/id
    user_weight_cap: Used to limit the maximum amount of weight that a user can have

    returns: The calculated user weight
    """

    user_labels, _ = _get_data_for_user(data, usr_idx)
    num_user_entries = len(user_labels)
    return min(num_user_entries / user_weight_cap, 1)


def _calc_standard_dev(noise_scale, sensitivity, usr_sel_prob, weight_sum):
    """
    Calculates the standard deviation
    Parameters are tailored to this sim impl
    """
    return (noise_scale * sensitivity) / (usr_sel_prob * weight_sum)


def _get_data_for_user(data, user_id):
    """
    Pick all the labels/feature that belong to a user for a round.

    data: The entire dataset passed into the sim
    user_id: The user idx to extract data for
    """

    labels, feats = data
    return labels[user_id], feats[user_id]


def _gen_gausian_rand_noise(stndrd_dev, vec_len):
    """
    Generates gausian noise and applies to all elements in a vector.

    stndrd_dev: The standard deviation of the distrubution to sample from
    vec_len: The number of elements in the vector

    returns: The vector after noise has been applied
    """

    return np.random.normal(loc=0.0, scale=stndrd_dev, size=vec_len)


def _moments_accountant_accum_priv_spending(noise_scale):
    return -1  # TODO


def _calc_privacy_spent():
    return -1  # TODO


def _get_coef_and_inter_slice_from_theta(theta, num_features, num_labels):
    """
    Helper function to extract slices of the coefs/intercepts of theta (which is stored as a 1D array)

    returns: A tuple containing slices of the coefs and intercepts (coefs, intercepts)
    """

    coef_len = num_features * num_labels
    coef_arr_slices = []

    for i in range(0, coef_len, num_features):
        coef_arr_slices.append(theta[i : i + num_features])

    return coef_arr_slices, theta[coef_len:]


def _set_coef_and_inter_on_theta(theta, coef_arrs, inter):
    """
    Helper function to update the coefs/intercepts of a theta.
    coef_arrs & inter are the coeficients/intercepts that the SGDClassifier outputs.

    coef_arr_slices: An array of arrays containing the coefs
    inter: An array containing the intercepts
    """

    num_features = len(coef_arrs[0])
    coef_len = num_features * len(inter)

    for i, slice in enumerate(coef_arrs):
        coef_start_idx = i * num_features
        theta[coef_start_idx : coef_start_idx + num_features] = coef_arrs[i]

    theta[coef_len:] = inter
