import random
import numpy as np
from sklearn.linear_model import SGDClassifier


class FedAvgWithDpParams:
    def __init__(self, num_users, num_features, weight_mod, sensitivity, noise_scale):
        self.num_users = num_users
        self.num_features = num_features
        self.weight_mod = weight_mod  # w_hat
        self.sensitivity = sensitivity
        self.noise_scale = noise_scale


def run_fed_avg_with_dp(prms, data):
    """
    Runs federated averaging with differential privacy
    prms: The parameters needed to run (FedAvgWithDpParams)
    data: Data that conforms to the format... (TODO)
    """

    user_weights, weight_sum = _init_user_weights_and_weight_sum(
        prms.num_users, prms.weight_mod
    )
    theta = None
    theta_0 = _init_theta_from_moment_accountant(prms.num_users)
    prev_theta = np.array(theta_0, copy=True)
    user_sel_prob = 1 / prms.num_users
    standard_dev = _calc_standard_dev(
        prms.noise_scale, prms.sensitivity, prms.usr_sel_prob, weight_sum
    )  # This feels weird being a constant...
    user_updates_buf = []

    for round_t in range(prms.num_rounds):
        # Pick a random set of users to sample
        random_user_idxs_sample = _get_random_selection_of_user_idxs(prms.num_users)

        # Query the selected users
        user_updates_buf.clear()
        for user_idx in random_user_idxs_sample:
            user_round_feats, user_round_labels = _get_data_for_user_for_round(
                data, user_idx, round_t
            )
            user_updates_buf.append(
                user_update_fed_avg(prms, user_round_feats, user_round_labels, theta_0)
            )

        # Merge (fc)
        merged_user_values = _merge_all_user_weights(
            prms.num_features, user_sel_prob, weight_sum, user_updates_buf, user_weights
        )  # Note that we start at t + 1

        # Note: Assuming for now that S is defined before we run
        rand_gauss_noise = _gen_gausian_rand_noise(
            standard_dev, len(merged_user_values)
        )
        theta = prev_theta + merged_user_values + rand_gauss_noise
        prev_theta = theta

        print("Theta for round {}: \n {}".format(round_t, theta))
        _moments_accountant_accum_priv_spending(prms.noise_scale)

    privacy_spent = _calc_privacy_spent()
    print("Total privacy spent from privacy budget: {}".format(privacy_spent))


# TODO: Give better function name...
def _merge_all_user_weights(
    num_feats, user_sel_prob, weight_sum, user_updates_buf, user_weights
):
    num_users_in_batch = len(user_updates_buf)
    merged_weights = np.zeros(num_feats)

    for i in range(num_users_in_batch):
        weighted_user_val = np.multiply(user_updates_buf[i], user_weights[i])
        merged_weights = np.add(merged_weights, weighted_user_val)

    merged_weights = np.divide(merged_weights, user_sel_prob * weight_sum)
    return merged_weights


def flat_clip(sensitivity, vec):
    return vec * min(1, sensitivity / np.linalg.norm(vec))


def user_update_fed_avg(prms, round_user_features, round_user_labels, theta_0):

    batches_features = []
    batches_labels = []

    epoch = 10
    batch_size = 10

    classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=1)

    round_num_entries = len(round_user_features)
    num_batches = (
        round_num_entries // batch_size
    )  # For now just truncate any batches that are smaller than batch_size

    coef = np.zeros((prms.num_labels, prms.num_features), dtype=np.float64, order="C")
    intercept = np.zeros(prms.num_labels, dtype=np.float64, order="C")
    theta = np.array(theta_0, copy=True)

    # Split the round data into seperate batches
    for i in range(0, round_num_entries, batch_size):
        batches_features.append(round_user_features[i : i + batch_size])
        batches_labels.append(round_user_labels[i : i + batch_size])

    for _ in range(epoch):
        for j in num_batches:

            classifier.fit(
                batches_features[j],
                batches_labels[j],
                coef_init=coef,
                intercept_init=intercept,
            )

            coef = classifier.coef_
            intercept = classifier.intercept_

            # trained_weight = [coef, intercept]

            theta = np.subtract(
                theta, coef
            )  # NOTE: THIS IS WRONG! We also need to involve intercept somehow...
            difference = np.subtract(theta, theta_0)
            theta = np.add(theta_0, flat_clip(prms.sensitivity, difference))

    return theta - theta_0


def _get_random_selection_of_user_idxs(num_users):
    return np.array(random.sample(range(num_users), num_users))


def _init_theta_from_moment_accountant(num_users):
    # Just a placeholder value for now
    return [0.5] * num_users


def _init_user_weights_and_weight_sum(num_users, weight_mod):
    weights = []
    init_weight = min(num_users / weight_mod, 1)

    for _ in range(num_users):
        weights.append(init_weight)

    weight_sum = num_users * init_weight

    return weights, weight_sum


def _calc_standard_dev(noise_scale, sensitivity, usr_sel_prob, weight_sum):
    return (noise_scale * sensitivity) / (usr_sel_prob * weight_sum)


def _get_data_for_user_for_round(data, user_id, round_t):
    """
    Just extract an entry (label and feature set) from the generated data for the given user.
    """
    labels, feats = data

    round_label = labels[user_id][round_t]
    round_feats = feats[user_id][round_t]

    return round_label, round_feats


def _gen_gausian_rand_noise(stndrd_dev, vec_len):
    return np.random.normal(loc=0.0, scale=stndrd_dev, size=vec_len)


def _moments_accountant_accum_priv_spending(noise_scale):
    return -1  # TODO


def _calc_privacy_spent():
    return -1  # TODO
