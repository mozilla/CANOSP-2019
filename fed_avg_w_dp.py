import random
import numpy as np
from sklearn.linear_model import SGDClassifier


class FedAvgWithDpParams:
    def __init__(self, weight_mod, num_users, num_features, sensitivity, noise_scale):
        self.weight_mod = weight_mod  # w_hat
        self.num_users = num_users
        self.num_features = num_features
        self.sensitivity = sensitivity
        self.noise_scale = noise_scale


def fed_avg_w_dp(prms, data):

    user_weights, weight_sum = _init_user_weights_and_weight_sum(prms.num_users)
    theta = None
    prev_theta = _init_theta_from_moment_accountant()
    user_sel_prob = 1 / prms.num_users
    standard_dev = _calc_standard_dev(
        prms.noise_scale, prms.sensitivity, prms.usr_sel_prob, prms.weight_sum
    )  # This feels weird being a constant...
    user_updates_buf = []

    for round_t in range(prms.num_rounds):
        # Pick a random set of users to sample
        random_user_idxs_sample = _get_random_selection_of_user_idxs(prms.num_users)

        # Query the selected users
        user_updates_buf.clear()
        for user_idx in random_user_idxs_sample:
            user_data = _get_data_for_user_for_round(user_idx, round_t)
            user_updates_buf.append(user_update_fed_avg(user_data, prev_theta))

        # Merge (fc)
        merged_user_values = _merge_all_user_weights(
            prms, user_sel_prob, user_updates_buf, user_weights
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
    params, user_sel_prob, weight_sum, user_updates_buf, user_weights
):
    num_users_in_batch = len(user_updates_buf)
    merged_weights = np.zeros(params.num_features)

    for i in range(num_users_in_batch):
        weighted_user_val = np.multiply(user_updates_buf[i], user_weights[i])
        merged_weights = np.add(merged_weights, weighted_user_val)

    merged_weights = np.divide(merged_weights, user_sel_prob * weight_sum)
    return merged_weights


def flat_clip(difference):


    pass


def user_update_fed_avg(features, labels , theta):

    batches_features = []
    batches_labels = []

    coef = list(theta[0])
    intercept = list(theta[1])

    epoch = 10
    batch_size = 10
    stepsize = 0.1

    classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=1)

    for i in range(0, len(k), batch_size):
        batches_features.append(k[i : i + batch_size])
        batches_labels.append(labels[i : i + batch_size])


    for i in range(epoch):
        for j in range(batch_size):

            classifier.fit(
                batches_features[i],
                batches_labels[i],
                coef_init=coef,
                intercept_init=intercept,
            )

            coef = classifier.coef_
            intercept = classifier.intercept_

    weights = [coef, intercept]

    return weights
            

    


def _get_random_selection_of_user_idxs(num_users):
    return np.array(random.sample(range(num_users), num_users))


def _init_theta_from_moment_accountant():
    pass  # TODO


def _init_user_weights_and_weight_sum(num_users, weight_mod):
    weights = []
    init_weight = min(num_users / weight_mod, 1)

    for i in range(num_users):
        weights.append(init_weight)

    weight_sum = num_users * init_weight

    return weights, weight_sum


def _calc_standard_dev(noise_scale, sensitivity, usr_sel_prob, weight_sum):
    return (noise_scale * sensitivity) / (usr_sel_prob * weight_sum)


def _get_data_for_user_for_round(user_id, rount_t):
    pass  # TODO


def _gen_gausian_rand_noise(stndrd_dev, vec_len):
    return np.random.normal(loc=0.0, scale=stndrd_dev, size=vec_len)


def _moments_accountant_accum_priv_spending(noise_scale):
    pass  # TODO


def _calc_privacy_spent():
    pass  # TODO
