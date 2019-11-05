import random
import numpy as np
from sklearn.linear_model import SGDClassifier


class FedAvgWithDpParams:
    def __init__(
        self,
        num_users,
        num_features,
        num_labels,
        num_rounds,
        batch_size,
        num_epochs,
        weight_mod,
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
        self.weight_mod = weight_mod  # w_hat
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
        prms.num_users, prms.weight_mod
    )
    theta = None
    theta_0 = _init_theta_from_moment_accountant(prms.num_features, prms.num_labels)
    prev_theta = np.array(theta_0, copy=True)
    user_sel_prob = _calc_user_sel_proc(prms.num_users, 30)
    standard_dev = _calc_standard_dev(
        prms.noise_scale, prms.sensitivity, user_sel_prob, weight_sum
    )  # This feels weird being a constant...
    user_updates_buf = []
    num_theta_elems = prms.num_features * prms.num_labels + prms.num_labels

    for round_t in range(prms.num_rounds):
        # Pick a random set of users to sample
        random_user_idxs_sample = _get_random_selection_of_user_idxs(
            prms.num_users, user_sel_prob
        )

        # Query the selected users
        user_updates_buf.clear()
        for user_idx in random_user_idxs_sample:
            user_round_labels, user_round_feats = _get_data_for_user_for_round(
                prms, data, user_idx
            )
            user_updates_buf.append(
                user_update_fed_avg(prms, user_round_feats, user_round_labels, theta_0)
            )

        # Merge (fc)
        merged_user_values = _merge_all_user_thetas(
            user_sel_prob, weight_sum, user_updates_buf, user_weights, num_theta_elems
        )  # Note that we start at t + 1

        # Note: Assuming for now that S is defined before we run
        rand_gauss_noise = _gen_gausian_rand_noise(
            standard_dev, len(merged_user_values)
        )

        theta = prev_theta + merged_user_values + rand_gauss_noise
        prev_theta = theta

        print("Theta ([coef, inter] for round {}: \n {}".format(round_t, theta))
        _moments_accountant_accum_priv_spending(prms.noise_scale)

    privacy_spent = _calc_privacy_spent()
    print("Total privacy spent from privacy budget: {}".format(privacy_spent))

    coef_slice, inter_slice = _get_coef_and_inter_slice_from_theta(
        theta, prms.num_features, prms.num_labels
    )

    return np.array(coef_slice), np.array(inter_slice)


# TODO: Give better function name...
def _merge_all_user_thetas(user_sel_prob, weight_sum, user_updates_buf, user_weights, num_theta_elems):
    """
    Merge all user updates for a round into a single delta (vector).
    """

    num_users_in_batch = len(user_updates_buf)
    merged_weights = np.zeros(num_theta_elems, dtype=np.float64, order="C")

    for i in range(num_users_in_batch):
        weighted_user_val = np.multiply(user_updates_buf[i], user_weights[i])
        merged_weights = np.add(merged_weights, weighted_user_val)

    merged_weights = np.divide(merged_weights, user_sel_prob * weight_sum)
    return merged_weights


def flat_clip(sensitivity, vec):
    vec *= min(1, sensitivity / np.linalg.norm(vec))
    return vec


def user_update_fed_avg(prms, round_user_features, round_user_labels, theta_0):

    batches_features = []
    batches_labels = []

    classifier = SGDClassifier(
        loss="hinge", penalty="l2", max_iter=1, random_state=prms.rand_seed
    )

    round_num_entries = len(round_user_features)

    # For now just skip any batches that are smaller than batch_size
    num_batches = round_num_entries // prms.batch_size

    theta = np.array(theta_0, copy=True)

    # Split the round data into seperate batches
    for i in range(0, round_num_entries, prms.batch_size):
        batches_features.append(round_user_features[i : i + prms.batch_size])
        batches_labels.append(round_user_labels[i : i + prms.batch_size])

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
    # This is definitely not the optimal way to do this, but it gets the job done
    # Very dumb brute force impl...

    sel_user_idxs = []

    for i in range(num_users):
        rand_val = random.random()
        if rand_val < user_sel_prob:
            sel_user_idxs.append(i)

    return sel_user_idxs


def _init_theta_from_moment_accountant(num_features, num_labels):
    # Just a placeholder value for now

    # theta_0 is internally a linear array. We will use functon to get slices of the features/labels.
    return np.zeros(num_labels * num_features + num_labels, dtype=np.float64, order="C")


def _init_user_weights_and_weight_sum(num_users, weight_mod):
    weights = (
        []
    )  # NOTE: Since user weights appear to be constant for all users, we may not need an array of weights.
    init_weight = min(num_users / weight_mod, 1)

    for _ in range(num_users):
        weights.append(init_weight)

    weight_sum = num_users * init_weight

    return weights, weight_sum


def _calc_standard_dev(noise_scale, sensitivity, usr_sel_prob, weight_sum):
    return (noise_scale * sensitivity) / (usr_sel_prob * weight_sum)


def _get_data_for_user_for_round(prms, data, user_id):
    """
    Pick a random amount of entries per user for a given round.
    """
    labels, feats = data

    # TODO: np is a temp hack! Probably should do this to data at the start of the sim instead.
    user_labels = np.array(labels[user_id])
    user_feats = np.array(feats[user_id])

    num_entries_for_user = len(user_labels)
    num_entries_to_choose = random.randint(prms.batch_size, num_entries_for_user)

    return _choose_n_labels_and_feautures_from_user_labels_and_data(
        user_labels, user_feats, num_entries_to_choose
    )


def _gen_gausian_rand_noise(stndrd_dev, vec_len):
    return np.random.normal(loc=0.0, scale=stndrd_dev, size=vec_len)


def _choose_n_labels_and_feautures_from_user_labels_and_data(
    user_labels, user_feats, num_entries_to_choose
):
    labels_for_round = []
    feats_for_round = []

    chosen_idxs = np.random.choice(len(user_labels), size=num_entries_to_choose)

    for i in chosen_idxs:
        labels_for_round.append(user_labels[i])
        feats_for_round.append(user_feats[i])

    return labels_for_round, feats_for_round


def _calc_user_sel_proc(num_users, expected_num_users_picked_per_round):
    return (1 / num_users) * expected_num_users_picked_per_round


def _moments_accountant_accum_priv_spending(noise_scale):
    return -1  # TODO


def _calc_privacy_spent():
    return -1  # TODO


def _get_coef_and_inter_slice_from_theta(theta, num_features, num_labels):
    coef_len = num_features * num_labels
    coef_arr_slices = []

    for i in range(0, coef_len, num_features):
        coef_arr_slices.append(theta[i : i + num_features])

    return coef_arr_slices, theta[coef_len:]


def _set_coef_and_inter_on_theta(theta, coef_arr_slices, inter):
    num_features = len(coef_arr_slices[0])
    coef_len = num_features * len(inter)

    for i, slice in enumerate(coef_arr_slices):
        coef_start_idx = i * num_features
        theta[coef_start_idx : coef_start_idx + num_features] = coef_arr_slices[i]

    theta[coef_len:] = inter
