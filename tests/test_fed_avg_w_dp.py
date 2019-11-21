# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
import numpy as np
import pandas as pd

from mozfldp import fed_avg_w_dp, runner, random_data_gen
from mozfldp.runner import Runner

import warnings

# Module globals because class variables can't be used as default variables
DEFAULT_USER_SEL_PROB = 0.1
DEFAULT_WEIGHT_SUM = 5


class TestMergeAllUserThetas:

    user_thetas = [[1, 2, 3, 4], [2, 5, 7, 9]]
    user_theta_len = 4

    def test_two_weights(self):

        # By hand:
        # Numerator:    [7.5, 17.5, 25, 32.5]
        # Denominator:  0.1 * 5.0

        res = self._run_merge_func(TestMergeAllUserThetas.user_thetas[:2])
        assert _lists_equal(res, [15, 35, 50, 65])

    def test_one_weight_returns_theta_over_user_sel_prob(self):

        # By Hand: (weight & weight_sum cancel)
        # Numerator: [1, 2, 3, 4]
        # Denominator: [0.1]

        res = self._run_merge_func(TestMergeAllUserThetas.user_thetas[:1])
        assert _lists_equal(res, [10, 20, 30, 40])

    def _run_merge_func(
        self,
        user_thetas,
        user_sel_prob=DEFAULT_USER_SEL_PROB,
        weight_sum=DEFAULT_WEIGHT_SUM,
    ):
        num_users = len(user_thetas)
        user_weights = [DEFAULT_WEIGHT_SUM / num_users] * num_users

        return fed_avg_w_dp._merge_all_user_thetas(
            user_sel_prob,
            weight_sum,
            user_thetas,
            user_weights,
            TestMergeAllUserThetas.user_theta_len,
        )


class TestSelectingRandomUserIdxs:

    DEFAULT_NUM_USERS = 10000

    def test_selects_user_idxs_follows_user_select_probability(self):
        num_runs_for_average = 100

        user_sel_prob = 0.2
        tot_users_sel = 0
        max_perc_variation_from_sel_prob = 0.01

        for _ in range(num_runs_for_average):
            res = fed_avg_w_dp._get_random_selection_of_user_idxs(
                TestSelectingRandomUserIdxs.DEFAULT_NUM_USERS, user_sel_prob
            )
            tot_users_sel += len(res)

        avg_users_sel_per_iter = tot_users_sel / num_runs_for_average
        expected_avg_sel_users_per_iter = (
            user_sel_prob * TestSelectingRandomUserIdxs.DEFAULT_NUM_USERS
        )
        assert _vals_are_within_percent_diff_range(
            avg_users_sel_per_iter,
            expected_avg_sel_users_per_iter,
            max_perc_variation_from_sel_prob,
        )

    def test_can_handle_a_0_select_probability(self):
        res = fed_avg_w_dp._get_random_selection_of_user_idxs(
            TestSelectingRandomUserIdxs.DEFAULT_NUM_USERS, 0
        )
        assert len(res) == 0

    def test_a_user_select_probability_of_1_returns_all_user_idxs(self):
        res = fed_avg_w_dp._get_random_selection_of_user_idxs(
            TestSelectingRandomUserIdxs.DEFAULT_NUM_USERS, 1
        )
        assert len(res) == TestSelectingRandomUserIdxs.DEFAULT_NUM_USERS


class TestFlatClipping:

    FLAT_CLIP_TEST_VEC = np.array([3.0, 4.0])

    def test_flatclipping_will_never_increase_the_magnitude(self):
        extreme_sensitivity = 50000.0

        old_norm = np.linalg.norm(TestFlatClipping.FLAT_CLIP_TEST_VEC)
        clipped_vec = fed_avg_w_dp.flat_clip(
            extreme_sensitivity, TestFlatClipping.FLAT_CLIP_TEST_VEC
        )
        new_norm = np.linalg.norm(clipped_vec)

        assert new_norm == old_norm

    def test_flatclipping_val_test(self):
        sensitivity = 1.0
        clipped_vec = fed_avg_w_dp.flat_clip(
            sensitivity, TestFlatClipping.FLAT_CLIP_TEST_VEC
        )

        assert np.allclose(clipped_vec, np.array([0.6, 0.8]))


class TestFedAvgWDPHighLevel:

    BLOB_DATA_PATH = "datasets/blob_S20000_L3_F4_U100.csv"
    RAND_DATA_PATH = "datasets/rand_S20000_L3_F4_U100.csv"

    SEEDS_TO_TEST_WITH = [42, 117, 1337, 981]

    @pytest.mark.slow
    def test_converges_resonably_with_blob_dataset(self):
        # Ensure that the score is at least 70%
        TestFedAvgWDPHighLevel._run_sim_on_dataset(
            TestFedAvgWDPHighLevel.BLOB_DATA_PATH, 1.0, 0.3
        )

    @pytest.mark.slow
    def test_does_not_converge_nicely_with_rand_dataset(self):
        # Ensure that the score hovers around 33%
        TestFedAvgWDPHighLevel._run_sim_on_dataset(
            TestFedAvgWDPHighLevel.RAND_DATA_PATH, 0.33, 0.3
        )

    def _run_sim_on_dataset(data_set_path, target_perc, max_diff_perc):
        # hide the warning message temporarily
        warnings.simplefilter("ignore")

        data = TestFedAvgWDPHighLevel._load_data_from_file(data_set_path)

        for seed in TestFedAvgWDPHighLevel.SEEDS_TO_TEST_WITH:
            print('Running "{}" with seed {}...'.format(data_set_path, seed))

            sim_prms = TestFedAvgWDPHighLevel._create_sim_prms_with_rand_seed(seed)
            score = runner.fed_avg_with_dp(sim_prms, data)

            assert _vals_are_within_percent_diff_range(
                score, target_perc, max_diff_perc
            )

    def _load_data_from_file(path):
        g_prms = random_data_gen.InputGenParams(20000, 3, 4, 100, 1.0, 2)
        df = pd.read_csv(path)
        return random_data_gen.transform_data_for_simulator_format(df, g_prms)

    def _create_sim_prms_with_rand_seed(seed):
        return {
            Runner.P_KEY_NUM_SAMPLES: 20000,
            Runner.P_KEY_NUM_LABELS: 3,
            Runner.P_KEY_NUM_FEATURES: 4,
            Runner.P_KEY_NUM_USERS: 100,
            Runner.P_KEY_BATCH_SIZE: 40,
            Runner.P_KEY_NUM_EPOCHS: 5,
            Runner.P_KEY_NUM_ROUNDS: 10,
            Runner.P_KEY_WEIGHT_MOD: 1,
            Runner.P_KEY_USER_SEL_PROB: 0.1,
            Runner.P_KEY_SENSITIVITY: 0.5,
            Runner.P_KEY_NOISE_SCALE: 1.0,
            Runner.P_KEY_RAND_SEED: seed,
        }


# Small tests


def test_setting_theta_values_are_set_properly():
    num_feats = 3
    num_labels = 2

    coefs = [[1, 2, 3], [4, 5, 6]]

    intercepts = [7, 8]

    theta_len = (num_feats * num_labels) + num_labels
    theta = np.zeros(theta_len)

    fed_avg_w_dp._set_coef_and_inter_on_theta(theta, coefs, intercepts)
    theta_coefs, theta_intercepts = fed_avg_w_dp._get_coef_and_inter_slice_from_theta(
        theta, num_feats, num_labels
    )

    assert np.array_equal(coefs, theta_coefs)
    assert np.array_equal(intercepts, theta_intercepts)


def test_standard_dev():
    z = 2
    s = 3
    q = 0.2
    W = 4

    #   (z * s) / (q * W)
    #   (2 * 3) / (0.2 * 4)
    # = 6 / 0.8 = 7.5

    res = fed_avg_w_dp._calc_standard_dev(z, s, q, W)
    assert res == 7.5


def test_batching_user_data():
    batch_size = 2

    user_features = [[1, 2], [3, 4], [5, 6], [7, 8]]
    user_labels = [7, 8, 9, 10]

    batched_feats, batched_labels = fed_avg_w_dp._break_user_update_data_into_batches(
        batch_size, 4, user_features, user_labels
    )

    assert _lists_equal(batched_labels[0], [7, 8])
    assert _lists_equal(batched_feats[0], [[1, 2], [3, 4]])


# Helper functions


def _lists_equal(l1, l2):
    return all(a == b for a, b in zip(l1, l2))


def _vals_are_within_percent_diff_range(v1, v2, max_perc_diff):
    perc_diff = (max(v1, v2) / min(v1, v2)) - 1
    print("v1: {} v2: {}, perc_diff: {}".format(v1, v2, perc_diff))
    return perc_diff <= max_perc_diff
