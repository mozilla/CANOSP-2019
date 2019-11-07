# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from mozfldp import fed_avg_w_dp


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

        res = TestMergeAllUserThetas._run_merge_func(
            TestMergeAllUserThetas.user_thetas[:2]
        )
        assert _lists_equal(res, [15, 35, 50, 65])

    def test_one_weight_returns_theta_over_user_sel_prob(self):

        # By Hand: (weight & weight_sum cancel)
        # Numerator: [1, 2, 3, 4]
        # Denominator: [0.1]

        res = TestMergeAllUserThetas._run_merge_func(
            TestMergeAllUserThetas.user_thetas[:1]
        )
        assert _lists_equal(res, [10, 20, 30, 40])

    def _run_merge_func(
        user_thetas, user_sel_prob=DEFAULT_USER_SEL_PROB, weight_sum=DEFAULT_WEIGHT_SUM
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

        print(clipped_vec)
        assert np.allclose(clipped_vec, np.array([0.6, 0.8]))


#### Small tests ####


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


#### Helper functions ####


def _lists_equal(l1, l2):
    return all(a == b for a, b in zip(l1, l2))


def _vals_are_within_percent_diff_range(v1, v2, max_perc_diff):
    perc_diff = (max(v1, v2) / min(v1, v2)) - 1
    print("v1: {} v2: {}, perc_diff: {}".format(v1, v2, perc_diff))
    return perc_diff <= max_perc_diff
