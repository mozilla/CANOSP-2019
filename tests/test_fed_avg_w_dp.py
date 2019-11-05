# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

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


def _lists_equal(l1, l2):
    return all(a == b for a, b in zip(l1, l2))
