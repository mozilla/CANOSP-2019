# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
from exceptions import StandardError

import numpy as np
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from mozfldp.server import Server


@pytest.mark.skip("some reason goes here")
def test_some_known_bug_but_we_want_to_skip_the_test():
    raise StandardError("some message here")


def test_sgd_example():
    """
    This test case was copied directly from:

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#
    """
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    Y = np.array([1, 1, 2, 2])
    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(X, Y)

    assert [1] == clf.predict([[-0.8, -1]])


def test_server():
    classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=1)
    classifier.fit(
        batches_features[i],
        batches_labels[i],
        coef_init=coef,
        intercept_init=intercept,
    )
