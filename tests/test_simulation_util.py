# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
from mozfldp.simulation_util import client_update, server_update
import numpy as np

def test_simulation_util():
    pass

def test_client_update():
    features = [[1, 4, 3], [0, 2, 2], [1, 4, 0], [0, 5, 3], [1, 2, 1], [0, 2, 9]]
    labels = [1, 0, 1, 0, 1, 0]
    all_classes = [0, 1]

    coefs = np.array([29., 0., 0.])
    intercepts = np.array([-9.])
    weights = [coefs, intercepts]

    epochs = 3
    batch_size = 3

    new_weights = client_update(
        weights,
        epochs,
        batch_size,
        features,
        labels,
        all_classes,
        rand_seed=0
    )
    new_coefs = new_weights[0][0].tolist()
    new_intercepts = new_weights[1].tolist()

    expected_coefs = [28.486725914466938, -0.02850360575641961, -0.04347976804480432]
    expected_intercepts = [-9.009666726361056]

    assert new_coefs == expected_coefs
    assert new_intercepts == expected_intercepts

def test_server_update():
    np.random.seed(0)

    init_weights = [np.array([0., 0., 0.]), np.array([0.])] 
    num_client = 10
    samples_per_client = 10
    num_features = 3
    features = np.random.randint(10, size=(num_client, samples_per_client, num_features))
    labels = np.random.randint(2, size=(num_client, samples_per_client))

    classifier = server_update(
        init_weights,
        client_fraction=0.5,
        num_rounds=2,
        features=features,
        labels=labels,
        epoch=2,
        batch_size=3,
        display_weight_per_round=False,
        rand_seed=0
    )
    new_coefs = classifier.coef_.tolist()[0]
    new_intercepts = classifier.intercept_.tolist()

    expected_coefs = [-42.21600518468813, 3.4301014509740986, 30.442381328564878]
    expected_intercepts = [-1.0702357064270938]

    assert new_coefs == expected_coefs
    assert new_intercepts == expected_intercepts


@pytest.mark.skip("some reason goes here")
def test_some_known_bug_but_we_want_to_skip_the_test():
    raise StandardError("some message here")
