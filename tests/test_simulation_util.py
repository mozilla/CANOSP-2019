# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
from mozfldp.simulation_util import client_update
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


@pytest.mark.skip("some reason goes here")
def test_some_known_bug_but_we_want_to_skip_the_test():
    raise StandardError("some message here")
