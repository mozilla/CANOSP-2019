# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
import numpy as np

from mozfldp.client import Client


def check_batch_indices(batch_ind, data_n, main_batch_size, final_batch_size):
    # batch_ind is a list containing index lists.
    all_ind = np.concatenate(batch_ind)
    # indices in batch_ind should cover 0, 1, ..., n-1.
    assert len(np.setxor1d(all_ind, np.arange(data_n))) == 0
    # number of batches:
    assert len(batch_ind) == np.ceil(data_n / main_batch_size)
    # all index lists should be of the same size, except possibly the last one.
    for i in range(len(batch_ind) - 1):
        assert len(batch_ind[i]) == main_batch_size
    assert len(batch_ind[-1]) == final_batch_size


def test_batching():
    features = np.array(
        [
            [6, 9, 6],
            [9, 2, 8],
            [0, 5, 4],
            [2, 6, 1],
            [6, 8, 1],
            [6, 1, 6],
            [5, 8, 5],
            [7, 6, 6],
        ]
    )
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])

    client = Client(client_id=None, features=features, labels=labels, model=None)

    batches_size_div = client._get_batch_indices(2)
    check_batch_indices(batches_size_div, 8, 2, 2)

    batches_size_nondiv = client._get_batch_indices(3)
    check_batch_indices(batches_size_nondiv, 8, 3, 2)


def test_update_weights():
    # TODO: check weights and iteration counters t_, n_iter_ for correctness.
    pass


def test_update_weights_dp():
    pass
