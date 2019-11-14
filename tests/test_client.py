# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
import numpy as np

from mozfldp.client import Client
from mozfldp.model import SGDModel

import random

FEATURES = [
    [6, 9, 6],
    [9, 2, 8],
    [0, 5, 4],
    [2, 6, 1],
    [6, 8, 1],
    [6, 1, 6],
    [5, 8, 5],
    [7, 6, 6],
]
LABELS = [1, 0, 1, 0, 1, 0, 1, 0]

# Expected result of batching permuted indices after calling reset_random_seed().
SEEDED_BATCHED_2 = [[1, 5], [0, 7], [2, 4], [3, 6]]
SEEDED_BATCHED_3 = [[1, 5, 0], [7, 2, 4], [3, 6]]


@pytest.fixture
def features():
    return np.array(FEATURES)


@pytest.fixture
def labels():
    return np.array(LABELS)


@pytest.fixture
def model():
    return SGDModel()


@pytest.fixture
def batched_indices_2():
    return [np.array(idx) for idx in SEEDED_BATCHED_2]


@pytest.fixture
def batched_indices_3():
    return [np.array(idx) for idx in SEEDED_BATCHED_3]


def reset_random_seed():
    random.seed(42)
    np.random.seed(42)


def test_batching(features, labels, batched_indices_2, batched_indices_3):
    client = Client(client_id=None, features=features, labels=labels, model=None)

    reset_random_seed()
    # Batch size divides data evenly.
    batches_size_div = client._get_batch_indices(2)
    assert len(batches_size_div) == len(batched_indices_2)
    for actual, expected in zip(batches_size_div, batched_indices_2):
        assert (actual == expected).all()

    reset_random_seed()
    # Batch size does not divide data evenly.
    batches_size_nondiv = client._get_batch_indices(3)
    assert len(batches_size_nondiv) == len(batched_indices_3)
    for actual, expected in zip(batches_size_nondiv, batched_indices_3):
        assert (actual == expected).all()


def test_update_weights():
    # TODO: check weights and iteration counters t_, n_iter_ for correctness.
    pass


def test_update_weights_dp():
    pass
