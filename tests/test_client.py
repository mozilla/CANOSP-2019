# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
import numpy as np
from unittest.mock import Mock, patch

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
def client(features, labels, model):
    return Client(client_id=None, features=features, labels=labels, model=model)


@pytest.fixture
def batched_indices_2():
    return [np.array(idx) for idx in SEEDED_BATCHED_2]


@pytest.fixture
def batched_indices_3():
    return [np.array(idx) for idx in SEEDED_BATCHED_3]


def reset_random_seed():
    random.seed(42)
    np.random.seed(42)


def compare_batch_indices(actual, expected):
    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        assert np.array_equal(act, exp)


def test_batching(client, batched_indices_2, batched_indices_3):
    reset_random_seed()
    # Batch size divides data evenly.
    batches_size_div = client._get_batch_indices(2)
    compare_batch_indices(batches_size_div, batched_indices_2)

    reset_random_seed()
    # Batch size does not divide data evenly.
    batches_size_nondiv = client._get_batch_indices(3)
    compare_batch_indices(batches_size_nondiv, batched_indices_3)

    reset_random_seed()
    # If batch size exceeds data size, all data should be in a single batch.
    batches_size_exceed = client._get_batch_indices(20)
    batch_ind = [np.ravel(batched_indices_2)]
    compare_batch_indices(batches_size_exceed, batch_ind)

def test_update_weights(client, monkeypatch, batched_indices_2):
    # TODO: check weights and iteration counters t_, n_iter_ for correctness.

    # For now, skip model updating and just record what data gets passed.
    model_update_data = []

    def mock_model_update(X, y):
        model_update_data.append((X, y))

    monkeypatch.setattr(client, "_run_model_update_step", mock_model_update)

    reset_random_seed()
    # mock the post request 
    mock_post_patcher = patch('mozfldp.client.requests.post')
    mock_post = mock_post_patcher.start()
    mock_post.return_value.ok = True
    client.update_and_submit_weights(
        current_coef=None, current_intercept=None, num_epochs=1, batch_size=2
    )
    assert len(model_update_data) == len(batched_indices_2)
    for (feat, lab), exp_ind in zip(model_update_data, batched_indices_2):
        assert np.array_equal(feat, client._features[exp_ind])
        assert np.array_equal(lab, client._labels[exp_ind])

    reset_random_seed()
    model_update_data = []
    client.update_and_submit_weights(
        current_coef=None, current_intercept=None, num_epochs=3, batch_size=2
    )
    batched_ind = batched_indices_2 * 3
    print(batched_ind)
    assert len(model_update_data) == len(batched_ind)
    for (feat, lab), exp_ind in zip(model_update_data, batched_ind):
        assert np.array_equal(feat, client._features[exp_ind])
        assert np.array_equal(lab, client._labels[exp_ind])

    # stop mocking the request
    mock_post_patcher.stop()

def test_update_weights_dp():
    pass
