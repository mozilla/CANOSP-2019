# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest
import numpy as np

from mozfldp.client import Client
from mozfldp.model import SGDModel
from mozfldp.server import client_data_url
from tests.utils import reset_random_seed

import json

CLIENT_ID = 123

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
def model(labels):
    # Turning off shuffling should be enough to make this reproducible.
    model = SGDModel(shuffle=False)
    model.set_training_classes(labels)
    return model


@pytest.fixture
def client(features, labels, model):
    return Client(
        client_id=CLIENT_ID, features=features, labels=labels, model=model.get_clone()
    )


@pytest.fixture
def api_url():
    return client_data_url(str(CLIENT_ID))


@pytest.fixture
def batched_indices_2():
    return [np.array(idx) for idx in SEEDED_BATCHED_2]


@pytest.fixture
def batched_indices_3():
    return [np.array(idx) for idx in SEEDED_BATCHED_3]


def batched_epoch_weights(client, batch_ind, num_epochs, init_coef, init_int):
    # Simulate training for the client to get expected weights.
    model = client._model.get_clone(trained=True)
    model.set_weights(init_coef, init_int)
    for _ in range(num_epochs):
        for bi in batch_ind:
            model.minibatch_update(client._features[bi], client._labels[bi])
    return model.get_weights()


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


def test_update_weights(client, batched_indices_2, api_url, monkeypatch):
    # TODO: check weights and iteration counters t_, n_iter_ for correctness.

    # Don't actually issue any requests, just return the contents.
    def mock_post(url, data=None, json=None):
        return url, data, json

    monkeypatch.setattr("mozfldp.server.requests.post", mock_post)
    init_coefs = np.array([29.0, 0.0, 0.0])
    init_intercept = np.array([-9.0])

    # Train for a single epoch.
    expected_coefs, expected_int = batched_epoch_weights(
        client,
        batch_ind=batched_indices_2,
        num_epochs=1,
        init_coef=init_coefs,
        init_int=init_intercept,
    )
    expected_coef_update = expected_coefs - init_coefs
    expected_int_update = expected_int - init_intercept

    reset_random_seed()
    url, _, json_data = client.submit_weight_updates(
        current_coef=init_coefs,
        current_intercept=init_intercept,
        num_epochs=1,
        batch_size=2,
    )

    assert url == api_url
    request_data = json.loads(json_data)
    assert request_data["user_contrib_weight"] == len(LABELS)
    assert np.array_equal(request_data["coef_update"], expected_coef_update)
    assert np.array_equal(request_data["intercept_update"], expected_int_update)

    # Train over multiple epochs.
    expected_coefs, expected_int = batched_epoch_weights(
        client,
        batch_ind=batched_indices_2,
        num_epochs=3,
        init_coef=init_coefs,
        init_int=init_intercept,
    )
    expected_coef_update = expected_coefs - init_coefs
    expected_int_update = expected_int - init_intercept

    reset_random_seed()
    url, _, json_data = client.submit_weight_updates(
        current_coef=init_coefs,
        current_intercept=init_intercept,
        num_epochs=3,
        batch_size=2,
    )

    assert url == api_url
    request_data = json.loads(json_data)
    assert request_data["user_contrib_weight"] == len(LABELS)
    assert np.array_equal(request_data["coef_update"], expected_coef_update)
    assert np.array_equal(request_data["intercept_update"], expected_int_update)


def test_update_weights_dp():
    pass
