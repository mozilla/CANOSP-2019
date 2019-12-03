from unittest.mock import MagicMock
import json

import numpy as np
import pytest

from mozfldp.server import ServerFacade
from mozfldp.server import app as base_app
from mozfldp.server import compute_new_weights


NUM_LABELS = 10
NUM_FEATURES = 784
CLIENT_FRACTION = 0.5
NUM_CLIENTS = 10
NUM_SAMPLES = 10


@pytest.fixture
def coef():
    return np.zeros((NUM_LABELS, NUM_FEATURES), dtype=np.float64, order="C")


@pytest.fixture
def intercept():
    return np.zeros(NUM_LABELS, dtype=np.float64, order="C")


@pytest.fixture
def server(coef, intercept):
    return ServerFacade(coef, intercept)


def test_server_initialization(server, coef, intercept):
    np.testing.assert_array_equal(server._coef, coef)
    np.testing.assert_array_equal(server._intercept, intercept)

    assert len(server._client_coef_updates) == 0
    assert len(server._client_intercept_updates) == 0
    assert len(server._user_contrib_weights) == 0


def test_ingest(server, coef, intercept):
    # send weights from the client to the server
    payload = {
        "coef_update": coef.tolist(),
        "intercept_update": intercept.tolist(),
        "user_contrib_weight": 5,
    }
    server.ingest_client_data(json.dumps(payload))

    # check that the payload got injested
    assert len(server._client_coef_updates) == 1
    assert len(server._client_intercept_updates) == 1
    assert len(server._user_contrib_weights) == 1

    # check if the payload is correct
    np.testing.assert_array_equal(server._client_coef_updates[0], coef)
    np.testing.assert_array_equal(server._client_intercept_updates[0], intercept)
    assert server._user_contrib_weights[0] == 5


def test_compute_new_weights(server, coef, intercept):
    # client 1 coefs and intercepts matrices are filled with the value 1
    coef1 = np.copy(coef)
    intercept1 = np.copy(intercept)
    coef1.fill(1)
    intercept1.fill(1)

    # client 2 coefs and intercepts matrices are filled with the value 2
    coef2 = np.copy(coef)
    intercept2 = np.copy(intercept)
    coef2.fill(2)
    intercept2.fill(2)

    # add the weights to the server storage before averaging
    server._client_coef_updates.append(coef1)
    server._client_intercept_updates.append(intercept1)
    server._user_contrib_weights.append(4)

    server._client_coef_updates.append(coef2)
    server._client_intercept_updates.append(intercept2)
    server._user_contrib_weights.append(6)

    new_coef, new_intercept = server.compute_new_weights()

    # since client 1 has 4 samples and client 2 has 6 samples, the update
    # from client 2 is weighted more - the weighted average between 1 and 2
    # is 1.6
    expected_coef_update = np.copy(coef)
    expected_intercept_update = np.copy(intercept)
    expected_coef_update.fill(1.6)
    expected_intercept_update.fill(1.6)

    np.testing.assert_array_equal(new_coef, coef + expected_coef_update)
    np.testing.assert_array_equal(new_intercept, intercept + expected_intercept_update)


@pytest.fixture
def app():
    app = base_app
    app.facade = MagicMock()
    app.facade.compute_new_weights = MagicMock(
        return_value=[np.array([1, 2, 3]), np.array([4, 5, 6])]
    )
    return app


def test_request_compute_weights(app):
    """
    Test that compute_new_weights returns valid JSON
    """
    result = compute_new_weights()
    assert result == {"result": "ok", "weights": [[1, 2, 3], [4, 5, 6]]}
