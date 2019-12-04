from unittest.mock import MagicMock
import json

import numpy as np
import pytest

from mozfldp.server import ServerFacade
from mozfldp.server import app as base_app
from mozfldp.server import compute_new_weights
from tests.utils import reset_random_seed


NUM_LABELS = 2
NUM_FEATURES = 3
CLIENT_FRACTION = 0.5
NUM_CLIENTS = 10
NUM_SAMPLES = 10

NORMAL_NOISE = (
    [[0.49671, -0.13826, 0.64769], [1.52303, -0.23415, -0.23414]],
    [1.57921, 0.76743],
)


@pytest.fixture
def coef():
    return np.zeros((NUM_LABELS, NUM_FEATURES), dtype=np.float64, order="C")


@pytest.fixture
def intercept():
    return np.zeros(NUM_LABELS, dtype=np.float64, order="C")


@pytest.fixture
def normal_noise():
    return [np.array(x) for x in NORMAL_NOISE]


@pytest.fixture
def make_server(coef, intercept):
    def _make_server(avg_denom=None, standard_dev=None):
        return ServerFacade(
            coef, intercept, avg_denom=avg_denom, standard_dev=standard_dev
        )

    return _make_server


@pytest.fixture
def server(make_server):
    return make_server()


@pytest.fixture
def make_server_populated(make_server, coef, intercept):
    def _make_server_populated(avg_denom=None, standard_dev=None):
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

        # add the weights to the server storage
        server = make_server(avg_denom=avg_denom, standard_dev=standard_dev)
        server._client_coef_updates.append(coef1)
        server._client_intercept_updates.append(intercept1)
        server._user_contrib_weights.append(4)

        server._client_coef_updates.append(coef2)
        server._client_intercept_updates.append(intercept2)
        server._user_contrib_weights.append(6)

        return server

    return _make_server_populated


@pytest.fixture
def app():
    app = base_app
    app.facade = MagicMock()
    app.facade.compute_new_weights = MagicMock(
        return_value=(np.array([1, 2, 3]), np.array([4, 5, 6]))
    )
    return app


def test_server_initialization(server, coef, intercept):
    np.testing.assert_array_equal(server._coef, coef)
    np.testing.assert_array_equal(server._intercept, intercept)

    assert len(server._client_coef_updates) == 0
    assert len(server._client_intercept_updates) == 0
    assert len(server._user_contrib_weights) == 0


@pytest.mark.skip("TODO")
def test_reset_dp_params(server):
    avgdenom = 20
    stdev = 4.0
    server.reset_dp_params(avg_denom=avgdenom, standard_dev=stdev)
    assert server._avg_denom == avgdenom
    assert server._standard_dev == stdev

    server.reset_dp_params()
    assert server._avg_denom is None
    assert server._standard_dev is None


def with_noise_array_equals(obs, exp):
    return np.array_equal(np.round(obs, 5), exp)


def test_add_gaussian_noise(make_server, coef, normal_noise):
    server = make_server()
    assert np.array_equal(server._add_gaussian_noise(coef), coef)
    server = make_server(standard_dev=1.0)
    reset_random_seed()
    assert with_noise_array_equals(server._add_gaussian_noise(coef), normal_noise[0])


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


def check_populated_server_output(server, expected, coef, intercept):
    new_coef, new_intercept = server.compute_new_weights()

    expected_coef = np.zeros_like(coef)
    expected_coef.fill(expected)
    expected_intercept = np.zeros_like(intercept)
    expected_intercept.fill(expected)

    assert np.array_equal(new_coef, expected_coef)
    assert np.array_equal(new_intercept, expected_intercept)

    # client data should get cleared
    assert len(server._client_coef_updates) == 0
    assert len(server._client_intercept_updates) == 0
    assert len(server._user_contrib_weights) == 0


def test_compute_new_weights(make_server_populated, coef, intercept, normal_noise):
    # standard fed averaging
    server = make_server_populated()
    # since client 1 has 4 samples and client 2 has 6 samples, the update
    # from client 2 is weighted more - the weighted average between 1 and 2
    # is 1.6
    check_populated_server_output(server, 1.6, coef, intercept)

    # with given denominator
    server = make_server_populated(avg_denom=2)
    check_populated_server_output(server, 8, coef, intercept)

    # with dp params
    server = make_server_populated(avg_denom=2, standard_dev=1.0)
    reset_random_seed()
    new_coef, new_inter = server.compute_new_weights()
    expected_coef = np.zeros_like(coef) + 8 + normal_noise[0]
    expected_intercept = np.zeros_like(intercept) + 8 + normal_noise[1]
    assert with_noise_array_equals(new_coef, expected_coef)
    assert with_noise_array_equals(new_inter, expected_intercept)


def test_request_compute_weights(app):
    """
    Test that compute_new_weights returns valid JSON
    """
    result = compute_new_weights()
    assert result == {"result": "ok", "coef": [1, 2, 3], "intercept": [4, 5, 6]}
