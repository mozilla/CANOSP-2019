import pytest
import numpy as np
import json

from mozfldp.server import ServerFacade

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
    return ServerFacade(
        coef, intercept, num_client=NUM_CLIENTS, client_fraction=CLIENT_FRACTION
    )


def test_server_initialization(server, coef, intercept):
    np.testing.assert_array_equal(server._coef, coef)
    np.testing.assert_array_equal(server._intercept, intercept)

    assert server._num_client == NUM_CLIENTS
    assert server._client_fraction == CLIENT_FRACTION

    assert len(server._client_coefs) == 0
    assert len(server._client_intercepts) == 0
    assert len(server._num_samples) == 0


def test_injest(server, coef, intercept):
    # send weights from the client to the server
    payload = {
        "coefs": coef.tolist(),
        "intercept": intercept.tolist(),
        "num_samples": 5,
    }
    server.ingest_client_data(json.dumps(payload))

    # check that the payload got injested
    assert len(server._client_coefs) == 1
    assert len(server._client_intercepts) == 1
    assert len(server._num_samples) == 1

    # check if the payload is correct
    np.testing.assert_array_equal(server._client_coefs[0], coef)
    np.testing.assert_array_equal(server._client_intercepts[0], intercept)
    assert server._num_samples[0] == 5


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
    server._client_coefs.append(coef1)
    server._client_intercepts.append(intercept1)
    server._num_samples.append(4)

    server._client_coefs.append(coef2)
    server._client_intercepts.append(intercept2)
    server._num_samples.append(6)

    new_coef, new_intercept = server.compute_new_weights()

    # since client 1 has 4 samples and client 2 has 6 samples, the update
    # from client 2 is weighted more - the weighted average between 1 and 2
    # is 1.6
    expected_coef = np.copy(coef)
    expected_intercept = np.copy(intercept)
    expected_coef.fill(1.6)
    expected_intercept.fill(1.6)

    np.testing.assert_array_equal(new_coef, expected_coef)
    np.testing.assert_array_equal(new_intercept, expected_intercept)

