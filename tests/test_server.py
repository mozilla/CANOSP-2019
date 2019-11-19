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
    return ServerFacade(coef, intercept, num_client=NUM_CLIENTS, client_fraction=CLIENT_FRACTION)


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
        "num_samples": 5
    };
    server.ingest_client_data(json.dumps(payload))

    # check that the payload got injested
    assert len(server._client_coefs) == 1
    assert len(server._client_intercepts) == 1
    assert len(server._num_samples) == 1

    # check if the payload is correct
    np.testing.assert_array_equal(server._client_coefs[0], coef)
    np.testing.assert_array_equal(server._client_intercepts[0], intercept)
    assert server._num_samples[0] == 5
