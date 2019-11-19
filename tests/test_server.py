import pytest
import numpy as np 

from mozfldp.server import ServerFacade

NUM_LABELS = 10
NUM_FEATURES = 784
CLIENT_FRACTION = 0.5
NUM_CLIENTS = 10

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

    assert len(server._num_samples) == 0



