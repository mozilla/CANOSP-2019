# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import requests
from decouple import config

import argparse
import json

app = Flask(__name__)

HOSTNAME = config("FLDP_HOST", "127.0.0.1")
PORT = config("FLDP_PORT", 8000)

API_BASE_URL = "http://{hostname:s}:{port:d}/api/v1".format(
    hostname=HOSTNAME, port=PORT
)


class ServerRequestError(Exception):
    """When issuing a request to the server fails for some reason."""

    def __init__(self, message, response):
        Exception.__init__(self)
        self.message = message
        self.response = response


def client_data_url(client_id):
    """Return the server API URL for submitting weights for the given client."""
    return "{base_url}/ingest_client_data/{id}".format(
        base_url=API_BASE_URL, id=client_id
    )


def _jsonify_array(arr):
    """Prepare a numpy array for passing to json.dump."""
    return arr.tolist()


def _check_repsonse_code(response, message):
    if response.status_code != requests.codes.ok:
        raise ServerRequestError(message, response)


def submit_client_data_request(
    client_id, coef_update, intercept_update, user_contrib_weight
):
    """Wrap and submit model updates to the currently running server.

    coef_update, intercept_update: numpy arrays containing weight updates
    user_contrib_weight: scalar weight value
    """
    client_data = {
        "coef_update": _jsonify_array(coef_update),
        "intercept_update": _jsonify_array(intercept_update),
        "user_contrib_weight": user_contrib_weight,
    }
    payload = json.dumps(client_data)

    # send the post request to update the weights
    api_endpoint = client_data_url(client_id)
    response = requests.post(url=api_endpoint, json=payload)
    _check_repsonse_code(
        response, "Error submitting client data for {}".format(client_id)
    )


def compute_weights_request():
    """Request centrally averaged weights from the server.

    Returns the updated coef and intercept.
    """
    response = requests.get("{}/compute_new_weights".format(API_BASE_URL))
    _check_repsonse_code(response, "Error computing server weights")

    try:
        payload = response.json()
        return np.array(payload["coef"]), np.array(payload["intercept"])
    except Exception as exc:
        raise ServerRequestError(
            "Error retrieving server weights: {}".format(str(exc.message)), response
        )


def clear_client_data_request():
    """Clear client data in the currently running server."""
    response = requests.get("{}/clear_client_data".format(API_BASE_URL))
    _check_repsonse_code(response, "Error clearing client data")


def reset_server_params_request(coef, intercept, avg_denom=None, standard_dev=None):
    """Reset the server params to initialize a new training session."""
    params = {"coef": _jsonify_array(coef), "intercept": _jsonify_array(intercept)}
    if avg_denom is not None:
        params["avg_denom"] = avg_denom
    if standard_dev is not None:
        params["standard_dev"] = standard_dev
    payload = json.dumps(params)

    response = requests.post(
        "{}/reset_server_params".format(API_BASE_URL), json=payload
    )
    _check_repsonse_code(response, "Error resetting server params")


def dump_server_state_request():
    """Request full server state.

    Returns the server response.
    """
    response = requests.get("{}/dump_state".format(API_BASE_URL))
    _check_repsonse_code(response, "Error dumping server state")

    try:
        payload = response.json()["server_state"]
        payload["coef"] = np.array(payload["coef"])
        payload["intercept"] = np.array(payload["intercept"])
        payload["client_coef_updates"] = [
            np.array(x) for x in payload["client_coef_updates"]
        ]
        payload["client_intercept_updates"] = [
            np.array(x) for x in payload["client_intercept_updates"]
        ]
        return payload
    except Exception as exc:
        raise ServerRequestError("Error retrieving server state", response)


class ServerFacade:
    """
    This class acts as a facade around the functions in
    `simulation_util` to provide a consistent interface to load data
    into a classifier.

    Args:
        coef: initial coefficients to initialize the server with
        intercept: initial intercepts to initialize the server with
    """

    def __init__(self, coef, intercept, avg_denom=None, standard_dev=None):
        self._coef = np.copy(coef)
        self._intercept = np.copy(intercept)
        self._client_coef_updates = []
        self._client_intercept_updates = []
        self._user_contrib_weights = []
        # DP params
        self._avg_denom = avg_denom
        self._standard_dev = standard_dev

    def dump_state(self):
        return {
            "coef": self._coef,
            "intercept": self._intercept,
            "client_coef_updates": self._client_coef_updates,
            "client_intercept_updates": self._client_intercept_updates,
            "user_contrib_weights": self._user_contrib_weights,
            "avg_denom": self._avg_denom,
            "standard_dev": self._standard_dev,
        }

    def clear_client_data(self):
        self._client_coef_updates.clear()
        self._client_intercept_updates.clear()
        self._user_contrib_weights.clear()

    def reset_server_params(self, coef, intercept, avg_denom=None, standard_dev=None):
        """Reset the server parameters on initializing a training session."""
        self._coef = np.copy(coef)
        self._intercept = np.copy(intercept)
        self._avg_denom = avg_denom
        self._standard_dev = standard_dev
        self.clear_client_data()

    def ingest_client_data(self, client_json):
        """
        Accepts weight updates from a client and stores them on the server side
        for averaging

        Args:
            client_json: a json object containing coef_update, intercept_update,
                and user_contrib_weights
        """
        client_json = json.loads(client_json)
        self._client_coef_updates.append(client_json["coef_update"])
        self._client_intercept_updates.append(client_json["intercept_update"])
        self._user_contrib_weights.append(client_json["user_contrib_weight"])

    def compute_new_weights(self):
        """Apply Federated Averaging on the stored client weight updates.

        DP protection is applied if the necessary DP parameters are available.

        Returns the new model weights.
        """

        final_coef_udpate = np.zeros(self._coef.shape, dtype=np.float64, order="C")
        final_int_update = np.zeros(self._intercept.shape, dtype=np.float64, order="C")

        avg_denom = self._avg_denom
        if avg_denom is None:
            avg_denom = sum(self._user_contrib_weights)

        for (coef_update, int_update, w_k) in zip(
            self._client_coef_updates,
            self._client_intercept_updates,
            self._user_contrib_weights,
        ):
            final_coef_udpate += np.array(coef_update) * w_k
            final_int_update += np.array(int_update) * w_k

        # update the server weights to newly calculated weights
        self._coef += final_coef_udpate / avg_denom
        self._intercept += final_int_update / avg_denom

        # add noise if required
        # this only modifies the weights if self._standard_dev is set.
        self._coef = self._add_gaussian_noise(self._coef)
        self._intercept = self._add_gaussian_noise(self._intercept)

        # reset all client data so it doesn't get used for the next round
        self.clear_client_data()

        return np.copy(self._coef), np.copy(self._intercept)

    def _add_gaussian_noise(self, arr):
        """Add independent Gaussian random noise to each element of an array.

        Noise uses the current instance-level standard deviation value. If none
        is set, the array is returned unchanged.
        """
        if self._standard_dev is None:
            return arr

        return arr + np.random.normal(loc=0.0, scale=self._standard_dev, size=arr.shape)


class ServerError(Exception):
    status_code = 500

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


@app.errorhandler(ServerError)
def handle_server_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/api/v1/ingest_client_data/<string:client_id>", methods=["POST"])
def ingest_client_data(client_id):
    payload = request.get_json()
    try:
        app.facade.ingest_client_data(payload)
        return {"result": "ok"}
    except Exception as exc:
        raise ServerError("Error updating client", payload={"exception": str(exc)})


@app.route("/api/v1/compute_new_weights")
def compute_new_weights():
    try:
        coef, inter = app.facade.compute_new_weights()
        return {
            "result": "ok",
            "coef": _jsonify_array(coef),
            "intercept": _jsonify_array(inter),
        }
    except Exception as exc:
        raise ServerError("Error computing weights", payload={"exception": str(exc)})


@app.route("/api/v1/clear_client_data")
def clear_client_data():
    try:
        app.facade.clear_client_data()
        return {"result": "ok"}
    except Exception as exc:
        raise ServerError("Error clearing client data", payload={"exception": str(exc)})


@app.route("/api/v1/reset_server_params", methods=["POST"])
def reset_server_params():
    payload = request.get_json()
    try:
        params_dict = json.loads(payload)
        params_dict["coef"] = np.array(params_dict["coef"])
        params_dict["intercept"] = np.array(params_dict["intercept"])
        app.facade.reset_server_params(**params_dict)
        return {"result": "ok"}
    except Exception as exc:
        raise ServerError(
            "Error resetting server params", payload={"exception": str(exc)}
        )


@app.route("/api/v1/dump_state")
def dump_server_state():
    """Dump internal server data (mainly for testing)."""
    try:
        state = app.facade.dump_state()
        state["coef"] = _jsonify_array(state["coef"])
        state["intercept"] = _jsonify_array(state["intercept"])
        state["client_coef_updates"] = [
            _jsonify_array(x) for x in state["client_coef_updates"]
        ]
        state["client_intercept_updates"] = [
            _jsonify_array(x) for x in state["client_intercept_updates"]
        ]
        return {"result": "ok", "server_state": state}
    except Exception as exc:
        raise ServerError("Error dumping server state", payload={"exception": str(exc)})


def flaskrun(app, default_host=HOSTNAME, default_port=PORT):
    """
    Takes a flask.Flask instance and runs it. Parses
    command-line flags to configure the app.
    """

    # Set up the command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-H",
        "--host",
        help="Hostname of the Flask app " + "[default %s]" % default_host,
        default=default_host,
    )
    parser.add_argument(
        "-P",
        "--port",
        help="Port for the Flask app " + "[default %s]" % default_port,
        default=default_port,
    )

    # Two options useful for debugging purposes, but
    # a bit dangerous so not exposed in the help message.
    parser.add_argument("-d", "--debug", action="store_true", dest="debug")
    parser.add_argument("-p", "--profile", action="store_true", dest="profile")

    args = parser.parse_args()

    # If the user selects the profiling option, then we need
    # to do a little extra setup
    if args.profile:
        from werkzeug.contrib.profiler import ProfilerMiddleware

        app.config["PROFILE"] = True
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
        args.debug = True

    NUM_LABELS = 10
    NUM_FEATURES = 784
    coef = np.zeros((NUM_LABELS, NUM_FEATURES), dtype=np.float64, order="C")
    intercept = np.zeros(NUM_LABELS, dtype=np.float64, order="C")

    app.facade = ServerFacade(coef, intercept)

    app.run(debug=args.debug, host=args.host, port=int(args.port))


if __name__ == "__main__":
    with app.app_context():
        flaskrun(app)
