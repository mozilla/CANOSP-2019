# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import argparse
import json


app = Flask(__name__)


class ServerFacade:
    """
    This class acts as a facade around the functions in
    `simulation_util` to provide a consistent interface to load data
    into a classifier.

    Args:
        coef: initial coefficients to initialize the server with
        inomtercept: initial intercepts to initialize the server with
    """

    def __init__(self, coef, intercept):
        self._coef = coef
        self._intercept = intercept

        self.reset_client_data()

    def reset_client_data(self):
        self._client_coefs = []
        self._client_intercepts = []
        self._num_samples = []

    def ingest_client_data(self, client_json):
        """
        Accepts new weights from a client and stores them on the server side for averaging

        Args:
            client_json: a json object containing coefs, intercepts, and num_samples
        """
        client_json = json.loads(client_json)
        self._client_coefs.append(client_json["coefs"])
        self._client_intercepts.append(client_json["intercept"])
        self._num_samples.append(client_json["num_samples"])

    def compute_new_weights(self):
        """
        Applies the federated averaging on the stored client weights for this round
        and return the new weights
        """

        new_coefs = np.zeros(self._coef.shape, dtype=np.float64, order="C")
        new_intercept = np.zeros(self._intercept.shape, dtype=np.float64, order="C")

        total_samples = sum(self._num_samples)

        for index, (client_coef, client_intercept, n_k) in enumerate(
            zip(self._client_coefs, self._client_intercepts, self._num_samples)
        ):
            added_coef = np.array(client_coef) * n_k / total_samples
            added_intercept = np.array(client_intercept) * n_k / total_samples

            new_coefs = np.add(new_coefs, added_coef)
            new_intercept = np.add(new_intercept, added_intercept)

        # update the server weights to newly calculated weights
        self._coef = new_coefs
        self._intercept = new_intercept

        # reset all client data so it doesn't get used for the next round
        self.reset_client_data()

        return self._coef, self._intercept

    def compute_new_weights_dp(self, standard_dev, avg_denom, indiv_client_weights, user_sel_prob):
        """
        Applies the DP-protected federated averaging on the stored client weights
        for this round and return the new weights.

        standard_dev: the standard deviation of the random noise to apply
        avg_denom: the denominator to use in computing the average
        """
        # TODO: DP version of fed averaging.

        coefs, inters = self._merge_all_user_thetas(avg_denom, indiv_client_weights, user_sel_prob)

        coefs += self._gen_gausian_rand_noise(standard_dev, len(self._client_coefs))
        inters += self._gen_gausian_rand_noise(standard_dev, len(self._client_intercepts))

        self._client_coefs += coefs
        self._client_intercepts += inters

        self.reset_client_data()

        return self._client_coefs, self._client_intercepts


    def _gen_gausian_rand_noise(standard_dev, vec_len):
        """
        Generates gausian noise and applies to all elements in a vector.

        stndrd_dev: The standard deviation of the distrubution to sample from
        vec_len: The number of elements in the vector

        returns: The vector after noise has been applied
        """

        return np.random.normal(loc=0.0, scale=stndrd_dev, size=vec_len)

    def _merge_all_user_thetas(self, weight_sum, user_weights, user_sel_prob):
        """
        Merge all user updates for a round into a single delta (vector).

        user_sel_prob: Probability of any given user being selected for a round
        weight_sum: The sum of all user weights
        user_updates_buf: The user updates (thetas) that we are merging.
        user_weights: The weights applied to each user. Users with more data have more weight.
        num_theta_elems: The number of elements in theta.
        """

        num_users_in_batch = len(self._client_coefs)
        merged_coefs = np.zeros(num_users_in_batch, dtype=np.float64, order="C")
        merged_inters = np.zeros(num_users_in_batch, dtype=np.float64, order="C")

        for i in range(num_users_in_batch):
            weighted_user_coefs = np.multiply(self._client_coefs[i], user_weights[i])
            weighted_user_inters = np.multiply(self._client_intercepts[i], user_weights[i])

            merged_coefs = np.add(merged_coefs, weighted_user_coefs)
            merged_inters = np.add(merged_inters, weighted_user_inters)

        divisor = user_sel_prob * weight_sum
        merged_coefs = np.divide(merged_coefs, divisor)
        merged_inters = np.divide(merged_inters, divisor)

        return merged_coefs, merged_inters




class InvalidClientData(Exception):
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


@app.errorhandler(InvalidClientData)
def handle_invalid_client_data(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/api/v1/ingest_client_data", methods=["POST"])
def ingest_client_data(client_id):
    payload = request.get_json()
    try:
        app.facade.ingest_client_data(payload)
        return {"result": "ok"}
    except Exception as exc:
        raise InvalidClientData(
            "Error updating client", payload={"exception": str(exc)}
        )


@app.route("/api/v1/compute_new_weights", methods=["POST"])
def compute_new_weights():
    try:
        weights = app.facade.compute_new_weights()
        json_safe_weights = [w.tolist() for w in weights]
        return {"result": "ok", "weights": json_safe_weights}
    except Exception as exc:
        raise InvalidClientData(
            "Error computing weights", payload={"exception": str(exc)}
        )


def flaskrun(app, default_host="0.0.0.0", default_port="8000"):
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
    flaskrun(app)
