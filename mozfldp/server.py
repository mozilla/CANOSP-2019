# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from flask import Flask
from flask import request
import numpy as np
import argparse
import json

# TODO: this needs to manually copied in right now.

# we should package up the canosp project as a pypi project so we can
# just import it like any other module.

app = Flask(__name__)


class ServerFacade:
    """
    This class acts as a facade around the functions in
    `simulation_util` to provide a consistent interface to load data
    into a classifier.

    Args: 
        coef: initial coefficients to initialize the server with
        intercept: initial intercepts to initialize the server with
        num_client: number of clients that are available
        client_fraction: (C in algorithm) - fraction of clients that we want to use in a round
    """

    def __init__(self, coef, intercept, num_client, client_fraction):
        self._coef = coef
        self._intercept = intercept

        self._num_client = num_client
        self._client_fraction = client_fraction

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

        for index, (client_coef, client_intercept) in enumerate(
            zip(self._client_coefs, self._client_intercepts)
        ):
            n_k = self._num_samples[index]
            added_coef = [
                np.array(value) * (n_k) / sum(self._num_samples) for value in client_coef
            ]
            added_intercept = [
                np.array(value) * (n_k) / sum(self._num_samples) for value in client_intercept
            ]

            new_coefs = np.add(new_coefs, added_coef)
            new_intercept = np.add(new_intercept, added_intercept)

        # update the server weights to newly calculated weights
        print("Updated Weights: ", new_coefs, new_intercept)

        self._coef = new_coefs
        self._intercept = new_intercept

        # reset the state for the next round of learning
        self._client_coefs = None
        self._client_intercepts = None
        self._num_samples = []

        return self._coef, self._intercept


# TODO: you'll need to add server routes for `update_classifier` and
# `classify` to allow them to be invoked over the web.


@app.route("/api/v1/client_update/<string:client_id>", methods=["POST"])
def client_update(client_id):
    payload = request.get_json()

    # This is some debugging information so you can see how data gets
    # ingested at the server.
    msg = "Client ID: [{}].\nJSON Payload: {}".format(client_id, str(payload))
    print(msg)

    # TODO: you probably want to send some kind of useful feedback to
    # clients that the data was ingested by the server
    return {"result": "ok", "message": msg}


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

    app.run(debug=args.debug, host=args.host, port=int(args.port))


if __name__ == "__main__":
    flaskrun(app)
