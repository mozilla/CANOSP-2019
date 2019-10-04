# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from flask import Flask
from flask import request
import argparse

# TODO: this needs to manually copied in right now.

# we should package up the canosp project as a pypi project so we can
# just import it like any other module.

from mozfldp.simulation_util import server_update  # noqa

from sklearn.linear_model import SGDClassifier


app = Flask(__name__)


class ServerFacade:
    """
    This class acts as a facade around the functions in
    `simulation_util` to provide a consistent interface to load data
    into a classifier.
    """

    def __init__(self):
        self._classifier = SGDClassifier(loss="hinge", penalty="l2")

        self._buffer = {"some_client_uuid_1": [], "some_client_uuid_2": []}

    def ingest_client_data(self, client_json):
        # TODO: we need to read one sample of data from a client and
        # add it to a sample buffer.  The intent here is that we want
        # to collect enough data that we can safely sample from the
        # buffer

        # Assume that client_json is some JSON blob with everything we
        # need to call client_update successfully.  We should have a
        # `client_id` key that uniquely identifies the client so that
        # we can pin data to a particular client.

        # TODO: maybe something like this?
        # self._buffer[client_json['client_id']].append(client_json)
        pass

    def update_classifier(self):
        # TODO: this is where the magic happens
        #
        # I think roughly, this should be applying a sample of client
        # data through a training loop.
        #
        # I believe you want to take the code from:
        # https://github.com/mozilla/CANOSP-2019/blob/master/simulation_util.py#L98
        # to
        # https://github.com/mozilla/CANOSP-2019/blob/master/simulation_util.py#L149
        # and update the parameters of the SGDClassifier.
        #
        # The samples `S` in simulation_util.py would be equivalent to self._buffer
        # in this class
        pass

    def classify(self, some_input_json):
        """
        Apply the classifer to some sample input
        """
        reshaped_X_test = None
        reshaped_Y_test = None
        score = self._classifier.score(reshaped_X_test, reshaped_Y_test)
        return score


server = ServerFacade()

# TODO: you'll need to add server routes for `update_classifier` and
# `classify` to allow them to be invoked over the web.


@app.route("/api/v1/client_update/<string:client_id>", methods=["POST"])
def client_update(client_id):
    payload = request.get_json()

    # This is some debugging information so you can see how data gets
    # ingested at the server.
    msg = "Client ID: [{}].\nJSON Payload: {}".format(client_id, str(payload))
    print(msg)

    server.ingest_client_data(payload)

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
