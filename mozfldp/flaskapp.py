from flask import Flask
from flask import request
import argparse

app = Flask(__name__)


@app.route("/api/v1/client_update/<string:client_id>", methods=["POST"])
def client_update(client_id):
    payload = request.get_json()

    # This is some debugging information so you can see how data gets
    # ingested at the server.
    msg = "Client ID: [{}].\nJSON Payload: {}".format(client_id, str(payload))
    print(msg)

    # TODO: hook the real server

    return {"result": "ok", "message": "something here"}


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
