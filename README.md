[![CircleCI](https://circleci.com/gh/mozilla/CANOSP-2019/tree/master.svg?style=svg)](https://circleci.com/gh/mozilla/CANOSP-2019/tree/master)

# CANOSP-2019

This project implements a minimal server that can perform federated learning
with differential privacy and accepts messages from clients.


## Getting Started

We are using Miniconda to manage the environment. It can be installed using one
of the installers available [here](https://docs.conda.io/en/latest/miniconda.html).
For MacOS, the bash installer is recommended.

Make sure to have `conda init` run during conda installation so that
your PATH is set properly.


## Installing locally to run tests

To install and run the tests for this project you can run:

```bash
# Set up the environment.
$ make setup_conda
$ conda activate mozfldp

# Run tests
$ make pytest
```

## Running the server

You can run the server locally, serving requests on port 8000, using:

```bash
$ python -m mozfldp.server
```

## Building a release

```bash
python setup.py sdist
```


## Running from Docker

The server can also be built and run as a Docker container. 
First, install [Docker](https://docs.docker.com/get-docker/).

Once you have Docker installed, you can build the container and run tests using:

```bash
$ make build_image
$ make docker_tests
```

To run the service in the container, use:

```bash
$ make up
```

Note that in the above command, we are exposing the container's port
8000 by binding it to port 8090 on the host computer.


## Sending data to the server


You can submit arbitrary JSON blobs to the server using HTTP POST.

A sample curl invocation that will work is:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/compute_new_weights

{"result":"ok","weights":[[[0.0,0.0,0.0,.... }
```

Note: If you are running locally, the port will be 8000. Port 8090 is used if you are running in a docker container.

