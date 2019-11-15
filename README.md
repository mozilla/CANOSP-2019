[![CircleCI](https://circleci.com/gh/mozilla/CANOSP-2019/tree/master.svg?style=svg)](https://circleci.com/gh/mozilla/CANOSP-2019/tree/master)

# CANOSP-2019

This project implements a minimal server that implements federated
learning and accepts messages from clients.

## Setting up the test environment


* Install Python 3.6 or higher
* Setup your virtual environment `python3 -m venv venv`
* Install your dependencies `pip install -r requirements.txt`


## Building a release

* `python setup.py sdist`


## Build Requirements

* docker-ce : https://docs.docker.com/install/


## Build Instructions

Once you have docker-ce installed, you should be able to build this
project using either GNUMake or you can use docker directly.


Build the docker image using:

```
docker build . -t mozfldp:latest
```

## Running the server

You can run the server locally serving requests on port 8000 using:

```
python -m mozfldp.server
```

Alternately, you can run the service in a container using :

```
	docker run -dit -p 127.0.0.1:8090:8000 --name mozfldp -t --rm mozfldp:latest -m mozfldp.server
```

Note that in the above command, we are exposing the container's port
8000 by binding it to port 8090 on the host computer.


## Sending data to the server


You can submit arbitrary JSON blobs to the server using HTTP POST.

A sample curl invocation that will work is:

Note the port `8090` below.

If you are running native locally -the port will be 8000.  Port 8090
is used if you are running in a docker container.

```
curl -d '{"key1":"value1", "key2":"value2"}' \
   -H "Content-Type: application/json" \
   -X POST http://127.0.0.1:8090/api/v1/client_update/some_client_id

Client ID: [some_client_id].
JSON Payload: {'key1': 'value1', 'key2': 'value2'}%
```
