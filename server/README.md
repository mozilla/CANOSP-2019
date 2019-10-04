# FLDP Server

This project implements a minimal server that implements federated
learning and accepts messages from clients.



## Build Requirements

* docker-ce : https://docs.docker.com/install/


## Build Instructions

Once you have docker-ce installed, you should be able to build this
project using either GNUMake or you can use docker directly.


Build the docker image using:
```
docker build . -t mozfldp:latest
```




## Sending data to the server


You can submit arbitrary JSON blobs to the server using HTTP POST.

A sample curl invocation that will work is:

```
curl -d '{"key1":"value1", "key2":"value2"}' \
   -H "Content-Type: application/json" \
   -X POST http://127.0.0.1:8090/api/v1/client_update/some_client_id

Client ID: [some_client_id].
JSON Payload: {'key1': 'value1', 'key2': 'value2'}%
```
