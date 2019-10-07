.PHONY: tests

all:
	cp ../simulation_util.py mozfldp
	docker build . -t mozfldp:latest

# Bind 127.0.0.1, port 8090 to the container's port 8000
# and start the server.
#
# Name the container 'mozfldp' so that we can stop the container
# easily
run:
	docker container stop mozfldp || true
	docker run -dit -p 127.0.0.1:8090:8000 --name mozfldp -t --rm mozfldp:latest -m mozfldp.server

stop:
	docker stop mozfldp

tests:
	pytest
