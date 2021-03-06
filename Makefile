.PHONY: upload_pypi tests image up stop setup_conda lint pytest

IMAGE_NAME=mozfldp:latest

all: pytest

build_image:
	# Build the docker image
	docker build . -t $(IMAGE_NAME)

upload_pypi:
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

setup_conda:
	# Install dependencies
	conda env update -n mozfldp -f environment.yml

pytest: lint
	pytest

# tests:
# 	python setup.py pytest

lint:
	flake8 mozfldp tests

docker_tests:
	docker run -it \
		-p 127.0.0.1:8090:8000 \
		--name mozfldp \
		-t --rm $(IMAGE_NAME) \
		test

####
#### Use `make up` and `make stop` to run the Docker container locally
####

up:
	# Bind 127.0.0.1, port 8090 to the container's port 8000
	# and start the server.
	#
	# Name the container 'mozfldp' so that we can stop the container
	# easily
	# docker container stop mozfldp || true
	docker run -eit \
		-p 127.0.0.1:8090:8000 \
		--name mozfldp \
		-t --rm $(IMAGE_NAME) \
		web

stop:
	docker stop mozfldp
