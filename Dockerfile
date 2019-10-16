FROM python:3.6.9-buster
ENV PYTHONDONTWRITEBYTECODE 1

MAINTAINER Victor Ng <vng@mozilla.com>

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gettext curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# First copy requirements.txt so we can take advantage of docker
# caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN python setup.py install

ENTRYPOINT ["/usr/local/bin/python"]
