FROM continuumio/miniconda3
ENV PYTHONDONTWRITEBYTECODE 1

MAINTAINER Victor Ng <vng@mozilla.com>

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gettext curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# First copy config files so we can take advantage of docker caching.
COPY . /app

RUN make setup_conda

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate mozfldp && \
    python setup.py install

ENTRYPOINT ["/bin/bash", "/app/bin/run"]
