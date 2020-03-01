FROM ubuntu:18.04
LABEL maintainer "Chris Ohk <utilforever@gmail.com>"

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ca-certificates \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app/build

RUN git clone https://bitbucket.org/blaze-lib/blaze/src/master/ && \
    cd master && \
    cp -r ./blaze /usr/local/include && \
    cd ../

RUN cmake .. && \
    make -j "$(nproc)" && \
    make install && \
    bin/UnitTests