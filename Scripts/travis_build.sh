#!/usr/bin/env bash

set -e

export NUM_JOBS=1

git clone https://bitbucket.org/blaze-lib/blaze/src/master/
cd master
cp -r ./blaze /usr/local/include
cd ../
mkdir build
cd build
cmake ..
make
bin/UnitTests
