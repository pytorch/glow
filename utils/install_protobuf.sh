#!/bin/sh
set -ex
wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz
tar -xzvf protobuf-2.6.1.tar.gz
cd protobuf-2.6.1 && ./configure CC=gcc CXX=g++ CXX_FOR_BUILD=g++ --prefix=/usr && make && sudo make install
