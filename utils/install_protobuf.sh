#!/bin/sh
set -ex

pb_dir="$HOME/.cache/pb"
mkdir -p "$pb_dir"

wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz
tar -xzf protobuf-2.6.1.tar.gz -C "$pb_dir" --strip-components 1
cd "$pb_dir" && ./configure CC=gcc CXX=g++ CXX_FOR_BUILD=g++ --prefix=/usr && make -j2 && sudo make install
