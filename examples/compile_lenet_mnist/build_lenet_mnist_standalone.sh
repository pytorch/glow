#!/usr/bin/env bash

# Bash strict mode
set -euo pipefail

# Configure the following environment variables based on your setup.

# Path to the Glow loader executable.
LOADER="${LOADER:-~/src/build/Glow/bin/loader}"
# The command to invoke a c++ compiler.
CXX="${CXX:-clang++}"

# Get path to this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# The root directory of the Glow source code.
GLOW_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# This file downloads the lenet_mnist model, uses the loader to create a
# bundle from it and then links it with the client-code from
# lenet_mnist_standalone.cpp to create a small standalone executable.

# Download the lenet_mnist Caffe2 model.
for file in predict_net.pbtxt predict_net.pb init_net.pb; do
  wget http://fb-glow-assets.s3.amazonaws.com/models/lenet_mnist/$file -P lenet_mnist -nc 
done

# Use loader to create a bundle.
# This command will produce two files:
# lenet_mnist.o is the object file containing the compiled model
# lenet_mnist.weights is the file with weights
mkdir -p build
${LOADER} ${GLOW_ROOT}/tests/images/mnist/5_1087.png -image_mode=0to1 -d lenet_mnist -cpu -emit-bundle build -g

# Compile lenet_mnist_standalone.cpp.
${CXX} -std=c++11 -c -g ${SCRIPT_DIR}/lenet_mnist_standalone.cpp -o build/lenet_mnist_standalone.o

# Produce a standalone executable by linking
# lenet_mnist_standalone.o and lenet_mnist.o
${CXX} -o build/lenet_mnist_standalone build/lenet_mnist_standalone.o build/lenet_mnist.o -lpng

pushd build

# Test the compiled model in the mnist images.
for image in ${GLOW_ROOT}/tests/images/mnist/*; do
  ./lenet_mnist_standalone ${image}
done

popd
