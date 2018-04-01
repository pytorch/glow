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

# The name of the neural network model.
NETWORK_MODEL=resnet50

# This file downloads the ${NETWORK_MODEL} model, uses the loader to create a
# bundle from it and then links it with the client-code from
# ${NETWORK_MODEL}_standalone.cpp to create a small standalone executable.

# Download the Caffe2 model.
for file in predict_net.pbtxt predict_net.pb init_net.pb; do
  wget http://fb-glow-assets.s3.amazonaws.com/models/$NETWORK_MODEL/$file -P $NETWORK_MODEL -nc 
done

# Use loader to create a bundle.
# This command will produce two files:
# ${NETWORK_MODEL}.o is the object file containing the compiled model
# ${NETWORK_MODEL}.weights is the file with weights
mkdir -p build
${LOADER} ${GLOW_ROOT}/tests/images/imagenet/cat_285.png -image_mode=0to1 -d ${NETWORK_MODEL} -cpu -emit-bundle build -g

# Compile lenet_mnist_standalone.cpp.
${CXX} -std=c++11 -c -g ${SCRIPT_DIR}/${NETWORK_MODEL}_standalone.cpp -o build/${NETWORK_MODEL}_standalone.o

# Produce a standalone executable by linking
# ${NETWORK_MODEL}_standalone.o and ${NETWORK_MODEL}.o
${CXX} -o build/${NETWORK_MODEL}_standalone build/${NETWORK_MODEL}_standalone.o build/${NETWORK_MODEL}.o -lpng

pushd build

# Test the compiled model using the imagenet images.
for image in ${GLOW_ROOT}/tests/images/imagenet/*; do
  ./${NETWORK_MODEL}_standalone ${image}
done

popd
