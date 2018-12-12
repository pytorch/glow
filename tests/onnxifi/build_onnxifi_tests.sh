#!/bin/bash

# Should be run from Glow source directory.
# Builds Onnxifi test driver.

GLOW_DIR=$PWD
ONNX_DIR="${GLOW_DIR}/thirdparty/onnx"
ONNX_BUILD_DIR="${ONNX_DIR}/build_onnx"

build() {
  rm -rf ${ONNX_BUILD_DIR}
  cd ${ONNX_DIR}
  mkdir build_onnx && cd build_onnx
  cmake -GNinja -DONNX_BUILD_TESTS=ON -DONNXIFI_DUMMY_BACKEND=OFF ../
  ninja onnxifi_test_driver_gtests
}

build
