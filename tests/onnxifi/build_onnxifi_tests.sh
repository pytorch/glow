#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
