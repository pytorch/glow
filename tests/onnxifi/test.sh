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
# Build Glow in glow/build and build_onnxifi_tests.sh then run this script.
# Accepts one parameter, the name of the test to run. If no test name is
# provided then it will run all ONNXIFI tests except those excluded by
# crashed_test.txt and failed_tests.txt.

GLOW_DIR=$PWD
GLOW_BUILD_DIR="${GLOW_DIR}/build"
ONNX_DIR="${GLOW_DIR}/thirdparty/onnx"
ONNX_BUILD_DIR="${ONNX_DIR}/build_onnx"
GLOW_ONNXIFI_DIR="${GLOW_BUILD_DIR}/lib/Onnxifi"

# Name of the specific test to run, if empty will run all tests except
# crashing/failing tests.
TEST_NAME=$1

# Check that glow/build exists.
check_glow_build_dir_exist() {
  if [ ! -d "$GLOW_BUILD_DIR" ]; then
    echo "Expected Glow to be built in ${GLOW_BUILD_DIR}"
    exit 1
  fi
}

# Copy libonnxifi-glow(.so|.dylib") dynamic library to libonnxifi(.so|.dylib").
copy_glow_lib() {
  COPIED=0

  if [ -f "${GLOW_ONNXIFI_DIR}/libonnxifi-glow.dylib" ]; then
   cp "${GLOW_ONNXIFI_DIR}/libonnxifi-glow.dylib" "${GLOW_ONNXIFI_DIR}/libonnxifi.dylib"
   export DYLD_LIBRARY_PATH=${GLOW_ONNXIFI_DIR}
   COPIED=1
  fi

  if [ -f "${GLOW_ONNXIFI_DIR}/libonnxifi-glow.so" ]; then
   cp "${GLOW_ONNXIFI_DIR}/libonnxifi-glow.so" "${GLOW_ONNXIFI_DIR}/libonnxifi.so"
   export LD_LIBRARY_PATH=${GLOW_ONNXIFI_DIR}
   COPIED=1
  fi

  if [ $COPIED -eq 0 ]; then
   echo "Could not find the libonnxifi-glow dynamic library in ${GLOW_ONNXIFI_DIR}"
   exit 1
  fi
}

# Run ONNXIFI node test specified by TEST_NAME or if it's empty run all tests
# except those listed in failed_tests.txt or crashed_tests.txt.
run_node_tests() {
  cd "${GLOW_BUILD_DIR}"
  ONNX_TESTDATA_DIR="${ONNX_DIR}/onnx/backend/test/data/node"
  if [ "$TEST_NAME" != "" ]; then
    GTEST_FILTER="*$TEST_NAME*"
  else
    CRASHED_TEST_CASES="$(paste -sd: "${GLOW_DIR}"/tests/onnxifi/crashed_tests.txt)"
    FAILED_TEST_CASES="$(paste -sd: "${GLOW_DIR}"/tests/onnxifi/failed_tests.txt)"
    EXCLUDED_TEST_CASES="${CRASHED_TEST_CASES}:${FAILED_TEST_CASES}"
    GTEST_FILTER="*-${EXCLUDED_TEST_CASES}"
  fi
  # TODO: reenable this when we can use onnxSetIOAndRunGraph for node tests.
  # GTEST_FILTER=$GTEST_FILTER "${ONNX_BUILD_DIR}/onnxifi_test_driver_gtests" "${ONNX_TESTDATA_DIR}"
}

check_glow_build_dir_exist
copy_glow_lib
run_node_tests
