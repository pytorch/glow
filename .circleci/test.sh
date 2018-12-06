#!/bin/bash

# This script runs all tests in glow, including onnxifi gtests

set -euxo pipefail

export GLOW_SRC=$PWD
export GLOW_BUILD_DIR=${GLOW_SRC}/build
export LOADER=${GLOW_BUILD_DIR}/bin/image-classifier

# Pass in which tests to run (one of {test, test_unopt}).
run_unit_tests() {
    CTEST_PARALLEL_LEVEL=4 ninja "${1}" || ( cat Testing/Temporary/LastTest.log && exit 1 )
}

# Pass one of {YES, NO} for QUANTIZE.
run_and_check_bundle() {
    echo "Checking lenet_mnist bundle with QUANTIZE=${1}"
    cd "${GLOW_SRC}/examples/bundles/lenet_mnist/"
    ( QUANTIZE=${1} make &> raw_results.txt ) || ( cat raw_results.txt && exit 1 )
    ( tail -n72 raw_results.txt | grep -F "Result: " > results.txt ) || ( cat raw_results.txt && exit 1 )
    diff results.txt "${GLOW_SRC}/.ci/lenet_mnist_expected_output.txt"
    rm results.txt raw_results.txt
    echo "Successfully completed checking lenet_mnist bundle with QUANTIZE=${1}"
}

run_onnxifi() {
    # Run ONNX test
    ONNX_DIR="${GLOW_SRC}/thirdparty/onnx"
    # ONNX test data dir
    TESTDATA_DIR="${ONNX_DIR}/onnx/backend/test/data/node"

    # Banned known buggy test cases from gtest
    CRASHED_TEST_CASES="$(paste -sd: "${GLOW_SRC}"/.circleci/onnxifi_driver_test/crashed.txt)"
    FAILED_TEST_CASES="$(paste -sd: "${GLOW_SRC}"/.circleci/onnxifi_driver_test/failed.txt)"
    EXCLUDED_TEST_CASES="${CRASHED_TEST_CASES}:${FAILED_TEST_CASES}"

    # Setup glow onnxifi backend so test driver can load it
    cp "${GLOW_BUILD_DIR}/lib/Onnxifi/libonnxifi-glow.so" "${GLOW_SRC}/libonnxifi.so"
    export LD_LIBRARY_PATH=${GLOW_SRC}

    # Run Onnxifi gtest
    GTEST_FILTER="*-${EXCLUDED_TEST_CASES}" "${GLOW_BUILD_DIR}/onnxifi_test_driver_gtests" "${TESTDATA_DIR}"
}

# Run unit tests and bundle tests.
cd "${GLOW_BUILD_DIR}"
case ${CIRCLE_JOB} in
    ASAN)
        # ASAN is not enabled in onnx, therefore we should skip it for now.
        # TODO: Enable ASAN test.
        run_unit_tests test
        ;;

    DEBUG)
        run_unit_tests test
        run_unit_tests test_unopt
        run_and_check_bundle YES
        run_and_check_bundle NO
        run_onnxifi
        ;;

    SHARED)
        # No tests with shared libs; it's similar to DEBUG.
        ;;

    RELEASE_WITH_LIT)
        run_unit_tests check
        ;;

    *)
        echo "Error, '${CIRCLE_JOB}' not valid mode; Must be one of {ASAN, DEBUG, SHARED, RELEASE_WITH_LIT}."
        exit 1
        ;;
esac
