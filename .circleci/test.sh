#!/bin/bash

# This script runs all tests in glow, including onnxifi gtests


set -euxo pipefail

GLOW_DIR=$PWD
GLOW_BUILD_DIR=${GLOW_DIR}/build
cd ${GLOW_BUILD_DIR}

TEST_NAME=$CIRCLE_JOB

# Pass in which tests to run (one of {test, test_unopt}).
run_unit_tests() {
    CTEST_PARALLEL_LEVEL=4 ninja ${1} || ( cat Testing/Temporary/LastTest.log && exit 1 )
}

# Pass one of {YES, NO} for QUANTIZE.
run_and_check_bundle() {
    echo "Checking lenet_mnist bundle with QUANTIZE=${1}"
    cd ${GLOW_SRC}/examples/bundles/lenet_mnist/
    ( QUANTIZE=${1} make &> raw_results.txt ) || ( cat raw_results.txt && exit 1 )
    ( tail -n72 raw_results.txt | grep -F "Result: " > results.txt ) || ( cat raw_results.txt && exit 1 )
    diff results.txt ${GLOW_SRC}/.ci/lenet_mnist_expected_output.txt
    rm results.txt raw_results.txt
    echo "Successfully completed checking lenet_mnist bundle with QUANTIZE=${1}"
}

export GLOW_SRC=${PWD}/..
export LOADER=${GLOW_SRC}/build/bin/image-classifier

case ${TEST_NAME} in
    ASAN)
        run_unit_tests test
        ;;

    DEBUG)
        run_unit_tests test
        run_unit_tests test_unopt
        run_and_check_bundle YES
        run_and_check_bundle NO
        ;;

    *)
        echo "Error, '${TEST_NAME}' not valid mode; Must be one of {ASAN, DEBUG}."
        exit 1
        ;;
esac


# Run ONNX test
ONNX_DIR="${GLOW_DIR}/thirdparty/onnx"
# ONNX test data dir
TESTDATA_DIR="${ONNX_DIR}/onnx/backend/test/data/node"

# Asan is not enbaled in onnx, therefore we should skip it for now.
# TODO: Enable asan test. Rui Zhu.
if [[ "$CIRCLE_JOB" != ASAN ]]; then
    # Banned known buggy test cases from gtest
    CRASHED_TEST_CASES="*test_softmax_axis_0*:*test_batchnorm_epsilon*:*test_batchnorm_example*:*test_sum_example*:*test_flatten_axis0*:*test_transpose_default*:*test_sum_one_input*"
    FAILED_TEST_CASES="*test_averagepool_1d_default*:*test_average_2d_precomputed_same_upper*:*test_reshape_reduced_dims*:*test_maxpool_with_argmax_2d_precomputed_pads*:*test_reshape_negative_dim*:*test_maxpool_3d_default*:*test_maxpool_2d_same_upper*:*test_averagepool_2d_precomputed_same_upper*:*test_reshape_extended_dims*:*test_averagepool_2d_same_upper*:*test_gemm_broadcast*:*test_gemm_broadcast*:*test_reshape_one_dim*:*test_averagepool_2d_pads*:*test_maxpool_1d_default*:*test_maxpool_2d_same_lower*:*test_gemm_nobroadcast*:*test_maxpool_2d_precomputed_same_upper*:*test_maxpool_with_argmax_2d_precomputed_strides*:*test_averagepool_2d_precomputed_pads*:*test_reshape_reordered_dims*:*test_averagepool_2d_same_lower*:*test_averagepool_3d_default*"
    EXCLUDED_TEST_CASES=${CRASHED_TEST_CASES}":"${FAILED_TEST_CASES}

    # Setup glow onnxifi backend so test driver can load it
    cp ${GLOW_BUILD_DIR}/lib/Onnxifi/libonnxifi-glow.so ${GLOW_DIR}/libonnxifi.so
    export LD_LIBRARY_PATH=${GLOW_DIR}

    # Run Onnxifi gtest
    GTEST_FILTER=*-${EXCLUDED_TEST_CASES} ${GLOW_BUILD_DIR}/onnxifi_test_driver_gtests ${TESTDATA_DIR}
fi
