#!/bin/bash

# This script runs all tests in glow, including onnxifi gtests

set -euxo pipefail

export GLOW_SRC=$PWD
export GLOW_BUILD_DIR=${GLOW_SRC}/build
export LOADER=${GLOW_BUILD_DIR}/bin/image-classifier
export LSAN_OPTIONS="suppressions=$GLOW_SRC/.circleci/suppressions.txt"
export ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer
export IMAGES_DIR=${GLOW_SRC}/tests/images/

# Pass in which tests to run (one of {test, test_unopt}).
run_unit_tests() {
    CTEST_PARALLEL_LEVEL=4 ninja "${1}" || ( cat Testing/Temporary/LastTest.log && exit 1 )
}

run_and_check_lenet_mnist_bundle() {
    for q in "" "quantized_"
    do
      cd "${GLOW_BUILD_DIR}/bundles/${q}lenet_mnist/"
      rm -f raw_results.txt
      for f in ${IMAGES_DIR}/mnist/*
      do
        # Assume that there is only one file with this format (prepended with Quantized or not)
        ./*LeNetMnistBundle ${f} | grep "Result: " >> raw_results.txt
      done
      diff raw_results.txt "${GLOW_SRC}/.ci/lenet_mnist_expected_output.txt"
      cd -
    done
}

run_and_check_resnet50_bundle() {
    for q in "" "quantized_"
    do
      cd "${GLOW_BUILD_DIR}/bundles/${q}resnet50/"
      rm -f raw_results.txt
      for f in ${IMAGES_DIR}/imagenet/*
      do
        # Assume that there is only one file with this format (prepended with Quantized or not)
        ./*ResNet50Bundle ${f} | grep "Result: " >> raw_results.txt
      done
      diff raw_results.txt "${GLOW_SRC}/.ci/resnet50_expected_output.txt"
      cd -
    done
}

# Run unit tests and bundle tests.
cd "${GLOW_BUILD_DIR}"
case ${CIRCLE_JOB} in
    ASAN)
        run_unit_tests check
        ;;
    OPENCL)
        run_unit_tests check
        ;;
    TSAN)
        # Run only Glow tests.
        run_unit_tests check
        ;;
    DEBUG)
        run_unit_tests check
        run_unit_tests test_unopt
        ;;
    SHARED)
        # No tests with shared libs; it's similar to DEBUG.
        ;;
    RELEASE_WITH_EXPENSIVE_TESTS)
        run_unit_tests check_expensive
        run_and_check_lenet_mnist_bundle
        run_and_check_resnet50_bundle
        ;;
    COVERAGE)
        cd "${GLOW_SRC}"
        cd build
        ../.circleci/run_coverage.sh
        ;;
    CHECK_CLANG_FORMAT)
        cd "${GLOW_SRC}"
        sudo ln -s /usr/bin/clang-format-7 /usr/bin/clang-format
        ./utils/format.sh check
        ;;
    *)
        echo "Error, '${CIRCLE_JOB}' not valid mode; Please, check .circleci/test.sh for list of supported tests."
        exit 1
        ;;
esac
