#!/bin/bash

# This script runs all tests in glow, including onnxifi gtests

set -euxo pipefail

export GLOW_SRC=$PWD
export GLOW_BUILD_DIR=${GLOW_SRC}/build
export LOADER=${GLOW_BUILD_DIR}/bin/image-classifier
export LSAN_OPTIONS="suppressions=$GLOW_SRC/.circleci/lsan_suppressions.txt"
export ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer
export IMAGES_DIR=${GLOW_SRC}/tests/images/

# Pass in which tests to run (one of {test, test_unopt}).
run_unit_tests() {
    CTEST_PARALLEL_LEVEL=4 GLOG_minloglevel=3 ninja "${1}" || ( cat Testing/Temporary/LastTest.log && exit 1 )
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

run_and_check_bundle_instrument() {
    cd "${GLOW_BUILD_DIR}/bundles/bundle_instrument/"
    # Compare console output.
    ./BundleInstrument ${IMAGES_DIR}/mnist/0_1009.png >> raw_results.txt
    diff raw_results.txt "${GLOW_SRC}/.ci/bundle_instrument_expected_output.txt"
    # Compare binary dumps between instrument-debug and instrument-ir.
    for file in ./instrument-debug-data/*.bin
    do
      file_name=$(basename $file)
      diff ./instrument-debug-data/${file_name} ./instrument-ir-data/${file_name}
    done
    cd -
}

run_and_check_bundle_with_multiple_entries() {
    cd "${GLOW_BUILD_DIR}/bundles/bundle_with_multiple_entries/"
    # Compare console output.
    ./bundle_with_multiple_entries >> raw_results.txt
    diff raw_results.txt "${GLOW_SRC}/.ci/bundle_with_multiple_entries_expected_output.txt"
    cd -
}

run_and_check_bundle_with_extra_objects() {
    cd "${GLOW_BUILD_DIR}/bundles/bundle_with_extra_objects/"
    # Compare console output.
    ./BundleWithExtraObjects >> raw_results.txt
    diff raw_results.txt "${GLOW_SRC}/.ci/bundle_with_extra_objects_expected_output.txt"
    cd -
}

run_and_check_bundle_tflite_custom() {
    cd "${GLOW_BUILD_DIR}/bundles/bundle_tflite_custom/"
    # Compare console output.
    ./BundleTFLiteCustom >> raw_results.txt
    diff raw_results.txt "${GLOW_SRC}/.ci/bundle_tflite_custom_expected_output.txt"
    cd -
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
    FEATURE_COMPILATION)
        # FEATURE_COMPILATION is a compilation only CI job, thus tests
        # are not requited.
        ;;
    DEBUG)
        run_unit_tests check
        run_unit_tests test_unopt
        ;;
    SHARED)
        # No tests with shared libs; it's similar to DEBUG.
        ;;
    32B_DIM_T)
        # A lot of 32b dim_t issues are not revealed at build time, thus
        # run the unit test suite also.
        run_unit_tests check
        ;;
    COVERAGE)
        cd "${GLOW_SRC}"
        cd build
        ../.circleci/run_coverage.sh
        ;;
    CHECK_CLANG_AND_PEP8_FORMAT)
        cd "${GLOW_SRC}"
        sudo ln -s /usr/bin/clang-format-11 /usr/bin/clang-format
        source /tmp/venv/bin/activate
        ./utils/format.sh check
        ;;
    *)
        echo "Error, '${CIRCLE_JOB}' not valid mode; Please, check .circleci/test.sh for list of supported tests."
        exit 1
        ;;
esac
