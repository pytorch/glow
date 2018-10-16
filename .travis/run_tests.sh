#!/bin/bash

set -euo pipefail

: ${1?"Usage: \`$0 TEST_NAME\` (TEST_NAME must be one of {ASAN, DEBUG})"}
TEST_NAME=${1}

# Note: keep_alive() prevents travis from exiting early on tests that take a while.
keep_alive() { while true; do echo -en "\a"; sleep 540; done }

# Pass in which tests to run (one of {test, test_unopt}).
run_unit_tests() {
    CTEST_PARALLEL_LEVEL=2 ninja ${1} || ( cat Testing/Temporary/LastTest.log && exit 1 )
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

keep_alive &

case ${TEST_NAME} in
ASAN)
    ninja all
    run_unit_tests test
    ;;

DEBUG)
    ninja all
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
