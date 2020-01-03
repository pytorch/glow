# Smoke tests the OpenCL backend's disk based kernel program cache by
# executing the OCLTest twice with the disk caching on.

set(OPENCL_CACHE_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencl-cached-binaries)

file(REMOVE_RECURSE ${OPENCL_CACHE_DIR})

function(RUN_OCLTEST)
  execute_process(COMMAND "${CMAKE_CURRENT_BINARY_DIR}/tests/OCLTest" "--gtest_output=xml:OCLTestCached.xml" "--opencl-program-cache-dir=${OPENCL_CACHE_DIR}"
    RESULT_VARIABLE CMD_RES
    ERROR_QUIET
    OUTPUT_QUIET)
endfunction(RUN_OCLTEST)

run_ocltest()
if(CMD_RES)
  message(FATAL_ERROR "Error in the OpenCL cold cache test run.")
endif()

run_ocltest()
if(CMD_RES)
  message(FATAL_ERROR "Error in the warm cache OpenCL test run.")
endif()

file(REMOVE_RECURSE ${OPENCL_CACHE_DIR})
