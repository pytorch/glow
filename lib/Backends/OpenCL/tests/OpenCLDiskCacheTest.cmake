# Smoke tests the OpenCL backend's disk based kernel program cache by
# executing the OCLTest twice with the disk caching on.

set(OPENCL_CACHE_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencl-cached-binaries)

file(REMOVE_RECURSE ${OPENCL_CACHE_DIR})

function(RUN_OCLTEST)
  execute_process(COMMAND "${GLOW_BINARY_DIR}/tests/OpenCLBackendCorrectnessTest" "--gtest_output=xml:OpenCLBackendCorrectnessTestCached.xml" "--opencl-program-cache-dir=${OPENCL_CACHE_DIR}")
  set(RES ${CMD_RES} PARENT_SCOPE)
endfunction(RUN_OCLTEST)

run_ocltest()
if(RES)
  message(FATAL_ERROR "Error in the OpenCL cold cache test run: ${RES}")
endif()

run_ocltest()
if(RES)
  message(FATAL_ERROR "Error in the warm cache OpenCL test run: ${RES}")
endif()

file(REMOVE_RECURSE ${OPENCL_CACHE_DIR})
