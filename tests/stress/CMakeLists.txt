set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/tests/stress/)

# Loop through all backends present in lib/Backends (see FindBackends.cmake)
# and instantiate a backend-parameterized test for each of them.
foreach(backend ${GLOW_BACKENDS})
  add_backend_test(TEST ParameterSweepTest BACKEND "${backend}" UNOPT)
  add_backend_test(TEST SparseLengthsSumTest BACKEND "${backend}" UNOPT)
endforeach()
