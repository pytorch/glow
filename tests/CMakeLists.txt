set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/tests/)
add_subdirectory(Testing)
add_subdirectory(unittests)
add_subdirectory(images)
add_subdirectory(benchmark)
add_subdirectory(models)

# Function which creates a new OutputCheck test that runs test_script.
function(add_output_check_test test_name test_script)
  set (OUTPUTCHECK_PATH "${GLOW_SOURCE_DIR}/tests/OutputCheck/bin/OutputCheck")
  set (OUTPUTCHECK_RUNNER_PATH "${GLOW_SOURCE_DIR}/tests/run_outputcheck.sh")

  # Command to setup test environment variables.
  set(SETUP_OUTPUTCHECK_ENV ${CMAKE_COMMAND} -E env
    MODELS_DIR=${GLOW_MODELS_DIR}
    OUTPUTCHECK=${OUTPUTCHECK_PATH}
    TEXT_TRANSLATOR=${GLOW_OUTPUT_DIR}/text-translator)

  set(TEST_COMMAND ${SETUP_OUTPUTCHECK_ENV} ${OUTPUTCHECK_RUNNER_PATH} ${test_script})

  # Add Glow test.
  add_glow_test(
    NAME ${test_name}
    COMMAND ${TEST_COMMAND}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS text-translator)

  # Append test command to OUTPUTCHECK_TESTS so that it will be run with the output_check_tests target.
  LIST(APPEND OUTPUTCHECK_TESTS ${TEST_COMMAND})

  # Propogate changes to OUTPUTCHECK_TESTS outside the function.
  set(OUTPUTCHECK_TESTS ${OUTPUTCHECK_TESTS} PARENT_SCOPE)
endfunction()

if (GLOW_BUILD_OUTPUTCHECK_TESTS)
  if(NOT EXISTS "${GLOW_MODELS_DIR}")
    message(FATAL_ERROR "Must provide a valid models directory with -DGLOW_MODELS_DIR.")
  endif()

  # TODO not all tests may require CPU. We should make this requirement more
  # flexible in the future.
  if (NOT GLOW_WITH_CPU)
    message(FATAL_ERROR "Cannot run OutputCheck tests without CPU. Configure with -DGLOW_WITH_CPU.")
  endif()

  if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "Some tests run with OutputCheck are extremely slow when not built with Release. To build with release use -DCMAKE_BUILD_TYPE=Release")
  endif()

  set (TEXT_TRANSLATOR_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/text-translator")

  add_output_check_test(en2gr_cpu_test
                        ${TEXT_TRANSLATOR_TESTS_DIR}/en2gr_cpu_test.sh)
  add_output_check_test(en2gr_quantization_test
                        ${TEXT_TRANSLATOR_TESTS_DIR}/en2gr_quantization_test.sh)

  # Create target.
  LIST(APPEND OUTPUTCHECK_TESTS true)
  add_custom_target(output_check_tests
    COMMAND ${OUTPUTCHECK_TESTS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS text-translator)
endif ()
