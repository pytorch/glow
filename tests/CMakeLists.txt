set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/tests/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/unittests)

add_subdirectory(Testing)
add_subdirectory(unittests)
add_subdirectory(images)
add_subdirectory(benchmark)
add_subdirectory(models)
add_subdirectory(runtime_test)
add_subdirectory(stress)

# External backends
ExternalBackendsTest()

# Function which creates a new OutputCheck test that runs test_script.
function(add_output_check_test)
  set(oneValueArgs NAME)
  set(multiValueArgs COMMAND REQUIRES)
  cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARG_NAME)
    message(FATAL_ERROR "Name mandatory")
  endif()

  if (NOT ARG_COMMAND)
    message(FATAL_ERROR "Command mandatory")
  endif()

  # Skip test if it requires CPU and DGLOW_WITH_CPU is OFF.
  if ("CPU" IN_LIST ARG_REQUIRES)
    if (NOT GLOW_WITH_CPU)
      message("Skipping adding test ${ARG_NAME} because it requires CPU backend. Configure with -DGLOW_WITH_CPU.")
      return()
    endif()
  endif()

  # Skip test if it requires downloaded models that aren't provided.
  if ("MODELS" IN_LIST ARG_REQUIRES)
    if(NOT EXISTS "${GLOW_MODELS_DIR}")
      message("Skipping adding test ${ARG_NAME} because it requires a models directory. Configure with -DGLOW_MODELS_DIR.")
      return()
    endif()
  endif()

  # Skip test if it requires RELEASE and was compiled with a different build type.
  if ("RELEASE" IN_LIST ARG_REQUIRES)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
      message("Skipping adding test ${ARG_NAME} because it requires a Release build. Configure with -DCMAKE_BUILD_TYPE=Release.")
      return()
    endif()
  endif()

  set (OUTPUTCHECK_PATH "${GLOW_SOURCE_DIR}/tests/OutputCheck/bin/OutputCheck")
  set (OUTPUTCHECK_RUNNER_PATH "${GLOW_SOURCE_DIR}/tests/run_outputcheck.sh")

  # Command to setup test environment variables.
  set(SETUP_OUTPUTCHECK_ENV ${CMAKE_COMMAND} -E env
    MODELS_DIR=${GLOW_MODELS_DIR}
    OUTPUTCHECK=${OUTPUTCHECK_PATH}
    BIN=${GLOW_OUTPUT_DIR}
    IMAGES_DIR=${CMAKE_BINARY_DIR}/tests/images)

  set(TEST_COMMAND ${SETUP_OUTPUTCHECK_ENV} ${OUTPUTCHECK_RUNNER_PATH} ${ARG_COMMAND})

  # Add expensive Glow test.
  add_glow_test(
    EXPENSIVE
    NAME ${ARG_NAME}
    COMMAND ${TEST_COMMAND}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS text-translator resnet-runtime)

  # Append test command to OUTPUTCHECK_TESTS so that it will be run with the output_check_tests target.
  LIST(APPEND OUTPUTCHECK_TESTS ${TEST_COMMAND})

  # Propogate changes to OUTPUTCHECK_TESTS outside the function.
  set(OUTPUTCHECK_TESTS ${OUTPUTCHECK_TESTS} PARENT_SCOPE)
endfunction()

set (TEXT_TRANSLATOR_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/text-translator")
set (EXAMPLES_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples_tests")

# Create en2gr_cpu_test OutputCheck test.
add_output_check_test(NAME en2gr_cpu_test
                      COMMAND ${TEXT_TRANSLATOR_TESTS_DIR}/en2gr_cpu_test.sh
                      REQUIRES MODELS CPU RELEASE)

# Create en2gr_quantization_test OutputCheck test.
add_output_check_test(NAME en2gr_quantization_test
                      COMMAND ${TEXT_TRANSLATOR_TESTS_DIR}/en2gr_quantization_test.sh
                      REQUIRES MODELS RELEASE)

# Create en2gr_cpu_partition_test OutputCheck test.
add_output_check_test(NAME en2gr_cpu_partition_test
                      COMMAND ${TEXT_TRANSLATOR_TESTS_DIR}/en2gr_cpu_partition_test.sh
                      REQUIRES MODELS RELEASE)

# Create en2gr_cpu_config_test OutputCheck test.
add_output_check_test(NAME en2gr_cpu_config_test
                      COMMAND ${TEXT_TRANSLATOR_TESTS_DIR}/en2gr_cpu_config_test.sh
                      REQUIRES MODELS RELEASE)

# Create an OutputCheck test to run resnet-runtime-test.
add_output_check_test(NAME resnet_runtime_test
                      COMMAND ${EXAMPLES_TESTS_DIR}/resnet-runtime-test.sh
                      REQUIRES MODELS)

# Create target for OUTPUTCHECK_TESTS.
LIST(APPEND OUTPUTCHECK_TESTS true)
add_custom_target(output_check_tests
  COMMAND ${OUTPUTCHECK_TESTS}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS text-translator resnet-runtime resnet-concurrent)
