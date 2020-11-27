# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A function to add a test to be driven through the 'check' target.
# Unlike the 'test' target, the 'check' target rebuilds the executables
# before invoking the tests.
function(add_glow_test)
  set(options OPTIONAL EXPENSIVE STRESS)
  set(oneValueArgs NAME)
  set(multiValueArgs COMMAND DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

  if (NOT ARG_NAME)
    list(GET ARG_UNPARSED_ARGUMENTS 0 ARG_NAME)
    list(REMOVE_AT ARG_UNPARSED_ARGUMENTS 0)
  endif()

  if (NOT ARG_NAME)
    message(FATAL_ERROR "Name mandatory")
  endif()

  if (NOT ARG_COMMAND)
    set(ARG_COMMAND ${ARG_UNPARSED_ARGUMENTS})
  endif()

  if (NOT ARG_COMMAND)
    message(FATAL_ERROR "Command mandatory")
  endif()

  list(GET ARG_COMMAND 0 TEST_EXEC)
  list(APPEND ARG_DEPENDS ${TEST_EXEC})

  set_property(GLOBAL APPEND PROPERTY GLOW_TEST_DEPENDS ${ARG_DEPENDS})

  # Produce the specific test rule using the default built-in.
  add_test(NAME ${ARG_NAME} COMMAND ${ARG_COMMAND})

  # If the EXPENSIVE argument is passed, add the EXPENSIVE label to the test
  # so that it will only be run with the other expensive tests with the
  # ninja check_expensive command
  if (ARG_EXPENSIVE)
    set_property(TEST ${ARG_NAME} PROPERTY LABELS EXPENSIVE)
  endif()
  # If the STRESS argument is passed, add the STRESS label to the test so that
  # it will only be run with the other expensive and stress tests with the ninja
  # check_expensive_stress command
  if (ARG_STRESS)
    set_property(TEST ${ARG_NAME} PROPERTY LABELS STRESS)
  endif()
endfunction()

# Adds a backend-parameterized test.  These tests can be instantiated for any
# backend present in lib/Backends, and allow similar functionality to easily be
# tested across all defined backends.
function(add_backend_test)
  set(options UNOPT EXPENSIVE STRESS)
  set(oneValueArgs BACKEND TEST)
  set(multiValueArgs PRIVATE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"  ${ARGN})
  set(test "${ARG_BACKEND}${ARG_TEST}")
  set(SOURCES "${ARG_TEST}.cpp")
  set(BLACKLIST_SOURCE "${GLOW_BACKENDS_DIR}/${ARG_BACKEND}/tests/${test}.cpp")
  if(NOT EXISTS ${BLACKLIST_SOURCE})
    return()
  endif()
  if (DEFINED "${ARG_BACKEND}_MAJOR_VERSION")
    add_definitions(-D${ARG_BACKEND}_MAJOR_VERSION=${${ARG_BACKEND}_MAJOR_VERSION})
  endif()
  if (DEFINED "${ARG_BACKEND}_MINOR_VERSION")
    add_definitions(-D${ARG_BACKEND}_MINOR_VERSION=${${ARG_BACKEND}_MINOR_VERSION})
  endif()
  list(APPEND SOURCES ${BLACKLIST_SOURCE})
  add_executable("${test}" ${SOURCES})
  target_compile_definitions("${test}" PRIVATE -DGLOW_TEST_BACKEND=${ARG_BACKEND})
  target_link_libraries("${test}"
    PRIVATE BackendTestUtils TestMain ${ARG_PRIVATE})
  if(${ARG_EXPENSIVE})
    set(ARG_EXPENSIVE EXPENSIVE)
  else()
    set(ARG_EXPENSIVE)
  endif()
  if(${ARG_STRESS})
    set(ARG_STRESS STRESS)
    set(STRESS_PREFIX "stress/")
  else()
    set(ARG_STRESS)
    set(STRESS_PREFIX)
  endif()
  add_glow_test("${test}" ${ARG_EXPENSIVE} ${ARG_STRESS} ${GLOW_BINARY_DIR}/tests/${STRESS_PREFIX}${test} --gtest_output=xml:${test}.xml )
  if(${ARG_UNOPT})
    list(APPEND UNOPT_TESTS ./tests/${STRESS_PREFIX}${test} -optimize-ir=false &&)
  endif()
endfunction()
