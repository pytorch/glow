# Copyright (c) 2017-present, Facebook, Inc.
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
  set(oneValueArgs NAME)
  set(multiValueArgs COMMAND DEPENDS)
  cmake_parse_arguments(ARG "" "${oneValueArgs}"
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
endfunction()
