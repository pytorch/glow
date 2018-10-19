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

option(GLOW_USE_COVERAGE "Define whether coverage report needs to be generated" OFF)

if(GLOW_USE_COVERAGE)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Code coverage must be run in debug mode, otherwise it might be misleading.")
  endif()

  find_program(GCOV_PATH gcov)
  if(NOT GCOV_PATH)
    message(FATAL_ERROR "Make sure gcov is installed.")
  endif()

  find_program(LCOV_PATH NAMES lcov lcov.bat lcov.exe lcov.perl)
  if(NOT LCOV_PATH)
    message(FATAL_ERROR "Make sure lcov is installed.")
  endif()

  find_program(GENHTML_PATH NAMES genhtml genhtml.perl genhtml.bat)
  if(NOT GENHTML_PATH)
    message(FATAL_ERROR "Make sure genhtml is installed.")
  endif()

  # Add compilation flags for coverage.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")

  # Add glow_coverage target.
  add_custom_target(glow_coverage
    # Cleanup lcov counters.
    COMMAND ${LCOV_PATH} --directory . --zerocounters
    COMMAND echo "Cleaning is done. Running tests"

    # Run all tests.
    COMMAND ctest -j 4

    # Capture lcov counters based on the test run.
    COMMAND ${LCOV_PATH} --no-checksum --directory . --capture --output-file glow_coverage.info

    # Ignore not related files.
    COMMAND ${LCOV_PATH} --remove glow_coverage.info '*v1*' '/usr/*' '*tests/*' '*llvm_install*' --output-file ${PROJECT_BINARY_DIR}/glow_coverage_result.info

    # Generate HTML report based on the profiles.
    COMMAND ${GENHTML_PATH} -o glow_coverage ${PROJECT_BINARY_DIR}/glow_coverage_result.info

    # Cleanup info files.
    COMMAND ${CMAKE_COMMAND} -E remove glow_coverage.info ${PROJECT_BINARY_DIR}/glow_coverage_result.info

    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )

  add_custom_command(TARGET glow_coverage POST_BUILD
    COMMAND ;
    COMMENT "Coverage report: ./glow_coverage/index.html. Open it in your browser."
  )
endif()

