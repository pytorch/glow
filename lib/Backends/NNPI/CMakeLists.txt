# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

cmake_minimum_required (VERSION 3.3)

if(DEFINED NNPI_COLLECT_MEM_USAGE)
    set (NNPI_MEM_PROFILING ${NNPI_COLLECT_MEM_USAGE})
else()
    set (NNPI_MEM_PROFILING 0)
endif()
add_definitions("-DNNPI_COLLECT_MEM_USAGE=${NNPI_MEM_PROFILING}")

if (DEFINED NNPI_LIB_DIR)
    set(NNPI_LIB_SEARCH_PATH ${NNPI_LIB_DIR})
else()
    set(NNPI_LIB_SEARCH_PATH "/opt/intel_nnpi/lib/")
endif()

if (DEFINED NNPI_INF_LIB_DIR)
    set(NNPI_INF_LIB_SEARCH_PATH ${NNPI_INF_LIB_DIR})
else()
    set(NNPI_INF_LIB_SEARCH_PATH "/opt/intel_nnpi/lib/")
endif()

if (DEFINED NNPI_MG_LIB_DIR)
    set(NNPI_MG_LIB_SEARCH_PATH ${NNPI_MG_LIB_DIR})
else()
    set(NNPI_MG_LIB_SEARCH_PATH "/opt/intel_nnpi/lib/")
endif()


if (DEFINED NNPI_API_DIR)
    set(NNPI_API_SEARCH_PATH ${NNPI_API_DIR})
else()
    set(NNPI_API_SEARCH_PATH "/opt/intel_nnpi/include/")
endif()

if (DEFINED NNPI_INF_API_DIR)
    set(NNPI_INF_API_SEARCH_PATH ${NNPI_INF_API_DIR})
else()
    set(NNPI_INF_API_SEARCH_PATH "/opt/intel_nnpi/include/")
endif()

if (DEFINED NNPI_MG_API_DIR)
    set(NNPI_MG_SEARCH_PATH ${NNPI_MG_API_DIR})
else()
    set(NNPI_MG_SEARCH_PATH "/opt/intel_nnpi/include/")
endif()

find_library(NNPI_TRANSFORMER_LIB nnpi_transformer ${NNPI_LIB_SEARCH_PATH})
if(NOT NNPI_TRANSFORMER_LIB)
  message(FATAL_ERROR "nnpi_transformer library not found at ${NNPI_LIB_SEARCH_PATH}")
endif()

find_library(NNPI_INFERENCE_LIB nnpi_inference ${NNPI_INF_LIB_SEARCH_PATH})
if(NOT NNPI_INFERENCE_LIB)
  message(FATAL_ERROR "nnpi_inference library not found at ${NNPI_INF_LIB_SEARCH_PATH}")
endif()

find_path(NNPI_TRANSFORMER_API nnpi_transformer.h ${NNPI_API_SEARCH_PATH})
if(NOT NNPI_TRANSFORMER_API)
  message(FATAL_ERROR "nnpi_transformer include files not found at ${NNPI_API_DIR}")
endif()

find_path(NNPI_INFERENCE_API nnpi_inference.h ${NNPI_INF_API_SEARCH_PATH})
if(NOT NNPI_INFERENCE_API)
  message(FATAL_ERROR "nnpi_inference include files  not found at ${NNPI_INF_LIB_DIR}")
endif()

find_path(NNPI_MG_API nnpiml.h ${NNPI_MG_SEARCH_PATH})
if(NOT NNPI_MG_API)
  message(FATAL_ERROR "nnpiml include files not found at ${NNPI_MG_API_DIR}")
endif()

find_library(NNPI_MG_LIB nnpiml ${NNPI_MG_LIB_SEARCH_PATH})
if(NOT NNPI_MG_LIB)
  message(FATAL_ERROR "nnpiml library not found at ${NNPI_MG_LIB_SEARCH_PATH}")
endif()

message(STATUS "[NNPI] NNPI_API_DIR                = ${NNPI_API_DIR}")
message(STATUS "[NNPI] NNPI_LIB_DIR                = ${NNPI_TRANSFORMER_LIB}")

message(STATUS "[NNPI] NNPI_INF_API_DIR            = ${NNPI_INF_API_DIR}")
message(STATUS "[NNPI] NNPI_INF_LIB_DIR            = ${NNPI_INFERENCE_LIB}")
message(STATUS "[NNPI] NNPI_MG_API_DIR             = ${NNPI_MG_API}")
message(STATUS "[NNPI] NNPI_MG_LIB_DIR             = ${NNPI_MG_LIB}")
message(STATUS "[NNPI] GLOW_BINARY_DIR             = ${GLOW_BINARY_DIR}")
message(STATUS "[NNPI] NNPI_COLLECT_MEM_USAGE      = ${NNPI_MEM_PROFILING}")

if (UNIX)
  EXEC_PROGRAM(cat ARGS "/proc/cpuinfo" OUTPUT_VARIABLE CPUINFO)
  STRING(REGEX REPLACE "^.*(avx512f).*$" "\\1" AVX512F_FLAG ${CPUINFO})
  if("avx512f" STREQUAL "${AVX512F_FLAG}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512vl -mavx512bw")
    add_definitions(-DUSE_AVX)
  endif()
endif()
add_subdirectory(ClassGen)

include_directories(
    ${NNPI_TRANSFORMER_API}
    ${NNPI_INFERENCE_API}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NNPI_MG_API}
)

add_library(NNPI
    NNPI.cpp
    NNPICompiledFunction.cpp
    Importer.cpp
    InferencePool.cpp
    NNPIDeviceManager.cpp
    NNPIFactory.cpp
    BlockStream.cpp
    NNPIOptions.cpp
    NNPITracing.cpp
    NNPIMLTraceWrapper.cpp
)

target_link_libraries(NNPI
                      PRIVATE
                      Backend
                      Base
                      Graph
                      GraphOptimizer
                      CodeGen
                      IR
                      Support
                      ${NNPI_TRANSFORMER_LIB}
                      ${NNPI_INFERENCE_LIB}
                      ${NNPI_MG_LIB}
)

set(linked_backends ${linked_backends} NNPI PARENT_SCOPE)

