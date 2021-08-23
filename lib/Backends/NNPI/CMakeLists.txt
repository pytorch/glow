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
if (NNPI_USE_STATIC_TRANSFORMER)
  find_library(NNPI_TRANSFORMER_LIB nnpi_transformer_static ${NNPI_LIB_SEARCH_PATH})
  if(NOT NNPI_TRANSFORMER_LIB)
    message(FATAL_ERROR "nnpi_transformer_static library not found at ${NNPI_LIB_SEARCH_PATH}")
  endif()
else()
  find_library(NNPI_TRANSFORMER_LIB nnpi_transformer ${NNPI_LIB_SEARCH_PATH})
  if(NOT NNPI_TRANSFORMER_LIB)
    message(FATAL_ERROR "nnpi_transformer library not found at ${NNPI_LIB_SEARCH_PATH}")
  endif()
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

find_path(NNPI_MG_API nnpi_ice_caps.h ${NNPI_MG_SEARCH_PATH})
if(NOT NNPI_MG_API)
  message(FATAL_ERROR "nnpi_ice_caps include files not found at ${NNPI_MG_API_DIR}")
endif()

find_library(NNPI_MG_LIB nnpi_icecaps ${NNPI_MG_LIB_SEARCH_PATH})
if(NOT NNPI_MG_LIB)
  message(FATAL_ERROR "nnpi_icecaps library not found at ${NNPI_MG_LIB_SEARCH_PATH}")
endif()

file (STRINGS "${NNPI_TRANSFORMER_LIB}" verString REGEX "DL VERSION: ([0-9]+.[0-9]+)" LIMIT_COUNT 1)
if (NOT DEFINED NNPI_MAJOR_VERSION)
  string(REGEX MATCH "DL VERSION: ([0-9]+).[0-9]+.[0-9]+.[0-9]+" _ ${verString})
  set(NNPI_MAJOR_VERSION ${CMAKE_MATCH_1})
endif()
if (NOT DEFINED NNPI_MINOR_VERSION)
  string(REGEX MATCH "DL VERSION: [0-9]+.([0-9]+).[0-9]+.[0-9]+" _ ${verString})
  set(NNPI_MINOR_VERSION ${CMAKE_MATCH_1})
endif()

file (STRINGS "${NNPI_INFERENCE_LIB}" verString REGEX "VERSION: v.([0-9]+.[0-9]+)" LIMIT_COUNT 1)
if (NOT DEFINED NNPI_INF_MAJOR_VERSION)
  string(REGEX MATCH "VERSION: v.([0-9]+).[0-9]+.[0-9]+.[0-9]+" _ ${verString})
  set(NNPI_INF_MAJOR_VERSION ${CMAKE_MATCH_1})
endif()
if (NOT DEFINED NNPI_INF_MINOR_VERSION)
  string(REGEX MATCH "VERSION: v.[0-9]+.([0-9]+).[0-9]+.[0-9]+" _ ${verString})
  set(NNPI_INF_MINOR_VERSION ${CMAKE_MATCH_1})
endif()

message(STATUS "[NNPI] NNPI_MAJOR_VERSION          = ${NNPI_MAJOR_VERSION}")
message(STATUS "[NNPI] NNPI_MINOR_VERSION          = ${NNPI_MINOR_VERSION}")
message(STATUS "[NNPI] NNPI_INF_MAJOR_VERSION      = ${NNPI_INF_MAJOR_VERSION}")
message(STATUS "[NNPI] NNPI_INF_MINOR_VERSION      = ${NNPI_INF_MINOR_VERSION}")
message(STATUS "[NNPI] NNPI_API_DIR                = ${NNPI_API_DIR}")
message(STATUS "[NNPI] NNPI_LIB_DIR                = ${NNPI_TRANSFORMER_LIB}")

message(STATUS "[NNPI] NNPI_INF_API_DIR            = ${NNPI_INF_API_DIR}")
message(STATUS "[NNPI] NNPI_INF_LIB_DIR            = ${NNPI_INFERENCE_LIB}")
message(STATUS "[NNPI] NNPI_MG_API_DIR             = ${NNPI_MG_API}")
message(STATUS "[NNPI] NNPI_MG_LIB_DIR             = ${NNPI_MG_LIB}")
message(STATUS "[NNPI] GLOW_BINARY_DIR             = ${GLOW_BINARY_DIR}")
message(STATUS "[NNPI] NNPI_COLLECT_MEM_USAGE      = ${NNPI_MEM_PROFILING}")
message(STATUS "[NNPI] NNPI_USE_STATIC_TRANSFORMER = ${NNPI_USE_STATIC_TRANSFORMER}")

add_subdirectory(ClassGen)

include_directories(
    ${NNPI_TRANSFORMER_API}
    ${NNPI_INFERENCE_API}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NNPI_MG_API}
    ${CMAKE_CURRENT_SOURCE_DIR}/CustomKernels
    ${CMAKE_CURRENT_SOURCE_DIR}/CustomKernels/DSPInjectors
    ${CMAKE_CURRENT_SOURCE_DIR}/CustomKernels/IAInjectors
)

add_definitions(-DNNPI_MAJOR_VERSION=${NNPI_MAJOR_VERSION})
add_definitions(-DNNPI_MINOR_VERSION=${NNPI_MINOR_VERSION})
add_definitions(-DNNPI_INF_MAJOR_VERSION=${NNPI_INF_MAJOR_VERSION})
add_definitions(-DNNPI_INF_MINOR_VERSION=${NNPI_INF_MINOR_VERSION})

SET_SOURCE_FILES_PROPERTIES(NNPIUtils_AVX512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vl -mavx512bw")

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
    InferenceContext.cpp
    NNPIResource.cpp
    NNPIUtils_AVX512.cpp
    NNPIAdapterContainer.cpp
    NNPIUtils.cpp
    SpecializedLookupTables.cpp
    CustomKernels/GetNNPIKernels.cpp
    CustomKernels/IAInjectors/IAInjectors.cpp
    CustomKernels/DSPInjectors/DSPInjectors.cpp
    CustomKernels/DSPInjectors/DSPInjectorUtils.cpp
    CustomKernels/DSPInjectors/ReluInjector.cpp
    CustomKernels/DSPInjectors/CmpInt32Injector.cpp
)

target_link_libraries(NNPI
                      PRIVATE
                      Backend
                      Base
                      Flags
                      Graph
                      GraphOptimizer
                      CodeGen
                      IR
                      Support
                      gomp
                      ${NNPI_TRANSFORMER_LIB}
                      ${NNPI_INFERENCE_LIB}
                      ${NNPI_MG_LIB}
)

set(linked_backends ${linked_backends} NNPI PARENT_SCOPE)
