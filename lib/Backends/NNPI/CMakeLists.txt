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

if (NOT DEFINED NNPI_API_DIR OR NOT DEFINED NNPI_LIB_DIR)
    message(FATAL_ERROR "Missing NNPI Transformer Dirs")
endif()

message(STATUS "[NNPI] NNPI_API_DIR                = ${NNPI_API_DIR}")
message(STATUS "[NNPI] NNPI_LIB_DIR                = ${NNPI_LIB_DIR}")

if (NOT DEFINED NNPI_INF_API_DIR OR NOT DEFINED NNPI_INF_LIB_DIR)
    message(FATAL_ERROR "Missing NNPI Inference Dirs")
endif()

message(STATUS "[NNPI] NNPI_INF_API_DIR            = ${NNPI_INF_API_DIR}")
message(STATUS "[NNPI] NNPI_INF_LIB_DIR            = ${NNPI_INF_LIB_DIR}")
message(STATUS "[NNPI] GLOW_BINARY_DIR             = ${GLOW_BINARY_DIR}")

include_directories(
    ${NNPI_API_DIR}
    ${NNPI_INF_API_DIR}
)

add_library(NNPI
    NNPI.cpp
    NNPICompiledFunction.cpp
    Importer.cpp
    InferencePool.cpp
    NNPIDeviceManager.cpp
    NNPIFactory.cpp
    BlockStream.cpp
)

target_link_libraries(NNPI
                      PRIVATE
                      Backend
                      Base
                      Graph
                      CodeGen
                      IR
                      Support
                      ${NNPI_LIB_DIR}/libnnpi_transformer.so
                      ${NNPI_INF_LIB_DIR}/libnnpi_inference.so
)

set(linked_backends ${linked_backends} NNPI PARENT_SCOPE)
