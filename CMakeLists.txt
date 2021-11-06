cmake_minimum_required(VERSION 3.5)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

project(Glow C CXX)
enable_testing()

option(GLOW_WITH_CPU "Build the LLVM-based JIT CPU backend" ON)
option(GLOW_WITH_LLVMIRCODEGEN "Build the LLVM-based code generation library" ON)
option(GLOW_WITH_OPENCL "Build the OpenCL backend" OFF)
option(GLOW_WITH_NNPI "Build the NNPI backend" OFF)
option(GLOW_WITH_HABANA "Build the Habana backend" OFF)
option(GLOW_USE_PNG_IF_REQUIRED "Link with libpng if required" ON)
option(GLOW_BUILD_EXAMPLES "Build the examples" ON)
option(GLOW_BUILD_PYTORCH_INTEGRATION "Build integration for PyTorch" OFF)
option(GLOW_BUILD_TESTS "Build the tests" ON)
option(GLOW_BUILD_ONNXIFI_DYNLIB "Build dynamic library for Onnxifi" OFF)
option(GLOW_WITH_BUNDLES "Build bundles" OFF)
option(LINK_PROTOBUF_AS_DLL "Link against protobuf build as dynamic libray." OFF)
option(TENSOR_DIMS_32_BITS "Set the max bitwidth of the tensor dimension and related indices to 32b instead of 64b." OFF)
option(GLOW_LIBJIT_FAST_MATH "Build the LLVM backend LIBJIT kernel library with fast math option." ON)

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT BUILD_SHARED_LIBS)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
endif ()

# Export a JSON file with the compilation commands that external tools can use
# to analyze the source code of the project.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(GLOW_USE_LLD)
  if(NOT CMAKE_C_COMPILER_ID MATCHES Clang)
    message(FATAL_ERROR "lld requires the use of the clang compiler")
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    message(FATAL_ERROR "lld does not support MachO yet")
  endif()
  include(CheckCXXCompilerFlag)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fuse-ld=lld")
  CHECK_CXX_COMPILER_FLAG("-fuse-ld=lld" HAS_LLD_FLAG)
  if(NOT HAS_LLD_FLAG)
    message(FATAL_ERROR "lld is not supported by the compiler")
  endif()
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
endif()

include(Glog)
include(Gflags)
include(GlowDefaults)
include(GlowTestSupport)
include(GlowExternalBackends)
include(SanitizerSupport)
include(CoverageSupport)
include(DoxygenSupport)
include(FindBackends)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/CMakeGraphVizOptions.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/CMakeGraphVizOptions.cmake COPYONLY)

set(GLOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GLOW_THIRDPARTY_DIR ${GLOW_SOURCE_DIR}/thirdparty)
set(GLOW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(GLOW_OUTPUT_DIR ${GLOW_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_OUTPUT_DIR})
set(GLOW_MODELS_DIR "" CACHE PATH
    "Directory that contains the various models, like en2gr, resnet50")

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include_directories(${GLOW_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${GLOW_SOURCE_DIR})

file(GLOB_RECURSE header_files include/*.h tools/*.h lib/*.h)
add_custom_target(CollectHeaders SOURCES ${header_files})

find_package(Git)
# Get the commit's short SHA1
execute_process(COMMAND
  "${GIT_EXECUTABLE}" log -1 --pretty=format:%h
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_SHA1
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions("-DGIT_SHA1=\"${GIT_SHA1}\"")
# Get the date of the commit
execute_process(COMMAND
  "${GIT_EXECUTABLE}" log -1 --pretty=format:%ad --date=short
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_DATE
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions("-DGIT_DATE=\"${GIT_DATE}\"")
# Get the name of the most recent tag.
execute_process(COMMAND
  "${GIT_EXECUTABLE}" describe --tags --abbrev=0
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_TAG
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions("-DGIT_TAG=\"${GIT_TAG}\"")

# Compile definition for Glow build date in Y-M-D format.
string(TIMESTAMP GLOW_BUILD_DATE "%Y-%m-%d")
add_definitions("-DGLOW_BUILD_DATE=\"${GLOW_BUILD_DATE}\"")
add_definitions("-DGLOW_VERSION=\"${GLOW_BUILD_DATE} (${GIT_SHA1}) (${GIT_TAG})\"")

if(GLOW_USE_PNG_IF_REQUIRED)
  find_package(PNG)
  if(PNG_FOUND)
    add_definitions(-DWITH_PNG)
  endif()
endif()

if(GLOW_WITH_LLVMIRCODEGEN)
  add_definitions(-DGLOW_WITH_LLVMIRCODEGEN=1)
endif()

if(GLOW_WITH_CPU)
  if(NOT GLOW_WITH_LLVMIRCODEGEN)
    message(FATAL_ERROR "Cannot use -DGLOW_WITH_CPU without -DGLOW_WITH_LLVMIRCODEGEN")
  endif ()
  add_definitions(-DGLOW_WITH_CPU=1)
endif ()

if (GLOW_WITH_OPENCL)
  add_definitions(-DGLOW_WITH_OPENCL=1)
  find_package(OpenCL REQUIRED)
endif ()

if(GLOW_WITH_NNPI)
  add_definitions(-DGLOW_WITH_NNPI=1)
endif()

if (GLOW_WITH_HABANA)
  add_definitions(-DGLOW_WITH_HABANA=1)

  # Find habanalabs libraries.
  list(APPEND CMAKE_PREFIX_PATH /usr/lib/habanalabs)
  find_path(SYNAPSE_INCLUDE_DIR synapse.h)
  find_library(SYNAPSE_LIB Synapse)
  find_library(TPCSIM_SHARED_LIB tpcsim_shared)
  find_library(HL_THUNK_LIB hl-thunk)

  # Create interface library to encapsulate necessary .so's.
  add_library(Synapse INTERFACE)
  target_link_libraries(Synapse
    INTERFACE
    "${SYNAPSE_LIB}"
    "${TPCSIM_SHARED_LIB}"
    "${HL_THUNK_LIB}")
endif ()

set(LLVMCPURuntimeExtraFlags)
if (TENSOR_DIMS_32_BITS)
  add_definitions(-DDIM_T_32)
  list(APPEND LLVMCPURuntimeExtraFlags "-DDIM_T_32")
  message(STATUS "Using 32b tensor dimensions.")
else()
  message(STATUS "Using 64b tensor dimensions.")
endif ()

if (GLOW_LIBJIT_FAST_MATH)
  list(APPEND LLVMCPURuntimeExtraFlags "-DFFAST_MATH")
  list(APPEND LLVMCPURuntimeExtraFlags "-ffast-math")
  list(APPEND LLVMCPURuntimeExtraFlags "-fno-finite-math-only")
  message(STATUS "Libjit option 'fast-math' enabled.")
else()
  message(STATUS "Libjit option 'fast-math' disabled.")
endif ()

# Top level setup for external backends
ExternalBackendsInit()

find_package(LLVM CONFIG)
if(NOT LLVM_FOUND OR LLVM_VERSION VERSION_LESS 7.0)
  message(SEND_ERROR "LLVM >= 7.0 is required to build Glow")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

if(NOT EXISTS "${GLOW_SOURCE_DIR}/tests/googlebenchmark/src")
  message(FATAL_ERROR "No googlebenchmark git submodule. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${GLOW_SOURCE_DIR}/tests/googletest/googletest")
  message(FATAL_ERROR "No googletest git submodule. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${GLOW_SOURCE_DIR}/tests/OutputCheck/bin/OutputCheck")
  message(FATAL_ERROR "No OutputCheck git submodule. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${GLOW_THIRDPARTY_DIR}/onnx")
  message(FATAL_ERROR "No onnx git submodule. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${GLOW_THIRDPARTY_DIR}/foxi")
  message(FATAL_ERROR "No foxi git submodule. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${GLOW_THIRDPARTY_DIR}/fp16")
  message(FATAL_ERROR "No fp16 git submodule. Run: git submodule update --init --recursive")
else()
  include_directories(${GLOW_THIRDPARTY_DIR}/fp16/include)
endif()
if (MSVC)
    find_package(folly CONFIG REQUIRED)
    find_library(JEMALLOC_LIB jemalloc)
    add_library(folly_jemalloc INTERFACE)
    target_link_libraries(folly_jemalloc INTERFACE Folly::folly Folly::folly_deps Folly::follybenchmark Folly::folly_test_util ${JEMALLOC_LIB})
elseif(NOT EXISTS "${GLOW_THIRDPARTY_DIR}/folly/folly")
  message(FATAL_ERROR "No folly git submodule. Run: git submodule update --init --recursive")
else()
  # Apple-specific definitions for folly.
  if(APPLE)
    set(FOLLY_HAVE_WEAK_SYMBOLS ON CACHE BOOL "Compiler supports weak symbols")
    set(OPENSSL_ROOT_DIR /usr/local/opt/openssl)
    find_package(OpenSSL REQUIRED)
  endif()

  # Remove Glow-specific CMAKE_CXX_FLAGS to build folly.
  set(SAVE_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  if (CONFIG_HAVE_KERNEL_TIMESPEC)
    set(CMAKE_CXX_FLAGS "-DCONFIG_HAVE_KERNEL_TIMESPEC")
  else()
    set(CMAKE_CXX_FLAGS "")
  endif()
  set(SAVE_CMAKE_CXX_STANDARD "${CMAKE_CXX_STANDARD}")
  unset(CMAKE_CXX_STANDARD)
  set(CXX_STD "c++14" CACHE STRING "Force c++14 for folly")
  set(COMPILER_HAS_F_ALIGNED_NEW OFF CACHE BOOL "turn off -faligned_new for folly")
  add_subdirectory("${GLOW_THIRDPARTY_DIR}/folly" EXCLUDE_FROM_ALL)
  target_include_directories(folly PUBLIC "${OPENSSL_INCLUDE_DIR}")
  set(CMAKE_CXX_FLAGS "${SAVE_CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_STANDARD "${SAVE_CMAKE_CXX_STANDARD}")
  find_library(JEMALLOC_LIB jemalloc)
  add_library(folly_jemalloc INTERFACE)
  target_link_libraries(folly_jemalloc INTERFACE folly ${JEMALLOC_LIB})
endif()

add_library(Miniz
  ${GLOW_THIRDPARTY_DIR}/miniz-2.0.8/miniz.c)
target_include_directories(Miniz PUBLIC ${GLOW_THIRDPARTY_DIR}/miniz-2.0.8)

add_subdirectory(lib)
add_subdirectory(tools)

if (GLOW_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if (GLOW_BUILD_PYTORCH_INTEGRATION)
  if(NOT EXISTS "${GLOW_THIRDPARTY_DIR}/pybind11")
    message(FATAL_ERROR "No pybind11 git submodule. Run: git submodule update --init --recursive")
  else()
    add_subdirectory(${GLOW_THIRDPARTY_DIR}/pybind11)
  endif()

  add_subdirectory(torch_glow/src)
endif()

if (GLOW_WITH_BUNDLES AND NOT GLOW_WITH_CPU)
  message(FATAL_ERROR "Cannot create bundles without CPU backend. Configure with -DGLOW_WITH_BUNDLES and -DGLOW_WITH_CPU to build bundles.")
endif()

if (GLOW_BUILD_TESTS)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable tests for Google benchmark" FORCE)
  add_subdirectory(tests/googlebenchmark EXCLUDE_FROM_ALL)
  add_subdirectory(tests/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(tests)

  # Fetch the dependencies for all the tests.
  get_property(GLOW_TEST_DEPENDS GLOBAL PROPERTY GLOW_TEST_DEPENDS)

  # All tests except expensive tests and stress tests.
  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -LE EXPENSIVE\\|STRESS
                      DEPENDS ${GLOW_TEST_DEPENDS} USES_TERMINAL)

  # All tests including expensive tests but not stress tests.
  add_custom_target(check_expensive COMMAND ${CMAKE_CTEST_COMMAND} -LE STRESS
                      DEPENDS ${GLOW_TEST_DEPENDS} USES_TERMINAL)

  # All tests including expensive and stress tests.
  add_custom_target(check_expensive_stress COMMAND ${CMAKE_CTEST_COMMAND}
                      DEPENDS ${GLOW_TEST_DEPENDS} USES_TERMINAL)
endif()

add_custom_target(dependency_graph
                  "${CMAKE_COMMAND}" "--graphviz=dependency_graph" .
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
