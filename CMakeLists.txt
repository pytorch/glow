cmake_minimum_required(VERSION 3.5)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

project(Glow C CXX)
enable_testing()

option(GLOW_WITH_CPU "Build the LLVM-based JIT CPU backend" ON)
option(GLOW_WITH_LLVMIRCODEGEN "Build the LLVM-based code generation library" ON)
option(GLOW_WITH_OPENCL "Build the OpenCL backend" OFF)
option(GLOW_WITH_HABANA "Build the Habana backend" OFF)
option(GLOW_BUILD_EXAMPLES "Build the examples" ON)
option(GLOW_BUILD_TESTS "Build the tests" ON)
option(GLOW_WITH_BUNDLES "Build bundles" OFF)
option(LINK_PROTOBUF_AS_DLL "Link against protobuf build as dynamic libray." OFF)

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
include(GlowDefaults)
include(GlowTestSupport)
include(GlowExternalBackends)
include(SanitizerSupport)
include(CoverageSupport)
include(DoxygenSupport)
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

find_package(PNG)
if(PNG_FOUND)
  add_definitions(-DWITH_PNG)
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

if (GLOW_WITH_HABANA)
  list(APPEND CMAKE_PREFIX_PATH /usr/lib/habanalabs)
  find_path(SYNAPSE_INCLUDE_DIR synapse.h)
  find_library(SYNAPSE_LIB Synapse)
  add_definitions(-DGLOW_WITH_HABANA=1)
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

add_subdirectory(lib)
add_subdirectory(tools)

if (GLOW_BUILD_EXAMPLES)
  add_subdirectory(examples)
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

  # All tests except expensive tests
  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} -LE EXPENSIVE
                      DEPENDS ${GLOW_TEST_DEPENDS} USES_TERMINAL)

  # All tests including expensive tests
  add_custom_target(check_expensive COMMAND ${CMAKE_CTEST_COMMAND}
                      DEPENDS ${GLOW_TEST_DEPENDS} USES_TERMINAL)
endif()

add_custom_target(dependency_graph
                  "${CMAKE_COMMAND}" "--graphviz=dependency_graph" .
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
