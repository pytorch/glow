cmake_minimum_required(VERSION 3.7)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

project(Glow C CXX)
enable_testing()

option(GLOW_WITH_CPU "Build the LLVM-based JIT CPU backend" ON)
option(GLOW_WITH_OPENCL "Build the OpenCL backend" ON)
option(GLOW_BUILD_EXAMPLES "Build the examples" ON)
option(GLOW_BUILD_TESTS "Build the tests" ON)
option(GLOW_WITH_BUNDLES "Build bundles" OFF)

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

include(GlowDefaults)
include(GlowTestSupport)
include(SanitizerSupport)
include(CoverageSupport)
include(DoxygenSupport)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/CMakeGraphVizOptions.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/CMakeGraphVizOptions.cmake COPYONLY)

set(GLOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GLOW_THIRDPARTY_DIR ${GLOW_SOURCE_DIR}/thirdparty)
set(GLOW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bin)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include_directories(${GLOW_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB_RECURSE header_files include/*.h tools/*.h lib/*.h)
add_custom_target(CollectHeaders SOURCES ${header_files})

find_package(PNG)
if(PNG_FOUND)
  add_definitions(-DWITH_PNG)
endif()

if(GLOW_WITH_CPU)
  add_definitions(-DGLOW_WITH_CPU=1)
endif ()

if (GLOW_WITH_OPENCL)
  add_definitions(-DGLOW_WITH_OPENCL=1)
  find_package(OpenCL REQUIRED)
endif ()

# Prefer LLVM 7.
find_package(LLVM 7 CONFIG)

# If LLVM 7 not found, try building with LLVM 6.
if (NOT LLVM_FOUND)
  # Fallback to LLVM 6
  find_package(LLVM 6 CONFIG)

  # Fallback to whatever is available.
  if (NOT LLVM_FOUND)
    find_package(LLVM CONFIG)
  endif()
endif()
if (NOT LLVM_FOUND)
  message(FATAL_ERROR "Could not find LLVM. Build LLVM manually and configure the project with -DCMAKE_PREFIX_PATH=/path/to/llvm/install")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

if(NOT EXISTS "${GLOW_SOURCE_DIR}/tests/googletest/googletest")
  message(FATAL_ERROR "No googletest git submodule. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${GLOW_THIRDPARTY_DIR}/onnx")
  message(FATAL_ERROR "No onnx git submodule. Run: git submodule update --init --recursive")
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
  add_subdirectory(tests/googletest)
  add_subdirectory(tests)

  # Fetch the dependencies for all the tests.
  get_property(GLOW_TEST_DEPENDS GLOBAL PROPERTY GLOW_TEST_DEPENDS)

  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                      DEPENDS ${GLOW_TEST_DEPENDS} USES_TERMINAL)
endif()

add_custom_target(dependency_graph
                  "${CMAKE_COMMAND}" "--graphviz=dependency_graph" .
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
