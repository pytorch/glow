
cmake_minimum_required(VERSION 3.7)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

project(Glow C CXX)
enable_testing()

option(GLOW_WITH_CPU "Build the LLVM-based JIT CPU backend" OFF)
option(GLOW_WITH_OPENCL "Build the OpenCL backend" ON)

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Export a JSON file with the compilation commands that external tools can use
# to analyze the source code of the project.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GlowDefaults)
include(SanitizerSupport)
include(CoverageSupport)
include(DoxygenSupport)

set(GLOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GLOW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bin)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include_directories(${GLOW_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB_RECURSE header_files include/*.h tools/*.h src/*.h)
add_custom_target(CollectHeaders SOURCES ${header_files})

find_package(PNG)

if(GLOW_WITH_CPU)
  add_definitions(-DGLOW_WITH_CPU=1)
endif ()

if (GLOW_WITH_OPENCL)
  add_definitions(-DGLOW_WITH_OPENCL=1)
  find_package(OpenCL REQUIRED)
endif ()

find_package(LLVM CONFIG)
if (NOT LLVM_FOUND)
  message("Could not find LLVM. Build LLVM manually and")
  message("configure the project with -DCMAKE_PREFIX_PATH=/path/to/llvm/install")
  return()
endif()
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

if(NOT EXISTS "${GLOW_SOURCE_DIR}/tests/googletest/googletest")
  message("The git submodules are not available. ")
  message("Run git submodule update --init --recursive")
  return()
endif()

add_subdirectory(tests/googletest/)

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(tests)

add_custom_target(dependency_graph ALL
                  "${CMAKE_COMMAND}" "--graphviz=dependency_graph" .
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
