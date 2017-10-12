
cmake_minimum_required(VERSION 3.7)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

project(Glow)
enable_testing()

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export a JSON file with the compilation commands that external tools can use
# to analyze the source code of the project.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GlowDefaults)
include(SanitizerSupport)

if(MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /EHsc- /GR-")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /fp:fast")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Oy /Od")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-exceptions -fno-rtti")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
endif()

set(GLOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GLOW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bin)

set(TRAINING_TENSOR_ELEMENT_TYPE float)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

include_directories (${GLOW_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB_RECURSE header_files include/glow/*.h)
find_package(PNG)

configure_file (
  "${GLOW_SOURCE_DIR}/include/Config.h.in"
  "${GLOW_BINARY_DIR}/Config.h"
  )

add_subdirectory(external)
add_subdirectory(src/glow/)
add_subdirectory(examples/)
add_subdirectory(tools/)

if(ENABLE_TESTING)
  add_subdirectory(tests)
endif()

