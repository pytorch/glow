cmake_minimum_required(VERSION 3.0i6)
project(Glow)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++14")

# Export a JSON file with the compilation commands that external tools can use
# to analyze the source code of the project.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -fsanitize=address -fno-omit-frame-pointer")

set(GLOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GLOW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

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

SET(LinkLibs "")

if (PNG_FOUND)
  SET(LinkLibs ${LinkLibs} ${PNG_LIBRARY})
endif (PNG_FOUND)


SET(LinkLibs ${LinkLibs} Importer Support Network)

enable_testing()
add_subdirectory(src/glow/)
add_subdirectory(examples/)
add_subdirectory(tests/)

add_executable(glow glow.cpp ${header_files})
target_link_libraries (glow LINK_PUBLIC ${LinkLibs})

