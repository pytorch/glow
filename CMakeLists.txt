cmake_minimum_required(VERSION 3.0i6)
project(Noether)

set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++14")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -fsanitize=address -fno-omit-frame-pointer")

set(NOETHER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NOETHER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(TRAINING_TENSOR_ELEMENT_TYPE float)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

include_directories (${NOETHER_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB_RECURSE header_files include/noether/*.h)

find_package(PNG)

configure_file (
  "${NOETHER_SOURCE_DIR}/include/Config.h.in"
  "${NOETHER_BINARY_DIR}/Config.h"
  )

SET(LinkLibs "")

if (PNG_FOUND)
  SET(LinkLibs ${LinkLibs} ${PNG_LIBRARY})
endif (PNG_FOUND)

add_subdirectory(caffe/)

SET(LinkLibs ${LinkLibs} ${PROTOBUF_LIBRARIES})
SET(LinkLibs ${LinkLibs} libNoether libCaffe)

add_subdirectory(src/noether/)
add_subdirectory(examples/)


add_executable(noether noether.cpp ${header_files})
target_link_libraries (noether LINK_PUBLIC ${LinkLibs})

