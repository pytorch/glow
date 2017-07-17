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


include_directories ("${NOETHER_BINARY_DIR}")

file(GLOB_RECURSE header_files include/noether/*.h)

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS caffe/caffe.proto)

add_executable(noether noether.cpp
                       ./src/noether/Train.cpp
                       ./src/noether/Image.cpp
                       ./src/noether/Network.cpp
                       ./src/noether/Node.cpp
                       ./src/noether/Nodes.cpp
                       ${header_files}
                       ${PROTO_SRCS}
                       ${PROTO_HDRS})

find_package (PNG)
if (PNG_FOUND)
  include_directories(${PNG_INCLUDE_DIR})
  target_link_libraries(noether ${PNG_LIBRARY})
endif (PNG_FOUND)

target_link_libraries(noether ${PROTOBUF_LIBRARIES})

configure_file (
  "${NOETHER_SOURCE_DIR}/include/Config.h.in"
  "${NOETHER_BINARY_DIR}/Config.h"
  )
