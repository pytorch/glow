cmake_minimum_required(VERSION 3.0i6)
project(Noether)

set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++14")

message("Configure with -DCMAKE_BUILD_TYPE=Release for optimized builds")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")

set(NOETHER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NOETHER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

file(GLOB_RECURSE header_files include/noether/*.h)

add_executable(noether noether.cpp ./src/noether/Network.cpp ${header_files})

target_compile_options(noether PUBLIC -std=c++14 -Wall)

find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIR})

target_link_libraries(noether ${PNG_LIBRARY})


