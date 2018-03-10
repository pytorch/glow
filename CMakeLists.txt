
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

option(BUILD_DOCS "Build documentation" OFF)
# Documentation Option
if(BUILD_DOCS)
  # check if Doxygen is installed
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    message("Generating documentation")

    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/docs)
      file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/docs)
    endif (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/docs)

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating C++ API documentation with Doxygen"
        VERBATIM)

  else (DOXYGEN_FOUND)
    message(FATAL_ERROR "Doxygen needs to be installed to generate the documentation")
  endif (DOXYGEN_FOUND)
endif (BUILD_DOCS)

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
