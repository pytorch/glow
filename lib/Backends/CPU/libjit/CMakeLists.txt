
set(SAVED_CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})

# -- ensure that we have clang as the compiler
if(NOT CMAKE_CXX_COMPILER_ID MATCHES Clang)
  find_program(CLANG_BIN clang++)
  if(NOT CLANG_BIN)
    message(SEND_ERROR "unable to find clang, cannot build the CPU runtime")
  endif()
  set(CMAKE_CXX_COMPILER ${CLANG_BIN})
endif()

# -- ensure that we have llvm-link to link with
if(NOT LLVM_LINK_BIN)
  if(MSVC)
    set(LLVM_LINK_BIN ${LLVM_BINARY_DIR}/$(Configuration)/bin/llvm-link)
  else()
    set(LLVM_LINK_BIN ${LLVM_BINARY_DIR}/bin/llvm-link)
  endif()
endif()
if(NOT EXISTS ${LLVM_LINK_BIN})
  message(SEND_ERROR "unable to find llvm-link, cannot build the CPU runtime")
endif()

# -- setup rules to link the JIT runtime
set(CMAKE_LLIR_CREATE_SHARED_MODULE "${LLVM_LINK_BIN} -o <TARGET> <OBJECTS>")

# -- build the JIT runtime
add_library(JIT MODULE
  libjit.cpp
  libjit_conv.cpp
  libjit_matmul.cpp)
set_target_properties(JIT PROPERTIES
  CXX_EXTENSIONS FALSE
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED TRUE
  LINKER_LANGUAGE LLIR
  OUTPUT_NAME libjit.bc
  PREFIX ""
  SUFFIX "")
target_compile_options(JIT PRIVATE
  -ffast-math -fno-finite-math-only -emit-llvm -g0 -O0)

set(CMAKE_CXX_COMPILER ${SAVED_CMAKE_CXX_COMPILER})

