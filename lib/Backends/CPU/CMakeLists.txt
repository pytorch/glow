if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CLANG_BIN ${CMAKE_CXX_COMPILER})
else()
  find_program(CLANG_BIN clang++)
endif()

find_program(LLVM_LINK_BIN
             NAMES
               llvm-link-7
               llvm-link-6.0
               llvm-link)

set(CMAKE_LLIR_CREATE_SHARED_LIBRARY "${LLVM_LINK_BIN} -o <TARGET> <OBJECTS>")
set(CMAKE_LLIR_CREATE_SHARED_MODULE "${LLVM_LINK_BIN} -o <TARGET> <OBJECTS>")
set(CMAKE_LLIR_CREATE_STATIC_LIBRARY "${LLVM_LINK_BIN} -o <TARGET> <OBJECTS>")

set(CPURunttimeCompilationOptions
      -std=c++14
      -ffast-math
      -fno-finite-math-only
      -g0
      -emit-llvm
      -O0)

set(libjit_files "libjit;libjit_conv;libjit_matmul")

set(libjit_obj_file_path ${CMAKE_CURRENT_BINARY_DIR}/CPURuntime)
file(MAKE_DIRECTORY ${libjit_obj_file_path})

set(CPURuntime_OBJS)
set(CPURuntime_SRCS)

foreach(libjit_src_file ${libjit_files})
  set(libjit_obj_file ${libjit_obj_file_path}/${libjit_src_file}${CMAKE_C_OUTPUT_EXTENSION})
  set(libjit_src_file_path ${CMAKE_CURRENT_LIST_DIR}/libjit/${libjit_src_file}.cpp)

  add_custom_command(
    OUTPUT  ${libjit_obj_file}
    COMMAND ${CLANG_BIN} -c ${libjit_src_file_path} ${CPURunttimeCompilationOptions} -o ${libjit_obj_file}
    DEPENDS ${libjit_src_file_path}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

  list(APPEND CPURuntime_OBJS ${libjit_obj_file})
  list(APPEND CPURuntime_SRCS ${libjit_src_file_path})
endforeach()

add_custom_command(
    OUTPUT ${GLOW_BINARY_DIR}/CPU/libjit.bc
    COMMAND ${LLVM_LINK_BIN} -o ${GLOW_BINARY_DIR}/CPU/libjit.bc ${CPURuntime_OBJS}
    DEPENDS  ${CPURuntime_OBJS} ${CPURuntime_SRCS}
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/glow/CPU/libjit_bc.inc
    COMMAND include-bin "${CMAKE_BINARY_DIR}/CPU/libjit.bc" "${CMAKE_BINARY_DIR}/glow/CPU/libjit_bc.inc"
    DEPENDS ${GLOW_BINARY_DIR}/CPU/libjit.bc
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_custom_target(CPURuntime
  DEPENDS ${CMAKE_BINARY_DIR}/glow/CPU/libjit_bc.inc
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

if (NOT MSVC)
  add_library(CPURuntimeNative
              libjit/libjit.cpp
              libjit/libjit_conv.cpp
              libjit/libjit_matmul.cpp)
endif(NOT MSVC)

add_library(CPUBackend
            "${CMAKE_BINARY_DIR}/glow/CPU/libjit_bc.inc"
            CPUFunction.cpp
            Transforms.cpp
            CPUBackend.cpp
            CPULLVMIRGen.cpp)

target_link_libraries(CPUBackend
                      PUBLIC
                        Backend
                        Base
                        CodeGen
                        Graph
                        IR
                        Optimizer
                        QuantizationBase
                        LLVMIRCodeGen)
add_dependencies(CPUBackend CPURuntime)

add_library(CPUDeviceManager
            CPUDeviceManager.cpp)
target_link_libraries(CPUDeviceManager
                      PRIVATE
                        Backends
                        Base
                        CodeGen
                        CPUBackend
                        Graph
                        IR
                        Optimizer
                        ThreadPool)
