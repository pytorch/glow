add_subdirectory(ClassGen)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CLANG_BIN ${CMAKE_CXX_COMPILER})
else()
  find_program(CLANG_BIN clang++)
endif()
if(NOT CLANG_BIN)
  message(SEND_ERROR "unable to find clang, cannot build the CPU runtime")
endif()

if(NOT LLVM_LINK_BIN)
  if(MSVC)
    set(LLVM_LINK_BIN ${LLVM_BINARY_DIR}/$(Configuration)/bin/llvm-link)
  else()
    find_program(LLVM_LINK_BIN NAMES llvm-link)
    if(NOT EXISTS ${LLVM_LINK_BIN})
      set(LLVM_LINK_BIN ${LLVM_BINARY_DIR}/bin/llvm-link)
    endif()
  endif()
endif()
if(NOT EXISTS ${LLVM_LINK_BIN} AND NOT MSVC)
  message(SEND_ERROR "unable to find llvm-link, cannot build the CPU runtime")
endif()

set(CPURunttimeCompilationOptions
      -I ${CMAKE_CURRENT_SOURCE_DIR}/../../../../glow/include
      -std=c++14
      -ffast-math
      -fno-finite-math-only
      -g
      -I${CMAKE_SOURCE_DIR}/include
      -emit-llvm
      -O0

      # When building for a custom target, use this as an example of how to
      # set up cross-compilation correctly. Then, when building bundles,
      # do not forget to specify the -target= option with the same target
      # string. Also, when building bundles, specify the -llvm-compiler option
      # together with -llvm-compiler-opt to specify custom compilation flags for
      # cross-compilation.

      # --sysroot=/usr/arm-linux-gnueabihf/
      # -target armv7-neon-linux-gnueabihf
      # -I/usr/arm-linux-gnueabihf/include/c++/7.4.0/arm-linux-gnueabihf/
      ${LLVMCPURuntimeExtraFlags})

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

file(MAKE_DIRECTORY ${GLOW_BINARY_DIR}/CPU)
add_custom_command(
    OUTPUT ${GLOW_BINARY_DIR}/CPU/libjit.bc
    COMMAND ${LLVM_LINK_BIN} -o ${GLOW_BINARY_DIR}/CPU/libjit.bc ${CPURuntime_OBJS}
    DEPENDS  ${CPURuntime_OBJS} ${CPURuntime_SRCS}
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

file(MAKE_DIRECTORY ${GLOW_BINARY_DIR}/glow/CPU)
add_custom_command(
    OUTPUT ${GLOW_BINARY_DIR}/glow/CPU/libjit_bc.inc
    COMMAND include-bin "${GLOW_BINARY_DIR}/CPU/libjit.bc" "${GLOW_BINARY_DIR}/glow/CPU/libjit_bc.inc"
    DEPENDS ${GLOW_BINARY_DIR}/CPU/libjit.bc
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_custom_target(CPURuntime
  DEPENDS ${GLOW_BINARY_DIR}/glow/CPU/libjit_bc.inc
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

if (NOT MSVC)
  add_library(CPURuntimeNative
              libjit/libjit.cpp
              libjit/libjit_conv.cpp
              libjit/libjit_matmul.cpp)
endif(NOT MSVC)

add_library(CPUBackend
            "${GLOW_BINARY_DIR}/glow/CPU/libjit_bc.inc"
            CPUBackend.cpp
            CPUDeviceManager.cpp
            CPUFactory.cpp
            CPUFunction.cpp
            CPULLVMIRGen.cpp
            Transforms.cpp)
target_link_libraries(CPUBackend
                      PUBLIC
                        Backend
                        Base
                        CodeGen
                        Graph
                        IR
                        IROptimizer
                        IROptimizerPipeline
                        GraphOptimizerPipeline
                        QuantizationBase
                        Runtime
                        LLVMIRCodeGen)
add_dependencies(CPUBackend CPURuntime)

set(linked_backends ${linked_backends} CPUBackend PARENT_SCOPE)
