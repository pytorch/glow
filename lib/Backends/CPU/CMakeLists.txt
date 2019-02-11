if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CLANG_BIN ${CMAKE_CXX_COMPILER})
else()
  find_program(CLANG_BIN clang++)
endif()

find_program(LLVM_LINK_BIN
             NAMES
               llvm-link-6.0
               llvm-link)

set(CMAKE_LLIR_CREATE_SHARED_LIBRARY "${LLVM_LINK_BIN} -o <TARGET> <OBJECTS>")
set(CMAKE_LLIR_CREATE_SHARED_MODULE "${LLVM_LINK_BIN} -o <TARGET> <OBJECTS>")
set(CMAKE_LLIR_CREATE_STATIC_LIBRARY "${LLVM_LINK_BIN} -o <TARGET> <OBJECTS>")

set(CPURunttimeCompileOptions
      -std=c++11
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
    COMMAND ${CLANG_BIN} -c ${libjit_src_file_path} ${CPURunttimeCompileOptions} -o ${libjit_obj_file}
    DEPENDS ${libjit_src_file_path}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

  list(APPEND CPURuntime_OBJS ${libjit_obj_file})
  list(APPEND CPURuntime_SRCS ${libjit_src_file_path})
endforeach()

add_custom_command(
    OUTPUT ${GLOW_BINARY_DIR}/libjit.bc
    COMMAND ${LLVM_LINK_BIN} -o ${GLOW_BINARY_DIR}/libjit.bc ${CPURuntime_OBJS}
    DEPENDS  ${CPURuntime_OBJS} ${CPURuntime_SRCS}
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/glow/libjit_bc.inc
    COMMAND include-bin "${CMAKE_BINARY_DIR}/libjit.bc" "${CMAKE_BINARY_DIR}/glow/libjit_bc.inc"
    DEPENDS ${GLOW_BINARY_DIR}/libjit.bc
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_custom_target(CPURuntime
  DEPENDS ${CMAKE_BINARY_DIR}/glow/libjit_bc.inc
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

if (NOT MSVC)
  add_library(CPURuntimeNative
              libjit/libjit.cpp
              libjit/libjit_conv.cpp
              libjit/libjit_matmul.cpp)
endif(NOT MSVC)

add_library(CPUBackend
            "${CMAKE_BINARY_DIR}/glow/libjit_bc.inc"
            AllocationsInfo.cpp
            BundleSaver.cpp
            CommandLine.cpp
            CPUFunction.cpp
            DebugInfo.cpp
            FunctionSpecializer.cpp
            GlowJIT.cpp
            Pipeline.cpp
            Transforms.cpp
            LLVMIRGen.cpp
            CPUBackend.cpp)

llvm_map_components_to_libnames(LLVM_TARGET_LIBRARIES ${LLVM_TARGETS_TO_BUILD})
target_link_libraries(CPUBackend
                      PUBLIC
                        Backend
                        Base
                        CodeGen
                        BackendUtils
                        Graph
                        IR
                        Optimizer
                        QuantizationBase
                        ${LLVM_TARGET_LIBRARIES}
                        LLVMAnalysis
                        LLVMBitWriter
                        LLVMCodeGen
                        LLVMCore
                        LLVMExecutionEngine
                        LLVMIRReader
                        LLVMInstCombine
                        LLVMInterpreter
                        LLVMMC
                        LLVMObject
                        LLVMPasses
                        LLVMScalarOpts
                        LLVMSupport
                        LLVMTarget
                        LLVMTransformUtils
                        LLVMVectorize
                        LLVMipo)
if(LLVM_VERSION_MAJOR VERSION_GREATER 6)
  target_link_libraries(CPUBackend
                        PRIVATE
                          LLVMOrcJIT)
endif()
add_dependencies(CPUBackend CPURuntime)

add_library(CPUDeviceManager
            CPUDeviceManager.cpp)
target_link_libraries(CPUDeviceManager
                      PRIVATE
                        Backends
                        BackendUtils
                        Base
                        CodeGen
                        CPUBackend
                        Graph
                        IR
                        Optimizer
                        ThreadPool)
