if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CLANG_BIN ${CMAKE_CXX_COMPILER})
else()
  find_program(CLANG_BIN clang++)
endif()

find_program(LLVM_LINK_BIN
             NAMES
               llvm-link-6.0
               llvm-link)

set(CPURunttimeCompileOptions -ffast-math -g0 -emit-llvm -O0)
set (CPURuntime_OBJS
  ${CMAKE_CURRENT_BINARY_DIR}/CPURuntime/libjit.bc.o
  ${CMAKE_CURRENT_BINARY_DIR}/CPURuntime/libjit_conv.bc.o
  ${CMAKE_CURRENT_BINARY_DIR}/CPURuntime/libjit_matmul.bc.o
)

# NOTE(abdulras) explicitly override the compiler and linker invocations with
# custom rules.
add_custom_command(
  OUTPUT ${CPURuntime_OBJS}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CPURuntime
  COMMAND ${CLANG_BIN} -c libjit/libjit.cpp ${CPURunttimeCompileOptions} -o ${CMAKE_CURRENT_BINARY_DIR}/CPURuntime/libjit.bc.o
  COMMAND ${CLANG_BIN} -c libjit/libjit_conv.cpp ${CPURunttimeCompileOptions} -o ${CMAKE_CURRENT_BINARY_DIR}/CPURuntime/libjit_conv.bc.o
  COMMAND ${CLANG_BIN} -c libjit/libjit_matmul.cpp ${CPURunttimeCompileOptions} -o ${CMAKE_CURRENT_BINARY_DIR}/CPURuntime/libjit_matmul.bc.o
  DEPENDS libjit/libjit.cpp libjit/libjit_conv.cpp libjit/libjit_matmul.cpp
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

add_custom_target(CPURuntime
  COMMAND ${LLVM_LINK_BIN} -o ${GLOW_BINARY_DIR}/libjit.bc ${CPURuntime_OBJS}
  DEPENDS ${CPURuntime_OBJS}
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/glow/libjit_bc.inc"
  COMMAND include-bin "${GLOW_BINARY_DIR}/libjit.bc" "${CMAKE_BINARY_DIR}/glow/libjit_bc.inc"
  DEPENDS include-bin CPURuntime)

add_library(CPURuntimeNative
              libjit/libjit.cpp
              libjit/libjit_conv.cpp
              libjit/libjit_matmul.cpp)

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
                      PRIVATE
                        Base
                        CodeGen
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
