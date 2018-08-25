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

add_library(CPURuntimeNative
              libjit/libjit.cpp
              libjit/libjit_conv.cpp
              libjit/libjit_matmul.cpp)

add_library(CPUBackend
            AllocationsInfo.cpp
            BundleSaver.cpp
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
                        QuantizationBase
                        LLVMAnalysis
                        LLVMCodeGen
                        LLVMCore
                        LLVMipo
                        LLVMIRReader
                        LLVMInstCombine
                        LLVMMC
                        LLVMScalarOpts
                        LLVMSupport
                        LLVMTarget
                        LLVMTransformUtils
                        LLVMVectorize
                        ${LLVM_TARGET_LIBRARIES}
                        LLVMCore
                        LLVMExecutionEngine
                        LLVMOrcJIT
                        LLVMInterpreter
                        LLVMSupport
                        LLVMPasses)
add_dependencies(CPUBackend CPURuntime)
