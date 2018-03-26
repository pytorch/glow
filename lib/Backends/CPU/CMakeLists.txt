
if(NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  message(SEND_ERROR "clang is required to build the JIT library")
endif()

find_program(LLVM_LINK_BIN llvm-link)

add_library(CPURuntime
            SHARED
              libjit.cpp)
set_target_properties(CPURuntime
                      PROPERTIES
                        CXX_STANDARD 11)
target_compile_options(CPURuntime
                       PRIVATE
                         -ffast-math
                         -g
                         -emit-llvm
                         -O0)
set_target_properties(CPURuntime
                      PROPERTIES
                        RULE_LAUNCH_LINK
                          "${LLVM_LINK_BIN} -o ${CMAKE_BINARY_DIR}/libjit.bc <OBJECTS> # "
                        OUTPUT_NAME
                          libjit.bc)

add_library(CPUBackend
            AllocationsInfo.cpp
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
                        Quantization
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
                        LLVMInterpreter
                        LLVMSupport
                        LLVMPasses)
add_dependencies(CPUBackend CPURuntime)
