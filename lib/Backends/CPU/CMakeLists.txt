find_program(CLANG_EXECUTABLE NAMES clang DOC "Path to the clang front-end.")

if(NOT CLANG_EXECUTABLE)
  message(FATAL_ERROR "clang is required to build the JIT library.")
endif()

set(LIBJITBC ${GLOW_BINARY_DIR}/libjit.bc)
set(LIBJITSRC ${CMAKE_CURRENT_SOURCE_DIR}/libjit.c)

add_custom_command(OUTPUT
                   ${LIBJITBC}
                   COMMAND ${CLANG_EXECUTABLE} -Os -ffast-math -emit-llvm -o ${LIBJITBC} -c ${LIBJITSRC}
                   DEPENDS ${LIBJITSRC}
                   COMMENT "Clang: Generating runtime bytecode library." VERBATIM)
add_library(CPUBackend
            AllocationsInfo.cpp
            FunctionSpecializer.cpp
            GlowJIT.cpp
            Pipeline.cpp
            Transforms.cpp
            LLVMIRGen.cpp
            CPUBackend.cpp)

llvm_map_components_to_libnames(LLVM_TARGET_LIBRARIES ${LLVM_TARGETS_TO_BUILD})
target_link_libraries(CPUBackend
                      PRIVATE
                        Optimizer
                        Base
                        Graph
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

add_custom_target(
  generate-libjitbc
  DEPENDS ${LIBJITBC}
  )
add_dependencies(CPUBackend generate-libjitbc)

