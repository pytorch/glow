include(GlowSerialize)

add_library(LLVMIRCodeGen
            AllocationsInfo.cpp
            BundleSaver.cpp
            CommandLine.cpp
            LLVMCompiledFunction.cpp
            DebugInfo.cpp
            JITFilePrinter.cpp
            FunctionSpecializer.cpp
            GlowJIT.cpp
            Pipeline.cpp
            LLVMIRGen.cpp
            LLVMBackend.cpp)

llvm_map_components_to_libnames(LLVM_TARGET_LIBRARIES ${LLVM_TARGETS_TO_BUILD})
target_link_libraries(LLVMIRCodeGen
                      PUBLIC
                        Backend
                        Base
                        CodeGen
                        Flags
                        Graph
                        IR
                        IROptimizer
                        IROptimizerPipeline
                        GraphOptimizerPipeline
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
  target_link_libraries(LLVMIRCodeGen
                        PRIVATE
                          LLVMOrcJIT)
endif()

# List the object files from "libobj" directory.
FILE(GLOB objects "${CMAKE_CURRENT_LIST_DIR}/libobj/*.o")

# Add object file for testing.
if (GLOW_WITH_BUNDLES)
  list(APPEND objects "${GLOW_SOURCE_DIR}/examples/bundles/bundle_with_extra_objects/test.o")
endif()

# Serialize bundle external object files.
glow_serialize(
  INP_FILES ${objects}
  OUT_DIR "${GLOW_BINARY_DIR}/libobj"
  OUT_FILE "${GLOW_BINARY_DIR}/libobj/BundleObjectRegister.h"
  OUT_VAR "bundleObjectRegister"
  OUT_TARGET BundleObjectRegisterTarget
)
add_dependencies(LLVMIRCodeGen BundleObjectRegisterTarget)
include_directories(${GLOW_BINARY_DIR}/libobj)
