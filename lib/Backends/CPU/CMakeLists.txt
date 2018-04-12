
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CLANG_BIN ${CMAKE_CXX_COMPILER})
else()
  find_program(CLANG_BIN clang++)
endif()

find_program(LLVM_LINK_BIN llvm-link)

add_library(CPURuntime
            SHARED
              libjit/libjit.cpp
              libjit/libjit_matmul.cpp)
set_target_properties(CPURuntime
                      PROPERTIES
                        CXX_STANDARD 11)
target_compile_options(CPURuntime
                       PRIVATE
                         -ffast-math
                         -g
                         -emit-llvm
                         -O0)
# NOTE(abdulras) explicitly override the compiler and linker invocations with
# custom rules.  The trailing `#` is the comment leader intended to nullify the
# compile and link commands.  Doing this allows us to override the compiler
# (driver) and the linker used for building the runtime which requires clang (to
# emit LLVM IR for the LTO'ed AOT JIT runtime) and uses the `llvm-link` as the
# linker to merge the object files.
set_target_properties(CPURuntime
                      PROPERTIES
                        RULE_LAUNCH_COMPILE
                          "${CLANG_BIN} <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE> # "
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
