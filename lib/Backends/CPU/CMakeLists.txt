
if(NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  message(SEND_ERROR "clang is required to build the JIT library")
endif()

add_library(CPURuntime
            OBJECT
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

find_program(LLVM_LINK_BIN llvm-link)

add_custom_command(OUTPUT
                     ${CMAKE_BINARY_DIR}/libjit.bc
                   DEPENDS
                     $<TARGET_OBJECTS:CPURuntime>
                     CPURuntime
                   COMMAND
                     echo \"$<TARGET_OBJECTS:CPURuntime>\" | tr '$<SEMICOLON>' ' ' | xargs -- ${LLVM_LINK_BIN} -o ${CMAKE_BINARY_DIR}/libjit.bc)
get_target_property(CPURuntime_SOURCES CPURuntime SOURCES)
add_custom_target(GlowCPURuntime
                  ALL
                  DEPENDS
                    ${CMAKE_BINARY_DIR}/libjit.bc
                  SOURCES
                    ${CPURuntime_SOURCES}
                    $<TARGET_OBJECTS:CPURuntime>)

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
add_dependencies(CPUBackend GlowCPURuntime)
