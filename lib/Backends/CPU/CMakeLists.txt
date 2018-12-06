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

add_library(CPURuntime
              libjit/libjit.cpp
              libjit/libjit_conv.cpp
              libjit/libjit_matmul.cpp)

# NOTE(abdulras) explicitly override the compiler invocations with a custom
# rule.  The trailing `#` is the comment leader intended to nullify the compile
# commands.  Doing this allows us to override the compiler (driver) used for
# building the runtime which requires clang (to emit LLVM IR for the LTO'ed AOT
# JIT runtime).  Use a custom linker language with the rule specified above to
# invoke our custom linker to merge the bitcode files.
set_target_properties(CPURuntime
                      PROPERTIES
                        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                        CXX_STANDARD 11
                        CXX_STANDARD_REQUIRED YES
                        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                        LINKER_LANGUAGE LLIR
                        OUTPUT_NAME libjit.bc
                        POSITION_INDEPENDENT_CODE YES
                        PREFIX ""
                        SUFFIX ""
                        RULE_LAUNCH_COMPILE "${CLANG_BIN} <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE> #")

# Due to the usage for NaN in the runtime, '-fno-finite-math-only' is needed
# to disable '-ffinite-math-only' activated by '-ffast-math', while benefiting
# from math optimizations.
target_compile_options(CPURuntime
                       PRIVATE
                         -ffast-math
                         -fno-finite-math-only
                         -g0
                         -emit-llvm
                         -O0)

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/glow/libjit_bc.inc"
  COMMAND include-bin "${CMAKE_BINARY_DIR}/libjit.bc" "${CMAKE_BINARY_DIR}/glow/libjit_bc.inc"
  DEPENDS include-bin CPURuntime)

if (NOT MSVC)
add_library(CPURuntimeNative
              libjit/libjit.cpp
              libjit/libjit_conv.cpp
              libjit/libjit_matmul.cpp)
endif()

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
                      DeviceManager
                      Graph
                      IR
                      Optimizer
                      QuantizationBase)
