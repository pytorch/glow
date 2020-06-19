add_subdirectory(ClassGen)

include(AddLibjit)

# LIBJIT CPU source files.
set(LIBJIT_CPU_SOURCE_FILES)
list(APPEND LIBJIT_CPU_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit/libjit.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit/libjit_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit/libjit_matmul.cpp
)

# LIBJIT CPU compile options.
set(LIBJIT_CPU_COMPILE_OPTIONS
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../../../glow/include
  -std=c++14
  -g
  -I${CMAKE_SOURCE_DIR}/include
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
  ${LLVMCPURuntimeExtraFlags}
)

# Add CPU LIBJIT.
glow_add_libjit(
  NAME libjit_cpu
  SOURCE_FILES ${LIBJIT_CPU_SOURCE_FILES}
  COMPILE_OPTIONS ${LIBJIT_CPU_COMPILE_OPTIONS}
)

# Add native CPU LIBJIT library used for testing.
if (NOT MSVC)
  add_library(CPURuntimeNative
              ${LIBJIT_CPU_SOURCE_FILES})
endif(NOT MSVC)

add_library(CPUBackend
            ${LIBJIT_INCLUDE_FILE}
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

add_dependencies(CPUBackend libjit_cpu_TARGET)

set(linked_backends ${linked_backends} CPUBackend PARENT_SCOPE)
