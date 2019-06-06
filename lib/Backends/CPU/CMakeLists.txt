
add_subdirectory(libjit)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libjit.inc
  COMMAND include-bin $<TARGET_FILE:JIT> "${CMAKE_CURRENT_BINARY_DIR}/libjit.inc"
  DEPENDS include-bin JIT)
add_custom_target(CPURuntime
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libjit.inc)

if (NOT MSVC)
  add_library(CPURuntimeNative
              libjit/libjit.cpp
              libjit/libjit_conv.cpp
              libjit/libjit_matmul.cpp)
endif(NOT MSVC)

add_library(CPUBackend
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
                        Optimizer
                        QuantizationBase
                        LLVMIRCodeGen)
add_dependencies(CPUBackend CPURuntime)
target_include_directories(CPUBackend PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR})

set(linked_backends ${linked_backends} CPUBackend PARENT_SCOPE)
