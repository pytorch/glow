add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/glow/OpenCL/kernels.cl.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl"
          "${CMAKE_BINARY_DIR}/glow/OpenCL/kernels.cl.inc"
  DEPENDS include-bin "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl")

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/glow/OpenCL/kernels_fwd_conv.cl.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_conv.cl"
          "${CMAKE_BINARY_DIR}/glow/OpenCL/kernels_fwd_conv.cl.inc"
  DEPENDS include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_conv.cl")

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/glow/OpenCL/kernels_fwd_quantized_conv.cl.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_quantized_conv.cl"
          "${CMAKE_BINARY_DIR}/glow/OpenCL/kernels_fwd_quantized_conv.cl.inc"
  DEPENDS include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_quantized_conv.cl")

add_library(OpenCLBackend
            "${CMAKE_BINARY_DIR}/glow/OpenCL/kernels.cl.inc"
            "${CMAKE_BINARY_DIR}/glow/OpenCL/kernels_fwd_conv.cl.inc"
            "${CMAKE_BINARY_DIR}/glow/OpenCL/kernels_fwd_quantized_conv.cl.inc"
            OpenCL.cpp
            OpenCLDeviceManager.cpp
            OpenCLFactory.cpp
            Transforms.cpp)

target_link_libraries(OpenCLBackend
                      PRIVATE
                      Backend
                      Base
                      Graph
                      CodeGen
                      IR
                      LLVMCore
                      Optimizer
                      QuantizationBase)

target_link_libraries(OpenCLBackend
                      PUBLIC
                      OpenCL::OpenCL)

set(linked_factories ${linked_factories} OpenCLBackend PARENT_SCOPE)
set(linked_device_managers ${linked_device_managers} OpenCLBackend PARENT_SCOPE)
