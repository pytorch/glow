add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/glow/kernels.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl"
          "${CMAKE_BINARY_DIR}/glow/kernels.inc"
  DEPENDS include-bin CPURuntime "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl")

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/glow/kernels_fwd_conv.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_conv.cl"
          "${CMAKE_BINARY_DIR}/glow/kernels_fwd_conv.inc"
  DEPENDS include-bin CPURuntime
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_conv.cl")

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/glow/kernels_fwd_quantized_conv.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_quantized_conv.cl"
          "${CMAKE_BINARY_DIR}/glow/kernels_fwd_quantized_conv.inc"
  DEPENDS include-bin CPURuntime
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_quantized_conv.cl")

add_library(OpenCL
            "${CMAKE_BINARY_DIR}/glow/kernels.inc"
            "${CMAKE_BINARY_DIR}/glow/kernels_fwd_conv.inc"
            "${CMAKE_BINARY_DIR}/glow/kernels_fwd_quantized_conv.inc"
            OpenCL.cpp
            Transforms.cpp)

target_link_libraries(OpenCL
                      PRIVATE
                      Backend
                      BackendUtils
                      Base
                      Graph
                      CodeGen
                      IR
                      Optimizer
                      QuantizationBase)

target_link_libraries(OpenCL
                      PRIVATE
                      OpenCL::OpenCL)
