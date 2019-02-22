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

add_library(OpenCLBackend
            "${CMAKE_BINARY_DIR}/glow/kernels.inc"
            "${CMAKE_BINARY_DIR}/glow/kernels_fwd_conv.inc"
            "${CMAKE_BINARY_DIR}/glow/kernels_fwd_quantized_conv.inc"
            OpenCL.cpp
            Transforms.cpp)

target_link_libraries(OpenCLBackend
                      PRIVATE
                      Backend
                      Base
                      Graph
                      CodeGen
                      IR
                      Optimizer
                      QuantizationBase)

target_link_libraries(OpenCLBackend
                      PUBLIC
                      OpenCL::OpenCL)

add_library(OpenCLDeviceManager
            OpenCLDeviceManager.cpp)
target_link_libraries(OpenCLDeviceManager
                      PRIVATE
                        Backend
                        OpenCLBackend
                        Base
                        CodeGen
                        Graph
                        IR
                        Optimizer
                        ThreadPool)

add_library(OpenCLFactory
              OpenCLFactory.cpp)
target_link_libraries(OpenCLFactory
                      PRIVATE
                        OpenCLBackend)
