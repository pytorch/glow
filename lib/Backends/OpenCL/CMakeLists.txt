add_subdirectory(ClassGen)

set(OpenCL_DIR ${CMAKE_BINARY_DIR}/glow/OpenCL)

add_custom_command(
    OUTPUT ${OpenCL_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OpenCL_DIR})
add_custom_target(make_opencl_dir
    DEPENDS ${OpenCL_DIR})

add_custom_command(
  OUTPUT "${OpenCL_DIR}/kernels.cl.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl"
          "${OpenCL_DIR}/kernels.cl.inc"
  DEPENDS make_opencl_dir include-bin "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl")

add_custom_command(
  OUTPUT "${OpenCL_DIR}/kernels_fwd_conv.cl.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_conv.cl"
          "${OpenCL_DIR}/kernels_fwd_conv.cl.inc"
  DEPENDS make_opencl_dir include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_conv.cl")

add_custom_command(
  OUTPUT "${OpenCL_DIR}/kernels_specialized_no_local_mem_conv.cl.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_specialized_no_local_mem_conv.cl"
          "${OpenCL_DIR}/kernels_specialized_no_local_mem_conv.cl.inc"
  DEPENDS make_opencl_dir include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_specialized_no_local_mem_conv.cl")

add_custom_command(
  OUTPUT "${OpenCL_DIR}/kernels_fwd_quantized_conv.cl.inc"
  COMMAND include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_quantized_conv.cl"
          "${OpenCL_DIR}/kernels_fwd_quantized_conv.cl.inc"
  DEPENDS make_opencl_dir include-bin
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_quantized_conv.cl")

add_library(OpenCLBackend
            "${OpenCL_DIR}/kernels.cl.inc"
            "${OpenCL_DIR}/kernels_fwd_conv.cl.inc"
            "${OpenCL_DIR}/kernels_fwd_quantized_conv.cl.inc"
            "${OpenCL_DIR}/kernels_specialized_no_local_mem_conv.cl.inc"
            OpenCL.cpp
            OpenCLDeviceManager.cpp
            OpenCLFactory.cpp
            OpenCLTensorLayout.cpp
            Transforms.cpp)

target_link_libraries(OpenCLBackend
                      PRIVATE
                      Backend
                      Base
                      Flags
                      Graph
                      CodeGen
                      IR
                      LLVMCore
                      IROptimizer
                      IROptimizerPipeline
                      GraphOptimizerPipeline
                      QuantizationBase
                      Runtime)

target_link_libraries(OpenCLBackend
                      PUBLIC
                      OpenCL::OpenCL)

set(linked_backends ${linked_backends} OpenCLBackend PARENT_SCOPE)
