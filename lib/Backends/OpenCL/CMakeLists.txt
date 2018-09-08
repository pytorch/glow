set(KERNELS_HDR ${GLOW_BINARY_DIR}/glow/kernels_cl.inc)
set(KERNELS_FWD_CONV_HDR ${GLOW_BINARY_DIR}/glow/kernels_fwd_conv_cl.inc)
set(KERNELS_FWD_QUANTIZED_CONV_HDR ${GLOW_BINARY_DIR}/glow/kernels_fwd_quantized_conv_cl.inc)

add_custom_command(
    OUTPUT "${KERNELS_HDR}"
    COMMAND ${CMAKE_COMMAND}
        -DBIN_FILE=kernels.cl
        -DINC_FILE=${KERNELS_HDR}
        -P ${GLOW_SOURCE_DIR}/cmake/GlowConvertCL.cmake
    DEPENDS ${GLOW_SOURCE_DIR}/cmake/GlowConvertCL.cmake kernels.cl
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_command(
    OUTPUT "${KERNELS_FWD_CONV_HDR}"
    COMMAND ${CMAKE_COMMAND}
        -DBIN_FILE=kernels_fwd_conv.cl
        -DINC_FILE=${KERNELS_FWD_CONV_HDR}
        -P ${GLOW_SOURCE_DIR}/cmake/GlowConvertCL.cmake
    DEPENDS ${GLOW_SOURCE_DIR}/cmake/GlowConvertCL.cmake kernels_fwd_conv.cl
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_command(
    OUTPUT "${KERNELS_FWD_QUANTIZED_CONV_HDR}"
    COMMAND ${CMAKE_COMMAND}
        -DBIN_FILE=kernels_fwd_quantized_conv.cl
        -DINC_FILE=${KERNELS_FWD_QUANTIZED_CONV_HDR}
        -P ${GLOW_SOURCE_DIR}/cmake/GlowConvertCL.cmake
    DEPENDS ${GLOW_SOURCE_DIR}/cmake/GlowConvertCL.cmake kernels_fwd_quantized_conv.cl
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(OpenCL
          ${KERNELS_HDR}
          ${KERNELS_FWD_CONV_HDR}
          ${KERNELS_FWD_QUANTIZED_CONV_HDR}
          Shaders.cpp
          OpenCL.cpp
          Transforms.cpp)

target_link_libraries(OpenCL
                      PRIVATE
                      Base
                      Graph
                      CodeGen
                      IR
                      QuantizationBase)

target_link_libraries(OpenCL
                      PRIVATE
                      OpenCL::OpenCL)
