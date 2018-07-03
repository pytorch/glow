set(KERNELS_HDR ${GLOW_BINARY_DIR}/kernels_cl.h)
set(KERNELS_FWD_CONV_HDR ${GLOW_BINARY_DIR}/kernels_fwd_conv_cl.h)

add_custom_command(
        OUTPUT "${KERNELS_HDR}"
        COMMAND xxd -i < ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl > ${KERNELS_HDR})

add_custom_command(
        OUTPUT "${KERNELS_FWD_CONV_HDR}"
        COMMAND xxd -i < ${CMAKE_CURRENT_SOURCE_DIR}/kernels_fwd_conv.cl > ${KERNELS_FWD_CONV_HDR})

add_library(OpenCL
            ${KERNELS_HDR}
            ${KERNELS_FWD_CONV_HDR}
            OpenCL.cpp
            Transforms.cpp)

target_link_libraries(OpenCL
                      PRIVATE
                      Base
                      Graph
                      CodeGen
                      IR)

target_link_libraries(OpenCL
                      PRIVATE
                      OpenCL::OpenCL)
