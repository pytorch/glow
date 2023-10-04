#include(GlowAddLibjit)
#
#set(LIBJIT_CMSIS_SOURCE_FILES)
#list(APPEND LIBJIT_CMSIS_SOURCE_FILES
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c
#  ${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c
#)
#
#set(LIBJIT_CMSIS_COMPILE_OPTIONS
#    -I${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Include
#    -std=c89
#    -g
#    -O0
#)
#
#glow_add_libjit(
#  NAME "libjit_cmsis"
#  SOURCE_FILES ${LIBJIT_CMSIS_SOURCE_FILES}
#  COMPILE_OPTIONS ${LIBJIT_CMSIS_COMPILE_OPTIONS}
#  CLANG_BIN /usr/bin/gcc
#)

    #            ${libjit_cmsis_INCLUDE_FILE}
add_library(CMSISBackend
            CMSISBackend.cpp
	    CMSISFactory.cpp
            CMSISLLVMIRGen.cpp)

	#add_dependencies(CMSISBackend libjit_cmsis_TARGET)

set(linked_backends ${linked_backends} CMSISBackend PARENT_SCOPE)
