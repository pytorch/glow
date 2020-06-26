#include <iostream>
#include "libjit_defs.h"
#include "__merlin_define.h"

//namespace {

	// void libjit_conv_acc(int num_layers, int * s) {
	// 	__merlin_load();
	// 	__merlin_run();
	// }

	void libjit_falcon_merge_f(float *weights_in[13], int config_for_kernel[13][96],
		float config_array_layer[24][1024*4], float * layer_0_in, float * layer_15_out,
		float * layer_22_out) {
		std::cout << "libjit_falcon_merge called." << std::endl;
		__merlin_load_weight(weights_in, config_for_kernel, config_array_layer);
		__merlin_exec_top_kernel_overlap(layer_0_in, layer_15_out, layer_22_out);
	}

//}