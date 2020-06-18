#ifndef GLOW_EXTRACTOR_H
#define GLOW_EXTRACTOR_H

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <utility>
#include "glow/Graph/Node.h"
#include "glow/Graph/Graph.h"
#include "../../lib/IR/GraphScheduler.h"

class Layer {
	// A representation of a fused conv layer.
	// 0: enable     0, 1 for maxpool and upsample, k for conv
	// 1: in_w     width of input
	// 2: in_h     height of input
	// 3: in_c     channel of input
	// 4: batch     batch of input
	// 5: out_w     width of output
	// 6: out_h     height of output
	// 7: out_c     channel of output
	// 8: size      size of kernel
	// 9: stride 
	// 10: pad
	// 11: n     the number of convolution calculation
	// 12: bn     if do batch normalization
	// 13: goup
	// 14: active     the type of activation function 0:leaky 1:linear
	// 15: to_ddr     if saved to ddr(for branch nodes)
	// 16: w_h_in     in_w * in_h
	// 17: w_h_c_in    in_w * in_h * in_channel
	// 18: w_h_out     out_w * out_h
	// 19: w_h_c_out     out_w * out_h * out_channel
	// 20: f_f      size * size
	// 21: f_f_c     size * size * in_c(?)
	// 22: f_f_c_n     size * size * in_c(?) * n
	std::vector<int> config_;
	std::string type_;
	void extract_info(glow::Node * node);
public:
	Layer();
	Layer(glow::Node * node);
	void set_config(int i, int k) {config_[i] = k;}
	std::string get_type() {return type_;}
	std::vector<int> get_config() {return config_;}
};

class Merged_Layer {
	// 0: convolution
	// 1: maxpool
	// 2: upsample (resizenearist)
	std::vector<bool> layer_added_;
	std::vector<Layer> layer_list_;
public:
	Merged_Layer();
	void add_layer(Layer);
	bool validate_layer(Layer);
	bool add_to_ddr(int idx) {
		if (! layer_added_[idx]) return false;
		layer_list_[idx].set_config(15, 1);
		return true;
	}
	bool add_relu(int idx) {
		if (! layer_added_[idx]) return false;
		layer_list_[idx].set_config(14, 0);
		return true;
	}
	// print the config for debug.
	void print_layers() {
		std::cout << "Merged_Layer :" << std::endl;
		for (auto l : layer_list_) {
			for (auto e : l.get_config()) {
				std::cout << e << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "End of the merged layer." << std::endl;
	}
};

class Graph_Config {
	std::vector<Merged_Layer> merged_layer_list_;
public:
	Graph_Config();
	void append_merged_layer(Merged_Layer merged_layer);
};

class Extractor {
public:
	Graph_Config extract(glow::Function * function);
};

#endif 