#include "glow/Extractor/Extractor.h"

using namespace std;
using namespace glow;

Layer::Layer() {
	type_ = "";
	config_ = std::vector<int>(32, 0);
}

Layer::Layer(glow::Node * node){
	type_ = node->getKindName();
	config_ = std::vector<int>(32, 0);
	extract_info(node);
}

void Layer::extract_info(glow::Node * node){
	if (node->getKindName() == "Convolution") {
		auto *convNode = llvm::dyn_cast<glow::ConvolutionNode>(node);
		int enable = convNode->getKernels()[0];
		int in_width = convNode->getInput().dims()[1];
		int in_height = convNode->getInput().dims()[2];
		int in_channel = convNode->getInput().dims()[3];
		int batch = convNode->getInput().dims()[0];
		int out_width = convNode->getResult().dims()[1];
		int out_height = convNode->getResult().dims()[2];
		int out_channel = convNode->getResult().dims()[3];
		int kernel_size = convNode->getKernels()[0];
		int stride = convNode->getStrides()[0];
		int pad = convNode->getPads()[0];
		int n = convNode->getResult().dims()[3];
		int bn = 0;
		int goup = 0;
		int active = 0;
		int to_ddr = 0;
		config_[0] = enable;
		config_[1] = in_width;
		config_[2] = in_height;
		config_[3] = in_channel;
		config_[4] = batch;
		config_[5] = out_width;
		config_[6] = out_height;
		config_[7] = out_channel;
		config_[8] = kernel_size;
		config_[9] = stride;
		config_[10] = pad;
		config_[11] = n;
		config_[12] = bn;
		config_[13] = goup;
		config_[14] = active;
		config_[15] = to_ddr;
		config_[16] = in_width * in_height;
		config_[17] = in_width * in_height * in_channel;
		config_[18] = out_width * out_height;
		config_[19] = out_width * out_height * out_channel;
		config_[20] = kernel_size * kernel_size;
		config_[21] = kernel_size * kernel_size * in_channel;
		config_[22] = kernel_size * kernel_size * in_channel * n;
	}
	else if (node->getKindName() == "MaxPool") {
		auto *maxpoolNode = llvm::dyn_cast<glow::MaxPoolNode>(node);
		int enable = 1;
		int in_width = maxpoolNode->getInput().dims()[1];
		int in_height = maxpoolNode->getInput().dims()[2];
		int in_channel = maxpoolNode->getInput().dims()[3];
		int batch = maxpoolNode->getInput().dims()[0];
		int out_width = maxpoolNode->getResult().dims()[1];
		int out_height = maxpoolNode->getResult().dims()[2];
		int out_channel = maxpoolNode->getResult().dims()[3];
		int kernel_size = maxpoolNode->getKernels()[0];
		int stride = maxpoolNode->getStrides()[0];
		int pad = maxpoolNode->getPads()[0];
		int n = maxpoolNode->getResult().dims()[3];
		config_[0] = enable;
		config_[1] = in_width;
		config_[2] = in_height;
		config_[3] = in_channel;
		config_[4] = batch;
		config_[5] = out_width;
		config_[6] = out_height;
		config_[7] = out_channel;
		config_[8] = kernel_size;
		config_[9] = stride;
		config_[10] = pad;
		config_[11] = n;
	}		
	else if (node->getKindName() == "ResizeNearest") {
		auto *resizeNode = llvm::dyn_cast<glow::ResizeNearestNode>(node);
		int enable = 1;
		int in_width = resizeNode->getInput().dims()[1];
		int in_height = resizeNode->getInput().dims()[2];
		int in_channel = resizeNode->getInput().dims()[3];
		int batch = resizeNode->getInput().dims()[0];
		int out_width = resizeNode->getResult().dims()[1];
		int out_height = resizeNode->getResult().dims()[2];
		int out_channel = resizeNode->getResult().dims()[3];
		int kernel_size = 1;
		int stride = 2;
		int pad = 1;
		int n = resizeNode->getResult().dims()[3];
		config_[0] = enable;
		config_[1] = in_width;
		config_[2] = in_height;
		config_[3] = in_channel;
		config_[4] = batch;
		config_[5] = out_width;
		config_[6] = out_height;
		config_[7] = out_channel;
		config_[8] = kernel_size;
		config_[9] = stride;
		config_[10] = pad;
		config_[11] = n;
	}
	else{
	}
}

Merged_Layer::Merged_Layer() {
	layer_list_ = std::vector<Layer>(3, Layer());
	layer_added_ = std::vector<bool>(3, false);
}

void Merged_Layer::add_layer(Layer l){
	if (! validate_layer(l)) {
		return;
	}
	if (l.get_type() == "Convolution") {
		layer_list_[0] = l;
		layer_added_[0] = true;
	}
	else if (l.get_type() == "MaxPool") {
		layer_list_[1] = l;
		layer_added_[1] = true;
	}
	else if (l.get_type() == "ResizeNearest") {
		layer_list_[2] = l;
		layer_added_[2] = true;
	}
	else {
	}
}

bool Merged_Layer::validate_layer(Layer l){
	if (l.get_type() == "Convolution" && !layer_added_[0]) {
		return true;
	}
	if (l.get_type() == "MaxPool" && !layer_added_[1]) {
		return true;
	}
	if (l.get_type() == "ResizeNearest" && ! layer_added_[2]) {
		return true;
	}
	return false;
}

Graph_Config::Graph_Config() {
	merged_layer_list_ = vector<Merged_Layer>(0);
}

void Graph_Config::append_merged_layer(Merged_Layer merged_layer){
	merged_layer_list_.push_back(merged_layer);
}


Graph_Config Extractor::extract(glow::Function * func){
	for (auto node : func->getNodes()) {

	}
}