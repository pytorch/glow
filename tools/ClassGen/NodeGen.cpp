// Copyright 2017 Facebook Inc.  All Rights Reserved.
#include "BackendNodes.h"
#include "NodeBuilder.h"

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " output.h output.cpp output.def\n";
    return -1;
  }

  std::cout << "Writing node descriptors to:\n\t" << argv[1] << "\n\t"
            << argv[2] << "\n\t" << argv[3] << "\n";

  std::ofstream hFile(argv[1]);
  std::ofstream cFile(argv[2]);
  std::ofstream dFile(argv[3]);

  Builder BB(hFile, cFile, dFile);

  //===--------------------------------------------------------------------===//
  //                    Input/Output nodes
  //===--------------------------------------------------------------------===//

  BB.declareNode("Variable");

  BB.newNode("Save")
      .addInput("Input")
      .addInput("Output")
      .addExtraMethod("Variable *getVariable() const;",
                      "Variable *SaveNode::getVariable() const { return "
                      "llvm::cast<Variable>(Output_.getNode()); };")
      .addOverwrittenInput("Output")
      .setHasSideEffects(true)
      .setDocstring("Specifies a node whose Input will be copied to Output."
                    "This node prevents graph optimizations from eliminating "
                    "this node and all of its ancestor nodes. Generally "
                    "intended to save the final result of a network.");

  //===--------------------------------------------------------------------===//
  //                   Convolution / Pool / FC
  //===--------------------------------------------------------------------===//

  BB.newNode("Convolution")
      .addInput("Input")
      .addInput("Filter")
      .addInput("Bias")
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::SizeT, "Pad")
      .addMember(MemberType::SizeT, "Depth")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring("Performs Convolution using a given Input, Filter, and "
                    "Bias tensors, as well as provided Kernel, Stride, Pad, "
                    "and Depth.");

  BB.newNode("Pool")
      .addEnumCase("Max")
      .addEnumCase("Avg")
      .addInput("Input")
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::SizeT, "Pad")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring("Performs a Pool operation (either Max or Avg) on the "
                    "Input given provided Kernel, Stride, and Pad.");

  BB.newNode("FullyConnected")
      .addInput("Input")
      .addInput("Weights")
      .addInput("Bias")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring("Creates a FullyConnected node where the Input tensor and "
                    "Weights tensor are multiplied, and then the Bias tensor "
                    "is added to it, producing the Output.");

  //===--------------------------------------------------------------------===//
  //                     Normalization
  //===--------------------------------------------------------------------===//

  BB.newNode("BatchNormalization")
      .addInput("Input")
      .addInput("Scale")
      .addInput("Bias")
      .addInput("Mean")
      .addInput("Var")
      .addMember(MemberType::SizeT, "ChannelIdx")
      .addMember(MemberType::Float, "Epsilon")
      .addMember(MemberType::Float, "Momentum")
      .addResult("Input.getType()")
      .addGradient()
      .setDocstring("Performs batch normalization on the Input tensor with the "
                    "provided Scale, Bias, Mean, Var, ChannelIdx, Epsilon, and "
                    "Momentum. Similar to Caffe2 SpatialBN.");

  BB.newNode("LocalResponseNormalization")
      .addInput("Input")
      .addMember(MemberType::SizeT, "HalfWindowSize")
      .addMember(MemberType::Float, "Alpha")
      .addMember(MemberType::Float, "Beta")
      .addMember(MemberType::Float, "K")
      .addResult("Input.getType()")
      .addGradient()
      .setDocstring("Performs local response normalization on the Input tensor "
                    "with the provided Scale, Bias, Mean, Var, ChannelIdx, "
                    "Epsilon, and Momentum. Similar to Caffe2 LRN.");

  //===--------------------------------------------------------------------===//
  //                      Loss operations
  //===--------------------------------------------------------------------===//

  BB.newNode("SoftMax")
      .addInput("Input")
      .addInput("Selected")
      .addResult("Input.getType()")
      .addGradient()
      .setDocstring("Performs SoftMax normalization on the Input tensor.");

  BB.newNode("CrossEntropyLoss")
      .addInput("P")
      .addInput("Labels")
      .addResultFromCtorArg("CE")
      .addGradient()
      .setDocstring("Computes the average cross entropy loss of the input.");

  BB.newNode("Regression")
      .addInput("Input")
      .addInput("Expected")
      .addResult("Input.getType()")
      .addGradient()
      .setDocstring(
          "Takes an Input tensor and creates a regression output layer.");

  //===--------------------------------------------------------------------===//
  //                      Arithmetic
  //===--------------------------------------------------------------------===//

  BB.newNode("Arithmetic")
      .addEnumCase("Add")
      .addEnumCase("Mul")
      .addEnumCase("Sub")
      .addEnumCase("Div")
      .addEnumCase("Max")
      .addEnumCase("Min")
      .addEnumCase("CmpLTE")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring("Performs arithmetic operations on the LHS and RHS "
                    "operands. The Compare operations generates a mask that's "
                    "consumed by the select instruction. The format of the "
                    "result is target- and type-specific.");

  BB.newNode("Select")
      .addInput("Cond")
      .addInput("LHS")
      .addInput("RHS")
      .addResult("LHS.getType()")
      .setDocstring("Selects between values on the LHS or RHS, depending on "
                    "the value of Cond. Cond is generated by the compare "
                    "instruction, and is target- and type-specific.");

  BB.newNode("BatchedArithmetic")
      .addEnumCase("Add")
      .addInput("Batch")
      .addInput("Slice")
      .addResultFromCtorArg()
      .setDocstring(
          "Adds the 'Slice' operand to each one of the slices in the batch.");

  BB.newNode("BatchedMatMul")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .setDocstring("Performs matrix multiplication between the LHS RHS. The "
                    "operands are a stack of two dimensional matrices. If one "
                    "of the matrices has a batch size of one then the matrix "
                    "is broadcasted to match the batch size of the other one."
                    "The result is a three dimensional tensor."
                    "Example: (1, A, Z) x (N, Z, B) => (N, A, B)");

  BB.newNode("BatchedReduce")
      .addInput("Batch")
      .addEnumCase("Add")
      .addResultFromCtorArg()
      .setDocstring("Accumulates all of the layers in the batch and produce a "
                    "tensor that has the same dimensions as the input tensor "
                    "without the first dimension.");

  //===--------------------------------------------------------------------===//
  //                Non-linearities
  //===--------------------------------------------------------------------===//

  BB.newNode("Relu")
      .addInput("Input")
      .addResult("Input.getType()")
      .addGradient()
      .setDocstring(
          "Applies ReLU, max(0, x), to each element in the Input tensor.");

  BB.newNode("Sigmoid")
      .addInput("Input")
      .addResult("Input.getType()")
      .addGradient()
      .setDocstring("Applies Sigmoid, 1 / (1 + exp(-x)), to each element in "
                    "the Input tensor.");

  BB.newNode("Tanh")
      .addInput("Input")
      .addResult("Input.getType()")
      .addGradient()
      .setDocstring("Applies hyperbolic tangent to each element in the Input "
                    "tensor.");

  //===--------------------------------------------------------------------===//
  //                Shape transformations
  //===--------------------------------------------------------------------===//

  BB.newNode("Reshape")
      .addInput("Input")
      .addMember(MemberType::VectorSizeT, "Dims")
      .addResultFromCtorArg()
      .setDocstring("Reshape the Input tensor to shape Dims.");

  BB.newNode("Transpose")
      .addInput("Input")
      .addMember(MemberType::VectorUnsigned, "Shuffle")
      .addResultFromCtorArg()
      .setDocstring("Transpose the Input tensor based on the vector Shuffle, "
                    "which assigns a new axis for each dimension in Input.");

  BB.newNode("Broadcast")
      .addInput("Input")
      .addMember(MemberType::VectorSizeT, "Shape")
      .addMember(MemberType::Unsigned, "Axis")
      .addResultFromCtorArg()
      .setDocstring("Performs broadcasting on the Input tensor so that its "
                    "shape matches the provided Shape. The provided Axis "
                    "represents the offset of the Input's shape (from the "
                    "leading dimension) when comparing dimensions to the "
                    "destination Shape.");

  BB.newNode("Concat")
      .addMember(MemberType::VectorNodeValue, "Inputs")
      .addMember(MemberType::Unsigned, "Dim")
      .addResultFromCtorArg()
      .setDocstring("The concat operator adds two tensors together.\nThe "
                    "parameter 'dim' specifies the dimension to use when "
                    "joining the tensors.");

  BB.newNode("Slice")
      .addInput("Input")
      .addMember(MemberType::VectorSizeT, "Start")
      .addResultFromCtorArg()
      .setDocstring("Produces a slice of the Input tensor. The Start vector "
                    "defines the starting indices for each dimension from "
                    "which the slice should be taken. The end index for each "
                    "dimension is determined from the input type's shape.");

  BB.newNode("InsertTensor")
      .addInput("Big")
      .addInput("Small")
      .addMember(MemberType::VectorSizeT, "Start")
      .addResult("Big.getType()")
      .setDocstring("Insert a tensor Small into tensor Big given indices "
                    "Start. The resulting Tensor will have the same type as "
                    "the input Big tensor.");

  BB.newNode("Gather")
      .addInput("Data")
      .addInput("Indices")
      .addResultFromCtorArg()
      .setDocstring("Gathers entries of the outer-most dimension of Data  "
                    "indexed by Indices, and concatenates them. Output tensor  "
                    "will have dimensions: "
                    "{I_0, I_1, ... I_n, D_1, D_2, ... D_m}, where D_i and I_j "
                    "denote Data and Indices dimensions respectively.");

  //===--------------------------------------------------------------------===//
  //                Nodes used for network training
  //===--------------------------------------------------------------------===//

  BB.newNode("Splat")
      .addMember(MemberType::Float, "Value")
      .addResultFromCtorArg()
      .setDocstring("Generate a tensor of a specific type filled with 'Value'");

  BB.newNode("SGD")
      .addInput("Gradient")
      .addInput("Weight")
      .addInput("Gsum")
      .addMember(MemberType::Float, "L1Decay")
      .addMember(MemberType::Float, "L2Decay")
      .addMember(MemberType::Float, "LearningRate")
      .addMember(MemberType::Float, "Momentum")
      .addMember(MemberType::Unsigned, "BatchSize")
      .addOverwrittenInput("Weight")
      .setHasSideEffects(true)
      .setDocstring("Stochastic Gradient Descent node used during training.");

  //===--------------------------------------------------------------------===//
  //                Nodes used by quantization.
  //===--------------------------------------------------------------------===//

  BB.newNode("QuantizationProfile")
      .addInput("Input")
      .addInput("Histogram")
      .addInput("ComputationInfo")
      .addMember(MemberType::String, "ProfiledNodeName")
      .addExtraMethod("Variable *getHistogramVar() const ;",
                      "Variable *QuantizationProfileNode::getHistogramVar() "
                      "const { return "
                      "llvm::cast<Variable>(Histogram_.getNode()); };")
      .addExtraMethod(
          "Variable *getComputationInfoVar() const;",
          "Variable *QuantizationProfileNode::getComputationInfoVar() const { "
          "return llvm::cast<Variable>(ComputationInfo_.getNode()); };")
      .addOverwrittenInput("ComputationInfo")
      .addOverwrittenInput("Histogram")
      .setHasSideEffects(true)
      .setDocstring("Generate profile (distribution of values) of the Input"
                    "tensor. This data is used for quantization of the tensor "
                    "later on.");

  BB.newNode("Quantize")
      .addInput("Input")
      .addResultFromCtorArg()
      .setDocstring("Quantize floating point tensor. This operation converts "
                    "floating point numbers to integers based on the given "
                    "Scale and Offset. Scale and Offset are deduced from the"
                    "type of the output."
                    "x_q = clip(round(x/Scale) + Offset, -128, 127)");

  BB.newNode("Dequantize")
      .addInput("Input")
      .addResultFromCtorArg()
      .setDocstring("Convert quantized input tensor into the float "
                    "representation. x = Scale * (x_q - Offset).");

  BB.newNode("RescaleQuantized")
      .addInput("Input")
      .addResultFromCtorArg()
      .setDocstring("Rescale input quantized tensor to a new Scale and "
                    "Offset.");

  //===--------------------------------------------------------------------===//
  //                Nodes used by generic transformations
  //===--------------------------------------------------------------------===//
  BB.newNode("Intrinsic")
      .addMember(MemberType::VectorNodeValue, "Inputs")
      .addMember(MemberType::String, "Identifier")
      .addIntrinsicOutput()
      .setDocstring("This is a general intrinsic node that represents an "
                    "opaque unknown operation. The node is variadic, which "
                    "means that it has an unspecified number of inputs and "
                    "outputs. The node has an identifier to identify the kind "
                    "of node.");

  //===--------------------------------------------------------------------===//
  //                Nodes used by RNN
  //===--------------------------------------------------------------------===//

  BB.newNode("TopK")
      .addInput("Input")
      .addMember(MemberType::SizeT, "K")
      .addResultFromCtorArg("Values")
      .addResultFromCtorArg("Indices")
      .setDocstring("Finds the top K maximal elements for each vector in the "
                    "tensor. Vectors are defined as the last dimension in the "
                    "tensor. The input shape {D_0, D_1, ... D_n} results in "
                    "theoutputs {D_0, D_1, ... D_n-1, K}, sorted in "
                    "non-decreasing order.");

  //===--------------------------------------------------------------------===//
  //                Backend-Specific Nodes
  //===--------------------------------------------------------------------===//

  addBackendSpecificNodes(BB);

  return 0;
}
