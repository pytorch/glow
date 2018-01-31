// Copyright 2017 Facebook Inc.  All Rights Reserved.
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
      .setHasSideEffects(true);
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
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .addGradient();

  BB.newNode("Pool")
      .addEnumCase("Max")
      .addEnumCase("Avg")
      .addInput("Input")
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::SizeT, "Pad")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .addGradient();

  BB.newNode("FullyConnected")
      .addInput("Input")
      .addInput("Filter")
      .addInput("Bias")
      .addMember(MemberType::SizeT, "Depth")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy", "Output")
      .addGradient();

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
      .addGradient();

  BB.newNode("LocalResponseNormalization")
      .addInput("Input")
      .addMember(MemberType::SizeT, "HalfWindowSize")
      .addMember(MemberType::Float, "Alpha")
      .addMember(MemberType::Float, "Beta")
      .addMember(MemberType::Float, "K")
      .addResult("Input.getType()")
      .addGradient();

  //===--------------------------------------------------------------------===//
  //                      Loss operations
  //===--------------------------------------------------------------------===//

  BB.newNode("SoftMax")
      .addInput("Input")
      .addInput("Selected")
      .addResult("Input.getType()")
      .addGradient();

  BB.newNode("Regression")
      .addInput("Input")
      .addInput("Expected")
      .addResult("Input.getType()")
      .addGradient();

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
      .addResult("LHS.getType()")
      .setDocstring("Performs arithmetic operations on the LHS and RHS "
                    "operands. The Compare operations generates a mask that's "
                    "consumed by the select instruction. The format of the "
                    "result is target- and type-specific.")
      .addGradient();

  BB.newNode("Select")
      .addInput("Cond")
      .addInput("LHS")
      .addInput("RHS")
      .addResult("LHS.getType()")
      .setDocstring("Selects between values on the LHS or RHS, depending on "
                    "the value of Cond. Cond is generated by the compare "
                    "instruction, and is target- and type-specific.");

  BB.newNode("BatchedArithmetic")
      .addInput("Batch")
      .addInput("Slice")
      .addEnumCase("Add")
      .addResult("Batch.getType()")
      .setDocstring(
          "Adds the 'Slice' operand to each one of the slices in the batch.");

  BB.newNode("BatchedMatMul")
      .addInput("LHS")
      .addInput("RHS")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .setDocstring("Performs matrix multiplication between the LHS RHS. The "
                    "operands are a stack of two dimensional matrices. If one "
                    "of the matrices has a batch size of one then the matrix "
                    "is broadcasted to match the batch size of the other one."
                    "The result is a three dimensional tensor."
                    "Example: (1, Z, A) x (N, B, Z) => (N, A, B)");

  BB.newNode("BatchedReduce")
      .addInput("Batch")
      .addEnumCase("Add")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .setDocstring("Accumulates all of the layers in the batch and produce a "
                    "tensor that has the same dimensions as the input tensor "
                    "without the first dimension.");

  //===--------------------------------------------------------------------===//
  //                Non-linearities
  //===--------------------------------------------------------------------===//

  BB.newNode("Relu")
      .addInput("Input")
      .addResult("Input.getType()")
      .addGradient();

  BB.newNode("Sigmoid")
      .addInput("Input")
      .addResult("Input.getType()")
      .addGradient();

  BB.newNode("Tanh")
      .addInput("Input")
      .addResult("Input.getType()")
      .addGradient();

  //===--------------------------------------------------------------------===//
  //                Shape transformations
  //===--------------------------------------------------------------------===//

  BB.newNode("Reshape")
      .addInput("Input")
      .addMember(MemberType::VectorSizeT, "Dims")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy");

  BB.newNode("Transpose")
      .addInput("Input")
      .addMember(MemberType::VectorUnsigned, "Shuffle")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy");

  BB.newNode("Broadcast")
      .addInput("Input")
      .addMember(MemberType::VectorSizeT, "Shape")
      .addMember(MemberType::Unsigned, "Axis")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .setDocstring("Performs broadcasting on the Input tensor so that its "
                    "shape matches the provided Shape. The provided Axis "
                    "represents the offset of the Input's shape (from the "
                    "leading dimension) when comparing dimensions to the "
                    "destination Shape.");

  BB.newNode("Concat")
      .addMember(MemberType::VectorNodeValue, "Inputs")
      .addMember(MemberType::SizeT, "Dim")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .setDocstring("The concat operator adds two tensors together.\nThe "
                    "parameter 'dim' specifies the dimension to use when "
                    "joining the tensors.");

  BB.newNode("Slice")
      .addInput("Input")
      .addMember(MemberType::VectorSizeT, "Start")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy");

  BB.newNode("InsertTensor")
      .addInput("Big")
      .addInput("Small")
      .addMember(MemberType::VectorSizeT, "Start")
      .addResult("Big.getType()");

  BB.newNode("Gather")
      .addInput("Data")
      .addInput("Indices")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .setDocstring(
          "Gathers entries of the outer-most dimension of Data indexed by "
          "Indices, and concatenates them. Output tensor will have dimensions: "
          "{I_0, I_1, ... I_n, D_1, D_2, ... D_m}, where D_i and I_j denote "
          "Data and Indices dimensions respectively.");

  //===--------------------------------------------------------------------===//
  //                Nodes used for network training
  //===--------------------------------------------------------------------===//

  BB.newNode("Splat")
      .addMember(MemberType::Float, "Value")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
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
      .setHasSideEffects(true);

  //===--------------------------------------------------------------------===//
  //                Nodes used by quantization.
  //===--------------------------------------------------------------------===//

  BB.newNode("QuantizationProfile")
      .addInput("Input")
      .addInput("Histogram")
      .addInput("ComputationInfo")
      .addExtraMethod("Variable *getHistogramVar() const ;",
                      "Variable *QuantizationProfileNode::getHistogramVar() "
                      "const { return "
                      "llvm::cast<Variable>(Histogram_.getNode()); };")
      .addExtraMethod(
          "Variable *getComputationInfoVar() const;",
          "Variable *QuantizationProfileNode::getComputationInfoVar() const { "
          "return llvm::cast<Variable>(ComputationInfo_.getNode()); };")
      .setDocstring(
          "Generate profile (distribution of values) of the Input tensor. "
          "This data is used for quantization of the tensor later on.")
      .setHasSideEffects(true);

  BB.newNode("Quantize")
      .addInput("Input")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .setDocstring("Quantize floating point tensor. This operation converts "
                    "floating point numbers to integers based on the given "
                    "Scale and Offset. Scale and Offset are deduced from the"
                    "type of the output."
                    "x_q = clip(round(x/Scale) + Offset, -128, 127)");

  BB.newNode("Dequantize")
      .addInput("Input")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .setDocstring(
          "Convert quantized input tensor into the float representation. "
          "x = Scale * (x_q - Offset).");

  BB.newNode("RescaleQuantized")
      .addInput("Input")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .setDocstring(
          "Rescale input quantized tensor to a new Scale and Offset.");

  //===--------------------------------------------------------------------===//
  //                Nodes used by generic transformations
  //===--------------------------------------------------------------------===//
  BB.newNode("Intrinsic")
      .addMember(MemberType::VectorNodeValue, "Inputs")
      .addMember(MemberType::String, "Identifier")
      .addMember(MemberType::VoidStar, "Saved")
      .addIntrinsicOutput()
      .setDocstring(
          "This is a general intrinsic node that represents an opaque "
          "unknown operation. The node is variadic, which means that "
          "it has an unspecified number of inputs and outputs. The "
          "node has an identifier to identify the kind of node.");

  //===--------------------------------------------------------------------===//
  //                Nodes used by unit tests.
  //===--------------------------------------------------------------------===//

  /// This is a test node that's used by the node unittests.
  BB.newNode("Distribute")
      .addInput("Input")
      .addResult("Input.getType()", "Left")
      .addResult("Input.getType()", "Right");

  //===--------------------------------------------------------------------===//
  //                Nodes used by RNN
  //===--------------------------------------------------------------------===//

  BB.newNode("TopK")
      .addInput("Input")
      .addMember(MemberType::SizeT, "K")
      .addExtraParam("TypeRef", "outTy1")
      .addExtraParam("TypeRef", "outTy2")
      .addResult("outTy1", "Values")
      .addResult("outTy2", "Indices")
      .setDocstring(
          "Finds the top K maximal elements for each vector in the tensor. "
          "Vectors are defined as the last dimension in the tensor. "
          "The input shape {D_0, D_1, ... D_n} results in the outputs "
          "{D_0, D_1, ... D_n-1, K}, sorted in non-decreasing order.");

  return 0;
}
