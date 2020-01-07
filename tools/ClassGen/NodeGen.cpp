/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

  BB.includeHeader("glow/Graph/Nodes.h");

  BB.declareNode("Storage");
  BB.declareNode("Constant");
  BB.declareNode("Placeholder");

  BB.newNode("Save")
      .addInput("Input")
      .addInput("Output")
      .addExtraMethod("Placeholder *getPlaceholder() const;",
                      "Placeholder *SaveNode::getPlaceholder() const { return "
                      "llvm::cast<Placeholder>(Output_.getNode()); };")
      .addOverwrittenInput("Output")
      .setHasSideEffects(true)
      .dataParallel()
      .setDocstring("Specifies a node whose Input will be copied to Output."
                    "This node prevents graph optimizations from eliminating "
                    "this node and all of its ancestor nodes. Generally "
                    "intended to save the final result of a network.");

  //===--------------------------------------------------------------------===//
  //                   Convolution / Pool / FC
  //===--------------------------------------------------------------------===//

  BB.newNode("Pad")
      .addInput("Input")
      .addMember(MemberType::Enum, "Mode")
      .addMember(MemberType::VectorSigned, "Pads")
      .addMember(MemberType::Float, "Value")
      .addResultFromCtorArg()
      .setDocstring(
          "Performs padding of a given input tensor. The Padding information "
          "must be specified for each dimension of the tensor in Pads (start "
          "and end padding). In case the padding is negative, it means that "
          "the tensor must be cropped. Mode defines how extra padding elements "
          "are created. Supported modes are defined in the PaddingMode enum: "
          "CONSTANT, REFLECT, EDGE. Value is only used with the CONSTANT "
          "mode.");

  BB.newNode("Convolution")
      .addInput("Input")
      .addInput("Filter")
      .addInput("Bias")
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Group")
      .addMember(MemberType::Unsigned, "Dilation")
      .addMember(MEMBER_TYPE_INFO(glow::ConvolutionLayout), "Layout")
      .addFusedActivation()
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring(
          "Performs 2D Convolution using a given Input, Filter, and "
          "Bias tensors, as well as provided Kernels, Strides, Pads, "
          "Group and Dilation. Supported Layouts are defined in the "
          "ConvolutionLayout enum: NHWC and NCHW. Supported FusedActivations "
          "are defined in the FusedActivation enum.");

  BB.newNode("ChannelwiseQuantizedConvolution")
      .addInput("Input")
      .addInput("Filter")
      .addInput("Bias")
      .addInput("Scales")
      .addInput("Offsets")
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Group")
      .addMember(MemberType::Boolean, "Groupwise")
      .addResultFromCtorArg()
      .setDocstring("Performs 2D Convolution using a given Input, Filter, and "
                    "Bias tensors, as well as provided Kernels, Strides, Pads, "
                    "and Group. Quantization parameters are provided by Scales "
                    "and Offsets. If Groupwise is true then the quantization "
                    "is per-group otherwise it is per-channel.");

  BB.newNode("Convolution3D")
      .addInput("Input")
      .addInput("Filter")
      .addInput("Bias")
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Group")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring("Performs 3D Convolution using a given Input, Filter, and "
                    "Bias tensors, as well as provided Kernels, Strides, Pads, "
                    "and Group.");

  BB.newNode("MaxPool")
      .addInput("Input")
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Enum, "Layout")
      .addResultFromCtorArg("Result")
      .addResultFromCtorArg("Argmax")
      .addGradient()
      .setDocstring(
          "Performs a Max Pool with Argmax operation on the Input "
          "given provided Kernels, Strides, and Pads. Argmax is a flattened "
          "index corresponding to respective max element. Supported layouts "
          "are defined in the ConvolutionLayout enum: NHWC and NCHW.");

  BB.newNode("ArgMax")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Axis")
      .addMember(MemberType::Boolean, "KeepDims")
      .addResultFromCtorArg("Argmax")
      .setDocstring("Finds index of a maximum element along Axis."
                    "If KeepDims is not true, the axis is removed from output");

  BB.newNode("AvgPool")
      .addInput("Input")
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Enum, "Layout")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring(
          "Performs an Average Pool operation on the Input given "
          "provided Kernels, Strides, and Pads. Supported layouts are defined "
          "in the ConvolutionLayout enum: NHWC and NCHW.");

  BB.newNode("AdaptiveAvgPool")
      .addInput("Input")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring(
          "Performs an Adaptive Average Pool operation on the Input given");

  BB.newNode("FullyConnected")
      .addInput("Input")
      .addInput("Weights")
      .addInput("Bias")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring("Creates a FullyConnected node where the Input tensor and "
                    "Weights tensor are multiplied, and then the Bias tensor "
                    "is added to it, producing the Output.");

  BB.newNode("RowwiseQuantizedFullyConnected")
      .addInput("Input")
      .addInput("Weights")
      .addInput("Scales")
      .addInput("Offsets")
      .addInput("Bias")
      .addResultFromCtorArg()
      .setDocstring(
          "Creates a RowwiseQuantizedFullyConnected node where the Input "
          "matrix and the transpose of Weights matrix are multiplied, and "
          "then the Bias vector is broadcast-added to the result. Input, "
          "Bias and Result are regularly quantized, while Weights use row-wise"
          "quantization.");

  //===--------------------------------------------------------------------===//
  //                     Normalization
  //===--------------------------------------------------------------------===//

  BB.newNode("BatchNormalization")
      .addInput("Input")
      .addInput("Scale")
      .addInput("Bias")
      .addInput("Mean")
      .addInput("Var")
      .addMember(MemberType::Unsigned, "ChannelIdx")
      .addMember(MemberType::Float, "Epsilon")
      .addMember(MemberType::Float, "Momentum")
      .addResult("Input.getType()")
      .addGradient()
      .setDocstring("Performs batch normalization on the Input tensor with the "
                    "provided Scale, Bias, Mean, Var, ChannelIdx, Epsilon, and "
                    "Momentum. Similar to Caffe2 SpatialBN, and ONNX "
                    "BatchNormalization operator.");

  BB.newNode("MeanVarNormalization")
      .addInput("Input")
      .addInput("Mean")
      .addInput("Var")
      .addMember(MemberType::Unsigned, "ChannelIdx")
      .addMember(MemberType::Float, "Momentum")
      .addResult("Mean.getType()", "NewMean")
      .addResult("Var.getType()", "NewVar")
      .setDocstring("Calculates new normalized mean and variance based on the "
                    "input mean, variance, and input.");

  BB.newNode("LocalResponseNormalization")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "HalfWindowSize")
      .addMember(MemberType::Float, "Alpha")
      .addMember(MemberType::Float, "Beta")
      .addMember(MemberType::Float, "K")
      .addResult("Input.getType()")
      .addGradient()
      .setDocstring("Performs local response normalization on the Input tensor "
                    "with the provided Scale, Bias, Mean, Var, ChannelIdx, "
                    "Epsilon, and Momentum. Similar to Caffe2 and ONNX LRN.");

  BB.newNode("LayerNormalization")
      .addInput("Input")
      .addInput("Scale")
      .addInput("Bias")
      .addMember(MemberType::Float, "Epsilon")
      .addResult("Input.getType()")
      .setDocstring("Performs layer normalization on the Input tensor with the "
                    "provided Scale, Bias, and Epsilon. Layer sizes are "
                    "determined by the dimensions of Scale and Bias. Similar "
                    "to PyTorch layer_norm.");

  BB.newNode("BatchBoxCox")
      .addInput("Input")
      .addInput("Lambda1")
      .addInput("Lambda2")
      .addMember(MemberType::Float, "Epsilon")
      .addResult("Input.getType()")
      .setDocstring("Apply box-cox transform for each column for each column "
                    "in NxD input tensor");

  //===--------------------------------------------------------------------===//
  //                     Bucketing
  //===--------------------------------------------------------------------===//

  BB.newNode("Bucketize")
      .addInput("Input")
      .addMember(MemberType::VectorFloat, "Boundaries")
      .addResultFromCtorArg()
      .setDocstring("Performs bucketization on the input given Boundaries");

  //===--------------------------------------------------------------------===//
  //                      Loss operations
  //===--------------------------------------------------------------------===//

  BB.newNode("SoftMax")
      .addInput("Input")
      .addInput("Selected")
      .addResultFromCtorArg()
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

  BB.newNode("SigmoidCrossEntropyWithLogits")
      .addInput("Logits")
      .addInput("Targets")
      .addResultFromCtorArg()
      .setDocstring("Computes the sigmoid cross entropy between two inputs.");

  //===--------------------------------------------------------------------===//
  //                      Arithmetic
  //===--------------------------------------------------------------------===//

  BB.newNode("Add")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .addGradient()
      .setDocstring("Performs Add on the LHS and RHS operands.");

  BB.newNode("Mul")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .addGradient()
      .setDocstring("Performs Mul on the LHS and RHS operands.");

  BB.newNode("Sub")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .addGradient()
      .setDocstring("Performs Sub on the LHS and RHS operands.");

  BB.newNode("Div")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .addGradient()
      .setDocstring("Performs Div on the LHS and RHS operands.");

  BB.newNode("Max")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs Max on the LHS and RHS operands.");

  BB.newNode("Min")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs Min on the LHS and RHS operands.");

  BB.newNode("Clip")
      .addInput("Input")
      .addMember(MemberType::Float, "Min")
      .addMember(MemberType::Float, "Max")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Clip range of inputs to lie in [Min, Max].");

  BB.newNode("CmpLTE")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs CmpLTE on the LHS and RHS operands. Generates a "
                    "mask that's consumed by the select instruction. The "
                    "format of the result is target- and type-specific.");

  BB.newNode("CmpEQ")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise equal comparison on the LHS and "
                    "RHS operands. Inputs must be integer.");

  BB.newNode("CmpLT")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring(
          "Compares X and Y element wise sets Dest[i] true if LHS[i] < "
          "RHS[i] otherwise false. Final result is a mask consumed by "
          "Select, ONNX Where, operator.");

  BB.newNode("Pow")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs elementwise pow(LHS, RHS).");

  // clang-format off
  BB.newNode("Log")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs element-wise natural log to the Input.");

  BB.newNode("Exp")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs element-wise exponential to the Input.");
  // clang-format on

  BB.newNode("Select")
      .addInput("Cond")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Selects between values on the LHS or RHS, depending on "
                    "the value of Cond. Cond is generated by the compare "
                    "instruction, and is target- and type-specific.");

  BB.newNode("BatchedAdd")
      .addInput("Batch")
      .addInput("Slice")
      .addResultFromCtorArg()
      .setDocstring(
          "Adds the 'Slice' operand to each one of the slices in the batch.");

  BB.newNode("MatMul")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .setDocstring("Performs matrix multiplication between the LHS and RHS."
                    "Example: (A, Z) x (Z, B) => (A, B)");

  BB.newNode("BatchMatMul")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .setDocstring("Performs batch matrix multiplication between the LHS and "
                    "RHS. The operands are a stack of two dimensional "
                    "matrices. Example: (N, A, Z) x (N, Z, B) => (N, A, B)");

  BB.newNode("BatchedReduceAdd")
      .addInput("Batch")
      .addMember(MemberType::Unsigned, "Axis")
      .addResultFromCtorArg()
      .setDocstring("Accumulates all of the layers in the batch and produce a "
                    "tensor that has the same dimensions as the input tensor "
                    "without the first dimension.");

  BB.newNode("BatchedReduceMean")
      .addInput("Batch")
      .addMember(MemberType::VectorUnsigned, "Axes")
      .addResultFromCtorArg()
      .setDocstring("Performs Average Mean operation on the Input given "
                    "Axes.");

  BB.newNode("BatchedReduceMin")
      .addInput("Batch")
      .addMember(MemberType::VectorUnsigned, "Axes")
      .addResultFromCtorArg()
      .setDocstring("Performs Reduce Min operation on the Input given "
                    "Axes.");

  BB.newNode("ChannelShuffle")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Group")
      .addMember(MemberType::Unsigned, "Kernel")
      .addResultFromCtorArg()
      .setDocstring("Performs Channel shuffle.");

  BB.newNode("LengthsSum")
      .addInput("Data")
      .addInput("Lengths")
      .addResultFromCtorArg()
      .setDocstring("Sums slices of the outermost dimension of Data in groups "
                    "defined by Lengths. The first Lengths[0] slices are "
                    "added together and stored in Result[0], the subsequent "
                    "Lengths[1] slices are added together and stored in "
                    "Result[1], etc.");

  BB.newNode("SparseLengthsSum")
      .addInput("Data")
      .addInput("Indices")
      .addInput("Lengths")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring("Gathers slices of the outer-most dimension of Data "
                    "indexed by Indices vector, and then accumulates them into "
                    "len(Lengths) entries: first Lengths[0] slices are "
                    "aggregated to Result[0], next Lengths[1] slices are "
                    "aggregated to Result[1], etc. I.e. sum(Lengths) must be "
                    "equal to len(Indices).");

  BB.newNode("SparseLengthsWeightedSum")
      .addInput("Data")
      .addInput("Weights")
      .addInput("Indices")
      .addInput("Lengths")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring("Gathers slices of the outer-most dimension of Data "
                    "indexed by Indices vector, and then accumulates them into "
                    "len(Lengths) entries: first Lengths[0] slices are "
                    "aggregated to Result[0], next Lengths[1] slices are "
                    "aggregated to Result[1], etc. I.e. sum(Lengths) must be "
                    "equal to len(Indices). Before doing aggregation, each "
                    "individual slice is scaled by its weight: Result[0] = "
                    "Weights[0] * Slice(0) + Weights[1] * Slice(1) + ... "
                    "It implies that len(Weights) == len(Indices).");

  BB.newNode("EmbeddingBag")
      .addInput("Data")
      .addInput("Weights")
      .addInput("Indices")
      .addInput("Offsets")
      .addMember(MemberType::Boolean, "HasEndOffset")
      .addResultFromCtorArg()
      .setDocstring(
          "Gathers slices of the outer-most dimension of Data "
          "indexed by Indices vector, and then accumulates them into "
          "len(Offsets) entries: first slice between Offsets[0] and Offsets[1] "
          "(or total length if there's only one elem in Offsets) are "
          "aggregated to Result[0], etc. I.e. largest offset must be "
          "less than or equal to len(Indices). Before doing aggregation, each "
          "individual slice is scaled by its weight: Result[0] = "
          "Weights[0] * Slice(0) + Weights[1] * Slice(1) + ... "
          "It implies that len(Weights) == len(Indices).");

  BB.newNode("EmbeddingBagByteRowwiseOffsets")
      .addInput("Data")
      .addInput("Weights")
      .addInput("Indices")
      .addInput("Offsets")
      .addMember(MemberType::Boolean, "UseFP16Accumulation",
                 /* addSetter */ true)
      .addMember(MemberType::Boolean, "HasEndOffset")
      .addResultFromCtorArg()
      .setDocstring("Same as FusedRowwiseQuantizedSparseLengthsWeightedSum but "
                    "using offsets instead of lengths.");

  BB.newNode("RowwiseQuantizedSparseLengthsWeightedSum")
      .addInput("Data")
      .addInput("Scales")
      .addInput("Offsets")
      .addInput("Weights")
      .addInput("Indices")
      .addInput("Lengths")
      .addMember(MemberType::Boolean, "UseFP16Accumulation",
                 /* addSetter */ true)
      .addResultFromCtorArg()
      .setDocstring("Gathers slices of the outer-most dimension of Data "
                    "indexed by Indices vector, and then accumulates them into "
                    "len(Lengths) entries: first Lengths[0] slices are "
                    "aggregated to Result[0], next Lengths[1] slices are "
                    "aggregated to Result[1], etc. I.e. sum(Lengths) must be "
                    "equal to len(Indices). Before doing aggregation, each "
                    "individual slice is scaled by its weight: Result[0] = "
                    "Weights[0] * Slice(0) + Weights[1] * Slice(1) + ... "
                    "It implies that len(Weights) == len(Indices). The input "
                    "data is rowwise-quantized, where the Scales and Offsets "
                    "are 1D tensors of length equal to the first dim of Data.");

  BB.newNode("FusedRowwiseQuantizedSparseLengthsWeightedSum")
      .addInput("Data")
      .addInput("Weights")
      .addInput("Indices")
      .addInput("Lengths")
      .addMember(MemberType::Boolean, "UseFP16Accumulation",
                 /* addSetter */ true)
      .addResultFromCtorArg()
      .setDocstring("Gathers slices of the outer-most dimension of Data "
                    "indexed by Indices vector, and then accumulates them into "
                    "len(Lengths) entries: first Lengths[0] slices are "
                    "aggregated to Result[0], next Lengths[1] slices are "
                    "aggregated to Result[1], etc. I.e. sum(Lengths) must be "
                    "equal to len(Indices). Before doing aggregation, each "
                    "individual slice is scaled by its weight: Result[0] = "
                    "Weights[0] * Slice(0) + Weights[1] * Slice(1) + ... "
                    "It implies that len(Weights) == len(Indices). The input "
                    "data is fused rowwise-quantized, where the Scales and "
                    "Offsets are appended to the end of each row. Thus, Data "
                    "must be a two-dimensional tensor.");

  BB.newNode("FusedRowwiseQuantizedSparseLengthsSum")
      .addInput("Data")
      .addInput("Indices")
      .addInput("Lengths")
      .addMember(MemberType::Boolean, "UseFP16Accumulation",
                 /* addSetter */ true)
      .addResultFromCtorArg()
      .setDocstring("Gathers slices of the outer-most dimension of Data "
                    "indexed by Indices vector, and then accumulates them into "
                    "len(Lengths) entries: first Lengths[0] slices are "
                    "aggregated to Result[0], next Lengths[1] slices are "
                    "aggregated to Result[1], etc. I.e. sum(Lengths) must be "
                    "equal to len(Indices). The input "
                    "data is fused rowwise-quantized, where the Scales and "
                    "Offsets are appended to the end of each row. Thus, Data "
                    "must be a two-dimensional tensor.");

  BB.newNode("LengthsToRanges")
      .addInput("Lengths")
      .addResultFromCtorArg()
      .setDocstring("Given a vector of segment lengths, calculates offsets of "
                    "each segment and packs them next to the lengths. For the "
                    "input vector of length N the output is a Nx2 matrix with "
                    "(offset, lengths) packaged for each segment.");

  BB.newNode("LengthsRangeFill")
      .addInput("Lengths")
      .addResultFromCtorArg()
      .setDocstring(
          "Converts an input Lengths 1D vector into a range sequence.");

  BB.newNode("SparseToDense")
      .addInput("Indices")
      .addInput("Values")
      .addResultFromCtorArg()
      .setDocstring(
          "Converts the sparse representation specified by the pair "
          "(Indices, Values) into a dense one. This dense "
          "representation contains each value from Values at the "
          "corresponding index specified in Indices. Unspecified indices "
          "are filled with zeroes. Indices may contain duplicate values "
          "and in this case, all of the corresponding values in Values "
          "are added together.");

  BB.newNode("SparseToDenseMask")
      .addInput("Indices")
      .addInput("Values")
      .addInput("DefaultValue")
      .addInput("Lengths")
      .addMember(MemberType::VectorDimT, "Mask")
      .addResultFromCtorArg()
      .setDocstring(
          "Converts the sparse representation specified by the pair "
          "(Indices, Values) into a dense one, where compacted tensor only "
          "contains IDs from given Mask. Indices cannot contain duplicate "
          "values. Lengths is used to distinguish elements from different "
          "examples of one batch. That is, first Lengths[0] index-value pairs "
          "belong to batch's example 0, next Lengths[1] pairs belong to "
          "example 1, and so on.");

  // clang-format off
  BB.newNode("IsNaN")
    .addInput("Input")
    .addResultFromCtorArg()
    .dataParallel()
    .setDocstring("Determines whether each element of the Input is NaN and "
                  "generates a mask that can be consumed by a Select node.");
  // clang-format on

  BB.newNode("ReplaceNaN")
      .addInput("Input")
      .addMember(MemberType::Float, "Value")
      .addResultFromCtorArg()
      .setDocstring("Replaces NaNs found in Input with Value.");

  BB.newNode("Modulo")
      .addInput("Input")
      .addMember(MemberType::Int64, "Divisor")
      .addMember(MemberType::Boolean, "SignFollowDivisor")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs elementwise modulo operation on the input where "
                    "each element in the output is the corresponding element "
                    "in the input data modulo Divisor.");

  //===--------------------------------------------------------------------===//
  //                Non-linearities
  //===--------------------------------------------------------------------===//

  BB.newNode("Relu")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .addGradient()
      .setDocstring(
          "Applies ReLU, max(0, x), to each element in the Input tensor.");

  BB.newNode("PRelu")
      .addInput("Input")
      .addInput("Slope")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Applies PReLU, slope * min(0, x) + max(0, x), to each "
                    "element in the Input tensor.");

  BB.newNode("Sigmoid")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .addGradient()
      .setDocstring("Applies Sigmoid, 1 / (1 + exp(-x)), to each element in "
                    "the Input tensor.");

  BB.newNode("Tanh")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .addGradient()
      .setDocstring("Applies hyperbolic tangent to each element in the Input "
                    "tensor.");

  //===--------------------------------------------------------------------===//
  //                Shape transformations
  //===--------------------------------------------------------------------===//

  BB.newNode("Reshape")
      .addInput("Input")
      .addMember(MemberType::VectorDimT, "Dims")
      .addMember(MemberType::String, "Layout")
      .addResultFromCtorArg()
      .setDocstring("Reshape the Input tensor to shape Dims.");

  BB.newNode("Transpose")
      .addInput("Input")
      .addMember(MemberType::VectorUnsigned, "Shuffle")
      .addMember(MemberType::String, "Layout")
      .addResultFromCtorArg()
      .setDocstring("Transpose the Input tensor based on the vector Shuffle, "
                    "which assigns a new axis for each dimension in Input.");

  BB.newNode("Concat")
      .addMember(MemberType::VectorNodeValue, "Inputs")
      .addMember(MemberType::Unsigned, "Dim")
      .addResultFromCtorArg()
      .setDocstring("The concat operator adds two tensors together.\nThe "
                    "parameter 'dim' specifies the dimension to use when "
                    "joining the tensors.");

  BB.newNode("Slice")
      .addInput("Input")
      .addMember(MemberType::VectorDimT, "Start")
      .addResultFromCtorArg()
      .setDocstring("Produces a slice of the Input tensor. The Start vector "
                    "defines the starting indices for each dimension from "
                    "which the slice should be taken. The end index for each "
                    "dimension is determined from the input type's shape.");

  BB.newNode("InsertTensor")
      .addInput("Big")
      .addInput("Small")
      .addMember(MemberType::VectorDimT, "Start")
      .addMember(MemberType::Unsigned, "Count")
      .addMember(MemberType::Unsigned, "Axis")
      .addResult("Big.getType()")
      .setDocstring("Insert tensor Small into tensor Big given indices Start. "
                    "Small is inserted Count times along Axis. The resulting "
                    "Tensor will have the same type as the input Big tensor.");

  BB.newNode("Gather")
      .addInput("Data")
      .addInput("Indices")
      .addMember(MemberType::Unsigned, "BatchDims")
      .addResultFromCtorArg()
      .setDocstring("Gathers entries of the outer-most dimension of Data "
                    "indexed by Indices, and concatenates them. Output tensor "
                    "will have dimensions: {I_0, I_1, ... I_n, D_1, D_2, ... "
                    "D_m}, where D_i and I_j denote Data and Indices "
                    "dimensions respectively. If batchDims is not zero, the "
                    "gather operator will treat the first batchDims as the "
                    "batch and will concat the result of the gather operation "
                    "on each sample in the batch.");

  BB.newNode("GatherRanges")
      .addInput("Data")
      .addInput("Ranges")
      .addResultFromCtorArg("Output")
      .addResultFromCtorArg("Lengths")
      .setDocstring("Gathers entries of Data into Output in groups specified "
                    "by the elements of Ranges. Each element of Ranges "
                    "contains a list of pairs of indices of the form (index, "
                    "length) which specify which entries of data to gather. "
                    "The ordering of elements in Ranges and of pairs within an "
                    "element is preserved in Output. Lengths contains the "
                    "lengths of the ranges gathered by each list of pairs in "
                    "Ranges.");

  BB.newNode("ScatterData")
      .addInput("Data")
      .addInput("Indices")
      .addInput("Slices")
      .addMember(MemberType::Boolean, "Cumulative")
      .addResult("Data.getType()")
      .setDocstring(
          "Copies each slice from Slices into Data at the "
          "corresponding index in Indices. For example, given input "
          "Data {{1,2},{3,4},{5,6}}, Slices {{-3,-4}}, and Indices "
          "{{1}}, the result is {{1,2},{-3,-4},{5,6}}. It also supports "
          "multi-dimensional indices. For example, given input Data "
          "{{1,2},{3,4},{5,6}}, Slices {-3,-4}, and Indices {{1,0},{1,1}} also "
          "produces {{1,2},{-3,-4},{5,6}}. If Cumulative is true, the node "
          "adds values from Slices to Data instead of copying. For example, "
          "given input Data {{1,2},{3,4},{5,6}}, Slices {{-3,-4}}, and Indices "
          "{1}, the result is {{1,2},{0,0},{5,6}}. If an index is specified "
          "several times, its updates will be added several times as well.");

  BB.newNode("Tile")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Count")
      .addMember(MemberType::Unsigned, "Axis")
      .addResultFromCtorArg()
      .setDocstring("Tile an Input tensor Count times along Axis.");

  BB.newNode("BatchOneHot")
      .addInput("Data")
      .addInput("Lengths")
      .addInput("Values")
      .addResultFromCtorArg()
      .setDocstring("Expands each row of the Data to a row of zeros and ones, "
                    "according to One Hot Encoding. i-th element of Result's "
                    "row is one iff Values[i] equals to the corresponding "
                    "element of Data.");

  BB.newNode("SpaceToDepth")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "BlockSize")
      .addResultFromCtorArg()
      .setDocstring("Given Input tensor of [N,H,W,C], where N is the batch "
                    "axis, C is the channel or depth, H is the height and W is "
                    "the width. This produces Output tensor of [N, "
                    "H/BlockSize, W/BlockSize, C * "
                    "BlockSize * BlockSize].");

  BB.newNode("ResizeNearest")
      .addInput("Input")
      .addMember(MemberType::Float, "HeightScale")
      .addMember(MemberType::Float, "WidthScale")
      .addResultFromCtorArg()
      .setDocstring(
          "Given Input tensor of [N,H,W,C], where N is the batch, C is the "
          "channel or depth, H is the height and W is the width, Generates an "
          "Output tensor with resized spatial dimensions using nearest "
          "neighbor interpolation. The Output tensor is of shape [N, "
          "floor(H*HeightScale), floor(W*WidthScale), C]");

  //===--------------------------------------------------------------------===//
  //                Nodes used for network training
  //===--------------------------------------------------------------------===//

  BB.newNode("Splat")
      .addMember(MemberType::Float, "Value")
      .addResultFromCtorArg()
      .setDocstring("Generate a tensor of a specific type filled with 'Value'."
                    "Splat always keep floating point value internally but can"
                    "quantize it based on the output type.");

  BB.newNode("SGD")
      .addInput("Gradient")
      .addInput("Weight")
      .addMember(MemberType::Float, "L1Decay")
      .addMember(MemberType::Float, "L2Decay")
      .addMember(MemberType::Float, "LearningRate")
      .addMember(MemberType::Float, "Momentum")
      .addMember(MemberType::Unsigned, "BatchSize")
      .addResult("Weight.getType()", "UpdatedWeight")
      .setHasSideEffects(true)
      .setDocstring("Stochastic Gradient Descent node used during training. "
                    "Produces the updated weight that needs to be used "
                    "instead of Weight for the next iteration.");

  //===--------------------------------------------------------------------===//
  //             Nodes used for debugging/profiling/printing
  //===--------------------------------------------------------------------===//

  BB.newNode("TraceEvent")
      .addInput("Data")
      .addMember(MemberType::String, "EventName")
      .addMember(MemberType::String, "EventType")
      .addMember(MemberType::Unsigned, "Index")
      .setHasSideEffects(true)
      .setDocstring("Inserts a TraceEvent for profiling.");

  //===--------------------------------------------------------------------===//
  //                Nodes used by quantization.
  //===--------------------------------------------------------------------===//

  BB.newNode("QuantizationProfile")
      .addInput("Input")
      .addInput("Histogram")
      .addInput("ComputationInfo")
      .addMember(MemberType::String, "ProfiledNodeName")
      .addMember(MemberType::Unsigned, "ProfiledOutputNumber")
      .addExtraMethod(
          "Placeholder *getHistogramPlaceholder() const ;\n",
          "Placeholder *QuantizationProfileNode::getHistogramPlaceholder() "
          "const { return "
          "llvm::cast<Placeholder>(Histogram_.getNode()); };\n")
      .addExtraMethod(
          "Placeholder *getComputationInfoPlaceholder() const;\n",
          "Placeholder "
          "*QuantizationProfileNode::getComputationInfoPlaceholder() const "
          "{ "
          "return llvm::cast<Placeholder>(ComputationInfo_.getNode()); };\n")
      .addOverwrittenInput("ComputationInfo")
      .addOverwrittenInput("Histogram")
      .setHasSideEffects(true)
      .setDocstring(
          "Generate profile (distribution of values) of the Input "
          "tensor. This data is used for quantization of the tensor "
          "later on. ProfiledNodeName contains the name of the node "
          "which is profiled by the QuantizationProfile node. "
          "ProfiledNodeName is helpful as lowering might transform the "
          "original graph. "
          "ProfiledOutputNumber contains the position of the node's output "
          "which gets profiled.");

  BB.newNode("IntLookupTable")
      .addInput("Input")
      .addInput("Mapping")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Simple mapping between quantized numbers."
                    "This can be used as quantized sigmoid or tanh functions.");

  BB.newNode("Quantize")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Quantize floating point tensor. This operation converts "
                    "floating point numbers to integers based on the given "
                    "Scale and Offset. Scale and Offset are deduced from the "
                    "type of the output."
                    "x_q = clip(round(x/Scale) + Offset, -128, 127)");

  BB.newNode("Dequantize")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Convert quantized input tensor into the float "
                    "representation. x = Scale * (x_q - Offset).");

  BB.newNode("RescaleQuantized")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Rescale the input quantized tensor to a new Scale and "
                    "Offset. The new Scale and Offset are specified by the "
                    "output type passed to the constructor");

  //===--------------------------------------------------------------------===//
  //                Nodes used by RNN
  //===--------------------------------------------------------------------===//

  BB.newNode("TopK")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "K")
      .addResultFromCtorArg("Values")
      .addResultFromCtorArg("Indices")
      .setDocstring("Finds the top K maximal elements for each vector in the "
                    "tensor. Vectors are defined as the last dimension in the "
                    "tensor. The input shape {D_0, D_1, ... D_n} results in "
                    "the outputs {D_0, D_1, ... D_n-1, K}, sorted in "
                    "non-decreasing order.");
  //===--------------------------------------------------------------------===//
  //                Conversions
  //===--------------------------------------------------------------------===//

  BB.newNode("ConvertTo")
      .addInput("Input")
      .addResultFromCtorArg()
      .setDocstring(
          "Convert the input from its current type to the destination "
          "type. The input and output types must have the same shapes. "
          "Moreover the input and output types must not be quantized types. "
          "Quantized types should use the appropriate Quantize, Dequantize, "
          "and Rescale nodes.");

  //===--------------------------------------------------------------------===//
  //                Backend-Specific Nodes
  //===--------------------------------------------------------------------===//

#include "glow/NodeGenIncludes.h"

  return 0;
}
