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
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " output.h output.cpp output.def import.h export.h\n";
    return -1;
  }

  std::cout << "Writing node descriptors to:\n\t" << argv[1] << "\n\t"
            << argv[2] << "\n\t" << argv[3] << "\n\t" << argv[4] << "\n\t"
            << argv[5] << "\n";

  std::ofstream hFile(argv[1]);
  std::ofstream cFile(argv[2]);
  std::ofstream dFile(argv[3]);
  std::ofstream iFile(argv[4]);
  std::ofstream eFile(argv[5]);

  Builder BB(hFile, cFile, dFile, iFile, eFile);

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
      .skipAutogenSerialization()
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
      .addMember(MemberType::VectorUnsigned, "Pads", /* addSetter */ true)
      .addMember(MemberType::Unsigned, "Group", /* addSetter */ true)
      .addMember(MemberType::VectorUnsigned, "Dilation")
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
      .addInput("FilterScales")
      .addInput("FilterOffsets")
      .addInput("BiasScales")
      .addInput("BiasOffsets")
      .addMember(MemberType::VectorUnsigned, "Kernels", /* addSetter */ true)
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads", /* addSetter */ true)
      .addMember(MemberType::Unsigned, "Group", /* addSetter */ true)
      .addMember(MemberType::VectorUnsigned, "Dilation")
      .addFusedActivation()
      .addResultFromCtorArg()
      .setDocstring(
          "Performs 2D Convolution using a given Input, Filter, and "
          "Bias tensors, as well as provided Kernels, Strides, Pads, "
          "and Group. The filter channel wise quantization parameters "
          "are provided by FilterScales and FilterOffsets while the "
          "bias channel wise quantization parameters are provided by "
          "BiasScales and BiasOffsets.");

  BB.newNode("ConvTranspose")
      .addInput("Input")
      .addInput("Filter")
      .addInput("Bias")
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Group")
      .addMember(MemberType::VectorUnsigned, "Dilation")
      .addResultFromCtorArg()
      .setDocstring("Performs 2D Transposed Convolution using a given Input,"
                    "Filter, and Bias tensors, as well as provided Kernels,"
                    "Strides, Pads, and Group.");

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
      .addMember(MemberType::VectorUnsigned, "Pads", /* addSetter */ true)
      .addMember(MemberType::Enum, "Layout")
      .addMember(MemberType::Boolean, "FlattenIndices")
      .addResultFromCtorArg("Result")
      .addResultFromCtorArg("Argmax")
      .addGradient()
      .setDocstring(
          "Performs a Max Pool with Argmax operation on the Input "
          "given provided Kernels, Strides, and Pads. Argmax is a flattened "
          "index corresponding to respective max element. Supported layouts "
          "are defined in the ConvolutionLayout enum: NHWC and NCHW. If "
          "FlattenIndices is set to true, the returned indices are flattened "
          "and are relative to the whole tensor (similar to ONNX). If it is "
          "set to false, the returned indices are relative to the H and W "
          "dimensions (similar ot pytorch).");

  BB.newNode("ArgMax")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Axis")
      .addMember(MemberType::Boolean, "KeepDims")
      .addResultFromCtorArg()
      .setDocstring("Finds index of a maximum element along Axis. "
                    "If KeepDims is not true, the axis is removed from output");

  BB.newNode("ArgMin")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Axis")
      .addMember(MemberType::Boolean, "KeepDims")
      .addResultFromCtorArg()
      .setDocstring("Finds index of a minimum element along Axis. "
                    "If KeepDims is not true, the axis is removed from output");

  BB.newNode("AvgPool")
      .addInput("Input")
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads", /* addSetter */ true)
      .addMember(MemberType::Enum, "Layout")
      .addMember(MemberType::Boolean, "CountIncludePads")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring(
          "Performs an Average Pool operation on the Input given "
          "provided Kernels, Strides, and Pads. Supported layouts are defined "
          "in the ConvolutionLayout enum: NHWC, NCHW, NTHWC and NCTHW.");

  BB.newNode("AdaptiveAvgPool")
      .addInput("Input")
      .addResultFromCtorArg()
      .addGradient()
      .setDocstring(
          "Performs an Adaptive Average Pool operation on the Input given");

  BB.newNode("Gemm")
      .addInput("A")
      .addInput("B")
      .addInput("C")
      .addMember(MemberType::Float, "Alpha")
      .addMember(MemberType::Float, "Beta")
      .addMember(MemberType::Boolean, "TransposeA")
      .addMember(MemberType::Boolean, "TransposeB")
      .addResultFromCtorArg()
      .setDocstring(
          "Computes Y = Alpha * A * B + Beta * C where Alpha, Beta are scalars "
          "and A, B, C are matrices. If TransposeA or TransposeB is used then "
          "A or B is additionally transposed.");

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
      .addResultFromCtorArg()
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

  BB.newNode("VectorNorm")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Axis")
      .addMember(MemberType::Unsigned, "P")
      .addResultFromCtorArg()
      .setDocstring("Performs L2 norm of the Input operand based on Axis.");

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

  BB.newNode("FloorDiv")
      .addInput("LHS")
      .addInput("RHS")
      .addMember(MemberType::Boolean, "Truncate")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring(
          "Performs Div on the LHS and RHS operands, then Floor. If Truncate "
          "is set to true then truncate the quotient to zero instead.");

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

  BB.newNode("CmpEQ")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise EQUAL comparison between the "
                    "LHS and RHS operands.");

  BB.newNode("CmpNEQ")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise NOT EQUAL comparison between "
                    "the LHS and RHS operands.");

  BB.newNode("CmpLT")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise LESS THAN comparison between "
                    "the LHS and RHS operands.");

  BB.newNode("CmpLTE")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise LESS THAN OR EQUAL comparison "
                    "between the LHS and RHS operands.");

  BB.newNode("Pow")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs elementwise pow(LHS, RHS).");

  BB.newNode("And")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise logical AND between the LHS and "
                    "RHS operands.");

  BB.newNode("Or")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise logical OR between the LHS and "
                    "RHS operands.");

  BB.newNode("Xor")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise logical XOR between the LHS and "
                    "RHS operands.");

  BB.newNode("Not")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise logical NOT of the Input "
                    "operand.");

  BB.newNode("Neg")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise negation (sign flip) of the "
                    "Input operand.");

  BB.newNode("Abs")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise ABS(x) of the Input operand.");

  BB.newNode("Floor")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise FLOOR(x) of the Input operand.");

  BB.newNode("Sign")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise Sign(x) of the Input operand");

  BB.newNode("Ceil")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise CEIL(x) of the Input operand.");

  BB.newNode("Round")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise ROUND(x) of the Input operand.");

  BB.newNode("Truncate")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring(
          "Performs an element-wise TRUNCATE(x) of the Input operand.");

  BB.newNode("Sqrt")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise SQRT(x) of the Input operand.");

  BB.newNode("Rsqrt")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise RSQRT(x) = 1 / SQRT(x) of the "
                    "Input operand.");

  BB.newNode("Reciprocal")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise RECIPROCAL(x) = 1 / x of the "
                    "Input operand.");

  BB.newNode("Sin")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise SIN(x) of the Input operand.");

  BB.newNode("Cos")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise COS(x) of the Input operand.");

  // clang-format off
  BB.newNode("Log")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs element-wise natural log to the Input.");

  BB.newNode("Acos")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise Arccosine(x) of the Input operand.");

  BB.newNode("Asin")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise Arcsine(x) of the Input operand.");

  BB.newNode("Atan")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise Arctan(x) of the Input operand.");

  BB.newNode("Erf")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs an element-wise Erf(x) of the Input operand.");

  BB.newNode("Exp")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs element-wise exponential to the Input.");
  // clang-format on

  BB.newNode("Logit")
      .addInput("Input")
      .addMember(MemberType::Float, "Epsilon")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Computes elementwise: result = log(input / (1 - input)).");

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

  BB.newNode("BatchedMul")
      .addInput("Batch")
      .addInput("Slice")
      .addResultFromCtorArg()
      .setDocstring("Multiplies the 'Slice' operand to each one of the slices "
                    "in the batch.");

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

  BB.newNode("BatchedReduceMax")
      .addInput("Batch")
      .addMember(MemberType::VectorUnsigned, "Axes")
      .addResultFromCtorArg()
      .setDocstring("Performs Reduce Max operation on the Input given "
                    "Axes.");

  BB.newNode("BatchedReduceProd")
      .addInput("Batch")
      .addMember(MemberType::Unsigned, "Axis")
      .addResultFromCtorArg()
      .setDocstring("Accumulates the product all of the layers in the batch "
                    " and produce a tensor that has the same dimensions as "
                    " the input tensor without the first dimension.");

  BB.newNode("ChannelShuffle")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Group")
      .addMember(MemberType::Unsigned, "Kernel")
      .addResultFromCtorArg()
      .setDocstring("Performs Channel shuffle.");

  BB.newNode("CumSum")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Exclusive")
      .addMember(MemberType::Unsigned, "Reverse")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Performs a Cumulative Sum operation over a 1D vector with "
                    "flags for working in exclusive mode and in reverse. In "
                    "each case the output size is the same as in input size."
                    "e.g (default) [1, 2, 3, 4] -> [1, 3, 6, 10]. "
                    "(exclusive) [1, 2, 3, 4] -> [0, 1, 3, 6]. "
                    "(reverse) [1, 2, 3, 4] -> [10, 9, 7, 4]. ");

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
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
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
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
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

  BB.newNode("Embedding")
      .addInput("Weights")
      .addInput("Indices")
      .addMember(MemberType::Int64, "PadIdx")
      .addMember(MemberType::Boolean, "Scale")
      .addMember(MemberType::Boolean, "Sparse")
      .addResultFromCtorArg()
      .setDocstring("Gathers slices of the outer-most dimension of Weights "
                    "indexed by Indices tensor.");

  BB.newNode("EmbeddingBag")
      .addInput("Data")
      .addInput("Weights")
      .addInput("Indices")
      .addInput("Offsets")
      .addMember(MemberType::Boolean, "HasEndOffset")
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
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
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
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
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
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
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
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
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
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

  BB.newNode("BatchedPairwiseDotProduct")
      .addMember(MemberType::VectorNodeValue, "Inputs")
      .addResultFromCtorArg()
      .setDocstring(
          "Performs batched pairwise dot products of the input vectors");

  BB.newNode("BatchedPairwiseDotProductGrad")
      .addInput("OutputGrad")
      .hasExtraResults()
      .addMember(MemberType::VectorNodeValue, "OriginalInputs")
      .setDocstring(
          "Performs the gradient operation for BatchedPairwiseDotProduct");

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

  BB.newNode("Gelu")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Applies GeLU, to each element in the Input tensor.");

  BB.newNode("Clip")
      .addInput("Input")
      .addMember(MemberType::Float, "Min")
      .addMember(MemberType::Float, "Max")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Clip range of inputs to lie in [Min, Max].");

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

  BB.newNode("Swish")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring("Applies Swish, X * Sigmoid(X), to each element in "
                    "the Input tensor.");

  BB.newNode("Tanh")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .addGradient()
      .setDocstring("Applies hyperbolic tangent to each element in the Input "
                    "tensor.");

  BB.newNode("LeakyRelu")
      .addInput("Input")
      .addMember(MemberType::Float, "Alpha")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring(
          "Applies LeakyReLU = x for positive x and alpha * x for negative x "
          "to each element in the Input tensor.");

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

  BB.newNode("GatherND")
      .addInput("Data")
      .addInput("Indices")
      .addResultFromCtorArg()
      .setDocstring(
          "Given Data tensor of rank r >= 1, Indices tensor of rank q >= 1 "
          "This operator gathers slices of Data into "
          "an output tensor of rank q + r - Indices_shape[-1] - 1 .");

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
      .addMember(MemberType::VectorFloat, "Scale")
      .addResultFromCtorArg()
      .setDocstring(
          "Given Input tensor of 3D, 4D, 5D or 6D, generates an "
          "Output tensor with resized spatial dimensions using nearest "
          "neighbor interpolation. The Output tensor is of shape "
          "floor(input_dimension * scale)");

  BB.newNode("ResizeBilinear")
      .addInput("Input")
      .addMember(MemberType::VectorFloat, "Scale")
      .addResultFromCtorArg()
      .setDocstring(
          "Given Input tensor of [N,H,W,C], where N is the batch, C is the "
          "channel or depth, H is the height and W is the width, Generates an "
          "Output tensor with resized spatial dimensions using bilinear "
          "neighbor interpolation. The Output tensor is of shape "
          "floor(input_dimension * scale)");

  BB.newNode("Broadcast")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Axis")
      .addMember(MemberType::VectorDimT, "TargetDim")
      .addResultFromCtorArg()
      .setDocstring(
          "Broadcast the Input tensor to TargetDim using Axis to indicate the "
          "offset between Input dimension and TargetDim");

  //===--------------------------------------------------------------------===//
  //                Reorder transformations
  //===--------------------------------------------------------------------===//

  BB.newNode("Flip")
      .addInput("Input")
      .addMember(MemberType::Unsigned, "Axis")
      .addResultFromCtorArg()
      .setDocstring(
          "Reverse the order of elements in a tensor along the given axis. The "
          "shape of the tensor is preserved, but the elements are reordered. "
          "The node is inspired from Python numpy.");

  //===--------------------------------------------------------------------===//
  //                Nodes used for network training
  //===--------------------------------------------------------------------===//

  BB.newNode("Splat")
      .addMember(MemberType::Float, "Value")
      .addResultFromCtorArg()
      .setDocstring("Generate a tensor of a specific type filled with 'Value'."
                    "Splat always keep floating point value internally but can"
                    "quantize it based on the output type.");

  // clang-format off
  BB.newNode("Touch")
    .addResultFromCtorArg()
    .setDocstring(
      "Generate a tensor of a specific type without initializing "
      "it. This is useful when filling a big tensor entirely with "
      "multiple small slices using InsertTensor nodes such that "
      "the big tensor is not required to be initialized (filled) "
      "with some value prior to insertion. This node is intended "
      "to remove the overhead associated with the initialization "
      "in situations where it is not required.");
  // clang-format on

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

  BB.newNode("LSTMUnit")
      .addInput("Input")
      .addInput("C")
      .addResult("C.getType()", "newC")
      .addResult("C.getType()", "newH")
      .setDocstring(
          "A LSTM unit node, take Input as I, F, G, O,"
          "takes F from forget gate, I from input gate,"
          "O from output gate, G from cell gate and C from cell state. "
          "Calulates newC = sigmoid(F) * C + sigmoid(I) * tanh(G), "
          "newH = tanh(newC) * sigmoid(O).");

  //===--------------------------------------------------------------------===//
  //                Conversions
  //===--------------------------------------------------------------------===//

  BB.newNode("ConvertTo")
      .addInput("Input")
      .addResultFromCtorArg()
      .dataParallel()
      .setDocstring(
          "Convert the input from its current type to the destination "
          "type. The input and output types must have the same shapes. "
          "Moreover the input and output types must not be quantized types. "
          "Quantized types should use the appropriate Quantize, Dequantize, "
          "and Rescale nodes.");

  //===--------------------------------------------------------------------===//
  //                Custom kernels invocations
  //===--------------------------------------------------------------------===//
  BB.newNode("ExternalFunctionCall")
      .addMember(MemberType::VectorNodeValue, "Inputs")
      // For now use single output.
      .addResultFromCtorArg()
      .addMember(MemberType::String, "FunctionName")
      // Examples are function source code, binary, or as needed.
      // The use of the following two fields will vary depending
      // on which kind of external function is used.
      .addMember(MemberType::String, "FunctionImpl")
      // Function kind, e.g. CUDA, function pointer, binary, backend-specific
      // source code.
      .addMember(MemberType::String, "FunctionKind")
      .skipAutogenSerialization()
      .setHasSideEffects(true)
      .setDocstring("This is a node representing an external function call. "
                    "One possible use of this capability is to pass a source "
                    "code for a function/kernel. When processing this node, a "
                    "backend can compile and execute the source code. This "
                    "node can also be used to pass binary or pointers to "
                    "executable code. The semantics and implementation of this "
                    "node not standardized and is very backend-specific.");

  //===--------------------------------------------------------------------===//
  //                Pre Processing
  //===--------------------------------------------------------------------===//

  BB.newNode("AudioSpectrogram")
      .addInput("Input")
      .addInput("Window")
      .addInput("TwiddleFactors")
      .addInput("BitReverseIndices")
      .addInput("ComplexToRealWeights")
      .addMember(MemberType::Unsigned, "WindowSize")
      .addMember(MemberType::Unsigned, "WindowStride")
      .addMember(MemberType::Boolean, "MagnitudeSquared")
      .addResultFromCtorArg("Spectrogram")
      .setDocstring("Computes the spectrogram of a mono audio signal using "
                    "given window size and stride. The FFT length used to "
                    "compute the spectrogram is the next power of 2 (for a "
                    "window size of 640 the FFT length is 1024). The length "
                    "of each spectrogram window is FFT_length / 2 + 1. "
                    "This node is inspired from TensorFlow.");

  BB.newNode("MFCC")
      .addInput("Spectrogram")
      .addInput("MelWeights")
      .addInput("MelRanges")
      .addInput("DctMat")
      .addMember(MemberType::Float, "SampleRate")
      .addMember(MemberType::Float, "LowerFrequency")
      .addMember(MemberType::Float, "UpperFrequency")
      .addMember(MemberType::Unsigned, "FilterBankCount")
      .addMember(MemberType::Unsigned, "NumCoefficients")
      .addResultFromCtorArg("Coefficients")
      .setDocstring("Computes the MFCC (Mel Frequency Cepstral Coefficient) "
                    "for the given spectrogram. This node is mostly used as "
                    "feature extractor for voice/speech audio data in "
                    "voice command or keyword spotting applications. The input "
                    "is assumed to be a power spectrogram and not a magnitude."
                    "This node is inspired from TensorFlow.");

  //===--------------------------------------------------------------------===//
  //                Post Processing
  //===--------------------------------------------------------------------===//

  BB.newNode("NonMaxSuppression")
      .addInput("Boxes")
      .addInput("Scores")
      .addMember(MemberType::Unsigned, "CenterPointBox")
      .addMember(MemberType::Unsigned, "MaxOutputBoxesPerClass")
      .addMember(MemberType::Float, "IouThreshold")
      .addMember(MemberType::Float, "ScoreThreshold")
      .addMember(MemberType::Boolean, "IsTFVersion4")
      .addResultFromCtorArg("Indices")
      .addResultFromCtorArg("NumberOfSelectedIndices")
      .setDocstring("This is a mix of ONNX and TF NMSv4. It supports multiple "
                    "classes and does per class NMS. It also supports TF NMS "
                    "V4 by outputting indices and scalar tensor with number of "
                    "valid indices. It pads the rest with global MIN box.");

  //===--------------------------------------------------------------------===//
  //                Region of Interest nodes
  //===--------------------------------------------------------------------===//

  BB.newNode("ROIAlign")
      .addInput("FeatureMap")
      .addInput("Boxes")
      .addInput("BatchIndices")
      .addMember(MemberType::Enum, "Mode")
      .addMember(MemberType::Unsigned, "OutputHeight")
      .addMember(MemberType::Unsigned, "OutputWidth")
      .addMember(MemberType::Unsigned, "SamplingRatio")
      .addMember(MemberType::Float, "SpatialScale")
      .addMember(MemberType::Boolean, "Aligned")
      .addMember(MemberType::Boolean, "Rotated")
      .addResultFromCtorArg()
      .setDocstring(
          "Performs region of interest align (ROI) operator. "
          "FeatureMap - a tensor of [N,H,W,C]. N is the batch, C is the "
          "channel, H is the height, W is the width. "
          "Boxes - a tensor of [K,4] or [K,5] with format "
          "[[optinal_batch_index] x0, y0, x1, y1]. K is the number of boxes. "
          "BatchIndices - a tensor of [K,]. If N > 1 and Box shape is [K,4], "
          "BatchIndices must be valid. "
          "Output is a tensor with shape [K, OutputHeight, OutputWidth, C]. "
          "Aligned - if true, coordinates are aligned to a center of a pixel.");

  BB.newNode("BBoxTransform")
      .addInput("Rois")
      .addInput("Deltas")
      .addInput("ImInfo")
      .addMember(MemberType::VectorFloat, "Weights")
      .addMember(MemberType::Boolean, "ApplyScale")
      .addMember(MemberType::Boolean, "Rotated")
      .addMember(MemberType::Boolean, "AngleBoundOn")
      .addMember(MemberType::Int64, "AngleBoundLo")
      .addMember(MemberType::Int64, "AngleBoundHi")
      .addMember(MemberType::Float, "ClipAngleThresh")
      .addMember(MemberType::Boolean, "LegacyPlusOne")
      .addResultFromCtorArg("BoxOut")
      .addResultFromCtorArg("RoiBatchSplits")
      .setDocstring(
          "Transform proposal bounding boxes to target bounding box using "
          "bounding box regression deltas. "
          "Rois tensor's format is: "
          "<[optional_batch_index], x1, y1, x2, y2>, shape (M, 4) or (M, 5) "
          "where M is the number of Rois. "
          "For rotated boxes, this would have an additional angle (in degrees) "
          "in the format <[optional_batch_id], ctr_x, ctr_y, w, h, angle> "
          "Deltas are of shape (M, K*4) with format <dx, dy, dw, dh>, "
          "where K is the number of classes. "
          "For rotated Rois: shape (M, K*5), format <dx, dy, dw, dh, da>. "
          "ImInfo is of shape <batch_size, 3> with format <img_height, "
          "img_width, img_scale>."
          "If proposals from multiple images in a batch are present, they "
          "should be grouped sequentially and in incremental order.");

  //===--------------------------------------------------------------------===//
  //                Backend-Specific Nodes
  //===--------------------------------------------------------------------===//

#include "glow/NodeGenIncludes.h"

  return 0;
}
