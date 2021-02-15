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

#include "InstrBuilder.h"

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr
        << "Usage: " << argv[0]
        << " header.h impl.cpp enums.def irbuilder.h irbuilder.cpp irgen.h\n";
    return -1;
  }

  std::cout << "Writing instr descriptors to:\n\t" << argv[1] << "\n\t"
            << argv[2] << "\n\t" << argv[3] << "\n\t" << argv[4] << "\n\t"
            << argv[5] << "\n\t" << argv[6] << "\n";

  std::ofstream headerStream(argv[1]);
  std::ofstream cppStream(argv[2]);
  std::ofstream defStream(argv[3]);
  std::ofstream builderHeaderStream(argv[4]);
  std::ofstream builderCppStream(argv[5]);
  std::ofstream irGenStream(argv[6]);

  Builder BB(headerStream, cppStream, defStream, builderHeaderStream,
             builderCppStream, irGenStream);

  //===--------------------------------------------------------------------===//
  //               Memory / Buffer Management
  //===--------------------------------------------------------------------===//

  BB.declareValue("WeightVar");

  BB.newInstr("AllocActivation")
      .addMember(MemberType::TypeRef, "Ty")
      .setType("Ty")
      .addExtraMethod(
          "void setTy(TypeRef Ty);",
          "void AllocActivationInst::setTy(TypeRef Ty) { Ty_ = Ty; }");

  BB.newInstr("TensorView")
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::TypeRef, "Ty")
      .addMember(MemberType::VectorDimT, "Offsets")
      .setType("Ty");

  BB.newInstr("DeallocActivation")
      .addOperand("Src", OperandKind::Out)
      .addExtraMethod("AllocActivationInst *getAlloc() const; ",
                      "AllocActivationInst *DeallocActivationInst::getAlloc() "
                      "const { return  "
                      "llvm::cast<AllocActivationInst>(getSrc()); }")
      .setType("Src->getType()");

  BB.newInstr("Copy")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .setType("Src->getType()")
      .inplaceOperand({"Dest", "Src"})
      .dataParallel();

  //===--------------------------------------------------------------------===//
  //                   Convolution / Pool / FC
  //===--------------------------------------------------------------------===//

  BB.newInstr("Convolution")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Filter", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Group")
      .addMember(MemberType::VectorUnsigned, "Dilation")
      .addMember(MEMBER_TYPE_INFO(ConvolutionLayout), "Layout")
      .addFusedActivation()
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter"})
      .addGradientInstr({"Src", "Filter"}, {"Dest", "Src", "Filter", "Bias"});

  BB.newInstr("ChannelwiseQuantizedConvolution")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Filter", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .addOperand("FilterScales", OperandKind::In)
      .addOperand("FilterOffsets", OperandKind::In)
      .addOperand("BiasScales", OperandKind::In)
      .addOperand("BiasOffsets", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Group")
      .addMember(MemberType::VectorUnsigned, "Dilation")
      .addFusedActivation()
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType,
                  {"Dest", "Src", "Filter", "ElemKind::Int8QTy"});

  BB.newInstr("ConvTranspose")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Filter", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Group")
      .addMember(MemberType::VectorUnsigned, "Dilation")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter"});

  BB.newInstr("Convolution3D")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Filter", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Group")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter"})
      .addGradientInstr({"Src", "Filter"}, {"Dest", "Src", "Filter", "Bias"});

  BB.newInstr("BatchNormalization")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Scale", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .addOperand("Mean", OperandKind::In)
      .addOperand("Var", OperandKind::In)
      .addMember(MemberType::Unsigned, "ChannelIdx")
      .addMember(MemberType::Float, "Epsilon")
      .addMember(MemberType::Float, "Momentum")
      .autoIRGen()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .setType("Dest->getType()");

  // MaxPool version caching Argmax flattened coordinates. It is both used by
  // itself, and also to restore XY coordinates to speedup gradient-based
  // computations.
  BB.newInstr("MaxPoolWithArgmax")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Argmax", OperandKind::Out)
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Layout")
      .addMember(MemberType::Boolean, "FlattenIndices")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .addGradientInstr({"Dest", "Src", "Argmax"}, {"Dest", "Src"});

  BB.newInstr("MaxPool")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Layout")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

  BB.newInstr("AvgPool")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .addMember(MemberType::Unsigned, "Layout")
      .addMember(MemberType::Boolean, "CountIncludePads")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .addGradientInstr({"Dest", "Src"}, {"Dest", "Src"});

  BB.newInstr("ArgMax")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::Unsigned, "Axis")
      .addMember(MemberType::Boolean, "KeepDims")
      .autoIRGen()
      .autoVerify(VerifyKind::NoVerify);

  BB.newInstr("ArgMin")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::Unsigned, "Axis")
      .addMember(MemberType::Boolean, "KeepDims")
      .autoIRGen()
      .autoVerify(VerifyKind::NoVerify);

  BB.newInstr("AdaptiveAvgPool")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .addGradientInstr({"Dest"}, {"Dest", "Src"});

  BB.newInstr("FullyConnected")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Weights", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

  BB.newInstr("RowwiseQuantizedFullyConnected")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Weights", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .addOperand("Scales", OperandKind::In)
      .addOperand("Offsets", OperandKind::In)
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType,
                  {"Dest", "Src", "ElemKind::Int8QTy"});

  //===--------------------------------------------------------------------===//
  //                     Normalization
  //===--------------------------------------------------------------------===//

  BB.newInstr("LocalResponseNormalization")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Scale", OperandKind::Out)
      .addMember(MemberType::Unsigned, "HalfWindowSize")
      .addMember(MemberType::Float, "Alpha")
      .addMember(MemberType::Float, "Beta")
      .addMember(MemberType::Float, "K")
      .setType("Src->getType()")
      .autoVerify(VerifyKind::SameType, {"Dest", "Src", "Scale"})
      .addGradientInstr({"Dest", "Src", "Scale"}, {"Dest", "Src"});

  //===--------------------------------------------------------------------===//
  //                      Loss functions
  //===--------------------------------------------------------------------===//

  BB.newInstr("SoftMax")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("SoftMaxGrad")
      .addOperand("OrigDest", OperandKind::In)
      .addOperand("OrigSrc", OperandKind::In)
      .addOperand("Selected", OperandKind::In)
      .addOperand("SrcGrad", OperandKind::Out)
      .autoVerify(VerifyKind::SameType, {"OrigDest", "OrigSrc", "SrcGrad"});

  BB.newInstr("CrossEntropyLoss")
      .addOperand("P", OperandKind::In)
      .addOperand("Labels", OperandKind::In)
      .addOperand("CE", OperandKind::Out)
      .autoVerify(VerifyKind::NoVerify);

  BB.newInstr("CrossEntropyLossGrad")
      .addOperand("CEGrad", OperandKind::In)
      .addOperand("P", OperandKind::In)
      .addOperand("Labels", OperandKind::In)
      .addOperand("Pgrad", OperandKind::Out)
      .addOperand("Labelsgrad", OperandKind::Out)
      .autoVerify(VerifyKind::NoVerify);

  //===--------------------------------------------------------------------===//
  //                      Arithmetic
  //===--------------------------------------------------------------------===//

  /// Perform matrix multiplication between the 3d tensors LHS and RHS.
  /// If one of the sizes has a batch size of 1 the matrix is broadcasted.
  BB.newInstr("MatMul")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

  /// Performs batch matrix multiplication between the LHS and RHS. The operands
  /// are a stack of two dimensional matrices. Example: (N, A, Z) x (N, Z, B) =>
  /// (N, A, B).
  BB.newInstr("BatchMatMul")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

  /// Accumulates all of the layers in the batch along the Axis dimension and
  /// produce a tensor that has the same dimensions as the input tensor without
  /// the Axis dimension.
  BB.newInstr("BatchedReduceAdd")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Batch", OperandKind::In)
      .addMember(MemberType::Unsigned, "Axis")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Batch"})
      .autoIRGen();

  /// Calculates minimum of all of the layers in the batch along the axes
  /// dimensions and produce a tensor that has the same dimensions as the input
  /// tensor without the Axes dimension.
  BB.newInstr("BatchedReduceMin")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Batch", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Axes")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Batch"})
      .autoIRGen();

  /// Calculates maximum of all of the layers in the batch along the axes
  /// dimensions and produce a tensor that has the same dimensions as the input
  /// tensor without the Axes dimension.
  BB.newInstr("BatchedReduceMax")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Batch", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Axes")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Batch"})
      .autoIRGen();

  /// Accumulates all of the layers in the batch along the Axis dimension and
  /// produce a tensor that has the same dimensions as the input tensor without
  /// the Axis dimension.
  BB.newInstr("BatchedReduceProd")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Batch", OperandKind::In)
      .addMember(MemberType::Unsigned, "Axis")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Batch"})
      .autoIRGen();

  // Does a running accumulation of all values in input (inclusive).
  // e.g [1, 2, 3, 4] -> [1, 3, 6, 10]
  BB.newInstr("CumSum")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Input", OperandKind::In)
      .addMember(MemberType::Unsigned, "Exclusive")
      .addMember(MemberType::Unsigned, "Reverse")
      .inplaceOperand({"Dest", "Input"})
      .autoIRGen()
      .autoVerify(VerifyKind::SameType, {"Dest", "Input"});

  /// Sums together groups of consecutive slices of Data as per the group sizes
  /// specified by Lengths.
  BB.newInstr("LengthsSum")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Lengths", OperandKind::In)
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Data"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Lengths", "ElemKind::Int32ITy"});

  BB.newInstr("SparseLengthsSum")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .addOperand("Lengths", OperandKind::In)
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Data"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Lengths", "ElemKind::Int32ITy"})
      .addGradientInstr({"Data", "Indices", "Lengths"}, {"Dest", "Data"});

  BB.newInstr("SparseLengthsWeightedSum")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Weights", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .addOperand("Lengths", OperandKind::In)
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Data", "Weights"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Lengths", "ElemKind::Int32ITy"})
      .autoVerify(VerifyKind::SameShape, {"Weights", "Indices"})
      .addGradientInstr({"Data", "Weights", "Indices", "Lengths"},
                        {"Dest", "Data", "Weights"});

  BB.newInstr("Embedding")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Weights", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .addMember(MemberType::Int64, "PadIdx")
      .addMember(MemberType::Boolean, "Scale")
      .addMember(MemberType::Boolean, "Sparse")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Weights"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Indices", "ElemKind::Int64ITy"});

  BB.newInstr("EmbeddingBag")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Weights", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .addOperand("Offsets", OperandKind::In)
      .addMember(MemberType::Boolean, "HasEndOffset")
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Data", "Weights"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Indices", "ElemKind::Int64ITy"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Offsets", "ElemKind::Int64ITy"})
      .autoVerify(VerifyKind::SameShape, {"Weights", "Indices"});

  BB.newInstr("RowwiseQuantizedSparseLengthsWeightedSum")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Scales", OperandKind::In)
      .addOperand("Offsets", OperandKind::In)
      .addOperand("Weights", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .addOperand("Lengths", OperandKind::In)
      .addMember(MemberType::Boolean, "UseFP16Accumulation")
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Data", "ElemKind::UInt8QTy"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Lengths", "ElemKind::Int32ITy"})
      .autoVerify(VerifyKind::SameShape, {"Weights", "Indices"});

  BB.newInstr("FusedRowwiseQuantizedSparseLengthsWeightedSum")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Weights", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .addOperand("Lengths", OperandKind::In)
      .addMember(MemberType::Boolean, "UseFP16Accumulation")
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType,
                  {"Lengths", "ElemKind::Int32ITy"})
      .autoVerify(VerifyKind::SameShape, {"Weights", "Indices"});

  BB.newInstr("EmbeddingBagByteRowwiseOffsets")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Weights", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .addOperand("Offsets", OperandKind::In)
      .addMember(MemberType::Boolean, "UseFP16Accumulation")
      .addMember(MemberType::Boolean, "HasEndOffset")
      .addMember(MEMBER_TYPE_INFO(glow::LengthsMode), "LengthsMode")
      .addMember(MemberType::Float, "AvgLength")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType,
                  {"Indices", "ElemKind::Int64ITy"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Offsets", "ElemKind::Int64ITy"})
      .autoVerify(VerifyKind::SameShape, {"Weights", "Indices"});

  BB.newInstr("LengthsToRanges")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Lengths", OperandKind::In)
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::Int32ITy"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Lengths", "ElemKind::Int32ITy"});

  BB.newInstr("LengthsRangeFill")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Lengths", OperandKind::In)
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::Int32ITy"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Lengths", "ElemKind::Int32ITy"});

  /// Converts the given sparse representation into a dense one.
  BB.newInstr("SparseToDense")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Indices", OperandKind::In)
      .addOperand("Values", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Values"})
      .autoIRGen();

  BB.newInstr("SparseToDenseMask")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Indices", OperandKind::In)
      .addOperand("Values", OperandKind::In)
      .addOperand("DefaultValue", OperandKind::In)
      .addOperand("Lengths", OperandKind::In)
      .addMember(MemberType::VectorDimT, "Mask")
      .autoVerify(VerifyKind::SameElementType,
                  {"Dest", "Values", "DefaultValue"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Indices", "ElemKind::Int64ITy"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Lengths", "ElemKind::Int32ITy"})
      .autoIRGen();

  /// Adds the 'Slice' operand to each one of the slices in the batch.
  BB.newInstr("BatchedAdd")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Batch", OperandKind::In)
      .addOperand("Slice", OperandKind::In)
      .inplaceOperand({"Dest", "Batch"})
      .autoVerify(VerifyKind::SameShape, {"Batch", "Dest"})
      .autoIRGen();

  BB.newInstr("ElementAdd")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoIRGen("Add");

  BB.newInstr("ElementSub")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoIRGen("Sub");

  BB.newInstr("ElementMul")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoIRGen("Mul");

  BB.newInstr("ElementDiv")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoIRGen("Div");

  BB.newInstr("ElementMax")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoIRGen("Max");

  BB.newInstr("ElementMin")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoIRGen("Min");

  BB.newInstr("ElementCmpEQ")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::BoolTy"})
      .autoIRGen("CmpEQ");

  BB.newInstr("ElementCmpNEQ")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::BoolTy"})
      .autoIRGen("CmpNEQ");

  BB.newInstr("ElementCmpLT")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::BoolTy"})
      .autoIRGen("CmpLT");

  BB.newInstr("ElementCmpLTE")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::BoolTy"})
      .autoIRGen("CmpLTE");

  BB.newInstr("ElementIsNaN")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::BoolTy"})
      .autoIRGen("IsNaN");

  BB.newInstr("ElementPow")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoIRGen("Pow");

  BB.newInstr("ElementAnd")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"LHS", "ElemKind::BoolTy"})
      .autoVerify(VerifyKind::SameElementType, {"RHS", "ElemKind::BoolTy"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::BoolTy"})
      .autoIRGen("And");

  BB.newInstr("ElementOr")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"LHS", "ElemKind::BoolTy"})
      .autoVerify(VerifyKind::SameElementType, {"RHS", "ElemKind::BoolTy"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::BoolTy"})
      .autoIRGen("Or");

  BB.newInstr("ElementXor")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"LHS", "ElemKind::BoolTy"})
      .autoVerify(VerifyKind::SameElementType, {"RHS", "ElemKind::BoolTy"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::BoolTy"})
      .autoIRGen("Xor");

  BB.newInstr("ElementNot")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Src", "ElemKind::BoolTy"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::BoolTy"})
      .autoIRGen("Not");

  BB.newInstr("ElementNeg")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Neg");

  BB.newInstr("ElementAbs")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Abs");

  BB.newInstr("ElementFloor")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Floor");

  BB.newInstr("ElementSign")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Sign");

  BB.newInstr("ElementCeil")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Ceil");

  BB.newInstr("ElementTruncate")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Truncate");

  BB.newInstr("ElementRound")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Round");

  BB.newInstr("ElementSqrt")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Sqrt");

  BB.newInstr("ElementRsqrt")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Rsqrt");

  BB.newInstr("ElementReciprocal")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Reciprocal");

  BB.newInstr("ElementSin")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Sin");

  BB.newInstr("ElementCos")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Cos");

  BB.newInstr("ElementLog")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({
          "Dest",
          "Src",
      })
      .dataParallel()
      .autoVerify(VerifyKind::SameType, {"Dest", "Src"})
      .autoIRGen("Log");

  BB.newInstr("ElementExp")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({
          "Dest",
          "Src",
      })
      .dataParallel()
      .autoVerify(VerifyKind::SameType, {"Dest", "Src"})
      .autoIRGen("Exp");

  BB.newInstr("ElementAcos")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Acos");

  BB.newInstr("ElementAsin")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Asin");

  BB.newInstr("ElementAtan")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Atan");

  BB.newInstr("ElementErf")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen("Erf");

  BB.newInstr("ElementSelect")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Cond", OperandKind::In)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS", "Cond"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS", "Cond"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"})
      .autoVerify(VerifyKind::SameElementType, {"Cond", "ElemKind::BoolTy"})
      .autoIRGen("Select");

  BB.newInstr("Modulo")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::Int64, "Divisor")
      .addMember(MemberType::Boolean, "SignFollowDivisor")
      .inplaceOperand({
          "Dest",
          "Src",
      })
      .dataParallel()
      .autoVerify(VerifyKind::SameType, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("BatchedPairwiseDotProduct")
      .addOperand("Dest", OperandKind::Out)
      .addMember(MemberType::Unsigned, "NumInputs")
      .addMember(MemberType::Unsigned, "VectorSize")
      .autoVerify(VerifyKind::NoVerify);

  BB.newInstr("BatchedPairwiseDotProductGrad")
      .addOperand("DestGrad", OperandKind::In)
      .addMember(MemberType::Unsigned, "NumInputs")
      .addMember(MemberType::Unsigned, "VectorSize")
      .autoVerify(VerifyKind::NoVerify);

  //===--------------------------------------------------------------------===//
  //                Non-linearities
  //===--------------------------------------------------------------------===//
  BB.newInstr("Relu")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({
          "Dest",
          "Src",
      })
      .dataParallel()
      .autoIRGen()
      .addGradientInstr({"Dest"}, {"Dest", "Src"});

  BB.newInstr("Clip")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::Float, "Min")
      .addMember(MemberType::Float, "Max")
      .inplaceOperand({
          "Dest",
          "Src",
      })
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("Sigmoid")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({
          "Dest",
          "Src",
      })
      .dataParallel()
      .autoVerify(VerifyKind::SameType, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("Tanh")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({
          "Dest",
          "Src",
      })
      .dataParallel()
      .autoVerify(VerifyKind::SameType, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("LeakyRelu")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::Float, "Alpha")
      .inplaceOperand({
          "Dest",
          "Src",
      })
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen();

  //===--------------------------------------------------------------------===//
  //                Shape transformations
  //===--------------------------------------------------------------------===//

  BB.newInstr("Transpose")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Shuffle")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("Splat")
      .addMember(MemberType::Float, "Value")
      .addOperand("Dest", OperandKind::Out)
      .dataParallel()
      .autoVerify(VerifyKind::NoVerify)
      .autoIRGen();

  BB.newInstr("Touch")
      .addOperand("Dest", OperandKind::Out)
      .dataParallel()
      .autoVerify(VerifyKind::NoVerify)
      .autoIRGen();

  BB.newInstr("InsertTensor")
      .addOperand("Dest", OperandKind::InOut)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorDimT, "Offsets")
      .addMember(MemberType::Unsigned, "Count")
      .addMember(MemberType::Unsigned, "Axis");

  BB.newInstr("ExtractTensor")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorDimT, "Offsets");

  BB.newInstr("Gather")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .addMember(MemberType::Unsigned, "BatchDims")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Data"})
      .autoIRGen();

  BB.newInstr("GatherND")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Data"})
      .autoIRGen();

  BB.newInstr("GatherRanges")
      .addOperand("Output", OperandKind::Out)
      .addOperand("Lengths", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Ranges", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Data", "Output"})
      .autoVerify(VerifyKind::SameElementType, {"Ranges", "Lengths"})
      .autoIRGen();

  BB.newInstr("ScatterData")
      .addOperand("Data", OperandKind::InOut)
      .addOperand("Indices", OperandKind::In)
      .addOperand("Slices", OperandKind::In)
      .addMember(MemberType::Boolean, "Cumulative")
      .autoVerify(VerifyKind::NoVerify);

  BB.newInstr("BatchOneHot")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Lengths", OperandKind::In)
      .addOperand("Values", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Values", "Data", "Dest"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Lengths", "ElemKind::Int32ITy"})
      .autoIRGen();

  BB.newInstr("SpaceToDepth")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::Unsigned, "BlockSize")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("ResizeNearest")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorFloat, "Scale")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("ResizeBilinear")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorFloat, "Scale")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen();

  //===--------------------------------------------------------------------===//
  //                Reorder transformations
  //===--------------------------------------------------------------------===//

  BB.newInstr("Flip")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::Unsigned, "Axis")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoIRGen();

  //===--------------------------------------------------------------------===//
  //             Instructions used for debugging/profiling/printing
  //===--------------------------------------------------------------------===//

  BB.newInstr("DebugPrint")
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::String, "Format")
      .addMember(MemberType::String, "FileName")
      .autoVerify(VerifyKind::NoVerify);

  BB.newInstr("TraceEvent")
      .addOperand("Data", OperandKind::In)
      .addMember(MemberType::Unsigned, "Index")
      .autoVerify(VerifyKind::NoVerify);

  /// Instruction used to instrument other instructions. InstrRef is a reference
  /// of the instruction being instrumented, ID is a unique identifier assigned
  /// to the instrumented instruction and InstrumentKind is the instrumentation
  /// kind/type. OperandsInfo is a temporary buffer used to store the addresses
  /// and the sizes of the operands for the instrumented instruction.
  BB.newInstr("Instrument")
      .addOperand("OperandsInfo", OperandKind::Out)
      .addMember(MEMBER_TYPE_INFO(glow::Instruction *), "InstrRef")
      .addMember(MemberType::Unsigned, "ID")
      .addMember(MEMBER_TYPE_INFO(glow::InstrumentKind), "InstrumentKind")
      .autoVerify(VerifyKind::NoVerify);

  //===--------------------------------------------------------------------===//
  //             Instructions used for quantization
  //===--------------------------------------------------------------------===//

  BB.newInstr("QuantizationProfile")
      .addOperand("InputTensor", OperandKind::In)
      .addOperand("Histogram", OperandKind::InOut)
      .addOperand("ComputationInfo", OperandKind::InOut)
      .autoVerify(VerifyKind::SameElementType,
                  {"InputTensor", "ElemKind::FloatTy"});

  BB.newInstr("IntLookupTable")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Mapping", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoVerify(VerifyKind::TypeCheck, {"Dest", "isQuantizedType()"})
      .dataParallel()
      .autoIRGen();

  BB.newInstr("Quantize")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .autoVerify(VerifyKind::TypeCheck, {"Src", "isFPType()"})
      .autoVerify(VerifyKind::TypeCheck, {"Dest", "isQuantizedType()"})
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .dataParallel()
      .autoIRGen();

  BB.newInstr("Dequantize")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .autoVerify(VerifyKind::TypeCheck, {"Dest", "isFPType()"})
      .autoVerify(VerifyKind::TypeCheck, {"Src", "isQuantizedType()"})
      .dataParallel()
      .autoIRGen();

  BB.newInstr("RescaleQuantized")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoVerify(VerifyKind::TypeCheck, {"Dest", "isQuantizedType()"})
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .dataParallel()
      .autoIRGen();

  //===--------------------------------------------------------------------===//
  //                Instructions used by RNN
  //===--------------------------------------------------------------------===//

  BB.newInstr("TopK")
      .addOperand("Values", OperandKind::Out)
      .addOperand("Indices", OperandKind::Out)
      .addOperand("Input", OperandKind::In)
      .addOperand("Scratch", OperandKind::Scratch)
      .addMember(MemberType::Unsigned, "K")
      .autoVerify(VerifyKind::SameElementType, {"Values", "Input"})
      .autoVerify(VerifyKind::SameShape, {"Values", "Indices"});

  //===--------------------------------------------------------------------===//
  //                   Conversions
  //===--------------------------------------------------------------------===//

  BB.newInstr("ConvertTo")
      .addOperand("Result", OperandKind::Out)
      .addOperand("Input", OperandKind::In)
      .autoVerify(VerifyKind::NoVerify)
      .autoIRGen();

  //===--------------------------------------------------------------------===//
  //                Custom kernels invocations
  //===--------------------------------------------------------------------===//
  BB.newInstr("ExternalFunctionCall")
      .addOperand("Dest", OperandKind::Out)
      .addMember(MemberType::String, "FunctionName")
      .addMember(MemberType::String, "FunctionImpl")
      .addMember(MemberType::String, "FunctionKind")
      .autoVerify(VerifyKind::NoVerify);

  //===--------------------------------------------------------------------===//
  //                Pre Processing
  //===--------------------------------------------------------------------===//

  BB.newInstr("AudioSpectrogram")
      .addOperand("Spectrogram", OperandKind::Out)
      .addOperand("Input", OperandKind::In)
      .addOperand("Window", OperandKind::In)
      .addOperand("TwiddleFactors", OperandKind::In)
      .addOperand("BitReverseIndices", OperandKind::In)
      .addOperand("ComplexToRealWeights", OperandKind::In)
      .addOperand("WinOutScratch", OperandKind::Scratch)
      .addOperand("FftOutScratch", OperandKind::Scratch)
      .addMember(MemberType::Int64, "WindowSize")
      .addMember(MemberType::Int64, "WindowStride")
      .addMember(MemberType::Boolean, "MagnitudeSquared")
      .autoVerify(VerifyKind::SameElementType,
                  {"Spectrogram", "Input", "Window", "TwiddleFactors",
                   "ComplexToRealWeights", "ElemKind::FloatTy"})
      .autoVerify(VerifyKind::SameElementType,
                  {"BitReverseIndices", "ElemKind::Int32ITy"})
      .autoIRGen();

  BB.newInstr("MFCC")
      .addOperand("Coefficients", OperandKind::Out)
      .addOperand("Spectrogram", OperandKind::In)
      .addOperand("MelWeights", OperandKind::In)
      .addOperand("MelRanges", OperandKind::In)
      .addOperand("DctMat", OperandKind::In)
      .addOperand("Scratch", OperandKind::Scratch)
      .addMember(MemberType::Float, "SampleRate")
      .addMember(MemberType::Float, "LowerFrequency")
      .addMember(MemberType::Float, "UpperFrequency")
      .addMember(MemberType::Int64, "FilterBankCount")
      .addMember(MemberType::Int64, "NumCoefficients")
      .autoVerify(VerifyKind::SameElementType,
                  {"Coefficients", "Spectrogram", "MelWeights", "DctMat",
                   "ElemKind::FloatTy"})
      .autoVerify(VerifyKind::SameElementType,
                  {"MelRanges", "ElemKind::Int32ITy"})
      .autoIRGen();

  //===--------------------------------------------------------------------===//
  //                Post Processing
  //===--------------------------------------------------------------------===//

  BB.newInstr("NonMaxSuppression")
      .addOperand("Indices", OperandKind::Out)
      .addOperand("NumberOfSelectedIndices", OperandKind::Out)
      .addOperand("Boxes", OperandKind::In)
      .addOperand("Scores", OperandKind::In)
      .addMember(MemberType::Int64, "CenterPointBox")
      .addMember(MemberType::Int64, "MaxOutputBoxesPerClass")
      .addMember(MemberType::Float, "IouThreshold")
      .addMember(MemberType::Float, "ScoreThreshold")
      .addMember(MemberType::Boolean, "IsTFVersion4")
      .autoVerify(VerifyKind::SameElementType, {"Boxes", "Scores"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Indices", "NumberOfSelectedIndices"})
      .autoIRGen();

  //===--------------------------------------------------------------------===//
  //                Region of Interest (ROI)
  //===--------------------------------------------------------------------===//
  BB.newInstr("ROIAlign")
      .addOperand("Result", OperandKind::Out)
      .addOperand("FeatureMap", OperandKind::In)
      .addOperand("Boxes", OperandKind::In)
      .addOperand("BatchIndices", OperandKind::In)
      .addMember(MemberType::Enum, "Mode")
      .addMember(MemberType::Unsigned, "OutputHeight")
      .addMember(MemberType::Unsigned, "OutputWidth")
      .addMember(MemberType::Unsigned, "SamplingRatio")
      .addMember(MemberType::Float, "SpatialScale")
      .addMember(MemberType::Boolean, "Aligned")
      .addMember(MemberType::Boolean, "Rotated")
      .autoVerify(VerifyKind::SameElementType, {"FeatureMap", "Boxes"})
      .autoIRGen();

  BB.newInstr("BBoxTransform")
      .addOperand("BoxOut", OperandKind::Out)
      .addOperand("RoiBatchSplits", OperandKind::Out)
      .addOperand("Rois", OperandKind::In)
      .addOperand("Deltas", OperandKind::In)
      .addOperand("ImInfo", OperandKind::In)
      .addMember(MemberType::VectorFloat, "Weights")
      .addMember(MemberType::Boolean, "ApplyScale")
      .addMember(MemberType::Boolean, "Rotated")
      .addMember(MemberType::Boolean, "AngleBoundOn")
      .addMember(MemberType::Int64, "AngleBoundLo")
      .addMember(MemberType::Int64, "AngleBoundHi")
      .addMember(MemberType::Float, "ClipAngleThresh")
      .addMember(MemberType::Boolean, "LegacyPlusOne")
      .autoVerify(VerifyKind::SameElementType, {"Rois", "Deltas"})
      .autoVerify(VerifyKind::SameElementType, {"Rois", "ImInfo"})
      .autoIRGen();

  //===--------------------------------------------------------------------===//
  //                Backend-Specific Instructions
  //===--------------------------------------------------------------------===//

#include "glow/InstrGenIncludes.h"

  return 0;
}
