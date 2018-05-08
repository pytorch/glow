/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
      .setType("Ty");

  BB.newInstr("TensorView")
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::TypeRef, "Ty")
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
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::SizeT, "Pad")
      .addMember(MemberType::SizeT, "Group")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType,
                  {"Dest", "Src", "Filter", "Bias"})
      .addGradientInstr({"Src", "Filter"}, {"Dest", "Src", "Filter", "Bias"});

  // PoolMax version caching XY coordinates to speedup gradient-based
  // computations.
  BB.newInstr("PoolMaxWithXY")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("SrcXY", OperandKind::Out)
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::SizeT, "Pad")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .addGradientInstr({"Dest", "SrcXY"}, {"Dest", "Src"});

  BB.newInstr("PoolMax")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::SizeT, "Pad")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

  BB.newInstr("PoolAvg")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::SizeT, "Pad")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .addGradientInstr({"Dest"}, {"Dest", "Src"});

  //===--------------------------------------------------------------------===//
  //                     Normalization
  //===--------------------------------------------------------------------===//

  BB.newInstr("LocalResponseNormalization")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Scale", OperandKind::Out)
      .addMember(MemberType::SizeT, "HalfWindowSize")
      .addMember(MemberType::Float, "Alpha")
      .addMember(MemberType::Float, "Beta")
      .addMember(MemberType::Float, "K")
      .setType("Src->getType()")
      .inplaceOperand({
          "Dest",
          "Src",
      })
      .autoVerify(VerifyKind::SameType, {"Dest", "Src", "Scale"})
      .addGradientInstr({"Dest", "Src", "Scale"}, {"Dest", "Src"});

  //===--------------------------------------------------------------------===//
  //                      Loss functions
  //===--------------------------------------------------------------------===//

  BB.newInstr("SoftMax")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .inplaceOperand({"Dest", "Src"})
      .autoVerify(VerifyKind::SameType, {"Dest", "Src"})
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

  /// Accumulates all of the layers in the batch and produce a tensor that has
  /// the same dimensions as the input tensor without the first dimension.
  BB.newInstr("BatchedReduceAdd")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Batch", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Batch"})
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

  BB.newInstr("ElementCmpLTE")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoIRGen("CmpLTE");

  BB.newInstr("ElementPow")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Base", OperandKind::In)
      .addMember(MemberType::Float, "Exp")
      .inplaceOperand({"Dest", "Base"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Base"})
      .autoIRGen("Pow");

  BB.newInstr("ElementSelect")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Cond", OperandKind::In)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS", "Cond"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "Cond", "LHS", "RHS"})
      .autoIRGen("Select");

  //===--------------------------------------------------------------------===//
  //                Non-linearities
  //===--------------------------------------------------------------------===//

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

  //===--------------------------------------------------------------------===//
  //                Shape transformations
  //===--------------------------------------------------------------------===//

  BB.newInstr("Transpose")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Shuffle")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("Broadcast")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorSizeT, "Shape")
      .addMember(MemberType::Unsigned, "Axis")
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("Splat")
      .addMember(MemberType::Float, "Value")
      .addOperand("Dest", OperandKind::Out)
      .dataParallel()
      .autoVerify(VerifyKind::NoVerify)
      .autoIRGen();

  BB.newInstr("InsertTensor")
      .addOperand("Dest", OperandKind::InOut)
      .addOperand("Src", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .addMember(MemberType::VectorSizeT, "Offsets");

  BB.newInstr("ExtractTensor")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .addMember(MemberType::VectorSizeT, "Offsets");

  BB.newInstr("Gather")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Data", OperandKind::In)
      .addOperand("Indices", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Data"})
      .autoVerify(VerifyKind::SameElementType,
                  {"Indices", "ElemKind::IndexTy"});

  //===--------------------------------------------------------------------===//
  //             Instructions used for debugging/profiling/printing
  //===--------------------------------------------------------------------===//

  BB.newInstr("DebugPrint")
      .addOperand("Src", OperandKind::In)
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

  BB.newInstr("Quantize")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::Int8QTy"})
      .autoVerify(VerifyKind::SameElementType, {"Src", "ElemKind::FloatTy"})
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("Dequantize")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType, {"Dest", "ElemKind::FloatTy"})
      .autoVerify(VerifyKind::SameElementType, {"Src", "ElemKind::Int8QTy"})
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoIRGen();

  BB.newInstr("RescaleQuantized")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .autoVerify(VerifyKind::SameElementType,
                  {"Dest", "Src", "ElemKind::Int8QTy"})
      .autoVerify(VerifyKind::SameShape, {"Dest", "Src"})
      .autoIRGen();

  //===--------------------------------------------------------------------===//
  //                Instructions used by RNN
  //===--------------------------------------------------------------------===//

  BB.newInstr("TopK")
      .addOperand("Values", OperandKind::Out)
      .addOperand("Indices", OperandKind::Out)
      .addOperand("Input", OperandKind::In)
      .addOperand("Scratch", OperandKind::InOut)
      .addMember(MemberType::SizeT, "K")
      .autoVerify(VerifyKind::SameElementType, {"Values", "Input"})
      .autoVerify(VerifyKind::SameShape, {"Values", "Indices"});

  //===--------------------------------------------------------------------===//
  //                Backend-Specific Instructions
  //===--------------------------------------------------------------------===//

#include "Backends/CPU/CPUSpecificInstrs.h"
#include "Backends/OpenCL/OpenCLSpecificInstrs.h"

  return 0;
}
