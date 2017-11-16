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
      .addOperand("Input")
      .addOperand("Output")
      .setType("Input->getType()")
      .overrideGetter("Output", "Variable *getOutput() const { return "
                                "llvm::cast<Variable>(Output_.get()); };");

  //===--------------------------------------------------------------------===//
  //                   Convolution / Pool / FC
  //===--------------------------------------------------------------------===//

  BB.newNode("Convolution")
      .addOperand("Input")
      .addOperand("Filter")
      .addOperand("Bias")
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::SizeT, "Pad")
      .addMember(MemberType::SizeT, "Depth")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy");

  BB.newNode("Pool")
      .addEnumCase("Max")
      .addEnumCase("Avg")
      .addOperand("Input")
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::SizeT, "Pad")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy");

  BB.newNode("FullyConnected")
      .addOperand("Input")
      .addOperand("Filter")
      .addOperand("Bias")
      .addMember(MemberType::SizeT, "Depth")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy");

  //===--------------------------------------------------------------------===//
  //                     Normalization
  //===--------------------------------------------------------------------===//

  BB.newNode("BatchNormalization")
      .addOperand("Input")
      .addOperand("Scale")
      .addOperand("Bias")
      .addOperand("Mean")
      .addOperand("Var")
      .addMember(MemberType::SizeT, "ChannelIdx")
      .addMember(MemberType::Float, "Epsilon")
      .addMember(MemberType::Float, "Momentum")
      .setType("Input->getType()");

  BB.newNode("LocalResponseNormalization")
      .addOperand("Input")
      .addOperand("Scale")
      .addMember(MemberType::SizeT, "HalfWindowSize")
      .addMember(MemberType::Float, "Alpha")
      .addMember(MemberType::Float, "Beta")
      .addMember(MemberType::Float, "K")
      .setType("Input->getType()");

  //===--------------------------------------------------------------------===//
  //                      Loss operations
  //===--------------------------------------------------------------------===//

  BB.newNode("SoftMax")
      .addOperand("Input")
      .addOperand("Selected")
      .setType("Input->getType()");

  BB.newNode("Regression")
      .addOperand("Input")
      .addOperand("Expected")
      .setType("Input->getType()");

  //===--------------------------------------------------------------------===//
  //                      Arithmetic
  //===--------------------------------------------------------------------===//

  BB.newNode("Arithmetic")
      .addEnumCase("Add")
      .addEnumCase("Mul")
      .addOperand("LHS")
      .addOperand("RHS")
      .setType("LHS->getType()");

  //===--------------------------------------------------------------------===//
  //                Non-linearities
  //===--------------------------------------------------------------------===//

  BB.newNode("Relu").addOperand("Input").setType("Input->getType()");
  BB.newNode("Sigmoid").addOperand("Input").setType("Input->getType()");
  BB.newNode("Tanh").addOperand("Input").setType("Input->getType()");

  //===--------------------------------------------------------------------===//
  //                Shape transformations
  //===--------------------------------------------------------------------===//

  BB.newNode("Reshape")
      .addOperand("Input")
      .addMember(MemberType::VectorSizeT, "Dims")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy");

  BB.newNode("Transpose")
      .addOperand("Input")
      .addMember(MemberType::VectorUnsigned, "Shuffle")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy");

  BB.newNode("Concat")
      .addMember(MemberType::VectorNodeOperand, "Inputs")
      .addMember(MemberType::SizeT, "Dim")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy")
      .setDocstring("The concat operator adds two tensors together.\nThe "
                    "parameter 'dim' specifies the dimension to use when "
                    "joining the tensors.");

  BB.newNode("Slice")
      .addOperand("Input")
      .addMember(MemberType::VectorSizeT, "Start")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy");

  return 0;
}
