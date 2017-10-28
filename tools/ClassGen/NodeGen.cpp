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
      .addMember("size_t", "Kernel")
      .addMember("size_t", "Stride")
      .addMember("size_t", "Pad")
      .addMember("size_t", "Depth")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy");

  BB.newNode("Pool")
      .addEnumCase("Max")
      .addEnumCase("Avg")
      .addOperand("Input")
      .addMember("size_t", "Kernel")
      .addMember("size_t", "Stride")
      .addMember("size_t", "Pad")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy");

  BB.newNode("FullyConnected")
      .addOperand("Input")
      .addOperand("Filter")
      .addOperand("Bias")
      .addMember("size_t", "Depth")
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
      .addMember("size_t", "ChannelIdx")
      .addMember("float", "Epsilon")
      .addMember("float", "Momentum")
      .setType("Input->getType()");

  BB.newNode("LocalResponseNormalization")
      .addOperand("Input")
      .addOperand("Scale")
      .addMember("size_t", "HalfWindowSize")
      .addMember("float", "Alpha")
      .addMember("float", "Beta")
      .addMember("float", "K")
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
      .addMember("std::vector<size_t>", "Dims")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy")
      .overrideGetter(
          "Dims", "llvm::ArrayRef<size_t> getDims() const { return Dims_; }");

  BB.newNode("Transpose")
      .addOperand("Input")
      .addMember("std::vector<unsigned>", "Shuffle")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy")
      .overrideGetter(
          "Shuffle",
          "llvm::ArrayRef<unsigned> getShuffle() const { return Shuffle_; }");

  BB.newNode("Concat")
      .addOperand("LHS")
      .addOperand("RHS")
      .addMember("size_t", "Dim")
      .addExtraParam("TypeRef", "outTy")
      .setType("outTy");

  return 0;
}
