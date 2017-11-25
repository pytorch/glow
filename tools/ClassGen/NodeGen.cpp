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

  BB.newNode("Save").addInput("Input").addInput("Output").addExtraMethod(
      "Variable *getVariable() const { return "
      "llvm::cast<Variable>(Output_.getNode()); };");
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
      .addInput("Scale")
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
      .addInput("LHS")
      .addInput("RHS")
      .addResult("LHS.getType()")
      .addGradient();

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

  //===--------------------------------------------------------------------===//
  //                Nodes used for network training
  //===--------------------------------------------------------------------===//

  BB.newNode("Zero")
      .addExtraParam("TypeRef", "outTy")
      .addResult("outTy")
      .setDocstring("Generate the zero tensor of a specific type");

  BB.newNode("SGD")
      .addInput("Gradient")
      .addInput("Weight")
      .addMember(MemberType::Float, "L1Decay")
      .addMember(MemberType::Float, "L2Decay")
      .addMember(MemberType::Float, "LearningRate")
      .addMember(MemberType::Float, "Momentum");

  //===--------------------------------------------------------------------===//
  //                Nodes used by unit tests.
  //===--------------------------------------------------------------------===//

  /// This is a test node that's used by the node unittests.
  BB.newNode("Distribute")
      .addInput("Input")
      .addResult("Input.getType()", "Left")
      .addResult("Input.getType()", "Right");

  return 0;
}
