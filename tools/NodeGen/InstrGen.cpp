#include "InstrBuilder.h"

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " output.h output.cpp output.def\n";
    return -1;
  }

  std::cout << "Writing instr descriptors to:\n\t" << argv[1] << "\n\t"
            << argv[2] << "\n\t" << argv[3] << "\n";

  std::ofstream hFile(argv[1]);
  std::ofstream cFile(argv[2]);
  std::ofstream dFile(argv[3]);

  Builder BB(hFile, cFile, dFile);

  BB.newInstr("AllocActivation").addExtraParam("TypeRef", "Ty").setType("Ty");

  BB.newInstr("DeallocActivation")
      .addOperand("Src", OperandKind::Out)
      .overrideGetter("Src", "AllocActivationInst *getAlloc() const { return "
                             "cast<AllocActivationInst>(getOperand(0).first); "
                             "}")
      .setType("Src->getType()");

  BB.newInstr("Copy")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .setType("Src->getType()");

  BB.newInstr("Convolution")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Filter", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .addMember("size_t", "Kernel")
      .addMember("size_t", "Stride")
      .addMember("size_t", "Pad")
      .addMember("size_t", "Depth")
      .addExtraMethod("bool mayShareBuffers() const { return false; }")
      .setType("Dest->getType()");

  BB.newInstr("Pool")
      .addEnumCase("Max")
      .addEnumCase("Avg")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("SrcXY", OperandKind::InOut)
      .addMember("size_t", "Kernel")
      .addMember("size_t", "Stride")
      .addMember("size_t", "Pad")
      .addExtraMethod("bool mayShareBuffers() const { return false; }")
      .setType("Dest->getType()");

  BB.newInstr("FullyConnected")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Filter", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .addMember("size_t", "Depth")
      .addExtraMethod("bool mayShareBuffers() const { return false; }")
      .setType("Dest->getType()");

  BB.newInstr("Relu")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .setType("Dest->getType()");

  BB.newInstr("Sigmoid")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .setType("Dest->getType()");

  BB.newInstr("Tanh")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .setType("Dest->getType()");

  BB.newInstr("SoftMax")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("E", OperandKind::InOut)
      .addOperand("Selected", OperandKind::InOut)
      .setType("Dest->getType()");

  BB.newInstr("Regression")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Expected", OperandKind::InOut)
      .setType("Dest->getType()");

  BB.newInstr("Reshape")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember("std::vector<size_t>", "Dims")
      .setType("Dest->getType()")
      .overrideGetter(
          "Dims", "llvm::ArrayRef<size_t> getDims() const { return Dims_; }");

  BB.newInstr("Transpose")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember("std::vector<unsigned>", "Shuffle")
      .setType("Dest->getType()")
      .overrideGetter(
          "Shuffle",
          "llvm::ArrayRef<unsigned> getShuffle() const { return Shuffle_; }");

  BB.newInstr("BatchNormalization")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Scale", OperandKind::In)
      .addOperand("Bias", OperandKind::In)
      .addOperand("Mean", OperandKind::In)
      .addOperand("Var", OperandKind::In)
      .addMember("size_t", "ChannelIdx")
      .addMember("float", "Epsilon")
      .addMember("float", "Momentum")
      .setType("Src->getType()");

  BB.newInstr("LocalResponseNormalization")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("Scale", OperandKind::In)
      .addMember("size_t", "HalfWindowSize")
      .addMember("float", "Alpha")
      .addMember("float", "Beta")
      .addMember("float", "K")
      .setType("Src->getType()");

  BB.newInstr("Arithmetic")
      .addEnumCase("Add")
      .addEnumCase("Mul")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .setType("LHS->getType()");

  return 0;
}
