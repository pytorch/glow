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

  return 0;
}
