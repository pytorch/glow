
## Automatic class generation

This document describes the automatic code generation techniques (class-gen)
that Glow uses for defining instructions and nodes. Glow's class-gen is inspired
by LLVM's TableGen tools.

### Introduction

The purpose of the automatic code generation tools in glow is identical to the
motivation begind TableGen:

" ... to help a human develop and maintain records of domain-specific
information. Because there may be a large number of these records, it is
specifically designed to allow writing flexible descriptions and for common
features of these records to be factored out. This reduces the amount of
duplication in the description, reduces the chance of error, and makes it easier
to structure domain specific information. "

The current system is capable of generating two kinds of classes: Nodes and
Instructions. These structures are declared in the IR document. Here is a short
example of the code for generating the SoftMax instruction. This node generates
the SoftMax instruction as well as the gradient calculation instruction
SoftMaxGrad. The different methods of the builder (described in the builder
doxygen comments) construct the different kinds of fields that the Instruction
has.

  ```
  BB.newInstr("SoftMax")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addOperand("E", OperandKind::InOut)
      .addOperand("Selected", OperandKind::InOut)
      .inplaceOperand({"Dest", "Src"})
      .addGradientInstr({"Src", "E", "Selected"}, {"Dest", "Src"});
  ```

