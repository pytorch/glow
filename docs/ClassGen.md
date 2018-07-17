
## Automatic class generation

Glow uses automatic code generation techniques (class-gen) for defining
instructions and nodes. The purpose of the automatic code generation tools in
Glow is similar to the motivation behind LLVM's TableGen, which is to help a
human develop and maintain records of domain-specific information.

### Introduction

The purpose of the automatic code generation tools in glow is identical to the
motivation behind TableGen:

" ... to help a human develop and maintain records of domain-specific
information. Because there may be a large number of these records, it is
specifically designed to allow writing flexible descriptions and for common
features of these records to be factored out. This reduces the amount of
duplication in the description, reduces the chance of error, and makes it easier
to structure domain-specific information. "

The current system is capable of generating two kinds of classes: Nodes for the
high-level IR and Instructions for the low-level IR. Below is an example of the
code for generating the AvgPool instruction. ClassGen generates most of the
methods that instructions need to have, such as instruction equality and
hashing, cloning, printing, verification, etc. The different methods of the
builder (described in the builder Doxygen comments) construct the different
kinds of fields that the Instruction has.

  ```
  BB.newInstr("PoolAvg")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::SizeT, "Kernel")
      .addMember(MemberType::SizeT, "Stride")
      .addMember(MemberType::VectorSizeT, "Pads")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .addGradientInstr({"Dest"}, {"Dest", "Src"});
  ```
