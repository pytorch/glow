
## Automatic class generation

Glow uses automatic code generation techniques (class-gen) for defining
instructions and nodes. The purpose of the automatic code generation tools in
Glow is similar to the motivation behind LLVM's TableGen, which is to help a
human develop and maintain records of domain-specific information.

### Introduction

The purpose of the automatic code generation tools in Glow is similar to the
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
  BB.newInstr("AvgPool")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In)
      .addMember(MemberType::VectorUnsigned, "Kernels")
      .addMember(MemberType::VectorUnsigned, "Strides")
      .addMember(MemberType::VectorUnsigned, "Pads")
      .autoIRGen()
      .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
      .addGradientInstr({"Dest"}, {"Dest", "Src"});
  ```

### NodeGen

This tool allows defining nodes (operators at graph level) by inserting
descriptions in the file [NodeGen](https://github.com/pytorch/glow/blob/master/tools/ClassGen/NodeGen.cpp).
Such an example is the `Add` node which is defined like this:

  ```
  BB.newNode("Add")
      .addInput("LHS")
      .addInput("RHS")
      .addResultFromCtorArg()
      .dataParallel()
      .addGradient()
      .setDocstring("Performs Add on the LHS and RHS operands.");
  ```

The above description automatically generates the definition for a new node
in the form of a C++ class named `AddNode` which:
- has two input operands named `LHS` and `RHS`.
- has a single output (result) operand.
- has the `dataParallel()` method invoked which specifies that this node performs
  data parallel computation.
- has the `addGradient()` method invoked which defines a new node (class) named
  `AddGradNode` which corresponds to the differentiated node used during training.
- has a small description given with the `setDocstring()` primitive.

The `NodeGen` tool also creates common methods for the class: constructor, getters/setters for
its members, hash/clone functions, etc.

### InstrGen

This tool allows defining instructions (operators at IR level which is the level
right below the graph level) by inserting descriptions in the file [InstrGen](https://github.com/pytorch/glow/blob/master/tools/ClassGen/InstrGen.cpp).
Such an example is the `ElementAdd` instruction which is defined like this:

  ```
  BB.newInstr("ElementAdd")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("LHS", OperandKind::In)
      .addOperand("RHS", OperandKind::In)
      .inplaceOperand({"Dest", "LHS", "RHS"})
      .dataParallel()
      .autoVerify(VerifyKind::SameShape, {"Dest", "LHS", "RHS"})
      .autoIRGen("Add");
  ```

The above description automatically generates the definition for a new instruction
in the form of a C++ class named `ElementAddInst` which:
- has one output operand named `Dest`.
- has two input operands named `LHS` and `RHS`.
- has the `inplaceOperand()` method invoked which specifies that from the point of view
  of the implementation the memory buffer corresponding to the output operand `Dest`
  can reuse the memory buffers corresponding to the input operands (for in-place computation).
- has the `dataParallel()` method invoked which specifies that this instruction performs
  data parallel computation.
- has some compile-time verifications inserted with the `autoVerify()` method which in this
  case verifies that all 3 operands have same shape.
- the `autoIRGen("Add")` method specifies that during compile-time (more specifically during
  the IRGen compile phase) the `ElementAdd` instruction will be created from an `Add` node.
  The link between `ElementAddInst` and `AddNode` is created automatically by this tool. This
  only works if there is a 1:1 mapping in terms of operands/members between the two classes.

The `InstrGen` tool also creates common methods for the class: constructor, getters/setters for
its members, hash/clone functions, etc.

Another feature which the `InstrGen` has is to define `Scratch` operands for the instruction.
A `Scratch` operand type is one which provides a temporary memory buffer for the instruction to
write some intermediate computations before writing the final results using its output operands.
The `Scratch` operand type only exists at IR level. An example of an instruction which requires
scratch memory is the `TopK` operator which is defined as:

  ```
  BB.newInstr("TopK")
      .addOperand("Values", OperandKind::Out)
      .addOperand("Indices", OperandKind::Out)
      .addOperand("Input", OperandKind::In)
      .addOperand("Scratch", OperandKind::Scratch)
      .addMember(MemberType::Unsigned, "K")
      .autoVerify(VerifyKind::SameElementType, {"Values", "Input"})
      .autoVerify(VerifyKind::SameShape, {"Values", "Indices"});
  ```

What `InstrGen` does in this case is define a `Scratch` operand for the instruction and also declare
automatically a method called `getScratchSize()` which must be implemented by the instruction creator
in order to express the scratch size requirements for this specific instruction. Multiple scratch
operands can be defined for an instruction in which case multiple methods will be emitted. The
`autoIRGen` feature also works for instructions with `Scratch` type operands since `InstrGen` exempts
these operators from being present in the associated Node.

A special case is when the instruction already has a member called `ScratchSize` (for which a getter
`getScratchSize()` is emitted by default) in which case the method is not emitted again. In this
particular case the member will be used for scratch allocation and a new method is not required to be
implemented by the instruction creator.
