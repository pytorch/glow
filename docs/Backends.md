## Backend-Specific Functionality

Different backends may prefer to transform or optimize the graph differently for
their own specialized architecture. For example, Glow lowers ReLU down to a Max
node, taking as inputs the original tensor and a "Splat" tensor of matching
dimensions, filled with all `0`s. Glow's CPU JIT backend prefers to replace this
pattern -- a Max with a Splat input and another non-Splat input -- with a single
"CPUMaxSplat" operation that takes a scalar Splat value as input in place of an
entire Splat tensor.

### Backend-Specific Transformation

Backends have the opportunity to perform their own analysis and transformations
before or after lowering depending on their requirements. This is exposed via
`transformPreLowering()` and `transformPostLowering()` hooks, during which a
backend can transform the graph however it desires. For example, the backend
could use `transformPostLowering()` to search the graph looking for the above
`CPUMaxSplat` pattern.

#### Backend-Specific Nodes and Instructions

A backend may create its own custom Nodes and Instructions which it can insert
into the IR. This is done via [ClassGen](ClassGen.md) and included in
`tools/ClassGen/NodeGen.cpp`. For example, the CPU Backend defines `CPUMaxSplat`
in `tools/ClassGen/Backends/CPU/CPUSpecificNodes.h`:

```cpp
BB.newBackendSpecificNode("CPUMaxSplat")
    .addInput("Input")
    .addResult("Input.getType()")
    .addMember(MemberType::Float, "SplatValue")
    .setDocstring("A Max node with one splat input; CPU specific.");
```

During `transformPostLowering()`, this `CPUMaxSplat` node replaces the
aforementioned pattern. However, there must be a corresponding instruction for
this Node to be lowered to during the IRGen phase. Thus, we need a corresponding
backend-specific CPUMaxSplat instruction, defined in
`tools/ClassGen/Backends/CPU/CPUSpecificInstrs.h`:

```
BB.newBackendSpecificInstr("CPUMaxSplat")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "SplatValue")
    .inplaceOperand({"Dest", "Src"})
    .dataParallel()
    .autoIRGen();
```

These instructions will appear in the instruction stream sent to the CPU backend
JIT; its [standard library](JIT.md#usage-of-the-standard-library) has a kernel
for executing this `CPUMaxSplat` instruction. You can see such instructions in
the [LeNet MNIST example](Example.md#lowering-to-ir).

Note that backend-specific nodes and instructions can be treated just as any
other node or instruction defined in `tools/ClassGen/NodeGen.cpp` or
`tools/ClassGen/InstrGen.cpp`. For example, the `CPUMaxSplat` instruction
definition includes the `dataParallel()` property, allowing for data parallel
optimizations to take place.

The `tools/ClassGen/Backends/CPU/CPUSpecificNodes.h` and
`tools/ClassGen/Backends/CPU/CPUSpecificInstrs.h` files are included in
`tools/ClassGen/NodeGen.cpp` and `tools/ClassGen/InstrGen.cpp`, respectively.
