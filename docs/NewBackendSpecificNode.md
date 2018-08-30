## New Backend-Specific Node


This document describes how to add new backend-specific node to make further optimization on certain backend.

### Graph Optimization

Glow has two levels of IR. The high-level IR is a dataflow node-based graph. There are several strategies to make transformation on high-level IR to achieve better performance on certain backend. Two of these strategies will introduce new backend-specific node.

#### Operator Fusion

There are mainly two advantanges of operator fusion:

- Good locality and less memory access
- Reduce kernel launch time in GPU and specialized accelerators

In TensorRT, the convolution, bias and ReLU layers of various sizes can be combined into a single kernel called CBR. A simple analogy is making three separate trips to the supermarket to buy three items with close relation versus buying all three in a single trip, which reduce the time of seeking items and starting trips.

#### Data Layout Transformation

Actually, a Tensor is a view of a block of memory. Besides a pointer to the memory, we also have to get some other descriptions of this block of memory, such as shape, stride and layout.

Different layout leads to different implementation of the operator kernel on certain backend. 
For example, in the subgraph from ResNet50, a `CPUConvDKKC8` node with memory layout modified for efficient SIMD access is introduced to optimized for CPU backend. Please refer to "5.3 Use Case: Optimizing Resnet50 for the CPU" in [Glow paper](https://arxiv.org/abs/1805.00907).

We should take both fast operator kernel implementation and extra potential layout transformation into consideration to get better performance.

### Steps

This section is about (1) adding new backend-specific nodes and corresponding instructions, (2) how to utilize the APIs to add these backend-specifics nodes to the graph, and (3) how to have your backend correctly handle this new corresponding backend-specific instruction that is IRGen'd from your backend-specific Node.

Here are mainly three steps to add a new backend-specific node in Glow:

1. Add a backend-specific Node `FusedAB` to `XSpecificNodes.h`, and a corresponding backend-specific Instruction `FusedAB` to `XSpecificInstrs.h`. Note that the `FusedABInst` needs to be marked with `autoIRGen()` so that the node is automatically IRGen'd to the instruction, as we currently do not support backend-specific IRGen.

2. Add logic to `XBackend::transformPreLowering()` or `XBackend::transformPostLowering()` (or both) depending on when you want to do the transformation. This logic would look for the pattern of nodes you want to fuse (`A` with a single use by `B`), and replaces all uses of the result of B with the new backend-specific `FusedABNode`.

3. Have your backend `X` implement `FusedABInst`. For example, for the OpenCL backend, this would mean adding a case to enqueue a kernel for the `FusedABInst` to `OpenCLFunction::execute()`, and then adding the corresponding kernel in `kernels.cl`.


### Examples

#### Operator Fusion for ReLU in CPU

ReLU is max between zero and the input value. Glow lowers `ReLUNode` to two basic low-level linear algebra operator nodes, `SplatNode` and `MaxNode`. The `SplatNode` first fills a Tensor with zero, and `MaxNode` compare `Input` with the filling Tensor. We can fuse these two operations which work with the same shape of tensors into a single kernel.

Please refer to the document in [Backend](https://github.com/pytorch/glow/blob/master/docs/Backends.md#backend-specific-nodes-and-instructions) part for source code details on adding a new backend-specific CPUMaxSplatNode on CPU. 

#### Data Layout Transformation for Conv Operator in OpenCL

OpenCL Conv is faster in layout `NCHW`, but  the default layout of convolution operator in Glow is `NHWC`. So we transpose the inputs/output and replace the `ConvolutionNode` with a backend-specific `OCLConvolutionNode` that uses `NCHW`. The transposes mostly can get optimized away thanks to the high-level graph optimizations.

The OpenCL backend defines `OCLConvolution` in `tools/ClassGen/OpenCL/OpenCLSpecificNodes.h` to support layout `NCHW` input.

```cpp
BB.newNode("OCLConvolution")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring(
        "This is an OpenCL-specific convolution implementation where the "
        "filter, the bias and the input are in the NCHW format");
```

During `transformPostLowering()`, this `convertConvToNCHWConv` node which contains a `NCHWConvNode` node and multiple`Transpose` nodes for `Input`, `Filter` and `Result` replaces the aforementioned pattern. 

A corresponding backend-specific `OCLConvolution` instruction is also needed, defined in
`tools/ClassGen/Backends/OpenCL/OpenCLSpecificInstrs.h`:

```cpp
BB.newBackendSpecificInstr("OCLConvolution")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Unsigned, "Group")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

```


### References

- [Glow: Graph Lowering Compiler Techniques for Neural Networks](https://arxiv.org/abs/1805.00907)
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)
- [TensorRT 3: Faster TensorFlow Inference and Volta Support](https://devblogs.nvidia.com/tensorrt-3-faster-tensorflow-inference/)
- [Discussions in Glow issue 1549](https://github.com/pytorch/glow/issues/1549#issuecomment-416283664)