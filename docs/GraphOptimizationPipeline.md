## Glow Graph Optimization Pipeline

This document describes the optimization pipeline and how to configure it for
varying precision configurations. Note that the terms "Graph" and "Function" are
often used interchangably.

### Graph (Function) Optimization Stages and Order

`glow::optimizeFunction()` is used to drive the optimization pipeline of a
Function. Inside `optimizeFunction()`, a Function is optimized by the following
optimization stages in order:

- `fold()`
- `optimize()`
- `lower()`
- `transformForPrecisionMode()`
- `optimize()`
- `Backend::transformPostLowering()`
- `optimize()`
- `checkAllNodesSupported()`

Note that optimize is called many times, usually after every other stage. This
is because the other stages may add or change the graph in some ways that allow
for previously inapplicable optimizations to now occur, or new nodes may be able
to take advantage of the optimizations.

### Descriptions of Optimization Stages

- `glow::fold()`: Fold low-level Nodes into higher-level Nodes. This is useful
  when compiling an input model where some high-level operators have been
  lowered (this can be for instance a side effect of model converters, like
  converters from Tensorflow to ONNX). In this situation, such folding can then
  enable more optimizations and also improve the performance of backends that
  support natively such high-level operators. Folding is done first, as we want
  to raise the graph to a higher level in order to take advantage of high-level
  optimizations and allow for backends to prevent lowering on them as well if
  desired.

- `glow::lower()`: Lowers high-level Nodes into lower-level Nodes. This allows
  backends to be agnostic to higher-level representations of Nodes. For example,
  a backend may support an LSTM if it supports all of the sub-nodes found inside
  an LSTM. However, it does not need to be responsible for understanding that it
  supports an LSTM. This future-proofs the backend, allowing it to support
  future high-level complex nodes without understanding their higher-level
  semantics. Note that [Backends can prevent
  lowering](Backends.md#backend-abstract-class) if preferred via
  `Backend::shouldLower()`.

- `transformForPrecisionMode()`: Transforms the graph depending on requested
  precision configuration. There are a few different options here:

  - `Profile`: Add special profiling nodes to the graph to gather a histogram of
    values flowing through each Node in the graph. This can then be used to
    automatically quantize the graph.

  - `Quantize`: Quantize the graph, given a previously gathered profile as
    mentioned above. This converts all Float inputs and outputs of each Node if
    the backend supports the node as quantized.

  - `ConvertToFP16`: Convert all Float inputs and outputs of Nodes to
    Float16. Note that this does not require any sort of profile. It can also be
    performed alongside Quantization to get a mixed precision graph.

- `Backend::transformPostLowering()`: Allow the Backend to transform the
  graph. This includes swapping in its own backend-specific nodes. More info can
  be found [here](Backends.md#backend-abstract-class) and
  [here](NewBackendSpecificNode.md#steps).

- `glow::optimize()`: Performs a series of graph optimizations, as listed
  [here](Optimizations.md#set-of-supported-graph-optimizations). Many of these
  are common compiler optimizations, such as DCE and CSE. Others are more ML and
  linear algebra related, such as fusing BatchNormalization into Convolutions in
  inference mode, or combining a series of Transpose operations into a single
  Transpose.

- `checkAllNodesSupported()`: Given some Backend that we are optimizing the
  Graph for, this verifies that the backend supports each of the nodes after the
  entire optimization pipeline is complete, via
  [Backend::isOpSupported()](Backends.md#backend-abstract-class). If any node is
  not supported it will return an error, which `optimizeFunction()` will then
  return up the stack.

## How to call `glow::optimizeFunction()`

Here we describe the API for `glow::optimizeFunction()` and how to use it in
different modes.

```
llvm::Error glow::optimizeFunction(Function *F, const Backend &B,
                                   CompilationContext &cctx);
```

An error is returned if something goes wrong during the optimization pipeline,
for example if a Node is no longer supported by the backend after optimizations,
or if the CompilationContext is not set up correctly.

- `F`: The Function (graph) to optimize. It will be transformed; if the caller
  would like to keep an original copy of the Function it is responsible for
  cloning it before calling `optimizeFunction()`.

- `B`: The Backend for which we are optimizing `F`. It is able to control what
  is lowered (`Backend::shouldLower()`), to specify what should be quantized
  (`Backend::isOpSupported()`); and to transform the graph however it wishes
  (`Backend::transformPostLowering()`).

- `cctx`: The CompilationContext, containing important data structures to keep
  some state before/after compilation is complete, as well as configuration
  information for what kinds of transformations to perform.

  - `PlaceholderBindings *bindings`: Mapping between Placeholders and Tensors
    for this compilation run. For example, this is used when instrumenting a
    Function for quantizaiton profiling, to create Placeholders and their
    Tensors for inserted `QuantizationProfileNodes`.

  - `LoweredInfoMap *loweredInfoMap`: Mapping from each Node output name in a
    Function (corresponding to unique NodeValues in a Function), to a set of
    `NodeNameAndKind` (which is a pair of Node output name and Kind). This is
    used to keep track of what Nodes are lowered into what other Nodes, for use
    during Profiling and Quantization.

  - `enum class CompilationMode compMode`: This is set to either `Train` or
    `Infer`, depending on what the Function is being used for. Some
    optimizations may only be performed during one of these modes.

  - `struct BackendOptions backendOpts`: Different options for compilation after
    graph optimizations have been completed. [See
    here](Backends.md#backendoptions-helper-struct) for more info.

  - `struct PrecisionConfiguration precisionConfig`: Configuration for different
    precision modes, used during `transformForPrecisionMode()`. Contains the
    following:

    - `enum class QuantizationMode quantMode`: One of `None`, `Quantize`, or
      `Profile`, as previously mentioned above.

    - `quantization::QuantizationConfiguration quantConfig`: Configuration for
      quantization, including the quantizaion precision `ElemKind`
      (e.g. `Int8QTy`, `Int16QTy`, etc.),
      [schema](Quantization.md#how-to-perform-nn-conversion) (e.g. Asymmetric,
      Symmetric, etc.), whether to use rowwise quantization, etc.

    - `bool convertToFP16`: Whether to convert all found `FloatTy` to
      `Float16Ty` in the Function. This is performed after Quantization, and so
      can be used together to get a mixed precision model.

    - `KindSet precisionModeKindSet`: A set of node kinds. This has different
      uses depending on the precision mode. If in profiling mode, this
      represents node kinds that should not be lowered ([mentioned
      here](Quantization.md#how-to-perform-nn-conversion)). If performing
      quantization or FP16 conversion this represents node kinds that should not
      be converted, and should instead be left in FloatTy.
