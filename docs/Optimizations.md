## Glow optimization passes

This document describes the target-independent optimizations performed by the
graph and IR optimizers and some implementation details.

### Overview

Glow has two different optimizers: the graph optimizer and the IR optimizer.

The graph optimizer performs optimizations on the graph representation of a
neural network model. The nodes of the graph usually represent more coarse-grained 
operations than those represented by the IR instructions. These
operations also do not explicitly represent memory allocations and buffers.

The IR optimizer performs a number of optimizations on the IR representation of a
neural network model.

The optimizations have two major objectives. One is to improve the performance
of training and inference steps. The other one is to reduce the memory consumption
during the execution of neural network models.

It is worth mentioning that performing optimizations reducing memory consumption
is easier at the IR level, because memory allocations and deallocations are
explicitly represented in the IR, whereas they are not explicit in the graph
representation.

### Set of supported graph optimizations

Below you can see the list of currently supported graph optimizations in Glow:

  * Dead code elimination (DCE)

    This optimization removes computations whose results or side-effects are
    not used.

  * Optimization of transpose nodes

    This optimization combines multiple consecutive transpose nodes
    into a single node, eliminates identity transpose nodes, and optimizes
    transpose nodes into reshape nodes when they actually move no data.

  * Sinking of transpose operations below other operations

    This optimization sinks transposes below such operations like a batch
    normalization, RELU, sigmoid, ChannelShuffle, etc. By doing this, many
    transpose operations are brought closer to each other and it creates more
    opportunities for elimination of transpose operations.

  * Pool operations optimization

    This optimization swaps the order of Relu->MaxPool, to perform the RELU
    operation on a smaller tensor. This optimization is not a major performance
    win. The RELU operation takes a small fraction of the time, and reordering
    the nodes does not provide a lot of performance wins. However, reordering the
    buffers allows us to reuse the memory buffer of the pool operation and
    potentially save memory.

  * Optimizing of regression nodes in the inference mode

    In inference mode, Regression nodes simply forward their inputs.

  * Optimization of concat nodes

    This optimization merges multiple consequent concat nodes into a single concat
    node.

  * Common sub-expression elimination (CSE)

    This optimization performs a classic CSE with the goal of avoiding of any
    results that were computed already.

  * Optimization of ReduceMean nodes

    This optimization performs substitions of ReduceMean with AvgPool node if
    the reduce parameters are suitable: input is 4D with last two dimensions
    to be reduced.

#### Quantization specific optimizations

Majority of the common optimizations above can be used on a quantized graph.
But in addition to those there are quantization specific optimizations:

  * Quantize(Dequantize(X)) -> RescaleQuantized(X)

    If the Quantize-Dequantize sequence does not change the type then this
    sequence is simply dropped without adding nop RescaleQuantized node.
    If Dequantize node has an input type that is different from the Quantize
    node output type then a RescaleQuantized node replaces Quantize-Dequantize.

  * Dequantize(Quantize(X))

    A sequence of Dequantize(Quantize(X)) is a nop transformation and can be completely removed.

  * RescaleQuantized(RescaleQuantized(X)

    A sequence of RescaleQuantized operators can be replaced by just a single RescaleQuantized.

  * Constants optimization

    Constants which have single use could be quantized at the optimization
    phase. This optimization replaces Quantize(Constant) with just a Constant
    with updated quantized weights based on the quantization parameters from the
    Quantize node.

  * RescaleQuantized(Max(X,Y)) -> Max(RescaleQuantized(X), RescaleQuantized(Y))

    It's OK to rescale the operands because even if the output range is smaller then truncation
    would have happened during the rescaling. On values that are outside of the range, we just move
    the truncation to a different location.

  * Combine RescaleQuantized operator up into the operation

    There are a number of operations which can operate on varying quantized parameters
    for the output type. It's safe to just merge RescaleQuantized node into the operator itself if
    operator supports this, e.g., add, mul, etc.

    This optimization can be applied to:
    
      * Add
      * Sub
      * Mul
      * Div
      * Min
      * Max
      * Convolution
      * Splat

  * Combine Arithmetic operations into Batch Normalization.

    When a chain of Arithmetic nodes (each operating on a constant on one side) is right
    below BatchNorm node, the chain is folded into the Batch Norm node.

    When a chain of Arithmetic nodes (each operating on a constant on one side) is right
    below Convolution node, the chain is folded into a BatchNorm.

  * Combine RescaleQuantized operator down into the operation

    This optimization allows eliminating redundant rescale operations when the next
    operation supports quantized inputs of different scales and offsets, e.g., normal
    arithmetic operations: Add, Sub, Mul, Div, Min, Max.

  * Sinking RescaleQuantized operator below other operators

    This optimization sinks RescaleQuantized node below such operations as slice,
    reshape, transpose, etc. By doing this, many RescaleQuantized operators
    are brought closer to each other, and it creates more opportunities for
    elimination of RescaleQuantized operations.

  * RescaleQuantized(Quantize(X)) -> Quantize(X)

    A sequence of Quantize operation followed by RescaleQuantized operation
    is replaced by a single Quantize operation with the proper quantization
    parameters based on the RescaleQuantized operation.

  * Eliminate Max operation in Max(Splat(X), someOperand) or Max(someOperand, Splat(X))

    Splat and Max operations can be completely eliminated if Splat value cannot impact
    the result of the Max operation. For example, Max and Splat are removed if Splat
    value is smaller than the smallest possible value from the other operand. Smallest
    possible value from the operand can be calculated based on the quantization
    parameters which represent quantization range [min, max] in fp32.

#### Configuring a graph optimization pipeline

The graph optimizations listed above are each formulated as a FunctionPass,
which is run on a Function (graph). A series of FunctionPasses along with how to
configure each (via a `FunctionPassConfig`) are what constitutes a pipeline,
which is passed into a PassManager, which executes them. A pipeline
is simply a vector of `FunctionPassConfig`s. `FunctionPassConfig` is a class
made up of:

- `FunctionPassID`: An ID corresponding to a specific FunctionPass. For example,
  `FuncionPassID::OptimizeArithmeticNodes`. Default: `EmptyPass` (no-op pass).

- `ConvergenceMode`: An enum corresponding to whether this FunctionPass should
  be run a single time (OnePass) or repeatedly until a fixed point is reached
  (`UntilFixedPoint`). Default: `OnePass`.

- `DCERequiredMode`: An enum representing whether DCE is required before a pass
  is run (`BeforePass`), or not at all (`None`). Running DCE is often required
  in order to make sure the number of users for each Node is up to
  date. Default: `BeforePass`.

- `EnabledCompilationModes`: A set of `CompilationMode`s representing which
  mode(s) the pass should run under. Default: `{Infer, Train}`.


### Set of supported IR optimizations

Below you can see the list of currently supported optimizations:

  * Peephole optimizations

    These are small, local optimizations that look for specific sequences of
    instructions and replace them with more efficient sequences of instructions.

  * Dead store elimination (DSE)

    This optimization removes stores into weights or allocations if it can
    prove that the results of these stores are never going to be used.

  * Deallocations hoisting

    This optimization tries to place the buffer deallocation instructions right
    after the last use of a buffer. Doing so reduces the lifetime of the buffer
    and makes the freed memory available for the allocation of other buffers.
    It improves the memory consumption.

  * Allocations sinking

    This optimization tries to place the buffer allocation instructions right
    before the first use of a buffer. Doing so reduces the lifetime of the
    buffer and makes the unused memory available for the allocation of other
    buffers. It improves the memory consumption.

  * Dead allocations removal

    This optimization finds and removes allocations that are just allocated and
    deallocated, but are never used. Such situations may happen e.g. after
    performing other allocations. Performing this optimization improves the
    memory consumption.

  * Making weights constant

    This optimization marks weights that are never mutated as constant. This may
    allow for placing such weights in a read only memory segments and share it
    between simultaneous executions of the same neural network model.

  * Sharing of buffers

    The purpose of this optimization is to reduce the memory usage by reusing
    the memory buffers as much as possible. The overall idea is that it is fine
    to combine storage for two live intervals if they do not overlap. Typically,
    two live intervals are considered as candidates for sharing if they occur
    in the same instruction.

  * Stacking of data-parallel operations

    Stacking tries to combine multiple data parallel (i.e. element-wise) operations
    that work with the same shape of tensors into a single kernel.

    Executing such a kernel should be in theory more efficient than executing those
    operations sequentially one after the other, because such a combined kernel
    exposes a better cache locality.

    The stacked kernels should provide even more advantages on GPUs, because they
    reduce the number of kernel threads launches, which are rather expensive operations.
