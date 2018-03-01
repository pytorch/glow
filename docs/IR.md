## Design of the Glow IR

### Introduction

This document describes the motivation behind the Glow intermediate
representation and some implementation details.

Glow is a retargetable compiler that supports a number of different backends.
This means that the first few layers of the compiler are target-independent, but
as you get closer to the different backends things start to diverge.  The first
two levels of IR are shared between all targets. Different backends may have
additional layers of IR.

### High-level Graph

The high-level IR, is a graph-based representation that's similar to the graph
that you may find inside Caffe.  When we load the model from a file we construct
this graph in a direct translation of one operator to one node.  It's a simple
graph that allows basic transformations such as swapping the order of nodes and
removing nodes. The graph is strongly typed, which means that inputs and output
have a known tensor type (dimension and element type), and that the types must
match.

The Glow graph is structured as a Module that contains multiple functions that
contain a multiple nodes. Variables, which are similar to global variables in C
programs, are shared between the functions. Nodes inside functions are able to
reference variables, which are owned by the module. The picture below depicts a
module that contains two functions.  One of the functions does the training of
the weights, and the other function runs the inference.

![](module.png)

Glow functions contain nodes that represent the different operations of a neural
network. The function owns the nodes and has access to the variables in the
module. The picture below depicts a small part of a function.

![](nodes.png)

The compiler has a debug method for dumping a graphical representation of the
graph into a dotty file. The method is called 'dumpDAG'. The pictures above were
generated with this method. The textual representation of the graph is less
informative and it looks like this:

  ```
  pool
  name : "pool"
  input : float<8 x 28 x 28 x 16>
  output : float<8 x 9 x 9 x 16>
  kernel : 3
  stride : 3
  pad : 0
  kind : max

  convolution
  name : "conv"
  input : float<8 x 9 x 9 x 16>
  output : float<8 x 9 x 9 x 16>
  filter : float<16 x 5 x 5 x 16>
  bias : float<16>
  kernel : 5
  stride : 1
  pad : 2
  depth : 16

  relu
  name : "conv"
  input : float<8 x 9 x 9 x 16>
  ```

After optimizing the graph with target-independent optimizations the code is
lowered into the mid-level IR in a phase that's called "IRGen" (stands for IR
generation). This is a one-to-many translation where each operator is translated
into one or more instructions.

### Variable Visibility

Variables are persistent tensors that live across different executions of the ML
network.  Variables are annotated with Public or Private labels. These labels
specify whether the node is visible outside of the graph, or not. If the node is
public, then it means that C++ code from outside the graph may access the
variable directly and change it's content before or after the execution of the
program.  This means that the optimizer is not allowed to delete unused public
variables or change their dimensions. On the other hand, in the case of private
variables, the optimizer is allowed to delete unused variables, transpose,
perform constant propagation, etc. The semantics of variables in the program,
both private and public, is that all writes must happen before the end of the
execution of the program.

### Predicates

Predicates are boolean variables that control the execution of some node or
instruction. If the value of the predicate at runtime is set to 'false' then the
predicated node or instructions may return any value. The program should know to
ignore the output of the predicated instruction because it could be zeros or
uninitialized memory. Predication is a way to accelerate the performance of the
network by avoiding some computation. The type of the predicate must be a
scalar, or a vector that matches the batch size.

![](pred.png)

### Mid-level Graph

The low-level IR enables a different kind of target independent optimizations
that are not possible with the high-level graph format. For example, the ability
to share the memory buffers during the forward pass can't be expressed in the
Graph form because buffers are not explicit.

The mid-level IR is built like a sequence of instructions that perform things
like copy-memory and perform-convolution.  The IR is not Static Single
Assignment (SSA) based representation, because the IR does not support control
flow. The IR is strongly typed and each instruction operand kind has known
parameter types.  The IR representation is designed to be used as an in-memory
form. The IR can be dumped to human readable assembly-like format.

The IR has two sections: 'declare' and 'program'. In the first section of the IR
we declare a number of memory regions that live throughout the lifetime of the
program. This is similar to global variables in C++. The second part of the IR
is list of instructions. Each variable is annotated with the kind of
initialization that the program should do.

There are two kinds of memory regions. The global memory regions and locally
allocated regions. The locally allocated memory regions are similar to 'alloca'
in C++, and in LLVM. Memory regions are strongly typed, which means that the
kind of type of tensor that the region represents is known.

Instructions operate on either global variables or locally allocated buffers.
Each operand is annotated with one of the qualifiers '@in'/'@out'/'@inout'. In
means that the buffer is read from. Out means that the buffer is written into.
And InOut means that the instruction may read and write into the buffer. These
operand qualifiers help the optimizer decide when it is legal to share buffers.
Instructions may have other attributes that specify the legality of some
optimizations. For example, some operands require that the data from the forward
pass would be kept around for the backward pass, so if the program is not
optimized for inference-only mode then certain memory optimizations can't
happen.


This is an example of an unoptimized IR.

  ```
  declare {
    %input = weight float<8 x 28 x 28 x 1>, broadcast, 0.0
    %filter = weight float<16 x 5 x 5 x 1>, xavier, 25.0
    %filter0 = weight float<16>, broadcast, 0.100
    %weights = weight float<10 x 144>, xavier, 144.0
    %bias = weight float<10>, broadcast, 0.100
    %selected = weight index<8 x 1>
    ...
    %result = weight float<8 x 10>
  }

  program {
    %allo = alloc float<8 x 28 x 28 x 16>
    %conv = convolution [5 1 2 16] @out %allo, @in %input, @in %filter3, @in %bias0
    %allo0 = alloc float<8 x 28 x 28 x 16>
    %relu = relu @out %allo0, @in %allo
    %allo1 = alloc index<8 x 9 x 9 x 16 x 2>
    %allo2 = alloc float<8 x 9 x 9 x 16>
    %pool = pool max [3 3 0] @out %allo2, @in %allo0, @inout %allo1
    ...
    %deal6 = dealloc @out %allo6
    %deal7 = dealloc @out %allo7
    %deal8 = dealloc @out %allo8
    %deal9 = dealloc @out %allo9
  }
  ```

### The Lifetime of a Glow Instruction

This section describes how instructions make their way from the beginning of the
compilation pipeline, and through the different levels of IR and to the
backends.  The Glow compilation pipeline comes to solve the problem of targeting
a large number of opcodes to many different targets. For example, being able to
run the Div, Relu, ConvGrad opcodes on three different accelerators and two GPU
vendors. The approach that was taken by classic learning frameworks was to
implement each opcode for each hardware target. And so, Div would be implemented
once for the GPU, once for the CPU, once for mobile, etc. This approach does not
scale as the number of opcodes increase and the number of hardware targets
increase. Instead, Glow takes a different approach. Instead of compiling the
high-level operators directly, Glow performs "node-lowering". In this phase the
compiler breaks the high-level operators into low-level linear algebra
operators. For example, the FullyConnected layer is represented as a sequence of
matrix multiplication followed by element-wise add. Then, the different compiler
backends don't have to implement the FullyConnected layer and a dozen other
high-level opcodes, just the low-level matrix multiplication. This lowering
phase derives many of the design decisions of the compiler.

In Glow, lowering is performed as part of the high-level graph, that's described
above. The lowering process happens before IR-Gen for a number of reasons.
First, the new lowered graph may allow additional graph-level optimizations.
Second, the new graph structure may effect the decisions that the scheduler must
take. And third, after lowering we allow the backends to perform additional
target-specific optimizations. The lowering transformation does not preserve the
semantics of the graph, because it is not possible to differentiate the graph
for certain operators. For example, the Regression node becomes a nop, for the
forward pass, but is translated into a element-wise subtract for the backward
pass. Performing the lowering before differentiantion would prevent us from
performing the correct lowering of the Regression node.

This is a high-level overview of the compilation process:
1. The graph is constructed (via the c++ interface or graph loader).
2. The graph is optimized, and differentianted, if needed.
3. Linear Algebra lowering takes place.
4. Additional rounds of optimizations (both target independent and target specific.)
5. Scheduling of the graph into a linear sequence of nodes that minimizes the memory usage.
6. IRGen (convert the low-level graph into instructions).
7. IR-level optimizations.
8. Backend-specific optimizations and code generation.



