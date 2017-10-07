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
match. This compile has a debug method for dumping a graphical representation of
the graph into a dotty file. The method is called 'dumpDAG'. The textual
representation of the graph is less informative and it looks like this:

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



