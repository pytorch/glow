## Design of the CPU Backend JIT

### Introduction

The CPU Backend is a JIT ("Just In Time") compiler that generates code in memory
on demand for the host CPU. The host cpu can be X86, ARM or anything that LLVM
can target. In many ways, the JIT is similar to the interpreter. Both execute
code on the host CPU, except that the JIT is able to further optimize the code.

The Glow interpreter goes over the low-level IR one instruction at a time and
executes a switch statement that dispatches a C++ implementation for each
instruction. This is suboptimal for a number of reasons. First, after each
low-level instruction is executed, by calling a function call, we return to the
dispatch switch-loop. Second, the C++ implementation of the low-level
instruction had no knowledge of the specific situation in which the instruction
is being executed.

The JIT, on the other hand, generates a single stream of highly optimized
instructions that don't go back to the interpreter. Moreover, each instruction
is optimized based on specific information on the context in which the
instruction is executed. When a matrix multiplication is compiled the JIT knows
exactly the dimensions of the matrices that are being executed and where the
tensors are placed in memory. The JIT knows that the buffers do or do-not
alias, and exactly the number of iterations for the loop. The knowledge enables
much better code generation and vectorization. The JIT is also able to eliminate
all calls to 'malloc', because the memory is statically allocated. The whole
network is allocated by a single malloc call.

### How the JIT Works

The JIT accepts the low-level IR. At this point the high-level optimizer and the
low-level optimizers did their best to optimize the graph. This includes things
like tiling and memory sharing of buffers. The first thing that the JIT does is
allocate concrete memory addresses for the AllocActivation instructions in the
module. The allocation is done by scanning the module and updating the memory
allocator. After this process the allocator reports the high water mark, which
is the maximum number of bytes that the network consumes. The allocator assigns
offsets for each alloc activation within the buffer. Then, the JIT performs a
single call to 'malloc' to allocates the heap. At this point each activation and
each weight has a concrete address on the heap.

Next, the JIT opens a new LLVM functions and prepares for code generation. The
compiler goes over each low-level instruction and generates a sequence of
LLVM-IR. The next section describes how the LLVM-IR is generated.

After the LLVM module is generated, the compiler calls the LLVM optimizer to
optimize the generated module and the code generator to generate efficient
machine code. At this point the compilation phase is complete, and the network
is ready for execution.

### Usage of the Standard Library

During the compilation process, each Glow low-level instruction is converted into
a sequence of LLVM-IR instructions. One way to implement this lowering is to
use the IRBuilder to generate low-level programs. For example, the matmul
instruction would translate to LLVM-IR by creating new basic blocks and encoding
the internals of the multiplications. This is insane. Implementing and
maintaining the low-level implementations of so many operations using the
LLVM-IR is not scalable.

Instead, the CPU backend compiles a small standard library into LLVM bitcode
that it ships with the compiler. During the compilation process, Glow loads the
bitcode from disk and specializes the operator implementations for the specific
context. Glow replaces function arguments that represent the dimensions of some
tensor or buffer addresses with constants that LLVM can optimize to generate
efficient code. The compiler can decide on the kind and level of operator
specialization to perform, and trade compile time and binary size for
performance.

Most operators are very simple and the LLVM vectorizer is able to generate very
efficient code. Notice that by providing the exact tensor dimensions and loop
trip count the vectorizer is able to generate efficient code that does not
contain pre-header legality check and scalar loop to handle the remainder odd
iterations. The convolution and matrix multiplication operations are
hand-optimized in C++ using the clang extended OpenCL vector syntax, and LLVM
does a good job allocating registers and encoding the instructions, removing the
need to use inline assembly.

### Use Case: Optimizing Resnet50 for the CPU

In this section, we describe the way that Glow optimizes Resnet50 to generate an
efficient stream of x86 instructions. Resnet50 is a residual convolutional
neural network that contains 54 convolutions as well as other operators such as
element-wise addition, ReLU, batch normalization, max and average pooling,
FullyConnected, and SoftMax. Glow optimizes Resnet50 by performing both
high-level and low-level optimizations.

First, high-level transformations eliminate redundant transpose operations and
merge the batch normalization operation with a convolution node. Next, the CPU
backend transforms the graph into a target-specific graph that allows
device-specific optimization. The CPU backend identifies three kinds of
convolutions: convolutions with a small number of channels, convolutions where
the size of the input activation buffer is large, and convolutions where the
filter weight buffer is large. Each one of these convolutions requires a
different compilation strategy.  Next, the target-specific optimizer mutates the
graph and generates code that matches the selected convolution. Each convolution
kind uses a different filter memory layout and tile size. Below you can see the
transformed filter memory layout.

```
Filter layout before transformation:
   [depth, filter_x, filter_y, channel]
Filter layout after transformation:
   [depth/N, filter_x, filter_y, channel, N]
```

This 5-dimensional tensor layout allows for consecutive SIMD memory access. The
N parameter is selected based on the iteration order and the blocking strategy
for the convolution. The CPU backend traverses the graph and replaces any
convolutions it would like to optimize in this way with this specialized
convolution.

The second parameter that the compiler controls is the size of the convolution
tile. Glow selects a processing tile that depends on the size of the first level
cache of the processor.

Next, the low-level optimizer optimizes the instruction stream by shrinking the
lifetime of memory allocations for the activations, and then performs static
memory allocation for the whole network into a single buffer. This reduces the
mutable memory footprint of the network. From this point in the compilation
pipeline the compiled code can refer to pointers in memory.

Finally, the compiler performs efficient code generation for the non-convolution
parts of the network. For example, below the generated assembly for some part of
the network can be seen. The compiler fused two unrelated element-wise
operations into a single loop. The Add and Max operations are performed on the
same memory buffer without reading the memory twice.

```
LBB14_1:
  vmovaps 3211264(%rcx,%rax,4), %ymm1
  vmovaps 3211296(%rcx,%rax,4), %ymm2
  vmovaps 3211328(%rcx,%rax,4), %ymm3
  vaddps 6422528(%rcx,%rax,4), %ymm1, %ymm1
  vaddps 6422560(%rcx,%rax,4), %ymm2, %ymm2
  vmovaps 3211360(%rcx,%rax,4), %ymm4
  vaddps 6422592(%rcx,%rax,4), %ymm3, %ymm3
  vaddps 6422624(%rcx,%rax,4), %ymm4, %ymm4
  vmaxps %ymm0, %ymm1, %ymm1
  vmaxps %ymm0, %ymm2, %ymm2
  vmaxps %ymm0, %ymm3, %ymm3
  vmovaps %ymm1, 6422528(%rcx,%rax,4)
  vmovaps %ymm2, 6422560(%rcx,%rax,4)
  vmaxps %ymm0, %ymm4, %ymm1
  vmovaps %ymm3, 6422592(%rcx,%rax,4)
  vmovaps %ymm1, 6422624(%rcx,%rax,4)
  addq $32, %rax
```
