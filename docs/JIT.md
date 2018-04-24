## Design of the CPU JIT

### Introduction

The JIT ("Just In Time") compiler is a backend that generates code in memory on
demand for the host CPU. The host cpu can be X86, ARM or anything that LLVM can
target. In many ways, the JIT is similar to the interpreter. Both execute code on
the host CPU, except that the JIT is able to further optimize the code.

The Glow interpreter goes over the low-level IR one instruction at a time and
executes a switch statement that dispatches a C++ implementation for each
instruction. This is suboptimal for a number of reasons. First, after each
low-level instruction is executed, by calling a function call,  we return to the
dispatch switch-loop.  Second, the C++ implementation of the low-level
instruction had no knowledge of the specific situation in which the instruction
is being executed.

The JIT, on the other hand, generates a single stream of highly optimized
instructions that don't go back to the interpreter.  Moreover, each instruction
is optimized based on specific information on the context in which the
instruction is executed.  When a matrix multiplication is compiled the JIT knows
exactly the dimensions of the matrices that are being executed and where the
tensors are placed in memory.  The JIT knows that the buffers do or do-not
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
offsets for each alloc activation within the buffer.  Then, the JIT performs a
single call to 'malloc' to allocates the heap. At this point each activation and
each weight has a concrete address on the heap.

Next, the JIT opens a new LLVM functions and prepares for code generation. The
compiler goes over each low-level instruction and generates a sequence of
LLVM-IR.  The next section describes how the LLVM-IR is generated.

After the LLVM module is generated, the compiler calls the LLVM optimizer to
optimize the generated module and the code generator to generate efficient
machine code. At this point the compilation phase is complete, and the network
is ready for execution.

### Usage of the Standard Library

During the compilation process, each Glow low-level instruction is converted into
a sequence of LLVM-IR instructions.  One way to implement this lowering is to
use the IRBuilder to generate low-level programs. For example, the matmul
instruction would translate to LLVM-IR by creating new basic blocks and encoding
the internals of the multiplications. This is insane. Implementing and
maintaining the low-level implementations of so many operations using the
LLVM-IR is not scalable.

The approach that our JIT takes is the use of a standard library. Each glow
instruction is translated into a function call. The function is represented in
LLVM-IR that's loaded from disk. When the main function is optimized the LLVM
optimizer is able to propagate things like tensor addresses and loop counts, and
inline the standard library functions into the module.

### Optimizations

In this section, we describe some opportunities for optimizations.

1. Some libraries provide optimized versions of some operators, such as matrix
multiplication and convolution. It could be a good idea to call these functions.
One way to implement this is to add explicit calls inside of the standard
library. If-defs and regular ifs can protect calls to optimized library calls.
This can reduce the complexity of the JIT significantly.

2. The LLVM optimizer has a great vectorizer, so no need to vectorize code by
hand. However, it has a pretty bad loop optimizer. Consider this example:

  ```
    for (...) A[i] = 3;
    for (...) A[i] = 4;
    return A[0];
  ```

The LLVM optimizer is unable to remove the first loop, even though it's
obviously dead code. And second, the optimizer can't figure out that we return
the value '4' and will emit a load. This means that we need to optimize loops
ourselves. The JIT would need to pattern match, and play tricks to fuse
operators that belong together.

