# Block

Kernel Nano-compiler Yielding Flawless Execution (KNYFE) is a small, lightweight, code generator. It aims to generate performant kernels without the user having to worry about low-level HW architecture details.

Block provides a portable way to store `KNYFE` kernels and provides a `DSL`
for writing more expressive kernels without having to use `KNYFE`'s internal IR.

This document provides an overview of the `Block` format
and how to write kernels directly in `Block`.

`KNYFE` and `Block` are part of an early experiment for
defining kernels for `Glow` using DSLs.

## Kernel Declaration

To declare a new kernel in `Block` use the keyword
`kernel` followed by a string naming the kernel.

Said string should be followed by parentheses declaring
the *name* of the input/output tensors for the kernel.

The body of the kernel should be inside a `{}` just like functions in `C`.

Here's an example of a kernel `foo` that takes tensors `A` and `B`:


```
kernel foo(A; B) {
}
```

## Declaring Tensor's shape and distribution

The kernel `foo` showcased in the section above is incomplete.
Given arguments `A` and `B`, the bare minimum requirement is
to declare their parametrized shape and their distribution.

As an example, lets assume `A` is a two dimensional tensor wherein
each element is one byte, and `B` is a one dimensional tensor
wherein each element is 4 bytes.

Their `tensors` block inside the kernel `foo` will look as follows:

```
  tensors {
    A = {{d0}; {d1}; elem : 1}
    B = {{d1}; elem : 4}
  }
```

To declare The distribution of `A` and `B`, we use the standard `ROTE`
notations, for example:

```
  distribution {
    A = {(0,); (1,)}
    B = {(0,)}
  }
```

The `Block` language is intelligent enough to avoid requiring
the specification of default values during initialization.
however, it does allow the programmer to override said values.

Each tensor in the `tensors` block can have a custom
`align` for each and every dimension.
The default stride is one and the default alignment
is 32 for all dimensions.

Here's a `tensors` block that overrides said values:

```
  tensors {
    A = {{d0; align : 64}; {d1; align : 128}; elem : 1; addr : 0x000000}
    B = {{d1; align : 128}; elem : 4}
  }
```

In the example above, the first dimension of `A` got an alignment of 64.

The second dimension got alignment of 128.

The base address of tensor A is `0x000000`.

For tensor `B` the alignment has been overridden to 128.

## Declaring Buffers

A tensor `A`, or a slice of a tensor `A`, can be transferred into
a fast local memory bank.
`Block` calls said memory banks `Buffer`.

To declare buffers for tensors `A` and `B`, above, use a `buffers` block:

```
  buffers {
    B = {{d1}}
    A = {{NULL}; {d1}}
  }
```

In the example above, we declare that we are interested in having a
buffer for the first dimension of `B` and for the second dimension
of `A`.

We do not need to specify the slice size of the tensor's
sub-dimension we want to transfer into a buffer,
`KNYFE` can calculate said value on its own.

The special value `NULL` indicates that we don't want to
transfer *any* slice of `A`'s first dimension into a
buffer.

## Declaring Tachyons

It is *highly* recommended not to declare and/or use `Tachyon`
manually, `KNYFE`'s optimizer should make use of them on
its own.

However, advanced users that want to manually optimized kernels
are free to use tachyons in their `Block` code.

Tachyons represent a logical memory storage that acts similar
to a CPU's L1 cache or a register file.
Can also be thought of as something in-between a Scratchpad
and a Cache. It is a super-fast memory, that may or may not
physically exist in hardware,
for most inner loops of a heavy compute.

Here's the `tachyons` block for `A` and `B` described above:

```
tachyons {
    A = {{d0}; {d1}}
    B = {{d1}}
    Cregs = {{NULL}; {NULL}; elem : 1}
  }
```

## Declaring Kernel Arguments

Kernel argument blocks contain one or more variables,
which describe the default dimension / block size.

It is worth noting that variables are *not* mutable on the
`Block` level, however,
they may mutate during code generation or runtime.
For example, A convolution kernel with a filter might
change the output dimensions specified on the `Block` level.

Given that tensors `A` and `B` named three dimensions above
(`d0`, `d1`, and `d2`), these three variables *must*
be present in either an arguments block *or* as
local vars, described later, in a compute blocks.

If we want to iterate over said tensors, we *must* provide
block size arguments and/or local variables.

Here's an example arguments block:

```
  arguments {
    var d0 = 3
    var d1 = 1024
    var bS0 = 32
    var bS1 = 32
  }
```

In the example above we declared that `A` is with a default
shape of `[3][1024]` and `B` is of a shape `[1024]`.

We've declared two block size variables, `bs0` and `bs1`,
that have a default value of 32.

## Compute Blocks

Compute blocks optionally contain one `var` statements,
which are internal composition of arguments,
and one or more `Block` instructions.

`var` statements are a composition of declared `var` arguments
that will *not* be mutated by `KNYFE`'s codegen.

These local `var` statements are used when doing arithmetic
operations on arguments.

To continue the example from the previous section, if we want to
create a dimension `d2` that is the summation of
`d0` and `d1` with a block size `bs2` that is the division of
`bs0` and `bs1` add the following:

```
    var d2 = d0+d1
    var bS2 = bS0/bS1
```

Following variable declaration, compute blocks can contain
any number of instructions.

## Block Instructions

The following instructions are currently supported by `Block`:

### `dma_in`

Transfers data from a Tensor into the tensor's buffer.

Must take a tensor name as an input. Optionally takes a `stride` variable.

### `loop`

Iterates over one or more tensor partitions.

If iterating over multiple tensors, it is the programmer's
responsibility to insure all partitions have the same
total iterations count.

The tensors `loop` iterates over can be in *any* memory
hierarchy. i.e. a Tensor, Buffer or Tachyon.

Each partition *must* take a tensor dimension `dim` and
a block size `block` as an input. It optionally takes a
`step` and `filter` sizes and a `partial_contribution` flag.

If `step` var is not provided, the default is one.

If `filter` is not provided the default is block size.
A filter size different than block size is used for halo
partitioning, more details about that can be found below.

If `partial_contribution` is not provided the default is zero.
A partial contribution represent the concept of a tensor iteration making a partial contribution to the output.
i.e. since we do not have the concept of an explicit output tensor,
it showcases that we are iterating over multiple input tensor regions in order
to compute a single output tensor region.

### `matmul`

Does matrix multiplication. Takes two input tensors and
an output tensor as inputs.

Must provide storage hierarchy for each tensor.
Inputs must be in either Buffers or Tachyons,
Output can be anywhere.

### `copy`

Copies a tensor. Takes an input tensor and
an output tensor as inputs.

Must provide storage hierarchy for each tensor.

### `transpose`

Transposes a tensor. Takes an input tensor and
an output tensor as inputs.

Must provide storage hierarchy for each tensor.

### `adjust_buffer`

Increments the read and/or write pointers of a tensor's buffer.
Takes a tensor name as an input.

### `reduce`

Does a reduction. Must take a tensor name and its storage
hierarchy as an input.

### `conv`

Performs a convolution. Takes two input tensors and
an output tensor as inputs.

Must provide storage hierarchy for each tensor.
Inputs must be in either Buffers or Tachyons,
Output can be anywhere.

### `add`

Does element-wise addition. Takes two input tensors and
an output tensor as inputs.

Must provide storage hierarchy for each tensor.
Inputs must be in either Buffers or Tachyons,
Output can be anywhere.

### `mul`

Does element-wise multiplication. Takes two input tensors and
an output tensor as inputs.

Must provide storage hierarchy for each tensor.
Inputs must be in either Buffers or Tachyons,
Output can be anywhere.

### `macc`

Does element-wise multiply-accumulate. Takes two input tensors and
an output tensor as inputs.

Must provide storage hierarchy for each tensor.
Inputs must be in either Buffers or Tachyons,
Output can be anywhere.

### `reshape`

Reshapes an entire tensor. Takes an input tensor and
an output tensor as inputs.

This operation is a reshape operator on the data.

The number of elements in the output tensor must
equal the number of elements in the input tensor.

The result of this operation is either a simple cast,
which is a no-op, or actual data movement if
adding or removing padding is required.

A `KNYFE` backend is allowed to ignore the data transfer as
a fast-math-like optimization. i.e. if a cast can be used
instead of a full data transfer due to architecture
details and/or due to all the compute uses being on
elementwise operations.

### `first_iter`

Performs some Instructions on the *first* iteration of an
outer loop partition.

Takes a loop partition, described in `loop` above, as input.

### `last_iter`

Performs some Instructions on the *last* iteration of an
outer loop partition.

Optionally Takes a loop partition,
described in `loop` above, as input.

If loop partition is not provided, a preceding partition
from a `first_iter` is assumed.

### `mid_iter`

Performs some Instructions on the *middle* iterations of an
outer loop partition. i.e. neither the first nor last.

