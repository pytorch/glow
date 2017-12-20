## Tensors

This document describes the design of the Tensor data structure and explains how
to use it effectively.

A Tensor is a multi-dimensional array. Tensors are allocated as contiguous
chunks of memory. Tensors are heap allocated, and have a fixed size that's
determined during construction or during tensor reset. Tensors are non-generic,
and are types. This means that the tensor contains an Enum that records the type
of the data (float, double, int, etc.)

Making tensors non-generic (they don't depend on the type of the element) makes
the interfaces very clear.  The neural network inference function does not need
to know the types of input or output tensors. This allows us to implement
networks that have non-floating point types (quantization of nodes).

## Handles

The problem with using non-generic tensors is that calculating the location of
an element becomes very expensive.  In order to figure out the location of an
element in memory, the Tensor subscript function needs to iterate over all
dimensions and multiply the dimension of each sizes with the input indices and
later multiply by the size of the element. This is extremely slow.  Another
problem is that the compiler can't prove that there are no pointers that point
to the internal data structures of the Tensor and has to re-read the tensor
dimensions and element type from memory very frequently.

The solution of this problem is the use of Handles. Handles are stack-allocated
generic data structures that contain an optimized version of the tensor
metadata.  The fact that they are stack allocated allows the compiler to
assume that there are no aliases to the tensor metadata.  The fact that the
class is generic makes the size calculation for the element type constant.

In practice, iterating over tensors using handles allows the compiler to perform
extremely efficient index calculation, and vectorization of the pattern access!

Handles are only valid as long as the tensor is alive. Tensors are passed by
value, and are relatively inexpensive to construct.  However, it is better to
initialize Handles outsize of inner loops.

Code example of creating tensors and the APIs to create a Handle and mutate the
data:

```
  /// Create a tensor of type Float, of the shape {4 x 2}.
  Tensor inputs(ElemKind::FloatTy, {4, 2});

  /// Create a handle to the tensor.
  auto I = inputs.getHandle<float>();

  /// Store an element to the tensor at index {0, 0}.
  I.at({0, 0}) = 13.1;
```
