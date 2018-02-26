
# Quantization in Glow

## Introduction

Quantization is the process of constraining an input from a continuous or
otherwise large set of values (such as the real numbers) to a discrete set
(such as the integers). In the context of machine learning, Quantization is the
process of converting the inference phase of the neural network execution from
floating point arithmetic to integer arithmetic.

This is an external [link](https://www.tensorflow.org/performance/quantization)
that explains how quantization is done in TensorFlow.

Glow is able to convert floating-point based networks into 8-bit integer
networks. This document describes how this is done.

## Tensor Representation

In Glow, Tensors are typed, which means that they can represent
non-floating-point values such as Int8 (8-bit integers) and index types (either
32-bit or 64-bit, depending on the architecture. Quantized tensors are tensors
of 8-bit integers. In addition to the information that the tensor payload is
made of 8-bit integers, the possible range of the values in the tensor is
recorded. The extra range information is recorded using the 'scale' and 'offset'
fiends.  To convert from the 8-bit integer range of [-128..127] to the number
that they represent use the following conversion formula:

  ```
    value = (input - offset) * scale
  ```

Activations, weights and variables all use the same type-system and represent
information in a uniform way.

## Network Conversion

Different parts of the network contain floating-point values in different
ranges. In some parts the typical range of the numbers is between zero and one,
while in other parts of the network the possible range is in the hundreds.
Choosing a single conversion scale for the whole network would not work, because
a single scale value could be imprecise for small values and truncate large
values.

We use profile guided information to estimate the possible numeric range for
each stage of the neural network. Our Quantization conversion works using a two
phase process. At first, we instrument the network and add special nodes that
record the ranges of activations that flow in the network.  Next, we use the
profile information to convert the network into a quantized form. We convert
portions of the network into islands of integer computation and aim to generate
outputs in the range that the original floating point network produces.

## Compiler Optimizations

Glow features a number of compiler optimizations that transform the compute
graph and make it more efficient. There are a few classes of optimizations and
parameters to optimize.

First, we attempt to minimize the number of conversions between floating-point
tensors and integer pointers, in both directions. Some operations, such as
'transpose' and 'contact' operate on both types, and changing the representation
can minimize conversions.

Second, the neural network contains 'rescale' nodes that change the range of the
integers. These nodes are required to convert between numeric ranges that mimic
the original floating point network. However, in many cases it's possible to
fold the rescale operations into numeric-producing operations, and eliminate
them.

Third, it's possible to rescale the values in the network in order to allow fast
hardware implementations of the quantized operations. For example, consider the
'max' operations.  By converting both sides of the 'max' into the same scale we
allow the hardware to perform a simple comparison. By normalizing both sides of
the 'max' operation to the same scale we enable this efficient optimization.







