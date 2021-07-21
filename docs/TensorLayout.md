## Tensor Layout

This document describes the design of the tensor Layout requirements in Glow.

Certain operations (e.g. convolutions, gemms, etc) need to know the semantic
layout of their tensors, i.e. the logical ordering ordering of their dimensions
(e.g. `NHWC`). Some backends enforce additional backend-specific requirements
on said operations (e.g. tensor alignment).

A theoretical clever backend, might even go a step further and have said
layout requirements depend on the properties of the operation: a convolution
with a small filter may need the input operands in a format different from a
convolution with a big filter.

Tensor layout is a property of the operation, some operations, such as
element-wise operations, may not care about their input layout, we avoid adding
a layout field for said operations to reduce the dynamic memory consumption of
the compiler.

For operations that do have layout requirements, Glow has an easily extendable
string-based layout field. This allows backends to override Glow's default
requirements without the hassle of creating a custom, backend-specific, node.

Glow's string-based layout format is encoded as follows:

1. A mandatory one character representing the current dimension. Either an  alphabetic letter or `*` (any layout).
2. An optional token for the start of the current dimension's information: `[`.
3. An optional namespace identifier for non-standard information, such as tiling, followed by `:`. Must have `[` from 2. in place. Following said identifier, all subsequent data is considered as a "black box" until `]` is encountered.
4. Given that we have `[` from 2. in place, the closing bracket `]` for it.
5. Optionally go back to 2.

As an example for this encoding, here's how we add alignment information,
which is an officially supported extension, thus not requiring a namespace,
followed by a backend-specific extension:
`N[a=32][namespace_for_unsupported:<bla>]HWC` would represent 4-D tensor wherein
`N` needs an alignment of 32 + some private backends' requirements we don't know about.
`HWC` have no layout restrictions.
We can, of course, combine "any" dimensions in there, for example: `N[a=32]*H*[a=64]`
would represent "any" for the second dimension with no restrictions whatsoever while
we have an alignment restriction of 64 on the 4th.

Notes:

1. For each dimension, the identifier can be either a single english alphabet letter,
either upper or lower case, or the star symbol.
2. We assume that a single letter is enough for each dimension,
it makes parsing easier and avoids adding delimiters in the serialized format,
however, we do have a constructor that (theoretically) accepts multi-letter dimensions.
If we decide to expand the current support,
we will need to add delimiters to the serialized form.

## Layout Requirements Interface

Backends in Glow *may* derive from [base class `TensorLayoutCommon`](https://github.com/pytorch/glow/blob/master/include/glow/Graph/TensorLayout.h).
Which includes the following virtual methods they can override:

- `virtual std::string getDefaultNDLayout(unsigned dims) const`

  - This helper function takes a `unsigned dims` and returns the (current) default n-D layout.

- `virtual std::string getNthInputLayoutRequirements(const Node *node, size_t n)`

  - This function takes an operator `Node *node` and returns the layout requirements of the Nth input `n`.

- `virtual std::string getNthResultLayoutRequirements(const Node *node, size_t n)`

  - This function takes an operator `Node *node` and returns the layout requirements of the Nth result `n`.

- ```
  virtual bool isSatisfiedBy(TypeRef ty,
                               const TensorLayoutDescription &destLayout,
                               const TensorLayoutDescription *srcLayout) const
  ```
   - This function checks if `ty` satisfies `destLayout` layout requirements, if `srcLayout` is provided for `ty`, take that into account.

- `virtual llvm::ArrayRef<TensorLayoutDescription> getLayoutsForDims() const`

  - This helper function returns an array of predefined layouts for all dimensions from `0-D` to Glow's max tensor layout dimension.

- `bool isEnabled() const`
   - Indicates whatever checking for layout requirements is enabled or not. default is off.

An example of why backends may want to override such methods can be seen in the `OpenCL` backend:
Convolutions are more efficient in `NCHW` format, as such, we may lower a `ConvolutionNode`
into a `NHWC` to `NCHW` transpose + convolution.
The `OpenCL` verifier should expect `NCHW` for the input/output of the convolution instead of `NHWC`.
`OpenCL` opts-in to post-lowering verifications.

## Canonical Tensor Layout

Before lowering a Glow graph into a specific, we introduce a "Canonical"
representation that we expect for certain operations.
This allows us to verify the graph after every transformation and may expose `GraphOptimizer` bugs [^tl0].
[class `CanonicalTensorLayout`](https://github.com/pytorch/glow/blob/master/include/glow/Graph/TensorLayout.h)
derives from `TensorLayoutCommon` and overrides the following functions:

- `std::string getDefaultNDLayout(unsigned dims) const`

  - Overrides the default `n-D` layout from "any" into something else, e.g. 4-D any into `NHWC`.

- `std::string getNthInputLayoutRequirements(const Node *node, size_t n)`

  - This function takes an operator `Node *node` and returns the layout requirements of the Nth input `n`.
  - It returns Common layout constraints, for example, the input of `TransposeNode` is the same as layout of operation's result producing it.

- `std::string getNthResultLayoutRequirements(const Node *node, size_t n)`

  - This function takes an operator `Node *node` and returns the layout requirements of the Nth result `n`.
  - It returns Common layout constraints, for example, `ConvolutionNode` should be in `NHWC` format.

Notes:

1. Some nodes can accept any layout as input, they are either data parallel, e.g. `Add`,
or, while not data parallel, do not care about the order of dimensions for their operation,
e.g. `ReshapeNodeKind`. When adding new nodes to Glow, such a behavior should be explicitly
specified, by adding `.dataParallel()` in NodeGen for example.
2. Some nodes propagate the layout information of their input, e.g. `convertTo` node,
when adding such nodes to Glow the canonical layout verifier should be aware of them.
We currently do that in `getNthInputLayoutRequirements`.

## Placeholders and Constants

An important thing to note is that some operators may have a `Placeholder` or
a `Constant` as their input. We may need to know a specific layout for said
storage. For example, a Placeholder may need to be in `NHWC` format for a
`ConvolutionNode`. However, we do not want to pollute the code by making
this a hard requirement, especially since the canonical layout may accept
anything for certain tensors (e.g. `1-D` tensor), as such, we introduce the
notion of `ANY_LAYOUT` and initialize them with this wildcard by default.
Note, that loaders have the ability to specify the layout based on network
description, e.g. they might accept either `NCHW` or `NHWC` as an input for
operator, and they can propagate that information to Glow.


## Related Work

Other machine learning frameworks introduced similar concepts, this is not a
proposal unique to Glow, here are some notable mentions:

### PlaidML

Provides layout requirement information as a parameter to operations that need
to know tensor layouts instead of setting a global layout that would apply to
every operation. Allowing users to mix layouts throughout their network.

PlaidML made the conscious decision to make the layout a property the operation
instead of the tensor, making the implementation of certain operations more
intuitive [^tl1].

### TVM

TOPI is the operator collection library for TVM [^tl2]. Certain TOPI operations
include their layout requirements as a string. Here's layout section of
`topi.nn.pool` taken from version 0.6 of the document:

> layout (string) – Layout of the input data. The layout is supposed to be composed
> of upper cases, lower cases and numbers, where upper case indicates a dimension
> and the corresponding lower case with factor size indicates the split dimension.
> For example, NCHW16c can describe a 5-D tensor of [batch_size, channel, height,
> width, channel_block], in which channel_block=16 is a split of dimension channel.


### XLA

XLA adds backend specific layout constraints. Their CPU backend requires
constant arrays to be column major when all of their users are dot operations [^tl3]. While
Their GPU backend adds layout constraints on the cudnn custom-call instruction [^tl4].

It is also worth taking a look at XLA's layout optimizer [^tl5], part of their
effort to improve the out-of-the-box TensorFlow performance [^tl6].

Another thing to note is that their alter layout pass [^tl7] is similar,
in function, to the "Solver" we propose to automatically legalizes layouts in
the future work section of this document.

### MLIR

Does not currently have such support, but there are ongoing discussions to add such
support to MLIR Tensor Type [^tl8].

## Future Work

There are a few neat things we can, and probably should, do to expand this support:

### Remove `enum ConvolutionLayout`

Our string based representation is more generic and extendable as it is basically an
extendable enum that can be used in the backends without touching the generic code base.

### Remove shuffle arrays

Some operations, such as `TransposeNode`, have a shuffle that tells them what to do.
This can be deprecated and automatically deduced by specifying layout constraints.

There is some discrepancy is the fact that with currently use both typed tensor,
with named dimensions, and explicitly indexed dimensions like we currently do
everywhere in the code base, shuffle arrays being an example of that, This
may lead to potential inconsistency in certain cases.
We should gradually migrate towards typed tensors in the long run.

### Introduce a "Solver" that automatically legalizes layouts

Said solver will drastically reduce the complexity of loading models from other frameworks:
We no longer need to insert transposes based on if we are importing `NHWC` or `NCHW`.
We just need to annotate the `Placeholder` with the layout information we've get at load-time,
and which we "forget" afterwards, and let the solver transpose said `Placeholder` to our
canonical layout.

First we will start with a "raw" state of non compliance, Then we have a loop to sink and
clamp layout transformations together.

### Remove backend specific nodes

Today, Glow core and custom backends implicitly hard-code this knowledge about the operations
into (backend-specific) nodes and code that works with them. This is pretty fragile and
involves a lot of boiler plate code.

Combining the proposed solver with the backend-specified layout constraints would improve
this situation considerably:

- The backend would return this information and Glow core could insert all the required layout transformations

- The transformations can also be optimized "for free": Glow currently optimizes `TransposeNode`:
 - Multiple transposes can be combined into one
 - Opposite transposes can eliminate each other

- The functionality to insert the required layout transforms is handled by the Glow core,
which removes a lot of code duplication from backends.

[^tl0]: [Glow Issue: Fix bug in constant folding optimization](https://github.com/pytorch/glow/issues/3500)

[^tl1]: [Tensor Layout Design Decision in PlaidML](https://github.com/plaidml/plaidml/blob/master/plaidml2/op/lib/design.md#tensor-layout)

[^tl2]: [TVM Operator Inventory](https://docs.tvm.ai/api/python/topi.html)

[^tl3]: [XLA CPU Layout Assignment](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/cpu/cpu_layout_assignment.cc)

[^tl4]: [XLA GPU Layout Assignment](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.cc)

[^tl5]: [XLA Layout optimizer](https://github.com/tensorflow/tensorflow/blob/b6f7ce2b98b496886be4d900a6f88c24ae730f2c/tensorflow/core/grappler/optimizers/layout_optimizer.cc)

[^tl6]: [TensorFlow Graph Optimizations](https://web.stanford.edu/class/cs245/slides/TFGraphOptimizationsStanford.pdf)

[^tl7]: [XLA Alter Layout](https://github.com/dmlc/tvm/blob/025a6c8077cd1914bdd4132c6b86de007151344e/src/relay/pass/alter_op_layout.cc)

[^tl8]: [Proposal to add layout attribute to MLIR Tensor Type](https://groups.google.com/a/tensorflow.org/forum/#!topic/mlir/sCaIEKm2RxA)
