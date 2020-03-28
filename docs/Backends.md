## Backends in Glow

There are two directories used by backends in Glow:

1. [tools/ClassGen/Backends/](https://github.com/pytorch/glow/tree/master/tools/ClassGen/Backends):
Each backend directory here contains new
[backend-specific](#backend-specific-nodes-and-instructions-transformations)
Nodes and Instructions for the backends. If a backend provides its own
backend-specific nodes/instructions, they should be included in
[NodeGen](https://github.com/pytorch/glow/blob/master/tools/ClassGen/NodeGen.cpp)/[InstrGen](https://github.com/pytorch/glow/blob/master/tools/ClassGen/InstrGen.cpp).

2. [lib/Backends/](https://github.com/pytorch/glow/tree/master/lib/Backends): The
implementation of the backend is contained here. This includes derived classes
for [`Backend`](#backend-abstract-class) and
[`CompiledFunction`](#compiledfunction-abstract-class).
Each backend needs to be registered through its own registration factory in
order to be discovered by Glow, see [CPUBackend for example]
(https://github.com/pytorch/glow/blob/master/lib/Backends/CPU/CPUFactory.cpp).
And all factories must be linked to the Backends library, see
[here](https://github.com/pytorch/glow/blob/master/lib/Backends/CMakeLists.txt).

### `Backend` Abstract Class

All backends in Glow derive from the [abstract base class
`Backend`](https://github.com/pytorch/glow/blob/master/include/glow/Backend/Backend.h). There
are two pure virtual functions all backends must implement:

- `virtual std::unique_ptr<CompiledFunction> compile(Function *F) const;`

  - This function takes a `Function *F` to compile with default
  [`BackendOptions`](#backendoptions-helper-struct). It should return a unique pointer to the
    [`CompiledFunction`](#compiledfunction-abstract-class) of `F`. If the backend uses Glow low-level IR, it can call `generateAndOptimizeIR()` to generate an optimized `IRFunction`.

- `virtual std::unique_ptr<CompiledFunction> compile(Function *F, BackendOptions &opts) const;`

  - This function takes a `Function *F` and the provided
  [`BackendOptions`](#backendoptions-helper-struct). It should return a unique pointer to the
    [`CompiledFunction`](#compiledfunction-abstract-class) of `F`. If the backend uses Glow low-level IR, it can call `generateAndOptimizeIR()` to generate an optimized `IRFunction`.

- `virtual std::vector<std::unique_ptr<CompiledFunction>> compileFunctions(llvm::ArrayRef<Function *> functions, BackendOptions &opts) const;`
    - This function takes an `ArrayRef` of `Function *`s and compiles them using the same `BackendOptions` object for all functions. This allows the compiler to reason over things like shared constants between functions.

- `virtual bool isOpSupported(const NodeInfo &NI) const;`

  - Returns whether the backend can execute a node with given NodeInfo `NI`,
    containing the node kind and input and output types. For example, a backend
    may not support a specific bit-width quantization kind (e.g. `Int16QTy`) at
    all, or may only support it for certain operations
    (e.g. `ConvolutionNodeKind`). Any `(opKind, inputTypes, outputTypes)` passed
    in and returns true must be supported by the backend during `compile()` and
    `execute()`.

Additionally, there are several virtual functions that backends can override:

- `virtual Expected<bool> transformPostLowering(Function *F, CompilationContext &cctx) const;`

  - Allow the backend to transform the `Function *F` after [node
    lowering](https://github.com/pytorch/glow/blob/master/docs/IR.md#node-lowering)
    occurs, given the `CompilationContext`. For example, the CPU backend
    prefers to transform MaxNodes, which take a SplatNode as an input, into a
    [backend-specific](https://github.com/pytorch/glow/blob/master/docs/NewBackendSpecificNode.md)
    CPUMaxSplatNode, which takes a scalar value as a member input instead of a
    SplatNode. This should be done after node lowering, as ReluNodes are lowered
    into MaxNodes. See
    [below](#backend-specific-nodes-and-instructions-transformations) for more
    information.

- `virtual bool acceptForExecution(const NodeInfo &NI) const;`

  - Returns whether the backend would like to accept NodeInfo for execution. By
    default this falls back to checking for support via
    `Backend::isOpSupported()`, however this allows the backend to override to
    also take into account things like performance considerations.

- `virtual bool verify(const Function &F) const;`

  - Verifies that `Function &F` conforms to the backend-dependent graph constraints.

- `virtual bool verify(const IRFunction &IR) const;`

  - Verifies that `IRFunction &IR` conforms to the backend-specific constraints.

- `virtual TensorLayoutCommon &getTensorLayoutRequirements() const;`

  - Gets the backend-specific tensor layout requirements.

- `virtual bool shouldLower(const Node *N) const;`

  - Allow the backend to prevent lowering for some `Node *N`. For example, if a
    backend wants to fuse a `ReluNode` into a `ConvNode` to create some
    backend-specific node `ConvReluNode`, then it may prevent lowering for
    `ReluNode`. Then during `transformPostLowering()` it can look for patterns
    of `ConvNode` followed by `ReluNode` to swap out for `ConvReluNode`. Another
    example is if a backend supports executing a FullyConnected operator, it
    would want to prevent lowering for it and provide a backend-specific
    instruction for the FullyConnectedNode to be
    [IRGen'd](https://github.com/pytorch/glow/blob/master/docs/IR.md#low-level-ir)
    into. Note that IRGen for a Node can be specified via the
    [ClassGen](https://github.com/pytorch/glow/blob/master/docs/ClassGen.md)
    `autoIRGen("NodeName")` call. See
    [below](#backend-specific-nodes-and-instructions-transformations) for more
    information. Returns true if `N` should be lowered.

- `virtual bool shouldShareBuffers() const;`

  - Allow the backend to disable the buffer-sharing optimization. This may be
    preferred by backends which would like to do their own memory
    optimizations. Returns true by default.

- `virtual void save(Function *F, llvm::StringRef outputDir,
                     llvm::StringRef bundleName, llvm::StringRef mainEntryName) const;`

  - Save a [standalone executable
    bundle](https://github.com/pytorch/glow/blob/master/docs/AOT.md), where the
    provided `Function *F` is compiled and then saved to `outputDir` with bundle
    name `bundleName` and main entry name `mainEntryName`.

- `virtual bool generateInst(Node *N, IRGenVisitor &irgen) const;`

  - Allow the backend to perform custom lowering from Node to Instruction IR.
    Returns true if lowering is performed, false otherwise.

- `virtual FunctionPassPipeline getOptimizationPipeline() const;`

  - Allows the backend to customize the graph optimizations that are performed
    when compiling a Function. Backend returns the
    "DefaultGraphOptimizationPassPipeline", which contains nearly all of the
    optimizations discussed
    [here](Optimizations.md#set-of-supported-graph-optimizations). More
    information on how to configure this pipeline can be found
    [here](Optimizations.md#configuring-a-graph-optimization-pipeline).


### `CompiledFunction` Abstract Class

`CompiledFunction` is an abstract class that represents the result of
compilation of a `Function`. Backends must implement their own derived class
from `CompiledFunction`, which must be returned as a result of
`Backend::compile()` or `Backend::compileWithoutConstants()` .
 `CompiledFunction` contains a pure virtual function
that must be implemented: `virtual void execute();`. This function is
responsible for copying inputs to the device from all input
[Placeholders](https://github.com/pytorch/glow/blob/master/docs/IR.md#placeholders),
executing the function, and copying outputs back from the device to output
Placeholders. The `CompiledFunction` contains a [RuntimeBundle](#runtimebundle-helper-class)
which contains the symbol information and mappings of inputs and outputs. Thus after the
function returns, all Placeholders for the outputs of the function should have had
their backing tensor updated.
An optional method: `virtual void freeCompilationResources()` can be implemented to allow
freeing resources that are no longer needed after the function has been loaded on a device.

### `RuntimeBundle` Helper Class

`RuntimeBundle` is a helper class that contains the symbol information and collection
of constants needed at runtime. This allows a function to be compiled without being linked
to a context, and allows the `Function` to be freed after compilation. The symbol information
is stored in a table where the key is the symbol name and the payload contains symbol information
including, size, offset, type, and whether it is an input or output of the function. `RuntimeBundle` also  contains a pointer that may point to a block of memory that contains the constants for the `CompiledFunction` if that `Backend` uses it.

### `BackendOptions` Helper Struct

`BackendOptions` is a helper struct that contains the options relevant to the backend for compilation. The options include:
- `bool collectConstants` - Whether constants should be collected and stored in the `CompiledFunction`'s [RuntimeBundle](#runtimebundle-helper-class). Default: True
- `bool autoInstrument` - Whether `TraceEvents` should be inserted for profiling. Default: False

## Backend-Specific Nodes and Instructions Transformations

Different backends may prefer to transform or optimize the graph differently for
their own specialized architecture. For example, Glow lowers ReLU down to a Max
node, taking as inputs the original tensor and a "Splat" tensor of matching
dimensions, filled with all `0`s. Glow's CPU JIT backend prefers to replace this
pattern -- a Max with a Splat input and another non-Splat input -- with a single
"CPUMaxSplat" operation that takes a scalar Splat value as input in place of an
entire Splat tensor.

### Backend-Specific Transformation

Backends have the opportunity to perform their own analysis and transformations
after lowering. This is exposed via the `transformPostLowering()` hook, during
which a backend can transform the graph however it desires. For example, the
backend could use `transformPostLowering()` to search the graph looking for the
above `CPUMaxSplat` pattern.

#### Backend-Specific Nodes and Instructions

A backend may create its own custom Nodes and Instructions which it can insert
into the IR. This is done via [ClassGen](ClassGen.md) and implicitly included in
`tools/ClassGen/NodeGen.cpp` and `tools/ClassGen/InstrGen.cpp`.
These new nodes and instructions should be defined
inside the backend sub-directory, in files
`lib/Backends/<BackendName>/ClassGen/<BackendName>SpecificNodes.h` and
`lib/Backends/<BackendName>/ClassGen/<BackendName>SpecificInstrs.h`:

For example, the CPU Backend defines `CPUMaxSplat`
in `lib/Backends/CPU/ClassGen/CPUSpecificNodes.h`:

```cpp
BB.newBackendSpecificNode("CPUMaxSplat")
    .addInput("Input")
    .addResult("Input.getType()")
    .addMember(MemberType::Float, "SplatValue")
    .setDocstring("A Max node with one splat input; CPU specific.");
```

If tensor layout requirements are enabled for the backend, on should take
special care of updating the layout verifier when adding a new node.
See `TensorLayout.md` for more information.
To extend the example above, if the new node is data parallel, a `.dataParallel()`
line should be added.

During `transformPostLowering()`, this `CPUMaxSplat` node replaces the
aforementioned pattern. However, there must be a corresponding instruction for
this Node to be lowered to during the IRGen phase. Thus, we need a corresponding
backend-specific CPUMaxSplat instruction, defined in
`lib/Backends/CPU/ClassGen/CPUSpecificInstrs.h`:

```
BB.newBackendSpecificInstr("CPUMaxSplat")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "SplatValue")
    .inplaceOperand({"Dest", "Src"})
    .dataParallel()
    .autoIRGen();
```

These instructions will appear in the instruction stream sent to the CPU backend
JIT; its [standard library](JIT.md#usage-of-the-standard-library) has a kernel
for executing this `CPUMaxSplat` instruction. You can see such instructions in
the [LeNet MNIST example](Example.md#lowering-to-ir).

Note that backend-specific nodes and instructions can be treated just as any
other node or instruction defined in `tools/ClassGen/NodeGen.cpp` or
`tools/ClassGen/InstrGen.cpp`. For example, the `CPUMaxSplat` instruction
definition includes the `dataParallel()` property, allowing for data parallel
optimizations to take place.

The `lib/Backends/CPU/ClassGen/CPUSpecificNodes.h` and
`lib/Backends/CPU/ClassGen/CPUSpecificInstrs.h` files are implicitly included in
`tools/ClassGen/NodeGen.cpp` and `tools/ClassGen/InstrGen.cpp`, respectively.

#### Backend Parameterized Tests

Glow provides several test suites that are parameterized by backend.  An
example of such a suite is `tests/unittests/OperatorTest.cpp`, which defines
small tests of Glow operators.  These tests can be executed against any backend
to check compliance.

These tests will only be run for a backend if a corresponding
`lib/Backends/$BACKEND/tests` directory is found and contains a corresponding
`${BACKEND}${TEST}.cpp` file containing a blacklist definition, e.g.:
```
std::set<std::string> glow::backendTestBlacklist = {};
```

This blacklist can be used to exclude any unsupported tests while a backend is
a work-in-progress.  See the Interpreter and CPU backends for examples of
setting up and using these tests.  To bootstrap a blacklist, we recommend using
a simple shell script to check which tests already work:
```
for test in $(tests/ExampleBackendOperatorTest --gtest_list_tests); do
  if ! tests/ExampleBackendOperatorTest --gtest_filter="$test" >& /dev/null; then
    echo $test
  fi
done
```

#### External backends

External backends can be added to Glow without changing the Glow build infrastructure.

An external backend is provided as a single source directory. It can then be developed in a separate source management repository.

The external backend mechanism is for instance convenient for adding closed-source backends to Glow.

The structure of external backends is defined [here](https://github.com/pytorch/glow/blob/master/docs/ExternalBackend.md).
