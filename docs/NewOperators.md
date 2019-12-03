## How to implement a new operator in Glow

*Note: It's best to avoid implementing new Glow IR nodes and instructions. If support for a new operator is needed, it's best to first check to see if this operator can be implemented using an existing glow IR node or nodes. For example FCTransposed can be implemented using a regular FC node. If this is the case then the operator can be implemented by instantiating existing glow IR nodes in the loader or by simply creating a new node creation method in `Graph/Graph.cpp` which would create the sequence of operators that implements the new operator. Only in the event that this is not possible should new Glow IR nodes/instructions be created.*

> Example PR: https://github.com/pytorch/glow/pull/2334

### Implementation
#### High level IR
* Create a new Glow high level IR node in `ClassGen/NodeGen.cpp`. Run `ninja all` to generate the node. In the build directory, check `glow/AutoGenNodes.h` to ensure the node has been generated.
* Implement the `verify()` method for the new node in `Graph/Nodes.cpp`.
* Implement any node layout requirements, if any, see `TensorLayout.md` for details.
Specifically see the notes section under `Canonical Tensor Layout`.
* Implement a node creation method in `Graph/Graph.cpp`.
* Implement logic to load model that contains the operator in `Importer/Caffe2ModelLoader.cpp` or `Importer/ONNXModelLoader.cpp` depending on which type of model the operator comes from. Add the operator to `Importer/CommonOperatorLoader.h` instead if the loading logic can be shared between Caffe2 and ONNX. Add as much validation logic as possible here in the loader for the operator because it's crucial to catch errors at this stage. Once the operator is loaded, it is assumed that Glow will be able to successfully run the operator so any issues must be caught here.
#### Low level IR
* Create a new Glow low level IR instruction in `ClassGen/InstrGen.cpp`.
* If a custom translation from high level IR node to low level IR instruction is necessary then implement a new case in `IR/IRGen` otherwise just use `autoIRGen(name)` in the definition of the new IR instruction in `ClassGen/InstrGen.cpp`. `autoIRGen(name)` will create code to automatically generate IR at compile time from the high level IR node with the name `name`. If the high level IR node as the same name as the low level instruction then the `name` parameter can be omitted.
* If the new node can be natively supported on some backends but not all backends then consider implementing a lowering in `Optimizer/Lower.cpp` to lower the high level IR node to other high level IR nodes instead of generating the low level IR instruction for the backends which cannot natively implement the new instruction.
* Implement the interpreter implementation of the instruction in `Interpreter/InterpreterNodes.cpp`. An implementation for interpreter backend is required for all instructions but implementations for other backends are encouraged as well. For example, it's strongly encouraged to also create a CPU implementation for each instruction because of the significantly greater performance of the CPU backend over the interpreter.

### Testing
* Test the operator implementation by writing unit tests in `unittests/OperatorTest.cpp`.
* Test the operator importing logic by creating a new model in `tests/models/caffe2Models` or `tests/models/onnx2Models` directory then write unit tests in `unittests/Caffe2ImporterTest.cpp` or `tests/unittests/OnnxImporterTest.cpp` using that model.

### PR
* [Create a pull request](../PULL_REQUEST.md) and prefix the title with "[New operator]".
