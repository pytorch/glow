/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ExecutorTestBuilder.h"
#include "ExecutorBenchmarkWrapper.h"
#include "ExecutorUnitTestWrapper.h"
#include "TestDeviceManager.h"

using namespace glow;
using namespace glow::runtime;

void ExecutorTestBuilder::addNode(const std::string &name, DeviceIDTy deviceId,
                                  llvm::ArrayRef<llvm::StringRef> parents,
                                  llvm::ArrayRef<llvm::StringRef> inputs,
                                  llvm::ArrayRef<llvm::StringRef> outputs,
                                  RunIdentifierTy runId, bool success) {
  auto newNode = llvm::make_unique<DAGNode>();
  auto *newNodeRawPtr = newNode.get();

  // If this is the first node being added, record the run ID for the graph.
  // Otherwise, make sure that the runId matches that of the previous nodes.
  if (nodes_.empty()) {
    runId_ = runId;
  } else {
    assert(runId == runId_ && "Node run ID does not match rest of graph!");
  }

  // If the result for this node is false, set the expected
  // result for the entire test to false.
  success_ &= success;

  // Add parents to the list of parents in the new node and add the newNode
  // to the list of children in the parents. If the parent list is empty,
  // make the root the only parent. Also, update the set of known leaves
  // by removing any parents of the new node from it. This will be useful
  // later.
  if (!parents.empty()) {
    for (const auto &parent : parents) {
      auto it = nodes_.find(parent);
      if (it == nodes_.end()) {
        assert(!"Parent specified for node not found!");
      }
      DAGNode *parentPtr = (it->second).get();
      (newNode->parents).emplace_back(parentPtr);
      (parentPtr->children).emplace_back(newNodeRawPtr);
      leaves_.erase(parentPtr);
    }
  } else {
    (newNode->parents).emplace_back(root_.get());
    (root_->children).emplace_back(newNode.get());
  }

  // Iterate through inputs and outputs and:
  // 1) Create Placeholders and Tensors for inputs/output names that have not
  //    been mapped to a Placeholder yet.
  // 2) Assemble the input ExecutionContexts that the node is expected to be
  // called with
  //    and the ExecutionContexts that the node should produce as output.
  // 3) Generate the symbol table for the new node by generating
  //    RuntimeSymbolInfo objects for each input and output.
  SymbolTableTy symbolTable;
  size_t offset = 0;

  auto nodeInputContext = llvm::make_unique<ExecutionContext>();
  auto nodeOutputContext = llvm::make_unique<ExecutionContext>();

  for (const auto &input : inputs) {
    insertSymbolIntoPlaceholderBindings(
        input, nodeInputContext->getPlaceholderBindings());

    RuntimeSymbolInfo runtimeSymbolInfo;
    runtimeSymbolInfo.size = type_->getSizeInBytes();
    runtimeSymbolInfo.offset = offset;
    runtimeSymbolInfo.type = *type_;
    runtimeSymbolInfo.input = true;
    runtimeSymbolInfo.output = false;
    symbolTable.insert(std::make_pair(input, runtimeSymbolInfo));
    offset += type_->getSizeInBytes();
  }

  for (const auto &output : outputs) {
    insertSymbolIntoPlaceholderBindings(
        output, nodeOutputContext->getPlaceholderBindings());

    RuntimeSymbolInfo runtimeSymbolInfo;
    runtimeSymbolInfo.size = type_->getSizeInBytes();
    runtimeSymbolInfo.offset = offset;
    runtimeSymbolInfo.type = *type_;
    runtimeSymbolInfo.input = false;
    runtimeSymbolInfo.output = true;
    symbolTable.insert(std::make_pair(output, runtimeSymbolInfo));
    offset += type_->getSizeInBytes();
  }

  // Set the name, device ID, and RuntimeBundle of the new node.
  newNode->name = name;
  newNode->deviceID = deviceId;
  newNode->runtimeBundle =
      RuntimeBundle(symbolTable, /*constWeight=*/0, /*mutableWeight=*/0,
                    /*activations=*/0);

  // Register node result with the appropriate DeviceManager.
  auto it = deviceManagers_.find(deviceId);

  if (it == deviceManagers_.end()) {
    assert(!"No test device manager found for this device ID");
  }

  auto *deviceManagerPtr = it->second.get();
  auto testDeviceManagerPtr =
      static_cast<TestDeviceManager *>(deviceManagerPtr);

  bool registered = testDeviceManagerPtr->registerResult(
      name, runId, success, std::move(nodeInputContext),
      std::move(nodeOutputContext));

  (void)registered;
  assert(registered && "Node registration was not successful");

  // Add the new node to nodes_ and leaves_.
  nodes_.insert(std::make_pair(name, std::move(newNode)));
  leaves_.insert(newNodeRawPtr);
}

template <class TestType>
std::unique_ptr<TestType> ExecutorTestBuilder::emitTest() {
  // Get the input and output symbol names for the whole DAG.
  std::vector<std::string> inputSymbols = gatherInputSymbols();
  std::vector<std::string> outputSymbols = gatherOutputSymbols();

  // Generate the input and output ExecutionContexts for the test. This
  // input ExecutionContexts contains the input Placeholders of all root
  // nodes and output Placeholders of all leaves (but backed by zero tensors).
  // This is the ExecutionContexts that needs to be passed to
  // Executor::run() to run the test. The output ExecutionContexts contains
  // the same Placeholders as the input ExecutionContexts, but the leaves'
  // output Placeholders are mapped to their expected output Tensors. This
  // ExecutionContext is used to verify that the one returned by the
  // Executor is correct.
  auto inputContext = llvm::make_unique<ExecutionContext>();
  auto outputContext = llvm::make_unique<ExecutionContext>();

  for (const auto &symbol : inputSymbols) {
    insertSymbolIntoPlaceholderBindings(symbol,
                                        inputContext->getPlaceholderBindings());
    insertSymbolIntoPlaceholderBindings(
        symbol, outputContext->getPlaceholderBindings());
  }

  for (const auto &symbol : outputSymbols) {
    auto *placeholder = bindings_->getPlaceholderByName(symbol);
    if (!placeholder) {
      assert(!"Placeholder for DAG output not found!");
    }
    inputContext->getPlaceholderBindings()->allocate(placeholder)->zero();
    insertSymbolIntoPlaceholderBindings(
        symbol, outputContext->getPlaceholderBindings());
  }

  // Create the test object.
  auto test = llvm::make_unique<TestType>(
      executor_, std::move(root_), std::move(type_), std::move(nodes_),
      std::move(placeholders_), std::move(inputContext),
      std::move(outputContext), runId_, success_);

  // Reset builder state to allow a new test to be constructed with this
  // instance.
  root_ = llvm::make_unique<DAGNode>();
  bindings_->clear();
  type_ = std::unique_ptr<Type>(new Type(ElemKind::FloatTy, {1, 2, 2}));
  nodes_.clear();
  leaves_.clear();
  placeholders_.clear();
  success_ = true;

  return test;
}

// Explicit instantiations of ExecutorTestBuilder::emitTest().
template std::unique_ptr<ExecutorUnitTestWrapper>
ExecutorTestBuilder::emitTest<ExecutorUnitTestWrapper>();
template std::unique_ptr<ExecutorBenchmarkWrapper>
ExecutorTestBuilder::emitTest<ExecutorBenchmarkWrapper>();

std::vector<std::string> ExecutorTestBuilder::gatherInputSymbols() const {
  std::vector<std::string> inputSymbols;

  // Input symbols for the entire test are the inputs of all nodes that have
  // no parents.
  for (const auto &node : root_->children) {
    const SymbolTableTy &symbolTable = (node->runtimeBundle).getSymbolTable();

    for (const auto &symbolPair : symbolTable) {
      const auto &symbolName = symbolPair.first;
      const auto &symbolInfo = symbolPair.second;

      if (symbolInfo.input) {
        inputSymbols.emplace_back(symbolName);
      }
    }
  }

  return inputSymbols;
}

std::vector<std::string> ExecutorTestBuilder::gatherOutputSymbols() const {
  std::vector<std::string> outputSymbols;

  // Input symbols for the entire test are the outputs of all nodes that have
  // no children.
  for (const auto &node : leaves_) {
    const SymbolTableTy &symbolTable = (node->runtimeBundle).getSymbolTable();

    for (const auto &symbolPair : symbolTable) {
      const auto &symbolName = symbolPair.first;
      const auto &symbolInfo = symbolPair.second;

      if (symbolInfo.output) {
        outputSymbols.emplace_back(symbolName);
      }
    }
  }

  return outputSymbols;
}

void ExecutorTestBuilder::insertSymbolIntoPlaceholderBindings(
    llvm::StringRef name, PlaceholderBindings *bindings) {
  auto it = placeholders_.find(name);

  if (it == placeholders_.end()) {
    // This is a new symbol. Create a Placeholder and an initialize and new
    // Tensor for it.
    auto placeholder = llvm::make_unique<Placeholder>(name, type_.get(),
                                                      /*trainable=*/false);
    auto *tensor = bindings_->allocate(placeholder.get());
    tensor->init(Tensor::InitKind::Xavier, 1.0, rng_);
    bindings->insert(placeholder.get(), tensor->clone());
    placeholders_[name] = std::move(placeholder);
  } else {
    // This is a symbol that already has an associated Placeholder and Tensor.
    // Copy that Tensor.
    auto *placeholder = (it->second).get();
    const auto *tensor = bindings_->get(placeholder);
    bindings->insert(placeholder, tensor->clone());
  }
}
