/**
 * Copyright (c) 2018-present, Facebook, Inc.
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
#include "glow/Converter/FunctionConverter.h"

#include "glow/Graph/Graph.h" // For Function.
#include "glow/Graph/Node.h"  // For Node.
#include "glow/Graph/Nodes.h" // For Placeholder and Constant.
#include "glow/Graph/PlaceholderBindings.h"

#include "llvm/ADT/DenseMap.h"

using namespace glow;

TypeRef
FunctionConverter::getTargetTypeForOutput(const NodeValue &nodeVal) const {
  // Default implementation says there is nothing to do.
  return nullptr;
}

TypeRef FunctionConverter::getTargetTypeForInput(const Node &use,
                                                 unsigned idx) const {
  // Default implementation says there is nothing to do.
  return nullptr;
}

bool FunctionConverter::canConvert(const Node &node) const {
  // By default, we assume everything is convertible.
  switch (node.getKind()) {
  default:
    return true;
  case Kinded::Kind::PlaceholderKind:
  case Kinded::Kind::SaveNodeKind:
    // Save node or placeholder special because
    // they are or their effects are visible from
    // the outside of the function being converted.
    // Thus, we cannot convert them, unless we change
    // the semantic of this function and the related
    // placeholder.
    return false;
  }
}

NodeValue FunctionConverter::getConversionOutput(Node &conversion) const {
  assert(conversion.getNumResults() == 1 && "This method should be overloaded");
  return NodeValue(&conversion, 0);
}

Node &FunctionConverter::morphNode(Node &node) { return node; }

void FunctionConverter::postProcessing(Node &node) {}

void FunctionConverter::convertOutputs(Node &node) {
  using FunctionAndValIdx = std::pair<Function *, unsigned>;
  llvm::DenseMap<FunctionAndValIdx, NodeValue> functionAndValToConversion;
  for (unsigned idx = 0, end = node.getNumResults(); idx != end; ++idx) {
    NodeValue val = node.getNthResult(idx);
    TypeRef targetTy = getTargetTypeForOutput(val);
    if (!targetTy || targetTy == val.getType()) {
      continue;
    }
    // convert the node and create a conversion to keep the users happy.
    assert(targetTy->dims() == val.getType()->dims() &&
           "Conversion does not preserve shape");
    TypeRef origTy = val.getType();
    // Fake the morphing of the node so that the creation
    // of the conversion works properly.
    val.setType(targetTy);
    // Store the users in a temporary object because setOperand
    // will invalidate the iterator.
    llvm::SmallVector<NodeUse, 4> users(val.getUsers().begin(),
                                        val.getUsers().end());
    // We cannot use replaceAllUsesWith here because:
    // 1. At this point, val and conversion don't have the same type
    //    (one is converted the other is the original type), and that
    //    would trigger an assertion.
    // 2. We would end up replacing the use of val in "conversion" by
    //   "conversion".
    // 3. Node may be a module level value and we need one conversion per
    //    function.
    for (auto use : users) {
      Node *user = use.getUser();
      Function *parent = user->getParent();
      assert(parent && "User not in a function?!");

      SaveNode *saveNode = llvm::dyn_cast<SaveNode>(user);
      // The output of save nodes is special because it doesn't use
      // the value of the node, but its address.
      // Thus, if we want to change the value of the output of
      // a save node, we actually have to convert the input.
      if (saveNode && saveNode->getOutput() == val) {
        NodeValue input = saveNode->getInput();
        Node *conversion = createConversion(*parent, input, targetTy);
        saveNode->setNthInput(SaveNode::InputIdx,
                              getConversionOutput(*conversion));
        continue;
      }

      FunctionAndValIdx functionAndVal = std::make_pair(parent, idx);
      auto conversionValIt = functionAndValToConversion.find(functionAndVal);
      if (conversionValIt == functionAndValToConversion.end()) {
        // Create the conversion.
        Node *conversion = createConversion(*parent, val, origTy);
        // "conversion" uses val so after this call,
        // we will get a use of conversion inside conversion.
        NodeValue conversionVal = getConversionOutput(*conversion);
        auto insertion =
            functionAndValToConversion.insert({functionAndVal, conversionVal});
        assert(insertion.second && "Conversion already there?!");
        conversionValIt = insertion.first;
      }

      NodeValue conversionVal = conversionValIt->second;
      if (user == conversionVal.getNode()) {
        continue;
      }
      // Log the change of node input(operand).
      if (Function *F = node.getParent()) {
        F->getLogContext()->logNodeInputChange(*user, *(use.get()),
                                               conversionVal);
      }

      use.get()->setOperand(conversionVal.getNode(), conversionVal.getResNo());
    }
  }
}

void FunctionConverter::convertInputs(Node &node) {
  // We shouldn't have to convert the inputs of something that is not in
  // function_.
  assert((node.getNumInputs() == 0 || node.getParent() == &function_) &&
         "Invalid requested conversion");
  for (unsigned idx = 0, end = node.getNumInputs(); idx != end; ++idx) {
    NodeValue val = node.getNthInput(idx);
    TypeRef targetTy = getTargetTypeForInput(node, idx);
    if (!targetTy || targetTy == val.getType()) {
      continue;
    }
    // convert the node and create a conversion to keep the users happy.
    assert(targetTy->dims() == val.getType()->dims() &&
           "Conversion does not preserve shape");
    // Create the conversion.
    Node *conversion = createConversion(function_, val, targetTy);
    node.setNthInput(idx, getConversionOutput(*conversion));
  }
}

void FunctionConverter::convert() {
  assert(function_.verify() && "Input function must be valid");

  // Traverse all nodes.
  // Check what the conversion should look like, if any.
  // Convert the node appropriately.

  // For every unprocessed node in the graph we keep the invariant of having
  // all inputs to be of the uncovered type.
  // I.e., if we have:
  // res(outTy) = node arg1(in2Ty), arg2(in2Ty)
  //
  // after converting "node", we will have something that looks like:
  // newArg1(convertedIn1Ty) = conversion arg1
  // newArg2(convertedIn2Ty) = conversion arg2
  // newRes(convertedOutTy) = node newArg1, newArg2
  // res(outTy) = conversion newRes
  //
  // In other words, the boundaries (in and out) are unchanged.

  // The iterator looks weird because we only want to iterate through
  // the original nodes.
  auto nodeIt = function_.getNodes().end();
  auto stopIt = function_.getNodes().begin();
  do {
    --nodeIt;
    Node &node = *nodeIt;
    if (!canConvert(node)) {
      continue;
    }
    // Mutate the output types and insert the conversion to keep our
    // invariant.
    convertOutputs(node);
    // Convert the inputs of the node.
    convertInputs(node);
    // All the surrounding code is properly typed, finally the morph node.
    Node &morphedNode = morphNode(node);
    // Do some post processing if need be.
    postProcessing(morphedNode);
  } while (nodeIt != stopIt);

  // Allow a late clean-up before verifying the conversation produced a valid
  // function.
  cleanUp();

  assert(function_.verify() && "Conversion led to invalid function");
}

void FunctionConverter::convertPlaceholder(Placeholder &placeholder,
                                           PlaceholderBindings *bindings) {
  TypeRef destTy = getTargetTypeForOutput(placeholder.getOutput());
  if (!destTy || destTy == placeholder.getType()) {
    return;
  }
  convertOutputs(placeholder);
  if (!bindings) {
    return;
  }
  Tensor *tensor = bindings->get(&placeholder);
  if (tensor) {
    convertTensor(*tensor, destTy);
  }
}
