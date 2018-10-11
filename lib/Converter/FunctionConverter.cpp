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

#include "glow/Base/Tensor.h" // For Tensor.
#include "glow/Graph/Graph.h" // For Function.
#include "glow/Graph/Node.h"  // For Node.
#include "glow/Graph/Nodes.h" // For Placeholder and Variable.

using namespace glow;

FunctionConverter::FunctionConverter(Function &F)
    : mod_(*F.getParent()), function_(F) {}

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

NodeValue FunctionConverter::getConversionInput(Node &conversion) const {
  assert(conversion.getNumInputs() == 1 && "This method should be overloaded");
  return conversion.getNthInput(0);
}

Node &FunctionConverter::morphNode(Node &node) { return node; }

void FunctionConverter::postProcessing(Node &node) {}

void FunctionConverter::convert() {
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
      // Create the conversion.
      Node *conversion = createConversion(val, origTy);
      conversions_.insert(conversion);
      // "conversion" uses val so after this call,
      // we will get a use of conversion inside conversion.
      NodeValue conversionVal = getConversionOutput(*conversion);
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
      for (auto use : users) {
        if (use.getUser() == conversion) {
          continue;
        }
        use.get()->setOperand(conversionVal.getNode(),
                              conversionVal.getResNo());
      }
    }
    // Convert the inputs of the node.
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
      Node *conversion = createConversion(val, targetTy);
      conversions_.insert(conversion);
      node.setNthInput(idx, getConversionOutput(*conversion));
    }
    // All the surrounding code is properly typed, finally the morph node.
    Node &morphedNode = morphNode(node);
    // Do some post processing if need be.
    postProcessing(morphedNode);
  } while (nodeIt != stopIt);

  // Allow a late clean-up before verifying the conversation produced a valid
  // function.
  cleanUp();

  function_.verify();
}

bool FunctionConverter::canConvert(const Tensor &tensor, TypeRef dstTy) const {
  // The default implementation of convertTensor uses Tensor::convertToType
  // which does not support quantization for now.
  return !dstTy->isQuantizedType() && !tensor.getType().isQuantizedType();
}

void FunctionConverter::convertTensor(Tensor &tensor, TypeRef dstTy) {
  assert(canConvert(tensor, dstTy) && "We shouldn't call this method");
  assert(tensor.dims() == dstTy->dims() &&
         "Conversion should only change the values, not the shape");
  tensor.convertToType(dstTy->getElementType());
}

NodeValue FunctionConverter::convertConstant(Constant &constant,
                                             TypeRef dstTy) {
  const Tensor &tensor = constant.getPayload();
  if (!canConvert(tensor, dstTy)) {
    return NodeValue();
  }

  Constant *constantToBeModified = &constant;
  if (!constant.hasOneUse()) {
    // Could be a simple clone, but storage classes don't support cloning for
    // now.
    constantToBeModified =
        mod_.createConstant(constant.getType(), constant.getName());
    constantToBeModified->getPayload().assign(&constant.getPayload());
  }
  Tensor &tensorToBeModified = constantToBeModified->getPayload();
  constantToBeModified->setType(0, dstTy);
  convertTensor(tensorToBeModified, dstTy);
  return NodeValue(constantToBeModified, 0);
}

bool FunctionConverter::optimizeConversions() {
  // Traverse all the conversions introduced during the conversion step and
  // remove them.
  llvm::SmallPtrSet<Node *, 8> deadConversions;
  bool changed = false;
  for (Node *conversion : conversions_) {
    if (deadConversions.count(conversion)) {
      continue;
    }
    NodeValue conversionInput = getConversionInput(*conversion);
    NodeValue dstVal = getConversionOutput(*conversion);
    NodeValue srcVal;
    switch (conversionInput.getNode()->getKind()) {
    case Kinded::Kind::ConstantKind:
      srcVal = convertConstant(*llvm::cast<Constant>(conversionInput.getNode()),
                               dstVal.getType());
      // Reset conversionInput because it may not be valid anymore.
      conversionInput = NodeValue();
      break;
    default:
      // Check if the input of this conversion is a conversion that we know.
      if (!conversions_.count(conversionInput)) {
        break;
      }
      // So we have "conversion(conversion srcVal to tmpVal) to dstVal".
      // If the type of srcVal is equal to the type of dstVal, we can replace
      // the uses of dstVal with srcVal.
      // Note: This potentially changes the semantic of the program, for
      // instance if going through tmpVal implies a loss in precision. However,
      // since this method is not run by default that means the caller knows
      // what it is doing.
      srcVal = getConversionInput(*conversionInput.getNode());
      break;
    }
    // Check if we found a suitable new source for dstVal.
    if (srcVal == NodeValue() || srcVal.getType() != dstVal.getType()) {
      continue;
    }
    // Use srcVal instead of dstVal.
    dstVal.replaceAllUsesOfWith(srcVal, &function_);
    changed = true;
    bool inserted = deadConversions.insert(dstVal.getNode()).second;
    (void)inserted;
    assert(inserted && "Conversion was already dead");
    if (conversionInput != NodeValue() && conversionInput.hasOneUse()) {
      // The only user of conversionInput is outVal.
      // This conversion is now dead too.
      inserted = deadConversions.insert(conversionInput.getNode()).second;
      (void)inserted;
      assert(inserted && "Conversion was already dead");
    }
  }
  for (Node *conversion : deadConversions) {
    function_.eraseNode(conversion);
  }
  return changed;
}
