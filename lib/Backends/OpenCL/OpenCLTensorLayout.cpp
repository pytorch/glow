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
#include "OpenCLTensorLayout.h"
#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"

using namespace glow;

/// Definitions of different tensor layouts.
static std::string oclDimsNHWC[] = {
    {"N"},
    {"H"},
    {"W"},
    {"C"},
};
static std::string oclDimsNCHW[] = {
    {"N"},
    {"C"},
    {"H"},
    {"W"},
};
static TensorLayoutDescription oclLayoutNHWC(oclDimsNHWC);
static TensorLayoutDescription oclLayoutNCHW(oclDimsNCHW);

static std::string returnBaseReqOrNHWC(TensorLayoutDescription &baseReq,
                                       const Node *node) {
  if (!baseReq.isSameLayout(
          CanonicalTensorLayout::getInstance().getLayoutsForDims()[4])) {
    return baseReq.getSerializedLayout();
  }
  if (CanonicalTensorLayout::getInstance().acceptsAnyLayout(node)) {
    // These nodes accept any 4-D layout.
    return baseReq.getSerializedLayout();
  }
  // For Placeholders and Constants that were loaded from another tool, we don't
  // have the layout information in during load time. This makes us assume they
  // are in Glow's Canonical NHWC format, which is not correct if we are loading
  // an image with NCHW such as in resent loader. Weaken the verifier to avoid a
  // runtime crash: if this is a storage location, return the baseReq.
  if (llvm::dyn_cast<Storage>(node)) {
    return baseReq.getSerializedLayout();
  }

  return CanonicalTensorLayout::getInstance().getDefaultNDLayout(4);
}

/// Helper function, \returns either NHWC or NCHW layout based on the
/// instruction's layout enum. This will be removed and refactored if/when we
/// move to using strings for all layout specifications and get rid of the enum.
template <typename N>
static const TensorLayoutDescription *getLayoutFromEnum(const N &node) {
  if (node->getLayout() == NCHW) {
    return &oclLayoutNCHW;
  }
  return &oclLayoutNHWC;
}

/// \returns either NHWC or NCHW layout based on the instruction's layout enum
/// if it has one. Else returns nullptr. This will be removed and refactored
/// if/when we move to using strings for all layout specifications and get rid
/// of the enum.
static const TensorLayoutDescription *
getLayoutForTempEnumRep(size_t n, const Node *node) {
  if (const auto MP = llvm::dyn_cast<MaxPoolNode>(node)) {
    return getLayoutFromEnum(MP);
  }
  if (const auto MPG = llvm::dyn_cast<MaxPoolGradNode>(node)) {
    return getLayoutFromEnum(MPG);
  }
  if (const auto AP = llvm::dyn_cast<AvgPoolNode>(node)) {
    return getLayoutFromEnum(AP);
  }
  if (const auto APG = llvm::dyn_cast<AvgPoolGradNode>(node)) {
    return getLayoutFromEnum(APG);
  }

  if (const auto *CN = llvm::dyn_cast<ConvolutionNode>(node)) {
    switch (n) {
    case ConvolutionNode::InputIndices::BiasIdx:
      return &CanonicalTensorLayout::getInstance().getLayoutsForDims()[1];
    default: { return getLayoutFromEnum(CN); }
    }
  }
  return nullptr;
}

std::string OpenCLTensorLayout::getNthInputLayoutRequirements(const Node *node,
                                                              size_t n) {
  DCHECK_LT(n, node->getNumInputs()) << "Wrong input number";
  auto inputNode = node->getNthInput(n);
  auto dims = inputNode.getType()->dims();
  DCHECK_LE(dims.size(), max_tensor_dimensions) << "Too many dimensions";
  // TODO: Remove ->getLayout() enum and take a string like transpose. Refactor
  // the following after doing so.
  const auto *layout = getLayoutForTempEnumRep(n, node);
  if (layout) {
    return layout->getSerializedLayout();
  }
  auto baseReq = TensorLayoutCommon::getNthInputLayoutRequirements(node, n);
  auto baseReqHelper = TensorLayoutDescription(baseReq);
  return returnBaseReqOrNHWC(baseReqHelper, node);
}

std::string OpenCLTensorLayout::getNthResultLayoutRequirements(const Node *node,
                                                               size_t n) {
  DCHECK_LT(n, node->getNumResults()) << "Wrong output number";
  auto dims = node->getNthResult(n).getType()->dims();
  DCHECK_LE(dims.size(), max_tensor_dimensions) << "Too many dimensions";
  // TODO: Remove ->getLayout() enum and take a string like transpose. Refactor
  // the following after doing so.
  const auto *layout = getLayoutForTempEnumRep(n, node);
  if (layout) {
    return layout->getSerializedLayout();
  }
  auto baseReq = TensorLayoutCommon::getNthResultLayoutRequirements(node, n);
  auto baseReqHelper = TensorLayoutDescription(baseReq);
  return returnBaseReqOrNHWC(baseReqHelper, node);
}
