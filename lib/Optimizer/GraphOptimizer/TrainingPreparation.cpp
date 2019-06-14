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

#include "glow/Optimizer/GraphOptimizer/TrainingPreparation.h"

#include "glow/Base/Tensor.h"
#include "glow/Graph/PlaceholderBindings.h"

namespace glow {

namespace {
void defaultTensorInitializer(Function *F, Node *node, unsigned inputIdx,
                              Tensor *tensor) {
  switch (node->getKind()) {
  case Kinded::Kind::ConvolutionNodeKind: {
    if (ConvolutionNode::FilterIdx == inputIdx) {
      ConvolutionNode *CN = llvm::cast<ConvolutionNode>(node);
      ShapeNHWC idim = ShapeNHWC(CN->getInput().dims());
      ShapeHW kdim(CN->getKernels());
      size_t fanIn = kdim.height * kdim.width * idim.c;
      tensor->init(Tensor::InitKind::Xavier, fanIn, F->getPRNG());
    } else if (ConvolutionNode::BiasIdx == inputIdx) {
      tensor->init(Tensor::InitKind::Broadcast, 0.1, F->getPRNG());
    }
    break;
  }
  case Kinded::Kind::BatchNormalizationNodeKind: {
    if (BatchNormalizationNode::ScaleIdx == inputIdx) {
      tensor->init(Tensor::InitKind::Zero, 0, F->getPRNG());
    } else if (BatchNormalizationNode::BiasIdx == inputIdx) {
      tensor->init(Tensor::InitKind::Broadcast, 0.1, F->getPRNG());
    } else if (BatchNormalizationNode::MeanIdx == inputIdx) {
      tensor->init(Tensor::InitKind::Zero, 0, F->getPRNG());
    } else if (BatchNormalizationNode::VarIdx == inputIdx) {
      tensor->init(Tensor::InitKind::Broadcast, 1.0, F->getPRNG());
    }
    break;
  }
  case Kinded::Kind::FullyConnectedNodeKind: {
    if (FullyConnectedNode::WeightsIdx == inputIdx) {
      FullyConnectedNode *FCN = llvm::cast<FullyConnectedNode>(node);
      auto in = FCN->getInput();
      tensor->init(Tensor::InitKind::Xavier, in.dims()[1], F->getPRNG());
    } else if (FullyConnectedNode::BiasIdx == inputIdx) {
      tensor->init(Tensor::InitKind::Broadcast, 0.1, F->getPRNG());
    }
    break;
  }
  case Kinded::Kind::SoftMaxNodeKind: {
    if (SoftMaxNode::SelectedIdx == inputIdx) {
      tensor->zero();
    }
    break;
  }
  default:
    break;
  }
}
} // namespace

TensorInitializer getDefaultTensorInitializer() {
  return defaultTensorInitializer;
}

llvm::Error prepareFunctionForTraining(Function *F,
                                       PlaceholderBindings &bindings,
                                       Placeholder *&selected,
                                       TensorInitializer &&initializer) {

  auto &nodes = F->getNodes();

  selected = nullptr;
  // Lookup all nodes, skip Storage types, enumerate inputs,
  // replace Constant type with trainable Placeholders except special cases,
  // like BatchNormalization inputs (mean and variance). In special cases
  // replace Constant type with non-trainable Placeholders.
  for (auto &node : nodes) {
    // Skip storages.
    if (llvm::isa<Storage>(&node)) {
      continue;
    }

    const bool isSoftMax = node.getKind() == Kinded::Kind::SoftMaxNodeKind;
    const bool isBatchNormalization =
        node.getKind() == Kinded::Kind::BatchNormalizationNodeKind;

    for (unsigned idx = 0, e = node.getNumInputs(); idx < e; ++idx) {
      auto *IN = node.getNthInput(idx).getNode();
      Constant *C = llvm::dyn_cast<Constant>(IN);
      if (!C) {
        continue;
      }

      // Condition for NON trainable case
      // isSoftMax || isBatchNormalization &&
      //  (BatchNormalizationNode::MeanIdx == idx ||
      //   BatchNormalizationNode::VarIdx == idx)

      const bool isTrainable =
          !isSoftMax &&
          (!isBatchNormalization || (BatchNormalizationNode::MeanIdx != idx &&
                                     BatchNormalizationNode::VarIdx != idx));

      auto *PH = F->getParent()->createPlaceholder(C->getType(), C->getName(),
                                                   isTrainable);

      if (isSoftMax) {
        selected = PH;
      }
      C->getOutput().replaceAllUsesOfWith(PH, F);
      auto &tensor = C->getPayloadMutable();
      initializer(F, &node, idx, &tensor);
      bindings.insert(PH, std::move(tensor));
      RETURN_ERR_IF_NOT(!C->hasUsers(), "Constant is still in use.");
      F->getParent()->eraseConstant(C);
    }
  }

  return llvm::Error::success();
}
} // namespace glow
