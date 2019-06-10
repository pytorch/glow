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

#include "glow/Optimizer/TrainingPreparation.h"
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
      tensor->init(glow::Tensor::InitKind::Zero, 0, F->getPRNG());
    } else if (BatchNormalizationNode::BiasIdx == inputIdx) {
      tensor->init(glow::Tensor::InitKind::Broadcast, 0.1, F->getPRNG());
    } else if (BatchNormalizationNode::MeanIdx == inputIdx) {
      tensor->zero();
    } else if (BatchNormalizationNode::VarIdx == inputIdx) {
      tensor->init(glow::Tensor::InitKind::Broadcast, 1.0, F->getPRNG());
    }
    break;
  }
  case Kinded::Kind::FullyConnectedNodeKind: {
    if (FullyConnectedNode::WeightsIdx == inputIdx) {
      FullyConnectedNode *FCN = llvm::cast<FullyConnectedNode>(node);
      auto in = FCN->getInput();
      tensor->init(glow::Tensor::InitKind::Zero, in.dims()[1], F->getPRNG());
    } else if (FullyConnectedNode::BiasIdx == inputIdx) {
      tensor->init(glow::Tensor::InitKind::Broadcast, 0.1, F->getPRNG());
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
                                       TensorInitializer &&initializer) {

  auto &nodes = F->getNodes();

  Node *outputNode{nullptr};

  // Find SaveNode and its input. It can be SoftMax or other aggregation nodes.
  // Prevent them from being replaced by Placeholders.
  for (auto &node : nodes) {
    if (Kinded::Kind::SaveNodeKind == node.getKind()) {
      // Detected SaveNode.
      SaveNode *SN = llvm::cast<SaveNode>(&node);
      ReshapeNode *RSN = llvm::dyn_cast<ReshapeNode>(SN->getInput().getNode());
      if (!RSN) {
        continue;
      }
      // Found Input ReshapeNode. Assign its input.
      outputNode = RSN->getInput().getNode();
      break;
    }
  }

  RETURN_ERR_IF_NOT(outputNode, "Cannot find output node");
  // Lookup all nodes, skip Storage types, enumerate inputs,
  // replace Constant type with trainable Placeholders except special cases,
  // like BatchNormalization inputs (mean and variance). In special cases
  // replace Constant type with non-trainable Placeholders.
  for (auto &node : nodes) {
    // Skip storages.
    if (llvm::isa<Storage>(&node)) {
      continue;
    }

    // Skip output node.
    if (outputNode == &node) {
      continue;
    }

    const bool isBatchNormalization =
        node.getKind() == Kinded::Kind::BatchNormalizationNodeKind;

    for (unsigned idx = 0, e = node.getNumInputs(); idx < e; ++idx) {
      auto *IN = node.getNthInput(idx).getNode();
      Constant *C = llvm::dyn_cast<Constant>(IN);
      if (!C) {
        continue;
      }
      bool isTrainable =
          !isBatchNormalization || (BatchNormalizationNode::MeanIdx != idx &&
                                    BatchNormalizationNode::VarIdx != idx);
      auto *PH = F->getParent()->createPlaceholder(C->getType(), C->getName(),
                                                   isTrainable);
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
