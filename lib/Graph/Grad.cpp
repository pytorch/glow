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

#include "glow/Graph/Grad.h"
#include "glow/Base/Train.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

using llvm::cast;
using llvm::isa;

void GraphGradMapper::addGradient(NodeValue activation, NodeValue grad) {
  if (map_.count(activation)) {
    auto curr = map_[activation];
    auto *sum = F_->createAdd("updateGrad", curr, grad);
    map_[activation] = sum;
    return;
  }

  map_[activation] = grad;
}

bool GraphGradMapper::hasGradient(NodeValue activation) {
  return map_.count(activation);
}

NodeValue GraphGradMapper::getGradient(NodeValue activation) {
  return map_[activation];
}

//===----------------------------------------------------------------------===//
//        Code for automatically generating the back propagation code.
//===----------------------------------------------------------------------===//

Function *glow::differentiate(Function *F, const TrainingConfig &conf,
                              llvm::StringRef newFuncName,
                              VariableGradientsList *varGrads) {
  // Create a new name for the differentiated function, if none is given.
  std::string tmpName;
  if (newFuncName.empty()) {
    tmpName = std::string(F->getName()) + "_grad";
    newFuncName = tmpName;
  }

  // Clone the function.
  Function *G = F->clone(newFuncName);

  using Kind = glow::Kinded::Kind;
  GraphGradMapper map(G);

  // A list of nodes to add to the Function.
  std::vector<Node *> toAppend;

  // Generate the gradient nodes for each one of the nodes in the function.

  PostOrderVisitor pov;
  for (auto &N : G->getNodes()) {
    N.visit(nullptr, &pov);
  }

  auto nodes = pov.getPostOrder();

  for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
    Node *N = *it;
    if (isa<Storage>(N)) {
      continue;
    }

#define CONVERT_TO_GRAD_NODE(NodeKind)                                         \
  if (N->getKind() == Kind::NodeKind##Kind) {                                  \
    toAppend.push_back(cast<NodeKind>(N)->getGrad(map));                       \
    continue;                                                                  \
  }

    CONVERT_TO_GRAD_NODE(ConvolutionNode)
    CONVERT_TO_GRAD_NODE(MaxPoolNode)
    CONVERT_TO_GRAD_NODE(AvgPoolNode)
    CONVERT_TO_GRAD_NODE(FullyConnectedNode)
    CONVERT_TO_GRAD_NODE(LocalResponseNormalizationNode)
    CONVERT_TO_GRAD_NODE(SoftMaxNode)
    CONVERT_TO_GRAD_NODE(CrossEntropyLossNode)
    CONVERT_TO_GRAD_NODE(RegressionNode)
    CONVERT_TO_GRAD_NODE(AddNode)
    CONVERT_TO_GRAD_NODE(MulNode)
    CONVERT_TO_GRAD_NODE(SubNode)
    CONVERT_TO_GRAD_NODE(DivNode)
    CONVERT_TO_GRAD_NODE(ReluNode)
    CONVERT_TO_GRAD_NODE(SigmoidNode)
    CONVERT_TO_GRAD_NODE(TanhNode)

    if (N->getKind() == Kind::SaveNodeKind) {
      // Swap the src and dest. Send the Zero value as gradient for both sides.
      auto *X = new SplatNode(N->getName(),
                              cast<SaveNode>(N)->getInput().getType(), 0);
      toAppend.push_back(X);
      map.addGradient(cast<SaveNode>(N)->getInput(), X);
      map.addGradient(cast<SaveNode>(N)->getOutput(), X);
      continue;
    }

    if (N->getKind() == Kind::ReshapeNodeKind) {
      ReshapeNode *RN = cast<ReshapeNode>(N);
      NodeValue outputG = map.getGradient(RN->getResult());
      NodeValue inputW = RN->getInput();

      // Swap the src and dest.
      auto *X = new ReshapeNode(N->getName(), inputW.getType(), outputG,
                                inputW.getType()->dims());
      toAppend.push_back(X);
      map.addGradient(RN->getInput(), X);
      continue;
    }

    if (N->getKind() == Kind::ConvertToNodeKind) {
      auto *RN = cast<ConvertToNode>(N);
      NodeValue outputG = map.getGradient(RN->getResult());
      NodeValue inputW = RN->getInput();

      // Swap the src and dest.
      auto *X = new ConvertToNode(N->getName(), inputW.getType(), outputG);
      toAppend.push_back(X);
      map.addGradient(RN->getInput(), X);
      continue;
    }

    if (N->getKind() == Kind::TransposeNodeKind) {
      TransposeNode *TN = cast<TransposeNode>(N);
      NodeValue outputG = map.getGradient(TN->getResult());
      NodeValue inputW = TN->getInput();

      // Generate the reverse shuffle.
      auto shuffle = TN->getShuffle();
      std::vector<unsigned_t> reverseShuffle(shuffle.begin(), shuffle.end());
      for (unsigned int i = 0; i < shuffle.size(); i++) {
        reverseShuffle[shuffle[i]] = i;
      }

      // Swap the src and dest.
      auto *X = new TransposeNode(N->getName(), inputW.getType(), outputG,
                                  reverseShuffle);
      toAppend.push_back(X);
      map.addGradient(TN->getInput(), X);
      continue;
    }

    if (N->getKind() == Kind::SliceNodeKind) {
      SliceNode *SN = cast<SliceNode>(N);
      auto *zero = new SplatNode("expand", SN->getInput().getType(), 0);
      auto *insert = new InsertTensorNode(
          "insert.slice.grad", zero, map.getGradient(SN->getResult()),
          SN->getStart(), /* count */ 1, /* axis */ 0);

      toAppend.push_back(zero);
      toAppend.push_back(insert);
      map.addGradient(SN->getInput(), insert);
      continue;
    }

    if (N->getKind() == Kind::ConcatNodeKind) {
      auto *CC = cast<ConcatNode>(N);
      auto inputs = CC->getInputs();
      NodeValue outputG = map.getGradient(CC->getResult());

      // We start extracting the shape at (0,0, ... ).
      std::vector<size_t> offsets(CC->getResult().dims().size(), 0);
      unsigned_t dim = CC->getDim();
      for (auto &N : inputs) {
        auto *X = new SliceNode("extract", N.getType(), outputG, offsets);
        toAppend.push_back(X);
        // We are stacking the tensors along a specific dimension. This means
        // that we increase the size of the tensor along this dimension.
        offsets[dim] += N.dims()[dim];
        map.addGradient(N, X);
      }
      continue;
    }

    if (N->getKind() == Kind::SplatNodeKind)
      // Constant nodes don't have inputs therefore don't need grad
      // calculations.
      continue;

    if (N->getKind() == Kind::BatchNormalizationNodeKind) {
      auto *BN = cast<BatchNormalizationNode>(N);
      auto in = BN->getInput();
      auto mean = BN->getMean();
      auto var = BN->getVar();
      auto channelIdx = BN->getChannelIdx();
      auto momentum = BN->getMomentum();

      // Update the mean and variance via the MeanVarNormalizationNode.
      auto *MVN = new MeanVarNormalizationNode("mean_var_normalization", in,
                                               mean, var, channelIdx, momentum);
      toAppend.push_back(MVN);

      // Save the newly calculated mean and variance to the mean and variance
      // variables. These will be used during the next iteration of training.
      G->createSave("saveMean", MVN->getNewMean(),
                    llvm::cast<Placeholder>(mean.getNode()));
      G->createSave("saveVar", MVN->getNewVar(),
                    llvm::cast<Placeholder>(var.getNode()));

      // Replace the BN's mean and variance with the new mean and variance
      // calculated from MVN.
      BN->setNthInput(BatchNormalizationNode::MeanIdx, mean);
      BN->setNthInput(BatchNormalizationNode::VarIdx, var);

      toAppend.push_back(BN->getGrad(map));

      continue;
    }

    llvm_unreachable("Invalid instruction type.");
  } // End of the for-each instr loop.

  for (auto N : nodes) {
    // Iterate only through Placeholders used by the Function. These are
    // inserted during the post-order walk.
    Placeholder *PH = llvm::dyn_cast<Placeholder>(N);
    if (!PH)
      continue;

    // In this special differentiation mode we record the last gradient value
    // without performing the SGD update. This mode is used by the unit tests.
    if (varGrads) {
      if (map.hasGradient(PH)) {
        std::string nodeName = "_grad_" + PH->getName().str();
        // Save the gradient and return the destination variable.
        auto *saveNode = G->createSave(nodeName, map.getGradient(PH));
        Placeholder *GradV = saveNode->getPlaceholder();
        varGrads->push_back({PH, GradV});
      }
      continue;
    }

    // Don't update nodes that are not marked as trainable.
    if (!PH->isTraining()) {
      continue;
    }

    auto X = new SGDNode(PH->getName(), map.getGradient(PH), PH, conf.L1Decay,
                         conf.L2Decay, conf.learningRate, conf.momentum,
                         conf.batchSize);
    toAppend.push_back(X);
    // Now update the weight with the value computed by SGD.
    auto *save = new SaveNode(PH->getName().str() + ".saveGrad", {X, 0}, PH);
    toAppend.push_back(save);
  }

  // Add all of the new variables and instructions.
  for (auto &I : toAppend) {
    G->addNode(I);
  }

  return G;
}
