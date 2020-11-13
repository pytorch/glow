/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

using llvm::cast;
using llvm::isa;

#define DECORATE_NODE_NAME(Node, ...)                                          \
  llvm::join_items("_", Node->getName(), __VA_ARGS__)

void GraphGradMapper::addGradient(NodeValue activation, NodeValue grad) {
  auto p = map_.insert({activation, grad});
  if (!p.second) {
    auto curr = p.first->second;
    auto *sum = F_->createAdd("updateGrad", curr, grad);
    p.first->second = sum;
  }
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
    CONVERT_TO_GRAD_NODE(AvgPoolNode)
    CONVERT_TO_GRAD_NODE(AdaptiveAvgPoolNode)
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
    CONVERT_TO_GRAD_NODE(SparseLengthsWeightedSumNode)
    CONVERT_TO_GRAD_NODE(SparseLengthsSumNode)

    if (N->getKind() == Kind::BatchedPairwiseDotProductNodeKind) {
      BatchedPairwiseDotProductNode *BPDPN =
          cast<BatchedPairwiseDotProductNode>(N);
      auto outputGrad = map.getGradient(BPDPN->getResult());

      auto *X = new BatchedPairwiseDotProductGradNode(
          DECORATE_NODE_NAME(N, "grad"), outputGrad, BPDPN->getInputs());

      size_t i = 0;
      for (auto &in : BPDPN->getInputs()) {
        X->addExtraResult(in.getType());
        map.addGradient(in, X->getNthResult(i));
        ++i;
      }

      toAppend.push_back(X);
      continue;
    }

    if (N->getKind() == Kind::SaveNodeKind) {
      // Swap the src and dest. Send the Zero value as gradient for both sides.
      auto *X = new SplatNode(DECORATE_NODE_NAME(N, "grad"),
                              cast<SaveNode>(N)->getInput().getType(), 0);
      toAppend.push_back(X);
      map.addGradient(cast<SaveNode>(N)->getInput(), X);
      map.addGradient(cast<SaveNode>(N)->getOutput(), X);
      continue;
    }

    if (N->getKind() == Kind::MaxPoolNodeKind) {
      auto *MPN = llvm::cast<MaxPoolNode>(N);
      // Argmax cannot be differentiated. Assert it has no users, and use a zero
      // Splat for its grad input so it doesn't have a null input.
      assert(MPN->getArgmax().getNumUsers() == 0 &&
             "Argmax cannot be differentiated; must go unused.");
      auto *ZSN = new SplatNode(DECORATE_NODE_NAME(N, "grad"),
                                MPN->getArgmax().getType(), 0);
      toAppend.push_back(ZSN);
      map.addGradient(MPN->getArgmax(), ZSN);
      toAppend.push_back(MPN->getGrad(map));
      continue;
    }

    if (N->getKind() == Kind::ReshapeNodeKind) {
      ReshapeNode *RN = cast<ReshapeNode>(N);
      NodeValue outputG = map.getGradient(RN->getResult());
      NodeValue inputW = RN->getInput();

      // Swap the src and dest.
      auto *X = new ReshapeNode(DECORATE_NODE_NAME(RN, "grad", "reshape"),
                                inputW.getType(), outputG,
                                inputW.getType()->dims(), RN->getLayout());
      toAppend.push_back(X);
      map.addGradient(RN->getInput(), X);
      continue;
    }

    if (N->getKind() == Kind::TileNodeKind) {
      TileNode *TN = cast<TileNode>(N);
      NodeValue outputG = map.getGradient(TN->getResult());

      // To compute the gradient with respect to the input of the TileNode, all
      // of the slices in outputG corresponding to the tiled slices in the
      // forward pass need to be added together. This is achieved by reshaping
      // outputG to replace the tiling axis with {numTiles, tileDim}, and then
      // performing a BatchedReduceAdd on the axis with numTiles elements. For
      // example, if the tile creates a {n,x,h,w} output with a {n,c,h,w}
      // input where x = c * numTiles, then the {n,x,h,w} gradient with respect
      // to the output is reshaped to {n, numTiles, c, h, w} so that
      // BatchedReduceAddNode eliminates the numTiles axis and produces a
      // {n,c,h,w} output.
      auto *TNInputType = TN->getInput().getType();
      std::vector<dim_t> BRAInputDims{TNInputType->dims()};
      BRAInputDims.insert(BRAInputDims.begin() + TN->getAxis(), TN->getCount());
      auto *BRAInputType =
          F->getParent()->uniqueTypeWithNewShape(TNInputType, BRAInputDims);

      auto *RN =
          new ReshapeNode(DECORATE_NODE_NAME(TN, "grad", "reshape"),
                          BRAInputType, outputG, BRAInputType->dims(), "*");
      auto *BRA = new BatchedReduceAddNode(
          DECORATE_NODE_NAME(TN, "grad", "bra"), TN->getInput().getType(), RN,
          {TN->getAxis()});

      toAppend.push_back(RN);
      toAppend.push_back(BRA);
      map.addGradient(TN->getInput(), BRA);
      continue;
    }

    if (N->getKind() == Kind::ConvertToNodeKind) {
      auto *RN = cast<ConvertToNode>(N);
      NodeValue outputG = map.getGradient(RN->getResult());
      NodeValue inputW = RN->getInput();

      // Swap the src and dest.
      auto *X = new ConvertToNode(DECORATE_NODE_NAME(N, "grad"),
                                  inputW.getType(), outputG);
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
      auto layout = TN->getLayout();
      std::string reverseLayout;
      reverseLayout.resize(TN->getLayout().size());
      std::vector<unsigned_t> reverseShuffle(shuffle.begin(), shuffle.end());
      for (unsigned int i = 0; i < shuffle.size(); i++) {
        reverseShuffle[shuffle[i]] = i;
        reverseLayout[shuffle[i]] = layout[i];
      }

      // Swap the src and dest.
      auto *X =
          new TransposeNode(DECORATE_NODE_NAME(N, "grad"), inputW.getType(),
                            outputG, reverseShuffle, reverseLayout);
      toAppend.push_back(X);
      map.addGradient(TN->getInput(), X);
      continue;
    }

    if (N->getKind() == Kind::SliceNodeKind) {
      SliceNode *SN = cast<SliceNode>(N);
      auto *zero = new SplatNode(DECORATE_NODE_NAME(SN, "expand"),
                                 SN->getInput().getType(), 0);
      auto *insert =
          new InsertTensorNode(DECORATE_NODE_NAME(SN, "grad"), zero,
                               map.getGradient(SN->getResult()), SN->getStart(),
                               /* count */ 1, /* axis */ 0);

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
      std::vector<dim_t> offsets(CC->getResult().dims().size(), 0);
      unsigned_t dim = CC->getDim();
      for (auto &N : inputs) {
        // SliceNode's name will be auto incremented due to name uniqueness.
        auto *X = new SliceNode(DECORATE_NODE_NAME(CC, "extract"), N.getType(),
                                outputG, offsets);
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
      auto *MVN = new MeanVarNormalizationNode(
          DECORATE_NODE_NAME(BN, "grad"), in, mean, var, channelIdx, momentum);
      toAppend.push_back(MVN);

      // Save the newly calculated mean and variance to the mean and variance
      // variables. These will be used during the next iteration of training.
      G->createSave(DECORATE_NODE_NAME(MVN, "mean"), MVN->getNewMean(),
                    llvm::cast<Placeholder>(mean.getNode()));
      G->createSave(DECORATE_NODE_NAME(MVN, "var"), MVN->getNewVar(),
                    llvm::cast<Placeholder>(var.getNode()));

      // Replace the BN's mean and variance with the new mean and variance
      // calculated from MVN.
      BN->getParent()->getLogContext()->logNodeInputChange(
          *BN, BN->getNthInput(BatchNormalizationNode::MeanIdx), mean);
      BN->setNthInput(BatchNormalizationNode::MeanIdx, mean);
      BN->getParent()->getLogContext()->logNodeInputChange(
          *BN, BN->getNthInput(BatchNormalizationNode::VarIdx), var);
      BN->setNthInput(BatchNormalizationNode::VarIdx, var);

      toAppend.push_back(BN->getGrad(map));

      continue;
    }

    if (N->getKind() == Kind::MatMulNodeKind) {
      MatMulNode *MMN = cast<MatMulNode>(N);
      // Get gradient.
      NodeValue OutputG = map.getGradient(MMN->getResult());

      // Get LHS/RHS inputs and their transpose presentations.
      NodeValue InputLHS = MMN->getLHS();
      NodeValue InputRHS = MMN->getRHS();
      auto *LT = G->createTranspose(
          DECORATE_NODE_NAME(MMN, "grad", "lhs", "transpose"), InputLHS,
          {1, 0});
      auto *RT = G->createTranspose(
          DECORATE_NODE_NAME(MMN, "grad", "rhs", "transpose"), InputRHS,
          {1, 0});

      // Grad for LHS = outputG x transpose(RHS).
      auto *GradLHS = new MatMulNode(DECORATE_NODE_NAME(MMN, "grad", "lhs"),
                                     InputLHS.getType(), OutputG, RT);
      // Grad for RHS = transpose(LHS) x outputG.
      auto *GradRHS = new MatMulNode(DECORATE_NODE_NAME(MMN, "grad", "rhs"),
                                     InputRHS.getType(), LT, OutputG);

      toAppend.push_back(GradLHS);
      map.addGradient(InputLHS, GradLHS);
      toAppend.push_back(GradRHS);
      map.addGradient(InputRHS, GradRHS);
      continue;
    }

    if (N->getKind() == Kind::BatchMatMulNodeKind) {
      BatchMatMulNode *BMMN = cast<BatchMatMulNode>(N);
      // Get gradient.
      NodeValue OutputG = map.getGradient(BMMN->getResult());

      // The implementation below is a batched version of the gradient
      // computation for MatMul.
      NodeValue InputLHS = BMMN->getLHS();
      NodeValue InputRHS = BMMN->getRHS();
      auto *LT = G->createTranspose(
          DECORATE_NODE_NAME(BMMN, "grad", "lhs", "transpose"), InputLHS,
          {0, 2, 1});
      auto *RT = G->createTranspose(
          DECORATE_NODE_NAME(BMMN, "grad", "lhs", "transpose"), InputRHS,
          {0, 2, 1});

      // Grad for LHS = outputG x transpose(RHS).
      auto *GradLHS =
          new BatchMatMulNode(DECORATE_NODE_NAME(BMMN, "grad", "lhs"),
                              InputLHS.getType(), OutputG, RT);
      // Grad for RHS = transpose(LHS) x outputG.
      auto *GradRHS =
          new BatchMatMulNode(DECORATE_NODE_NAME(BMMN, "grad", "lhs"),
                              InputRHS.getType(), LT, OutputG);

      toAppend.push_back(GradLHS);
      map.addGradient(InputLHS, GradLHS);
      toAppend.push_back(GradRHS);
      map.addGradient(InputRHS, GradRHS);
      continue;
    }

    if (N->getKind() == Kind::BatchedReduceAddNodeKind) {
      BatchedReduceAddNode *BRA = cast<BatchedReduceAddNode>(N);
      // Get gradient.
      NodeValue OutputG = map.getGradient(BRA->getResult());
      // Get input value.
      NodeValue Input = BRA->getBatch();

      // Gradient for BatchedReduceAddNode is TileNode,
      // repeating OutputG batch times.
      auto Axis = BRA->getAxes()[0];
      // Copy input dimensions first.
      std::vector<dim_t> Dims{Input.dims()};
      // Then set to 1 dimension size on axis.
      Dims[Axis] = 1;
      auto *RSN = G->createReshape(DECORATE_NODE_NAME(BRA, "grad", "reshape"),
                                   OutputG, Dims);
      auto *TN =
          new TileNode(DECORATE_NODE_NAME(BRA, "grad", "tile"), Input.getType(),
                       RSN->getResult(), Input.dims()[Axis], Axis);

      toAppend.push_back(TN);
      map.addGradient(Input, TN);
      continue;
    }

    if (N->getKind() == Kind::GatherNodeKind) {
      GatherNode *GN = cast<GatherNode>(N);
      // Get gradient.
      NodeValue Result = GN->getResult();
      NodeValue OutputG = map.getGradient(Result);
      // Get Data & Indices.
      NodeValue Data = GN->getData();
      NodeValue Indices = GN->getIndices();

      // Reshape indices into a two-dimensional Tensor (Vector).
      std::vector<dim_t> IndicesDims{Indices.getType()->size(), 1};
      auto *RI =
          G->createReshape(DECORATE_NODE_NAME(GN, "grad", "reshape", "indices"),
                           Indices, IndicesDims);

      // Reshape Gradient into N-k dimension, where k is Index dimensions,
      // except the case when Indices is one-dimensional.
      ReshapeNode *RG = nullptr;
      auto K = Indices.dims().size();
      if (K != 1) {
        const auto &OrgDims = OutputG.dims();
        std::vector<dim_t> GDims{OrgDims.begin() + K - 1, OrgDims.end()};
        for (dim_t k = 0; k < K - 1; ++k) {
          GDims[0] *= OrgDims[k];
        }
        RG = G->createReshape(
            DECORATE_NODE_NAME(GN, "grad", "reshape", "output"), OutputG,
            GDims);
      }
      // Reshaped Indices Vector maps Reshaped Gradient Tensors
      // to the correspondent Data Tensors, where Vector value
      // points to Data Tensor.
      auto *SN =
          G->createSplat(DECORATE_NODE_NAME(GN, "splat"), Data.getType(), 0);
      auto *SA = new ScatterDataNode(DECORATE_NODE_NAME(GN, "scatter_assign"),
                                     SN->getResult(), RI->getResult(),
                                     RG ? RG->getResult() : OutputG,
                                     /*cumulative*/ false);
      toAppend.push_back(SA);
      map.addGradient(Data, SA);
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
    auto *save =
        new SaveNode(DECORATE_NODE_NAME(PH, "save", "grad"), {X, 0}, PH);
    toAppend.push_back(save);
  }

  // Add all of the new variables and instructions.
  for (auto &I : toAppend) {
    G->addNode(I);
  }

  return G;
}

#undef DECORATE_NODE_NAME
