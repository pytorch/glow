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

#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "glow/Backend/Backend.h"
#include "glow/Converter/Float16Converter.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Log.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Graph/TensorLayout.h"
#include "glow/Graph/Utils.h"
#include "glow/Graph/VerifierHelper.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"
#include "glow/Optimizer/GraphOptimizer/PassManager.h"
#include "glow/Optimizer/GraphOptimizerPipeline/Pipeline.h"
#include "glow/Optimizer/Lower/Lower.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Runtime/RuntimeTypes.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

llvm::cl::OptionCategory graphOptCat("Graph Optimizations Options");
llvm::cl::opt<unsigned> constDedupSizeOpt(
    "const_dedup_size",
    llvm::cl::desc(
        "Max number of elements allowed for deduplicating Constants"),
    llvm::cl::Optional, llvm::cl::init(256), llvm::cl::cat(graphOptCat));

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

static bool shouldDeleteNode(Node *N) {
  // In general, nodes who have side effects are retained.
  if (N->hasSideEffects()) {
    return false;
  }

  // Don't delete nodes that have users.
  if (N->hasUsers()) {
    return false;
  }

  return true;
}

/// Helper that \returns whether all sibling Functions of \p F (other Functions
/// inside its Module) are Loaded.
static bool shouldDeleteConstants(Function *F) {
  Module *mod = F->getParent();
  for (auto *MF : mod->getFunctions()) {
    if (MF->getState() < FunctionState::FuncLoaded) {
      return false;
    }
  }
  return true;
}

/// Helper that \returns the shuffle that inverts \p shuffle. For example, if
/// \p shuffle is {3, 0, 1, 2}, then this function returns {1, 2, 3, 0}.
static llvm::SmallVector<unsigned_t, max_tensor_dimensions>
invertShuffle(llvm::ArrayRef<unsigned_t> shuffle) {
  llvm::SmallVector<unsigned_t, max_tensor_dimensions> invertedShuffle;
  invertedShuffle.resize(shuffle.size());

  for (size_t i = 0; i < shuffle.size(); ++i) {
    invertedShuffle[shuffle[i]] = i;
  }

  return invertedShuffle;
}

/// Add a TranposeNode after \p C in \p F that has the same shuffle as \p TR.
/// This funcion assumes that the type of \p C and the output type of \p TR are
/// the same. \returns the newly added TransposeNode.
static TransposeNode *insertMatchingTransposeAfterConstant(Function *F,
                                                           Constant *C,
                                                           TransposeNode *TR) {
  auto *CT = C->getOutput().getType();
  auto *TRT = TR->getResult().getType();
  DCHECK(CT->isEqual(TRT));

  auto &T = C->getPayload();

  // In order for a new Transpose node with the same shuffle as TR to
  // be created at the output of the Constant, a new Constant should
  // created that has the same type as the input to TR.
  auto *NC = F->getParent()->createConstant(TR->getInput().getType(),
                                            C->getName().str() + ".transposed");

  // The payload of the original Constant C has the same type as the
  // output of TR. In order to preserve correctness, this payload must
  // be transposed using the inverse of the shuffle of TR and stored
  // into the payload of the new Constant.
  //
  // Another way to think of this is that we are inserting two
  // Transposes that are inverses of each other back to back after the original
  // Constant. The shuffle of the second Transpose must match that of TR.
  // In order to preserve correctness, the shuffle
  // of the first Transpose must be the inverse of that shuffle of the
  // second Transpose. The statement below statically computes this
  // first Transpose.
  T.transpose(&NC->getPayloadMutable(), invertShuffle(TR->getShuffle()));

  // Create Transpose on the LHS that has the same shuffle as TR.
  return F->createTranspose("transpose", NC, TR->getShuffle());
}

bool EmptyPass::run(Function *F, const CompilationContext &cctx) {
  return false;
}

bool DCE::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());

  auto &nodes = F->getNodes();
  auto &consts = F->getParent()->getConstants();

  std::vector<ConstList::iterator> erasedConsts{};
  std::vector<NodesList::iterator> erasedNodes{};

  bool changed = false;

  // Remove unused nodes.
  while (true) {
    bool changedLocally = false;
    for (auto it = nodes.begin(), e = nodes.end(); it != e;) {
      if (!shouldDeleteNode(&*it)) {
        ++it;
        continue;
      }

      erasedNodes.push_back(it);
      ++it;
      changedLocally = true;
      changed = true;
    }

    while (!erasedNodes.empty()) {
      auto it = erasedNodes.back();
      F->eraseNode(it);
      erasedNodes.pop_back();
    }

    if (!changedLocally) {
      break;
    }
  }

  if (!shouldDeleteConstants(F)) {
    return changed;
  }

  // Delete unused Constants.
  for (auto it = consts.begin(), e = consts.end(); it != e;) {
    if (!shouldDeleteNode(*it)) {
      ++it;
      continue;
    }
    erasedConsts.push_back(it);
    ++it;
  }

  while (!erasedConsts.empty()) {
    auto it = erasedConsts.back();
    F->getParent()->eraseConstant(it);
    erasedConsts.pop_back();
  }

  return changed;
}

/// \returns true if the \p shuffle corresponds to an identity operation, false
/// otherwise.
static bool isIdentityShuffle(llvm::ArrayRef<unsigned> shuffle) {
  for (size_t i = 0, e = shuffle.size(); i < e; i++) {
    if (shuffle[i] != i) {
      return false;
    }
  }
  return true;
}

/// \returns True if the node \p N always evaluates to \p val.
bool isSplatOfVal(Node *N, float val) {
  SplatNode *Z = dyn_cast<SplatNode>(N);
  if (!Z) {
    return false;
  }
  return (Z->getValue() == val);
}

/// \returns True if the node returns a constant value.
bool isConstant(Node *N) { return isa<SplatNode>(N); }

/// \returns the new simplified NodeValue or the original node's first result.
static NodeValue simplifyNode(Node *node, Function *F) {
// Simplify commutative nodes by moving the constant operator to the right-hand
// side.
// Example:  C + X  =>  X + C
#define COMMUTE_CONST_TO_RHS(NodeKind)                                         \
  if (auto *NN = dyn_cast<NodeKind##Node>(node))                               \
    if (isConstant(NN->getLHS()) && !isConstant(NN->getRHS())) {               \
      return F->create##NodeKind(NN->getName(), NN->getResult().getType(),     \
                                 NN->getRHS(), NN->getLHS());                  \
    }

  COMMUTE_CONST_TO_RHS(Add)
  COMMUTE_CONST_TO_RHS(Mul)
  COMMUTE_CONST_TO_RHS(Max)
  COMMUTE_CONST_TO_RHS(Min)
#undef COMMUTE_CONST_TO_RHS

  if (auto *AN = dyn_cast<AddNode>(node)) {
    // X + 0 => X
    if (isSplatOfVal(AN->getRHS(), 0)) {
      return AN->getLHS();
    }
  }

  if (auto *MN = dyn_cast<MulNode>(node)) {
    // X * 0 => 0
    if (isSplatOfVal(MN->getRHS(), 0)) {
      return MN->getRHS();
    }
    // X * 1 => X
    if (isSplatOfVal(MN->getRHS(), 1)) {
      return MN->getLHS();
    }
  }

  if (auto *DN = dyn_cast<DivNode>(node)) {
    // 0 / X => 0
    if (isSplatOfVal(DN->getLHS(), 0)) {
      return DN->getLHS();
    }
    // X / 1 => X
    if (isSplatOfVal(DN->getRHS(), 1)) {
      return DN->getLHS();
    }
  }

  // X - 0 => X
  if (auto *SN = dyn_cast<SubNode>(node)) {
    if (isSplatOfVal(SN->getRHS(), 0)) {
      return SN->getLHS();
    }
  }

  return node;
}

/// Sink Transpose below ChannelShuffle node.
static bool sinkTranposeBelowChannelShuffle(Function *F,
                                            ChannelShuffleNode *CS) {
  auto *TR = dyn_cast<TransposeNode>(CS->getInput());
  if (!TR) {
    return false;
  }

  // Create a new ChannelShuffle with kernel parameter transposed by the
  // sinking TR's shuffle because that Transpose will now be moved below this
  // ChannelShuffle operator.
  auto *newCS =
      F->createChannelShuffle(CS->getName(), TR->getInput(), CS->getGroup(),
                              TR->getShuffle()[CS->getKernel()]);

  // Create a copy of sinkingTR and insert after newChannelShuffle.
  auto *newTR = F->createTranspose(TR->getName(), newCS, TR->getShuffle(),
                                   TR->getLayout());

  CS->getResult().replaceAllUsesOfWith(newTR);

  return true;
}

/// Code Sinking.
bool SinkCode::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  auto &nodes = F->getNodes();
  // For each node:
  for (auto &N : nodes) {
    auto *node = &N;
    // Sink Transpose below batch normalization nodes:
    if (auto *BN = dyn_cast<BatchNormalizationNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(BN->getInput());

      if (!TR) {
        continue;
      }

      // Figure out where we transposed the channel index for batch
      // normalization.
      unsigned_t idx = BN->getChannelIdx();
      unsigned_t newChannelIdx = TR->getShuffle()[idx];

      auto *NewBN = F->createBatchNormalization(
          BN->getName(), TR->getInput(), BN->getBias(), BN->getScale(),
          BN->getMean(), BN->getVar(), newChannelIdx, BN->getEpsilon(),
          BN->getMomentum());
      NewBN->setPredicate(node->getPredicate());
      auto *newTR = F->createTranspose(TR->getName(), NewBN, TR->getShuffle(),
                                       TR->getLayout());
      newTR->setPredicate(node->getPredicate());

      BN->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
      continue;
    }

    // Sink Transpose below batch RELU nodes.
    if (auto *RL = dyn_cast<ReluNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(RL->getInput());

      if (!TR) {
        continue;
      }

      // Keep the same quantization parameters for ReLU output, but
      // change the shape to appropriate value.
      auto reluOutTy = F->getParent()->uniqueTypeWithNewShape(
          RL->getResult().getType(), TR->getInput().dims());
      auto *NRL = F->createRELU(RL->getName(), TR->getInput(), reluOutTy);
      NRL->setPredicate(node->getPredicate());
      auto *newTR = F->createTranspose(TR->getName(), NRL, TR->getShuffle(),
                                       TR->getLayout());
      newTR->setPredicate(node->getPredicate());
      RL->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
      continue;
    }

    // Sink Transpose below Clip nodes.
    if (auto *CL = dyn_cast<ClipNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(CL->getInput());

      if (!TR) {
        continue;
      }

      // Keep the same quantization parameters for Clip output, but
      // change the shape to appropriate value.
      auto clipOutTy = F->getParent()->uniqueTypeWithNewShape(
          CL->getResult().getType(), TR->getInput().dims());
      auto *NCL = F->createClip(CL->getName(), TR->getInput(), clipOutTy,
                                CL->getMin(), CL->getMax());
      NCL->setPredicate(node->getPredicate());
      auto *newTR = F->createTranspose(TR->getName(), NCL, TR->getShuffle());
      newTR->setPredicate(node->getPredicate());
      CL->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
      continue;
    }

    // Sink Transpose below Sigmoid nodes.
    if (auto *SI = dyn_cast<SigmoidNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(SI->getInput());

      if (!TR) {
        continue;
      }

      auto *NSI = F->createSigmoid(SI->getName(), TR->getInput());
      NSI->setPredicate(node->getPredicate());
      auto *newTR = F->createTranspose(TR->getName(), NSI, TR->getShuffle(),
                                       TR->getLayout());
      newTR->setPredicate(node->getPredicate());
      SI->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
      continue;
    }

    // Sink Transpose below Pad nodes.
    if (auto *padNode = dyn_cast<PadNode>(node)) {
      auto *transposeNode = dyn_cast<TransposeNode>(padNode->getInput());

      if (!transposeNode) {
        continue;
      }

      // The transpose shuffle specifies the source dimension.
      // When sinking Transpose below Pad, shuffle describes the target
      // dimension.
      auto shuffle = transposeNode->getShuffle();

      // Shuffle the Pad output type and the padding attribute.
      auto outPadType = padNode->getResult().getType();
      auto outPadShape = outPadType->dims();
      auto pads = padNode->getPads();
      size_t numDims = outPadShape.size();
      std::vector<dim_t> newOutPadShape(numDims);
      std::vector<int> newPads(2 * numDims);
      for (size_t i = 0; i < outPadShape.size(); i++) {
        newOutPadShape[shuffle[i]] = outPadShape[i];
        newPads[shuffle[i]] = pads[i];
        newPads[shuffle[i] + numDims] = pads[i + numDims];
      }

      // New pad
      auto newOutPadType =
          F->getParent()->uniqueTypeWithNewShape(outPadType, newOutPadShape);
      auto *NewPadNode = F->createPad(
          padNode->getName(), transposeNode->getInput(), newOutPadType,
          padNode->getMode(), newPads, padNode->getValue());
      NewPadNode->setPredicate(node->getPredicate());
      auto *newTransposeNode =
          F->createTranspose(transposeNode->getName(), NewPadNode, shuffle);
      newTransposeNode->setPredicate(node->getPredicate());
      padNode->getResult().replaceAllUsesOfWith(newTransposeNode);
      changed = true;
      continue;
    }

    // Sink Transpose below Tanh nodes.
    if (auto *TN = dyn_cast<TanhNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(TN->getInput());

      if (!TR) {
        continue;
      }

      auto *NTN = F->createTanh(TN->getName(), TR->getInput());
      NTN->setPredicate(node->getPredicate());
      auto *newTR = F->createTranspose(TR->getName(), NTN, TR->getShuffle(),
                                       TR->getLayout());
      newTR->setPredicate(node->getPredicate());
      TN->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
      continue;
    }

    // Remove 'identity' transpose operations.
    if (auto *TR = dyn_cast<TransposeNode>(node)) {
      auto mask = TR->getShuffle();

      if (isIdentityShuffle(mask)) {
        TR->getResult().replaceAllUsesOfWith(TR->getInput());
        changed = true;
        continue;
      }
    }

    // Merge consecutive Transpose operations.
    if (auto *TR1 = dyn_cast<TransposeNode>(node)) {
      auto *TR2 = dyn_cast<TransposeNode>(TR1->getInput());

      if (!TR2) {
        continue;
      }

      auto mask1 = TR1->getShuffle();
      auto mask2 = TR2->getShuffle();
      assert(mask1.size() == mask2.size() && "Invalid mask size");

      llvm::SmallVector<unsigned_t, max_tensor_dimensions> newMask;
      newMask.resize(mask2.size());

      for (size_t i = 0, end = mask2.size(); i < end; i++) {
        newMask[i] = mask2[mask1[i]];
      }

      auto *newTR = F->createTranspose("tranpose", TR2->getInput(), newMask);
      TR1->getResult().replaceAllUsesOfWith(newTR->getResult());
      changed = true;
      continue;
    }

    if (auto *CS = dyn_cast<ChannelShuffleNode>(node)) {
      // Sink Transpose below ChannelShuffle.
      if (sinkTranposeBelowChannelShuffle(F, CS)) {
        changed = true;
        continue;
      }
    }

    // Sink Transpose below Arithmetic nodes.
    if (node->isArithmetic()) {
      TransposeNode *LTR =
          dyn_cast<TransposeNode>(node->getNthInput(ArithmeticNode::LHSIdx));
      TransposeNode *RTR =
          dyn_cast<TransposeNode>(node->getNthInput(ArithmeticNode::RHSIdx));

      if (!LTR || !RTR) {
        // If one of the sides is a splat, it can be seen as
        // transpose (splat'). Similarly, if one of the sides is a Constant,
        // it can be seen as tranpose (Constant').
        if (isa<SplatNode>(node->getNthInput(ArithmeticNode::LHSIdx)) && RTR) {
          // Build splat' for LHS.
          auto *SN =
              dyn_cast<SplatNode>(node->getNthInput(ArithmeticNode::LHSIdx));
          auto *NS = F->createSplat("splat", RTR->getInput().getType(),
                                    SN->getValue());
          LTR = F->createTranspose("transpose", NS, RTR->getShuffle(),
                                   RTR->getLayout());
          changed = true;
        } else if (isa<SplatNode>(node->getNthInput(ArithmeticNode::RHSIdx)) &&
                   LTR) {
          // Build splat' for RHS.
          auto *SN =
              dyn_cast<SplatNode>(node->getNthInput(ArithmeticNode::RHSIdx));
          auto *NS = F->createSplat("splat", LTR->getInput().getType(),
                                    SN->getValue());
          RTR = F->createTranspose("transpose", NS, LTR->getShuffle(),
                                   LTR->getLayout());
          changed = true;
        } else if (isa<Constant>(node->getNthInput(ArithmeticNode::LHSIdx)) &&
                   RTR) {
          // Build Constant' for for LHS.
          auto *C = cast<Constant>(node->getNthInput(ArithmeticNode::LHSIdx));
          LTR = insertMatchingTransposeAfterConstant(F, C, RTR);
          changed = true;
        } else if (isa<Constant>(node->getNthInput(ArithmeticNode::RHSIdx)) &&
                   LTR) {
          // Build Constant' for for RHS.
          auto *C = cast<Constant>(node->getNthInput(ArithmeticNode::RHSIdx));
          RTR = insertMatchingTransposeAfterConstant(F, C, LTR);
          changed = true;
        } else {
          continue;
        }
      }
      // The masks of the transposes on both sizes must match.
      if (LTR->getShuffle() != RTR->getShuffle()) {
        continue;
      }

      Node *newAN = nullptr;

#define ARITHMETIC_CASE(NODE_NAME_)                                            \
  case glow::Kinded::Kind::NODE_NAME_##NodeKind:                               \
    newAN =                                                                    \
        F->create##NODE_NAME_(node->getName(),                                 \
                              F->getParent()->uniqueTypeWithNewShape(          \
                                  node->getType(ArithmeticNode::ResultIdx),    \
                                  LTR->getInput().getType()->dims()),          \
                              LTR->getInput(), RTR->getInput());               \
    break;

#define BOOLEAN_OP_CASE(NODE_NAME_)                                            \
  case glow::Kinded::Kind::NODE_NAME_##NodeKind:                               \
    newAN = F->create##NODE_NAME_(node->getName(), LTR->getInput(),            \
                                  RTR->getInput());                            \
    break;

      switch (node->getKind()) {
        ARITHMETIC_CASE(Add);
        ARITHMETIC_CASE(Mul);
        ARITHMETIC_CASE(Sub);
        ARITHMETIC_CASE(Div);
        ARITHMETIC_CASE(Max);
        ARITHMETIC_CASE(Min);
        BOOLEAN_OP_CASE(CmpLTE);
        BOOLEAN_OP_CASE(CmpEQ);
      default:
        llvm_unreachable("Unhandled node");
      }
#undef BOOLEAN_OP_CASE
#undef ARITHMETIC_CASE

      newAN->setPredicate(node->getPredicate());
      changed = true;
      auto *newTR = F->createTranspose(LTR->getName(), newAN, LTR->getShuffle(),
                                       LTR->getLayout());
      newTR->setPredicate(node->getPredicate());
      node->getNthResult(ArithmeticNode::ResultIdx).replaceAllUsesOfWith(newTR);
    }

    // Sink TransposeNode below QuantizedNode.
    // If it doesn't work out it will be re-sinked later.
    if (auto *Q = dyn_cast<QuantizeNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(Q->getInput());
      if (!TR) {
        continue;
      }

      auto newQType = F->getParent()->uniqueTypeWithNewShape(
          Q->getResult().getType(), TR->getInput().dims());
      auto *newQ = F->createQuantize(Q->getName(), TR->getInput(), newQType);
      auto *newTR = F->createTranspose(TR->getName(), newQ, TR->getShuffle());
      Q->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
    }

    // Sink TransposeNode below DequantizedNode.
    // If it doesn't work out it will be re-sinked later.
    if (auto *D = dyn_cast<DequantizeNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(D->getInput());
      if (!TR) {
        continue;
      }

      auto newDType = F->getParent()->uniqueTypeWithNewShape(
          D->getResult().getType(), TR->getInput().dims());
      auto *newD = F->createDequantize(D->getName(), TR->getInput(), newDType);
      auto *newTR = F->createTranspose(TR->getName(), newD, TR->getShuffle());
      D->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
    }

    // Sink Transpose below RescaleQuantized.
    // Potentially exposes opportunity to be combined up with Convolution.
    // If it doesn't work out it will be re-sinked later.
    if (auto *RQ = dyn_cast<RescaleQuantizedNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(RQ->getInput());
      if (!TR) {
        continue;
      }

      auto newRQType = F->getParent()->uniqueTypeWithNewShape(
          RQ->getResult().getType(), TR->getInput().getType()->dims());
      auto *newRQ =
          F->createRescaleQuantized(RQ->getName(), TR->getInput(), newRQType);
      auto *newTR = F->createTranspose(TR->getName(), newRQ, TR->getShuffle(),
                                       TR->getLayout());
      RQ->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
    }

    // Sink RELU below batch concat nodes.
    if (auto *CN = dyn_cast<ConcatNode>(node)) {
      llvm::SmallVector<NodeValue, 6> CNInputs;
      for (auto &input : CN->getInputs()) {
        auto *inputRL = dyn_cast<ReluNode>(input);
        if (!inputRL) {
          break;
        }
        CNInputs.push_back(inputRL->getInput());
      }

      if (CNInputs.size() == CN->getNumInputs()) {
        auto *newCN = F->createConcat(CN->getName(), CNInputs, CN->getDim());
        newCN->setPredicate(node->getPredicate());
        auto name = CN->getNthInput(0).getNode()->getName();
        auto *newRL = F->createRELU(name, newCN, CN->getResult().getType());
        newRL->setPredicate(node->getPredicate());
        CN->getResult().replaceAllUsesOfWith(newRL);
        changed = true;
      }
    }

    // Sink Transpose below concat nodes.
    if (auto *CN = dyn_cast<ConcatNode>(node)) {
      llvm::SmallVector<NodeValue, 6> transVector;
      auto inputIter = CN->getInputs().begin();
      auto *firstInput = dyn_cast<TransposeNode>(*inputIter);
      if (!firstInput) {
        continue;
      }

      transVector.push_back(firstInput->getInput());
      auto shuffle = firstInput->getShuffle();
      // If the shuffle masks don't agree or not all inputs are Transpose then
      // bail out.
      for (++inputIter; inputIter != CN->getInputs().end(); ++inputIter) {
        auto *tTR = dyn_cast<TransposeNode>(*inputIter);
        if (!tTR || tTR->getShuffle() != shuffle) {
          break;
        }
        transVector.push_back(tTR->getInput());
      }

      if (transVector.size() != CN->getNumInputs()) {
        continue;
      }

      // Figure out where we transposed the channel index for batch
      // normalization.
      unsigned_t idx = CN->getDim();
      unsigned_t newChannelIdx = shuffle[idx];

      auto *newCN = F->createConcat(CN->getName(), transVector, newChannelIdx);
      newCN->setPredicate(node->getPredicate());
      auto *newTR =
          F->createTranspose(firstInput->getName(), newCN,
                             firstInput->getShuffle(), firstInput->getLayout());
      newTR->setPredicate(node->getPredicate());
      CN->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
    }

    // Sink Clip below Reshape nodes.
    if (auto *RN = dyn_cast<ReshapeNode>(node)) {
      auto *CN = dyn_cast<ClipNode>(RN->getInput());
      if (!CN) {
        continue;
      }

      ReshapeNode *newRN = F->createReshape(RN->getName(), CN->getInput(),
                                            RN->getDims(), RN->getLayout());
      ClipNode *newCN = F->createClip(CN->getName(), newRN->getResult(),
                                      CN->getMin(), CN->getMax());
      RN->getResult().replaceAllUsesOfWith(newCN->getResult());
      newRN->setPredicate(RN->getPredicate());
      newCN->setPredicate(CN->getPredicate());
      changed = true;
      continue;
    }
  } // For all nodes in the graph.

  return changed;
}

/// \returns True if node A may depend on the result of B. The relationship
/// between the nodes does not have to be direct. For example, A can depend on
/// X which depends on B. In that case the method needs to return True.
/// Check the use-def dependency up to a depth of \p depth.
static bool mayDepend(Node *A, Node *B, unsigned depth = 6) {
  // We define the identify as a dependency.
  if (A == B) {
    return true;
  }

  // A does not depend on anything.
  if (A->getNumInputs() == 0) {
    return false;
  }

  // B has no users. Nothing can depend on it.
  if (B->getNumResults() == 0) {
    return false;
  }

  // We can't continue the search. Assume that the nodes depend on one another.
  if (depth == 0) {
    return true;
  }

  // Check all inputs of A. None of them may depend on B.
  for (int i = 0, e = A->getNumInputs(); i < e; i++) {
    auto *input = A->getNthInput(i).getNode();
    // The inputs of A must not depend on B.
    if (mayDepend(input, B, depth - 1)) {
      return true;
    }
  }

  // We checked all inputs of A and none of them depend on B.
  return false;
}

/// \returns True if the node \p N depends on any of the values in \p list, or
/// if any of the values in list depend on \p N.
static bool mayDependOnAny(llvm::ArrayRef<NodeValue> list, Node *N) {
  for (auto &ll : list) {
    if (mayDepend(ll.getNode(), N) || mayDepend(N, ll.getNode())) {
      return true;
    }
  }

  return false;
}

// Merge several two or more multiple matrix multiplications into a single
// large matmul. The large matmul is more likely to utilize the hardware. The
// result of the big matmul is the concatenated results.
//
//            ____      _________        _________
//   ----    |    |    |         |     M|  A * C  |
// M| A  |  T| B  | * K|    C    | =    |---------|
//   ---- ,  |    |    |         |     T|  B * C  |
//    K       ----      ---------        ---------
//             K            R                R
bool MergeMatMul::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  auto &nodes = F->getNodes();

  // These two maps record the list of matrix multipliers that use each node
  // value either as a right-hand-side user or a left-hand-user.
  llvm::DenseMap<Node *, std::vector<MatMulNode *>> rightMatrixUsers;
  llvm::DenseMap<Node *, std::vector<MatMulNode *>> leftMatrixUsers;

  // Collect the list of nodes that are used by the matrix multiplier.
  for (auto &node : nodes) {
    if (auto *MM = dyn_cast<MatMulNode>(&node)) {
      // Do not try to merge quantized matrix multiplications because their
      // quantized parameters may not match. Until we implement the logic to
      // match the scale and offset just avoid the optimization.
      if (MM->getResult().getType()->isQuantizedType()) {
        continue;
      }

      rightMatrixUsers[MM->getRHS().getNode()].push_back(MM);
      leftMatrixUsers[MM->getLHS().getNode()].push_back(MM);
    }
  }

  // Merge RHS matrices.
  for (auto &it : rightMatrixUsers) {
    auto &MMs = it.second;

    // Collects the LHS values to merge.
    std::vector<NodeValue> LHS;

    // For each matmul that depends on the rhs matrix.
    for (auto &MM : MMs) {
      auto L = MM->getLHS();
      // The operands to the matrix multiplier should not depend on one another
      // or else we won't be able to get rid of the original matrix
      // multiplication.
      if (mayDependOnAny(LHS, L.getNode())) {
        continue;
      }
      LHS.push_back(L);
    }

    // We need to have at least two matrices to merge.
    if (LHS.size() < 2) {
      continue;
    }

    // Merge the matmul:
    auto *CC = F->createConcat("mergeLHS", LHS, 0);
    auto *MM = F->createMatMul("bigMatMul", CC, it.first);

    dim_t R = MM->getResult().dims()[1];
    dim_t start = 0;
    for (auto *origMM : MMs) {
      dim_t H = origMM->getResult().dims()[0];
      auto *ex = F->createSlice("extract", MM, {start, 0}, {start + H, R});
      start += H;
      origMM->getResult().replaceAllUsesOfWith(ex);
      changed = true;
    }
  }
  return changed;
}

bool MergePadIntoConvolution::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  for (auto &node : F->getNodes()) {
    auto *CN = dyn_cast<ConvolutionNode>(&node);
    if (!CN) {
      continue;
    }

    auto *PN = dyn_cast<PadNode>(CN->getInput());
    if (!PN) {
      continue;
    }

    // Convolution only supports padding with 0 constant
    if ((PN->getMode() != PaddingMode::CONSTANT) || (PN->getValue() != 0.f)) {
      continue;
    }

    // The convolution needs to be the unique user
    if (!PN->hasOneUse()) {
      continue;
    }

    // Compute the new padding.
    // Note: - convolution only supports positive padding
    //       - the convolution takes NHWC input tensors.
    bool canMerge = true;
    auto padPads = PN->getPads();
    auto convPads = CN->getPads();

    // For now, there is a different interpretation of the ONNX spec for
    // Pad and Convolution. The 'pads' array won't have the same size because
    // only spatial dimensions are specified for the convolution while all
    // dimensions are specified for Pad.

    // The merge can apply only if no padding is requested for non spatial
    // dimensions.
    if ((padPads[0] != 0) || (padPads[3] != 0) || (padPads[4] != 0) ||
        (padPads[7] != 0)) {
      continue;
    }

    // Compute new spatial padding.
    const int H_INDEX = 1;
    std::vector<unsigned_t> newConvPads(4);
    auto numDims = PN->getResult().dims().size();
    for (size_t i = 0; i < 2; i++) {
      // Two pad integers per dimension (begin and end padding).
      for (size_t j = 0; j < 2; j++) {
        int newConvPadSigned =
            padPads[(i + H_INDEX) + j * numDims] + int(convPads[i + j * 2]);
        if (newConvPadSigned < 0) {
          canMerge = false;
          break;
        }
        newConvPads[i + j * 2] = unsigned_t(newConvPadSigned);
      }
    }
    if (!canMerge) {
      continue;
    }

    // New Convolution
    auto *newCN = F->createConv(CN->getName(), PN->getInput(), CN->getFilter(),
                                CN->getBias(), CN->getResult().getType(),
                                CN->getKernels(), CN->getStrides(), newConvPads,
                                CN->getGroup(), CN->getDilation());
    CN->getResult().replaceAllUsesOfWith(newCN);
    changed = true;
  }

  return changed;
}

/// Merge Transpose into MatMul or FC.
/// MatMul/FC(Reshape(Transpose(X)), Weights) ->
/// -> MatMul/FC(Reshape(X), reordered Weights)
/// Common sequence while using NCHW as input layout, because GLOW convolution
/// layout is NHWC:
/// Transpose([N, H, W, C]) -> [N, C, H, W]
/// Reshape([N, C, H, W]) -> [N, C * H * W]
/// MatMul/FC([N, C * H * W], [C * H * W, K]) -> [N, K]
bool MergeTransposeIntoMatMulOrFC::run(Function *F,
                                       const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  for (auto &node : F->getNodes()) {
    auto *MMN = dyn_cast<MatMulNode>(&node);
    auto *FCN = dyn_cast<FullyConnectedNode>(&node);
    Constant *W;
    ReshapeNode *RN;

    // Node is either MatMul or FC.
    if (MMN) {
      W = dyn_cast<Constant>(MMN->getRHS());
      RN = dyn_cast<ReshapeNode>(MMN->getLHS());
    } else if (FCN) {
      W = dyn_cast<Constant>(FCN->getWeights());
      RN = dyn_cast<ReshapeNode>(FCN->getInput());
    } else {
      continue;
    }

    // Weights node (or MatMul RHS) is constant.
    if (!W) {
      continue;
    }

    // Linearizing Reshape precedes MatMul/FC.
    // The first dimension must be kept unchanged, the others are linearized.
    if (!RN || !RN->hasOneUse() ||
        RN->getInput().dims()[0] != RN->getDims()[0]) {
      continue;
    }

    // Transpose precedes Reshape.
    // The first dimension must be kept unchanged, the others can be shuffled
    // in any way.
    auto *TN = dyn_cast<TransposeNode>(RN->getInput());
    if (!TN || !TN->hasOneUse() || TN->getShuffle()[0] != 0) {
      continue;
    }

    // MatMul/FC weights tensor is 2D. De-linearize the first dimension
    // according to Transpose output layout (original shape) and input layout
    // (reordered shape). Then we can do weights reordering by simply
    // transposing the tensor from original shape to reordered shape.
    //
    // Example for [N, H, W, C] -> [N, C, H, W] transpose (common case):
    //   De-linearized original shape: [C * H * W, K] -> [C, H, W, K]
    //   De-linearized reordered shape: [C * H * W, K] -> [H, W, C, K]
    //   Reorder weights by transposing them: [C, H, W, K] -> [H, W, C, K]
    ShapeVector shape, newShape;
    llvm::SmallVector<unsigned_t, max_tensor_dimensions> shuffle;
    shuffle.resize(TN->getShuffle().size() - 1);
    for (size_t i = 1; i < TN->getShuffle().size(); i++) {
      shape.push_back(TN->getResult().getType()->dims()[i]);
      newShape.push_back(TN->getInput().getType()->dims()[i]);
      shuffle[TN->getShuffle()[i] - 1] = i - 1;
    }
    shape.push_back(W->dims()[1]);
    newShape.push_back(W->dims()[1]);
    shuffle.push_back(TN->getShuffle().size() - 1);
    auto reshapedWTy =
        F->getParent()->uniqueTypeWithNewShape(W->getType(), shape);
    auto reshapedNewWTy =
        F->getParent()->uniqueTypeWithNewShape(W->getType(), newShape);

    // New reordered weights.
    auto *newW = F->getParent()->createConstant(W->getType(), W->getName(),
                                                W->getLayout());
    Tensor reshapedSrc(W->getPayload().getUnsafePtr(), reshapedWTy);
    Tensor reshapedDst(newW->getPayload().getUnsafePtr(), reshapedNewWTy);
    reshapedSrc.transpose(&reshapedDst, shuffle);

    // New Reshape and MatMul/FC.
    auto *newRN =
        F->createReshape(RN->getName(), TN->getInput(), RN->getDims());
    if (MMN) {
      auto *newMMN = F->createMatMul(MMN->getName(), MMN->getResult().getType(),
                                     newRN, newW);
      MMN->getResult().replaceAllUsesOfWith(newMMN);
    } else if (FCN) {
      auto *newFCN =
          F->createFullyConnected(FCN->getName(), newRN, newW, FCN->getBias(),
                                  FCN->getResult().getType());
      FCN->getResult().replaceAllUsesOfWith(newFCN);
    } else {
      llvm_unreachable("Unexpected node kind");
    }

    changed = true;
  }

  return changed;
}

/// \returns True if the two slices \p A and \p B access consecutive spacial
/// regions on the \p dim dimension. For example Slice(0..10) Slice(10..50)
/// are consecutive but Slice(0..10) Slice(20..30) are not.
static bool areSlicesConsecutive(SliceNode *A, SliceNode *B, unsigned_t dim) {
  // The slices must extract from the same input.
  if (A->getInput().getNode() != B->getInput().getNode()) {
    return false;
  }

  // The result element type must be identical.
  if (A->getResult().getType()->getElementType() !=
      B->getResult().getType()->getElementType()) {
    return false;
  }

  auto aStart = A->getStart();
  auto bStart = B->getStart();

  assert(aStart.size() > dim && "Invalid dimension");

  for (size_t i = 0, e = aStart.size(); i < e; i++) {
    if (i == dim) {
      auto resSize = A->getResult().dims();
      // This is the stride (the delta between the two slices on the requested
      // dimension).
      auto delta = bStart[i] - aStart[i];
      // The distance between the two slices must be identical to the size of
      // the result.
      if (resSize[dim] != delta) {
        return false;
      }

      continue;
    }

    // The non-consecutive dimensions must be identical.
    if (aStart[i] != bStart[i]) {
      return false;
    }
  }

  return true;
}

bool ConvertBroadcastedBatchMatMul::run(Function *F,
                                        const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  for (auto &node : F->getNodes()) {
    BatchMatMulNode *BMMN = dyn_cast<BatchMatMulNode>(&node);
    if (!BMMN) {
      continue;
    }

    NodeValue LHS = BMMN->getLHS();
    NodeValue RHS = BMMN->getRHS();

    // If RHS is a Tile along axis 0 and the input's dims()[0] == 1, then the
    // RHS is fully broadcasted and we can perform the optimization.
    TileNode *TN = dyn_cast<TileNode>(RHS);
    if (!TN || TN->getAxis() != 0 || TN->getInput().dims()[0] != 1) {
      continue;
    }

    // Can now convert the broadcasted BatchMatMul to a MatMul.
    // LHS = {numBatches, N, M}
    // RHS = {M, P}
    // Multiply each LHS matrix {N, M} by RHS {M, P} to get final matrix
    // {numBatches, N, P}
    const dim_t numBatches = LHS.dims()[0];
    const dim_t N = LHS.dims()[1];
    const dim_t M = LHS.dims()[2];
    const dim_t P = RHS.dims()[2];
    auto name = BMMN->getName();

    // Reshape the LHS to be a two-dimensional matrix, where each batch is
    // essentially concatenated onto itself in the 0th dimension.
    ReshapeNode *reshapeLHS =
        F->createReshape(name.str() + ".reshapeLHS", LHS, {numBatches * N, M});
    // Squeeze out the first dimension of the original Tile's input.
    ReshapeNode *squeezedRHS =
        F->createSqueeze(name.str() + ".squeezedRHS", TN->getInput(), {0});

    // Perform a normal matmul, implementing the batch matmul.
    MatMulNode *MMN = F->createMatMul(name, reshapeLHS, squeezedRHS);

    assert(MMN->getResult().dims()[0] == (numBatches * N) &&
           "Incorrect resulting dimension for batch matmul");
    assert(MMN->getResult().dims()[1] == P &&
           "Incorrect resulting dimension for batch matmul");

    // Reshape the result back to the expected batch output shape, with the
    // first dimension the number of batches.
    ReshapeNode *finalReshape = F->createReshape(name.str() + ".reshapeResult",
                                                 MMN, {numBatches, N, P});
    BMMN->getResult().replaceAllUsesOfWith(finalReshape);
    changed = true;
  }
  return changed;
}

/// Find a sequence of slices in \p input that span the whole input.
/// \returns True if a group of slices that span the whole input was found.
/// The order of the slices is recorded in \p order.
static bool findSlicesThatSpanInput(llvm::ArrayRef<SliceNode *> input,
                                    unsigned_t dimension,
                                    std::vector<SliceNode *> &order) {
  // This is the 'last' slice to be found in the sequence of slices.
  SliceNode *lastSlice = nullptr;

  // Find the 'first' slice in the sequence.
  for (SliceNode *SN : input) {
    auto start = SN->getStart();

    // Invalid dimension.
    if (start.size() <= dimension) {
      return false;
    }

    // Check if this slice extract the first element.
    if (start[dimension] == 0) {
      // We found the first element.
      lastSlice = SN;
      order.push_back(lastSlice);
      break;
    }
  }

  // We could not find a 'first' slice.
  if (!lastSlice) {
    return false;
  }

  // Now that we've found the first slice in the sequence, try to order the
  // rest of the slices after the first one.
  bool addedSlice = true;
  while (addedSlice) {
    addedSlice = false;

    // For each slice:
    for (SliceNode *SN : input) {
      // Ignore slices of invalid types. Ignore shapes for now, that's checked
      // next while ignoring the axis dimension.
      if (!lastSlice->getResult().getType()->isEqual(
              *SN->getResult().getType(),
              /* allowDifferentShape */ true)) {
        continue;
      }

      // Check if shapes match except for the axis dimension.
      bool skip = false;
      for (size_t i = 0, e = lastSlice->getResult().dims().size(); i < e; ++i) {
        if (i != dimension &&
            lastSlice->getResult().dims()[i] != SN->getResult().dims()[i]) {
          skip = true;
          break;
        }
      }
      if (skip) {
        continue;
      }

      // Check if SN comes after the last slice in the sequence.
      if (areSlicesConsecutive(lastSlice, SN, dimension)) {
        // Add the consecutive slice and schedule another iteration.
        lastSlice = SN;
        order.push_back(lastSlice);
        addedSlice = true;
        continue;
      }
    }
  } // While adding new slices.

  // Check that the last slice completes the tensor.
  auto startCoor = lastSlice->getStart();
  auto resDim = lastSlice->getResult().getType()->dims();
  auto inDim = lastSlice->getInput().getType()->dims();

  // Check if for all dimensions, the size of the result tensor plus the start
  // coordinate matches the size of the tensor.
  for (int i = 0, e = startCoor.size(); i < e; i++) {
    if (startCoor[i] + resDim[i] != inDim[i]) {
      return false;
    }
  }

  // Report success if we found at least two slices that extract from the
  // input.
  return order.size() > 1;
}

/// Merge multiple batched add nodes into a large batched-add node.
bool MergeBatchedAdd::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  auto &nodes = F->getNodes();

  // We index the batched add nodes by the slice operand.
  llvm::DenseMap<Node *, std::vector<BatchedAddNode *>> rightBAUsers;

  // Collect all of the batched add nodes and index them by the 'slice'
  // operand.
  for (auto &node : nodes) {
    if (auto *BA = dyn_cast<BatchedAddNode>(&node)) {
      rightBAUsers[BA->getSlice().getNode()].push_back(BA);
    }
  }

  // For each 'slice' that batched add nodes access:
  for (auto &it : rightBAUsers) {
    auto &BAs = it.second;

    // Collects the left-hand-side operands that the batched-adds add into. We
    // only collect 'slice' nodes.
    std::vector<SliceNode *> slices;

    for (auto *BA : BAs) {
      if (auto *S = dyn_cast<SliceNode>(BA->getBatch().getNode())) {
        slices.push_back(S);
      }
    }

    // Check if the slice nodes that we've collected cover a whole tensor.
    std::vector<SliceNode *> order;
    bool found = findSlicesThatSpanInput(slices, 0, order);

    if (!found) {
      continue;
    }

    // We found a sequence of batched-add-slice that cover the input tensor.
    // We can transform the graph and create one big batched-add.
    std::vector<Node *> newSlices;
    assert(order.size() > 1 && "order must contain at least 2 SliceNodes.");
    SliceNode *S = llvm::cast<SliceNode>(order[0]);
    auto *mergedBA = F->createBatchedAdd("mergedBA", S->getInput(), it.first);

    // Create the new slices. These slices will replace the original scalar
    // batched-add nodes.
    for (auto *orig : order) {
      newSlices.push_back(F->createSlice(orig->getName(), mergedBA,
                                         orig->getStart(),
                                         orig->getResult().getType()));
    }

    // Replace the original individual batched adds with corresponding slices
    // from the new merged batch add.
    for (auto *BA : BAs) {
      for (int i = 0, e = order.size(); i < e; i++) {
        if (BA->getBatch().getNode() == order[i]) {
          BA->getResult().replaceAllUsesOfWith(newSlices[i]);
          changed = true;
          break;
        }
      }
    }

  } // for each batched-add group.
  return changed;
}

/// Optimize ReduceMean configuration with AvgPool if possible: last two axes
/// in a 4D input must be reduced.
bool OptimizeReduceMean::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  auto &nodes = F->getNodes();

  // For each node:
  for (auto &node : nodes) {
    if (auto *RM = dyn_cast<BatchedReduceMeanNode>(&node)) {

      // Input shape must be 4D.
      if (RM->getBatch().dims().size() != 4) {
        continue;
      }

      // Last two axes must be reduced.
      auto axes = RM->getAxes();
      if (axes.size() != 2 || std::count(axes.begin(), axes.end(), 2) != 1 ||
          std::count(axes.begin(), axes.end(), 3) != 1) {
        continue;
      }

      // RM is already shaped to have the required output shape.
      NodeValue in = RM->getBatch();

      std::vector<unsigned_t> kernels = {static_cast<unsigned_t>(in.dims()[2]),
                                         static_cast<unsigned_t>(in.dims()[3])};
      std::vector<unsigned_t> strides = {1, 1};
      std::vector<unsigned_t> pads = {0, 0, 0, 0};

      // TODO: Fix bad assumption? See issue 3499, for now workaround it.
      // In Glow, AvgPool expects NHWC.
      auto *TR1 = F->createTranspose(
          RM->getName().str() + ".transposeNCHW2NHWC", in, NCHW2NHWC, "NHWC");
      auto *AP = F->createAvgPool(RM->getName().str() + ".avgPool", TR1,
                                  kernels, strides, pads);
      auto *TR2 = F->createTranspose(
          RM->getName().str() + ".transposeNHWC2NCHW", AP, NHWC2NCHW, "NCHW");

      // AvgPool keeps original shape. Add reshape to match expected output.
      std::vector<dim_t> shape = TR2->getResult().dims();

      ShapeVector shapeAxes(axes.begin(), axes.end());

      // Axes must be sorted for correct erase.
      std::sort(shapeAxes.rbegin(), shapeAxes.rend());
      for (const auto &axis : shapeAxes) {
        shape.erase(shape.begin() + axis);
      }

      auto *RN = F->createReshape(RM->getName().str() + ".reshape", TR2, shape);

      RM->getResult().replaceAllUsesOfWith(RN);
      changed = true;
      continue;
    }
  } // For all nodes in the graph.

  return changed;
}

/// \returns a uniquely used Constant with the same contents as \p node. If \p
/// node is not a Constant then \returns a nullptr. If \node is a Constant which
/// has a single use, \p node is returned. If \node is a Constant which has
/// multiple uses, then \returns a new duplicate Constant that has the same
/// contents as \p node contained in \p M.
static Constant *getUniquelyUsedConstant(Module *M, Node &node) {
  Constant *constant = dyn_cast<Constant>(&node);
  if (!constant) {
    return nullptr;
  }

  if (constant->hasOneUse()) {
    return constant;
  }

  // If constant has more than one use, duplicate it and return the duplicate.
  auto *NC = M->createConstant(constant->getType(), constant->getName(),
                               constant->getLayout());
  NC->getPayloadMutable().assign(&constant->getPayload());
  return NC;
}

/// Normalize the weight of \p CV with what \p BN is doing, given containing
/// Module \p M. \returns whether or not the normalization was possible.
template <typename ElemTy>
bool normalizeWeights(Module *M, ConvolutionNode &CV,
                      BatchNormalizationNode &BN) {
  static_assert(
      std::is_floating_point<ElemTy>::value ||
          std::is_same<float16_t, typename std::remove_cv<ElemTy>::type>::value,
      "This implementation is for floating-point values only");

  Constant *filterC = getUniquelyUsedConstant(M, *CV.getFilter().getNode());
  Constant *cbiasC = getUniquelyUsedConstant(M, *CV.getBias().getNode());

  if (!filterC || !cbiasC) {
    return false;
  }

  // Set the new filter and bias on CV if necessary.
  if (filterC != CV.getFilter().getNode()) {
    CV.getParent()->getLogContext()->logNodeInputChange(
        CV, CV.getNthInput(ConvolutionNode::FilterIdx), filterC);
    CV.setNthInput(ConvolutionNode::FilterIdx, filterC);
  }
  if (cbiasC != CV.getBias().getNode()) {
    CV.getParent()->getLogContext()->logNodeInputChange(
        CV, CV.getNthInput(ConvolutionNode::BiasIdx), cbiasC);
    CV.setNthInput(ConvolutionNode::BiasIdx, cbiasC);
  }

  // First, BN computation can be phrased as follows:
  //
  // (X - mean) * (1.0 / sqrt(var + eps)) * bn_scale + bias
  //
  // Thus, we can rewrite bn_scale as:
  //  X * bn_scale * 1.0 / (sqrt(var + eps)) +
  //    (bias - mean * (1.0 / sqrt(var + eps)) * bn_scale)
  //
  // Thus, can just have the affine transform:
  //
  //  X * A + B
  //
  //  where
  //
  //  A = bn_scale * 1.0 / (sqrt(running_var + eps))
  //  B =  (bias - mean * (1.0 / sqrt(var + eps)) * bn_scale)
  //
  // Now, we have that the computation made is the following:
  //
  // ((X `conv` W) + b) * A + B
  //
  // Then, we can simply fuse this as follows:
  //
  // (X `conv` (W * A)) + b * A + B
  //
  // which is simply
  //
  // (X `conv` Q) + C
  //
  // where
  //
  // Q = W * A
  // C = b * A + B

  Constant *scaleC = cast<Constant>(BN.getScale());
  Constant *biasC = cast<Constant>(BN.getBias());
  Constant *meanC = cast<Constant>(BN.getMean());
  Constant *var = cast<Constant>(BN.getVar());

  auto filterH = filterC->getHandle<ElemTy>();

  auto cbiasH = cbiasC->getHandle<ElemTy>();

  auto scaleH = scaleC->getHandle<ElemTy>();
  auto biasH = biasC->getHandle<ElemTy>();
  auto meanH = meanC->getHandle<ElemTy>();
  auto varH = var->getHandle<ElemTy>();

  // Update the filter/bias constants of the Conv node.
  auto epsilon = BN.getEpsilon();
  for (size_t i = 0, e = filterH.size(); i < e; i++) {
    // Dimension zero is the 'channel' dimension. If we ever change the
    // layout of the filter then we need to change this optimization.
    dim_t channelId = filterH.getDimForPtr(0, i);
    float value = varH.at({channelId});
    float stdvar = 1.0f / std::sqrt(value + epsilon);
    float gamma = scaleH.at({channelId});
    float A = gamma * stdvar;
    filterH.raw(i) = ElemTy(float(filterH.raw(i)) * A);
  }

  for (size_t i = 0, e = cbiasH.size(); i < e; i++) {
    // Dimension zero is the 'channel' dimension. If we ever change the
    // layout of the filter then we need to change this optimization.
    dim_t channelId = cbiasH.getDimForPtr(0, i);
    float mu = meanH.at({channelId});
    float value = varH.at({channelId});
    float stdvar = 1.0f / std::sqrt(value + epsilon);
    float gamma = scaleH.at({channelId});
    float beta = biasH.at({channelId});
    float A = gamma * stdvar;
    float B = beta - mu * A;
    cbiasH.raw(i) = ElemTy(float(cbiasH.raw(i)) * A + B);
  }
  return true;
}

bool OptimizeBatchNorm::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  auto &nodes = F->getNodes();
  auto *M = F->getParent();

  // For each node:
  for (auto &node : nodes) {
    // Merge the Batch Normalization operation into the convolution that comes
    // before it by updating the weights of the filter and bias.
    if (auto *BN = dyn_cast<BatchNormalizationNode>(&node)) {
      auto *CV = dyn_cast<ConvolutionNode>(BN->getInput());
      if (!CV) {
        continue;
      }

      // We can't modify conv operators that have multiple users.
      if (!CV->hasOneUse()) {
        continue;
      }

      bool normalizationHappened = false;
      switch (CV->getElementType(ConvolutionNode::ResultIdx)) {
      case ElemKind::FloatTy:
        normalizationHappened = normalizeWeights<float>(M, *CV, *BN);
        break;
      case ElemKind::Float16Ty:
        normalizationHappened = normalizeWeights<float16_t>(M, *CV, *BN);
        break;
      default:
        llvm_unreachable("Type not supported");
      }

      if (!normalizationHappened) {
        continue;
      }

      // Take the predicate of what was expected for the output.
      CV->setPredicate(BN->getPredicate());
      BN->getResult().replaceAllUsesOfWith(CV);
      changed = true;
      continue;
    }
  } // For all nodes in the graph.
  return changed;
}

/// \returns true if all dimensions of the \p input tensors are the same
/// except for the provided \p dimension, otherwise return false.
static bool checkConcatNodeUniformDims(llvm::ArrayRef<NodeValue> inputs,
                                       unsigned_t dimension) {
  for (size_t i = 1; i < inputs.size(); i++) {
    for (size_t j = 0; j < inputs[0].dims().size(); j++) {
      if (j == dimension) {
        continue;
      }
      if (inputs[0].dims()[j] != inputs[i].dims()[j]) {
        return false;
      }
    }
  }
  return true;
}

/// Given a tensor's dims \p firstDims and the desired leading/trailing dims
/// sizes \p leadingDimsProdOriginalConcatNode, \p
/// trailingDimsProdOriginalConcatNode. \returns the dimension, at which the
/// trailing/leading dimensions match the desired sizes, otherwise returns -1.
/// Example: Given a tensor <1,2,3,4,5>, and a desired trailing dimensions
/// size of 20, and a desired leading dimensions size of 2, this function will
/// return dimension 1 as the trailing dimensions after it are <4,5>, which
/// matches the size 20, and the leading dimensions are <1,2>, which matches
/// the size 2.
static ssize_t findMatchingConcatDimForSameTrailingAndLeadingDims(
    llvm::ArrayRef<dim_t> firstDims, size_t leadingDimsProdOriginalConcatNode,
    size_t trailingDimsProdOriginalConcatNode) {
  size_t trailingDimsProdCurNode = 1;
  for (ssize_t i = firstDims.size() - 1; i >= 0; i--) {
    if (trailingDimsProdCurNode == trailingDimsProdOriginalConcatNode) {
      size_t leadingDimsProdCurNode = 1;
      for (ssize_t j = 0; j < i; j++) {
        leadingDimsProdCurNode *= firstDims[j];
      }
      if (leadingDimsProdCurNode == leadingDimsProdOriginalConcatNode) {
        return i;
      }
    }
    trailingDimsProdCurNode *= firstDims[i];
  }
  return -1;
}

/// Given input tensors \p inputs and a original ConcatNode \p origConcatN,
/// try to find out if there is a dimension in the input tensors, with which
/// we can meet two requirements:
///   1) Input tensors are concatenate-able along this dimension.
///   2) The trailing/leading dimensions sizes after/before this dimension in
///      the input tensors, are of the same size as the trailing/leading
///      dimensions of the input of the original Concat node after/before the
///      concatenation dimension. It is required, because they ensure that the
///      payload of the new concat node should be the same as the payload of
///      the original concat node, and also won't affect the data order of the
///      entire tensor.
/// \returns this dimension if found, otherwise -1.
static int
findConcatDimForSameTrailingAndLeadingDims(llvm::ArrayRef<NodeValue> inputs,
                                           ConcatNode *originalConcatNode) {
  // For the purpose of the optimiztion
  // Concat(Reshape(X)*N)->Reshape(Concat(N*X)), we want to make sure the new
  // ConcatNode can concatenate on the trailing/leading dimensions which are
  // of the same size of those of the original Concate node.

  auto firstDims = inputs.front().dims();
  auto origConcatNInputDims = originalConcatNode->getInputs().front().dims();
  // The sizes of the trailing/leading dimensions of the original ConcatNode,
  // which are being concatenated. This sizes are simply the products of
  // dimensions following/before the dimension used for concatenation.
  dim_t trailingDimsProdOriginalConcatNode = 1;
  dim_t leadingDimsProdOriginalConcatNode = 1;
  for (size_t i = 0; i < origConcatNInputDims.size(); ++i) {
    if (i < originalConcatNode->getDim()) {
      leadingDimsProdOriginalConcatNode *= origConcatNInputDims[i];
    } else if (i > originalConcatNode->getDim()) {
      trailingDimsProdOriginalConcatNode *= origConcatNInputDims[i];
    }
  }

  // Try to find the dimension in the first input such that the
  // trailing/leading dimensions sizes are the same as the sizes of the
  // trailing/leading dimensions based on the concatenation dimension used by
  // the original ConcatNode.
  ssize_t dim = findMatchingConcatDimForSameTrailingAndLeadingDims(
      firstDims, leadingDimsProdOriginalConcatNode,
      trailingDimsProdOriginalConcatNode);
  if (dim == -1) {
    return -1;
  }

  // Now we have found the dimension, we need to check if all inputs can be
  // concatenated along this dimension.
  if (!checkConcatNodeUniformDims(inputs, dim)) {
    return -1;
  }
  return dim;
}

/// Given the inputs \p originalConcatInputs of one Concat Nodes, \returns
/// true if they are all ReshapeNode, and the input tensors of these input
/// nodes have same number of dimensions, otherwise returns false.
static bool
tryToGetNewConcatInputs(NodeValueArrayRef originalConcatInputs,
                        llvm::SmallVectorImpl<NodeValue> &newConcatInputs) {
  // Go through the input nodes of CN, check if they are all ReshapeNode,
  // and if the input tensors of these input nodes have same number of
  // dimensions.
  for (auto &I : originalConcatInputs) {
    if (auto *R = dyn_cast<ReshapeNode>(I)) {
      if (newConcatInputs.empty() || newConcatInputs.front().dims().size() ==
                                         R->getInput().dims().size()) {
        newConcatInputs.push_back(R->getInput());
        continue;
      }
    }
    return false;
  }
  return true;
}

/// Concat(Reshape(x) * N) -> Reshape(Concat(x * N)).
/// \returns a new simplified Concat node or nullptr.
static NodeValue tryToOptimizeConcatOfRehapes(Function *F, ConcatNode *CN) {
  llvm::SmallVector<NodeValue, 16> newConcatInputs;
  // The inputs of the collected input reshape nodes. They will be used as
  // inputs for the new Concat node if possible.
  if (!tryToGetNewConcatInputs(CN->getInputs(), newConcatInputs)) {
    return NodeValue(nullptr);
  }

  // Try to concatenate along the same size trailing/leading dimensions as of
  // the original Concat node.
  auto dim = findConcatDimForSameTrailingAndLeadingDims(newConcatInputs, CN);
  if (dim == -1) {
    return NodeValue(nullptr);
  }
  auto *newCN = F->createConcat(CN->getName(), newConcatInputs, dim);
  return F->createReshape(
      CN->getInputs().front().getNode()->getName(), newCN,
      CN->getResult().dims(),
      CanonicalTensorLayout::getInstance().getNthResultLayoutRequirements(
          CN, ConcatNode::ResultIdx));
}

/// Simplify concat node.
/// \returns a new simplified Concat node or nullptr.
static NodeValue simplifyConcatNode(Function *F, ConcatNode *CN) {
  /// concat(dim1, concat(dim2, X, Y), Z) -> concat(dim1, X, Y, Z),
  /// but only if dim1 == dim2

  LOG_SCOPE(F->getLogContext(), "simplifyConcatNode")

  {
    auto inputs = CN->getInputs();
    // Check if any of the inputs are ConcatNode.
    llvm::SmallVector<NodeValue, 16> newInputs;
    bool merged = false;
    for (auto &input : inputs) {
      newInputs.push_back(input);
      auto *CNI = dyn_cast<ConcatNode>(input);
      // Bail if it is not a ConcatNode or it is a concat node with a diffrent
      // dimension.
      if (!CNI || CNI->getDim() != CN->getDim()) {
        continue;
      }

      merged = true;
      // Replace current input by its own inputs, i.e. merge them into the
      // parent concat node.
      newInputs.pop_back();
      newInputs.append(CNI->getInputs().begin(), CNI->getInputs().end());
    }
    if (merged) {
      // Return a new simplified Concat node.
      return F->createConcat(CN->getName(), newInputs, CN->getDim());
    }
  }

  // If all of the inputs to the concat are extracted from the same input in
  // the right order then we can just use the extract-input instead of the
  // concat. Concat(Slice(X, 0..10), Slice(X, 10..20)) -> X.
  {
    std::vector<SliceNode *> order;
    std::vector<SliceNode *> slices;
    // Collect all of the inputs that are SliceNode.
    for (auto &I : CN->getInputs()) {
      if (auto *S = dyn_cast<SliceNode>(I.getNode())) {
        slices.push_back(S);
      }
    }
    // Check if the slices span the input value.
    bool found = findSlicesThatSpanInput(slices, CN->getDim(), order);
    if (found && order.size() == slices.size()) {
      auto orig = order[0]->getInput();
      // The original value that we extract from must be of the same shape as
      // the concat.
      if (CN->getResult().getType() == orig.getType()) {
        return orig;
      }
    }
  }

  // Try the optimization Concat(Reshape(x) * N) -> Reshape(Concat(x * N)).
  if (auto transformedConcatNode = tryToOptimizeConcatOfRehapes(F, CN)) {
    return transformedConcatNode;
  }

  // If the concat has a single input, replace the concat with that input.
  if (CN->getNumInputs() == 1) {
    return CN->getInputs()[0];
  }

  return NodeValue(nullptr);
}

/// If all of the outputs of \p CN are essentially piped from the inputs of the
/// concat (i.e. same shape, axis, order) then we can get rid of the slices and
/// concat. \returns true if this optimization is successful and changes the
/// Function.
static bool combineConcatSlices(ConcatNode *CN) {
  auto inputsToCN = CN->getInputs();
  std::vector<SliceNode *> slices;
  std::vector<SliceNode *> orderedSlices;
  for (auto &user : CN->getUsers()) {
    if (SliceNode *SN = dyn_cast<SliceNode>(user.getUser())) {
      slices.push_back(SN);
    }
  }

  // Check if the slices span the input value.
  bool found = findSlicesThatSpanInput(slices, CN->getDim(), orderedSlices);
  if (!found || orderedSlices.size() != slices.size() ||
      orderedSlices.size() != inputsToCN.size()) {
    return false;
  }

  // Now verify that all of the inputs to CN have the same shape as all of the
  // slices for the result of CN.
  for (size_t i = 0, e = orderedSlices.size(); i < e; ++i) {
    if (orderedSlices[i]->getResult().dims() != inputsToCN[i].dims()) {
      return false;
    }
  }

  // We can now replace all of the inputs to the concat to the result of
  // each slice.
  for (size_t i = 0, e = inputsToCN.size(); i < e; ++i) {
    orderedSlices[i]->getResult().replaceAllUsesOfWith(inputsToCN[i]);
  }
  return true;
}

/// Optimize Concat nodes.
bool OptimizeConcatNodes::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  auto &nodes = F->getNodes();

  // For each node:
  for (auto &node : nodes) {
    auto *CN = dyn_cast<ConcatNode>(&node);
    if (!CN) {
      continue;
    }
    NodeValue newCN = simplifyConcatNode(F, CN);
    if (newCN.getNode()) {
      CN->getResult().replaceAllUsesOfWith(newCN);
      changed = true;
      continue;
    }

    if (combineConcatSlices(CN)) {
      changed = true;
      continue;
    }
  }
  return changed;
}

/// Simplify and canonicalize arithmetic nodes by detecting simple arithmetic
/// identities.
bool OptimizeArithmeticNodes::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  // A worklist that contains the nodes to process.
  std::vector<Node *> worklist;

  // Add all of the arithmetic nodes to the worklist, with a node's
  // dependencies added after itself so they are processed before the node.
  GraphPreOrderVisitor visitor(*F);
  worklist.reserve(visitor.getPreOrder().size());
  for (auto *N : visitor.getPreOrder()) {
    if (N->isArithmetic()) {
      worklist.push_back(N);
    }
  }
  while (!worklist.empty()) {
    Node *N = worklist.back();
    assert(N->isArithmetic() && "Must be an Arithmetic node.");
    worklist.pop_back();

    auto SNV = simplifyNode(N, F);
    if (SNV.getNode() != N) {
      N->getNthResult(ArithmeticNode::ResultIdx).replaceAllUsesOfWith(SNV);
      changed = true;

      auto *SN = SNV.getNode();

      // The simplified node could be further simplified. Note that the
      // simplified node might not be arithmetic; it could be a splat.
      if (SN->isArithmetic()) {
        worklist.push_back(SN);
      }

      // The simplified node's operands could be further simplified as well.
      // Push them after the node so they are processed before the node.
      for (size_t i = 0, e = SN->getNumInputs(); i < e; i++) {
        Node *input = SN->getNthInput(i).getNode();
        if (input->isArithmetic()) {
          worklist.push_back(input);
        }
      }
      continue;
    }
  }
  return changed;
}

/// Statically transpose Constants.
bool TransposeConstants::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  auto &nodes = F->getNodes();
  bool changed = false;
  for (auto &node : nodes) {
    auto *TN = dyn_cast<TransposeNode>(&node);
    if (!TN) {
      continue;
    }
    auto *C = dyn_cast<Constant>(TN->getInput());
    if (!C) {
      continue;
    }
    // Create a new Constant NC to hold the transposed result.
    auto *NC = F->getParent()->createConstant(TN->getResult().getType(),
                                              C->getName(), TN->getLayout());
    // Transpose the value of C into NC.
    genericTranspose(&C->getPayload(), &NC->getPayloadMutable(),
                     TN->getShuffle());
    NC->getPayloadMutable().setType(NC->getType());
    // Rewrite uses of TN to reference NC.
    TN->getResult().replaceAllUsesOfWith(NC);
    changed = true;
  }
  return changed;
}

namespace {

/// A helper type for hasing Node pointers when they are used as keys in hash
/// maps.
struct NodeHasher {
  size_t operator()(Node *N) const { return N->getHash(); }
};

/// A helper type implementing the Node equality predicate that can be used
/// when Node pointers are used as keys in hash maps.
struct NodeEq {
  bool operator()(const Node *lhs, const Node *rhs) const {
    return lhs->isEqual(*rhs);
  }
};

/// This visitor is used to walk the graph and perform a common subexpression
/// evaluation.
struct CSEVisitor : NodeWalker {
  // Mapping from the original node to its canonical representation under CSE.
  std::unordered_map<Node *, Node *, NodeHasher, NodeEq> cseNodes_;
  // Set of visited nodes.
  std::unordered_set<Node *> visitedNodes_;

  /// This callback is called before visiting the children of \p N.
  void pre(Node *parent, Node *N) override {
    // Put the node into a visited set to make sure it is visited
    // only once.
    visitedNodes_.insert(N);
  }

  /// This callback is called after visiting the children of \p N.
  /// It means that all of its dependencies are processed already.
  void post(Node *parent, Node *N) override {
    // Try to find a node equivalent to the current one.
    auto FoundI = cseNodes_.find(N);
    if (FoundI == cseNodes_.end()) {
      // No node CSE-equivalent to the current one has been seen yet.
      // Remember this node, so that the next occurrence can be
      // replaced by this one.
      cseNodes_.insert({N, N});
      assert(cseNodes_.find(N) != cseNodes_.end());
      return;
    }
    Node *foundN = FoundI->second;

    // Same node cannot be visited.
    assert(N != foundN);
    // Replace current node by a found node, which is
    // equivalent to it.
    assert(N->isEqual(*foundN));
    for (unsigned i = 0; i < N->getNumResults(); i++) {
      NodeValue FV(foundN, i);
      N->getNthResult(i).replaceAllUsesOfWith(FV);
    }
    // TODO: Erase N during CSE? If we don't do it here,
    // DCE will remove it later anyways.
  }

  /// Make sure that each node is processed only once.
  bool shouldVisit(Node *parent, Node *N) override {
    return visitedNodes_.count(N) == 0;
  }
};

/// A helper type for hashing Constant pointers when they are used as keys in
/// hash maps for deduplication. The hash is based on the type of the Constant
/// (element type, dimensions), as well as a constant number of elements from
/// the backing Tensor to balance collisions with hash calclulation time.
struct ConstsHasherDedup {
  size_t operator()(Constant *V) const {
    auto hash = llvm::hash_value(V->getType());
    auto &T = V->getPayload();
    // Only use the first 8 elements in the hash. It's likely that if two
    // tensors have different content they will diverge quickly. Fall back to
    // full equality check in ConstsEqDedup.
    constexpr dim_t maxNumEls = 8;
    dim_t numEls = std::min((dim_t)T.getType().size(), maxNumEls);
    dim_t bufSize = T.getType().getElementSize() * numEls;
    auto *data = T.getUnsafePtr();
    for (size_t i = 0; i < bufSize; i++) {
      hash = llvm::hash_combine(hash, data[i]);
    }
    return hash;
  }
};

/// A helper type implementing the Constants equality predicate that can be
/// used when Constant pointers are used as keys in hash maps for
/// deduplication.
struct ConstsEqDedup {
  bool operator()(const Constant *lhs, const Constant *rhs) const {
    // Only consider Constants for deduplication if they have the same type.
    if (lhs->getType() != rhs->getType()) {
      return false;
    }
    // Only dedup Constants if they're bit exact matches.
    return lhs->getPayload().isBitwiseEqual(rhs->getPayload());
  }
};

} // namespace

/// Deduplicates Constants in the Module \p M. Applicable Constants for
/// deduplication must have the same data. \returns whether any Constants were
/// deduplicated.
static bool deduplicateConstants(Module *M) {
  // Map from Constants to other Constants that are equivalent for purposes of
  // deduplication.
  std::unordered_map<Constant *, Constant *, ConstsHasherDedup, ConstsEqDedup>
      duplicateConstants;

  bool changed = false;
  for (auto &C : M->getConstants()) {
    // Only perform deduplication on consts of small enough size. Otherwise
    // just skip them. constDedupSizeOpt defaults to 256 as a heuristic, to
    // keep compile time reasonable.
    size_t maxNumEls = constDedupSizeOpt;
    size_t numEls = C->getType()->size();
    if (numEls > maxNumEls) {
      continue;
    }

    // Try to find a Constant that has the same data as the current one.
    auto foundI = duplicateConstants.find(C);
    if (foundI == duplicateConstants.end()) {
      // No node equivalent to the current one has been seen yet. Remember
      // this Constant, so that the next occurrence can be replaced by this
      // one.
      duplicateConstants.emplace(C, C);
      assert(duplicateConstants.find(C) != duplicateConstants.end());
      continue;
    }
    Constant *foundC = foundI->second;
    assert(C != foundC && "Constants should not be visited multiple times.");

    // Replace current Constant by a found Constant, which is equivalent to
    // it.
    C->getOutput().replaceAllUsesOfWith(foundC);
    changed = true;
  }
  return changed;
}

/// Common Subexpression Elimination.
bool CSE::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  CSEVisitor visitor;

  bool changed = deduplicateConstants(F->getParent());

  // Perform CSE on all nodes.
  for (auto &N : F->getNodes()) {
    N.visit(nullptr, &visitor);
  }
  // TODO: Change Visitors to return whether they modified the Function they
  // are contained in. For now conservatively set changed to true;
  changed = true;
  return changed;
}

/// Fold Nodes into SplatNodes.
bool OptimizeSplat::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  for (Node &node : F->getNodes()) {
    // Slice(Splat(args)) -> Splat(args')
    if (SliceNode *sliceNode = dyn_cast<SliceNode>(&node)) {
      SplatNode *splatNode = dyn_cast<SplatNode>(sliceNode->getInput());
      if (!splatNode) {
        continue;
      }
      SplatNode *newSplatNode =
          F->createSplat(sliceNode->getName(), sliceNode->getResult().getType(),
                         splatNode->getValue());
      sliceNode->getResult().replaceAllUsesOfWith(newSplatNode);
      changed = true;
      continue;
    }

    // Clip(Splat(args)) -> Splat(args')
    if (ClipNode *clipNode = dyn_cast<ClipNode>(&node)) {
      SplatNode *splatNode = dyn_cast<SplatNode>(clipNode->getInput());
      if (!splatNode) {
        continue;
      }
      const float newSplatVal =
          std::min(std::max(splatNode->getValue(), clipNode->getMin()),
                   clipNode->getMax());

      SplatNode *newSplatNode = nullptr;
      if (newSplatVal == splatNode->getValue()) {
        // No need to crate a new Splat.
        newSplatNode = splatNode;
      } else {
        newSplatNode = F->createSplat(
            splatNode->getName().str() + clipNode->getName().str(),
            splatNode->getResult().getType(), newSplatVal);
      }

      clipNode->getResult().replaceAllUsesOfWith(newSplatNode->getResult());
      changed = true;
      continue;
    }
  }
  return changed;
}

/// Optimize TransposeNode into ReshapeNode when it actually moves no data.
bool OptimizeTransposeIntoReshape::run(Function *F,
                                       const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;

  for (auto &node : F->getNodes()) {
    auto *TR = dyn_cast<TransposeNode>(&node);
    if (!TR) {
      continue;
    }
    auto inputNode = TR->getInput();
    auto inputDims = inputNode.dims();
    auto outputDims = TR->getResult().dims();
    // The transformation is not possible if alignments different from 1 are
    // used for any dimension.
    if (!inputNode.getType()->isEqual(F->getParent()->uniqueTypeWithNewShape(
            inputNode.getType(), inputDims))) {
      continue;
    }
    if (!TR->getResult().getType()->isEqual(
            F->getParent()->uniqueTypeWithNewShape(TR->getResult().getType(),
                                                   outputDims))) {
      continue;
    }
    // Transpose moves no data if input/output dimensions match after they both
    // drop dimensions of size 1. E.g. transposing [1 5 1 15] into [5 15 1 1]
    // produces vectors (1, 3) for both dimensions so optimization is executed.
    auto shuffle = TR->getShuffle();
    ShapeVector inDims;
    ShapeVector outDims;
    for (size_t i = 0; i < inputDims.size(); i++) {
      if (inputDims[i] != 1) {
        inDims.push_back(i);
      }
      if (outputDims[i] != 1) {
        outDims.push_back(shuffle[i]);
      }
    }
    if (inDims != outDims) {
      continue;
    }
    auto *RS =
        F->createReshape(TR->getName(), inputNode, outputDims, TR->getLayout());
    TR->getResult().replaceAllUsesOfWith(RS);
    changed = true;
  }

  return changed;
}

bool EliminateNoopTile::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;

  for (auto &node : F->getNodes()) {
    if (auto *tileNode = dyn_cast<TileNode>(&node)) {
      // If the TileNode tiles only once, eliminate it.
      if (tileNode->getCount() == 1) {
        tileNode->getResult().replaceAllUsesOfWith(tileNode->getInput());
        changed = true;
      }
    }
  }

  return changed;
}

/// Eliminate noop Slice(Node) -> Node.
bool EliminateNoopSlice::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;

  for (auto &node : F->getNodes()) {
    SliceNode *sliceNode = dyn_cast<SliceNode>(&node);
    if (!sliceNode) {
      continue;
    }

    // If input and result have different types then this is not a noop.
    if (sliceNode->getInput().getType() != sliceNode->getResult().getType()) {
      continue;
    }

    sliceNode->getResult().replaceAllUsesOfWith(sliceNode->getInput());
    changed = true;
  }

  return changed;
}

/// Optimize reshape nodes.
bool OptimizeReshape::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  for (auto &node : F->getNodes()) {
    auto *reshapeNode = dyn_cast<ReshapeNode>(&node);
    if (!reshapeNode) {
      continue;
    }
    auto inputNode = reshapeNode->getNthInput(ReshapeNode::InputIdx);
    // Eliminate ReshapeNode when the input is already the correct shape.
    if (inputNode.dims() == reshapeNode->getResult().dims()) {
      reshapeNode->getResult().replaceAllUsesOfWith(inputNode);
      continue;
    }
    // Reshape(Splat(args)) -> Splat(args').
    auto *splatNode = dyn_cast<SplatNode>(inputNode);
    if (splatNode && splatNode->hasOneUse()) {
      // Splat node with more than one use can not be transformed, otherwise
      // we would increase the number of splats, which may lead to increased
      // memory consumption during the execution of the NN model.
      auto *newSplatNode = F->createSplat(splatNode->getName(),
                                          reshapeNode->getResult().getType(),
                                          splatNode->getValue());
      reshapeNode->getResult().replaceAllUsesOfWith(newSplatNode);
      changed = true;
      continue;
    }
    // Reshape(Reshape(x)) -> Reshape(x).
    auto *reshapeNodeInput = dyn_cast<ReshapeNode>(inputNode);
    if (reshapeNodeInput && reshapeNodeInput->hasOneUse()) {
      auto *newReshape = F->createReshape(
          reshapeNode->getName(), reshapeNodeInput->getInput(),
          reshapeNode->getResult().dims(), reshapeNode->getLayout());
      reshapeNode->getResult().replaceAllUsesOfWith(newReshape);
      changed = true;
      continue;
    }
    // Reshape(Constant) -> Constant'.
    if (auto *C = dyn_cast<Constant>(inputNode)) {
      // Create a new Constant with the type of the reshape.
      auto layout =
          CanonicalTensorLayout::getInstance().getNthResultLayoutRequirements(
              reshapeNode, ReshapeNode::ResultIndices::ResultIdx);
      auto *newC = F->getParent()->createConstant(
          reshapeNode->getResult().getType(), C->getName(), layout);
      // Create an unowned view of the original tensor with the correct shape,
      // and assign it to the new Constant.
      Tensor reshapedT = C->getPayload().getUnowned(reshapeNode->getDims());
      newC->assign(&reshapedT);
      reshapeNode->getResult().replaceAllUsesOfWith(newC);
      changed = true;
      continue;
    }
  }
  return changed;
}

/// Optimize: Max(Splat(), otherInput) or Max(otherInput, Splat()) for
/// quantized operations.
/// Splat and Max can be eliminated if Splat value cannot impact the result.
/// For example, Max and Splat can be removed if splat value is smaller
/// than quantization range [min, max].
/// \returns if anything was changed in the given function.
static bool optimizeQuantizedMaxSplat(Function *F) {
  LOG_SCOPE(F->getLogContext(), "optimizeQuantizedMaxSplat")

  bool changed = false;
  // The following optimizations need to be performed after all
  // quantize/dequantize/rescale optimizations are done.
  for (auto &node : F->getNodes()) {
    // Potentially nop quantized Max can be eliminated.
    // Likely MaxNode has same types for LHS/RHS and Result, make sure
    // it's the case.
    if (auto *MN = dyn_cast<MaxNode>(&node)) {
      if (!MN->getResult().getType()->isQuantizedType() ||
          MN->getResult().getType() != MN->getLHS().getType() ||
          MN->getResult().getType() != MN->getRHS().getType()) {
        continue;
      }

      // Check for input Splat node.
      if (!isa<SplatNode>(MN->getLHS()) && !isa<SplatNode>(MN->getRHS())) {
        continue;
      }

      Node *splatNode =
          isa<SplatNode>(MN->getLHS()) ? MN->getLHS() : MN->getRHS();
      Node *otherInput =
          isa<SplatNode>(MN->getLHS()) ? MN->getRHS() : MN->getLHS();

      // If splat value is smaller than values that can be covered by
      // quantization [min,max] range then just remove MaxNode operation.
      float splatValue = (dyn_cast<SplatNode>(splatNode))->getValue();
      float min = MN->getResult().getType()->getQuantizedValueRange().first;
      if (splatValue <= min) {
        changed = true;
        MN->getResult().replaceAllUsesOfWith(otherInput);
      }
      continue;
    }
    // Potentially nop quantized ReLU can be eliminated.
    if (auto *RN = dyn_cast<ReluNode>(&node)) {
      if (!RN->getResult().getType()->isQuantizedType() ||
          RN->getResult().getType() != RN->getInput().getType()) {
        continue;
      }

      Node *input = RN->getInput();

      // If zero is smaller or equal than values that can be covered by
      // quantization [min,max] range then just remove ReluNode operation.
      float min = RN->getResult().getType()->getQuantizedValueRange().first;
      if (0.0f <= min) {
        changed = true;
        RN->getResult().replaceAllUsesOfWith(input);
      }
      continue;
    }
  }
  return changed;
}

/// \returns A value representing the given \p constant converted
/// to the destination type \p dstTy. If the conversion is not
/// possible, this method returns NodeValue().
static NodeValue convertConstant(Module &mod, Constant &constant,
                                 TypeRef dstTy) {
  // Sort out the easy case first.
  if (constant.getType() == dstTy) {
    return constant.getOutput();
  }
  auto modifyConstantTyAndGet = [&]() -> Constant & {
    Constant *oneUseCst = getUniquelyUsedConstant(&mod, constant);
    assert(oneUseCst &&
           "We should always be able to get a constant from a constant!");
    // This type setting updates the type of the node.
    // The underlying tensor still needs to be converted after this call.
    oneUseCst->setType(Storage::OutputIdx, dstTy);
    return *oneUseCst;
  };
  const Tensor &tensor = constant.getPayload();
  switch (tensor.getElementType()) {
  case ElemKind::FloatTy:
  case ElemKind::Float16Ty:
    switch (dstTy->getElementType()) {
    case ElemKind::FloatTy:
    case ElemKind::Float16Ty: {
      // Plain conversion: {FloatTy, Float16Ty} -> {FloatTy, Float16Ty}.
      Constant &constantToBeModified = modifyConstantTyAndGet();
      constantToBeModified.getPayloadMutable().convertToType(
          dstTy->getElementType());
      return constantToBeModified.getOutput();
    }
    case ElemKind::Int32QTy:
    case ElemKind::Int16QTy:
    case ElemKind::Int8QTy: {
      // Quantization: {FloatTy, Float16Ty} -> Quantized type.
      Constant &constantToBeModified = modifyConstantTyAndGet();
      TensorQuantizationParams params{dstTy->getScale(), dstTy->getOffset()};
      Tensor &tensorToBeModified = constantToBeModified.getPayloadMutable();
      // Right now we only quantize fp32 value.
      // Add an assert on that, so that if it changes, we adapt the
      // following code. Adapting the code would required to
      // teach quantizeTensor how to deal with Float16Ty.
      assert(tensor.getType().isFPType() &&
             "Type quantization not implemented");
      tensorToBeModified = quantization::quantizeTensor(
          tensorToBeModified, params, dstTy->getElementType());
      return constantToBeModified.getOutput();
    }
    default:
      // Quantization: {FloatTy, Float16Ty} -> Int[16|32]QTy.
      // Plain conversion: {FloatTy, Float16Ty} -> Int64ITy.
      return NodeValue();
    }
  case ElemKind::UInt8FusedQTy: {
    if (dstTy->getElementType() != ElemKind::UInt8FusedFP16QTy) {
      return NodeValue();
    }
    auto *NC =
        mod.createConstant(dstTy, constant.getName(), constant.getLayout());
    NC->getPayloadMutable() =
        tensor.getCopyConvertedToType(dstTy->getElementType());
    return NC->getOutput();
  }

  default:
    // For now we don't see other quantize, dequantize, or rescale nodes
    // directly attached to constants.
    // Thus don't add code that will never be executed.
    // Dequantization: Int[8|16|32]QTy -> {FloatTy, Float16Ty, Int64I}.
    // Rescale: Int[8|16|32]QTy -> Int[8|16|32]QTy.
    // Plain conversion: Int64ITy -> {FloatTy, Float16Ty}.
    // Quantization: Int64ITy -> Int[8|16|32]QTy.
    return NodeValue();
  }
}

/// Compute number of significant bits that are used to represent data of type
/// \p kind. For FP, it is the number of bits in mantissa, for integers it's the
/// number of bits except sign bit.
/// \p returns number of significant bits of \p kind.
/// TODO: Currently, for all supported types wider mantissa also means wider
/// exponent. If we add a type for which this is not true, we should check both
/// mantissa and exponent.
static size_t numSignificantBits(ElemKind kind) {
  switch (kind) {
  case ElemKind::BoolTy:
    return std::numeric_limits<bool>::digits;
  case ElemKind::Int8QTy:
    return std::numeric_limits<int8_t>::digits;
  case ElemKind::UInt8QTy:
  case ElemKind::UInt8FusedQTy:
    return std::numeric_limits<uint8_t>::digits;
  case ElemKind::Float16Ty:
    // Custom type with layout 0 00000 0000000000.
    return 10;
  case ElemKind::Int16QTy:
    return std::numeric_limits<int16_t>::digits;
  case ElemKind::FloatTy:
    return std::numeric_limits<float>::digits;
  case ElemKind::Int32QTy:
  case ElemKind::Int32ITy:
    return std::numeric_limits<int32_t>::digits;
  case ElemKind::Int64ITy:
    return std::numeric_limits<int64_t>::digits;
  default:
    // Avoid compiler warning.
    break;
  }
  llvm_unreachable("Unknown type!");
}

/// Returns true if casting value from \p srcTy to \p destTy may change it. As
/// implication of this, casting value from \p srcTy to \p destTy and back may
/// produce different value than before cast.
static bool isValueChangingCast(TypeRef srcTy, TypeRef destTy) {
  // FP-to-Int conversion may lead to loss of fraction, so it's not NOOP.
  if (srcTy->isFPType() && !destTy->isFPType()) {
    return true;
  }
  // Narrowing transform (e.g. int64 to int32) may lead to loss of
  // significant senior bits, so it's not NOOP.
  ElemKind srcElKind = srcTy->getElementType();
  ElemKind convElKind = destTy->getElementType();
  if (numSignificantBits(srcElKind) > numSignificantBits(convElKind)) {
    return true;
  }
  return false;
}

/// Optimize away redundant ClipNodes.
/// We basically turn "Clip(Clip(Clip(A)))" to "Clip(A)".
bool OptimizeClips::run(Function *F, const CompilationContext &cctx) {
  int clipsEliminated = 0;
  for (Node &node : F->getNodes()) {
    ClipNode *clip = dyn_cast<ClipNode>(&node);
    if (!clip) {
      continue;
    }
    float min = clip->getMin();
    float max = clip->getMax();
    if (auto *clipPrev = dyn_cast<ClipNode>(clip->getInput().getNode())) {
      float minPrev = clipPrev->getMin();
      float maxPrev = clipPrev->getMax();
      auto *newClip =
          F->createClip(clipPrev->getName(), clipPrev->getInput().getNode(),
                        std::max(minPrev, min), std::min(maxPrev, max));
      clip->getResult().replaceAllUsesOfWith(newClip);
      ++clipsEliminated;
    }
  }

  return clipsEliminated;
}

/// Optimize away ConvertToNode.
/// This basically turns "conversion(conversion A to B) to C"
/// into noop if all of the conditions below are met:
///  - the type of A and C are the same;
///  - A->B is not a FP-to-Int conversion;
///  - A->B is not a narrowing conversion.
bool OptimizeConversions::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());

  bool changed = false;
  for (auto &node : F->getNodes()) {
    if (auto *CN = llvm::dyn_cast<ConvertToNode>(&node)) {

      // Eliminate no-op conversion.
      if (CN->getInput().getType() == CN->getResult().getType()) {
        CN->getResult().replaceAllUsesOfWith(CN->getInput());
        changed = true;
        continue;
      }

      // Perform conversion of constants.
      if (auto *BN = llvm::dyn_cast<Constant>(CN->getInput())) {
        auto newConst =
            convertConstant(*F->getParent(), *BN, CN->getResult().getType());
        CN->getResult().replaceAllUsesOfWith(newConst, F);
        changed = true;
        continue;
      }

      // Simplify a chain of conversions A -> B -> C to A -> C, unless A -> B
      // is a narrowing cast.
      if (auto *BN = llvm::dyn_cast<ConvertToNode>(CN->getInput())) {
        auto AN = BN->getInput();

        // Do not optimize away narrowing casts.
        if (!isValueChangingCast(AN.getType(), BN->getResult().getType())) {
          auto *newCast =
              F->createConvertTo(CN->getName(), AN, CN->getResult().getType());
          CN->getResult().replaceAllUsesOfWith(newCast);
          changed = true;
          continue;
        }
      }
    }
  }
  return changed;
}

/// \returns a cloned version of node \p N, but with each of the cloned node's
/// output types set to the corresponding type in \p types. The new node is
/// added to Function \p F.
/// \pre types.size() == N->getNumResults()
static Node *cloneNodeWithNewTypes(Function *F, Node *N,
                                   llvm::ArrayRef<TypeRef> types) {
  assert(N->getNumResults() == types.size() &&
         "Number of types must equal number of results of the node.");

  Node *newNode = F->addNode(N->clone());
  for (size_t i = 0; i < types.size(); i++) {
    newNode->setType(i, types[i]);
  }

  return newNode;
}

template <class T, class U>
using enable_if_same_t = std::enable_if<std::is_same<T, U>::value, U>;
#define FUNCTION_ENABLE_IF_TEMPLATE(NODE_NAME_)                                \
  template <class T, typename... Args>                                         \
  typename enable_if_same_t<T, NODE_NAME_##Node>::type static

FUNCTION_ENABLE_IF_TEMPLATE(AvgPool) * createNode(Function &F, Args... args) {
  return F.createAvgPool(args...);
}
FUNCTION_ENABLE_IF_TEMPLATE(MaxPool) * createNode(Function &F, Args... args) {
  return F.createMaxPool(args...);
}
FUNCTION_ENABLE_IF_TEMPLATE(Add)
*createNode(Function &F, Args... args) { return F.createAdd(args...); }
FUNCTION_ENABLE_IF_TEMPLATE(Sub)
*createNode(Function &F, Args... args) { return F.createSub(args...); }
FUNCTION_ENABLE_IF_TEMPLATE(Mul)
*createNode(Function &F, Args... args) { return F.createMul(args...); }
FUNCTION_ENABLE_IF_TEMPLATE(Div)
*createNode(Function &F, Args... args) { return F.createDiv(args...); }
FUNCTION_ENABLE_IF_TEMPLATE(Min)
*createNode(Function &F, Args... args) { return F.createMin(args...); }
FUNCTION_ENABLE_IF_TEMPLATE(Max)
*createNode(Function &F, Args... args) { return F.createMax(args...); }
FUNCTION_ENABLE_IF_TEMPLATE(MatMul)
*createNode(Function &F, Args... args) { return F.createMatMul(args...); }

/// Sink Rescale down with Pooling node.
/// PoolingNode(Rescale(X)) -> Rescale(PoolingNode(X)).
/// Apply this transformation for AvgPool and MaxPool.
template <typename T>
static bool sinkDownRescaleToPoolingNode(Function &F, T *PN) {
  LOG_SCOPE(F.getLogContext(), "sinkDownRescaleToPoolingNode")

  bool changed = false;

  if (auto *rescale = dyn_cast<RescaleQuantizedNode>(PN->getInput())) {
    T *newPN = createNode<T>(F, PN->getName(), rescale->getInput(),
                             PN->getKernels(), PN->getStrides(), PN->getPads());
    auto rescaleOutTy = F.getParent()->uniqueTypeWithNewShape(
        rescale->getResult().getType(), PN->getResult().dims());
    auto *newRescale = F.createRescaleQuantized(
        rescale->getName(), newPN->getResult(), rescaleOutTy);
    PN->getResult().replaceAllUsesOfWith(newRescale);
    for (size_t i = 1; i < PN->getNumResults(); i++) {
      PN->getNthResult(i).replaceAllUsesOfWith(newPN->getNthResult(i));
    }
    changed = true;
  }

  return changed;
}

/// Combine Rescale down with Arithmetic node.
///   ArithmeticNode(Rescale(X), Rescale(Y)) -> ArithmeticNode(X, Y).
///   ArithmeticNode(Rescale(X), Y) -> ArithmeticNode(X, Y).
///   ArithmeticNode(X, Rescale(Y)) -> ArithmeticNode(X, Y).
/// Apply this optimization for Add, Sub, Mul, Div, Min, Max.
template <typename T>
static bool combineDownRescaleToArithmeticNode(Function &F, T *AN) {
  LOG_SCOPE(F.getLogContext(), "combineDownRescaleToArithmeticNode")

  bool changed = false;

  if (auto *rescale = dyn_cast<RescaleQuantizedNode>(AN->getLHS())) {
    T *newAN = createNode<T>(F, AN->getName(), AN->getResult().getType(),
                             rescale->getInput(), AN->getRHS());
    AN->getResult().replaceAllUsesOfWith(newAN);
    AN = newAN;
    changed = true;
  }
  if (auto *rescale = dyn_cast<RescaleQuantizedNode>(AN->getRHS())) {
    T *newAN = createNode<T>(F, AN->getName(), AN->getResult().getType(),
                             AN->getLHS(), rescale->getInput());
    AN->getResult().replaceAllUsesOfWith(newAN);
    changed = true;
  }

  return changed;
}

/// Sink Rescale nodes down when possible.
/// \returns if anything was changed in the given function.
static bool sinkRescaleQuantizedNode(Function *F) {
  LOG_SCOPE(F->getLogContext(), "sinkRescaleQuantizedNode");
  bool changed = false;
  for (auto &node : F->getNodes()) {
    // Sink Rescale below Reshape node.
    // Reshape(Rescale(X)) -> Rescale(Reshape(X)).
    if (auto *reshape = dyn_cast<ReshapeNode>(&node)) {
      auto *rescale = dyn_cast<RescaleQuantizedNode>(reshape->getInput());
      if (!rescale) {
        continue;
      }

      auto *newReshape =
          F->createReshape(reshape->getName(), rescale->getInput(),
                           reshape->getResult().dims(), reshape->getLayout());
      auto *newRescale = F->createRescaleQuantized(
          rescale->getName(), newReshape, reshape->getResult().getType());
      reshape->getResult().replaceAllUsesOfWith(newRescale);

      changed = true;
      continue;
    }

    // Sink Rescale below Slice node.
    // Slice(Rescale(X)) -> Rescale(Slice(X)).
    if (auto *slice = dyn_cast<SliceNode>(&node)) {
      auto *rescale = dyn_cast<RescaleQuantizedNode>(slice->getInput());
      if (!rescale) {
        continue;
      }

      auto sliceOutTy = F->getParent()->uniqueTypeWithNewShape(
          rescale->getInput().getType(), slice->getResult().dims());
      auto *newSlice = F->createSlice(slice->getName(), rescale->getInput(),
                                      slice->getStart(), sliceOutTy);
      auto *newRescale = F->createRescaleQuantized(
          rescale->getName(), newSlice, slice->getResult().getType());
      slice->getResult().replaceAllUsesOfWith(newRescale);

      changed = true;
      continue;
    }

    // Sink Rescale below Transpose node.
    // Transpose(Rescale(X)) -> Rescale(Transpose(X)).
    if (auto *transpose = dyn_cast<TransposeNode>(&node)) {
      auto *rescale = dyn_cast<RescaleQuantizedNode>(transpose->getInput());
      if (!rescale) {
        continue;
      }

      auto *newTranspose =
          F->createTranspose(transpose->getName(), rescale->getInput(),
                             transpose->getShuffle(), transpose->getLayout());
      auto rescaleOutTy = F->getParent()->uniqueTypeWithNewShape(
          rescale->getResult().getType(), transpose->getResult().dims());
      auto *newRescale = F->createRescaleQuantized(rescale->getName(),
                                                   newTranspose, rescaleOutTy);
      transpose->getResult().replaceAllUsesOfWith(newRescale);

      changed = true;
      continue;
    }

    if (auto *PN = dyn_cast<AvgPoolNode>(&node)) {
      changed |= sinkDownRescaleToPoolingNode<AvgPoolNode>(*F, PN);
      continue;
    }

    if (auto *PN = dyn_cast<MaxPoolNode>(&node)) {
      changed |= sinkDownRescaleToPoolingNode<MaxPoolNode>(*F, PN);
      continue;
    }

    // Combine Rescale down with FullyConnected node.
    // FullyConnected(Rescale(X)) -> FullyConnected(X).
    if (auto *FC = dyn_cast<FullyConnectedNode>(&node)) {
      auto *rescale = dyn_cast<RescaleQuantizedNode>(FC->getInput());
      if (!rescale) {
        continue;
      }

      auto *newFC = F->createFullyConnected(FC->getName(), rescale->getInput(),
                                            FC->getWeights(), FC->getBias(),
                                            FC->getResult().getType());
      FC->getResult().replaceAllUsesOfWith(newFC);

      changed = true;
      continue;
    }

    // Combine Rescale down with Convolution node.
    // Convolution(Rescale(X), F, B) -> Convolution(X, F, B).
    // Convolution(X, Rescale(F), B) -> Convolution(X, F, B).
    // Convolution(X, F, Rescale(B)) -> Convolution(X, F, B).
    // ... and different combinations.
    if (auto *CN = dyn_cast<ConvolutionNode>(&node)) {
      auto *rescaleX = dyn_cast<RescaleQuantizedNode>(CN->getInput());
      auto *rescaleF = dyn_cast<RescaleQuantizedNode>(CN->getFilter());
      auto *rescaleB = dyn_cast<RescaleQuantizedNode>(CN->getBias());
      auto newX = rescaleX ? rescaleX->getInput() : CN->getInput();
      auto newF = rescaleF ? rescaleF->getInput() : CN->getFilter();
      auto newB = rescaleB ? rescaleB->getInput() : CN->getBias();
      if (rescaleX || rescaleF || rescaleB) {
        auto *newCN = F->createConv(CN->getName(), newX, newF, newB,
                                    CN->getResult().getType(), CN->getKernels(),
                                    CN->getStrides(), CN->getPads(),
                                    CN->getGroup(), CN->getDilation());
        CN->getResult().replaceAllUsesOfWith(newCN);
        changed = true;
      }
      continue;
    }

    if (auto *AN = dyn_cast<AddNode>(&node)) {
      changed |= combineDownRescaleToArithmeticNode<AddNode>(*F, AN);
      continue;
    }
    if (auto *AN = dyn_cast<SubNode>(&node)) {
      changed |= combineDownRescaleToArithmeticNode<SubNode>(*F, AN);
      continue;
    }
    if (auto *AN = dyn_cast<MulNode>(&node)) {
      changed |= combineDownRescaleToArithmeticNode<MulNode>(*F, AN);
      continue;
    }
    if (auto *AN = dyn_cast<DivNode>(&node)) {
      changed |= combineDownRescaleToArithmeticNode<DivNode>(*F, AN);
      continue;
    }
    if (auto *AN = dyn_cast<MinNode>(&node)) {
      changed |= combineDownRescaleToArithmeticNode<MinNode>(*F, AN);
      continue;
    }
    if (auto *AN = dyn_cast<MaxNode>(&node)) {
      changed |= combineDownRescaleToArithmeticNode<MaxNode>(*F, AN);
      continue;
    }

    // Combine Rescale down with Relu node.
    //   ReluNode(Rescale(in)) -> ReluNode(in).
    if (auto *RN = dyn_cast<ReluNode>(&node)) {
      if (auto *rescale = dyn_cast<RescaleQuantizedNode>(RN->getInput())) {
        auto *newRN = F->createRELU(RN->getName(), rescale->getInput(),
                                    RN->getResult().getType());
        RN->getResult().replaceAllUsesOfWith(newRN);
        changed = true;
      }
      continue;
    }

    if (auto *MN = dyn_cast<MatMulNode>(&node)) {
      changed |= combineDownRescaleToArithmeticNode<MatMulNode>(*F, MN);
      continue;
    }
  }

  return changed;
}

/// Eliminate node sequences that are related to quantization.
/// \returns if anything was changed in the given function.
bool OptimizeQuantization::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  // A worklist that contains the nodes to process.
  std::vector<Node *> worklist;

  // Add all of the interesting nodes to the worklist.
  for (auto &node : F->getNodes()) {
    if (isa<QuantizeNode>(node) || isa<DequantizeNode>(node) ||
        isa<RescaleQuantizedNode>(node)) {
      worklist.push_back(&node);
    }
  }

  while (!worklist.empty()) {
    // Take a node from the worklist.
    Node *node = worklist.back();
    worklist.pop_back();

    if (auto *Q = dyn_cast<QuantizeNode>(node)) {
      if (auto *DQ = dyn_cast<DequantizeNode>(Q->getInput())) {
        // Quantize(Dequantize(X)) -> RescaleQuantized(X)
        // If the quantization-dequantization sequence does not change the
        // type then we can simply drop them without adding a requantization
        // node.
        changed = true;
        if (DQ->getInput().getType() == Q->getResult().getType()) {
          Q->getResult().replaceAllUsesOfWith(DQ->getInput());
          continue;
        }

        auto *RS = F->createRescaleQuantized(Q->getName(), DQ->getInput(),
                                             Q->getResult().getType());
        Q->getResult().replaceAllUsesOfWith(RS);

        // We may be able to optimize this rescale node. Remember to visit
        // this new node and try to optimize it later.
        worklist.push_back(RS);
        continue;
      }

      if (auto *C = dyn_cast<Constant>(Q->getInput())) {
        // Quantize(Constant) -> Constant
        // Note, it does not really matter how many usages this Constant has.
        // Quantized graph will use optimized Constant and other functions will
        // refer to the floating point original Constant.
        NodeValue NC =
            convertConstant(*F->getParent(), *C, Q->getResult().getType());
        if (NC == NodeValue()) {
          continue;
        }
        changed = true;
        Q->getResult().replaceAllUsesOfWith(NC);
        continue;
      }

      if (auto *SN = dyn_cast<SplatNode>(Q->getInput())) {
        // Quantize(Splat) -> Splat'
        changed = true;
        SplatNode *newSN = F->createSplat(
            SN->getName(), Q->getResult().getType(), SN->getValue());
        Q->getResult().replaceAllUsesOfWith(newSN);
        continue;
      }
    }

    if (auto *DQ = dyn_cast<DequantizeNode>(node)) {
      if (auto *Q = dyn_cast<QuantizeNode>(DQ->getInput())) {
        // Dequantize(Quantize(X)) -> X
        changed = true;
        DQ->getResult().replaceAllUsesOfWith(Q->getInput());
        continue;
      }
      // Fold the rescale into the following Dequantize.
      // Dequantize(rescale) -> Dequantize()
      if (auto *RS = dyn_cast<RescaleQuantizedNode>(DQ->getInput())) {
        changed = true;
        auto *newRS = F->createDequantize(DQ->getName(), RS->getInput());
        DQ->getResult().replaceAllUsesOfWith(newRS);

        // We may be able to optimize this rescale node. Remember to visit
        // this new node and try to optimize it later.
        worklist.push_back(newRS);
        continue;
      }
      if (auto *SN = dyn_cast<SplatNode>(DQ->getInput())) {
        // Dequantize(Splat) -> Splat'
        changed = true;
        SplatNode *newSN = F->createSplat(
            SN->getName(), DQ->getResult().getType(), SN->getValue());
        DQ->getResult().replaceAllUsesOfWith(newSN);
        continue;
      }
    }

    if (auto *RS = dyn_cast<RescaleQuantizedNode>(node)) {
      if (RS->getInput().getType() == RS->getResult().getType()) {
        // If rescale does not change the type, then simply drop it.
        changed = true;
        RS->getResult().replaceAllUsesOfWith(RS->getInput());
        continue;
      }

      // All optimizations in this scope below combine a Rescale up into or
      // above the Rescale's input X. If X has multiple users then this merging
      // will duplicate X, just with a different output scale/offset. If X is
      // not a Splat then this is likely not desired, as it means a
      // computational node (e.g. Add) is duplicated.
      if (!RS->getInput().hasOneUse() && !isa<SplatNode>(RS->getInput())) {
        continue;
      }

      // Combine the rescale node up into its parent node.
      // Rescale(Node()) -> 'Node().
      bool addNewNodeToWorklist = false;
      switch (RS->getInput().getNode()->getKind()) {
      case Kinded::Kind::RescaleQuantizedNodeKind:
      case Kinded::Kind::QuantizeNodeKind:
        addNewNodeToWorklist = true;
      case Kinded::Kind::SplatNodeKind:
      case Kinded::Kind::AddNodeKind:
      case Kinded::Kind::SubNodeKind:
      case Kinded::Kind::MulNodeKind:
      case Kinded::Kind::DivNodeKind:
      case Kinded::Kind::MinNodeKind:
      case Kinded::Kind::MatMulNodeKind:
      case Kinded::Kind::ConvolutionNodeKind:
      case Kinded::Kind::SparseLengthsWeightedSumNodeKind: {
        changed = true;
        Node *newNode =
            cloneNodeWithNewTypes(F, RS->getInput(), RS->getResult().getType());
        RS->getResult().replaceAllUsesOfWith(newNode);
        if (addNewNodeToWorklist) {
          worklist.push_back(newNode);
        }
        continue;
      }
      default:;
      }

      if (auto *MN = dyn_cast<MaxNode>(RS->getInput())) {
        // Rescale(MAX(X, Y)) -> MAX(Rescale(X), Rescale(Y)).
        // It's okay to rescale the operands because even if the output range
        // is smaller then truncation would have happened during the rescale.
        // On values that are outside of the range we just moved the
        // truncation to a different location.
        changed = true;
        auto name = RS->getName();
        auto *L = F->createRescaleQuantized(name, MN->getLHS(),
                                            RS->getResult().getType());
        auto *R = F->createRescaleQuantized(name, MN->getRHS(),
                                            RS->getResult().getType());
        auto *newMN = F->createMax(MN->getName(), L, R);
        worklist.push_back(L);
        worklist.push_back(R);
        RS->getResult().replaceAllUsesOfWith(newMN);
        continue;
      }
    } // Handle RescaleQuantizedNode
  }   // For each item in the worklist.

  changed |= optimizeQuantizedMaxSplat(F);

  // If nothing has changed then sink rescale quantization nodes.
  if (!changed) {
    changed = sinkRescaleQuantizedNode(F);
  }
  return changed;
}

void glow::convertPlaceholdersToConstants(Function *F,
                                          const PlaceholderBindings &bindings,
                                          llvm::ArrayRef<Placeholder *> phs) {
  LOG_SCOPE(F->getLogContext(), "convertPlaceholdersToConstants")

  auto *M = F->getParent();
  for (auto &PH : F->findPlaceholders()) {
    if (std::find(phs.begin(), phs.end(), PH) != phs.end()) {
      continue;
    }
    auto *tensor = bindings.get(PH);
    if (!tensor) {
      continue;
    }
    auto *constant = M->createConstant(PH->getName(), *tensor, PH->getLayout());
    PH->getOutput().replaceAllUsesOfWith(constant, F);
  }
}

/// \returns True if the \p node sub-tree corresponds to a scalar
/// (Constant or Splat) and return the float value in \p retFloat.
static bool getFloatScalar(Node *node, float *retFloat) {
  // Iterate across potential Tile Nodes (implied by broadcasting if any).
  auto *n = node;
  while (auto *TN = dyn_cast<TileNode>(n)) {
    n = TN->getInput();
  }

  // After potential Tile nodes, it should be a singleton constant scalar node
  // with any shape corresponding to one single element.
  if (auto *constNode = dyn_cast<Constant>(n)) {
    if ((constNode->getType()->getElementType() != ElemKind::FloatTy) ||
        (constNode->getType()->size() != 1)) {
      return false;
    }
    auto valueH = constNode->getHandle<float>();
    std::vector<dim_t> coord(constNode->getType()->dims().size(), 0);
    *retFloat = valueH.at(coord);
    return true;
  }
  if (auto *splatNode = dyn_cast<SplatNode>(n)) {
    *retFloat = splatNode->getValue();
    return true;
  }

  return false;
}

/// Fold leakyRelu operations expressed as a sub-graph Max(A, Mul(A, scalar))
/// and replace it by PRelu(Splat).
bool FoldLeakyRelu::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());
  bool changed = false;
  auto &nodes = F->getNodes();
  for (auto &node : nodes) {
    auto *maxNode = dyn_cast<MaxNode>(&node);
    if (!maxNode) {
      continue;
    }
    NodeValue otherMaxOperand;
    MulNode *mulNode;
    if ((mulNode = dyn_cast<MulNode>(maxNode->getRHS()))) {
      otherMaxOperand = maxNode->getLHS();
    } else if ((mulNode = dyn_cast<MulNode>(maxNode->getLHS()))) {
      otherMaxOperand = maxNode->getRHS();
    } else {
      continue;
    }
    NodeValue otherMulOperand;
    float value;
    if (getFloatScalar(mulNode->getRHS(), &value)) {
      otherMulOperand = maxNode->getLHS();
    } else if (getFloatScalar(mulNode->getLHS(), &value)) {
      otherMulOperand = maxNode->getRHS();
    } else {
      continue;
    }
    if ((value <= 1.0f) && (otherMulOperand == otherMaxOperand)) {
      // The sub-tree is a Leaky-Relu, express it as a PRelu.
      auto *splat = F->createSplat(maxNode->getName(),
                                   mulNode->getResult().getType(), value);
      auto *PRelu = F->createPRELU(maxNode->getName(), otherMaxOperand, splat);
      maxNode->getResult().replaceAllUsesOfWith(PRelu);
      changed = true;
      continue;
    }
  }
  return changed;
}

/// Parameters that are used to define ChannelShuffle operators.
struct ChannelShuffleParams {
  size_t group;
  size_t kernel;
};

/// Compute the original parameters to the ChannelShuffle operator (represented
/// as ReshapeNode->TransposeNode->ReshapeNode) for which \p node is the leading
/// ReshapeNode. \returns The original ChannelShuffle parameters if possible and
/// empty Optional otherwise.
static llvm::Optional<ChannelShuffleParams>
getChannelShuffleParams(const ReshapeNode &node) {
  auto resM = llvm::Optional<ChannelShuffleParams>();

  llvm::ArrayRef<dim_t> inputDims = node.getInput().dims();
  llvm::ArrayRef<dim_t> resultDims = node.getDims();

  // Check that there is one more output dimension than input dimension.
  if (resultDims.size() != inputDims.size() + 1) {
    return resM;
  }

  // Find the first output dimension that doesn't match its corresponding input
  // dimension.
  ChannelShuffleParams params;
  bool found = false;
  for (size_t i = 0, e = resultDims.size(); i < e - 1; ++i) {
    if (inputDims[i] != resultDims[i]) {
      params.kernel = i;
      params.group = resultDims[i];
      found = true;
      break;
    }
  }

  // Double check the property that the mismatched output found dimension and
  // its successor together evenly multiply to the input dimension they
  // mismatched on.
  if (found && resultDims[params.kernel] * resultDims[params.kernel + 1] ==
                   inputDims[params.kernel]) {
    resM = params;
  }

  return resM;
}

// Fold Reshape->Transpose->Reshape into ChannelShuffle when applicable.
bool FoldChannelShuffle::run(Function *F, const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());

  bool changed = false;
  auto &nodes = F->getNodes();
  for (auto &node : nodes) {
    auto *RN2 = dyn_cast<ReshapeNode>(&node);
    if (!RN2) {
      continue;
    }

    auto *TR = dyn_cast<TransposeNode>(RN2->getInput());
    if (!TR) {
      continue;
    }

    auto *RN1 = dyn_cast<ReshapeNode>(TR->getInput());
    if (!RN1) {
      continue;
    }

    // Check that the input and output shapes match:
    if (RN1->getInput().getType() != RN2->getResult().getType()) {
      continue;
    }

    // Compute the original parameters to ChannelShuffle.
    auto paramsM = getChannelShuffleParams(*RN1);
    if (!paramsM.hasValue()) {
      continue;
    }

    // Create a new ChannelShuffle with kernel parameter tranposed by the
    // TR's shuffle.
    auto *newCS = F->createChannelShuffle("channel_shuffle", RN1->getInput(),
                                          paramsM->group, paramsM->kernel);
    RN2->getResult().replaceAllUsesOfWith(newCS);
    changed = true;
  }
  return changed;
}

// Fold Tile -> Add into BatchedAdd wherever applicable.
bool FoldTileAddIntoBatchedAdd::run(Function *F,
                                    const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());

  bool changed = false;
  for (const auto &node : F->getNodes()) {
    const auto *addNode = dyn_cast<AddNode>(&node);
    if (!addNode) {
      continue;
    }

    NodeValue batchNode, addedNode;
    const auto &LHS = addNode->getLHS();
    const auto &RHS = addNode->getRHS();
    const TileNode *tileNode = nullptr;

    // Check if LHS is a tile.
    if ((tileNode = dyn_cast<TileNode>(LHS))) {
      batchNode = RHS;
      addedNode = tileNode->getInput();
    }
    // Check if RHS is a tile.
    else if ((tileNode = dyn_cast<TileNode>(RHS))) {
      batchNode = LHS;
      addedNode = tileNode->getInput();
    }
    // If neither LHS or RHS is a tile, nothing to do.
    else {
      continue;
    }

    // If the tiling of the added node is not along the 0th axis,
    // 'Add' cannot be replaced with 'BatchedAdd'.
    if (tileNode->getAxis() != 0) {
      continue;
    }

    auto oldDims = addedNode.dims();
    // If the 0th dimension of the added node is not 1,
    // then reducing dimension via reshaping is more complicated.
    // Hence, Add will not be replaced with BatchedAdd.
    if (oldDims.size() == 0 || oldDims[0] != 1) {
      continue;
    }

    // Reshape the added node to create a slice for the batched add
    // such that its dim size is one less than that of the batch.
    const auto newDims = oldDims.take_back(oldDims.size() - 1);
    auto *slice = F->createReshape(tileNode->getName().str() + "_reshape",
                                   addedNode, newDims);

    // Create a new batched add node to replace existing add node.
    auto *newBA = F->createBatchedAdd(addNode->getName().str() + "_batched_add",
                                      batchNode, slice);
    addNode->getResult().replaceAllUsesOfWith(newBA);
    changed = true;
  }
  return changed;
}

/// Fold ElemKind conversion nodes (ConvertTo, Quantize) into
/// single-user Placeholders. Note that this changes the semantics
/// of the IO of the Function and so must be done carefully, i.e. should always
/// be opt-in and done alongside conversion of corresponding Tensors in
/// PlaceholderBindings. If
/// cctx.optimizationOpts.foldStaticPlaceholderConversions is set this will
/// only change Placeholders marked as static.
bool FoldElemKindConversionIntoInputs::run(Function *F,
                                           const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());

  bool changed = false;
  auto &nodes = F->getNodes();

  for (auto it = nodes.begin(), e = nodes.end(); it != e; it++) {
    Node *N = &*it;
    // Handle conversion of inputs (conversion of Placeholders):
    ConvertToNode *CTN = llvm::dyn_cast<ConvertToNode>(N);
    QuantizeNode *QN = llvm::dyn_cast<QuantizeNode>(N);
    if (CTN || QN) {
      NodeValue in = CTN ? CTN->getInput() : QN->getInput();
      Placeholder *P = llvm::dyn_cast<Placeholder>(in);
      if (!P || P->getUsers().size() != 1) {
        continue;
      }
      // If foldElemKindConversionIntoIO is not set and this is not a static
      // placeholder then skip.
      if (!cctx.optimizationOpts.foldElemKindConversionIntoIO &&
          !P->isStatic()) {
        continue;
      }

      // We have a conversion of a single-use placeholder to some other type, so
      // it is safe to do the requested conversion.
      NodeValue res = CTN ? CTN->getResult() : QN->getResult();

      // Convert the type of the Placeholder to the conversion type. If target
      // type is fused call setTypeUnsafe because the shape can change in this
      // case.
      if (isFusedQuantizedElemKind(res.getElementType())) {
        P->setTypeUnsafe(Storage::OutputIdx, res.getType());
      } else {
        P->setType(Storage::OutputIdx, res.getType());
      }

      // Replace all uses of the original ConvertTo to the Placeholder.
      res.replaceAllUsesOfWith(P);

      changed = true;
      continue;
    }
  }
  return changed;
}

/// Fold ElemKind conversion nodes (ConvertTo, Dequantize) into SaveNodes. Note
/// that this changes the semantics of the IO of the Function and so must be
/// done carefully, i.e. should always be opt-in and done alongside conversion
/// of corresponding Tensors in PlaceholderBindings.
bool FoldElemKindConversionIntoOutputs::run(Function *F,
                                            const CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), getName());

  std::unordered_set<SaveNode *> deadSaves;

  bool changed = false;
  // Since we will be adding in new SaveNodes, reverse iterate to be safe.
  auto &nodes = F->getNodes();
  for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
    Node *N = &*it;

    // Handle conversion of outputs (SaveNodes + Placeholders):
    if (SaveNode *SN = llvm::dyn_cast<SaveNode>(N)) {
      if (!SN) {
        continue;
      }
      if (SN->getPlaceholder()->getUsers().size() != 1) {
        continue;
      }
      ConvertToNode *CTN = llvm::dyn_cast<ConvertToNode>(SN->getInput());
      DequantizeNode *DQN = llvm::dyn_cast<DequantizeNode>(SN->getInput());
      if (!CTN && !DQN) {
        continue;
      }
      NodeValue in = CTN ? CTN->getInput() : DQN->getInput();

      // Set the type of the Placeholder to be same the conversion's input.
      SN->getPlaceholder()->setType(Storage::OutputIdx, in.getType());

      // Create a new SaveNode directly using the conversion's input.
      F->createSave(SN->getName(), in, SN->getPlaceholder());

      // Queue up deleting the original SaveNode as it won't be deleted via DCE.
      deadSaves.insert(SN);
      changed = true;
      continue;
    }
  }

  // Delete all the dead saves.
  for (SaveNode *SN : deadSaves) {
    F->eraseNode(SN);
  }

  return changed;
}

/// Looks for an activation directly following \p N from \p F that the backend
/// \p B supports for fusion.
template <class T> bool fuseActivation(T *N, Function *F, const Backend *B) {
  if (!N || N->hasFusedActivation() || !N->getResult().hasOneUse()) {
    return false;
  }

  // We know there is one result user so we can just deref the first result.
  Node *activation = (*N->getResult().getUsers().begin()).getUser();
  if (!B || !B->supportsFusedActivation(N, activation)) {
    return false;
  }

  FusedActivation activationType;
  NodeValue activationNV;
  switch (activation->getKind()) {
  case Kinded::Kind::ReluNodeKind:
    activationType = FusedActivation::RELU;
    activationNV = cast<ReluNode>(activation)->getResult();
    break;
  case Kinded::Kind::SigmoidNodeKind:
    activationType = FusedActivation::SIGMOID;
    activationNV = cast<SigmoidNode>(activation)->getResult();
    break;
  case Kinded::Kind::TanhNodeKind:
    activationType = FusedActivation::TANH;
    activationNV = cast<TanhNode>(activation)->getResult();
    break;
  default:
    return false;
  }

  N->setFusedActivation(activationType);
  activationNV.replaceAllUsesOfWith(N->getResult());
  return true;
}

static bool foldActivations(Function *F, CompilationContext &cctx,
                            const Backend *B) {
  bool changed = false;
  for (auto &node : F->getNodes()) {
    if (fuseActivation(dyn_cast<ConvolutionNode>(&node), F, B)) {
      changed = true;
      continue;
    }
  }
  return changed;
}

void glow::fold(Function *F, CompilationContext &cctx, const Backend *B) {
  LOG_SCOPE(F->getLogContext(), "glow::fold")

  FunctionPassManager FPM("FoldFPM", createDefaultFoldPassPipeline());
  FPM.run(F, cctx);

  foldActivations(F, cctx, B);
}

void glow::optimize(Function *F, CompilationContext &cctx, const Backend &B) {
  LOG_SCOPE(F->getLogContext(), "glow::optimize")

  FunctionPassManager FPM("TargetDependentGraphOptzFPM",
                          B.getOptimizationPipeline(), &B);
  FPM.run(F, cctx);
}

void glow::optimize(Function *F, CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), "glow::optimize")

  // Indicates if the given function is completely loaded. A temporary
  // workaround until #3213 is complete.
  F->setState(FunctionState::FuncLoaded);

  FunctionPassManager FPM("TargetIndependentGraphOptzFPM",
                          createDefaultGraphOptimizationPassPipeline());
  FPM.run(F, cctx);
}

void glow::optimize(Function *F, CompilationMode mode) {
  CompilationContext cctx;
  cctx.compMode = mode;
  optimize(F, cctx);
}

/// Helper to pass over all Nodes in \p F and set FP16 accumulation to true for
/// those Nodes in the SLS family which support and need it. \p precConfig
/// contains the black/whitelist for skipping.
static void setFP16AccumSLS(Function *F,
                            const PrecisionConfiguration &precConfig) {
  // Iterate from original end to beginning to avoid processing new Nodes added
  // during the pass.
  auto nodeIt = F->getNodes().end();
  auto stopIt = F->getNodes().begin();
  do {
    --nodeIt;
    Node &node = *nodeIt;
    // Only update allowed nodes based on black/whitelist.
    const bool inSet = precConfig.precisionModeKindSet.count(node.getKind());
    const bool allowConversion = precConfig.useSetAsWhitelist ? inSet : !inSet;
    if (!allowConversion) {
      continue;
    }

#define CASE_SET_SLS_FP16_ACCUM(NODE_)                                         \
  case Kinded::Kind::NODE_##NodeKind: {                                        \
    NODE_##Node *SLS = llvm::cast<NODE_##Node>(&node);                         \
    if (SLS->getResult().getElementType() != ElemKind::Float16Ty) {            \
      continue;                                                                \
    }                                                                          \
    SLS->setUseFP16Accumulation(true);                                         \
    continue;                                                                  \
  }

    switch (node.getKind()) {
      CASE_SET_SLS_FP16_ACCUM(RowwiseQuantizedSparseLengthsWeightedSum);
      CASE_SET_SLS_FP16_ACCUM(FusedRowwiseQuantizedSparseLengthsWeightedSum);
      CASE_SET_SLS_FP16_ACCUM(FusedRowwiseQuantizedSparseLengthsSum);
      CASE_SET_SLS_FP16_ACCUM(EmbeddingBagByteRowwiseOffsets);
    default:
      continue;
    }
  } while (nodeIt != stopIt);
}

void glow::transformForPrecisionMode(const Backend &B, Function *F,
                                     CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), "transformForPrecisionMode")
  const PrecisionConfiguration &precConfig = cctx.precisionConfig;

  switch (precConfig.quantMode) {
  case QuantizationMode::Profile: {
    assert(cctx.bindings);

    LOG_SCOPE(F->getLogContext(), "glow::profileQuantization")

    glow::profileQuantization(*cctx.bindings, F);
    break;
  }

  case QuantizationMode::Quantize: {
    LOG_SCOPE(F->getLogContext(), "quantization::quantizeFunction")

    quantization::quantizeFunction(F, precConfig.quantConfig, B,
                                   *cctx.loweredInfoMap,
                                   precConfig.precisionModeKindSet);
    break;
  }

  case QuantizationMode::None: {
    break;
  }
  }

  if (precConfig.convertToFP16) {
    LOG_SCOPE(F->getLogContext(), "glow::convertFunctionToFloat16")
    convertFunctionToFloat16(F, precConfig);
    FunctionPassManager FPM("FP16GraphOptzFPM",
                            createFP16GraphOptimizationPassPipeline());
    FPM.run(F, cctx);
  }

  // By default, FP16 SLS accumulation is not enabled.
  // If requested, Force all ops in the SLS family to use FP16 accumulation.
  if (precConfig.forceFP16AccumSLS) {
    setFP16AccumSLS(F, precConfig);
  }
}

Error glow::optimizeFunctionBeforeLowering(Function *F,
                                           CompilationContext &cctx) {
  LOG_SCOPE(F->getLogContext(), "glow::optimizeFunctionBeforeLowering")

  // If we only want to lower the Function, do nothing here.
  if (cctx.optimizationOpts.onlyLowerFuns.count(F)) {
    return Error::success();
  }

  // Verify the function pre-optimization/lowering.
  assert(F->verify() && "Function must be valid");

  // Verify that the CompilationContext is set up correctly.
  RETURN_IF_ERR(cctx.verify());

  // Fold low-level operators into higher-level operators.
  // This is useful when compiling an input model where some high-level
  // operators have been lowered (this can be for instance a side effect of
  // model converters, like converters from Tensorflow to ONNX). In this
  // situation, such folding can then enable more optimizations and also improve
  // the performance backends that support natively such high-level operators.
  ::glow::fold(F, cctx);

  // Optimize the graph. Only runs optimizations that are target-independent.
  ::glow::optimize(F, cctx);
  return Error::success();
}

// NOTE: When updating this function, please also update the documentation in
// docs/GraphOptimizationPipeline.md
Error glow::optimizeFunction(Function *F, const Backend &B,
                             CompilationContext &cctx,
                             const glow::runtime::DeviceInfo *devInfo) {
  LOG_SCOPE(F->getLogContext(), "glow::optimizeFunction")

  // If requested only lower the Function and early return.
  if (cctx.optimizationOpts.onlyLowerFuns.count(F)) {
    ::glow::lower(F, cctx, &B);
    // Cleanup from lowering via DCE.
    runDCEPass(F, cctx);

    if (!B.verify(*F, cctx.verboseCompile)) {
      return MAKE_ERR(
          ErrorValue::ErrorCode::COMPILE_UNSUPPORTED_NODE_AFTER_OPTIMIZE,
          "Unsupported node(s) found after only-lowering path for Function " +
              F->getName().str() + " for backend " + B.getBackendName());
    }
    return Error::success();
  }

  RETURN_IF_ERR(optimizeFunctionBeforeLowering(F, cctx));
  // Lower the graph into a sequence of low-level linear algebra operations.
  const PrecisionConfiguration &precConfig = cctx.precisionConfig;
  if (precConfig.quantMode == QuantizationMode::Profile) {
    // When profiling, pass a nullptr for the backend, signaling that all nodes
    // should be lowered. loweredInfoMap logs what is lowered from what for
    // later use when creating quantization infos. Also pass the precision mode
    // kind set as nodes to not lower, specified higher up in the stack.
    ::glow::lower(F, cctx, /* backend */ nullptr,
                  precConfig.precisionModeKindSet);
  } else {
    // Lower based on the backend's preferences.
    ::glow::lower(F, cctx, &B);
  }

  // Transform given precision mode; may quantize, convert to fp16, or
  // instrument with profiling nodes. This must be done after lowering.
  transformForPrecisionMode(B, F, cctx);

  // Lower once more, in case precision transform has introduced operators that
  // need to be lowered, e.g., Clip.
  ::glow::lower(F, cctx, &B);

  // Optimize the graph again now that we have a lowered representation.
  ::glow::optimize(F, cctx);

  // If requested fold ElemKind conversion Nodes into static Placeholders,
  // inputs, and outputs (Placeholders and SaveNodes).
  if (cctx.optimizationOpts.foldStaticPlaceholderConversions ||
      cctx.optimizationOpts.foldElemKindConversionIntoIO) {
    FunctionPassPipeline pipeline{
        FunctionPassID::FoldElemKindConversionIntoInputs};

    if (cctx.optimizationOpts.foldElemKindConversionIntoIO) {
      pipeline.pushBack({FunctionPassID::FoldElemKindConversionIntoOutputs});
    }
    FunctionPassManager FPM("FoldElemKindConversionIntoIO", pipeline);
    if (FPM.run(F, cctx)) {
      ::glow::optimize(F, cctx);
    }
  }

  // Allow the backend to transform the graph after lowering.
  if (B.transformPostLowering(F, cctx, devInfo)) {
    // If the backend made changes, optimize the graph again. Perform only
    // passes that the Backend has requested so we do not interfere with the
    // backend's transformations.
    ::glow::optimize(F, cctx, B);
  }

  // We already started using backend specific verification when the function
  // state became lowered. Do one more verification pass to make sure everything
  // is in order and to bail if it is not.
  if (!B.verify(*F, cctx.verboseCompile)) {
    return MAKE_ERR(
        ErrorValue::ErrorCode::COMPILE_UNSUPPORTED_NODE_AFTER_OPTIMIZE,
        "Unsupported node(s) found after optimizing Function " +
            F->getName().str() + " for backend " + B.getBackendName());
  }
  return Error::success();
}

bool glow::executeVerticalFCWeightsSplit(Function *F, unsigned numOfChunks,
                                         unsigned minKToSplit) {
  DCHECK(numOfChunks > 0) << "numOfChunks must be a positive number, given: "
                          << numOfChunks;
  DCHECK(minKToSplit > 0) << "minKToSplit must be a positive number, given: "
                          << minKToSplit;

  bool changed = false;
  for (auto it = F->getNodes().begin(), e = F->getNodes().end(); it != e;
       ++it) {
    auto *FC = dyn_cast<FullyConnectedNode>(it);
    if (!FC) {
      continue;
    }

    size_t K = FC->getWeights().dims()[1];
    if (K < minKToSplit) {
      continue;
    }

    auto input = FC->getInput();
    auto weights = FC->getWeights();
    auto bias = FC->getBias();

    dim_t elemPerChunk = (bias.dims()[0] + numOfChunks - 1) / numOfChunks;
    dim_t sliceStart = 0;
    std::vector<NodeValue> fcs(numOfChunks);

    // Split weights across second dimension into numOfChunks pieces.
    // Input dimension is [M;K] and kept untouched.
    // Bias dimension is [N], split into chunks.
    // Weight dimension is [K;N], split into numOfChunks chunks,
    // [K;N/numOfChunks] each.
    // Last chunk might require special handling in case
    // N is not divisible by numOfChunks.
    auto *fcType = F->getParent()->uniqueTypeWithNewShape(
        FC->getResult().getType(), {FC->getResult().dims()[0], elemPerChunk});

    for (unsigned i = 0; i < numOfChunks; ++i) {
      // Last chunk might need special handling if bias dimension
      // is not divisible by numOfChunks.
      if (i == numOfChunks - 1 && bias.dims()[0] % numOfChunks != 0) {
        elemPerChunk = bias.dims()[0] - (numOfChunks - 1) * elemPerChunk;
        fcType = F->getParent()->uniqueTypeWithNewShape(
            FC->getResult().getType(),
            {FC->getResult().dims()[0], elemPerChunk});
      }

      auto *weightSlice = F->createSlice(
          "weight_slice." + weights.getNode()->getName().str(), weights,
          {0, sliceStart}, {weights.dims()[0], sliceStart + elemPerChunk});
      auto *biasSlice =
          F->createSlice("bias_slice." + bias.getNode()->getName().str(), bias,
                         {sliceStart}, {sliceStart + elemPerChunk});
      fcs[i] = F->createFullyConnected("fc_slice." + FC->getName().str(), input,
                                       weightSlice->getResult(),
                                       biasSlice->getResult(), fcType);
      sliceStart += elemPerChunk;
    }

    auto *concat =
        F->createConcat("concat." + FC->getName().str(), fcs, /*dimension*/ 1);
    FC->getResult().replaceAllUsesOfWith(concat);
    changed = true;
  }

  return changed;
}

/// Helper to parallelize a node \p curNode from \p F into \p numOfChunksNode
/// Nodes by slicing its inputs, creating clones of it and changing the inputs
/// of the clones to the slices, and then concatenating all of the clones
/// together and replacing \p curNode with the concat. \p inputBatchIdx is the
/// input idx from \p curNode that will be split (there may be more than one
/// input to split, but their splitDim should all have the same size).
/// \p splitDim represents what dimension to split for each of the inputs to
/// \p curNode. \p resultDim is the dimension on which we are splitting and then
/// concatenating the results. \p resultIdx represents the result index from
/// \p curNode that is being split and later concatenated.
static void parallelizeAndReplaceNode(Function *F, Node *curNode,
                                      dim_t numOfChunksNode,
                                      dim_t inputBatchIdx, dim_t resultIdx,
                                      llvm::ArrayRef<int> splitDims,
                                      size_t resultDim) {
  const int inputIdx = splitDims[inputBatchIdx];
  CHECK_GE(inputIdx, 0) << "Input batch idx must be split";
  const dim_t batchSize = curNode->getNthInput(inputBatchIdx).dims()[inputIdx];
  const dim_t elemPerChunk = batchSize / numOfChunksNode;
  const dim_t remain = batchSize % numOfChunksNode;

  std::vector<NodeValue> newNodes(numOfChunksNode);
  for (dim_t i = 0; i < numOfChunksNode; ++i) {
    // Calculate the out type of this chunk.
    const dim_t sliceStart = i * elemPerChunk + std::min(i, remain);
    const dim_t sliceEnd = sliceStart + elemPerChunk + ((i < remain) ? 1 : 0);
    VLOG(1) << "\tChunk " << i << ": start: " << sliceStart
            << " end: " << sliceEnd << "\n";
    auto outDims = curNode->dims(resultIdx).vec();
    outDims[resultDim] = (sliceEnd - sliceStart);
    for (auto outDim : outDims) {
      VLOG(1) << "outDim: " << outDim << "\n";
    }

    // Clone the original Node, so that it keeps all of the inputs/members of
    // the original Node. Then modify the output type so that its new shape is
    // correct, and below change the inputs the sliced inputs.
    Node *clone = curNode->clone();
    clone->getNthResult(resultIdx).setTypeUnsafe(
        F->getParent()->uniqueTypeWithNewShape(curNode->getType(resultIdx),
                                               outDims));
    F->addNode(clone);

    // Loop over all of the inputs and slice those inputs that need to be
    // sliced, and set them on the clone.
    for (int j = 0, e = curNode->getNumInputs(); j < e; j++) {
      int dim = splitDims[j];
      if (dim == -1) {
        continue;
      }

      NodeValue currInput = curNode->getNthInput(j);
      auto sliceDimsStart = std::vector<dim_t>(currInput.dims().size(), 0);
      sliceDimsStart[dim] = sliceStart;
      auto sliceDimsEnd = currInput.dims().vec();
      sliceDimsEnd[dim] = sliceEnd;
      VLOG(1) << "start: ";
      for (auto sliceDimStart : sliceDimsStart) {
        VLOG(1) << sliceDimStart << "\n";
      }
      VLOG(1) << "end: ";
      for (auto sliceDimEnd : sliceDimsEnd) {
        VLOG(1) << sliceDimEnd << "\n";
      }
      VLOG(1) << "Input name: " << currInput.getNode()->getName().str() << "\n";

      auto *inputSlice =
          F->createSlice("dp_slice." + currInput.getNode()->getName().str() +
                             "." + std::to_string(i),
                         currInput, sliceDimsStart, sliceDimsEnd);
      clone->setNthInput(j, inputSlice);

      newNodes[i] = clone;
    }
  }

  // Now that we have split the node into many, concat all of the pieces back
  // together and replace the original by the concat.
  VLOG(1) << "Creating Concat";
  auto *concat = F->createConcat("concat." + curNode->getName().str(), newNodes,
                                 resultDim);
  curNode->getNthResult(resultIdx).replaceAllUsesOfWith(concat);
}

bool glow::parallelizeOps(
    Function *F, const llvm::DenseMap<Node *, size_t> &numOfChunksMap,
    const llvm::DenseMap<Node *, ParallelTransformKind> &parOpts,
    size_t numOfChunks) {
  // Since we will be transforming the original list of nodes, reverse iterate.
  auto &nodes = F->getNodes();
  size_t numProcessedNodes = 0;
  for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
    Node *curNode = &*it;
    auto numOfChunksIt = numOfChunksMap.find(curNode);
    if (numOfChunksIt != numOfChunksMap.end()) {
      numOfChunks = numOfChunksIt->second;
    }

    ParallelTransformKind parTransformMode = ParallelTransformKind::None;
    auto parOptsIt = parOpts.find(curNode);
    if (parOptsIt != parOpts.end()) {
      parTransformMode = parOptsIt->second;
      ++numProcessedNodes;
    }

    VLOG(1) << "Attempting to Parallelizing Node: " << curNode->getName().str()
            << "\n";

    // Use this vector to communicate what dims to split to
    // parallelizeAndReplaceNode(). -1 represents not splitting at all.
    llvm::SmallVector<int, 3> splitDims(curNode->getNumInputs(), -1);
    switch (parTransformMode) {
    case ParallelTransformKind::Data: {
      switch (curNode->getKind()) {
      case Kinded::Kind::FullyConnectedNodeKind: {
        splitDims[FullyConnectedNode::InputIdx] = 0;
        parallelizeAndReplaceNode(F, curNode, numOfChunks,
                                  FullyConnectedNode::InputIdx,
                                  FullyConnectedNode::ResultIdx, splitDims, 0);
        break;
      }
      case Kinded::Kind::AddNodeKind: {
        splitDims[AddNode::LHSIdx] = 0;
        splitDims[AddNode::RHSIdx] = 0;
        parallelizeAndReplaceNode(F, curNode, numOfChunks, AddNode::LHSIdx,
                                  AddNode::ResultIdx, splitDims, 0);
        break;
      }
      case Kinded::Kind::MulNodeKind: {
        splitDims[AddNode::LHSIdx] = 0;
        splitDims[AddNode::RHSIdx] = 0;
        parallelizeAndReplaceNode(F, curNode, numOfChunks, MulNode::LHSIdx,
                                  MulNode::ResultIdx, splitDims, 0);
        break;
      }
      case Kinded::Kind::SigmoidNodeKind: {
        splitDims[SigmoidNode::InputIdx] = 0;
        parallelizeAndReplaceNode(F, curNode, numOfChunks,
                                  SigmoidNode::InputIdx, SigmoidNode::ResultIdx,
                                  splitDims, 0);
        break;
      }
      case Kinded::Kind::TanhNodeKind: {
        splitDims[TanhNode::InputIdx] = 0;
        parallelizeAndReplaceNode(F, curNode, numOfChunks, TanhNode::InputIdx,
                                  TanhNode::ResultIdx, splitDims, 0);
        break;
      }
      case Kinded::Kind::TransposeNodeKind: {
        splitDims[TransposeNode::InputIdx] = 0;
        unsigned_t resultDim = cast<TransposeNode>(curNode)->getShuffle()[0];
        parallelizeAndReplaceNode(
            F, curNode, numOfChunks, TransposeNode::InputIdx,
            TransposeNode::ResultIdx, splitDims, resultDim);
        break;
      }
      case Kinded::Kind::ReluNodeKind: {
        splitDims[ReluNode::InputIdx] = 0;
        parallelizeAndReplaceNode(F, curNode, numOfChunks, ReluNode::InputIdx,
                                  ReluNode::ResultIdx, splitDims, 0);
        break;
      }
      case Kinded::Kind::ConvertToNodeKind: {
        splitDims[ConvertToNode::InputIdx] = 0;
        parallelizeAndReplaceNode(F, curNode, numOfChunks,
                                  ConvertToNode::InputIdx,
                                  ConvertToNode::ResultIdx, splitDims, 0);
        break;
      }
      default:
        VLOG(1) << "Attempted to parallelize op type " << curNode->getKindName()
                << "not yet supported"
                << "\n";
        break;
      }
      break;
    }

    case ParallelTransformKind::Model: {
      switch (curNode->getKind()) {
      case Kinded::Kind::FullyConnectedNodeKind: {
        splitDims[FullyConnectedNode::WeightsIdx] = 1;
        splitDims[FullyConnectedNode::BiasIdx] = 0;
        parallelizeAndReplaceNode(F, curNode, numOfChunks,
                                  FullyConnectedNode::WeightsIdx,
                                  FullyConnectedNode::ResultIdx, splitDims, 1);
        break;
      }
      case Kinded::Kind::ReluNodeKind: {
        if (curNode->getNthInput(ReluNode::InputIdx).dims().size() < 2) {
          break;
        }
        splitDims[ReluNode::InputIdx] = 1;
        parallelizeAndReplaceNode(F, curNode, numOfChunks, ReluNode::InputIdx,
                                  ReluNode::ResultIdx, splitDims, 1);
        break;
      }
      default:
        VLOG(1) << "Attempted to parallelize op type " << curNode->getKindName()
                << "not yet supported"
                << "\n";
        break;
      }
      break;
    }

    case ParallelTransformKind::None:
      break;
    }
  }

  // Because we transformed Node types unsafely, make sure all types of the
  // Function still are valid.
  bool ret = F->verify();
  DCHECK(ret) << "Verification issue post parallelization";

  const bool allProcessed = numProcessedNodes == parOpts.size();
  DCHECK(allProcessed) << "Not all Nodes specified in parOpts were processed.";

  return ret && allProcessed;
}
