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

#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Quantization/Base/Base.h"

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

/// Dead code elimination.
static void DCE(Function *F) {
  auto &nodes = F->getNodes();
  auto &consts = F->getParent()->getConstants();

  std::vector<ConstList::iterator> erasedConsts{};
  std::vector<NodesList::iterator> erasedNodes{};

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

/// \returns true if the masks \p shuffle1 and shuffle2 are
/// the inverse of on another. Applying both masks should result in the identity
/// shuffle.
static bool isIdentityShuffle(llvm::ArrayRef<unsigned_t> shuffle1,
                              llvm::ArrayRef<unsigned_t> shuffle2) {

  if (shuffle1.size() != shuffle2.size()) {
    return false;
  }

  // Check if the combined masks are the identity mask.
  for (unsigned i = 0, e = shuffle1.size(); i < e; i++) {
    unsigned_t idx = shuffle2[shuffle1[i]];
    if (idx != i) {
      return false;
    }
  }
  return true;
}

/// \returns True if the node \p N always evaluates to \p val.
bool isSplatOfVal(Node *N, float val) {
  SplatNode *Z = dyn_cast<SplatNode>(N);
  if (!Z)
    return false;

  return (Z->getValue() == val);
}

/// \returns True if the node returns a constant value.
bool isConstant(Node *N) { return isa<SplatNode>(N); }

/// \returns the new simplified node or the original node.
static Node *simplifyNode(Node *node, Function *F) {
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

  llvm::ArrayRef<size_t> inputDims = node.getInput().dims();
  llvm::ArrayRef<size_t> resultDims = node.getDims();

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

/// Sink Transpose below ChannelShuffle node sequence ending with \p
/// postShuffleRN. For example (Transpose_1->Reshape_1->Transpose_2->Reshape_2)
/// becomes (Reshape_1->Transpose_2->Reshape_2->Transpose_1). \returns true if
/// tranpose was sunk below ChannelShuffle node sequence and false otherwise.
static bool sinkTranposeBelowChannelShuffle(Function *F,
                                            ReshapeNode *postShuffleRN) {
  auto *shuffleTR = dyn_cast<TransposeNode>(postShuffleRN->getInput());
  if (!shuffleTR) {
    return false;
  }

  auto *preShuffleRN = dyn_cast<ReshapeNode>(shuffleTR->getInput());
  if (!preShuffleRN) {
    return false;
  }

  auto *sinkingTR = dyn_cast<TransposeNode>(preShuffleRN->getInput());
  if (!sinkingTR) {
    return false;
  }

  // Compute the original parameters to ChannelShuffle.
  auto paramsM = getChannelShuffleParams(*preShuffleRN);

  if (!paramsM.hasValue()) {
    return false;
  }

  // Create a new ChannelShuffle with kernel parameter tranposed by the
  // sinkingTR's shuffle because that Transpose will now be moved below this
  // ChannelShuffle operator.
  auto *newChannelShuffle = F->createChannelShuffle(
      "channel_shuffle", sinkingTR->getInput(), paramsM->group,
      sinkingTR->getShuffle()[paramsM->kernel]);

  // Create a copy of sinkingTR and insert after newChannelShuffle.
  auto *newSinkingTR = F->createTranspose(
      sinkingTR->getName(), newChannelShuffle, sinkingTR->getShuffle());

  postShuffleRN->getResult().replaceAllUsesOfWith(newSinkingTR);

  return true;
}

/// Code Sinking.
/// \returns true if code sinking was successful.
static bool sinkCode(Function *F) {
  auto &nodes = F->getNodes();
  bool changed = false;
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
      auto *newTR = F->createTranspose(TR->getName(), NewBN, TR->getShuffle());
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
      auto *newTR = F->createTranspose(TR->getName(), NRL, TR->getShuffle());
      newTR->setPredicate(node->getPredicate());
      RL->getResult().replaceAllUsesOfWith(newTR);
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
      auto *newTR = F->createTranspose(TR->getName(), NSI, TR->getShuffle());
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
      std::vector<size_t> newOutPadShape(numDims);
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
      auto *newTR = F->createTranspose(TR->getName(), NTN, TR->getShuffle());
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

      // The two transposes are reversing one another. We can skip both of
      // them alltogether.
      if (isIdentityShuffle(mask1, mask2)) {
        TR1->getResult().replaceAllUsesOfWith(TR2->getInput());
        changed = true;
        continue;
      }
    }

    if (auto *RN = dyn_cast<ReshapeNode>(node)) {
      // Sink Transpose below ChannelShuffle.
      if (sinkTranposeBelowChannelShuffle(F, RN)) {
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
        // transpose (splat').
        if (isa<SplatNode>(node->getNthInput(ArithmeticNode::LHSIdx)) && RTR) {
          // Build splat' for LHS.
          auto *SN =
              dyn_cast<SplatNode>(node->getNthInput(ArithmeticNode::LHSIdx));
          auto *NS = F->createSplat("splat", RTR->getInput().getType(),
                                    SN->getValue());
          LTR = F->createTranspose("transpose", NS, RTR->getShuffle());
          changed = true;
        } else if (isa<SplatNode>(node->getNthInput(ArithmeticNode::RHSIdx)) &&
                   LTR) {
          // Build splat' for RHS.
          auto *SN =
              dyn_cast<SplatNode>(node->getNthInput(ArithmeticNode::RHSIdx));
          auto *NS = F->createSplat("splat", LTR->getInput().getType(),
                                    SN->getValue());
          RTR = F->createTranspose("transpose", NS, LTR->getShuffle());
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
      auto *newTR =
          F->createTranspose(LTR->getName(), newAN, LTR->getShuffle());
      newTR->setPredicate(node->getPredicate());
      node->getNthResult(ArithmeticNode::ResultIdx).replaceAllUsesOfWith(newTR);
    }

    // Sink RELU below batch concat nodes.
    if (auto *CN = dyn_cast<ConcatNode>(node)) {
      if (CN->getInputs().size() != 2) {
        continue;
      }
      auto LInput = CN->getInputs()[0];
      auto RInput = CN->getInputs()[1];
      auto *L = dyn_cast<ReluNode>(LInput);
      auto *R = dyn_cast<ReluNode>(RInput);

      if (L && R) {
        auto *newCN = F->createConcat(
            CN->getName(), {L->getInput(), R->getInput()}, CN->getDim());
        newCN->setPredicate(node->getPredicate());
        auto *newRL =
            F->createRELU(L->getName(), newCN, CN->getResult().getType());
        newRL->setPredicate(node->getPredicate());
        CN->getResult().replaceAllUsesOfWith(newRL);
      }
    }

    // Sink Transpose below concat nodes.
    if (auto *CN = dyn_cast<ConcatNode>(node)) {
      if (CN->getInputs().size() != 2) {
        continue;
      }
      auto LInput = CN->getInputs()[0];
      auto RInput = CN->getInputs()[1];
      auto *L = dyn_cast<TransposeNode>(LInput);
      auto *R = dyn_cast<TransposeNode>(RInput);

      // Both sides must be a transpose instruction.
      if (!L || !R) {
        continue;
      }

      // If the shuffle masks don't agree then bail out.
      if (L->getShuffle() != R->getShuffle()) {
        continue;
      }

      // Figure out where we transposed the channel index for batch
      // normalization.
      unsigned_t idx = CN->getDim();
      unsigned_t newChannelIdx = L->getShuffle()[idx];

      auto *newCN = F->createConcat(
          CN->getName(), {L->getInput(), R->getInput()}, newChannelIdx);
      newCN->setPredicate(node->getPredicate());
      auto *newTR = F->createTranspose(L->getName(), newCN, L->getShuffle());
      newTR->setPredicate(node->getPredicate());
      CN->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
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
  for (auto ll : list) {
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
static void mergeMatMul(Function *F) {
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

    size_t R = MM->getResult().dims()[1];
    size_t start = 0;
    for (auto *origMM : MMs) {
      size_t H = origMM->getResult().dims()[0];
      auto *ex = F->createSlice("extract", MM, {start, 0}, {start + H, R});
      start += H;
      origMM->getResult().replaceAllUsesOfWith(ex);
    }
  }
}

static bool mergePadIntoConvolution(Function *F) {
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
                                CN->getGroup());
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
static bool mergeTransposeIntoMatMulOrFC(Function *F) {
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
    auto *newW = F->getParent()->createConstant(W->getType(), W->getName());
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
      // Ignore slices of invalid types.
      if (lastSlice->getResult().getType() != SN->getResult().getType()) {
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
    if (startCoor[i] + resDim[i] != inDim[i])
      return false;
  }

  // Report success if we found at least two slices that extract from the
  // input.
  return order.size() > 1;
}

/// Merge multiple batched add nodes into a large batched-add node.
static void mergeBatchedAdd(Function *F) {
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
    SliceNode *S = llvm::cast<SliceNode>(order[0]);
    auto *BA = F->createBatchedAdd("mergedBA", S->getInput(), it.first);

    // Create the new slices. These slices will replace the original scalar
    // batched-add nodes.
    for (auto *orig : order) {
      newSlices.push_back(F->createSlice(orig->getName(), BA, orig->getStart(),
                                         orig->getResult().getType()));
    }

    // Replace the original individual batched adds with corresponding slices
    // from the new merged batch add.
    for (auto *BA : BAs) {
      for (int i = 0, e = order.size(); i < e; i++) {
        if (BA->getBatch().getNode() == order[i]) {
          BA->getResult().replaceAllUsesOfWith(newSlices[i]);
          break;
        }
      }
    }

  } // for each batched-add group.
}

/// Pool optimization.
static void optimizePool(Function *F) {
  auto &nodes = F->getNodes();

  // For each node:
  for (auto &node : nodes) {
    // Swap the order of Relu->MaxPool, to perform the RELU operation on a
    // smaller tensor. This optimization is not a major performance win. The
    // RELU operation takes a small fraction of the time, and reordering the
    // nodes does not give us much. However, reordering the buffers allows us
    // to reuse the memory buffer of the pool operation and potentially save
    // memory.
    if (auto *PL = dyn_cast<MaxPoolNode>(&node)) {
      auto *RL = dyn_cast<ReluNode>(PL->getInput());

      if (!RL) {
        continue;
      }

      // We don't want to increase the number of operations in the program, so
      // perform this transformation if the relu has a single user, which is
      // the pooling operation.
      if (!RL->hasOneUse()) {
        continue;
      }

      auto *NPL =
          F->createMaxPool(PL->getName(), RL->getInput(), PL->getKernels(),
                           PL->getStrides(), PL->getPads());
      auto reluOutTy = F->getParent()->uniqueTypeWithNewShape(
          RL->getResult().getType(), NPL->getResult().dims());
      auto *NRL = F->createRELU(RL->getName(), NPL, reluOutTy);
      PL->getResult().replaceAllUsesOfWith(NRL);
      continue;
    }
  } // For all nodes in the graph.
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
  auto *NC = M->createConstant(constant->getType(), constant->getName());
  NC->getPayload().assign(&constant->getPayload());
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
    CV.setNthInput(ConvolutionNode::FilterIdx, filterC);
  }
  if (cbiasC != CV.getBias().getNode()) {
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
    size_t channelId = filterH.getDimForPtr(0, i);
    float value = varH.at({channelId});
    float stdvar = 1.0f / std::sqrt(value + epsilon);
    float gamma = scaleH.at({channelId});
    float A = gamma * stdvar;
    filterH.raw(i) = ElemTy(float(filterH.raw(i)) * A);
  }

  for (size_t i = 0, e = cbiasH.size(); i < e; i++) {
    // Dimension zero is the 'channel' dimension. If we ever change the
    // layout of the filter then we need to change this optimization.
    size_t channelId = cbiasH.getDimForPtr(0, i);
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

static void optimizeBatchNorm(Function *F) {
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
    }
  } // For all nodes in the graph.
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
    llvm::ArrayRef<size_t> firstDims, size_t leadingDimsProdOriginalConcatNode,
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
  size_t trailingDimsProdOriginalConcatNode = 1;
  size_t leadingDimsProdOriginalConcatNode = 1;
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
  return F->createReshape(CN->getInputs().front().getNode()->getName(), newCN,
                          CN->getResult().dims());
}

/// Simplify concat node.
/// \returns a new simplified Concat node or nullptr.
static NodeValue simplifyConcatNode(Function *F, ConcatNode *CN) {
  /// concat(dim1, concat(dim2, X, Y), Z) -> concat(dim1, X, Y, Z),
  /// but only if dim1 == dim2
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
      if (!CNI || CNI->getDim() != CN->getDim())
        continue;

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

/// Optimize Concat nodes.
static void optimizeConcatNodes(Function *F) {
  auto &nodes = F->getNodes();

  // For each node:
  for (auto &node : nodes) {
    if (auto *CN = dyn_cast<ConcatNode>(&node)) {
      NodeValue newCN = simplifyConcatNode(F, CN);
      if (newCN.getNode()) {
        CN->getResult().replaceAllUsesOfWith(newCN);
      }
    }
  }
}

/// Simplify and canonicalize arithmetic nodes by detecting simple arithmetic
/// identities.
static void optimizeArithmeticNodes(Function *F) {
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

    auto *SN = simplifyNode(N, F);
    if (SN != N) {
      N->getNthResult(ArithmeticNode::ResultIdx).replaceAllUsesOfWith(SN);

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
}

/// Statically transpose Constants.
static void transposeConstants(Function *F) {
  auto &nodes = F->getNodes();

  for (auto &node : nodes) {
    auto *TN = dyn_cast<TransposeNode>(&node);
    if (!TN) {
      continue;
    }
    auto *C = dyn_cast<Constant>(TN->getInput());
    // C must have a single use.
    if (!C || !C->hasOneUse()) {
      continue;
    }
    // Create a new Constant NC to hold the transposed result.
    auto *NC =
        F->getParent()->createConstant(TN->getResult().getType(), C->getName());
    // Transpose the value of C into NC.
    genericTranspose(&C->getPayload(), &NC->getPayload(), TN->getShuffle());
    // Rewrite uses of TN to reference NC.
    TN->getResult().replaceAllUsesOfWith(NC);
  }
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
    constexpr size_t maxNumEls = 8;
    size_t numEls = std::min(T.getType().size(), maxNumEls);
    size_t bufSize = T.getType().getElementSize() * numEls;
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
    // Only dedup Constants if their data matches exactly, so allowed error is
    // 0.0.
    return lhs->getPayload().isEqual(rhs->getPayload(),
                                     /* allowedError */ 0.0);
  }
};

} // namespace

/// Deduplicates Constants in the Module \p M. Applicable Constants for
/// deduplication must have the same data.
static void deduplicateConstants(Module *M) {
  // Map from Constants to other Constants that are equivalent for purposes of
  // deduplication.
  std::unordered_map<Constant *, Constant *, ConstsHasherDedup, ConstsEqDedup>
      duplicateConstants;

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
  }
}

/// Common Subexpression Elimination.
static void CSE(Function *F) {
  CSEVisitor visitor;

  deduplicateConstants(F->getParent());

  // Perform CSE on all nodes.
  for (auto &N : F->getNodes()) {
    N.visit(nullptr, &visitor);
  }
}

/// Eliminate SliceNode when the input is SplatNode.
/// Slice(Splat(args)) -> Splat(args')
static void optimizeSliceOfSplat(Function *F) {
  for (auto &node : F->getNodes()) {
    auto *sliceNode = dyn_cast<SliceNode>(&node);
    if (!sliceNode)
      continue;
    auto *splatNode = dyn_cast<SplatNode>(sliceNode->getInput());
    if (!splatNode)
      continue;
    auto *newSplatNode =
        F->createSplat(sliceNode->getName(), sliceNode->getResult().getType(),
                       splatNode->getValue());
    sliceNode->getResult().replaceAllUsesOfWith(newSplatNode);
  }
}

/// Optimize reshape nodes.
static void optimizeReshape(Function *F) {
  for (auto &node : F->getNodes()) {
    auto *reshapeNode = dyn_cast<ReshapeNode>(&node);
    if (!reshapeNode)
      continue;
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
      continue;
    }
    // Reshape(Reshape(x)) -> Reshape(x).
    auto *reshapeNodeInput = dyn_cast<ReshapeNode>(inputNode);
    if (reshapeNodeInput && reshapeNodeInput->hasOneUse()) {
      auto *newReshape =
          F->createReshape(reshapeNode->getName(), reshapeNodeInput->getInput(),
                           reshapeNode->getResult().dims());
      reshapeNode->getResult().replaceAllUsesOfWith(newReshape);
      continue;
    }
    // Reshape(Constant) -> Constant'.
    // Only do this if the Constant has a single use, as otherwise we would
    // duplicate the Constant and increase the memory footprint.
    auto *C = dyn_cast<Constant>(inputNode);
    if (C && C->hasOneUse()) {
      // Create a new Constant with the type of the reshape.
      auto *newC = F->getParent()->createConstant(
          reshapeNode->getResult().getType(), C->getName());
      // Create an unowned view of the original tensor with the correct shape,
      // and assign it to the new Constant.
      Tensor reshapedT = C->getPayload().getUnowned(reshapeNode->getDims());
      newC->assign(&reshapedT);
      reshapeNode->getResult().replaceAllUsesOfWith(newC);
      continue;
    }
  }
}

/// Optimize: Max(Splat(), otherInput) or Max(otherInput, Splat()) for
/// quantized operations.
/// Splat and Max can be eliminated if Splat value cannot impact the result.
/// For example, Max and Splat can be removed if splat value is smaller
/// than quantization range [min, max].
static void optimizeQuantizedMaxSplat(Function *F) {
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
      // quantition [min,max] range then just remove MaxNode operation.
      float splatValue = (dyn_cast<SplatNode>(splatNode))->getValue();
      float min = MN->getResult().getType()->getQuantizedValueRange().first;
      if (splatValue <= min) {
        MN->getResult().replaceAllUsesOfWith(otherInput);
      }
    }
  }
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
      constantToBeModified.getPayload().convertToType(dstTy->getElementType());
      return constantToBeModified.getOutput();
    }
    case ElemKind::Int32QTy:
    case ElemKind::Int16QTy:
    case ElemKind::Int8QTy: {
      // Quantization: {FloatTy, Float16Ty} -> Quantized type.
      Constant &constantToBeModified = modifyConstantTyAndGet();
      TensorQuantizationParams params{dstTy->getScale(), dstTy->getOffset()};
      Tensor &tensorToBeModified = constantToBeModified.getPayload();
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

/// Optimize away ConvertToNode.
/// This basically turns "conversion(conversion A to B) to C"
/// into noop if the type of A and C are the same.
///
/// This method potentially changes the semantic of the program
/// because it eliminates some precision loss steps.
/// However, this actually improves accuracy so we can always do it.
static void optimizeConversions(Function *F) {

  llvm::SmallVector<Node *, 8> conversions;

  for (auto &node : F->getNodes()) {
    if (isa<ConvertToNode>(&node)) {
      conversions.push_back(&node);
    }
  }

  llvm::SmallPtrSet<Node *, 8> deadConversions;
  Module &mod = *F->getParent();
  for (Node *node : conversions) {
    if (deadConversions.count(node)) {
      continue;
    }
    ConvertToNode &conversion = *cast<ConvertToNode>(node);
    NodeValue conversionInput = conversion.getInput();
    NodeValue dstVal = conversion.getResult();
    NodeValue srcVal;
    switch (conversionInput.getNode()->getKind()) {
    case Kinded::Kind::ConstantKind:
      srcVal =
          convertConstant(mod, *llvm::cast<Constant>(conversionInput.getNode()),
                          dstVal.getType());
      // Reset conversionInput because it may not be valid anymore.
      conversionInput = NodeValue();
      break;
    case Kinded::Kind::ConvertToNodeKind:
      // So we have "conversion(conversion srcVal to tmpVal) to dstVal".
      // If the type of srcVal is equal to the type of dstVal, we can replace
      // the uses of dstVal with srcVal.
      srcVal = cast<ConvertToNode>(conversionInput.getNode())->getInput();
      break;
    default:
      break;
    }
    // If it is a conversion between the same types, it can be eliminated.
    if (srcVal == NodeValue() &&
        conversionInput.getType() == dstVal.getType()) {
      srcVal = conversionInput;
    }
    // Check if we found a suitable new source for dstVal.
    if (srcVal == NodeValue() || srcVal.getType() != dstVal.getType()) {
      continue;
    }
    // Use srcVal instead of dstVal.
    dstVal.replaceAllUsesOfWith(srcVal, F);
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
    F->eraseNode(conversion);
  }
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

/// Eliminate node sequences that are related to quantization.
static void optimizeQuantization(Function *F) {
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
        Q->getResult().replaceAllUsesOfWith(NC);
        continue;
      }

      if (auto *SN = dyn_cast<SplatNode>(Q->getInput())) {
        // Quantize(Splat) -> Splat'
        SplatNode *newSN = F->createSplat(
            SN->getName(), Q->getResult().getType(), SN->getValue());
        Q->getResult().replaceAllUsesOfWith(newSN);
        continue;
      }
    }

    if (auto *DQ = dyn_cast<DequantizeNode>(node)) {
      if (auto *Q = dyn_cast<QuantizeNode>(DQ->getInput())) {
        // Dequantize(Quantize(X)) -> X
        DQ->getResult().replaceAllUsesOfWith(Q->getInput());
        continue;
      }
      // Fold the rescale into the following Dequantize.
      // Dequantize(rescale) -> Dequantize()
      if (auto *RS = dyn_cast<RescaleQuantizedNode>(DQ->getInput())) {
        auto *newRS = F->createDequantize(DQ->getName(), RS->getInput());
        DQ->getResult().replaceAllUsesOfWith(newRS);

        // We may be able to optimize this rescale node. Remember to visit
        // this new node and try to optimize it later.
        worklist.push_back(newRS);
        continue;
      }
    }

    if (auto *RS = dyn_cast<RescaleQuantizedNode>(node)) {
      if (RS->getInput().getType() == RS->getResult().getType()) {
        // If rescale does not change the type, then simply drop it.
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

  optimizeQuantizedMaxSplat(F);
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
  bool changed = false;

  if (auto *rescale = dyn_cast<RescaleQuantizedNode>(PN->getInput())) {
    T *newPN = createNode<T>(F, PN->getName(), rescale->getInput(),
                             PN->getKernels(), PN->getStrides(), PN->getPads());
    auto rescaleOutTy = F.getParent()->uniqueTypeWithNewShape(
        rescale->getResult().getType(), PN->getResult().dims());
    auto *newRescale =
        F.createRescaleQuantized(rescale->getName(), newPN, rescaleOutTy);
    PN->getResult().replaceAllUsesOfWith(newRescale);
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
static bool sinkRescaleQuantizedNode(Function *F) {
  bool changed = false;
  for (auto &node : F->getNodes()) {
    // Sink Rescale below Reshape node.
    // Reshape(Rescale(X)) -> Rescale(Reshape(X)).
    if (auto *reshape = dyn_cast<ReshapeNode>(&node)) {
      auto *rescale = dyn_cast<RescaleQuantizedNode>(reshape->getInput());
      if (!rescale) {
        continue;
      }

      auto *newReshape = F->createReshape(
          reshape->getName(), rescale->getInput(), reshape->getResult().dims());
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

      auto *newTranspose = F->createTranspose(
          transpose->getName(), rescale->getInput(), transpose->getShuffle());
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
        auto *newCN = F->createConv(
            CN->getName(), newX, newF, newB, CN->getResult().getType(),
            CN->getKernels(), CN->getStrides(), CN->getPads(), CN->getGroup());
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

void glow::convertPlaceholdersToConstants(Function *F, const Context &ctx,
                                          llvm::ArrayRef<Placeholder *> phs) {
  auto *M = F->getParent();
  auto &placeholders = M->getPlaceholders();
  for (auto &PH : placeholders) {
    if (std::find(phs.begin(), phs.end(), PH) != phs.end()) {
      continue;
    }
    auto *tensor = ctx.get(PH);
    if (!tensor) {
      continue;
    }
    auto *constant = M->createConstant(PH->getName(), *tensor);
    PH->getOutput().replaceAllUsesOfWith(constant, F);
  }
}

void glow::optimize(Function *F, CompilationMode mode) {
  // Optimize may be called after backend specific transformations and some
  // nodes may have become unused. It is a good idea to remove them, before
  // proceeding with any further optimizations.
  DCE(F);

  // Sink transpose operations in an attempt to cancel them out.
  // Perform code sinking until a fixed-point is reached.
  // On big functions, the number of iterations until the fixpoint
  // is usually at most 2 or 3 iterations.
  while (sinkCode(F)) {
    // Perform Dead Code Elimination between rounds of code sinking.
    DCE(F);
  }

  // Reshapes and transposes can prevent other optimizations from triggering,
  // so try to optimize them out first.
  optimizeReshape(F);
  if (mode == CompilationMode::Infer) {
    transposeConstants(F);
  }

  // Optimize the pooling operation.
  optimizePool(F);

  // Perform Common Subexpression Elimination.
  CSE(F);

  // Optimize Pad nodes
  mergePadIntoConvolution(F);

  // Perform Dead Code Elimination.
  DCE(F);

  // Merge multiple matmul nodes into a single large matmul.
  mergeMatMul(F);

  // Merge multiple batched adds into a larger batched add.
  mergeBatchedAdd(F);

  // Perform Dead Code Elimination.
  DCE(F);

  if (mode == CompilationMode::Infer) {
    // Merge batch normalization operations.
    // Do after transpose constant folding, as weight transposes can prevent
    // the optimization from triggering.
    optimizeBatchNorm(F);
  }

  // Perform Common Subexpression Elimination.
  CSE(F);

  // Optimize Concat nodes.
  optimizeConcatNodes(F);

  // Optimize arithmetic nodes based on algebraic identities.
  optimizeArithmeticNodes(F);

  // Optimize Tensor shape transformations.
  optimizeSliceOfSplat(F);

  // Merge Transpose into MatMul/FC.
  // Run DCE to ensure correct number of node users.
  DCE(F);
  mergeTransposeIntoMatMulOrFC(F);

  // Optimize away intermediate type conversions.
  optimizeConversions(F);

  // Optimize quantization related operators.
  optimizeQuantization(F);

  while (sinkRescaleQuantizedNode(F)) {
    DCE(F);
    optimizeQuantization(F);
  }

  // Perform Dead Code Elimination.
  DCE(F);
}
