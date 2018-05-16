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

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Quantization/Quantization.h"

#include "llvm/Support/Casting.h"

#include <unordered_map>
#include <unordered_set>

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

  if (Variable *V = dyn_cast<Variable>(N)) {
    // We don't want to delete unused public variables because they are
    // accessible to the outside world that may hold a reference to them.
    if (V->getVisibilityKind() == VisibilityKind::Public)
      return false;
  }

  return true;
}

/// Dead code elimination.
static void DCE(Function *F) {
  auto &nodes = F->getNodes();
  auto &vars = F->getParent()->getVars();

  std::vector<VariablesList::iterator> erasedVars{};
  std::vector<NodesList::iterator> erasedNodes{};

  // Remove unused nodes. Do not remove unused vars because they are the
  // interface to the user program.
  bool changedLocally = true;
  do {
    changedLocally = false;
    for (auto it = nodes.begin(), e = nodes.end(); it != e;) {
      if (!shouldDeleteNode(*it)) {
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

  } while (changedLocally);

  // Delete unused variables.
  for (auto it = vars.begin(), e = vars.end(); it != e;) {
    if (!shouldDeleteNode(*it)) {
      ++it;
      continue;
    }
    erasedVars.push_back(it);
    ++it;
  }

  while (!erasedVars.empty()) {
    auto it = erasedVars.back();
    F->getParent()->eraseVariable(it);
    erasedVars.pop_back();
  }
}

/// \returns true if the masks \p shuffle1 and shuffle2 are
/// the inverse of on another. Applying both masks should result in the identity
/// shuffle.
static bool isIdentityShuffle(llvm::ArrayRef<unsigned> shuffle1,
                              llvm::ArrayRef<unsigned> shuffle2) {

  if (shuffle1.size() != shuffle2.size()) {
    return false;
  }

  // Check if the combined masks are the identity mask.
  for (unsigned i = 0, e = shuffle1.size(); i < e; i++) {
    unsigned idx = shuffle2[shuffle1[i]];
    if (idx != i) {
      return false;
    }
  }
  return true;
}

/// \returns True if the node \p N always evaluates to zero.
bool isZero(Node *N) {
  SplatNode *Z = dyn_cast<SplatNode>(N);
  if (!Z)
    return false;

  return (Z->getValue() == 0);
}

/// \returns True if the node returns a constant value.
bool isConstant(Node *N) { return isa<SplatNode>(N); }

/// \returns the new simplified node or the original node.
static Node *simplifyNode(Node *node, Function *F) {

// Recursively simplify the operands of arithmetic nodes.
#define SIMPLIFY_OPERANDS(NodeKind)                                            \
  if (auto *NN = dyn_cast<NodeKind##Node>(node)) {                             \
    Node *LHS = simplifyNode(NN->getLHS(), F);                                 \
    Node *RHS = simplifyNode(NN->getRHS(), F);                                 \
    if (LHS != NN->getLHS()) {                                                 \
      return simplifyNode(                                                     \
          F->create##NodeKind(NN->getName(), LHS, NN->getRHS()), F);           \
    }                                                                          \
    if (RHS != NN->getRHS()) {                                                 \
      return simplifyNode(                                                     \
          F->create##NodeKind(NN->getName(), NN->getLHS(), RHS), F);           \
    }                                                                          \
  }

  SIMPLIFY_OPERANDS(Add)
  SIMPLIFY_OPERANDS(Mul)
  SIMPLIFY_OPERANDS(Div)
  SIMPLIFY_OPERANDS(Sub)
  SIMPLIFY_OPERANDS(Max)
  SIMPLIFY_OPERANDS(Min)
  SIMPLIFY_OPERANDS(CmpLTE)
#undef SIMPLIFY_OPERANDS

// Simplify commutative nodes by moving the constant operator to the right-hand
// side.
// Example:  C + X  =>  X + C
#define COMMUTE_CONST_TO_RHS(NodeKind)                                         \
  if (auto *NN = dyn_cast<NodeKind##Node>(node))                               \
    if (isConstant(NN->getLHS()) && !isConstant(NN->getRHS())) {               \
      return F->create##NodeKind(NN->getName(), NN->getRHS(), NN->getLHS());   \
    }

  COMMUTE_CONST_TO_RHS(Add)
  COMMUTE_CONST_TO_RHS(Mul)
  COMMUTE_CONST_TO_RHS(Max)
  COMMUTE_CONST_TO_RHS(Min)
#undef COMMUTE_CONST_TO_RHS

  if (auto *AN = dyn_cast<AddNode>(node)) {
    // X + 0 => X
    if (isZero(AN->getRHS())) {
      return AN->getLHS();
    }
  }

  if (auto *MN = dyn_cast<MulNode>(node)) {
    // X * 0 => 0
    if (isZero(MN->getRHS())) {
      return MN->getRHS();
    }
  }

  // 0 / X => 0
  if (auto *DN = dyn_cast<DivNode>(node)) {
    if (isZero(DN->getLHS())) {
      return DN->getLHS();
    }
  }

  // X - 0 => X
  if (auto *SN = dyn_cast<SubNode>(node)) {
    if (isZero(SN->getRHS())) {
      return SN->getLHS();
    }
  }

  return node;
}

/// Code Sinking.
/// \returns true if code sinking was successful.
static bool sinkCode(Function *F) {
  auto &nodes = F->getNodes();
  bool changed = false;
  // For each node:
  for (auto const &node : nodes) {
    // Sink Transpose below batch normalization nodes:
    if (auto *BN = dyn_cast<BatchNormalizationNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(BN->getInput());

      if (!TR) {
        continue;
      }

      // Figure out where we transposed the channel index for batch
      // normalization.
      unsigned idx = BN->getChannelIdx();
      unsigned newChannelIdx = TR->getShuffle()[idx];

      auto *NewBN = F->createBatchNormalization(
          BN->getName(), TR->getInput(), BN->getBias(), BN->getScale(),
          BN->getMean(), BN->getVar(), newChannelIdx, BN->getEpsilon(),
          BN->getMomentum());
      auto *newTR = F->createTranspose(TR->getName(), NewBN, TR->getShuffle());

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

      auto *NRL = F->createRELU(RL->getName(), TR->getInput());
      auto *newTR = F->createTranspose(TR->getName(), NRL, TR->getShuffle());
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
      auto *newTR = F->createTranspose(TR->getName(), NSI, TR->getShuffle());
      SI->getResult().replaceAllUsesOfWith(newTR);
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
      auto *newTR = F->createTranspose(TR->getName(), NTN, TR->getShuffle());
      TN->getResult().replaceAllUsesOfWith(newTR);
      changed = true;
      continue;
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

    // Sink Transpose below Arithmetic nodes. Note: For simplicity, we
    // assume for the arithmetic node, LHS is the 0th input, RHS is 1st, and
    // Result is 0th result.
    if (node->isArithmetic()) {
#define GET_LHS(NODE_) NODE_->getNthInput(0)
#define GET_RHS(NODE_) NODE_->getNthInput(1)
      TransposeNode *LTR = dyn_cast<TransposeNode>(GET_LHS(node));
      TransposeNode *RTR = dyn_cast<TransposeNode>(GET_RHS(node));

      if (!LTR || !RTR) {
        // If one of the sides is a splat, it can be seen as
        // transpose (splat').
        if (isa<SplatNode>(GET_LHS(node)) && RTR) {
          // Build splat' for LHS.
          auto *SN = dyn_cast<SplatNode>(GET_LHS(node));
          auto *NS = F->createSplat("splat", RTR->getInput().getType(),
                                    SN->getValue());
          LTR = F->createTranspose("transpose", NS, RTR->getShuffle());
          changed = true;
        } else if (isa<SplatNode>(GET_RHS(node)) && LTR) {
          // Build splat' for RHS.
          auto *SN = dyn_cast<SplatNode>(GET_RHS(node));
          auto *NS = F->createSplat("splat", LTR->getInput().getType(),
                                    SN->getValue());
          RTR = F->createTranspose("transpose", NS, LTR->getShuffle());
          changed = true;
        } else {
          continue;
        }
      }
#undef GET_LHS
#undef GET_RHS
      // The masks of the transposes on both sizes must match.
      if (LTR->getShuffle() != RTR->getShuffle()) {
        continue;
      }

      Node *newAN = nullptr;

#define ARITHMETIC_CASE(NODE_NAME_)                                            \
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
        ARITHMETIC_CASE(CmpLTE);
      default:
        llvm_unreachable("Unhandled node");
      }
#undef ARITHMETIC_CASE

      changed = true;
      auto *newTR =
          F->createTranspose(LTR->getName(), newAN, LTR->getShuffle());
#define GET_RESULT(NODE_) NODE_->getNthResult(0)
      GET_RESULT(node).replaceAllUsesOfWith(newTR);
#undef GET_RESULT
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
        auto *newRL = F->createRELU(L->getName(), newCN);
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
      unsigned idx = CN->getDim();
      unsigned newChannelIdx = L->getShuffle()[idx];

      auto *newCN = F->createConcat(
          CN->getName(), {L->getInput(), R->getInput()}, newChannelIdx);
      auto *newTR = F->createTranspose(L->getName(), newCN, L->getShuffle());
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
  for (auto const &node : nodes) {
    if (auto *MM = dyn_cast<MatMulNode>(node)) {
      // Do not try to merge quantized matrix multiplications because their
      // quantized parameters may not match. Until we implement the logic to
      // match the scale and offset just avoid the optimization.
      if (MM->getType()->isQuantizedType()) {
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

    size_t R = MM->getNthResult(0).dims()[1];
    size_t start = 0;
    for (auto *origMM : MMs) {
      size_t H = origMM->dims()[0];
      auto *ex = F->createSlice("extract", MM, {start, 0}, {start + H, R});
      start += H;
      NodeValue(origMM).replaceAllUsesOfWith(ex);
    }
  }
}

/// Pool optimization.
static void optimizePool(Function *F) {
  auto &nodes = F->getNodes();

  // For each node:
  for (auto const &node : nodes) {
    // Swap the order of Relu->MaxPool, to perform the RELU operation on a
    // smaller tensor. This optimization is not a major performance win. The
    // RELU operation takes a small fraction of the time, and reordering the
    // nodes does not give us much. However, reordering the buffers allows us to
    // reuse the memory buffer of the pool operation and potentially save
    // memory.
    if (auto *PL = dyn_cast<PoolMaxNode>(node)) {
      auto *RL = dyn_cast<ReluNode>(PL->getInput());

      if (!RL) {
        continue;
      }

      // We don't want to increase the number of operations in the program, so
      // perform this transformation if the relu has a single user, which is the
      // pooling operation.
      if (!RL->hasOneUse()) {
        continue;
      }

      auto *NPL =
          F->createPoolMax(PL->getName(), RL->getInput(), PL->getKernel(),
                           PL->getStride(), PL->getPad());
      auto *NRL = F->createRELU(RL->getName(), NPL);
      PL->getResult().replaceAllUsesOfWith(NRL);
      continue;
    }
  } // For all nodes in the graph.
}

static void optimizeBatchNorm(Function *F) {
  auto &nodes = F->getNodes();

  // For each node:
  for (auto const &node : nodes) {
    // Merge the Batch Normalization operation into the convolution that comes
    // before it by updating the weights of the filter.
    if (auto *BN = dyn_cast<BatchNormalizationNode>(node)) {
      auto *CV = dyn_cast<ConvolutionNode>(BN->getInput());
      if (!CV) {
        continue;
      }

      // We can't modify conv operators that have multiple users.
      if (!CV->hasOneUse()) {
        continue;
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

      auto filterH = cast<Variable>(CV->getFilter())->getHandle<>();
      auto cbiasH = cast<Variable>(CV->getBias())->getHandle<>();

      auto scaleH = cast<Variable>(BN->getScale())->getHandle<>();
      auto biasH = cast<Variable>(BN->getBias())->getHandle<>();
      auto meanH = cast<Variable>(BN->getMean())->getHandle<>();
      auto varH = cast<Variable>(BN->getVar())->getHandle<>();

      // Update the filter/bias variables of the Conv node.
      auto epsilon = BN->getEpsilon();
      for (size_t i = 0, e = filterH.size(); i < e; i++) {
        // Dimension zero is the 'channel' dimension. If we ever change the
        // layout of the filter then we need to change this optimization.
        size_t channelId = filterH.getDimForPtr(0, i);
        float var = varH.at({channelId});
        float stdvar = 1.0f / std::sqrt(var + epsilon);
        float gamma = scaleH.at({channelId});
        float A = gamma * stdvar;
        filterH.raw(i) = filterH.raw(i) * A;
      }

      for (size_t i = 0, e = cbiasH.size(); i < e; i++) {
        // Dimension zero is the 'channel' dimension. If we ever change the
        // layout of the filter then we need to change this optimization.
        size_t channelId = cbiasH.getDimForPtr(0, i);
        float mu = meanH.at({channelId});
        float var = varH.at({channelId});
        float stdvar = 1.0f / std::sqrt(var + epsilon);
        float gamma = scaleH.at({channelId});
        float beta = biasH.at({channelId});
        float A = gamma * stdvar;
        float B = beta - mu * A;
        cbiasH.raw(i) = cbiasH.raw(i) * A + B;
      }

      BN->getResult().replaceAllUsesOfWith(CV);
    }
  } // For all nodes in the graph.
}

/// Concat nodes merging.
/// concat(dim1, concat(dim2, X, Y), Z) -> concat(dim1, X, Y, Z)
/// but only if dim1 == dim2
static void optimizeConcatNodes(Function *F) {
  auto &nodes = F->getNodes();

  // For each node:
  for (auto const &node : nodes) {
    if (auto *CN = dyn_cast<ConcatNode>(node)) {
      auto inputs = CN->getInputs();
      // Check if any of the inputs is a ConcatNode.
      llvm::SmallVector<NodeValue, 16> newInputs;
      bool changed = false;
      for (auto input : inputs) {
        newInputs.push_back(input);
        auto *CNI = dyn_cast<ConcatNode>(input);
        // Bail if it is not a ConcatNode or it is a concat node with a diffrent
        // dimension.
        if (!CNI || CNI->getDim() != CN->getDim())
          continue;

        changed = true;
        // Replace current input by its own inputs, i.e. merge them into the
        // parent concat node.
        newInputs.pop_back();
        newInputs.append(CNI->getInputs().begin(), CNI->getInputs().end());
      }
      if (!changed)
        continue;
      // Create a new Concat node.
      auto newCN = F->createConcat(CN->getName(), newInputs, CN->getDim());
      CN->getResult().replaceAllUsesOfWith(newCN);
    }
  }
}

/// Simplify and canonicalize arithmetic nodes by detecting simple arithmetic
/// identities.
static void optimizeArithmeticNodes(Function *F) {
  // A worklist that contains the nodes to process.
  std::vector<Node *> worklist;

  // Add all of the interesting nodes to the worklist.
  for (auto *node : F->getNodes()) {
    if (node->isArithmetic()) {
      worklist.push_back(node);
    }
  }

  while (!worklist.empty()) {
    Node *node = worklist.back();
    worklist.pop_back();

    auto *sn = simplifyNode(node, F);
    if (sn != node) {
      node->getNthResult(0).replaceAllUsesOfWith(sn);
      // The simplified node could be further simplified.
      worklist.push_back(sn);
      continue;
    }
  }
}

/// Statically transpose private variables.
static void optimizeTranspose(Function *F) {
  auto &nodes = F->getNodes();

  for (auto const &node : nodes) {
    auto *TN = dyn_cast<TransposeNode>(node);
    if (!TN) {
      continue;
    }
    auto *V = dyn_cast<Variable>(TN->getInput());
    // V must have a single use and be private.
    if (!V || !V->hasOneUse() || !V->isPrivate()) {
      continue;
    }
    // Create a new variable NV to hold the transposed result.
    auto *NV = F->getParent()->createVariable(TN->getType(), V->getName(),
                                              V->getVisibilityKind(),
                                              V->getTrainKind(), V->getValue());
    // Transpose the value of V into NV.
    genericTranspose(&V->getPayload(), &NV->getPayload(), TN->getShuffle());
    // Rewrite uses of TN to reference NV.
    TN->getResult().replaceAllUsesOfWith(NV);
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

/// This visitor is used to walk the the graph and
/// perform a common subexpression evaluation.
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

} // namespace

/// Common Subexpression Elimination.
static void CSE(Function *F) {
  CSEVisitor visitor;

  // No need to perform CSE on variables because
  // all variables are distinct from each other.

  // Perform CSE on all nodes.
  for (auto const &N : F->getNodes()) {
    N->visit(nullptr, &visitor);
  }
}

/// Eliminate SliceNode when the input is SplatNode.
/// Slice(Splat(args)) -> Splat(args')
static void optimizeSliceOfSplat(Function *F) {
  for (const auto &node : F->getNodes()) {
    auto *sliceNode = dyn_cast<SliceNode>(node);
    if (!sliceNode)
      continue;
    auto *splatNode = dyn_cast<SplatNode>(sliceNode->getInput());
    if (!splatNode)
      continue;
    auto *newSplatNode = F->createSplat(
        sliceNode->getName(), sliceNode->getType(), splatNode->getValue());
    sliceNode->getResult().replaceAllUsesOfWith(newSplatNode);
  }
}

/// Eliminate ReshapeNode when the input is already the correct shape.
static void optimizeReshape(Function *F) {
  for (const auto &node : F->getNodes()) {
    auto *reshapeNode = dyn_cast<ReshapeNode>(node);
    if (!reshapeNode)
      continue;
    auto inputNode = reshapeNode->getInput();
    if (inputNode.dims() == reshapeNode->dims()) {
      reshapeNode->getResult().replaceAllUsesOfWith(inputNode);
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
  for (auto node : F->getNodes()) {
    // Potentially nop quantized Max can be eliminated.
    // Likely MaxNode has same types for LHS/RHS and Result, make sure
    // it's the case.
    if (auto *MN = dyn_cast<MaxNode>(node)) {
      if (!node->getType()->isQuantizedType() ||
          MN->getResult()->getType() != MN->getLHS()->getType() ||
          MN->getResult()->getType() != MN->getRHS()->getType()) {
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

      float splatValue = (dyn_cast<SplatNode>(splatNode))->getValue();
      // Calculate quantization [min,max] range.
      TensorQuantizationParams TQP{MN->getResult().getType()->getScale(),
                                   MN->getResult().getType()->getOffset()};
      float min =
          quantization::dequantize(std::numeric_limits<int8_t>::min(), TQP);

      // If splat value is smaller than values that can be covered by
      // quantition [min,max] range then just remove MaxNode operation.
      if (splatValue <= min) {
        MN->getResult().replaceAllUsesOfWith(otherInput);
      }
    }
  }
}

/// Eliminate node sequences that are related to quantization.
static void optimizeQuantization(Function *F) {
  // A worklist that contains the nodes to process.
  std::vector<Node *> worklist;

  // Add all of the interesting nodes to the worklist.
  for (auto *node : F->getNodes()) {
    if (isa<QuantizeNode>(node) || isa<DequantizeNode>(node) ||
        isa<RescaleQuantizedNode>(node)) {
      worklist.push_back(node);
    }
  }

  while (!worklist.empty()) {
    // Take a node from the worklist.
    Node *node = worklist.back();
    worklist.pop_back();

    if (auto *Q = dyn_cast<QuantizeNode>(node)) {
      if (auto *DQ = dyn_cast<DequantizeNode>(Q->getInput())) {
        // Quantize(Dequantize(X)) -> RescaleQuantized(X)
        // If the quantization-dequantization sequence does not change the type
        // then we can simply drop them without adding a requantization node.
        if (DQ->getInput()->getType() == Q->getType()) {
          Q->getResult().replaceAllUsesOfWith(DQ->getInput());
          continue;
        }

        auto *RS = F->createRescaleQuantized(Q->getName(), DQ->getInput(),
                                             Q->getType());
        Q->getResult().replaceAllUsesOfWith(RS);

        // We may be able to optimize this rescale node. Remember to visit this
        // new node and try to optimize it later.
        worklist.push_back(RS);
        continue;
      }

      if (auto *V = dyn_cast<Variable>(Q->getInput())) {
        // Quantize(Variable) -> Variable
        // V must have a single use and be private.
        if (!V || !V->hasOneUse() || !V->isPrivate()) {
          continue;
        }
        // Create a new variable NV to hold the quantized result.
        auto *NV = F->getParent()->createVariable(Q->getType(), V->getName(),
                                                  V->getVisibilityKind(),
                                                  V->getTrainKind(), 1.0);
        // Quantize V into NV.
        auto srcHandle = V->getHandle();
        auto destHandle = NV->getHandle<int8_t>();
        TensorQuantizationParams params{Q->getType()->getScale(),
                                        Q->getType()->getOffset()};
        for (size_t i = 0, e = destHandle.size(); i < e; ++i) {
          destHandle.raw(i) = quantization::quantize(srcHandle.raw(i), params);
        }
        Q->getResult().replaceAllUsesOfWith(NV);
        continue;
      }
    }

    if (auto *DQ = dyn_cast<DequantizeNode>(node)) {
      if (auto *Q = dyn_cast<QuantizeNode>(DQ->getInput())) {
        // Dequantize(Quantize(X)) -> X
        DQ->getResult().replaceAllUsesOfWith(Q->getInput());
        continue;
      }
    }

    if (auto *RS = dyn_cast<RescaleQuantizedNode>(node)) {
      if (RS->getInput()->getType() == RS->getType()) {
        // If rescale does not change the type, then simply drop it.
        RS->getResult().replaceAllUsesOfWith(RS->getInput());
        continue;
      }

      if (auto *MN = dyn_cast<MaxNode>(RS->getInput())) {
        // Rescale(MAX(X, Y)) -> MAX(Rescale(X), Rescale(Y)).
        // It's okay to rescale the operands because even if the output range is
        // smaller then truncation would have happened during the rescale. On
        // values that are outside of the range we just moved the truncation to
        // a different location.
        auto name = RS->getName();
        auto *L = F->createRescaleQuantized(name, MN->getLHS(), RS->getType());
        auto *R = F->createRescaleQuantized(name, MN->getRHS(), RS->getType());
        auto *newMN = F->createMax(MN->getName(), L, R);
        worklist.push_back(L);
        worklist.push_back(R);
        RS->getResult().replaceAllUsesOfWith(newMN);
        continue;
      }

// Combine the rescale node up into the arithmetic node.
// Rescale(Arithmetic()) -> Arithmetic().
// Not all arithmetic nodes support explicit output quantized type.
// Combine up rescale with add, sub, mul, div.
#define COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(NODE_NAME_)                      \
  if (auto *AN = dyn_cast<NODE_NAME_##Node>(RS->getInput())) {                 \
    auto *newAN = F->create##NODE_NAME_(AN->getName(), RS->getType(),          \
                                        AN->getLHS(), AN->getRHS());           \
    RS->getResult().replaceAllUsesOfWith(newAN);                               \
                                                                               \
    continue;                                                                  \
  }

      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Add);
      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Sub);
      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Mul);
      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Div);
#undef COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE

      // Combine the rescale node up into the convolution.
      // Rescale(Conv()) -> Conv()
      if (auto *CN = dyn_cast<ConvolutionNode>(RS->getInput())) {
        // Create the exact same convolution but with a different scaling
        // return type.
        auto *newCN =
            F->createConv(CN->getName(), CN->getInput(), CN->getFilter(),
                          CN->getBias(), RS->getType(), CN->getKernel(),
                          CN->getStride(), CN->getPad(), CN->getGroup());
        RS->getResult().replaceAllUsesOfWith(newCN);
        continue;
      }

      // Merge splat and rescale nodes.
      // Rescale(Splat()) -> Splat()
      if (auto *SP = dyn_cast<SplatNode>(RS->getInput())) {
        auto *newRS =
            F->createSplat(SP->getName(), RS->getType(), SP->getValue());

        worklist.push_back(newRS);
        RS->getResult().replaceAllUsesOfWith(newRS);
        continue;
      }

      // Fold the rescale into the previous rescale.
      // Rescale(Rescale()) -> Rescale()
      if (auto *RS2 = dyn_cast<RescaleQuantizedNode>(RS->getInput())) {
        auto *newRS = F->createRescaleQuantized(RS->getName(), RS2->getInput(),
                                                RS->getType());
        worklist.push_back(newRS);
        RS->getResult().replaceAllUsesOfWith(newRS);
        continue;
      }

      // Fold the rescale into the previous quantize.
      // Rescale(Quantize()) -> Quantize()
      if (auto *QN = dyn_cast<QuantizeNode>(RS->getInput())) {
        auto *newQ =
            F->createQuantize(QN->getName(), QN->getInput(), RS->getType());
        worklist.push_back(newQ);
        RS->getResult().replaceAllUsesOfWith(newQ);
        continue;
      }
    } // Handle RescaleQuantizedNode
  }   // For each item in the worklist.

  optimizeQuantizedMaxSplat(F);
}

/// Sink Rescale nodes down when possible.
static bool sinkRescaleQuantizedNode(Function *F) {
  bool changed = false;
  for (auto const &node : F->getNodes()) {
    // Sink Rescale below Reshape node.
    // Reshape(Rescale(X)) -> Rescale(Reshape(X)).
    if (auto *reshape = dyn_cast<ReshapeNode>(node)) {
      auto *rescale = dyn_cast<RescaleQuantizedNode>(reshape->getInput());
      if (!rescale) {
        continue;
      }

      auto *newReshape = F->createReshape(reshape->getName(),
                                          rescale->getInput(), reshape->dims());
      auto *newRescale = F->createRescaleQuantized(
          rescale->getName(), newReshape, reshape->getResult().getType());
      reshape->getResult().replaceAllUsesOfWith(newRescale);

      changed = true;
      continue;
    }

    // Sink Rescale below Slice node.
    // Slice(Rescale(X)) -> Rescale(Slice(X)).
    if (auto *slice = dyn_cast<SliceNode>(node)) {
      auto *rescale = dyn_cast<RescaleQuantizedNode>(slice->getInput());
      if (!rescale) {
        continue;
      }

      auto sliceOutTy = F->getParent()->uniqueTypeWithNewShape(
          rescale->getInput().getType(), slice->dims());
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
    if (auto *transpose = dyn_cast<TransposeNode>(node)) {
      auto *rescale = dyn_cast<RescaleQuantizedNode>(transpose->getInput());
      if (!rescale) {
        continue;
      }

      auto *newTranspose = F->createTranspose(
          transpose->getName(), rescale->getInput(), transpose->getShuffle());
      auto rescaleOutTy = F->getParent()->uniqueTypeWithNewShape(
          rescale->getResult().getType(), transpose->dims());
      auto *newRescale = F->createRescaleQuantized(rescale->getName(),
                                                   newTranspose, rescaleOutTy);
      transpose->getResult().replaceAllUsesOfWith(newRescale);

      changed = true;
      continue;
    }

    // Combine Rescale down with FullyConnected node.
    // FullyConnected(Rescale(X)) -> FullyConnected(X).
    if (auto *FC = dyn_cast<FullyConnectedNode>(node)) {
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

// Combine Rescale down with Arithmetic node.
//   ArithmeticNode(Rescale(X), Rescale(Y)) -> ArithmeticNode(X, Y).
//   ArithmeticNode(Rescale(X), Y) -> ArithmeticNode(X, Y).
//   ArithmeticNode(X, Rescale(Y)) -> ArithmeticNode(X, Y).
// Apply this optimization for Add, Sub, Mul and Div.
#define COMBINE_DOWN_RESCALE_TO_ARITHMETIC_NODE(NODE_NAME_)                    \
  if (auto *AN = dyn_cast<NODE_NAME_##Node>(node)) {                           \
    if (auto *rescale = dyn_cast<RescaleQuantizedNode>(AN->getLHS())) {        \
      auto *newAN =                                                            \
          F->create##NODE_NAME_(AN->getName(), AN->getResult().getType(),      \
                                rescale->getInput(), AN->getRHS());            \
      AN->getResult().replaceAllUsesOfWith(newAN);                             \
      AN = newAN;                                                              \
      changed = true;                                                          \
    }                                                                          \
    if (auto *rescale = dyn_cast<RescaleQuantizedNode>(AN->getRHS())) {        \
      auto *newAN =                                                            \
          F->create##NODE_NAME_(AN->getName(), AN->getResult().getType(),      \
                                AN->getLHS(), rescale->getInput());            \
      AN->getResult().replaceAllUsesOfWith(newAN);                             \
      changed = true;                                                          \
    }                                                                          \
    continue;                                                                  \
  }
    COMBINE_DOWN_RESCALE_TO_ARITHMETIC_NODE(Add);
    COMBINE_DOWN_RESCALE_TO_ARITHMETIC_NODE(Sub);
    COMBINE_DOWN_RESCALE_TO_ARITHMETIC_NODE(Mul);
    COMBINE_DOWN_RESCALE_TO_ARITHMETIC_NODE(Div);
#undef COMBINE_DOWN_RESCALE_TO_ARITHMETIC_NODE
  }

  return changed;
}

void glow::optimize(Function *F, CompilationMode mode) {
  // Sink transpose operations in an attempt to cancel them out.
  // Perform code sinking until a fixed-point is reached.
  // On big functions, the number of iterations until the fixpoint
  // is usually at most 2 or 3 iterations.
  while (sinkCode(F)) {
    // Perform Dead Code Elimination between rounds of code sinking.
    DCE(F);
  }

  // Optimize the pooling operation.
  optimizePool(F);

  // Perform Common Subexpression Elimination.
  CSE(F);

  // Merge multiple matmul nodes into a single large matmul.
  mergeMatMul(F);

  // Perform Dead Code Elimination.
  DCE(F);

  if (mode == CompilationMode::Infer) {
    // Merge batch normalization operations.
    optimizeBatchNorm(F);

    // Constant-fold transpose operations.
    optimizeTranspose(F);
  }

  // Perform Common Subexpression Elimination.
  CSE(F);

  // Optimize Concat nodes.
  optimizeConcatNodes(F);

  // Optimize arithmetic nodes based on algebraic identities.
  optimizeArithmeticNodes(F);

  // Optimize Tensor shape transformations.
  optimizeSliceOfSplat(F);

  optimizeReshape(F);

  // Optimize things that are related to quantization.
  optimizeQuantization(F);

  while (sinkRescaleQuantizedNode(F)) {
    DCE(F);
    optimizeQuantization(F);
  }

  // Perform Dead Code Elimination.
  DCE(F);
}
