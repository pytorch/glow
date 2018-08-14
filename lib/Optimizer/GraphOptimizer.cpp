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
#include "glow/Quantization/Base/Base.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#include <unordered_map>
#include <unordered_set>

llvm::cl::OptionCategory graphOptCat("Graph Optimizations Options");
llvm::cl::opt<unsigned> constVarDedupSizeOpt(
    "const_var_dedup_size",
    llvm::cl::desc(
        "Max number of elements allowed for deduplicating constant variables"),
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
    if (LHS != NN->getLHS()) {                                                 \
      return simplifyNode(                                                     \
          F->create##NodeKind(NN->getName(), LHS, NN->getRHS()), F);           \
    }                                                                          \
    Node *RHS = simplifyNode(NN->getRHS(), F);                                 \
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
  SIMPLIFY_OPERANDS(CmpEQ)
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
    newAN = F->create##NODE_NAME_(                                             \
        node->getName(),                                                       \
        F->getParent()->uniqueTypeWithNewShape(                                \
            node->getType(0), LTR->getInput().getType()->dims()),              \
        LTR->getInput(), RTR->getInput());                                     \
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
      unsigned_t idx = CN->getDim();
      unsigned_t newChannelIdx = L->getShuffle()[idx];

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
      NodeValue(origMM).replaceAllUsesOfWith(ex);
    }
  }
}

/// \returns True if the two slices \p A and \p B access consecutive spacial
/// regions on the \p dim dimension. For example Slice(0..10) Slice(10..50) are
/// consecutive but Slice(0..10) Slice(20..30) are not.
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
/// \returns True if a group of slices that span the whole input was found. The
/// order of the slices is recorded in \p order.
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

  // Now that we've found the first slice in the sequence, try to order the rest
  // of the slices after the first one.
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

  // Report success if we found at least two slices that extract from the input.
  return order.size() > 1;
}

/// Merge multiple batched add nodes into a large batched-add node.
static void mergeBatchedAdd(Function *F) {
  auto &nodes = F->getNodes();

  // We index the batched add nodes by the slice operand.
  llvm::DenseMap<Node *, std::vector<BatchedAddNode *>> rightBAUsers;

  // Collect all of the batched add nodes and index them by the 'slice' operand.
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

    // We found a sequence of batched-add-slice that cover the input tensor. We
    // can transform the graph and create one big batched-add.
    std::vector<Node *> newSlices;
    SliceNode *S = llvm::cast<SliceNode>(order[0]);
    auto *BA = F->createBatchedAdd("mergedBA", S->getInput(), it.first);

    // Create the new slices. These slices will replace the the original scalar
    // batched-add nodes.
    for (int i = 0, e = order.size(); i < e; i++) {
      auto *orig = order[i];
      newSlices.push_back(F->createSlice(orig->getName(), BA, orig->getStart(),
                                         orig->getResult().getType()));
    }

    // Replace the original individual batched adds with corresponding slices
    // from the new merged batch add.
    for (auto *BA : BAs) {
      for (int i = 0, e = order.size(); i < e; i++) {
        if (BA->getBatch().getNode() == order[i]) {
          NodeValue(BA).replaceAllUsesOfWith(newSlices[i]);
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
    // nodes does not give us much. However, reordering the buffers allows us to
    // reuse the memory buffer of the pool operation and potentially save
    // memory.
    if (auto *PL = dyn_cast<MaxPoolNode>(&node)) {
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
          F->createMaxPool(PL->getName(), RL->getInput(), PL->getKernels(),
                           PL->getStrides(), PL->getPads());
      auto *NRL = F->createRELU(RL->getName(), NPL);
      PL->getResult().replaceAllUsesOfWith(NRL);
      continue;
    }
  } // For all nodes in the graph.
}

/// \returns The uniquely used variable from node or nullptr
/// if node has more than one user or is not a variable.
static Variable *getUniquelyUsedVariable(Node &node) {
  // If that node has more than one use, it may not
  // be okay to modify the underlying variable.
  if (!node.hasOneUse()) {
    return nullptr;
  }
  return dyn_cast<Variable>(&node);
}

static void optimizeBatchNorm(Function *F) {
  auto &nodes = F->getNodes();

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

      // FIXME: We need to use a temporary node to hold on
      // the variables, because using the result directly
      // will create a temporary NodeValue that is going
      // to be otherwise alive at the call site and that
      // messes up with the number of users.
      Node *tmp = CV->getFilter().getNode();
      Variable *filterV = getUniquelyUsedVariable(*tmp);
      tmp = CV->getBias().getNode();
      Variable *cbiasV = getUniquelyUsedVariable(*tmp);

      if (!filterV || !cbiasV) {
        continue;
      }

      Variable *scaleV = cast<Variable>(BN->getScale());
      Variable *biasV = cast<Variable>(BN->getBias());
      Variable *meanV = cast<Variable>(BN->getMean());
      Variable *var = cast<Variable>(BN->getVar());

      auto filterH = filterV->getHandle<>();

      auto cbiasH = cbiasV->getHandle<>();

      auto scaleH = scaleV->getHandle<>();
      auto biasH = biasV->getHandle<>();
      auto meanH = meanV->getHandle<>();
      auto varH = var->getHandle<>();

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

/// \returns true if all dimensions of the \p input tensors are the same except
/// for the provided \p dimension, otherwise return false.
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
/// Example: Given a tensor <1,2,3,4,5>, and a desired trailing dimensions size
/// of 20, and a desired leading dimensions size of 2, this function will return
/// dimension 1 as the trailing dimensions after it are <4,5>, which matches the
/// size 20, and the leading dimensions are <1,2>, which matches the size 2.
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

/// Given input tensors \p inputs and a original ConcatNode \p origConcatN, try
/// to find out if there is a dimension in the input tensors, with which we can
/// meet two requirements:
///   1) Input tensors are concatenate-able along this dimension.
///   2) The trailing/leading dimensions sizes after/before this dimension in
///      the input tensors, are of the same size as the trailing/leading
///      dimensions of the input of the original Concat node after/before the
///      concatenation dimension. It is required, because they ensure that the
///      payload of the new concat node should be the same as the payload of the
///      original concat node, and also won't affect the data order of the
///      entire tensor.
/// \returns this dimension if found, otherwise -1.
static int
findConcatDimForSameTrailingAndLeadingDims(llvm::ArrayRef<NodeValue> inputs,
                                           ConcatNode *originalConcatNode) {
  // For the purpose of the optimiztion
  // Concat(Reshape(X)*N)->Reshape(Concat(N*X)), we want to make sure the new
  // ConcatNode can concatenate on the trailing/leading dimensions which are of
  // the same size of those of the original Concate node.

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

  // Try to find the dimension in the first input such that the trailing/leading
  // dimensions sizes are the same as the sizes of the trailing/leading
  // dimensions based on the concatenation dimension used by the original
  // ConcatNode.
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

/// Given the inputs \p originalConcatInputs of one Concat Nodes, \returns true
/// if they are all ReshapeNode, and the input tensors of these input nodes have
/// same number of dimensions, otherwise returns false.
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

  // If all of the inputs to the concat are extracted from the same input in the
  // right order then we can just use the extract-input instead of the concat.
  // Concat(Slice(X, 0..10), Slice(X, 10..20)) -> X.
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

  // Add all of the interesting nodes to the worklist.
  for (auto &node : F->getNodes()) {
    if (node.isArithmetic()) {
      worklist.push_back(&node);
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

  for (auto &node : nodes) {
    auto *TN = dyn_cast<TransposeNode>(&node);
    if (!TN) {
      continue;
    }
    auto *V = dyn_cast<Variable>(TN->getInput());
    // V must have a single use and be private.
    if (!V || !V->hasOneUse() || !V->isPrivate()) {
      continue;
    }
    // Create a new variable NV to hold the transposed result.
    auto *NV =
        F->getParent()->createVariable(TN->getResult().getType(), V->getName(),
                                       V->getVisibilityKind(), V->isTraining());
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

/// A helper type for hashing Variable pointers when they are used as keys in
/// hash maps for deduplication. The hash is based on the type of the Variable
/// (element type, dimensions), as well as a constant number of elements from
/// the Variable to balance collisions with hash calclulation time.
struct VarsHasherDedup {
  size_t operator()(Variable *V) const {
    auto hash = llvm::hash_value(V->getType());
    auto &T = V->getPayload();
    // Only use the first 8 elements in the hash. It's likely that if two
    // tensors have different content they will diverge quickly. Fall back to
    // full equality check in VarsEqDedup.
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

/// A helper type implementing the Variable equality predicate that can be used
/// when Variable pointers are used as keys in hash maps for deduplication. It
/// is assumed the Visibility and training mode are the same, as deduplication
/// only inserts if Private and None, respectively.
struct VarsEqDedup {
  bool operator()(const Variable *lhs, const Variable *rhs) const {
    // Only consider Vars for deduplication if they have the same type. The
    // train kind and visibility must already be the same.
    if (lhs->getType() != rhs->getType()) {
      return false;
    }
    assert(lhs->getVisibilityKind() == rhs->getVisibilityKind() &&
           "Should only be comparing Variables with same VisibilityKind.");
    assert(lhs->isTraining() == rhs->isTraining() &&
           "Should only be comparing Variables with same training mode.");
    // Only combine Vars if their data matches exactly, so allowed error is 0.0.
    return lhs->getPayload().isEqual(rhs->getPayload(), /* allowedError */ 0.0);
  }
};

} // namespace

/// \returns true if Variable \p V is written into, either as a result, or as an
/// overwritten input.
static bool hasWriters(Variable *V) {
  for (auto &U : V->getUsers()) {
    auto *N = U.getUser();

    // See if V is used as a result anywhere.
    for (unsigned i = 0; i < N->getNumResults(); i++)
      if (V == N->getNthResult(i))
        return true;

    // See if V is used as an overwritten input anywhere.
    for (unsigned i = 0; i < N->getNumInputs(); i++)
      if (N->isOverwrittenNthInput(i))
        if (V == N->getNthInput(i))
          return true;
  }
  return false;
}

/// Deduplicates constant variables in the Module \p M. Applicable constant
/// variables for deduplication must have the same data, have
/// VisibilityKind::Private, not trainable, and have no writers.
static void deduplicateConstants(Module *M) {
  // Map from Variables to other Variables that are equivalent for purposes of
  // deduplication.
  std::unordered_map<Variable *, Variable *, VarsHasherDedup, VarsEqDedup>
      duplicateVars;

  for (auto &V : M->getVars()) {
    // Only perform deduplication on vars of small enough size. Otherwise just
    // skip them. constVarDedupSizeOpt defaults to 256 as a heuristic, to keep
    // compile time reasonable.
    size_t maxNumEls = constVarDedupSizeOpt;
    size_t numEls = V->getType()->size();
    if (numEls > maxNumEls) {
      continue;
    }

    // Only perform deduplication on private vars that have no train kind.
    if (V->getVisibilityKind() != VisibilityKind::Private || V->isTraining()) {
      continue;
    }

    // Only perform deduplication on vars that have no writers.
    if (hasWriters(V)) {
      continue;
    }

    // Try to find a var that has the same data as the current one.
    auto foundI = duplicateVars.find(V);
    if (foundI == duplicateVars.end()) {
      // No node equivalent to the current one has been seen yet. Remember this
      // variable, so that the next occurrence can be replaced by this one.
      duplicateVars.emplace(V, V);
      assert(duplicateVars.find(V) != duplicateVars.end());
      continue;
    }
    Variable *foundV = foundI->second;
    assert(V != foundV && "Variables should not be visited multiple times.");

    // Replace current var by a found var, which is equivalent to it.
    V->getOutput().replaceAllUsesOfWith(foundV);
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
    auto inputNode = reshapeNode->getNthInput(0);
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
        // If the quantization-dequantization sequence does not change the type
        // then we can simply drop them without adding a requantization node.
        if (DQ->getInput().getType() == Q->getResult().getType()) {
          Q->getResult().replaceAllUsesOfWith(DQ->getInput());
          continue;
        }

        auto *RS = F->createRescaleQuantized(Q->getName(), DQ->getInput(),
                                             Q->getResult().getType());
        Q->getResult().replaceAllUsesOfWith(RS);

        // We may be able to optimize this rescale node. Remember to visit this
        // new node and try to optimize it later.
        worklist.push_back(RS);
        continue;
      }

      if (auto *V = dyn_cast<Variable>(Q->getInput())) {
        // Quantize(Variable) -> Variable
        // V must be a private variable.
        // Note, it does not really matter how many usages this var has.
        // Quantized graph will use optimized var and other functions will
        // refer to the floating point original var.
        if (!V || !V->isPrivate()) {
          continue;
        }
        // Create a new variable NV to hold the quantized result.
        auto *NV = F->getParent()->createVariable(
            Q->getResult().getType(), V->getName(), V->getVisibilityKind(),
            false);
        // Quantize V into NV.
        auto srcHandle = V->getHandle();
        auto destHandle = NV->getHandle<int8_t>();
        TensorQuantizationParams params{Q->getResult().getType()->getScale(),
                                        Q->getResult().getType()->getOffset()};
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
      if (RS->getInput().getType() == RS->getResult().getType()) {
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

// Combine the rescale node up into the arithmetic node.
// Rescale(Arithmetic()) -> Arithmetic().
// Not all arithmetic nodes support explicit output quantized type.
// Combine up rescale with Add, Sub, Mul, Div, Min, Max.
#define COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(NODE_NAME_)                      \
  if (auto *AN = dyn_cast<NODE_NAME_##Node>(RS->getInput())) {                 \
    auto *newAN = F->create##NODE_NAME_(                                       \
        AN->getName(), RS->getResult().getType(), AN->getLHS(), AN->getRHS()); \
    RS->getResult().replaceAllUsesOfWith(newAN);                               \
                                                                               \
    continue;                                                                  \
  }

      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Add);
      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Sub);
      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Mul);
      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Div);
      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Min);
      COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE(Max);
#undef COMBINE_UP_RESCALE_TO_ARITHMETIC_NODE

      // Combine the rescale node up into the convolution.
      // Rescale(Conv()) -> Conv()
      if (auto *CN = dyn_cast<ConvolutionNode>(RS->getInput())) {
        // Create the exact same convolution but with a different scaling
        // return type.
        auto *newCN = F->createConv(
            CN->getName(), CN->getInput(), CN->getFilter(), CN->getBias(),
            RS->getResult().getType(), CN->getKernels(), CN->getStrides(),
            CN->getPads(), CN->getGroup());
        RS->getResult().replaceAllUsesOfWith(newCN);
        continue;
      }

      // Merge splat and rescale nodes.
      // Rescale(Splat()) -> Splat()
      if (auto *SP = dyn_cast<SplatNode>(RS->getInput())) {
        auto *newRS = F->createSplat(SP->getName(), RS->getResult().getType(),
                                     SP->getValue());

        worklist.push_back(newRS);
        RS->getResult().replaceAllUsesOfWith(newRS);
        continue;
      }

      // Fold the rescale into the previous rescale.
      // Rescale(Rescale()) -> Rescale()
      if (auto *RS2 = dyn_cast<RescaleQuantizedNode>(RS->getInput())) {
        auto *newRS = F->createRescaleQuantized(RS->getName(), RS2->getInput(),
                                                RS->getResult().getType());
        worklist.push_back(newRS);
        RS->getResult().replaceAllUsesOfWith(newRS);
        continue;
      }

      // Fold the rescale into the previous quantize.
      // Rescale(Quantize()) -> Quantize()
      if (auto *QN = dyn_cast<QuantizeNode>(RS->getInput())) {
        auto *newQ = F->createQuantize(QN->getName(), QN->getInput(),
                                       RS->getResult().getType());
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

// Combine Rescale down with Arithmetic node.
//   ArithmeticNode(Rescale(X), Rescale(Y)) -> ArithmeticNode(X, Y).
//   ArithmeticNode(Rescale(X), Y) -> ArithmeticNode(X, Y).
//   ArithmeticNode(X, Rescale(Y)) -> ArithmeticNode(X, Y).
// Apply this optimization for Add, Sub, Mul, Div, Min, Max.
#define COMBINE_DOWN_RESCALE_TO_ARITHMETIC_NODE(NODE_NAME_)                    \
  if (auto *AN = dyn_cast<NODE_NAME_##Node>(&node)) {                          \
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
    COMBINE_DOWN_RESCALE_TO_ARITHMETIC_NODE(Min);
    COMBINE_DOWN_RESCALE_TO_ARITHMETIC_NODE(Max);
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

  // Merge multiple batched adds into a larger batched add.
  mergeBatchedAdd(F);

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
