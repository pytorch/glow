// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

#include <unordered_map>
#include <unordered_set>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;

/// Dead code elimination.
static void DCE(Graph &G, CompilationMode mode) {
  auto &nodes = G.getNodes();
  auto &vars = G.getVars();

  std::vector<VariablesList::iterator> erasedVars{};
  std::vector<NodesList::iterator> erasedNodes{};

  // Remove unused nodes. Do not remove unused vars because they are the
  // interface to the user program.
  bool changedLocally = true;
  do {
    changedLocally = false;
    for (auto it = nodes.begin(), e = nodes.end(); it != e;) {
      bool used = (*it)->hasUsers();
      if (used || llvm::isa<SaveNode>(*it)) {
        ++it;
        continue;
      }

      // In training mode we should not remove the training nodes.
      if (mode == CompilationMode::Train) {
        if (llvm::isa<SGDNode>(*it)) {
          ++it;
          continue;
        }
      }

      erasedNodes.push_back(it);
      ++it;
      changedLocally = true;
    }

    while (!erasedNodes.empty()) {
      auto it = erasedNodes.back();
      G.eraseNode(it);
      erasedNodes.pop_back();
    }

  } while (changedLocally);

  // Delete unused variables.
  for (auto it = vars.begin(), e = vars.end(); it != e;) {
    if ((*it)->hasUsers()) {
      ++it;
      continue;
    }
    erasedVars.push_back(it);
    ++it;
  }

  while (!erasedVars.empty()) {
    auto it = erasedVars.back();
    G.eraseVariable(it);
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

/// Code Sinking.
static void sinkCode(Graph &G) {
  auto &nodes = G.getNodes();

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

      auto *NewBN = G.createBatchNormalization(
          BN->getName(), TR->getInput(), BN->getBias(), BN->getScale(),
          BN->getMean(), BN->getVar(), newChannelIdx, BN->getEpsilon(),
          BN->getMomentum());
      auto *newTR = G.createTranspose(TR->getName(), NewBN, TR->getShuffle());

      BN->replaceAllUsesOfWith(newTR);
      continue;
    }

    // Sink Transpose below batch RELU nodes.
    // TODO: support other similar activation functions, such as sigmoid, etc.
    if (auto *RL = dyn_cast<ReluNode>(node)) {
      auto *TR = dyn_cast<TransposeNode>(RL->getInput());

      if (!TR) {
        continue;
      }

      auto *NRL = G.createRELU(RL->getName(), TR->getInput());
      auto *newTR = G.createTranspose(TR->getName(), NRL, TR->getShuffle());
      RL->replaceAllUsesOfWith(newTR);
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

      // The two transposes are reversing one another. We can skip both of them
      // alltogether.
      if (isIdentityShuffle(mask1, mask2)) {
        TR1->replaceAllUsesOfWith(TR2->getInput());
        continue;
      }
    }

    // Sink Transpose below Arithmetic nodes.
    if (auto *AN = dyn_cast<ArithmeticNode>(node)) {
      auto *LTR = dyn_cast<TransposeNode>(AN->getLHS());
      auto *RTR = dyn_cast<TransposeNode>(AN->getRHS());

      if (!LTR || !RTR) {
        continue;
      }
      // The masks of the transposes on both sizes must match.
      if (LTR->getShuffle() != RTR->getShuffle()) {
        continue;
      }

      auto *newAN = G.createArithmetic(AN->getName(), LTR->getInput(),
                                       RTR->getInput(), AN->getMode());
      auto *newTR = G.createTranspose(LTR->getName(), newAN, LTR->getShuffle());
      AN->replaceAllUsesOfWith(newTR);
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
        auto *newCN = G.createConcat(
            CN->getName(), {L->getInput(), R->getInput()}, CN->getDim());
        auto *newRL = G.createRELU(L->getName(), newCN);
        CN->replaceAllUsesOfWith(newRL);
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

      auto *newCN = G.createConcat(
          CN->getName(), {L->getInput(), R->getInput()}, newChannelIdx);
      auto *newTR = G.createTranspose(L->getName(), newCN, L->getShuffle());
      CN->replaceAllUsesOfWith(newTR);
    }

  } // For all nodes in the graph.
}

/// Pool optimization.
static void OptimizePool(Graph &G) {
  auto &nodes = G.getNodes();

  // For each node:
  for (auto const &node : nodes) {
    // Swap the order of Relu->MaxPool, to perform the RELU operation on a
    // smaller tensor. This optimization is not a major performance win. The
    // RELU operation takes a small fraction of the time, and reordering the
    // nodes does not give us much. However, reordering the buffers allows us to
    // reuse the memory buffer of the pool operation and potentially save
    // memory.
    if (auto *PL = dyn_cast<PoolNode>(node)) {
      auto *RL = dyn_cast<ReluNode>(PL->getInput());

      if (!RL) {
        continue;
      }

      // This optimization is only valid on max pooling.
      if (PL->getMode() != PoolNode::Mode::Max) {
        continue;
      }

      // We don't want to increase the number of operations in the program, so
      // perform this transformation if the relu has a single user, which is the
      // pooling operation.
      if (!RL->hasOneUse()) {
        continue;
      }

      auto *NPL = G.createPool(PL->getName(), RL->getInput(), PL->getMode(),
                               PL->getKernel(), PL->getStride(), PL->getPad());
      auto *NRL = G.createRELU(RL->getName(), NPL);
      PL->replaceAllUsesOfWith(NRL);
      continue;
    }
  } // For all nodes in the graph.
}

static void OptimizeBatchNorm(Graph &G) {
  auto &nodes = G.getNodes();

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

      BN->replaceAllUsesOfWith(CV);
    }
  } // For all nodes in the graph.
}

/// Concat nodes merging.
/// concat(dim1, concat(dim2, X, Y), Z) -> concat(dim1, X, Y, Z)
/// but only if dim1 == dim2
static void optimizeConcatNodes(Graph &G) {
  auto &nodes = G.getNodes();

  // For each node:
  for (auto const &node : nodes) {
    if (auto *CN = dyn_cast<ConcatNode>(node)) {
      auto Inputs = CN->getInputs();
      // Check if any of the inputs is a ConcatNode.
      llvm::SmallVector<Node *, 16> NewInputs;
      bool Changed = false;
      for (auto input : Inputs) {
        NewInputs.push_back(input);
        auto *CNI = dyn_cast<ConcatNode>(input);
        // Bail if it is not a ConcatNode or it is a concat node with a diffrent
        // dimension.
        if (!CNI || CNI->getDim() != CN->getDim())
          continue;

        Changed = true;
        // Replace current input by its own inputs, i.e. merge them into the
        // parent concat node.
        NewInputs.pop_back();
        NewInputs.append(CNI->getInputs().begin(), CNI->getInputs().end());
      }
      if (!Changed)
        continue;
      // Create a new Concat node.
      auto NewCN = G.createConcat(CN->getName(), NewInputs, CN->getDim());
      CN->replaceAllUsesOfWith(NewCN);
    }
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
  bool operator()(const Node *LHS, const Node *RHS) const {
    return LHS->isEqual(*RHS);
  }
};

/// This visitor is used to walk the the graph and
/// perform a common subexpression evaluation.
struct CSEVisitor : NodeWalker {
  // Mapping from the original node to its canonical representation under CSE.
  std::unordered_map<Node *, Node *, NodeHasher, NodeEq> CSENodes;
  // Set of visited nodes.
  std::unordered_set<Node *> VisitedNodes;

  /// This callback is called before visiting the children of \p N.
  void pre(Node *parent, Node *N) override {
    // Put the node into a visited set to make sure it is visited
    // only once.
    VisitedNodes.insert(N);
  }

  /// This callback is called after visiting the children of \p N.
  /// It means that all of its dependencies are processed already.
  void post(Node *parent, Node *N) override {
    // Try to find a node equivalent to the current one.
    auto FoundI = CSENodes.find(N);
    if (FoundI == CSENodes.end()) {
      // No node CSE-equivalent to the current one has been seen yet.
      // Remember this node, so that the next occurrence can be
      // replaced by this one.
      CSENodes.insert({N, N});
      assert(CSENodes.find(N) != CSENodes.end());
      return;
    }
    Node *FoundN = FoundI->second;
    // Bail if the equivalent node is the same node.
    if (FoundN == N)
      return;
    // Replace current node by a found node, which is
    // equivalent to it.
    assert(N->isEqual(*FoundN));
    N->replaceAllUsesOfWith(FoundN);
    // TODO: Erase N during CSE? If we don't do it here,
    // DCE will remove it later anyways.
  }

  /// Make sure that each node is processed only once.
  bool shouldVisit(Node *parent, Node *N) override {
    return VisitedNodes.count(N) == 0;
  }
};

} // namespace

/// Common Subexpression Elimination.
static void CSE(Graph &G) {
  CSEVisitor Visitor;

  // No need to perform CSE on variables because
  // all variables are distinct from each other.

  // Perform CSE on all nodes.
  //
  // TODO: Make sure that nodes are visited after nodes that dominate them.
  // This code may need to be updated if we allow for non-linear control flow
  // in the future.
  for (auto const &N : G.getNodes()) {
    N->visit(nullptr, &Visitor);
  }
}

void glow::optimize(Graph &G, CompilationMode mode) {
  // Sink transpose operations in an attempt to cancel them out.
  sinkCode(G);

  // Optimize the pooling operation.
  OptimizePool(G);

  // Perform Common Subexpression Elimination.
  CSE(G);

  // Perform Dead Code Elimination.
  DCE(G, mode);

  if (mode == CompilationMode::Infer) {
    // Merge batch normalization operations.
    OptimizeBatchNorm(G);
  }

  // Perform Common Subexpression Elimination.
  CSE(G);

  optimizeConcatNodes(G);

  // Perform Dead Code Elimination.
  DCE(G, mode);
}
