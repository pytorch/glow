#ifndef GLOW_GRAPH_GRAPH_H
#define GLOW_GRAPH_GRAPH_H

#include "glow/IR/Instrs.h"

#include "llvm/ADT/ArrayRef.h"

#include <vector>

namespace glow {

class Node;
class Variable;
class ConvolutionNode;
class PoolNode;
class FullyConnectedNode;
class ReluNode;
class SigmoidNode;
class TanhNode;
class SoftMaxNode;
class RegressionNode;
class ReshapeNode;
class TransposeNode;
class ConcatNode;
class BatchNormalizationNode;
class LocalResponseNormalizationNode;
class ArithmeticNode;

/// Represents the compute graph.
class Graph final {
  /// A list of nodes that the graph owns.
  std::vector<Node *> nodes_;
  /// A list of variables.
  std::vector<Variable *> vars_;

  Module &M_;

  /// Inserts the node \p N to the list of nodes, and returns the inserted node.
  template <class NodeTy> NodeTy *addNode(NodeTy *N) {
    nodes_.push_back(N);
    return N;
  }

  /// Inserts the variable \p V to the list of variables.
  Variable *addVar(Variable *V) {
    vars_.push_back(V);
    return V;
  }

public:
  Graph(Module &M) : M_(M) {}
  ~Graph();

  /// @name High-level, operation-level IRBuilder.
  ///@{

  Variable *
  createVariable(TypeRef T, llvm::StringRef name,
                 WeightVar::InitKind initKind = WeightVar::InitKind::Broadcast,
                 float val = 0.0);

  Variable *
  createVariable(ElemKind T, llvm::ArrayRef<size_t> dims, llvm::StringRef name,
                 WeightVar::InitKind initKind = WeightVar::InitKind::Broadcast,
                 float val = 0.0);

  ConvolutionNode *createConv(Node *input, size_t depth, size_t kernel,
                              size_t stride, size_t pad);

  PoolNode *createPool(Node *input, PoolInst::OpKind kind, size_t kernel,
                       size_t stride, size_t pad);

  FullyConnectedNode *createFullyConnected(Node *input, size_t outDepth);

  ReluNode *createRELU(Node *input);

  SigmoidNode *createSigmoid(Node *input);

  TanhNode *createTanh(Node *input);

  SoftMaxNode *createSoftMax(Node *input, Node *selected);

  RegressionNode *createRegression(Node *input, Node *expected);

  ReshapeNode *createReshape(Node *input, llvm::ArrayRef<size_t> shape);

  TransposeNode *createTranspose(Node *input, llvm::ArrayRef<unsigned> shuffle);

  ConcatNode *createConcat(llvm::ArrayRef<Node *> inputs, unsigned dimension);

  BatchNormalizationNode *createBatchNormalization(Node *input,
                                                   size_t channelIdx = 0,
                                                   float epsilon = 1e-5,
                                                   float momentum = 0.9);

  LocalResponseNormalizationNode *
  createLocalResponseNormalization(Node *input, size_t halfWindowSize = 2,
                                   float alpha = 1e-4, float beta = 0.75,
                                   float k = 2.0);

  ArithmeticNode *createArithmetic(Node *LHS, Node *RHS,
                                   ArithmeticInst::OpKind op);
};

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
