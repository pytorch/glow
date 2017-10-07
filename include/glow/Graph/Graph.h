#ifndef GLOW_GRAPH_GRAPH_H
#define GLOW_GRAPH_GRAPH_H

#include "glow/IR/Instrs.h"

#include "llvm/ADT/ArrayRef.h"

#include <unordered_map>
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
class ReturnNode;

/// Represents the compute graph.
class Graph final {
  /// A list of nodes that the graph owns.
  std::vector<Node *> nodes_;
  /// A list of variables that the graph owns.
  std::vector<Variable *> vars_;
  /// A reference to the low-level IR module.
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
  /// Holds the mapping between graph nodes to IR variables.
  using NodeToInstrTy = std::unordered_map<Node *, Value *>;

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

  ConvolutionNode *createConv(llvm::StringRef name, Node *input, size_t depth,
                              size_t kernel, size_t stride, size_t pad);

  PoolNode *createPool(llvm::StringRef name, Node *input, PoolInst::OpKind kind,
                       size_t kernel, size_t stride, size_t pad);

  FullyConnectedNode *createFullyConnected(llvm::StringRef name, Node *input,
                                           size_t outDepth);

  ReluNode *createRELU(llvm::StringRef name, Node *input);

  SigmoidNode *createSigmoid(llvm::StringRef name, Node *input);

  TanhNode *createTanh(llvm::StringRef name, Node *input);

  SoftMaxNode *createSoftMax(llvm::StringRef name, Node *input, Node *selected);

  RegressionNode *createRegression(llvm::StringRef name, Node *input,
                                   Node *expected);

  ReshapeNode *createReshape(llvm::StringRef name, Node *input,
                             llvm::ArrayRef<size_t> shape);

  TransposeNode *createTranspose(llvm::StringRef name, Node *input,
                                 llvm::ArrayRef<unsigned> shuffle);

  ConcatNode *createConcat(llvm::StringRef name, llvm::ArrayRef<Node *> inputs,
                           unsigned dimension);

  BatchNormalizationNode *createBatchNormalization(llvm::StringRef name,
                                                   Node *input,
                                                   size_t channelIdx = 0,
                                                   float epsilon = 1e-5,
                                                   float momentum = 0.9);

  LocalResponseNormalizationNode *createLocalResponseNormalization(
      llvm::StringRef name, Node *input, size_t halfWindowSize = 2,
      float alpha = 1e-4, float beta = 0.75, float k = 2.0);

  ArithmeticNode *createArithmetic(llvm::StringRef name, Node *LHS, Node *RHS,
                                   ArithmeticInst::OpKind op);

  ReturnNode *createReturn(llvm::StringRef name, Node *input);

  /// @}

  /// Generate IR from the nodes in the graph into the module.
  NodeToInstrTy generateIR();

  /// Dumps the textual representation of the network.
  void dump();

  /// Dump a dotty graph that depicts the module.
  void dumpDAG();
};

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
