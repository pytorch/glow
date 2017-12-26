#ifndef GLOW_GRAPH_GRAPH_H
#define GLOW_GRAPH_GRAPH_H

#include "glow/Base/Type.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/ArrayRef.h"

#include <list>
#include <unordered_map>
#include <vector>

namespace glow {

using TypesList = std::list<Type>;
using NodesList = std::list<Node *>;
using VariablesList = std::list<Variable *>;
using VariableGradientsList = std::list<std::pair<Variable *, Variable *>>;
using UnsignedArrayRef = llvm::ArrayRef<size_t>;

/// Represents the compute graph.
class Graph final : public Named {
  /// A uniqued list of types in the module. Types in this list can be equated
  /// by comparing their addresses.
  TypesList types_{};
  /// A list of nodes that the graph owns.
  NodesList nodes_;
  /// A list of variables that the graph owns.
  VariablesList vars_;

  /// A list of (var, grad_var) pairs associating variables with their
  /// gradient variables.
  VariableGradientsList grads_;

  /// Unique index for producing unique names.
  size_t uniqueIdx_;

  /// \returns unique name with a prefix \p Name.
  std::string uniqueName(llvm::StringRef Name);

  /// Unique names defined by node \p N.
  void uniqueNames(Node *N);

public:
  Graph(llvm::StringRef Name = {}) : Named(Name), uniqueIdx_(1) {}

  ~Graph();

  /// Inserts the node \p N to the list of nodes, and returns the inserted node.
  template <class NodeTy> NodeTy *addNode(NodeTy *N) {
    uniqueNames(N);
    nodes_.push_back(N);
    return N;
  }

  /// Inserts the variable \p V to the list of variables.
  Variable *addVar(Variable *V) {
    uniqueNames(V);
    vars_.push_back(V);
    return V;
  }

  /// Return a pointer to a uniqued type \p t in the current module.
  TypeRef uniqueType(const Type &T);

  /// Return a pointer to a uniqued type \p t in the current module.
  TypeRef uniqueType(ElemKind elemTy, UnsignedArrayRef dims);

  /// Return the void type.
  TypeRef getVoidTy();

  /// @name High-level, operation-level IRBuilder.
  ///@{

  Variable *
  createVariable(TypeRef T, llvm::StringRef name,
                 Variable::InitKind initKind = Variable::InitKind::Broadcast,
                 float val = 0.0);

  Variable *
  createVariable(ElemKind T, UnsignedArrayRef dims, llvm::StringRef name,
                 Variable::InitKind initKind = Variable::InitKind::Broadcast,
                 float val = 0.0);

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              size_t depth, size_t kernel, size_t stride,
                              size_t pad);

  PoolNode *createPool(llvm::StringRef name, NodeValue input,
                       PoolNode::Mode mode, size_t kernel, size_t stride,
                       size_t pad);

  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, Variable *W,
                                           Variable *B, size_t outDepth);
  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, size_t outDepth);

  ReluNode *createRELU(llvm::StringRef name, NodeValue input);

  SigmoidNode *createSigmoid(llvm::StringRef name, NodeValue input);

  TanhNode *createTanh(llvm::StringRef name, NodeValue input);

  SoftMaxNode *createSoftMax(llvm::StringRef name, NodeValue input,
                             NodeValue selected);

  RegressionNode *createRegression(llvm::StringRef name, NodeValue input,
                                   NodeValue expected);

  ReshapeNode *createReshape(llvm::StringRef name, NodeValue input,
                             UnsignedArrayRef shape);

  TransposeNode *createTranspose(llvm::StringRef name, NodeValue input,
                                 llvm::ArrayRef<unsigned> shuffle);

  ConcatNode *createConcat(llvm::StringRef name, llvm::ArrayRef<Node *> inputs,
                           unsigned dimension);

  SliceNode *createSlice(llvm::StringRef name, NodeValue input,
                         UnsignedArrayRef begin, UnsignedArrayRef end);

  BatchNormalizationNode *createBatchNormalization(llvm::StringRef name,
                                                   NodeValue input,
                                                   size_t channelIdx = 0,
                                                   float epsilon = 1e-5,
                                                   float momentum = 0.9);

  BatchNormalizationNode *
  createBatchNormalization(llvm::StringRef name, NodeValue input,
                           NodeValue beta, NodeValue gamma, NodeValue mean,
                           NodeValue var, size_t channelIdx = 0,
                           float epsilon = 1e-5, float momentum = 0.9);

  LocalResponseNormalizationNode *createLocalResponseNormalization(
      llvm::StringRef name, NodeValue input, size_t halfWindowSize = 2,
      float alpha = 1e-4, float beta = 0.75, float k = 2.0);

  ArithmeticNode *createArithmetic(llvm::StringRef name, NodeValue LHS,
                                   NodeValue RHS, ArithmeticNode::Mode op);

  SelectNode *createSelect(llvm::StringRef name, NodeValue Cond, NodeValue LHS,
                           NodeValue RHS);

  SplatNode *createSplat(llvm::StringRef name, TypeRef ty, float value);

  BatchedMatMulNode *createBatchedMatMul(llvm::StringRef name, NodeValue batch,
                                         NodeValue filter);

  BatchedReduceNode *createBatchedReduce(llvm::StringRef name,
                                         BatchedReduceNode::Mode mode,
                                         NodeValue batch);

  BatchedArithmeticNode *
  createBatchedArithmetic(llvm::StringRef name,
                          BatchedArithmeticNode::Mode mode, NodeValue batch,
                          NodeValue sample);

  SaveNode *createSave(llvm::StringRef name, NodeValue input);
  SaveNode *createSave(llvm::StringRef name, NodeValue input, Variable *output);

  /// @}

  /// Erase a variable from the graph.
  void eraseVariable(Variable *N);
  void eraseVariable(VariablesList::iterator I);

  /// Erase a node from the graph.
  void eraseNode(Node *N);
  void eraseNode(NodesList::iterator I);

  /// Verify the correctness of the graph.
  void verify() const;

  /// Dumps the textual representation of the network.
  void dump() const;

  /// Dump a dotty graph that depicts the module.
  void dumpDAG();

  /// Dump a dotty graph that depicts the module.
  void dumpDAG(const char *dotFilename);

  /// \returns the list of nodes that the graph owns.
  NodesList &getNodes() { return nodes_; }

  /// \returns a pointer to the first variable with the name \p name or nullptr
  /// if no node has this name.
  Variable *getVariableByName(llvm::StringRef name);

  /// \returns the list of variables that the graph owns.
  VariablesList &getVars() { return vars_; }

  /// Associates a gradient variable \p GradV with the variable \p V.
  void addGradientVariable(Variable *V, Variable *GradV);

  /// \returns a gradient variable associated with \p V.
  /// Returns nullptr if there is no gradient variable
  /// related to this variable.
  Variable *getGradientVariable(Variable *V);
};

struct TrainingConfig;

/// Mutate the inference graph and turn it into a training graph by inserting
/// training (gradient calculation) nodes.
void generateGradientNodes(Graph &G, TrainingConfig &config,
                           CompilationMode mode);

/// \returns a variable that accumulates the gradients that update \p V.
/// Given the variable \p V, find the SGD node that trains it and record the
/// gradients that flow into V.
Variable *generateRecordGradientNode(Graph &G, Variable *V);

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
