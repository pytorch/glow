#ifndef GLOW_GRAPH_GRAPH_H
#define GLOW_GRAPH_GRAPH_H

#include "glow/Base/Type.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

#include <list>
#include <unordered_map>
#include <vector>

namespace glow {

using TypesList = std::list<Type>;
using NodesList = std::list<Node *>;
using FunctionList = std::list<Function *>;
using VariablesList = std::list<Variable *>;
using UnsignedArrayRef = llvm::ArrayRef<size_t>;

class Module final {
  /// Stores the functions in the module.
  FunctionList functions_;
  /// A uniqued list of types. Types in this list can be equated by comparing
  /// their addresses.
  TypesList types_{};

  /// A list of variables that the Module owns.
  VariablesList vars_;

  /// Unique index for producing unique names.
  size_t uniqueIdx_{1};

  /// \returns unique name with a prefix \p Name.
  std::string uniqueName(llvm::StringRef Name);

public:
  Module() = default;

  ~Module();

  /// Inserts the variable \p V to the list of variables.
  Variable *addVar(Variable *V) {
    assignUniqueName(V);
    vars_.push_back(V);
    return V;
  }

  /// Assign unique name to node \p N.
  void assignUniqueName(Node *N);

  /// Return a pointer to a uniqued type \p T.
  TypeRef uniqueType(const Type &T);

  /// Return a pointer to a uniqued type \p T.
  TypeRef uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims);

  /// Return a pointer to a uniqued type \p T.
  TypeRef uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims, float scale,
                     int32_t offset);

  /// Return a pointer to a uniqued type \p T.
  /// The new type is identical to \p T, with a new shape \p dims.
  TypeRef uniqueTypeWithNewShape(TypeRef T, llvm::ArrayRef<size_t> dims);

  /// Return the void type.
  TypeRef getVoidTy();

  /// \returns True if a function by the name \p name exists in the module.
  bool hasFunction(llvm::StringRef name);
  /// \returns the function with the name \p name, or nullptr if the function
  /// does not exist.
  Function *getFunction(llvm::StringRef name);
  /// \returns a new function with the name \p name.
  Function *createFunction(llvm::StringRef name);
  /// \returns the list of Functions that the Module owns.
  FunctionList &getFunctions() { return functions_; }

  const FunctionList &getFunctions() const { return functions_; }

  /// Erase the variable \p N from the Module.
  void eraseVariable(Variable *N);

  /// Erase the variable \p I from the Module.
  void eraseVariable(VariablesList::iterator I);

  /// \returns a pointer to the first variable with the name \p name or nullptr
  /// if no node has this name.
  Variable *getVariableByName(llvm::StringRef name);

  /// \returns the list of variables that the Module owns.
  VariablesList &getVars() { return vars_; }

  const VariablesList &getVars() const { return vars_; }

  /// @name High-level Variable builders.
  ///@{

  Variable *createVariable(
      TypeRef T, llvm::StringRef name,
      Variable::VisibilityKind visibility = Variable::VisibilityKind::Private,
      Variable::TrainKind train = Variable::TrainKind::Broadcast,
      float val = 0.0);

  Variable *createVariable(
      ElemKind T, llvm::ArrayRef<size_t> dims, llvm::StringRef name,
      Variable::VisibilityKind visibility = Variable::VisibilityKind::Private,
      Variable::TrainKind train = Variable::TrainKind::Broadcast,
      float val = 0.0);

  Variable *createVariable(
      ElemKind T, llvm::ArrayRef<size_t> dims, float scale, int32_t offset,
      llvm::StringRef name,
      Variable::VisibilityKind visibility = Variable::VisibilityKind::Private,
      Variable::TrainKind train = Variable::TrainKind::Broadcast,
      float val = 0.0);
  ///@}

  /// Verify the correctness of the Module.
  void verify() const;

  /// Dumps the textual representation of the network.
  void dump() const;

  /// Dump a dotty graph that depicts the Module.
  void dumpDAG();

  /// Dump a dotty graph that depicts the Module.
  void dumpDAG(const char *dotFilename);
};

/// Represents the compute graph.
class Function final : public Named {
  /// A list of nodes that the Function owns.
  NodesList nodes_;

  /// A reference to the owner of the function.
  Module *parent_;

public:
  Function(Module *parent, llvm::StringRef Name = {})
      : Named(Name), parent_(parent) {}

  ~Function();

  Module *getParent() { return parent_; }
  const Module *getParent() const { return parent_; }

  /// Inserts the node \p N to the list of nodes, and returns the inserted node.
  template <class NodeTy> NodeTy *addNode(NodeTy *N) {
    getParent()->assignUniqueName(N);
    nodes_.push_back(N);
    return N;
  }

  /// @name High-level, operation-level IRBuilder.
  ///@{

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              size_t depth, size_t kernel, size_t stride,
                              size_t pad);

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              NodeValue filter, NodeValue bias, TypeRef outTy,
                              size_t depth, size_t kernel, size_t stride,
                              size_t pad);

  PoolMaxNode *createPoolMax(llvm::StringRef name, NodeValue input,
                             size_t kernel, size_t stride, size_t pad);

  PoolAvgNode *createPoolAvg(llvm::StringRef name, NodeValue input,
                             size_t kernel, size_t stride, size_t pad);

  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, Variable *W,
                                           Variable *B);

  /// Create a fully connected node with the specified output type.
  /// Note, outputDepth is infered based on the output type.
  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, Node *W, Node *B,
                                           TypeRef outTy);

  /// Create a fully connected node with the given \p name, \p input and \p
  /// output depth. Trainable weight and bias variables are created implicitly.
  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, size_t outDepth);

  ReluNode *createRELU(llvm::StringRef name, NodeValue input);

  SigmoidNode *createSigmoid(llvm::StringRef name, NodeValue input);

  TanhNode *createTanh(llvm::StringRef name, NodeValue input);

  SoftMaxNode *createSoftMax(llvm::StringRef name, NodeValue input,
                             NodeValue selected);

  CrossEntropyLossNode *createCrossEntropyLoss(llvm::StringRef name,
                                               NodeValue input,
                                               NodeValue labels);

  RegressionNode *createRegression(llvm::StringRef name, NodeValue input,
                                   NodeValue expected);

  ReshapeNode *createReshape(llvm::StringRef name, NodeValue input,
                             UnsignedArrayRef shape);

  TransposeNode *createTranspose(llvm::StringRef name, NodeValue input,
                                 llvm::ArrayRef<unsigned> shuffle);

  BroadcastNode *createBroadcast(llvm::StringRef name, NodeValue input,
                                 UnsignedArrayRef shape, unsigned axis);

  /// Create concat node which concatenates input tensors along \p dimension.
  ConcatNode *createConcat(llvm::StringRef name, llvm::ArrayRef<Node *> inputs,
                           unsigned dimension);

  /// Create concat node with the given return type \p outTy.
  ConcatNode *createConcat(llvm::StringRef name, llvm::ArrayRef<Node *> inputs,
                           unsigned dimension, TypeRef outTy);

  SliceNode *createSlice(llvm::StringRef name, NodeValue input,
                         UnsignedArrayRef begin, UnsignedArrayRef end);

  /// Create a slice node with the given starting point for each dimension.
  /// End points will be calculated based on the output type during execution.
  SliceNode *createSlice(llvm::StringRef name, NodeValue input,
                         llvm::ArrayRef<size_t> start, TypeRef outTy);

  /// Shuffles dimension number \p kernel. Suppose original size is D. It will
  /// be represented as groupX(D/group) matrix, transposed and concatenated back
  /// to size D. For example, shuffle of {1, 2, 3, 4, 5, 6} with \p group = 2 is
  /// {1, 4, 2, 5, 3, 6}
  Node *createChannelShuffle(llvm::StringRef name, NodeValue input,
                             size_t group, size_t kernel);

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

#define ARITHMETIC_FUN_DECL(NODE_NAME_)                                        \
  NODE_NAME_##Node *create##NODE_NAME_(llvm::StringRef name, NodeValue LHS,    \
                                       NodeValue RHS);                         \
  NODE_NAME_##Node *create##NODE_NAME_(llvm::StringRef name, TypeRef Ty,       \
                                       NodeValue LHS, NodeValue RHS);
  ARITHMETIC_FUN_DECL(Add);
  ARITHMETIC_FUN_DECL(Mul);
  ARITHMETIC_FUN_DECL(Sub);
  ARITHMETIC_FUN_DECL(Div);
  ARITHMETIC_FUN_DECL(Max);
  ARITHMETIC_FUN_DECL(Min);
  ARITHMETIC_FUN_DECL(CmpLTE);
#undef ARITHMETIC_FUN_DECL

  PowNode *createPow(llvm::StringRef name, NodeValue Base, float exp);

  SelectNode *createSelect(llvm::StringRef name, NodeValue Cond, NodeValue LHS,
                           NodeValue RHS);

  SplatNode *createSplat(llvm::StringRef name, TypeRef ty, float value);

  MatMulNode *createMatMul(llvm::StringRef name, NodeValue lhs, NodeValue rhs);

  MatMulNode *createMatMul(llvm::StringRef name, TypeRef outTy, NodeValue lhs,
                           NodeValue rhs);

  BatchedReduceAddNode *createBatchedReduceAdd(llvm::StringRef name,
                                               NodeValue batch);

  BatchedAddNode *createBatchedAdd(llvm::StringRef name, NodeValue batch,
                                   NodeValue sample);

  BatchedAddNode *createBatchedAdd(llvm::StringRef name, TypeRef outTy,
                                   NodeValue batch, NodeValue sample);

  SaveNode *createSave(llvm::StringRef name, NodeValue input);
  SaveNode *createSave(llvm::StringRef name, NodeValue input, Variable *output);
  /// Create quantization profile node named \p name for the output tensor from
  /// \p input. Capture observed node name in quantization profile node as
  /// original node can be replaced during lowering phase.
  QuantizationProfileNode *createQuantizationProfile(llvm::StringRef name,
                                                     NodeValue input);

  TopKNode *createTopK(llvm::StringRef name, NodeValue input, size_t k);

  GatherNode *createGather(llvm::StringRef name, NodeValue data,
                           NodeValue indices);

  /// Create quantization node which transforms floating point tensor to a
  /// quantized one with given Scale and Offset. Scale and Offset params are
  /// part of the \p outTy.
  QuantizeNode *createQuantize(llvm::StringRef name, NodeValue input,
                               TypeRef outTy);

  /// Create dequantization node which transforms quantized tensor to a
  /// floating point one with given Scale and Offset. Scale and Offset params
  /// are part of the \p input.
  DequantizeNode *createDequantize(llvm::StringRef name, NodeValue input);

  /// Create transformation for quantized tensors to rescale based on the new
  /// Scale and Offset.
  RescaleQuantizedNode *createRescaleQuantized(llvm::StringRef name,
                                               NodeValue input, TypeRef outTy);

  /// Create an unrolled single-layer Simple RNN cell with \p hiddenSize
  /// dimensionality of the hidden state and \p outputSize dimensionality of the
  /// output state. \p inputs define the input for the cell at each time step
  /// and the number of time steps is equal to the size of the \p inputs. The
  /// names of the created variables are prefixed by \p namePrefix.
  /// The output variables are written to \p outputs, they represent the
  /// activations of the output layer, unrolled over time.
  // The dimensionality of the output variables is \p batchSize x \p outputSize.
  void createSimpleRNN(llvm::StringRef namePrefix,
                       const llvm::ArrayRef<Node *> inputs, unsigned batchSize,
                       unsigned hiddenSize, unsigned outputSize,
                       std::vector<Node *> &outputs);

  /// Create an unrolled single-layer GRU cell with \p hiddenSize
  /// dimensionality of the hidden state and \p outputSize dimensionality of the
  /// output state. \p inputs define the input for the cell at each time step
  /// and the number of time steps is equal to the size of the \p inputs. The
  /// names of the created variables are prefixed by \p namePrefix.
  /// The output variables are written to \p outputs, they represent the
  /// activation of the output layer, unrolled over time.
  // The dimensionality of the output variables is \p batchSize x \p outputSize.
  void createGRU(llvm::StringRef namePrefix,
                 const llvm::ArrayRef<Node *> inputs, unsigned batchSize,
                 unsigned hiddenSize, unsigned outputSize,
                 std::vector<Node *> &outputs);

  /// Create an unrolled single-layer LSTM cell with \p hiddenSize
  /// dimensionality of the hidden state and \p outputSize dimensionality of the
  /// output state. \p inputs define the input for the cell at each time step
  /// and the number of time steps is equal to the size of the \p inputs. The
  /// names of the created variables are prefixed by \p namePrefix.
  /// The output variables are written to \p outputs, they represent the
  /// activation of the output layer, unrolled over time.
  // The dimensionality of the output variables is \p batchSize x \p outputSize.
  void createLSTM(llvm::StringRef namePrefix,
                  const llvm::ArrayRef<Node *> inputs, unsigned batchSize,
                  unsigned hiddenSize, unsigned outputSize,
                  std::vector<Node *> &outputs);

  /// @}

  /// Erase the node \p N from the Function.
  void eraseNode(Node *N);

  /// Erase the node \p I from the Function.
  void eraseNode(NodesList::iterator I);

  /// Clone the current function into a new function with the name \p newName.
  /// If \p map is non-null then the procedure records the mapping between the
  /// old node to the new node in \p map.
  /// \returns a new function that is a copy of the current function.
  Function *clone(llvm::StringRef newName,
                  llvm::DenseMap<Node *, Node *> *map = nullptr);

  /// Verify the correctness of the Function.
  void verify() const;

  /// Dumps the textual representation of the network.
  void dump() const;

  /// Dump a dotty graph that depicts the function.
  void dumpDAG();

  /// Dump a dotty graph that depicts the function.
  void dumpDAG(const char *dotFilename);

  /// \returns the list of nodes that the Function owns.
  NodesList &getNodes() { return nodes_; }

  const NodesList &getNodes() const { return nodes_; }
};

struct TrainingConfig;

using VariableGradientsList = std::list<std::pair<Variable *, Variable *>>;

/// Create a new Function that 'trains' the input Function. We differentiate the
/// nodes and insert code to update the weights based on the \p config
/// parameters.
/// If \p varGrads is set then instead of inserting code to update the weights,
/// the procedure adds code to record the last gradient value: a list of
/// (var, grad_var) pairs associating variables with their gradient variables.
/// This feature is used by the gradient-check unit tests.
/// \returns a new function with the name \p newFuncName.
Function *differentiate(Function *F, TrainingConfig &config,
                        llvm::StringRef newFuncName = "",
                        VariableGradientsList *varGrads = nullptr);

/// \returns a variable that accumulates the gradients that update \p V.
/// Given the variable \p V, find the SGD node that trains it and record the
/// gradients that flow into V.
Variable *generateRecordGradientNode(Function &G, Variable *V);

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
