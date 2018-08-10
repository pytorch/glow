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
#ifndef GLOW_GRAPH_GRAPH_H
#define GLOW_GRAPH_GRAPH_H

#include "glow/Base/Type.h"
#include "glow/Graph/Nodes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

#include <list>
#include <vector>

namespace glow {

/// List of Types.
using TypesList = std::list<Type>;
/// Intrusive list of Nodes.
using NodesList = llvm::iplist<glow::Node>;
/// List of pointers to Nodes. The nodes are not owned by the list.
using NodesPtrList = std::list<glow::Node *>;
/// List of Functions.
using FunctionList = std::list<Function *>;
/// List of Variables.
using VariablesList = std::list<Variable *>;
using UnsignedArrayRef = llvm::ArrayRef<size_t>;

class Module final {
  /// Stores the functions in the module.
  FunctionList functions_;
  /// A uniqued list of types. Types in this list can be equated by comparing
  /// their addresses.
  TypesList types_{};
  /// Stores a list of unique variable names that were used by the module at
  /// some point.
  llvm::StringSet<> uniqueVariableNames_{};
  /// A list of variables that the Module owns.
  VariablesList vars_;
  /// Deterministic PRNG used to initialize weights in this module.
  PseudoRNG PRNG_;

public:
  Module() = default;

  ~Module();

  /// \returns unique legal name that's based on the string \p name. Legal
  /// names are legal C identifiers in the form: "[a-zA-Z_][a-zA-Z0-9_]*".
  static llvm::StringRef uniqueName(llvm::StringRef name,
                                    llvm::StringSet<> &stringTable);

  /// Inserts the variable \p V to the list of variables.
  Variable *addVar(Variable *V) {
    V->setName(uniqueName(V->getName(), uniqueVariableNames_));
    vars_.push_back(V);
    return V;
  }

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

  Variable *createVariable(TypeRef T, llvm::StringRef name,
                           VisibilityKind visibility = VisibilityKind::Private,
                           bool isTrainable = true);

  Variable *createVariable(ElemKind T, llvm::ArrayRef<size_t> dims,
                           llvm::StringRef name,
                           VisibilityKind visibility = VisibilityKind::Private,
                           bool isTrainable = true);

  Variable *createVariable(ElemKind T, llvm::ArrayRef<size_t> dims, float scale,
                           int32_t offset, llvm::StringRef name,
                           VisibilityKind visibility = VisibilityKind::Private,
                           bool isTrainable = true);

  Variable *createVariable(llvm::StringRef name, const Tensor &tensor,
                           VisibilityKind visibility = VisibilityKind::Private,
                           bool isTrainable = true);

  ///@}

  /// Verify the correctness of the Module.
  void verify() const;

  /// Get the pseudo-random number generator used by this module.
  PseudoRNG &getPRNG() { return PRNG_; }

  /// Dumps the textual representation of the network.
  void dump() const;

  /// Dump a dotty graph that depicts the Module.
  void dumpDAG();

  /// Dump a dotty graph that depicts the Module.
  void dumpDAG(llvm::StringRef dotFilename);

  /// Dump a dotty graph that depicts the Module.
  void dumpDAG(const char *dotFilename);

  /// Erase all of the functions from the module.
  void eraseFunctions();
};

/// Represents the compute graph.
class Function final : public Named {
  /// A list of nodes that the Function owns.
  NodesList nodes_;

  /// Stores a list of unique node names that were used by the module at some
  /// point.
  llvm::StringSet<> uniqueNodeNames_{};

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
    N->setName(Module::uniqueName(N->getName(), uniqueNodeNames_));
    nodes_.push_back(N);
    return N;
  }

  /// Get the pseudo-random number generator used by this module.
  PseudoRNG &getPRNG() { return getParent()->getPRNG(); }

  /// @name High-level, operation-level IRBuilder.
  ///@{

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              size_t depth, llvm::ArrayRef<size_t> kernels,
                              llvm::ArrayRef<size_t> strides,
                              llvm::ArrayRef<size_t> pads, size_t group);

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              NodeValue filter, NodeValue bias, TypeRef outTy,
                              llvm::ArrayRef<size_t> kernels,
                              llvm::ArrayRef<size_t> strides,
                              llvm::ArrayRef<size_t> pads, size_t group);

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              size_t depth, size_t kernel, size_t stride,
                              size_t pad, size_t group);

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              NodeValue filter, NodeValue bias, TypeRef outTy,
                              size_t kernel, size_t stride, size_t pad,
                              size_t group);

  MaxPoolNode *createMaxPool(llvm::StringRef name, NodeValue input,
                             llvm::ArrayRef<size_t> kernels,
                             llvm::ArrayRef<size_t> strides,
                             llvm::ArrayRef<size_t> pads);

  MaxPoolNode *createMaxPool(llvm::StringRef name, NodeValue input,
                             size_t kernel, size_t stride, size_t pad);

  AvgPoolNode *createAvgPool(llvm::StringRef name, NodeValue input,
                             llvm::ArrayRef<size_t> kernels,
                             llvm::ArrayRef<size_t> strides,
                             llvm::ArrayRef<size_t> pads);

  AvgPoolNode *createAvgPool(llvm::StringRef name, NodeValue input,
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

  LogNode *createLog(llvm::StringRef name, NodeValue input);

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

  /// Create a series of nodes that implement a Broadcast operation. The \p
  /// input Tensor is broadcasted based on \p newShape and along the \p axis,
  /// which defines the offset from the leading dimension under which
  /// broadcasting is performed.
  Node *createBroadcast(llvm::StringRef name, NodeValue input,
                        UnsignedArrayRef newShape, unsigned axis);

  /// Create concat node which concatenates input tensors along \p dimension.
  ConcatNode *createConcat(llvm::StringRef name,
                           llvm::ArrayRef<NodeValue> inputs,
                           unsigned dimension);

  /// Create concat node with the given return type \p outTy.
  ConcatNode *createConcat(llvm::StringRef name,
                           llvm::ArrayRef<NodeValue> inputs, unsigned dimension,
                           TypeRef outTy);

  /// Create an insert tensor node that implements a tile with provided \p name,
  /// \p input, \p tiles, and \p axis. For example, an input tensor {{1,2,3,4}}
  /// of dimension 1x4 with tiles = 2 and axis = 0 would result in an output
  /// tensor {{1,2,3,4}, {1,2,3,4}} of dimension 2x4.
  InsertTensorNode *createTile(llvm::StringRef name, NodeValue input,
                               unsigned tiles, unsigned axis);

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

  /// Removes single-dimensional entries from the shape of a tensor. The
  /// parameter \p axes is a list of positive integers, indicating the
  /// dimensions to squeeze. Impelmented as a single ReshapeNode. This is the
  /// opposite of ExpandDims.
  /// https://github.com/onnx/onnx/blob/master/docs/Operators.md#squeeze
  ReshapeNode *createSqueeze(llvm::StringRef name, NodeValue input,
                             llvm::ArrayRef<size_t> axes);

  /// Add single-dimensional entries to the shape of the \p input tensor at
  /// locations in \p axes. \p axes is listed as seen in the output tensor.
  /// Implemented as a single ReshapeNode. This is the opposite of Squeeze.
  ReshapeNode *createExpandDims(llvm::StringRef name, NodeValue input,
                                llvm::ArrayRef<size_t> axes);

  /// Flattens the input tensor into a 2D matrix. If input tensor has shape
  /// (d_0, d_1, ... d_n) then the output will have shape:
  /// (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X d_n).
  ReshapeNode *createFlatten(llvm::StringRef name, NodeValue input,
                             size_t axis);

  /// Create \p outputNum slice nodes of \p input. Slices happen along dimension
  /// number \p axis. Array \p split defines lengths of slices. If \p split is
  /// empty, \p input is split to equal sized parts.
  void createSplit(llvm::StringRef name, NodeValue input, size_t outputNum,
                   size_t axis, llvm::ArrayRef<size_t> split,
                   std::vector<Node *> &outputs);

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
  ARITHMETIC_FUN_DECL(CmpEQ);
  ARITHMETIC_FUN_DECL(Pow);
#undef ARITHMETIC_FUN_DECL

  PowNode *createPow(llvm::StringRef name, NodeValue base, float exp);

  SelectNode *createSelect(llvm::StringRef name, NodeValue Cond, NodeValue LHS,
                           NodeValue RHS);

  SelectNode *createSelect(llvm::StringRef name, TypeRef outTy, NodeValue Cond,
                           NodeValue LHS, NodeValue RHS);

  SplatNode *createSplat(llvm::StringRef name, TypeRef ty, float value);

  MatMulNode *createMatMul(llvm::StringRef name, NodeValue lhs, NodeValue rhs);

  MatMulNode *createMatMul(llvm::StringRef name, TypeRef outTy, NodeValue lhs,
                           NodeValue rhs);

  /// \p lhs is a 3d matrix, where the leading dimension is the batch size. \p
  /// rhs is a 2d matrix, which every batch from \p lhs (a 2d matrix) is
  /// multiplied by. This is implemented via reshaping \p lhs to be a 2d matrix,
  /// multiplying by \p rhs, and then reshaping the result back to 3d.
  Node *createBatchMatMul(llvm::StringRef name, NodeValue lhs, NodeValue rhs);

  BatchedReduceAddNode *createBatchedReduceAdd(llvm::StringRef name,
                                               NodeValue batch, size_t axis);

  BatchedReduceAddNode *createBatchedReduceAdd(llvm::StringRef name,
                                               TypeRef outTy, NodeValue batch,
                                               size_t axis);

  /// Implements a batched reduce mean of the \p batch on the provided \p axis
  /// with output type \p outTy with three nodes: a BatchedReduceAdd followed by
  /// a DivNode with a SplatNode of the length of the \p axis
  /// dimension. \returns the final DivNode.
  DivNode *createBatchedReduceMean(llvm::StringRef name, TypeRef outTy,
                                   NodeValue batch, size_t axis);

  /// Implements a batched reduce mean of the \p batch on the provided \p axis
  /// with three nodes: a BatchedReduceAdd followed by a DivNode with a
  /// SplatNode of the length of the \p axis dimension. \returns the final
  /// DivNode.
  DivNode *createBatchedReduceMean(llvm::StringRef name, NodeValue batch,
                                   size_t axis);

  BatchedAddNode *createBatchedAdd(llvm::StringRef name, NodeValue batch,
                                   NodeValue sample);

  BatchedAddNode *createBatchedAdd(llvm::StringRef name, TypeRef outTy,
                                   NodeValue batch, NodeValue sample);

  /// Create a node, performing SparseLengthsSum operation:
  /// Gathers slices of the outer-most dimension of Data indexed by Indices
  /// vector, and then accumulates them into len(Lengths) entries:
  /// first Lengths[0] slices are aggregated to Result[0], next Lengths[1]
  /// slices are aggregated to Result[1], etc. I.e. sum(Lengths) must be equal
  /// to len(Indices).
  SparseLengthsWeightedSumNode *createSparseLengthsSum(llvm::StringRef name,
                                                       NodeValue data,
                                                       NodeValue indices,
                                                       NodeValue lengths);

  /// Same as SparseLengthsSum, but i-th slice is multiplied by weights[i].
  /// len(weights) must be equal to len(indices).
  SparseLengthsWeightedSumNode *
  createSparseLengthsWeightedSum(llvm::StringRef name, NodeValue data,
                                 NodeValue weights, NodeValue indices,
                                 NodeValue lengths);

  SaveNode *createSave(llvm::StringRef name, NodeValue input);
  SaveNode *createSave(llvm::StringRef name, NodeValue input, Variable *output);

  /// Create quantization profile node named \p name for the output tensor from
  /// \p input. Capture observed node name in quantization profile node as
  /// original node can be replaced during lowering phase.
  QuantizationProfileNode *createQuantizationProfile(llvm::StringRef name,
                                                     NodeValue input);

  /// Create lookup table for mapping between quantized numbers.
  /// \p input and \p outTy must have quantized type.
  /// Table contains all numbers from the quantized range, e.g.,
  /// 256 entries for int8. Position 0 in the \p initValues
  /// corresponds to the -128 input number, position 255 to 127.
  IntLookupTableNode *createIntLookupTable(llvm::StringRef name,
                                           NodeValue input,
                                           llvm::ArrayRef<int8_t> initValues,
                                           TypeRef outTy);

  /// Create quantized tanh.
  IntLookupTableNode *createIntTanh(llvm::StringRef name, NodeValue input,
                                    TypeRef outTy);

  /// Create quantized sigmoid.
  IntLookupTableNode *createIntSigmoid(llvm::StringRef name, NodeValue input,
                                       TypeRef outTy);

  TopKNode *createTopK(llvm::StringRef name, NodeValue input, size_t k);

  /// Gathers entries of the outer-most dimension of \p data indexed by
  /// \p indices, and concatenates them. A non-zero \p batchDims specifies the
  /// batch, and the result is the concatenation of the operation on each sample
  /// in the batch.
  GatherNode *createGather(llvm::StringRef name, NodeValue data,
                           NodeValue indices, unsigned batchDims = 0);

  /// Copies each slice from \p slices into \p data at the corresponding index
  /// in \p indices, and \returns this new version of data. For example, given
  /// input data {{1,2},{3,4},{5,6}}, slices {{-3,-4}}, and indices {1}, the
  /// result is {{1,2},{-3,-4},{5,6}}.
  ScatterAssignNode *createScatterAssign(llvm::StringRef name, NodeValue data,
                                         NodeValue indices, NodeValue slices);

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

  /// Create a series of nodes that implement a weighted sum. \p data and \p
  /// weights should have the same number of elements. The nodes in \p weights
  /// should all be of size 1. Each node d_i in \p data is element-wise
  /// multiplied by the corresponding weight value w_i found in \p weights,
  /// broadcasted to the same shape as d_i, and resulting in r_i. All r_i are
  /// element-wise summed, and the final add node in this sum is returned.
  Node *createWeightedSum(llvm::StringRef name, llvm::ArrayRef<NodeValue> data,
                          llvm::ArrayRef<NodeValue> weights);

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
                       std::vector<NodeValue> &outputs);

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
                 std::vector<NodeValue> &outputs);

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
                  std::vector<NodeValue> &outputs);

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
  void dumpDAG(llvm::StringRef dotFilename);

  /// Dump a dotty graph that depicts the function.
  void dumpDAG(const char *dotFilename);

  /// \returns the list of nodes that the Function owns.
  NodesList &getNodes() { return nodes_; }

  /// \returns a node with the name \p name or nullptr if no node was found.
  Node *getNodeByName(llvm::StringRef name);

  const NodesList &getNodes() const { return nodes_; }

  /// \returns pointer to the class member for the nodes list.
  static NodesList Function::*getNodesMemberPtr() { return &Function::nodes_; }
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
Function *differentiate(Function *F, const TrainingConfig &config,
                        llvm::StringRef newFuncName = "",
                        VariableGradientsList *varGrads = nullptr);

/// \returns a variable that accumulates the gradients that update \p V.
/// Given the variable \p V, find the SGD node that trains it and record the
/// gradients that flow into V.
Variable *generateRecordGradientNode(Function &G, Variable *V);

/// Helper vectors for common transpose shuffles.
#define NCHW2NHWC                                                              \
  { 0u, 2u, 3u, 1u }
#define NHWC2NCHW                                                              \
  { 0u, 3u, 1u, 2u }

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
