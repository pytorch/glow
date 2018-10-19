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
class Context;

/// List of Types.
using TypesList = std::list<Type>;
/// Intrusive list of Nodes.
using NodesList = llvm::iplist<glow::Node>;
/// List of pointers to Nodes. The nodes are not owned by the list.
using NodesPtrList = std::list<glow::Node *>;
/// List of Functions.
using FunctionList = std::list<Function *>;
using ConstList = std::list<Constant *>;
using PlaceholderList = std::list<Placeholder *>;
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
  /// A list of constants that the Module owns.
  ConstList constants_;
  /// A list of placeholder nodes that the Module owns.
  PlaceholderList placeholders_;
  /// Deterministic PRNG used to initialize weights in this module.
  PseudoRNG PRNG_;

public:
  Module() = default;

  ~Module();

  /// \returns unique legal name that's based on the string \p name. Legal
  /// names are legal C identifiers in the form: "[a-zA-Z_][a-zA-Z0-9_]*".
  static llvm::StringRef uniqueName(llvm::StringRef name,
                                    llvm::StringSet<> &stringTable);

  /// Inserts the constant \p V to the list of constants.
  Constant *addConstant(Constant *V);

  /// Inserts the placeholder node \p ph to the list of variables.
  Placeholder *addPlaceholder(Placeholder *ph);

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

  /// Erase the constant \p N from the Module.
  void eraseConstant(Constant *N);

  /// Erase the variable \p I from the Module.
  void eraseConstant(ConstList::iterator I);

  /// \returns a pointer to the first variable with the name \p name or nullptr
  /// if no node has this name.
  Constant *getConstantByName(llvm::StringRef name);

  /// \returns the list of constants that the Module owns.
  ConstList &getConstants() { return constants_; }

  const ConstList &getConstants() const { return constants_; }

  /// \returns the list of placeholders that the Module owns.
  PlaceholderList &getPlaceholders() { return placeholders_; }

  const PlaceholderList &getPlaceholders() const { return placeholders_; }

  /// \returns a pointer to the placeholder with the name \p name or
  /// nullptr if no placeholder has this name.
  Placeholder *getPlaceholderByName(llvm::StringRef name);

  /// @name High-level Variable builders.
  ///@{

  Placeholder *createPlaceholder(ElemKind T, llvm::ArrayRef<size_t> dims,
                                 llvm::StringRef name, bool isTrainable);

  Placeholder *createPlaceholder(TypeRef T, llvm::StringRef name,
                                 bool isTrainable);

  Placeholder *createPlaceholder(ElemKind T, llvm::ArrayRef<size_t> dims,
                                 float scale, int32_t offset,
                                 llvm::StringRef name, bool isTrainable);

  Constant *createConstant(TypeRef T, llvm::StringRef name);

  Constant *createConstant(ElemKind T, llvm::ArrayRef<size_t> dims,
                           llvm::StringRef name);

  Constant *createConstant(ElemKind T, llvm::ArrayRef<size_t> dims, float scale,
                           int32_t offset, llvm::StringRef name);

  Constant *createConstant(llvm::StringRef name, const Tensor &tensor);

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

  /// Erase all the functions, variables, etc.
  void clear();

  /// Erase a function \p F from the module.
  void eraseFunction(Function *F);

  // Don't copy or move this class around.
  // The destructor will wipe the functions leaving
  // the original Module only dangling pointers.
  Module(const Module &) = delete;
  Module(Module &&) = delete;
  Module &operator=(const Context &) = delete;
  Module &operator=(Context &&) = delete;
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
                              NodeValue filter, NodeValue bias, TypeRef outTy,
                              llvm::ArrayRef<unsigned_t> kernels,
                              llvm::ArrayRef<unsigned_t> strides,
                              llvm::ArrayRef<unsigned_t> pads,
                              unsigned_t group);

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              NodeValue filter, NodeValue bias, TypeRef outTy,
                              unsigned_t kernel, unsigned_t stride,
                              unsigned_t pad, unsigned_t group);

  ConvertToNode *createConvertTo(llvm::StringRef name, NodeValue input,
                                 TypeRef outTy);

  MaxPoolNode *createMaxPool(llvm::StringRef name, NodeValue input,
                             llvm::ArrayRef<unsigned_t> kernels,
                             llvm::ArrayRef<unsigned_t> strides,
                             llvm::ArrayRef<unsigned_t> pads);

  MaxPoolNode *createMaxPool(llvm::StringRef name, NodeValue input,
                             unsigned_t kernel, unsigned_t stride,
                             unsigned_t pad);

  AvgPoolNode *createAvgPool(llvm::StringRef name, NodeValue input,
                             llvm::ArrayRef<unsigned_t> kernels,
                             llvm::ArrayRef<unsigned_t> strides,
                             llvm::ArrayRef<unsigned_t> pads);

  AvgPoolNode *createAvgPool(llvm::StringRef name, NodeValue input,
                             unsigned_t kernel, unsigned_t stride,
                             unsigned_t pad);

  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, Storage *W,
                                           Storage *B);

  /// Create a fully connected node with the specified output type.
  /// Note, outputDepth is infered based on the output type.
  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, Node *W, Node *B,
                                           TypeRef outTy);

  /// Create a row-wise quantized fully connected node. This node is only used
  /// in quantization. Args \p input and \p B are quantized in regular way, \p W
  /// is the constant weights and will be row-wise quantized during node
  /// creation time. The output is quantized in the regular way, and its type
  /// \p outTy is a quantized type.
  RowwiseQuantizedFullyConnectedNode *
  createRowwiseQuantizedFullyConnected(llvm::StringRef name, NodeValue input,
                                       Constant *W, Node *B, TypeRef outTy);

  /// Implement an operation that computes the row-wise dot product of its
  /// inputs. Consequently, \p X and \p Y must be either 1D or 2D tensors. This
  /// lowered to a Mul node, and is followed by a BatchedReduceAdd if \p X and
  /// \p Y are 2D. \returns either the Mul or BatchedReduceAdd node.
  Node *createDotProduct(llvm::StringRef name, NodeValue X, NodeValue Y);

  /// Create a ReLU node with the given \p name and \p input.
  /// Result type will be implicitly set based on the \p input type.
  ReluNode *createRELU(llvm::StringRef name, NodeValue input);

  /// Create a ReLU node with the given \p name, \p input and
  /// output type \p outTy.
  ReluNode *createRELU(llvm::StringRef name, NodeValue input, TypeRef outTy);

  SigmoidNode *createSigmoid(llvm::StringRef name, NodeValue input);

  TanhNode *createTanh(llvm::StringRef name, NodeValue input);

  /// Create a Log node with \p name, which calculates element-wise natural log
  /// of \p input, with output type \p outTy.
  LogNode *createLog(llvm::StringRef name, NodeValue input,
                     TypeRef outTy = nullptr);

  SoftMaxNode *createSoftMax(llvm::StringRef name, NodeValue input,
                             NodeValue selected, TypeRef outTy = nullptr);

  CrossEntropyLossNode *createCrossEntropyLoss(llvm::StringRef name,
                                               NodeValue input,
                                               NodeValue labels);

  RegressionNode *createRegression(llvm::StringRef name, NodeValue input,
                                   NodeValue expected);

  /// Creates a node, which computes sigmoid cross entropy between two inputs.
  SigmoidCrossEntropyWithLogitsNode *
  createSigmoidCrossEntropyWithLogits(llvm::StringRef name, NodeValue logits,
                                      NodeValue targets);

  ReshapeNode *createReshape(llvm::StringRef name, NodeValue input,
                             UnsignedArrayRef shape);

  TransposeNode *createTranspose(llvm::StringRef name, NodeValue input,
                                 llvm::ArrayRef<unsigned_t> shuffle);

  /// Create a series of nodes that implement a Broadcast operation. The \p
  /// input Tensor is broadcasted based on \p newShape and along the \p axis,
  /// which defines the offset from the leading dimension under which
  /// broadcasting is performed.
  Node *createBroadcast(llvm::StringRef name, NodeValue input,
                        UnsignedArrayRef newShape, unsigned_t axis);

  /// Create concat node which concatenates input tensors along \p dimension.
  ConcatNode *createConcat(llvm::StringRef name,
                           llvm::ArrayRef<NodeValue> inputs,
                           unsigned_t dimension);

  /// Create concat node with the given return type \p outTy.
  ConcatNode *createConcat(llvm::StringRef name,
                           llvm::ArrayRef<NodeValue> inputs,
                           unsigned_t dimension, TypeRef outTy);

  /// Create a quantized TileNode with \p name, \p input, \p tiles, and \p axis.
  /// For example, an input tensor {{1,2,3,4}} of dimension 1x4 with tiles = 2
  /// and axis = 0 would result in an output tensor {{1,2,3,4}, {1,2,3,4}} of
  /// dimension 2x4.
  TileNode *createTile(llvm::StringRef name, NodeValue input, unsigned_t tiles,
                       unsigned_t axis, TypeRef outTy = nullptr);

  /// Create an insert tensor node \p name, which inserts \p small into \p big
  /// at offset into big \p start \p count times along \p axis.
  InsertTensorNode *createInsertTensor(llvm::StringRef name, NodeValue big,
                                       NodeValue small,
                                       llvm::ArrayRef<size_t> start,
                                       unsigned_t count = 1,
                                       unsigned_t axis = 0);

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
                             unsigned_t axis);

  /// Create \p outputNum slice nodes of \p input. Slices happen along dimension
  /// number \p axis. Array \p split defines lengths of slices. If \p split is
  /// empty, \p input is split to equal sized parts.
  void createSplit(llvm::StringRef name, NodeValue input, unsigned_t outputNum,
                   unsigned_t axis, llvm::ArrayRef<size_t> split,
                   std::vector<Node *> &outputs);

  BatchNormalizationNode *
  createBatchNormalization(llvm::StringRef name, NodeValue input,
                           NodeValue beta, NodeValue gamma, NodeValue mean,
                           NodeValue var, unsigned_t channelIdx = 0,
                           float epsilon = 1e-5, float momentum = 0.9);

  LocalResponseNormalizationNode *createLocalResponseNormalization(
      llvm::StringRef name, NodeValue input, unsigned_t halfWindowSize = 2,
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

  /// Create a node that produces an boolean output of the same shape as
  /// \p input in which each element indicates whether or not the corresponding
  /// element in \p input is NaN or not.
  IsNaNNode *createIsNaN(llvm::StringRef name, NodeValue input);

  /// Implements an operation that replaces all instances of NaN in \p input
  /// with \p value. This operation is lowered to a Select node with \p input
  /// as one of the inputs, a Splat node created using \p value as the other
  /// input, and an IsNaN node as the comparator input.
  /// \returns the Select node.
  Node *createReplaceNaN(llvm::StringRef name, NodeValue input, float value);

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
  Node *createBroadcastedBatchMatMul(llvm::StringRef name, NodeValue lhs,
                                     NodeValue rhs);

  /// \p lhs and \p rhs are 3d matrices, where the leading dimension is the
  /// batch size. For each batch element number i, lhs.slice(i) is multiplied by
  /// rhs.slice(i). This is implemented by unrolling loop over batch size and
  /// issuing multiple slice, reshape and matmul instructions, and then
  /// concatenating results and bringing them back to 3d shape.
  Node *createParallelBatchMatMul(llvm::StringRef name, NodeValue lhs,
                                  NodeValue rhs);

  BatchedReduceAddNode *createBatchedReduceAdd(llvm::StringRef name,
                                               NodeValue batch,
                                               unsigned_t axis);

  BatchedReduceAddNode *createBatchedReduceAdd(llvm::StringRef name,
                                               TypeRef outTy, NodeValue batch,
                                               unsigned_t axis);

  /// Implements a batched reduce mean of the \p batch on the provided \p axis
  /// with output type \p outTy with three nodes: a BatchedReduceAdd followed by
  /// a DivNode with a SplatNode of the length of the \p axis
  /// dimension. \returns the final DivNode.
  DivNode *createBatchedReduceMean(llvm::StringRef name, TypeRef outTy,
                                   NodeValue batch, unsigned_t axis);

  /// Implements a batched reduce mean of the \p batch on the provided \p axis
  /// with three nodes: a BatchedReduceAdd followed by a DivNode with a
  /// SplatNode of the length of the \p axis dimension. \returns the final
  /// DivNode.
  DivNode *createBatchedReduceMean(llvm::StringRef name, NodeValue batch,
                                   unsigned_t axis);

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

  /// Given a vector of segment lengths, calculates offsets of each segment and
  /// packs them next to the lengths. For the input vector of length N the
  /// output is a Nx2 matrix with (offset, lengths) packaged for each segment.
  LengthsToRangesNode *createLengthsToRanges(llvm::StringRef name,
                                             NodeValue lengths);

  SaveNode *createSave(llvm::StringRef name, NodeValue input);
  SaveNode *createSave(llvm::StringRef name, NodeValue input,
                       Placeholder *output);

  /// Create quantization profile node named \p name for the output tensor from
  /// \p input in context \p ctx. Capture observed node name in quantization
  /// profile node as original node can be replaced during lowering phase.
  QuantizationProfileNode *createQuantizationProfile(Context &ctx,
                                                     llvm::StringRef name,
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

  TopKNode *createTopK(llvm::StringRef name, NodeValue input, unsigned_t k);

  /// Gathers entries of the outer-most dimension of \p data indexed by
  /// \p indices, and concatenates them. A non-zero \p batchDims specifies the
  /// batch, and the result is the concatenation of the operation on each sample
  /// in the batch.
  GatherNode *createGather(llvm::StringRef name, NodeValue data,
                           NodeValue indices, unsigned_t batchDims = 0);

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

  /// Create a series of nodes that implements a two-parameter
  /// rowwise Box-Cox transform. For each element of the \p input x, this is
  /// defined as:
  ///
  /// y = ln(max(x + lambda2, 1e-6)), if lambda1 == 0
  ///     (max(x + lambda2, 1e-6)^lambda1 - 1)/lambda1, if lambda1 != 0
  ///
  /// The transform parameters \p lambda1 and \p lambda2 are vectors of size D
  /// that are broadcasted to match the size of \p input (NxD). The transform
  /// itself is implemented using elementwise Max, Add, Log (if lambda1 == 0),
  /// Pow, Splat, Sub, and Div (if lambda1 != 0) nodes with a Splat and Select
  /// node to select between the two cases listed above. \returns the final
  /// Select node.
  Node *createBatchBoxCox(llvm::StringRef name, NodeValue input,
                          NodeValue lambda1, NodeValue lambda2);

  /// Create a series of nodes for the Clip operator. It limits the given input
  /// within an interval specified by the `min` and `max` arguments.
  Node *createClip(llvm::StringRef name, NodeValue input, float min, float max);
  /// @}

  /// @name The builder functions below are identical to the builder functions
  /// above except that they create nodes that use Placeholder instead of
  /// Variables. The methods create and initialize the tensors in the context.
  /// As soon as we finish the Placeholder migration we'll delete these methods
  /// and merge them with the builder methods above.
  /// See issue #1334.
  ///@{

  BatchNormalizationNode *
  createBatchNormalization(Context &ctx, llvm::StringRef name, NodeValue input,
                           unsigned_t channelIdx = 0, float epsilon = 1e-5,
                           float momentum = 0.9);

  ConvolutionNode *createConv(Context &ctx, llvm::StringRef name,
                              NodeValue input, size_t depth,
                              llvm::ArrayRef<unsigned_t> kernels,
                              llvm::ArrayRef<unsigned_t> strides,
                              llvm::ArrayRef<unsigned_t> pads,
                              unsigned_t group);

  ConvolutionNode *createConv(Context &ctx, llvm::StringRef name,
                              NodeValue input, size_t depth, unsigned_t kernel,
                              unsigned_t stride, unsigned_t pad,
                              unsigned_t group);

  /// Create a fully connected node with the given \p name, \p input and \p
  /// output depth. Trainable weight and bias variables are created implicitly.
  FullyConnectedNode *createFullyConnected(Context &ctx, llvm::StringRef name,
                                           NodeValue input, size_t outDepth);

  /// Create an unrolled single-layer Simple RNN cell with \p hiddenSize
  /// dimensionality of the hidden state and \p outputSize dimensionality of the
  /// output state. \p inputs define the input for the cell at each time step
  /// and the number of time steps is equal to the size of the \p inputs. The
  /// names of the created variables are prefixed by \p namePrefix.
  /// The output variables are written to \p outputs, they represent the
  /// activations of the output layer, unrolled over time.
  // The dimensionality of the output variables is \p batchSize x \p outputSize.
  void createSimpleRNN(Context &ctx, llvm::StringRef namePrefix,
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
  void createGRU(Context &ctx, llvm::StringRef namePrefix,
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
  void createLSTM(Context &ctx, llvm::StringRef namePrefix,
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

using VariableGradientsList =
    std::list<std::pair<Placeholder *, Placeholder *>>;

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

/// Helper vectors for common transpose shuffles.
#define NCHW2NHWC                                                              \
  { 0u, 2u, 3u, 1u }
#define NHWC2NCHW                                                              \
  { 0u, 3u, 1u, 2u }

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
