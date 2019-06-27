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
#include "glow/Graph/Log.h"
#include "glow/Graph/Nodes.h"
#include "glow/Quantization/Base/Base.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

#include <list>
#include <vector>

namespace glow {
class PlaceholderBindings;

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

  /// Module log context that stores all logs related to this module.
  ModuleLogContext moduleLogCtx_;

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

  /// Erase the constant \p I from the Module.
  void eraseConstant(ConstList::iterator I);

  /// \returns a pointer to the first variable with the name \p name or nullptr
  /// if no node has this name.
  Constant *getConstantByName(llvm::StringRef name) const;

  /// \returns the list of constants that the Module owns.
  ConstList &getConstants() { return constants_; }

  const ConstList &getConstants() const { return constants_; }

  /// \returns the list of placeholders that the Module owns.
  PlaceholderList &getPlaceholders() { return placeholders_; }

  const PlaceholderList &getPlaceholders() const { return placeholders_; }

  /// \returns a pointer to the placeholder with the name \p name or
  /// nullptr if no placeholder has this name.
  Placeholder *getPlaceholderByName(llvm::StringRef name) const;

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

  Constant *createConstant(llvm::StringRef name, Tensor &&tensor);

  ///@}

  /// Verify the correctness of the Module.
  /// \returns true when the function is valid. False otherwise.
  bool verify() const;

  /// Get the pseudo-random number generator used by this module.
  PseudoRNG &getPRNG() { return PRNG_; }

  /// Dump a textual representation of the Module into default output stream.
  void dump() const;

  /// Dump a textual representation of the Module to std::string.
  std::string toString() const;

  /// Dump a textual representation of the Module into provided output stream.
  void dump(llvm::raw_ostream &os) const;

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

  /// Strips payloads from constants. This is useful when
  /// the Module will be kept around for metadata but we want to reduce memory
  /// use. Unlike clear this leaves PHs and Constants in the module.
  void strip();

  /// Erase a function \p F from the module.
  void eraseFunction(Function *F);

  /// \Returns the size in bytes of data used by constants.
  uint64_t getConstantsSize();

  /// \Returns the module log context.
  ModuleLogContext &getModuleLogContext() { return moduleLogCtx_; };

  // Don't copy or move this class around.
  // The destructor will wipe the functions leaving
  // the original Module only dangling pointers.
  Module(const Module &) = delete;
  Module(Module &&) = delete;
  Module &operator=(const PlaceholderBindings &) = delete;
  Module &operator=(PlaceholderBindings &&) = delete;
};

/// Represents the compute graph.
class Function final : public Named {
  /// A list of nodes that the Function owns.
  NodesList nodes_;

  /// A list of metadata PHs associated with the function.
  std::vector<Placeholder *> metadataPlaceholders_;

  /// Stores a list of unique node names that were used by the module at some
  /// point.
  llvm::StringSet<> uniqueNodeNames_{};

  /// A reference to the owner of the function.
  Module *parent_;

  /// The log context associated with this function.
  std::shared_ptr<LogContext> logCtx_;

public:
  Function(Module *parent, llvm::StringRef Name = {})
      : Named(Name), parent_(parent) {
    logCtx_ = std::make_shared<LogContext>();
    logCtx_->setParent(this);
  }

  ~Function();

  /// Return the log context.
  std::shared_ptr<LogContext> getLogContext() { return logCtx_; }

  /// Add placeholder for metadata such as profiling.
  void addMetadataPlaceholder(Placeholder *PH) {
    metadataPlaceholders_.push_back(PH);
  }

  /// Get list of metadata placeholders.
  const std::vector<Placeholder *> &getMetadataPlaceholders() const {
    return metadataPlaceholders_;
  }

  Module *getParent() { return parent_; }

  /// Perform ordering of nodes_ based on node's name.
  /// This is to make sure that performing optimizations have a deterministic
  /// behavior on the graphs which have the same ops but different ordering in
  /// nodes_.
  void orderNodes() {
    nodes_.sort(
        [](const Node &a, const Node &b) { return a.getName() < b.getName(); });
  }

  /// Search the Module containing the function to gather and return a list of
  /// placeholders that are used by the Function.
  PlaceholderList findPlaceholders();
  PlaceholderList findPlaceholders() const;

  /// Search the Module containing the function to gather and return a list of
  /// constants that are used by the Function.
  ConstList findConstants();
  ConstList findConstants() const;

  const Module *getParent() const { return parent_; }

  /// Inserts the node \p N to the list of nodes, and returns the inserted node.
  template <class NodeTy> NodeTy *addNode(NodeTy *N) {
    N->setName(Module::uniqueName(N->getName(), uniqueNodeNames_));
    nodes_.push_back(N);

    // Log the node creation.
    logCtx_->logNodeCreation(*N);

    return N;
  }

  /// Take ownership of \p N by removing it from its original Function, add it
  /// to the current Function, and also unique its name.
  void takeOwnershipOfNode(Node *N) {
    N->getParent()->getNodes().remove(N);
    N->setName(Module::uniqueName(N->getName(), uniqueNodeNames_));
    nodes_.push_back(N);
  }

  /// Get the pseudo-random number generator used by this module.
  PseudoRNG &getPRNG() { return getParent()->getPRNG(); }

  /// @name High-level, operation-level IRBuilder.
  ///@{

  PadNode *createPad(llvm::StringRef name, NodeValue input, TypeRef outTy,
                     unsigned_t mode, llvm::ArrayRef<int> pads, float value);

  /// Creates a ConvolutionNode with the given \p name which convolves the 4D
  /// \p input with \p filter and \bias. \p kernels defines the size of the
  /// height and width dimensions of the filters. \p strides defines the number
  /// of steps to take in the input for each output cell. \p pads defines how
  /// many zero padding cells should be added to the input during convolution.
  /// \p group defines the number of groups the input and output channels should
  /// be divided into and convolved separately. \p dilation defines factor by
  /// which gap between 2 neighboring kernel elements is expanded along each
  /// axis.

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              NodeValue filter, NodeValue bias, TypeRef outTy,
                              llvm::ArrayRef<unsigned_t> kernels,
                              llvm::ArrayRef<unsigned_t> strides,
                              llvm::ArrayRef<unsigned_t> pads, unsigned_t group,
                              unsigned_t dilation = 1);

  /// Creates a ConvolutionNode with the given \p name which convolves the 4D
  /// \p input with \p filter and \bias. \p kernel defines the size of the
  /// height and width dimensions of the filters. \p stride defines the number
  /// of steps to take in the input for each output cell. \p pad defines how
  /// many zero padding cells should be added to the input during convolution.
  /// \p group defines the number of groups the input and output channels should
  /// be divided into and convolved separately. \p dilation defines factor by
  /// which gap between 2 neighboring kernel elements is expanded along each
  /// axis.

  ConvolutionNode *createConv(llvm::StringRef name, NodeValue input,
                              NodeValue filter, NodeValue bias, TypeRef outTy,
                              unsigned_t kernel, unsigned_t stride,
                              unsigned_t pad, unsigned_t group,
                              unsigned_t dilation = 1);

  /// Creates a Convolution3DNode with the given \p name which convolves the 5D
  /// \p input with \p filter and \bias. \p kernels defines the size of the
  /// height, width, and depth dimensions of the filters. \p strides defines the
  /// the number of steps to take in the input for each output cell. \p pads
  /// defines how many zero padding cells should be added to the input during
  /// convolution. \p group defines the number of groups the input and output
  /// channels should be divided into and convolved separately. \p outTy defines
  /// the type of the output of the 3d convolution.
  Convolution3DNode *createConv3D(llvm::StringRef name, NodeValue input,
                                  NodeValue filter, NodeValue bias,
                                  TypeRef outTy,
                                  llvm::ArrayRef<unsigned_t> kernels,
                                  llvm::ArrayRef<unsigned_t> strides,
                                  llvm::ArrayRef<unsigned_t> pads,
                                  unsigned_t group);

  /// Creates a Convolution3DNode with the given \p name which convolves the 5D
  /// \p input with \p filter and \bias. \p kernel defines the size of the
  /// height, width, and depth dimensions of the filters. \p stride defines the
  /// the number of steps to take in the input for each output cell. \p pad
  /// defines how many zero padding cells should be added to the input during
  /// convolution. \p group defines the number of groups the input and output
  /// channels should be divided into and convolved separately. \p outTy defines
  /// the type of the output of the 3d convolution.
  Convolution3DNode *createConv3D(llvm::StringRef name, NodeValue input,
                                  NodeValue filter, NodeValue bias,
                                  TypeRef outTy, unsigned_t kernel,
                                  unsigned_t stride, unsigned_t pad,
                                  unsigned_t group);

  /// Creates a ChannelwiseQuantizedConvolutionNode with the given \p name which
  /// convolves the 4D \p input with \p filter and \bias. \p scales and \p
  /// offsets provide individual quantization parameters for each filter group
  /// in \p filter. \p kernels defines the size of the height and width
  /// dimensions of the filters. \p strides defines the number of steps to take
  /// in the input for each output cell. \p pads defines how many zero padding
  /// cells should be added to the input during convolution. \p group defines
  /// the number of groups the input and output channels should be divided into
  /// and convolved separately.
  /// NOTE: ChannelwiseQuantizedConvolutionNode does not yet have an
  /// implementation so attempting to run a graph containing this node fails.
  ChannelwiseQuantizedConvolutionNode *createChannelwiseQuantizedConv(
      llvm::StringRef name, NodeValue input, Constant *filter, Constant *bias,
      Constant *scales, Constant *offsets, TypeRef outTy,
      llvm::ArrayRef<unsigned_t> kernels, llvm::ArrayRef<unsigned_t> strides,
      llvm::ArrayRef<unsigned_t> pads, unsigned_t group);

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
                             TypeRef outTy, llvm::ArrayRef<unsigned_t> kernels,
                             llvm::ArrayRef<unsigned_t> strides,
                             llvm::ArrayRef<unsigned_t> pads);

  AvgPoolNode *createAvgPool(llvm::StringRef name, NodeValue input,
                             unsigned_t kernel, unsigned_t stride,
                             unsigned_t pad);

  /// Creates and \returns a FullyConnectedNode with \p name, \p input, weights
  /// \p W, bias \p B. If \p input is not 2 dimensional then it is flattened
  /// along \p axis. Note, output type and outputDepth are inferred based on
  /// the input types.
  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, Storage *W,
                                           Storage *B, unsigned_t axis = 1);

  /// Creates and \returns a FullyConnectedNode with \p name, \p input, weights
  /// \p W, bias \p B, and \p outTy. If \p input is not 2 dimensional then it is
  /// flattened along \p axis. Note, outputDepth is inferred based on \p outTy.
  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, NodeValue W,
                                           NodeValue B, TypeRef outTy,
                                           unsigned_t axis = 1);

  /// Create a row-wise quantized fully connected node. This node is only used
  /// in quantization. Args \p input and \p B are quantized in regular way, \p W
  /// is the constant weights and will be row-wise quantized during node
  /// creation time. The output is quantized in the regular way, and its type
  /// \p outTy is a quantized type. if \p transposeWeight is true, \p W need to
  /// be transposed first.
  RowwiseQuantizedFullyConnectedNode *createRowwiseQuantizedFullyConnected(
      llvm::StringRef name, NodeValue input, Constant *W, NodeValue B,
      TypeRef outTy, quantization::Schema schema, bool transposeWeight = false);

  /// Implement an operation that computes the row-wise dot product of its
  /// inputs. Consequently, \p X and \p Y must be either 1D or 2D tensors. This
  /// lowered to a Mul node, and is followed by a BatchedReduceAdd if \p X and
  /// \p Y are 2D. \returns either the Mul or BatchedReduceAdd node.
  Node *createDotProduct(llvm::StringRef name, NodeValue X, NodeValue Y);

  /// Create a node that implements the elementwise linear operator. \p X is
  /// 2D and \p w and \p b are 1D. \p w and \p b are broadcasted to match the
  /// shape of \p X and then the output is computed by multiplying \p X and
  /// broadcasted \p w and adding broadcasted \p b. \returns the
  /// ElementwiseLinearNode. \p axis indicates the axis of the inputs (the other
  /// axis of \p X is assumed to be the batch index).
  Node *createElementwiseLinear(llvm::StringRef name, NodeValue X, NodeValue w,
                                NodeValue b, unsigned axis);

  /// Create a ReLU node with the given \p name and \p input.
  /// Result type will be implicitly set based on the \p input type.
  ReluNode *createRELU(llvm::StringRef name, NodeValue input);

  /// Create a ReLU node with the given \p name, \p input and
  /// output type \p outTy.
  ReluNode *createRELU(llvm::StringRef name, NodeValue input, TypeRef outTy);

  /// Create a PReLU node with the given \p name, \p input and  \p slope.
  /// Result type will be implicitly set based on the \p input type.
  PReluNode *createPRELU(llvm::StringRef name, NodeValue input,
                         NodeValue slope);

  /// Create a PReLU node with the given \p name, \p input, \p slope and
  /// output type \p outTy.
  PReluNode *createPRELU(llvm::StringRef name, NodeValue input, NodeValue slope,
                         TypeRef outTy);

  /// Create a Sigmoid node with the given \p name, \p input and
  /// output type \p outTy.
  SigmoidNode *createSigmoid(llvm::StringRef name, TypeRef outTy,
                             NodeValue input);

  /// Create a Sigmoid node with the given \p name and \p input.
  /// Result type will be implicitly set based on the \p input type.
  SigmoidNode *createSigmoid(llvm::StringRef name, NodeValue input);

  /// Create a Tanh node with the given \p name, \p input and
  /// output type \p outTy.
  TanhNode *createTanh(llvm::StringRef name, TypeRef outTy, NodeValue input);

  /// Create a Tanh node with the given \p name and \p input.
  /// Result type will be implicitly set based on the \p input type.
  TanhNode *createTanh(llvm::StringRef name, NodeValue input);

  /// Create an Exp  node with \p name, which calculates element-wise
  /// exponential of \p input.
  ExpNode *createExp(llvm::StringRef name, NodeValue input);

  /// Create a Log node with \p name, which calculates element-wise natural log
  /// of \p input, with output type \p outTy.
  LogNode *createLog(llvm::StringRef name, NodeValue input,
                     TypeRef outTy = nullptr);

  /// Create a series of nodes with \p name that implements an element-wise
  /// logit transform. For each element of the \p input x, this is
  /// defined as:
  ///
  /// y = log(x / (1 - x))
  ///
  /// where the \p input is clamped in (\p eps, 1 - \p eps), and
  /// the transform parameter \p eps is a positive value (< 0.5)
  /// (needed to avoid degenerate probabilities of 0 or 1,
  /// which would result in taking the logarithm of zero).
  /// The transform itself is implemented using element-wise Clip, Sub,
  /// Splat, Div, and Log nodes.
  /// \returns the final node.
  Node *createLogit(llvm::StringRef name, NodeValue input, float eps);

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
                   std::vector<SliceNode *> &outputs);

  BatchNormalizationNode *
  createBatchNormalization(llvm::StringRef name, NodeValue input,
                           NodeValue beta, NodeValue gamma, NodeValue mean,
                           NodeValue var, unsigned_t channelIdx = 0,
                           float epsilon = 1e-5, float momentum = 0.9);

  LocalResponseNormalizationNode *createLocalResponseNormalization(
      llvm::StringRef name, NodeValue input, unsigned_t halfWindowSize = 2,
      float alpha = 1e-4, float beta = 0.75, float k = 2.0);

  /// Create a ModuloNode which performs the modulo operation elementwise on the
  /// \p input such that each element in the output is equal to the
  /// corresponding element in the input modulo \p divisor. If \p
  /// signFollowDivisor is true then any negative elements in the output will
  /// have divisor added to their final values.
  ModuloNode *createModulo(llvm::StringRef name, NodeValue input,
                           int64_t divisor, bool signFollowDivisor = false);

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

  /// \returns a ReplaceNaNNode given \p name, \p input, and \p value.
  ReplaceNaNNode *createReplaceNaN(llvm::StringRef name, NodeValue input,
                                   float value);

  PowNode *createPow(llvm::StringRef name, NodeValue base, float exp);

  SelectNode *createSelect(llvm::StringRef name, NodeValue Cond, NodeValue LHS,
                           NodeValue RHS);

  SelectNode *createSelect(llvm::StringRef name, TypeRef outTy, NodeValue Cond,
                           NodeValue LHS, NodeValue RHS);

  SplatNode *createSplat(llvm::StringRef name, TypeRef ty, float value);

  MatMulNode *createMatMul(llvm::StringRef name, NodeValue lhs, NodeValue rhs);

  MatMulNode *createMatMul(llvm::StringRef name, TypeRef outTy, NodeValue lhs,
                           NodeValue rhs);

  /// \p lhs and \p rhs are 3d matrices, where the leading dimension is the
  /// batch size. For each batch element number i, lhs.slice(i) is multiplied by
  /// rhs.slice(i).
  BatchMatMulNode *createBatchMatMul(llvm::StringRef name, NodeValue lhs,
                                     NodeValue rhs);

  /// Create a node, performing BatchedReduceAdd operation. Output type is
  /// based on the input \p batch type with dimensions specified with \p axes
  /// removed.
  BatchedReduceAddNode *createBatchedReduceAdd(llvm::StringRef name,
                                               NodeValue batch,
                                               llvm::ArrayRef<unsigned_t> axes);

  /// Create a node, performing BatchedReduceAdd operation. Output type
  /// matches input \p outTy type.
  BatchedReduceAddNode *createBatchedReduceAdd(llvm::StringRef name,
                                               TypeRef outTy, NodeValue batch,
                                               llvm::ArrayRef<unsigned_t> axes);

  /// Create a node, performing BatchedReduceMean operation. Output type
  /// matches input \p outTy type.
  BatchedReduceMeanNode *
  createBatchedReduceMean(llvm::StringRef name, TypeRef outTy, NodeValue batch,
                          llvm::ArrayRef<unsigned_t> axes);

  /// Create a node, performing BatchedReduceMean operation. Output type is
  /// based on the input \p batch type with dimensions specified with \p axes
  /// removed.
  BatchedReduceMeanNode *
  createBatchedReduceMean(llvm::StringRef name, NodeValue batch,
                          llvm::ArrayRef<unsigned_t> axes);

  BatchedAddNode *createBatchedAdd(llvm::StringRef name, NodeValue batch,
                                   NodeValue sample);

  BatchedAddNode *createBatchedAdd(llvm::StringRef name, TypeRef outTy,
                                   NodeValue batch, NodeValue sample);

  /// Implements an operation that accumulates the values in \p data along the
  /// first dimension into len(\p lengths) entries by summing together the first
  /// lengths[0] values, then the subsequent lengths[1] values, etc.
  /// sum(\p lengths) must equal the first dimension of \p data. This operation
  /// is similar to SparseLengthsSum but the input is a dense represention
  /// instead of a sparse one. In other words, it has already been Gathered.
  LengthsSumNode *createLengthsSum(llvm::StringRef name, NodeValue data,
                                   NodeValue lengths);

  /// Create a node, performing SparseLengthsSum operation:
  /// Gathers slices of the outer-most dimension of Data indexed by Indices
  /// vector, and then accumulates them into len(Lengths) entries:
  /// first Lengths[0] slices are aggregated to Result[0], next Lengths[1]
  /// slices are aggregated to Result[1], etc. I.e. sum(Lengths) must be equal
  /// to len(Indices).
  SparseLengthsSumNode *createSparseLengthsSum(llvm::StringRef name,
                                               NodeValue data,
                                               NodeValue indices,
                                               NodeValue lengths);

  /// Same as SparseLengthsSum, but i-th slice is multiplied by weights[i].
  /// len(weights) must be equal to len(indices).
  SparseLengthsWeightedSumNode *
  createSparseLengthsWeightedSum(llvm::StringRef name, NodeValue data,
                                 NodeValue weights, NodeValue indices,
                                 NodeValue lengths);

  /// Same as \ref createSparseLengthsWeightedSum(), but with \p outTy
  /// specified.
  SparseLengthsWeightedSumNode *
  createSparseLengthsWeightedSum(llvm::StringRef name, TypeRef outTy,
                                 NodeValue data, NodeValue weights,
                                 NodeValue indices, NodeValue lengths);

  /// Creates and \returns a node of \p name, performing the SparseLengthsSum
  /// operation, using rowwise quantization for the input \p data with the \p
  /// scales and \p offsets as separate input tensors. Gathers slices of the
  /// outer-most dimension of data indexed by the \p indices vector, and then
  /// accumulates them into len(\p lengths) entries: first Lengths[0] slices are
  /// aggregated to Result[0], next Lengths[1] slices are aggregated to
  /// Result[1], etc. I.e. sum(Lengths) must be equal to len(Indices).
  RowwiseQuantizedSparseLengthsWeightedSumNode *
  createRowwiseQuantizedSparseLengthsSum(llvm::StringRef name, Constant *data,
                                         Constant *scales, Constant *offsets,
                                         NodeValue indices, NodeValue lengths);

  /// Same as \ref createRowwiseQuantizedSparseLengthsSum(), but expects
  /// float input \p data, which is rowwise-quantized internally.
  RowwiseQuantizedSparseLengthsWeightedSumNode *
  createRowwiseQuantizedSparseLengthsSum(llvm::StringRef name, Tensor &data,
                                         NodeValue indices, NodeValue lengths,
                                         quantization::Schema schema);

  /// Same as \ref createRowwiseQuantizedSparseLengthsSum(), but i-th slice is
  /// multiplied by weights[i]. len(weights) must be equal to len(indices).
  RowwiseQuantizedSparseLengthsWeightedSumNode *
  createRowwiseQuantizedSparseLengthsWeightedSum(
      llvm::StringRef name, Constant *data, Constant *scales, Constant *offsets,
      NodeValue weights, NodeValue indices, NodeValue lengths);

  /// Same as \ref createRowwiseQuantizedSparseLengthsWeightedSum(), but expects
  /// float input \p data, which is rowwise-quantized internally.
  RowwiseQuantizedSparseLengthsWeightedSumNode *
  createRowwiseQuantizedSparseLengthsWeightedSum(
      llvm::StringRef name, Tensor &data, NodeValue weights, NodeValue indices,
      NodeValue lengths, quantization::Schema schema);

  /// Creates and \returns a node of \p name, performing the SparseLengthsSum
  /// operation, using fused rowwise quantization for the input \p data wherein
  /// the scales and offsets are fused inline with each row of data. \p data
  /// must be ElemKind::UInt8FusedQTy. Gathers slices of the outer-most
  /// dimension of data indexed by the \p indices vector, and then accumulates
  /// them into len(\p lengths) entries: first Lengths[0] slices are aggregated
  /// to Result[0], next Lengths[1] slices are aggregated to Result[1], etc.
  /// I.e. sum(Lengths) must be equal to len(Indices).
  FusedRowwiseQuantizedSparseLengthsSumNode *
  createFusedRowwiseQuantizedSparseLengthsSum(llvm::StringRef name,
                                              Constant *data, NodeValue indices,
                                              NodeValue lengths);

  /// Same as \ref createFusedRowwiseQuantizedSparseLengthsSum(), but expects
  /// float input \p data, which is rowwise-quantized and fused internally.
  FusedRowwiseQuantizedSparseLengthsSumNode *
  createFusedRowwiseQuantizedSparseLengthsSum(llvm::StringRef name,
                                              Tensor &data, NodeValue indices,
                                              NodeValue lengths);

  /// Same as \ref createFusedRowwiseQuantizedSparseLengthsSum(), but i-th slice
  /// is multiplied by weights[i]. len(weights) must be equal to len(indices).
  FusedRowwiseQuantizedSparseLengthsWeightedSumNode *
  createFusedRowwiseQuantizedSparseLengthsWeightedSum(llvm::StringRef name,
                                                      NodeValue data,
                                                      NodeValue weights,
                                                      NodeValue indices,
                                                      NodeValue lengths);

  /// Same as \ref createFusedRowwiseQuantizedSparseLengthsWeightedSum(), but
  /// expects float input \p data, which is rowwise-quantized and fused
  /// internally.
  FusedRowwiseQuantizedSparseLengthsWeightedSumNode *
  createFusedRowwiseQuantizedSparseLengthsWeightedSum(llvm::StringRef name,
                                                      Tensor &data,
                                                      NodeValue weights,
                                                      NodeValue indices,
                                                      NodeValue lengths);

  /// Given a vector of segment lengths, calculates offsets of each segment and
  /// packs them next to the lengths. For the input vector of length N the
  /// output is a Nx2 matrix with (offset, lengths) packaged for each segment.
  LengthsToRangesNode *createLengthsToRanges(llvm::StringRef name,
                                             NodeValue lengths);

  /// Given a vector of \p lengths, \returns a LengthsRangeFillNode. This Node
  /// calculates a range sequence given \p lengths, where the sum of the
  /// elements of \p lengths must be no greater than \p maxOutputSize, which is
  /// used to set the output type.
  LengthsRangeFillNode *createLengthsRangeFill(llvm::StringRef name,
                                               NodeValue lengths,
                                               unsigned_t maxOutputSize);

  /// Implements an operation that converts the sparse representation given by
  /// the pair of \p indices and \p values into a dense representation.
  /// This representation contains each value of \p values at the corresponding
  /// index given by \p indices. All indices that are not present in \p indices
  /// are filled with zeroes. \p indices can contain duplicates, and in this
  /// case, the corresponding values in \p values are added.
  ///
  /// \p dataToInferDim acts as a hint about the shape of the output. The first
  /// dimension of the output is the first dimension of this tensor.
  SparseToDenseNode *createSparseToDense(llvm::StringRef name,
                                         NodeValue indices, NodeValue values,
                                         NodeValue dataToInferDim);

  /// Implements an operation that converts the sparse representation given by
  /// the pair of \p indices and \p values into a dense representation, which
  /// only contains IDs from given \p mask. Indices cannot contain duplicates.
  /// \p lengths is used to distinguish elements that belong to different
  /// examples of one batch. That is, first \p lengths[0] index-value pairs
  /// belong to batch's example 0, next \p lengths[1] pairs belong to example 1
  /// and so on.
  SparseToDenseMaskNode *
  createSparseToDenseMask(llvm::StringRef name, NodeValue indices,
                          NodeValue values, NodeValue defaultValue,
                          NodeValue lengths, llvm::ArrayRef<int64_t> mask);

  SaveNode *createSave(llvm::StringRef name, NodeValue input);
  SaveNode *createSave(llvm::StringRef name, NodeValue input,
                       Placeholder *output);

  /// Create quantization profile node named \p name for the output tensor from
  /// \p input in PlaceholderBindings \p bindings. Capture observed node name in
  /// quantization profile node as original node can be replaced during lowering
  /// phase.
  QuantizationProfileNode *
  createQuantizationProfile(PlaceholderBindings &bindings, llvm::StringRef name,
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

  /// Create a node, performing GatherRanges operation:
  /// Gathers entries of \p data in groups specified by the "examples" in
  /// \p ranges. Each example in \p ranges contains a list of pairs of
  /// indices of the form (index, length) which specify which entries of \p
  /// data to gather. The ordering of elements in \p ranges and of pairs
  /// within an element is preserved in the output. In addition to the result
  /// of gathering ("output"), the lengths of the ranges gathered by each
  /// example in \p ranges is also produced as an output ("lengths").
  /// \p maxOutputSize is the maximum possible size of "output" and is used to
  /// set its type. Users must use "lengths" to interpret "output" correctly.
  /// \returns the GatherRangesNode.
  GatherRangesNode *createGatherRanges(llvm::StringRef name, NodeValue data,
                                       NodeValue ranges,
                                       unsigned_t maxOutputSize);

  /// Copies each slice from \p slices into \p data at the corresponding index
  /// in \p indices, and \returns this new version of data. For example, given
  /// input data {{1,2},{3,4},{5,6}}, slices {{-3,-4}}, and indices {1}, the
  /// result is {{1,2},{-3,-4},{5,6}}.
  ScatterAssignNode *createScatterAssign(llvm::StringRef name, NodeValue data,
                                         NodeValue indices, NodeValue slices);

  /// Given 2D matrix \p data, 1D vector \p lengths (of the same size as width
  /// of \p data), and 1D vector \p values (of the same size as sum of
  /// \p lengths), expand each row of the \p data to a row of zeros and ones,
  /// according to One Hot Encoding. j-th element of resulting i-th row is one
  /// iff \p values[j] == \p data[i][some index within range of j].
  BatchOneHotNode *createBatchOneHot(llvm::StringRef name, NodeValue data,
                                     NodeValue lengths, NodeValue values);

  /// Given Input tensor of [N,H,W,C], where N is the batch
  /// axis, H is the height, W is
  /// the width, C is the channel or depth. This produces Output tensor of [N,
  /// H/blockSize, W/blockSize, C * blockSize * blockSize].
  SpaceToDepthNode *createSpaceToDepth(llvm::StringRef name, NodeValue input,
                                       unsigned blockSize);

  /// Create quantization node which transforms floating point tensor to a
  /// quantized one with given Scale and Offset. Scale and Offset params are
  /// part of the \p outTy.
  QuantizeNode *createQuantize(llvm::StringRef name, NodeValue input,
                               TypeRef outTy);

  /// Create dequantization node which transforms quantized tensor to a
  /// floating point one with given Scale and Offset. Scale and Offset params
  /// are part of the \p input.
  DequantizeNode *createDequantize(llvm::StringRef name, NodeValue input);

  /// Create dequantization node which transforms quantized tensor to a
  /// floating point type \p outTy one with given Scale and Offset. Scale and
  /// Offset params are part of the \p input.
  DequantizeNode *createDequantize(llvm::StringRef name, NodeValue input,
                                   TypeRef outTy);

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
  /// Select node. \p epsilon is used to ensure we do not divide by zero when
  /// calculating the lambda == 0 case, as we use a Select to choose which
  /// result to use, and so both paths are executed.
  Node *createBatchBoxCox(llvm::StringRef name, NodeValue input,
                          NodeValue lambda1, NodeValue lambda2,
                          float epsilon = std::numeric_limits<float>::min());

  /// Create a series of nodes for the Clip operator. It limits the given input
  /// within an interval specified by the `min` and `max` arguments.
  Node *createClip(llvm::StringRef name, NodeValue input, float min, float max);
  /// @}

  /// @name The builder functions below are identical to the builder functions
  /// above except that they create nodes that use Placeholder instead of
  /// Variables. The methods create and initialize the tensors in the
  /// PlaceholderBindings. As soon as we finish the Placeholder migration we'll
  /// delete these methods and merge them with the builder methods above. See
  /// issue #1334.
  ///@{

  BatchNormalizationNode *
  createBatchNormalization(PlaceholderBindings &bindings, llvm::StringRef name,
                           NodeValue input, unsigned_t channelIdx = 0,
                           float epsilon = 1e-5, float momentum = 0.9);

  /// Creates a ConvolutionNode with the given \p name which convolves the 4D
  /// \p input. \p kernels defines the size of the height and width dimensions
  /// of the convolutional filters. \p stride defines the the number of steps
  /// to take in the input for each output cell. \p pads defines how many zero
  /// padding cells should be added to the input during convolution. \p group
  /// defines the number of groups the input and output channels should be
  /// divided into and convolved separately. \p dilation defines factor by
  /// which gap between 2 neighboring kernel elements is expanded along each
  /// axis.
  ConvolutionNode *createConv(PlaceholderBindings &bindings,
                              llvm::StringRef name, NodeValue input,
                              size_t outChannels,
                              llvm::ArrayRef<unsigned_t> kernels,
                              llvm::ArrayRef<unsigned_t> strides,
                              llvm::ArrayRef<unsigned_t> pads, unsigned_t group,
                              unsigned_t dilation = 1);

  /// Creates a ConvolutionNode with the given \p name which convolves the 4D
  /// \p input. \p kernel defines the size of the height and width dimensions of
  /// the convolutional filters. \p stride defines the the number of steps to
  /// take in the input for each output cell. \p pad defines how many zero
  /// padding cells should be added to the input during convolution. \p group
  /// defines the number of groups the input and output channels should be
  /// divided into and convolved separately.\p dilation defines factor by
  /// which gap between 2 neighboring kernel elements is expanded along each
  /// axis.
  ConvolutionNode *createConv(PlaceholderBindings &bindings,
                              llvm::StringRef name, NodeValue input,
                              size_t outChannels, unsigned_t kernel,
                              unsigned_t stride, unsigned_t pad,
                              unsigned_t group, unsigned_t dilation = 1);

  /// Creates a Convolution3DNode with the given \p name which convolves the 5D
  /// \p input. \p kernels defines the size of the height, width, and depth
  /// dimensions of the convolutional filters. \p strides defines the the number
  /// of steps to take in the input for each output cell. \p pads defines how
  /// many zero padding cells should be added to the input during convolution.
  /// \p group defines the number of groups the input and output channels should
  /// be divided into and convolved separately.
  Convolution3DNode *createConv3D(PlaceholderBindings &bindings,
                                  llvm::StringRef name, NodeValue input,
                                  size_t outChannels,
                                  llvm::ArrayRef<unsigned_t> kernels,
                                  llvm::ArrayRef<unsigned_t> strides,
                                  llvm::ArrayRef<unsigned_t> pads,
                                  unsigned_t group);

  /// Creates a Convolution3DNode with the given \p name which convolves the 5D
  /// \p input. \p kernel defines the size of the height, width, and depth
  /// dimensions of the convolutional filters. \p stride defines the the number
  /// of steps to take in the input for each output cell. \p pad defines how
  /// many zero padding cells should be added to the input during convolution.
  /// \p group defines the number of groups the input and output channels should
  /// be divided into and convolved separately.
  Convolution3DNode *createConv3D(PlaceholderBindings &bindings,
                                  llvm::StringRef name, NodeValue input,
                                  size_t outChannels, unsigned_t kernel,
                                  unsigned_t stride, unsigned_t pad,
                                  unsigned_t group);

  /// Creates and \returns a FullyConnectedNode with \p name, \p input, weights
  /// \p W, bias \p B. If \p input is not 2 dimensional then it is flattened
  /// along \p axis. Note, output type is inferred based on the input
  /// types. Trainable weight and bias variables are created implicitly.
  FullyConnectedNode *createFullyConnected(PlaceholderBindings &bindings,
                                           llvm::StringRef name,
                                           NodeValue input, size_t outDepth,
                                           unsigned_t axis = 1);

  /// Create an unrolled single-layer Simple RNN cell with \p hiddenSize
  /// dimensionality of the hidden state and \p outputSize dimensionality of the
  /// output state. \p inputs define the input for the cell at each time step
  /// and the number of time steps is equal to the size of the \p inputs. The
  /// names of the created variables are prefixed by \p namePrefix.
  /// The output variables are written to \p outputs, they represent the
  /// activations of the output layer, unrolled over time.
  // The dimensionality of the output variables is \p batchSize x \p outputSize.
  void createSimpleRNN(PlaceholderBindings &bindings,
                       llvm::StringRef namePrefix,
                       const llvm::ArrayRef<NodeValue> inputs,
                       unsigned batchSize, unsigned hiddenSize,
                       unsigned outputSize, std::vector<NodeValue> &outputs);

  /// Create an unrolled single-layer GRU cell with \p hiddenSize
  /// dimensionality of the hidden state and \p outputSize dimensionality of the
  /// output state. \p inputs define the input for the cell at each time step
  /// and the number of time steps is equal to the size of the \p inputs. The
  /// names of the created variables are prefixed by \p namePrefix.
  /// The output variables are written to \p outputs, they represent the
  /// activation of the output layer, unrolled over time.
  // The dimensionality of the output variables is \p batchSize x \p outputSize.
  void createGRU(PlaceholderBindings &bindings, llvm::StringRef namePrefix,
                 const llvm::ArrayRef<NodeValue> inputs, unsigned batchSize,
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
  void createLSTM(PlaceholderBindings &bindings, llvm::StringRef namePrefix,
                  const llvm::ArrayRef<NodeValue> inputs, unsigned batchSize,
                  unsigned hiddenSize, unsigned outputSize,
                  std::vector<NodeValue> &outputs);
  /// @}

  /// Create a TraceEvent in the runtime profile, which triggers collection of
  /// runtime statistics.
  TraceEventNode *createTraceEvent(llvm::StringRef eventName,
                                   llvm::StringRef eventType, Node *data,
                                   unsigned index);

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
  /// \returns true when the function is valid. False otherwise.
  bool verify() const;

  /// Dump a textual representation of the Function into provided output stream.
  void dump() const;

  /// Dump a textual representation of the Function to std::string.
  std::string toString() const;

  /// Dump a textual representation of the Function into default output stream.
  void dump(llvm::raw_ostream &os) const;

  /// Dump a dotty graph that depicts the function into a file.
  /// \returns full path to the file.
  std::string dumpDAG();

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

/// \returns the first SaveNode user of the placeholder \p PH or
/// nullptr if none are found.
SaveNode *getOutputSave(Function *F, Placeholder *PH);

/// Helper vectors for common transpose shuffles.
#define NCHW2NHWC                                                              \
  { 0u, 2u, 3u, 1u }
#define NHWC2NCHW                                                              \
  { 0u, 3u, 1u, 2u }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Module &mod);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Module *mod);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Function &F);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Function *F);

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
