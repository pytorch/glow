/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
using UnsignedArrayRef = llvm::ArrayRef<dim_t>;
/// Map from original Nodes to cloned Nodes.
using NodeMap = llvm::DenseMap<Node *, Node *>;
/// State of a function. This can be used to control optimizations which depend
/// on the state of the Function. This is a temporary workaround until GH Issue
/// #3213 is complete.
enum class FunctionState {
  /// Indicates that the function has been created but not completely loaded.
  FuncCreated,
  /// Indicates that the function has been completely loaded.
  FuncLoaded,
};

/// Helper names for common tensor layouts.
#define ANY_LAYOUT "*"

class Module final {
  /// Stores the functions in the module.
  FunctionList functions_;
  /// A uniqued list of types. Types in this list can be equated by comparing
  /// their addresses.
  TypesList types_{};
  /// Stores a list of unique Storage names that were used by the module at
  /// some point.
  llvm::StringSet<> usedStorageNames_{};
  /// Stores a list of node names that were used by Functions of this module at
  /// some point.
  llvm::StringSet<> usedNodeNames_{};
  /// Stores a list of node names that were present in the original model and
  /// are good to be retained.
  llvm::StringSet<> originalNames_{};
  /// A list of constants that the Module owns.
  ConstList constants_;
  /// A list of placeholder nodes that the Module owns.
  PlaceholderList placeholders_;
  /// Deterministic PRNG used to initialize weights in this module.
  PseudoRNG PRNG_;

  /// Module log context that stores all logs related to this module.
  LogContext moduleLogCtx_{nullptr};

  /// Inserts the constant \p V to the list of constants.
  Constant *addConstant(Constant *V);

  friend class Function;

public:
  Module() = default;

  ~Module();

  /// \returns the prefix part of the provided \p name. E.g. for an input
  /// of "relu__2" returns "relu".
  static std::string getPrefix(llvm::StringRef name);

  /// \returns unique legal name that's based on the string \p name. Legal
  /// names are legal C identifiers in the form: "[a-zA-Z_][a-zA-Z0-9_]*".
  /// The name may not be in \p stringTable or \p updateTable and will be
  /// inserted into \p updateTable.
  static llvm::StringRef uniqueName(llvm::StringRef name,
                                    const llvm::StringSet<> &stringTable,
                                    llvm::StringSet<> &updateTable,
                                    const llvm::StringSet<> &originalNames);

  /// Registers a \p name as used by some Node in this module.
  void registerNodeName(llvm::StringRef name) {
    // Don't care if it's already in the set.
    usedNodeNames_.insert(name);
  }

  /// Registers a \p name from the original model, good to be retained.
  void registerOriginalName(llvm::StringRef name) {
    // Don't care if it's already in the set.
    if (name.size()) {
      originalNames_.insert(name);
    }
  }

  /// \returns the pointer to list of original node names, good to be retained;
  const llvm::StringSet<> *getOriginalNames() const { return &originalNames_; }

  /// Registers a name as used by a Storage node (Constant or Placeholder) in
  /// this module.
  void registerStorageName(llvm::StringRef name) {
    usedStorageNames_.insert(name);
  }

  /// \returns whether there's a Storage node already registered with \p name.
  bool hasStorageName(llvm::StringRef name) {
    return usedStorageNames_.count(name);
  }

  /// Return a pointer to a uniqued type \p T.
  TypeRef uniqueType(const Type &T);

  /// Return a pointer to a uniqued type \p T.
  TypeRef uniqueType(ElemKind elemTy, llvm::ArrayRef<dim_t> dims);

  /// Return a pointer to a uniqued type \p T.
  TypeRef uniqueType(ElemKind elemTy, llvm::ArrayRef<dim_t> dims, float scale,
                     int32_t offset);

  /// Return a pointer to a uniqued type \p T.
  /// The new type is identical to \p T, with a new shape \p dims.
  TypeRef uniqueTypeWithNewShape(TypeRef T, llvm::ArrayRef<dim_t> dims);

  /// The new type is identical to \p T, with a new shape \p dims and new \p
  /// alignments.
  TypeRef uniqueTypeWithNewShape(TypeRef T, llvm::ArrayRef<dim_t> dims,
                                 llvm::ArrayRef<dim_t> alignments);

  /// Return a pointer to a uniqued type \p T.
  /// The new type is identical to \p T, with a new shape and strides taken from
  /// the type \p shapeType.
  TypeRef uniqueTypeWithNewShape(TypeRef T, TypeRef shapeType);

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

  /// Clears out all Functions from \ref functions_.
  void clearFunctions();

  /// \returns the list of types that the Module owns.
  const TypesList &getTypes() const { return types_; }

  /// Erase the constant \p N from the Module.
  void eraseConstant(Constant *N);

  /// Erase the constant \p I from the Module.
  void eraseConstant(ConstList::iterator I);

  /// Erase the placeholder \p I from the Module.
  /// Note: we only provide an iterator version of this, as erasing Placeholders
  /// is often unsafe.
  void erasePlaceholder(PlaceholderList::iterator I);

  /// \returns a pointer to the first Constant with the name \p name or nullptr
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
  Placeholder *getPlaceholderByNameSlow(llvm::StringRef name) const;

  /// @name High-level Storage builders.
  ///@{

  Placeholder *createPlaceholder(ElemKind T, llvm::ArrayRef<dim_t> dims,
                                 llvm::StringRef name, bool isTrainable,
                                 const std::string &layout = ANY_LAYOUT);

  Placeholder *createPlaceholder(TypeRef T, llvm::StringRef name,
                                 bool isTrainable,
                                 const std::string &layout = ANY_LAYOUT);

  Placeholder *createPlaceholder(ElemKind T, llvm::ArrayRef<dim_t> dims,
                                 float scale, int32_t offset,
                                 llvm::StringRef name, bool isTrainable,
                                 const std::string &layout = ANY_LAYOUT);

  Constant *createConstant(TypeRef T, llvm::StringRef name,
                           const std::string &layout = ANY_LAYOUT);

  Constant *createConstant(ElemKind T, llvm::ArrayRef<dim_t> dims,
                           llvm::StringRef name,
                           const std::string &layout = ANY_LAYOUT);

  Constant *createConstant(ElemKind T, llvm::ArrayRef<dim_t> dims, float scale,
                           int32_t offset, llvm::StringRef name,
                           const std::string &layout = ANY_LAYOUT);

  Constant *createConstant(llvm::StringRef name, const Tensor &tensor,
                           const std::string &layout = ANY_LAYOUT);

  Constant *createConstant(llvm::StringRef name, Tensor &&tensor,
                           const std::string &layout = ANY_LAYOUT);

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

  /// Erase all the functions, Placeholders, Constants, etc.
  void clear();

  /// Clone a module.
  /// \returns a new module that is a copy of the current module.
  Module *clone() const;

  /// Clone the current module into a user-provided module \p M.
  /// \returns the user-provided module \p M that now contains a clone of the
  /// current module.
  Module *clone(Module *M) const;

  /// Strips payloads from constants. This is useful when
  /// the Module will be kept around for metadata but we want to reduce memory
  /// use. Unlike clear this leaves PHs and Constants in the module.
  void strip();

  /// Erase a function \p F from the module.
  void eraseFunction(Function *F);

  /// \Returns the size in bytes of data used by constants.
  uint64_t getConstantsSize();

  /// \Returns the module log context.
  LogContext *getModuleLogContext() { return &moduleLogCtx_; };

  /// \returns whether any Node in the module are non-fused quantized with
  /// scale == or != dummyScale, depending on \p expectDummy.
  Error verifyDummyQParams(bool expectDummies);

  // Don't copy or move this class around.
  // The destructor will wipe the functions leaving
  // the original Module only dangling pointers.
  Module(const Module &) = delete;
  Module(Module &&) = delete;
  Module &operator=(const PlaceholderBindings &) = delete;
  Module &operator=(PlaceholderBindings &&) = delete;
};

// Forward Declaration for verify's optional parameter
class Backend;
struct CompilationContext;

/// Represents the compute graph.
class Function final : public IRContainer {
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

  /// The state of this function.
  FunctionState state_;

public:
  Function(Module *parent, llvm::StringRef Name = {})
      : IRContainer(Name), parent_(parent), state_(FunctionState::FuncCreated) {
    logCtx_ = std::make_shared<LogContext>(parent);
    logCtx_->pushEvent(parent->getModuleLogContext()->getClonedScope());
  }

  ~Function();

  /// Clear out \ref nodes_ and \ref uniqueNodeNames_.
  void clear();

  /// Sets the state of the function.
  void setState(FunctionState state) { state_ = state; }

  /// Gets the state of the function.
  FunctionState getState() { return state_; }

  std::string getFilename() { return getName().rsplit('/').second.str(); }

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
  /// Please do not call this in the middle of PyTorchModelLoading, since
  /// constant propagation is heavily relied on the order of nodes in nodelist.
  /// If the order is changed during model loading, the constant propagation may
  /// cause unpredictable fatal error when building the graph.
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
    N->setName(Module::uniqueName(N->getName(), parent_->usedStorageNames_,
                                  uniqueNodeNames_, parent_->originalNames_));
    parent_->registerNodeName(N->getName());
    nodes_.push_back(N);

    // Log the node creation.
    logCtx_->logNodeCreation(*N);

    return N;
  }

  /// Take ownership of \p N by removing it from its original Function, add it
  /// to the current Function, and also unique its name.
  void takeOwnershipOfNode(Node *N) {
    N->getParent()->getNodes().remove(N);
    N->setName(Module::uniqueName(N->getName(), parent_->usedStorageNames_,
                                  uniqueNodeNames_, parent_->originalNames_));
    parent_->registerNodeName(N->getName());
    nodes_.push_back(N);
  }

  /// Get the pseudo-random number generator used by this module.
  PseudoRNG &getPRNG() { return getParent()->getPRNG(); }

  /// @name High-level, operation-level IRBuilder.
  ///@{

  /// Creates a PadNode with the given \p name and output type \p outTy which
  /// pads the given \p input with the explicit pads \p pads according to the
  /// padding mode \p mode and with the given value \p value. The padding mode
  /// \p mode is one of enumeration values from \ref PaddingMode. For an input
  /// with N dimensions (rank N) the \p pads must be a vector with 2*N values
  /// with the following format:
  /// pads = [pad_before(D1), pad_before(D2), ..., pad_before(DN),
  ///         pad_after (D1), pad_after (D2), ..., pad_after (DN)].
  /// The mode PaddingMode::CONSTANT pads the input using the constant value
  /// \p value and currently is the only mode supported.
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
  /// axis. \p layout defines the Tensor layout and must be either NHWC or NCHW.

  ConvolutionNode *
  createConv(llvm::StringRef name, NodeValue input, NodeValue filter,
             NodeValue bias, TypeRef outTy, llvm::ArrayRef<unsigned_t> kernels,
             llvm::ArrayRef<unsigned_t> strides,
             llvm::ArrayRef<unsigned_t> pads, unsigned_t group,
             llvm::ArrayRef<unsigned_t> dilation = {1, 1},
             ConvolutionLayout layout = ConvolutionLayout::NHWC);

  /// Creates a ConvolutionNode with the given \p name which convolves the 4D
  /// \p input with \p filter and \bias. \p kernel defines the size of the
  /// height and width dimensions of the filters. \p stride defines the number
  /// of steps to take in the input for each output cell. \p pad defines how
  /// many zero padding cells should be added to the input during convolution.
  /// \p group defines the number of groups the input and output channels should
  /// be divided into and convolved separately. \p dilation defines factor by
  /// which gap between 2 neighboring kernel elements is expanded along each
  /// axis. \p layout defines the Tensor layout and must be either NHWC or NCHW.

  ConvolutionNode *
  createConv(llvm::StringRef name, NodeValue input, NodeValue filter,
             NodeValue bias, TypeRef outTy, unsigned_t kernel,
             unsigned_t stride, unsigned_t pad, unsigned_t group,
             llvm::ArrayRef<unsigned_t> dilation = {1, 1},
             ConvolutionLayout layout = ConvolutionLayout::NHWC);

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
  /// convolves the 4D/5D \p input with \p filter and \p bias. \p filterScales
  /// and \p filterOffsets provide individual quantization parameters for each
  /// filter group in \p filter while \p biasScales and \p biasOffsets provide
  /// individual quantization parameters for each bias element corresponding to
  /// each output channel. \p kernels defines the size of the height and width
  /// dimensions of the filters. \p strides defines the number of steps to take
  /// in the input for each output cell. \p pads defines how many zero padding
  /// cells should be added to the input during convolution. \p group defines
  /// the number of groups the input and output channels should be divided into
  /// and convolved separately. \p dilation defines the filter dilation.
  /// This function is flexible and has the following features:
  /// - it can be provided with a floating-point \p filter and the function will
  ///   quantize automatically the filter channelwise using the given schema
  ///   \p schema and type \p filterElemQTy.
  /// - it can be provided with a floating-point \p bias and the function will
  ///   quantize automatically the bias channelwise using the given schema
  ///   \p schema and type \p biasElemQTy.
  /// - if \p filter is floating-point and \p filterScales or \p filterOffsets
  ///   are not provided then this function will derive them automatically.
  /// - if \p filter is quantized then \p filterScales or \p filterOffsets are
  ///   mandatory.
  /// - if \p bias is floating-point and \p biasScales or \p biasOffsets are not
  ///   provided then this function will derive them automatically.
  /// - if \p bias is quantized  and \p biasScales or \p biasOffsets are not
  ///   provided then this function will assume the implicit parameters
  ///   biasScales[i] = inputScale * filterScales[i] and biasOffsets[i] = 0.
  ///   To be noted that this case can handle safely only INT32 bias data type
  ///   because for INT8 type the bias will almost certainly be saturated.
  /// This function will only quantize the filter if \p quantizeFilter is set
  /// to true and will only quantize the bias if \p quantizeBias is set to true
  /// such that a floating-point filter/bias can be attached to the node as-is
  /// without any modifications in order for the backends to perform their own
  /// custom quantization later if desired.
  /// This function requires \p filter and \p bias operands to be constants.
  ChannelwiseQuantizedConvolutionNode *createChannelwiseQuantizedConv(
      llvm::StringRef name, NodeValue input, NodeValue filter, NodeValue bias,
      NodeValue filterScales, NodeValue filterOffsets, NodeValue biasScales,
      NodeValue biasOffsets, TypeRef outTy, llvm::ArrayRef<unsigned_t> kernels,
      llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
      unsigned_t group, llvm::ArrayRef<unsigned_t> dilation = {1, 1},
      bool quantizeFilter = true, bool quantizeBias = true,
      quantization::Schema schema = quantization::Schema::Asymmetric,
      ElemKind filterElemQTy = ElemKind::Int8QTy,
      ElemKind biasElemQTy = ElemKind::Int32QTy);

  /// Creates a ConvTransposeNode with the given \p name which does transposed
  /// convolution of the 4D \p input with \p filter and \bias. \p kernels define
  /// the size of the height and width dimensions of the filters. \p strides
  /// define the number of steps to take in the input for each output cell.
  /// \p pads define how many zero padding cells should be added to the input
  /// during convolution. \p group defines the number of groups the input and
  /// output channels should be divided into and convolved separately.
  ConvTransposeNode *createConvTranspose(
      llvm::StringRef name, NodeValue input, NodeValue filter, NodeValue bias,
      TypeRef outTy, llvm::ArrayRef<unsigned_t> kernels,
      llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
      unsigned_t group, llvm::ArrayRef<unsigned_t> dilation = {1, 1});

  /// Creates a createConvTransposeNode with the given \p name which does
  /// transposed convolution of the 4D \p input with \p filter and \bias. \p
  /// kernel defines the size of the height and width dimensions of the filters.
  /// \p stride defines the number of steps to take in the input for each output
  /// cell. \p pad defines how many zero padding cells should be added to the
  /// input during convolution. \p group defines the number of groups the input
  /// and output channels should be divided into and convolved separately.
  ConvTransposeNode *
  createConvTranspose(llvm::StringRef name, NodeValue input, NodeValue filter,
                      NodeValue bias, TypeRef outTy, unsigned_t kernel,
                      unsigned_t stride, unsigned_t pad, unsigned_t group,
                      llvm::ArrayRef<unsigned_t> dilation = {1, 1});

  /// Creates and \returns a ConvertTo Node with name \p name of \p input to
  /// output type \p outTy.
  ConvertToNode *createConvertTo(llvm::StringRef name, NodeValue input,
                                 TypeRef outTy);

  /// Creates and \returns a ConvertTo Node with name \p name of \p input to
  /// output ElemKind \p k.
  ConvertToNode *createConvertTo(llvm::StringRef name, NodeValue input,
                                 ElemKind k);

  MaxPoolNode *createMaxPool(llvm::StringRef name, NodeValue input,
                             llvm::ArrayRef<unsigned_t> kernels,
                             llvm::ArrayRef<unsigned_t> strides,
                             llvm::ArrayRef<unsigned_t> pads,
                             ElemKind elemTyAMT = ElemKind::Int64ITy,
                             ConvolutionLayout layout = NHWC,
                             bool flattenIndices = true);

  MaxPoolNode *createMaxPool(llvm::StringRef name, NodeValue input,
                             unsigned_t kernel, unsigned_t stride,
                             unsigned_t pad,
                             ElemKind elemTyAMT = ElemKind::Int64ITy,
                             ConvolutionLayout layout = NHWC,
                             bool flattenIndices = true);

  AvgPoolNode *createAvgPool(llvm::StringRef name, NodeValue input,
                             llvm::ArrayRef<unsigned_t> kernels,
                             llvm::ArrayRef<unsigned_t> strides,
                             llvm::ArrayRef<unsigned_t> pads,
                             ConvolutionLayout layout = NHWC,
                             bool countIncludePads = true);

  AvgPoolNode *createAvgPool(llvm::StringRef name, NodeValue input,
                             TypeRef outTy, llvm::ArrayRef<unsigned_t> kernels,
                             llvm::ArrayRef<unsigned_t> strides,
                             llvm::ArrayRef<unsigned_t> pads,
                             ConvolutionLayout layout = NHWC,
                             bool countIncludePads = true);

  AvgPoolNode *createAvgPool(llvm::StringRef name, NodeValue input,
                             unsigned_t kernel, unsigned_t stride,
                             unsigned_t pad, ConvolutionLayout layout = NHWC,
                             bool countIncludePads = true);

  /// Creates and \returns an AdaptiveAvgPool node with \p name, \p input, and
  /// \p outTy. The AdaptiveAvgPoolNode will perform average pooling over the
  /// input so that the result is of the shape specified by \p outTy.
  AdaptiveAvgPoolNode *createAdaptiveAvgPool(llvm::StringRef name,
                                             NodeValue input, TypeRef outTy);

  /// Creates and \returns a General Matrix Multiplication (Gemm) node with
  /// given \p name which computes Y = alpha * A * B + beta * C. The operands
  /// \p A and \p B are 2D matrices, the \p C operand is an optional 1D or 2D
  /// matrix (broadcastable to the size of Y) and \p alpha and \p beta are float
  /// scalars. The \p C operand is optional, if nullptr is given then it is not
  /// used. If \p transposeA or \p transposeB is true then \p A or \p B is
  /// additionally transposed prior to matrix multiplication.
  /// If the output shape of Y is [M,N] then:
  /// - The shape of \p A must be [M,K] or [K,M] (if transposed).
  /// - The shape of \p B must be [K,N] or [N,K] (if transposed).
  /// - The shape of \p C must be [N] (if 1D) or [M,N] (if 2D).
  GemmNode *createGemm(llvm::StringRef name, NodeValue A, NodeValue B,
                       NodeValue C = nullptr, float alpha = 1.0,
                       float beta = 1.0, bool transposeA = false,
                       bool transposeB = false);

  GemmNode *createGemm(llvm::StringRef name, TypeRef outTy, NodeValue A,
                       NodeValue B, NodeValue C = nullptr, float alpha = 1.0,
                       float beta = 1.0, bool transposeA = false,
                       bool transposeB = false);

  /// Creates and \returns a FullyConnectedNode with \p name, \p input, weights
  /// \p W, bias \p B. If \p input is not 2 dimensional then it is flattened
  /// along \p axis. Note, output type and outputDepth are inferred based on
  /// the input types.
  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, Storage *W,
                                           Storage *B, unsigned_t axis = 1);

  /// Creates and \returns a FullyConnectedNode with \p name, \p input, weights
  /// \p W, bias \p B. If \p input is not 2 dimensional then it is flattened
  /// along \p axis. Note, output type and outputDepth are inferred based on
  /// the input types.
  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, NodeValue W,
                                           NodeValue B, unsigned_t axis = 1);

  /// Creates and \returns a FullyConnectedNode with \p name, \p input, weights
  /// \p W, bias \p B, and \p outTy. If \p input is not 2 dimensional then it is
  /// flattened along \p axis. Note, outputDepth is inferred based on \p outTy.
  FullyConnectedNode *createFullyConnected(llvm::StringRef name,
                                           NodeValue input, NodeValue W,
                                           NodeValue B, TypeRef outTy,
                                           unsigned_t axis = 1);

  /// Create a row-wise quantized fully connected node. This node is only used
  /// in quantization. Args \p input and \p B are quantized in regular way, \p W
  /// is the constant weights and is row-wise quantized using the given \p
  /// scales and \p offsets. The output is quantized in the regular way, and its
  /// type \p outTy is a quantized type.
  RowwiseQuantizedFullyConnectedNode *createRowwiseQuantizedFullyConnected(
      llvm::StringRef name, NodeValue input, NodeValue W, Constant *scales,
      Constant *offsets, NodeValue B, TypeRef outTy);

  /// Create a row-wise quantized fully connected node. This node is only used
  /// in quantization. Args \p input and \p B are quantized in regular way, \p W
  /// is the constant weights and will be row-wise quantized during node
  /// creation time. The output is quantized in the regular way, and its type
  /// \p outTy is a quantized type. if \p transposeWeight is true, \p W need to
  /// be transposed first.
  RowwiseQuantizedFullyConnectedNode *createRowwiseQuantizedFullyConnected(
      llvm::StringRef name, NodeValue input, NodeValue W, NodeValue B,
      TypeRef outTy, quantization::Schema schema, bool transposeWeight = false);

  /// Implement an operation that computes the row-wise dot product of its
  /// inputs. Consequently, \p X and \p Y must be either 1D or 2D tensors. This
  /// lowered to a Mul node, and is followed by a BatchedReduceAdd if \p X and
  /// \p Y are 2D. \returns either the Mul or BatchedReduceAdd node.
  Node *createDotProduct(llvm::StringRef name, NodeValue X, NodeValue Y);

  /// Create a node that computes the pairwise dot product of \p inputs, which
  /// must be a list of 2D tensors with identical shape. \returns the
  /// BatchedPairwiseDotProductNode.
  BatchedPairwiseDotProductNode *
  createBatchedPairwiseDotProduct(llvm::StringRef name,
                                  llvm::ArrayRef<NodeValue> inputs);

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

  /// Create a series of nodes representing a GeLU with the given \p name and \p
  /// input. Result type will be implicitly set based on the \p input type.
  Node *createGELU(llvm::StringRef name, NodeValue input);

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

  /// Create a Swish node with the given \p name and \p input.
  /// If \p OT is nullptr, then result type will be implicitly set based on the
  /// \p input type.
  SwishNode *createSwish(llvm::StringRef name, NodeValue input,
                         TypeRef OT = nullptr);

  /// Create a Tanh node with the given \p name, \p input and
  /// output type \p outTy.
  TanhNode *createTanh(llvm::StringRef name, TypeRef outTy, NodeValue input);

  /// Create a Tanh node with the given \p name and \p input.
  /// Result type will be implicitly set based on the \p input type.
  TanhNode *createTanh(llvm::StringRef name, NodeValue input);

  /// Create an Exp node with \p name, which calculates element-wise
  /// exponential of \p input.
  ExpNode *createExp(llvm::StringRef name, NodeValue input);

  /// Create an Exp node with \p name with output type \p outTy, which
  /// calculates element-wise exponential of \p input.
  ExpNode *createExp(llvm::StringRef name, TypeRef outTy, NodeValue input);

  /// Create a Log node with \p name, which calculates element-wise natural log
  /// of \p input, with output type \p outTy.
  LogNode *createLog(llvm::StringRef name, NodeValue input,
                     TypeRef outTy = nullptr);

  /// \returns a LogitNode with \p name given \p input and \p eps.
  LogitNode *createLogit(llvm::StringRef name, NodeValue input, float eps);

  SoftMaxNode *createSoftMax(llvm::StringRef name, NodeValue input,
                             NodeValue selected, TypeRef outTy = nullptr,
                             float beta = 1.0);

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
                             UnsignedArrayRef shape,
                             llvm::StringRef layout = ANY_LAYOUT);

  TransposeNode *createTranspose(llvm::StringRef name, NodeValue input,
                                 llvm::ArrayRef<unsigned_t> shuffle,
                                 const std::string &layout = ANY_LAYOUT);

  /// Create a node with the name \p name which flips (reorders) the elements
  /// of the input \p input along the given axis \p axis.
  FlipNode *createFlip(llvm::StringRef name, NodeValue input, unsigned_t axis);

  /// Create a Broadcast node that broadcasting the \p input Tensor based on
  /// \p newShape and along the \p axis, which defines the offset between the
  /// input dim and the newShape.
  /// e.g. For input: [3] and newShape: [2, 3, 2], the axis will be 1.
  ///      For input: [3] and newShape: [2, 2, 3], the axis will be 2.
  BroadcastNode *createBroadcast(llvm::StringRef name, NodeValue input,
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
                                       llvm::ArrayRef<dim_t> start,
                                       unsigned_t count = 1,
                                       unsigned_t axis = 0);

  /// Create a slice node \p name with the given starting points for each
  /// dimension \p begin and end points \p end (exclusive).
  SliceNode *createSlice(llvm::StringRef name, NodeValue input,
                         UnsignedArrayRef begin, UnsignedArrayRef end);

  /// Create a slice node with the given starting point for each dimension.
  /// End points will be calculated based on the output type during execution.
  SliceNode *createSlice(llvm::StringRef name, NodeValue input,
                         llvm::ArrayRef<dim_t> start, TypeRef outTy);

  /// Shuffles dimension number \p kernel. Suppose original size is D. It will
  /// be represented as groupX(D/group) matrix, transposed and concatenated back
  /// to size D. For example, shuffle of {1, 2, 3, 4, 5, 6} with \p group = 2 is
  /// {1, 4, 2, 5, 3, 6}
  Node *createChannelShuffle(llvm::StringRef name, NodeValue input,
                             size_t group, size_t kernel);

  /// Computes the indices of the max elements of the input tensor along the
  /// provided \p axis. The resulted tensor has the same rank as the input if \p
  /// keepDims equal 1. If \p keepdims equals 0, the resulted tensor has the
  /// reduced dimension pruned. The type of the output tensor is \p elemTy.
  ArgMaxNode *createArgMax(llvm::StringRef name, NodeValue input,
                           unsigned_t axis, bool keepDims,
                           ElemKind elemTy = ElemKind::Int64ITy);

  /// Computes the indices of the min elements of the input tensor along the
  /// provided \p axis. The resulted tensor has the same rank as the input if \p
  /// keepDims equal 1. If \p keepdims equals 0, the resulted tensor has the
  /// reduced dimension pruned. The type of the output tensor is \p elemTy.
  ArgMinNode *createArgMin(llvm::StringRef name, NodeValue input,
                           unsigned_t axis, bool keepDims,
                           ElemKind elemTy = ElemKind::Int64ITy);

  /// Removes single-dimensional entries from the shape of a tensor. The
  /// parameter \p axes is a list of positive integers, indicating the
  /// dimensions to squeeze. Impelmented as a single ReshapeNode. This is the
  /// opposite of ExpandDims.
  /// https://github.com/onnx/onnx/blob/master/docs/Operators.md#squeeze
  ReshapeNode *createSqueeze(llvm::StringRef name, NodeValue input,
                             llvm::ArrayRef<dim_t> axes);

  /// Add single-dimensional entries to the shape of the \p input tensor at
  /// locations in \p axes. \p axes is listed as seen in the output tensor.
  /// Implemented as a single ReshapeNode. This is the opposite of Squeeze.
  ReshapeNode *createExpandDims(llvm::StringRef name, NodeValue input,
                                llvm::ArrayRef<dim_t> axes);

  /// Flattens the input tensor into a 2D matrix. If input tensor has shape
  /// (d_0, d_1, ... d_n) then the output will have shape:
  /// (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X d_n).
  ReshapeNode *createFlatten(llvm::StringRef name, NodeValue input,
                             unsigned_t axis);

  /// Flattens the input tensor into a 2D matrix. If input tensor has shape
  /// (d_0, d_1, ... d_n) then the output will have shape:
  /// ((d_0 X d_1 ... d_(axis-1) X d_(axis+1) ... X d_n), d_axis).
  ReshapeNode *createFlattenV1(llvm::StringRef name, NodeValue input,
                               unsigned_t axis);

  /// Create \p outputNum slice nodes of \p input. Slices happen along dimension
  /// number \p axis. Array \p split defines lengths of slices. If \p split is
  /// empty, \p input is split to equal sized parts.
  void createSplit(llvm::StringRef name, NodeValue input, unsigned_t outputNum,
                   unsigned_t axis, llvm::ArrayRef<dim_t> split,
                   std::vector<SliceNode *> &outputs);

  BatchNormalizationNode *createBatchNormalization(
      llvm::StringRef name, TypeRef resType, NodeValue input, NodeValue beta,
      NodeValue scale, NodeValue mean, NodeValue var, unsigned_t channelIdx = 0,
      float epsilon = 1e-5, float momentum = 0.9);

  /// Creates and \returns a LayerNormalizationNode that computes the layer
  /// normalization of the inner most layers of \p input based on the shape of
  /// \p scale and \p bias. \p epsilon is a small perterbation used to avoid
  /// division by 0 during normalization.
  LayerNormalizationNode *createLayerNormalization(llvm::StringRef name,
                                                   NodeValue input,
                                                   NodeValue scale,
                                                   NodeValue bias,
                                                   float epsilon = 1e-5);

  /// Bucketizes the input tensor based on monotonically increasing \p
  /// boundaries for each value in \p input. For each value x in input, the
  /// operator \returns index i given boundaries[i-1] < x <= boundaries[i]. If
  /// the value x is beyond the bounds of boundaries, 0 or len(boundaries) is
  /// returned as appropriate.
  BucketizeNode *createBucketizeNode(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<float> boundaries);

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

  /// Create a logical NOT node with name \p name and input \p input.
  NotNode *createNot(llvm::StringRef name, NodeValue input);

#define UNARY_ARITHMETIC_FUN_DECL(NODE_NAME_)                                  \
  NODE_NAME_##Node *create##NODE_NAME_(llvm::StringRef name, NodeValue input); \
  NODE_NAME_##Node *create##NODE_NAME_(llvm::StringRef name, TypeRef Ty,       \
                                       NodeValue input);
  UNARY_ARITHMETIC_FUN_DECL(Abs)
  UNARY_ARITHMETIC_FUN_DECL(Neg)
  UNARY_ARITHMETIC_FUN_DECL(Floor)
  UNARY_ARITHMETIC_FUN_DECL(Sign)
  UNARY_ARITHMETIC_FUN_DECL(Ceil)
  UNARY_ARITHMETIC_FUN_DECL(Round)
  UNARY_ARITHMETIC_FUN_DECL(Sqrt)
  UNARY_ARITHMETIC_FUN_DECL(Rsqrt)
  UNARY_ARITHMETIC_FUN_DECL(Reciprocal)
  UNARY_ARITHMETIC_FUN_DECL(Sin)
  UNARY_ARITHMETIC_FUN_DECL(Cos)
  UNARY_ARITHMETIC_FUN_DECL(Erf)
  UNARY_ARITHMETIC_FUN_DECL(Truncate)
#undef UNARY_ARITHMETIC_FUN_DECL

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
  ARITHMETIC_FUN_DECL(CmpEQ);
  ARITHMETIC_FUN_DECL(CmpNEQ);
  ARITHMETIC_FUN_DECL(CmpLT);
  ARITHMETIC_FUN_DECL(CmpLTE);
  ARITHMETIC_FUN_DECL(And);
  ARITHMETIC_FUN_DECL(Or);
  ARITHMETIC_FUN_DECL(Xor);
  ARITHMETIC_FUN_DECL(Pow);
#undef ARITHMETIC_FUN_DECL

#define TRIGONOMETRIC_FUN_DECL(NODE_NAME_)                                     \
  NODE_NAME_##Node *create##NODE_NAME_(llvm::StringRef name, NodeValue input); \
  NODE_NAME_##Node *create##NODE_NAME_(llvm::StringRef name, TypeRef Ty,       \
                                       NodeValue input);
  TRIGONOMETRIC_FUN_DECL(Acos)
  TRIGONOMETRIC_FUN_DECL(Asin)
  TRIGONOMETRIC_FUN_DECL(Atan)
#undef TRIGONOMETRIC_FUN_DECL

  std::vector<NodeValue>
  broadcastInputs(int axis, const llvm::ArrayRef<NodeValue> inputs);

  template <class T, class U>
  using enable_if_same_t = std::enable_if<std::is_same<T, U>::value, U>;

#define BROADCAST_FUNC_COMMON_CODE(NUM_INPUTS)                                 \
  constexpr size_t numInputs = sizeof...(Args);                                \
  static_assert(numInputs == NUM_INPUTS,                                       \
                "Invalid input passed in to commonCreateBroadcast.");          \
  std::vector<NodeValue> inputs = broadcastInputs(axis, {inputArgs...});

#define DECLARE_BROADCAST_NODE(NODE_NAME, NUM_INPUTS)                          \
  template <class T, class... Args>                                            \
  typename enable_if_same_t<T, NODE_NAME##Node>::type *                        \
  createNodeWithBroadcast(const std::string &name, int axis,                   \
                          Args &&...inputArgs) {                               \
    BROADCAST_FUNC_COMMON_CODE(NUM_INPUTS)                                     \
    return create##NODE_NAME(name, inputs[0].getType(), inputs[0], inputs[1]); \
  }

  /// Template function that creates a node and normalizes its input shapes
  /// with the use of BroadCast nodes. If axis is -1, it calculates it
  /// automatically for multi directional broadcast.
  DECLARE_BROADCAST_NODE(Mul, /* NUM_INPUTS */ 2)
  DECLARE_BROADCAST_NODE(Div, /* NUM_INPUTS */ 2)
  DECLARE_BROADCAST_NODE(Add, /* NUM_INPUTS */ 2)
  DECLARE_BROADCAST_NODE(Sub, /* NUM_INPUTS */ 2)
  DECLARE_BROADCAST_NODE(And, /* NUM_INPUTS */ 2)
  DECLARE_BROADCAST_NODE(Xor, /* NUM_INPUTS */ 2)
  DECLARE_BROADCAST_NODE(Or, /* NUM_INPUTS */ 2)
  DECLARE_BROADCAST_NODE(Pow, /* NUM_INPUTS */ 2)

#define DECLARE_BROADCAST_NODE_WITH_OUT_TYPE(NODE_NAME, NUM_INPUTS,            \
                                             OUTTYPEREF)                       \
  template <class T, class... Args>                                            \
  typename enable_if_same_t<T, NODE_NAME##Node>::type *                        \
  createNodeWithBroadcastOutTy(const std::string &name, int axis,              \
                               TypeRef OUTTYPEREF, Args &&...inputArgs) {      \
    BROADCAST_FUNC_COMMON_CODE(NUM_INPUTS)                                     \
    return create##NODE_NAME(name, OUTTYPEREF, inputs[0], inputs[1]);          \
  }

  DECLARE_BROADCAST_NODE_WITH_OUT_TYPE(Add, /* NUM_INPUTS */ 2, outTy)
  DECLARE_BROADCAST_NODE_WITH_OUT_TYPE(Sub, /* NUM_INPUTS */ 2, outTy)
  DECLARE_BROADCAST_NODE_WITH_OUT_TYPE(Mul, /* NUM_INPUTS */ 2, outTy)
  DECLARE_BROADCAST_NODE_WITH_OUT_TYPE(Div, /* NUM_INPUTS */ 2, outTy)
  DECLARE_BROADCAST_NODE_WITH_OUT_TYPE(Min, /* NUM_INPUTS */ 2, outTy)
  DECLARE_BROADCAST_NODE_WITH_OUT_TYPE(Max, /* NUM_INPUTS */ 2, outTy)

#define DECLARE_CMP_BROADCAST_NODE(NODE_NAME)                                  \
  template <class T, class... Args>                                            \
  typename enable_if_same_t<T, NODE_NAME##Node>::type *                        \
  createNodeWithBroadcast(const std::string &name, int axis,                   \
                          Args &&...inputArgs) {                               \
    BROADCAST_FUNC_COMMON_CODE(2)                                              \
    return create##NODE_NAME(name, inputs[0], inputs[1]);                      \
  }

  /// Template function that creates a node and normalizes its input shapes
  /// with the use of BroadCast nodes. If axis is -1, it calculates it
  /// automatically for multi directional broadcast.
  DECLARE_CMP_BROADCAST_NODE(CmpLT)
  DECLARE_CMP_BROADCAST_NODE(CmpEQ)
  DECLARE_CMP_BROADCAST_NODE(CmpNEQ)
  DECLARE_CMP_BROADCAST_NODE(CmpLTE)
  DECLARE_CMP_BROADCAST_NODE(Min)
  DECLARE_CMP_BROADCAST_NODE(Max)

  /// Template function that creates a node and normalizes its input shapes
  /// with the use of BroadCast nodes. If axis is -1, it calculates it
  /// automatically for multi directional broadcast.
  template <class T, class... Args>
  typename enable_if_same_t<T, SelectNode>::type *
  createNodeWithBroadcast(const std::string &name, int axis,
                          Args &&...inputArgs) {
    BROADCAST_FUNC_COMMON_CODE(3)
    return createSelect(name, inputs[1].getType(), inputs[0], inputs[1],
                        inputs[2]);
  }

#undef BROADCAST_FUNC_COMMON_CODE
#undef DECLARE_BROADCAST_NODE
#undef DECLARE_BROADCAST_NODE_WITH_OUT_TYPE
#undef DECLARE_CMP_BROADCAST_NODE
#undef BROADCAST_FUNC_COMMON_CODE

  /// Create a FloorDivNode with given \p name which divides \p LHS with \p RHS
  /// and floors the quotient. If \p truncate is true then truncates the
  /// quotient instead of flooring.
  FloorDivNode *createFloorDiv(llvm::StringRef name, NodeValue LHS,
                               NodeValue RHS, bool truncate = false);

  /// Create a FloorDivNode with given \p name and output type \p outTy which
  /// divides \p LHS with \p RHS and floors the quotient. If \p truncate is true
  /// then truncates the quotient to zero instead of flooring.
  FloorDivNode *createFloorDiv(llvm::StringRef name, TypeRef outTy,
                               NodeValue LHS, NodeValue RHS,
                               bool truncate = false);

  /// Create a FloorDivNode with given \p name which divides \p LHS with \p RHS
  /// and floors the quotient. If \p truncate is true then truncates the
  /// quotient to zero instead of flooring. The inputs are broadcasted based on
  /// \p axis.
  FloorDivNode *createFloorDivWithBroadcast(llvm::StringRef name, int axis,
                                            NodeValue LHS, NodeValue RHS,
                                            bool truncate = false);

  /// Create a FloorDivNode with given \p name and output type \p outTy which
  /// divides \p LHS with \p RHS and floors the quotient. If \p truncate is true
  /// then truncates the quotient to zero instead of flooring. The inputs are
  /// broadcasted based on \p axis.
  FloorDivNode *createFloorDivWithBroadcast(llvm::StringRef name, int axis,
                                            TypeRef outTy, NodeValue LHS,
                                            NodeValue RHS,
                                            bool truncate = false);

  /// Create an element-wise GREATER THAN comparison between \p LHS and \p RHS
  /// by creating a CmpLTNode with given \p name and swapped inputs.
  CmpLTNode *createCmpGT(llvm::StringRef name, NodeValue LHS, NodeValue RHS);

  /// Create an element-wise GREATER THAN or EQUAL comparison between \p LHS and
  /// \p RHS by creating a CmpLTENode with given \p name and swapped inputs.
  CmpLTENode *createCmpGTE(llvm::StringRef name, NodeValue LHS, NodeValue RHS);

  /// Create a MulNode with given \p name which multiplies \p input with itself
  /// to produce an equivalent Square node.
  MulNode *createSquare(llvm::StringRef name, NodeValue input);

  /// Create a MulNode with given \p name and output type \p outTy which
  /// multiplies \p input with itself to produce an equivalent Square node.
  MulNode *createSquare(llvm::StringRef name, TypeRef outTy, NodeValue input);

  /// Create a LeakyRELU with \p name, \p input and slope \p alpha.
  LeakyReluNode *createLeakyRELU(llvm::StringRef name, NodeValue input,
                                 float alpha);

  /// Create a LeakyRELU with \p name, \p outTy, \p input and slope \p alpha.
  LeakyReluNode *createLeakyRELU(llvm::StringRef name, TypeRef outTy,
                                 NodeValue input, float alpha);

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

  TouchNode *createTouch(llvm::StringRef name, TypeRef ty);

  MatMulNode *createMatMul(llvm::StringRef name, NodeValue lhs, NodeValue rhs);

  MatMulNode *createMatMul(llvm::StringRef name, TypeRef outTy, NodeValue lhs,
                           NodeValue rhs);

  /// \p lhs and \p rhs are 3d matrices, where the leading dimension is the
  /// batch size. For each batch element number i, lhs.slice(i) is multiplied by
  /// rhs.slice(i).
  BatchMatMulNode *createBatchMatMul(llvm::StringRef name, NodeValue lhs,
                                     NodeValue rhs);

  /// Create a node, performing Norm operation. Output type is based on the
  /// input \p p type with dimensions specified with \p axes removed.
  VectorNormNode *createVectorNorm(llvm::StringRef name, NodeValue input,
                                   unsigned_t axis, unsigned_t p = 2);

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

  /// Create a node, performing BatchedReduceMin operation. Output type is
  /// based on the input \p batch type with dimensions specified with \p axes
  /// removed.
  BatchedReduceMinNode *createBatchedReduceMin(llvm::StringRef name,
                                               NodeValue batch,
                                               llvm::ArrayRef<unsigned_t> axes);

  /// Create a node, performing BatchedReduceMax operation. Output type is
  /// based on the input \p batch type with dimensions specified with \p axes
  /// removed.
  BatchedReduceMaxNode *createBatchedReduceMax(llvm::StringRef name,
                                               NodeValue batch,
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

  /// Create a node, performing BatchedReduceProd operation. Output type is
  /// based on the input \p batch type with dimensions specified with \p axes
  /// removed.
  BatchedReduceProdNode *
  createBatchedReduceProd(llvm::StringRef name, NodeValue batch,
                          llvm::ArrayRef<unsigned_t> axes);

  /// Create a node, performing BatchedReduceProd operation. Output type
  /// matches input \p outTy type.
  BatchedReduceProdNode *
  createBatchedReduceProd(llvm::StringRef name, TypeRef outTy, NodeValue batch,
                          llvm::ArrayRef<unsigned_t> axes);

  BatchedAddNode *createBatchedAdd(llvm::StringRef name, NodeValue batch,
                                   NodeValue slice);

  BatchedAddNode *createBatchedAdd(llvm::StringRef name, TypeRef outTy,
                                   NodeValue batch, NodeValue slice);

  BatchedMulNode *createBatchedMul(llvm::StringRef name, NodeValue batch,
                                   NodeValue slice);

  BatchedMulNode *createBatchedMul(llvm::StringRef name, TypeRef outTy,
                                   NodeValue batch, NodeValue slice);

  /// Create a node performing a Cumulative Sum operation, output type matches
  /// \p input type.
  CumSumNode *createCumSum(llvm::StringRef name, NodeValue input,
                           bool exclusive = false, bool reverse = false);

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
  /// to len(Indices). \p lengthsMode and \p avgLength represent meta
  /// information about the \p lengths, allowing the backend to use a
  /// specialized implementation.
  SparseLengthsSumNode *
  createSparseLengthsSum(llvm::StringRef name, NodeValue data,
                         NodeValue indices, NodeValue lengths,
                         LengthsMode lengthsMode = LengthsMode::Variable,
                         float avgLength = NAN);

  /// Same as SparseLengthsSum, but i-th slice is multiplied by weights[i].
  /// len(weights) must be equal to len(indices).
  SparseLengthsWeightedSumNode *createSparseLengthsWeightedSum(
      llvm::StringRef name, NodeValue data, NodeValue weights,
      NodeValue indices, NodeValue lengths,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Create an Embedding node
  /// weights is a 2D tensor capturing the embedding table
  /// indices is a tesnor of arbitrary shape containing the indices to extract
  /// padIdx, if given, zeros the output vector when encounters padIdx
  /// scale, if true, will scale gradients by the inverse of the frequency of
  /// words in mini-batch (currently not supported, default=false)
  /// sparse, if true, gradinet w.r.t. weight matrix will be a sparse tensor
  /// (currently not supported, default=false)
  EmbeddingNode *createEmbedding(llvm::StringRef name, NodeValue weights,
                                 NodeValue indices, int64_t padIdx, bool scale,
                                 bool sparse);

  /// Create an EmbeddingBag node. If \p hasEndOffset is true then the node
  /// expects an extra offset to be appended to \p offsets which marks the end
  /// of the last range. \p lengthsMode and \p avgLength represent meta
  /// information about the \p lengths, allowing the backend to use a
  /// specialized implementation.
  EmbeddingBagNode *createEmbeddingBag(
      llvm::StringRef name, NodeValue data, NodeValue weights,
      NodeValue indices, NodeValue offsets, bool hasEndOffset = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Create an EmbeddingBagByteRowwiseOffsetsNode node. If \p hasEndOffset is
  /// true then the node expects an extra offset to be appended to \p offsets
  /// which marks the end of the last range. \p lengthsMode and \p avgLength
  /// represent meta information about the \p lengths, allowing the backend to
  /// use a specialized implementation.
  EmbeddingBagByteRowwiseOffsetsNode *createEmbeddingBagByteRowwiseOffsets(
      llvm::StringRef name, NodeValue data, NodeValue weights,
      NodeValue indices, NodeValue offsets, bool useFP16Accumulation = false,
      bool hasEndOffset = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Same as \ref createEmbeddingBagByteRowwiseOffsets(), but
  /// expects float input \p data, which is rowwise-quantized and fused
  /// internally. \p fusedElemKind represents the element kind to use for the
  /// final fused rowwise-quantized data. If \p hasEndOffset is true then the
  /// node expects an extra offset to be appended to \p offsets which marks the
  /// end of the last range.
  EmbeddingBagByteRowwiseOffsetsNode *createEmbeddingBagByteRowwiseOffsets(
      llvm::StringRef name, Tensor &data, NodeValue weights, NodeValue indices,
      NodeValue offsets, ElemKind fusedElemKind = ElemKind::UInt8FusedQTy,
      bool useFP16Accumulation = false, bool hasEndOffset = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Same as \ref createSparseLengthsWeightedSum(), but with \p outTy
  /// specified.
  SparseLengthsWeightedSumNode *createSparseLengthsWeightedSum(
      llvm::StringRef name, TypeRef outTy, NodeValue data, NodeValue weights,
      NodeValue indices, NodeValue lengths,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Creates and \returns a node of \p name, performing the SparseLengthsSum
  /// operation, using rowwise quantization for the input \p data with the \p
  /// scales and \p offsets as separate input tensors. Gathers slices of the
  /// outer-most dimension of data indexed by the \p indices vector, and then
  /// accumulates them into len(\p lengths) entries: first Lengths[0] slices are
  /// aggregated to Result[0], next Lengths[1] slices are aggregated to
  /// Result[1], etc. I.e. sum(Lengths) must be equal to len(Indices).
  /// \p precision represents what precision to use for Scale, Offset, and
  /// Result. If \p useFP16Accumulation, then internal arithmetic will use FP16
  /// accumulation; otherwise defaults to FP32. \p lengthsMode and \p avgLength
  /// represent meta information about the \p lengths, allowing the backend to
  /// use a specialized implementation.
  RowwiseQuantizedSparseLengthsWeightedSumNode *
  createRowwiseQuantizedSparseLengthsSum(
      llvm::StringRef name, Storage *data, NodeValue scales, NodeValue offsets,
      NodeValue indices, NodeValue lengths,
      ElemKind precision = ElemKind::FloatTy, bool useFP16Accumulation = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Same as \ref createRowwiseQuantizedSparseLengthsSum(), but expects
  /// float input \p data, which is rowwise-quantized internally.
  RowwiseQuantizedSparseLengthsWeightedSumNode *
  createRowwiseQuantizedSparseLengthsSum(
      llvm::StringRef name, Tensor &data, NodeValue indices, NodeValue lengths,
      quantization::Schema schema, ElemKind precision = ElemKind::FloatTy,
      bool useFP16Accumulation = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Same as \ref createRowwiseQuantizedSparseLengthsSum(), but i-th slice is
  /// multiplied by weights[i]. len(weights) must be equal to len(indices).
  RowwiseQuantizedSparseLengthsWeightedSumNode *
  createRowwiseQuantizedSparseLengthsWeightedSum(
      llvm::StringRef name, Storage *data, NodeValue scales, NodeValue offsets,
      NodeValue weights, NodeValue indices, NodeValue lengths,
      ElemKind precision = ElemKind::FloatTy, bool useFP16Accumulation = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Same as \ref createRowwiseQuantizedSparseLengthsWeightedSum(), but expects
  /// float input \p data, which is rowwise-quantized internally.
  RowwiseQuantizedSparseLengthsWeightedSumNode *
  createRowwiseQuantizedSparseLengthsWeightedSum(
      llvm::StringRef name, Tensor &data, NodeValue weights, NodeValue indices,
      NodeValue lengths, quantization::Schema schema,
      ElemKind precision = ElemKind::FloatTy, bool useFP16Accumulation = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Creates and \returns a node of \p name, performing the SparseLengthsSum
  /// operation, using fused rowwise quantization for the input \p data wherein
  /// the scales and offsets are fused inline with each row of data. \p data
  /// must be of a fused ElemKind. Gathers slices of the outer-most dimension of
  /// data indexed by the \p indices vector, and then accumulates them into
  /// len(\p lengths) entries: first Lengths[0] slices are aggregated to
  /// Result[0], next Lengths[1] slices are aggregated to Result[1], etc.  I.e.
  /// sum(Lengths) must be equal to len(Indices).  The precision for the Result
  /// is determined by the \p data input's ElemKind used for Scale and
  /// Offset. If \p useFP16Accumulation, then internal arithmetic will use FP16
  /// accumulation; otherwise defaults to FP32. \p lengthsMode and \p avgLength
  /// represent meta information about the \p lengths, allowing the backend to
  /// use a specialized implementation.
  FusedRowwiseQuantizedSparseLengthsSumNode *
  createFusedRowwiseQuantizedSparseLengthsSum(
      llvm::StringRef name, Storage *data, NodeValue indices, NodeValue lengths,
      bool useFP16Accumulation = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Same as \ref createFusedRowwiseQuantizedSparseLengthsSum(), but expects
  /// float input \p data, which is rowwise-quantized and fused internally.
  /// \p fusedElemKind represents the element kind to use for the final fused
  /// rowwise-quantized data.
  FusedRowwiseQuantizedSparseLengthsSumNode *
  createFusedRowwiseQuantizedSparseLengthsSum(
      llvm::StringRef name, Tensor &data, NodeValue indices, NodeValue lengths,
      ElemKind fusedElemKind = ElemKind::UInt8FusedQTy,
      bool useFP16Accumulation = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Same as \ref createFusedRowwiseQuantizedSparseLengthsSum(), but i-th slice
  /// is multiplied by weights[i]. len(weights) must be equal to len(indices).
  FusedRowwiseQuantizedSparseLengthsWeightedSumNode *
  createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      llvm::StringRef name, NodeValue data, NodeValue weights,
      NodeValue indices, NodeValue lengths, bool useFP16Accumulation = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

  /// Same as \ref createFusedRowwiseQuantizedSparseLengthsWeightedSum(), but
  /// expects float input \p data, which is rowwise-quantized and fused
  /// internally. \p fusedElemKind represents the element kind to use for the
  /// final fused rowwise-quantized data.
  FusedRowwiseQuantizedSparseLengthsWeightedSumNode *
  createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      llvm::StringRef name, Tensor &data, NodeValue weights, NodeValue indices,
      NodeValue lengths, ElemKind fusedElemKind = ElemKind::UInt8FusedQTy,
      bool useFP16Accumulation = false,
      LengthsMode lengthsMode = LengthsMode::Variable, float avgLength = NAN);

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
                          NodeValue lengths, llvm::ArrayRef<dim_t> mask);

  SaveNode *createSave(llvm::StringRef name, NodeValue input);

  /// Creates and \returns a SaveNode of \p input to \p output. If \p skipSuffix
  /// then the name used is \p name, otherwise suffix "_save" is appended.
  SaveNode *createSave(llvm::StringRef name, NodeValue input,
                       Placeholder *output, bool skipSuffix = false);

  /// Create quantization profile node named \p name for the output tensor from
  /// \p input in PlaceholderBindings \p bindings. Capture observed node name in
  /// quantization profile node as original node can be replaced during lowering
  /// phase. Compute the histogram during profiling with \p numHistogramBins.
  QuantizationProfileNode *
  createQuantizationProfile(PlaceholderBindings &bindings, llvm::StringRef name,
                            NodeValue input, dim_t numHistogramBins = 10);

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

  TopKNode *createTopK(llvm::StringRef name, NodeValue input, unsigned_t k,
                       ElemKind outIndicesTyKind);

  /// Gathers entries of the outer-most dimension of \p data indexed by
  /// \p indices, and concatenates them. A non-zero \p batchDims specifies the
  /// batch, and the result is the concatenation of the operation on each sample
  /// in the batch.
  GatherNode *createGather(llvm::StringRef name, NodeValue data,
                           NodeValue indices, unsigned_t batchDims = 0);

  /// Given \p data tensor of rank r >= 1, \p indices tensor of rank q >= 1,
  /// and batch_dims integer b, this operator gathers slices of data
  /// into an output tensor of rank q + r - indices_shape[-1] - 1 - b.
  GatherNDNode *createGatherND(llvm::StringRef name, NodeValue data,
                               NodeValue indices);

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
  /// result is {{1,2},{-3,-4},{5,6}}. If \p cumulative is true, this node adds
  /// values instead of copying.
  ScatterDataNode *createScatterData(llvm::StringRef name, NodeValue data,
                                     NodeValue indices, NodeValue slices,
                                     bool cumulative = false);

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

  /// Create a sequence of Reshape and Transpose nodes representing DepthToSpace
  /// operator with \p blockSize in DCR or CRD mode based on \p isCRD flag.
  /// Assumes input layout to be NHWC. \returns the last node in the sequence.
  ReshapeNode *createDepthToSpace(llvm::StringRef name, NodeValue input,
                                  unsigned blockSize, bool isCRD = false);

  /// Given \p input tensor of [N,H,W,C], where N is the batch, C is the channel
  /// or depth, H is the height and W is the width, and \p scale tensor with
  /// tensor format same as \p input then ResizeNearest generates an Output
  /// tensor with resized spatial dimensions using nearest neighbor
  /// interpolation. The Output tensor is of shape [floor(N * \p scale[0]),
  /// floor(H * \p scale[1]), floor(W * \p scale[2]),
  /// floor(C * \p scale[3])]
  ResizeNearestNode *createResizeNearest(llvm::StringRef name, NodeValue input,
                                         llvm::ArrayRef<float> scale);

  /// Given \p input tensor of [N,H,W,C], where N is the batch, C is the channel
  /// or depth, H is the height and W is the width, with tensor format same as
  /// \p input then ResizeNearest generates an Output tensor with resized
  /// spatial dimensions using nearest neighbor interpolation. The Output tensor
  /// shape is specified with \p outTy.
  ResizeNearestNode *createResizeNearest(llvm::StringRef name, NodeValue input,
                                         TypeRef outTy);

  /// Given \p input tensor of [N,H,W,C], where N is the batch, C is the channel
  /// or depth, H is the height and W is the width, and \p scale tensor with
  /// tensor format same as \p input then ResizeBilinear generates an Output
  /// tensor with resized spatial dimensions using bilinear neighbor
  /// interpolation. The Output tensor is of shape [floor(N * \p scale[0]),
  /// floor(H * \p scale[1]), floor(W * \p scale[2]),
  /// floor(C * \p scale[3])]
  ResizeBilinearNode *createResizeBilinear(llvm::StringRef name,
                                           NodeValue input,
                                           llvm::ArrayRef<float> scale);

  /// Given \p input tensor of [N,H,W,C], where N is the batch, C is the channel
  /// or depth, H is the height and W is the width, with tensor format same as
  /// \p input then ResizeBilinear generates an Output tensor with resized
  /// spatial dimensions using bilinear neighbor interpolation. The Output
  /// tensor shape is specified with \p outTy.
  ResizeBilinearNode *createResizeBilinear(llvm::StringRef name,
                                           NodeValue input, TypeRef outTy);

  /// Create quantization node which transforms floating point tensor to a
  /// quantized one with given Scale and Offset. Scale and Offset params are
  /// part of the \p outTy.
  QuantizeNode *createQuantize(llvm::StringRef name, NodeValue input,
                               TypeRef outTy);

  /// Create quantization node which transforms floating point tensor to a
  /// quantized one of kind \p q with given \p scale and \p offset.
  QuantizeNode *createQuantize(llvm::StringRef name, NodeValue input,
                               ElemKind q, float scale, int32_t offset);

  /// Create dequantization node which transforms quantized tensor to a
  /// floating point one with given Scale and Offset. Scale and Offset params
  /// are part of the \p input. Result dequantization kind is \p k.
  DequantizeNode *createDequantize(llvm::StringRef name, NodeValue input,
                                   ElemKind k);

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

  /// Create a Clip node with the given \p name, \p input, minimum clip value
  /// \p min, maximum clip value \p max and output type \p outTy.
  ClipNode *createClip(llvm::StringRef name, NodeValue input, TypeRef outTy,
                       float min, float max);

  /// Create a Clip node with the given \p name, \p input, minimum clip value
  /// \p min, maximum clip value \p max. Result type will be implicitly set
  /// based on the \p input type.
  ClipNode *createClip(llvm::StringRef name, NodeValue input, float min,
                       float max);

  /// Creates and \returns a ClipNode to the min/max range of FP16 with \p name
  /// of \p input. Result type will be implicitly set based on the \p input
  /// type.
  ClipNode *createClipMinMaxFP16(llvm::StringRef name, NodeValue input);

  /// Creates and \returns a ClipNode to the min/max range of BFloat16 with \p
  /// name of \p input. Result type will be implicitly set based on the \p input
  /// type.
  ClipNode *createClipMinMaxBFloat16(llvm::StringRef name, NodeValue input);

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
  /// axis. \p layout defines the Tensor layout and must be either NHWC or NCHW.
  ConvolutionNode *createConv(PlaceholderBindings &bindings,
                              llvm::StringRef name, NodeValue input,
                              dim_t outChannels,
                              llvm::ArrayRef<unsigned_t> kernels,
                              llvm::ArrayRef<unsigned_t> strides,
                              llvm::ArrayRef<unsigned_t> pads, unsigned_t group,
                              llvm::ArrayRef<unsigned_t> dilation = {1, 1},
                              ConvolutionLayout layout = NHWC);

  /// Creates a ConvolutionNode with the given \p name which convolves the 4D
  /// \p input. \p kernel defines the size of the height and width dimensions of
  /// the convolutional filters. \p stride defines the the number of steps to
  /// take in the input for each output cell. \p pad defines how many zero
  /// padding cells should be added to the input during convolution. \p group
  /// defines the number of groups the input and output channels should be
  /// divided into and convolved separately.\p dilation defines factor by
  /// which gap between 2 neighboring kernel elements is expanded along each
  /// axis. \p layout defines the Tensor layout and must be either NHWC or NCHW.
  ConvolutionNode *createConv(PlaceholderBindings &bindings,
                              llvm::StringRef name, NodeValue input,
                              dim_t outChannels, unsigned_t kernel,
                              unsigned_t stride, unsigned_t pad,
                              unsigned_t group,
                              llvm::ArrayRef<unsigned_t> dilation = {1, 1},
                              ConvolutionLayout layout = NHWC);

  /// Creates a Convolution3DNode with the given \p name which convolves the 5D
  /// \p input. \p kernels defines the size of the height, width, and depth
  /// dimensions of the convolutional filters. \p strides defines the the number
  /// of steps to take in the input for each output cell. \p pads defines how
  /// many zero padding cells should be added to the input during convolution.
  /// \p group defines the number of groups the input and output channels should
  /// be divided into and convolved separately.
  Convolution3DNode *createConv3D(PlaceholderBindings &bindings,
                                  llvm::StringRef name, NodeValue input,
                                  dim_t outChannels,
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

  /// Creates a ConvTransposeNode with the given \p name which does transposed
  /// convolution on the 4D \p input. \p kernels define the size of the height
  /// and width dimensions of the convolution filters. \p strides define the
  /// number of steps to take in the input for each output cell. \p pads define
  /// how many zero padding cells should be added to the input during
  /// convolution. \p group defines the number of groups the input and output
  /// channels should be divided into and convolved separately.
  ConvTransposeNode *createConvTranspose(
      PlaceholderBindings &bindings, llvm::StringRef name, NodeValue input,
      dim_t outChannels, llvm::ArrayRef<unsigned_t> kernels,
      llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
      unsigned_t group, llvm::ArrayRef<unsigned_t> dilation = {1, 1});

  /// Creates a ConvTransposeNode with the given \p name which does transposed
  /// convolution on the 4D \p input. \p kernel defines the size of the height
  /// and width dimensions of the convolution filters. \p stride defines the
  /// number of steps to take in the input for each output cell. \p pad defines
  /// how many zero padding cells should be added to the input during
  /// convolution. \p group defines the number of groups the input and output
  /// channels should be divided into and convolved separately.
  ConvTransposeNode *
  createConvTranspose(PlaceholderBindings &bindings, llvm::StringRef name,
                      NodeValue input, dim_t outChannels, unsigned_t kernel,
                      unsigned_t stride, unsigned_t pad, unsigned_t group,
                      llvm::ArrayRef<unsigned_t> dilation = {1, 1});

  /// Creates and \returns a FullyConnectedNode with \p name, \p input, weights
  /// \p W, bias \p B. If \p input is not 2 dimensional then it is flattened
  /// along \p axis. Note, output type is inferred based on the input
  /// types. Trainable weight and bias variables are created implicitly.
  FullyConnectedNode *createFullyConnected(PlaceholderBindings &bindings,
                                           llvm::StringRef name,
                                           NodeValue input, dim_t outDepth,
                                           unsigned_t axis = 1);

  /// Creates an RMSNorm pair. \p X should be a 2D tensor, \p gamma and \p beta
  /// should be 1D tensors.
  std::array<Node *, 2> createRMSNorm(llvm::StringRef name, NodeValue X,
                                      NodeValue gamma, NodeValue beta,
                                      float epsilon = .0f);

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

  /// Create an LSTM Unit Node with \p Input which shape is [batch,
  /// 4*hiddenSize] and follow the order i, f, g, o, and \p C as current cell
  /// state.
  LSTMUnitNode *createLSTMUnit(llvm::StringRef namePrefix, NodeValue Input,
                               NodeValue C);

  /// Helper function create a PyTorch style LSTM for one direction, and returns
  /// every output in a vector. \p T should be an iterator or reverse_iterator
  /// of a NodeValue vector, and /p inputItr is an iterator pointer of the input
  /// vector. \p Wx, \p Wh, \p Bx, \p Bh, \p H and \p C is i, f, g, o, hidden
  /// state and cell state, whose shape should be the same to
  /// createSingleDirectionLSTM.
  template <class T>
  std::vector<NodeValue> createSingleDirectionLSTM(
      std::string nameBase, T inputItr, const int timeSteps, NodeValue Wx,
      NodeValue Wh, NodeValue Bx, NodeValue Bh, NodeValue &H, NodeValue &C);

  /// Create PyTorch style LSTM with fixed weights and biases.
  /// The order of \p Wx \p Wh \p Bx and \p Bh is i, f, g, o,
  /// The \p inputs shape should be (numSteps, batchSize, hiddenSize),
  /// while \p Wx shape should be (inputSize, hiddenSize * 4),
  /// Wh shape should be (hiddenSize, hiddenSize * 4),
  /// \p Bx and \p Bh shape should be (hiddenSize * 4).
  /// If \p isBidirectional == true, \p WxR, \p WhR, \p BxR and \p BhR
  /// also need to be provided, indicates the reversed weights and biases.
  /// \p Ht and \p Ct are initial hidden state and cell.
  /// For more details, please read:
  /// https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
  void createPyTorchLSTM(llvm::StringRef namePrefix, NodeValue inputs,
                         NodeValue Wx, NodeValue Wh, NodeValue Bx, NodeValue Bh,
                         NodeValue &Ht, NodeValue &Ct, NodeValue &outputs,
                         bool isBidirectional = false,
                         NodeValue WxR = NodeValue(),
                         NodeValue WhR = NodeValue(),
                         NodeValue BxR = NodeValue(),
                         NodeValue BhR = NodeValue());

  /// Type definition for the direction of an RNN module (RNN, GRU, LSTM).
  enum class RnnDirection {
    Forward,
    Reverse,
    Bidirectional,
  };

  /// Definition for a lambda used to create an activation node for RNN modules.
  using RnnActivation = std::function<Node *(llvm::StringRef, Node *)>;

  /// Create an unrolled multi-layer RNN according to the ONNX definition:
  /// https://github.com/onnx/onnx/blob/master/docs/Operators.md#RNN
  /// The RNN has the following inputs:
  /// - input \p X with size [S, B, ISize].
  /// - weigts \p W with size [N, HSize, ISize].
  /// - reccurence weights \p R with size [N, HSize, HSize].
  /// - bias weights \p B with size [N, 2 * HSize].
  /// - initial hidden state \p initial_h with size [N, B, HSize].
  /// where S is the sequence length, N is the number of directions, B is the
  /// batch size, ISize is the input size and HSize is the hidden size.
  /// The RNN has the following outputs:
  /// - output \p Y with size [S, N, B, HSize].
  /// - final hidden state \p Y_h with size [N, B, HSize].
  /// The direction of the instatiated RNN is given by \p direction. The RNN
  /// will use the activation functions defined by the \p activations array:
  /// - [f] in case the RNN is unidirectional (1 function).
  /// - [f] for the forward cell followed by [f] for the reverse cell in
  ///    case the RNN is bidirectional (4 functions).
  /// The inputs \p B and \p initial_h are optional (assumed 0 if nullptr is
  /// provided). The names of all the nodes created are prefixed with
  /// \p namePrefix.
  void createOnnxRNN(llvm::StringRef namePrefix, NodeValue X, NodeValue W,
                     NodeValue R, NodeValue B, NodeValue initial_h,
                     NodeValue &Y, NodeValue &Y_h, unsigned hiddenSize,
                     RnnDirection direction,
                     std::vector<RnnActivation> &activations);

  /// Create an unrolled multi-layer GRU according to the ONNX definition:
  /// https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
  /// The GRU has the following inputs:
  /// - input \p X with size [S, B, ISize].
  /// - weigts \p W with size [N, 3 * HSize, ISize].
  /// - reccurence weights \p R with size [N, 3 * HSize, HSize].
  /// - bias weights \p B with size [N, 6 * HSize].
  /// - initial hidden state \p initial_h with size [N, B, HSize].
  /// where S is the sequence length, N is the number of directions, B is the
  /// batch size, ISize is the input size and HSize is the hidden size.
  /// The GRU has the following outputs:
  /// - output \p Y with size [S, N, B, HSize].
  /// - final hidden state \p Y_h with size [N, B, HSize].
  /// The direction of the instatiated GRU is given by \p direction. The GRU
  /// will use the activation functions defined by the \p activations array:
  /// - [f,g] in case the GRU is unidirectional (2 functions).
  /// - [f,g] for the forward cell followed by [f,g] for the reverse cell in
  ///    case the GRU is bidirectional (4 functions).
  /// The inputs \p B and \p initial_h are optional (assumed 0 if nullptr is
  /// provided). The names of all the nodes created are prefixed with
  /// \p namePrefix. The boolean parameter \p linearBeforeReset defines whether
  /// the reset for the previous hidden state occurs before/after the linear
  /// expression.
  void createOnnxGRU(llvm::StringRef namePrefix, NodeValue X, NodeValue W,
                     NodeValue R, NodeValue B, NodeValue initial_h,
                     NodeValue &Y, NodeValue &Y_h, unsigned hiddenSize,
                     RnnDirection direction,
                     std::vector<RnnActivation> &activations,
                     bool linearBeforeReset = false);

  /// Create an unrolled multi-layer LSTM according to the ONNX definition:
  /// https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
  /// The LSTM has the following inputs:
  /// - input \p X with size [S, B, ISize].
  /// - weigts \p W with size [N, 4 * HSize, ISize].
  /// - reccurence weights \p R with size [N, 4 * HSize, HSize].
  /// - bias weights \p B with size [N, 8 * HSize].
  /// - initial hidden state \p initial_h with size [N, B, HSize].
  /// - initial cell state \p initial_c with size [N, B, HSize].
  /// - peephole weights \p P with size [N, 3 * HSize].
  /// where S is the sequence length, N is the number of directions, B is the
  /// batch size, ISize is the input size and HSize is the hidden size.
  /// The LSTM has the following outputs:
  /// - output \p Y with size [S, N, B, HSize].
  /// - final hidden state \p Y_h with size [N, B, HSize].
  /// - final cell state \p Y_c with size [N, B, HSize].
  /// The direction of the instatiated LSTM is given by \p direction. The LSTM
  /// will use the activation functions defined by \p activations array:
  /// - [f,g,h] in case the LSTM is unidirectional (3 functions).
  /// - [f,g,h] for the forward cell followed by [f,g,h] for the reverse cell in
  ///    case the LSTM is bidirectional (6 functions).
  /// The inputs \p B, \p initial_h, \p initial_c and \p P are optional (assumed
  /// 0 if nullptr is provided). The names of all the nodes created are prefixed
  /// with \p namePrefix. The boolean parameter \p inputForget defines whether
  /// the input and forget gates should be coupled (compute the input gate from
  /// the forget gate).
  void createOnnxLSTM(llvm::StringRef namePrefix, NodeValue X, NodeValue W,
                      NodeValue R, NodeValue B, NodeValue initial_h,
                      NodeValue initial_c, NodeValue P, NodeValue &Y,
                      NodeValue &Y_h, NodeValue &Y_c, unsigned hiddenSize,
                      RnnDirection direction,
                      std::vector<RnnActivation> &activations,
                      bool inputForget = false);
  /// @}

  /// Create a TraceEvent in the runtime profile, which triggers collection of
  /// runtime statistics.
  TraceEventNode *createTraceEvent(llvm::StringRef eventName,
                                   llvm::StringRef eventType, Node *data,
                                   unsigned index);

  /// Creates NMSv4 node that does NMS for one class.
  /// Inputs
  /// - \p boxes Tensor with box coordinates.
  /// - \p scores Tensor with scores per box.
  /// - \p centerPointBox Indicates format of the box per ONNX spec.
  /// - \p iouThreshold Threshold for box overlap.
  /// - \p scoreThreshold Threshold for box scores.
  NonMaxSuppressionNode *
  createNonMaxSuppressionV4(llvm::StringRef name, NodeValue boxes,
                            NodeValue scores, int64_t centerPointBox,
                            int64_t maxOutputBoxesPerClass, float iouThreshold,
                            float scoreThreshold);

  /// Creates NMSv4 node that does NMS for one class.
  /// Inputs
  /// - \p boxes Tensor with box coordinates.
  /// - \p scores Tensor with scores per box.
  /// - \p centerPointBox Indicates format of the box per ONNX spec.
  /// - \p iouThreshold Threshold for box overlap.
  /// - \p scoreThreshold Threshold for box scores.
  /// - \p ElemKind Output ElemKind.
  NonMaxSuppressionNode *
  createNonMaxSuppressionV4(llvm::StringRef name, NodeValue boxes,
                            NodeValue scores, int64_t centerPointBox,
                            int64_t maxOutputBoxesPerClass, float iouThreshold,
                            float scoreThreshold, ElemKind elTy);

  /// Creates NMSv4 node that does NMS for one class.
  /// Inputs
  /// - \p boxes Tensor with box coordinates.
  /// - \p scores Tensor with scores per box.
  /// - \p centerPointBox Indicates format of the box per ONNX spec.
  /// - \p iouThreshold Threshold for box overlap.
  /// - \p scoreThreshold Threshold for box scores.
  /// - \p indicesTy Type of indices output.
  /// - \p numberOfSelectedIndicesTy \p Type of second output for number of
  /// boxes detected.
  NonMaxSuppressionNode *
  createNonMaxSuppressionV4(llvm::StringRef name, NodeValue boxes,
                            NodeValue scores, int64_t centerPointBox,
                            int64_t maxOutputBoxesPerClass, float iouThreshold,
                            float scoreThreshold, TypeRef indicesTy,
                            TypeRef numberOfSelectedIndicesTy);

  /// Performs class wise NMS based on ONNX specification, with padding and ONNX
  /// layout output.
  /// Inputs
  /// - \p boxes Tensor with box coordinates.
  /// - \p scores Tensor with scores per box.
  /// - \p centerPointBox Indicates format of the box per ONNX spec.
  /// - \p iouThreshold Threshold for box overlap.
  /// - \p scoreThreshold Threshold for box scores.
  NonMaxSuppressionNode *
  createNonMaxSuppressionONNX(llvm::StringRef name, NodeValue boxes,
                              NodeValue scores, int64_t centerPointBox,
                              int64_t maxOutputBoxesPerClass,
                              float iouThreshold, float scoreThreshold);

  /// Performs class wise NMS based on ONNX specification, with padding and ONNX
  /// layout output.
  /// Inputs
  /// - \p boxes Tensor with box coordinates.
  /// - \p scores Tensor with scores per box.
  /// - \p centerPointBox Indicates format of the box per ONNX spec.
  /// - \p iouThreshold Threshold for box overlap.
  /// - \p scoreThreshold Threshold for box scores.
  NonMaxSuppressionNode *createNonMaxSuppressionONNX(
      llvm::StringRef name, NodeValue boxes, NodeValue scores,
      int64_t centerPointBox, int64_t maxOutputBoxesPerClass,
      float iouThreshold, float scoreThreshold, ElemKind elTy);

  /// Performs class wise NMS based on ONNX specification, with padding and ONNX
  /// layout output.
  /// Inputs
  /// - \p boxes Tensor with box coordinates.
  /// - \p scores Tensor with scores per box.
  /// - \p centerPointBox Indicates format of the box per ONNX spec.
  /// - \p iouThreshold Threshold for box overlap.
  /// - \p scoreThreshold Threshold for box scores.
  NonMaxSuppressionNode *createNonMaxSuppressionONNX(
      llvm::StringRef name, NodeValue boxes, NodeValue scores,
      int64_t centerPointBox, int64_t maxOutputBoxesPerClass,
      float iouThreshold, float scoreThreshold, TypeRef indicesTy);

  /// Create a constant node with a 1D cosine windowing function defined as:
  /// w[n] = 0.5 - 0.5 * cos(2 * pi * n / N) for n = 0 .. N - 1 where N
  /// is the window \p length. The node name will be \p name.
  Constant *createCosineWindow(llvm::StringRef name, dim_t length);

  /// Create a constant node with the twiddle factors for a 1D complex FFT:
  /// W(N, k) = exp(-j * 2 * pi * k / N) for k = 0 ... N -1, where N is the
  /// \p fftLength. The constant node will contain 2 * \p fftLength real float
  /// values corresponding to \p fftLength complex values with the real and
  /// imaginary parts interleaved: real[0], imag[0], real[1], imag[1], etc.
  /// The node name will be \p name.
  Constant *createFFTTwiddleFactors(llvm::StringRef name, dim_t fftLength);

  /// Create a constant node with the bit reverse indices for a 1D FFT, that
  /// is the corresponding index obtained after reversing the bit order for
  /// each of the values k = 0 ... N -1 where N is the \p fftLength. The node
  /// will contain \p fftLength int32 values. The node name will be \p name.
  Constant *createFFTBitReverseIndices(llvm::StringRef name, dim_t fftLength);

  /// Create a constant node with the complex weights used to map the results
  /// of N/2 point complex FFT to a N point real FFT. This allows an efficient
  /// implementation of the N point FFT for a real data x[n] with n = 0 .. N-1
  /// by first computing the N/2 complex FFT G[k] for the complex signal g[n]
  /// defined as g[n] = x[2*n+0] + j * x[2*n+1] with n = 0 ... N/2-1 and then
  /// computing the final N point FFT X[k] for the original data x[n] by using
  /// X[k] = G[k] * A[k] + conj(G[N/2-k]) * (1 - A[k]) for k = 0 ... N/2 (for
  /// a real signal the FFT is conjugate symmetrical and therefore only the
  /// first N/2+1 output points of X[k] should be computed, the others being
  /// redundant). The relation should also use the definitions G[N/2] = G[0] and
  /// then A[k] = 1/2 * (1 - j * exp(-j * 2 * pi * k / N)) for k = 0 ... N/2.
  /// The FFT length parameter N is given as \p fftLength. This constant node
  /// will contain the complex values of A[k] for k = 0 ... L-1 where L is the
  /// sequence length given as \p outLength (the required length L is smaller
  /// than N/2+1 since A[k] has such properties that the second half of the
  /// sequence can be easily deduced from first half). This constant node will
  /// contain 2 * \p outLength real float values corresponding to \p outLength
  /// complex values A[k] with the real and imaginary parts interleaved.
  Constant *createFFTComplexToRealWeights(llvm::StringRef name, dim_t fftLength,
                                          dim_t outLength);

  /// This node computes the spectrogram of a 1D mono audio signal \p input by
  /// extracting windows of size \p windowSize with stride \p windowStride and
  /// computing for each window the spectrum power (magnitude squared) or simply
  /// the magnitude depending on the flag \p magnitudeSquared. If the length of
  /// the \p input is [inputLength] samples then the size of the spectrogram is
  /// [windowCount, fftLength/2+1] where:
  /// - windowCount = floor((inputLength-windowSize)/windowStride)+1 is the
  ///   number of windows extracted from the input.
  /// - fftLength is the FFT length used to compute the spectrogram which is the
  ///   next power of 2 (e.g. for a window size of 640 the fftLength is 1024).
  /// The input audio data values are commonly float values scaled in the range
  /// [-1.0, 1.0]. If the audio data is decoded from a WAV file into int8/int16
  /// values then those values are commonly scaled down with 2^7/2^15 before
  /// using this node. The node name will be \p name. This node is inspired from
  /// TensorFlow (tensorflow.python.ops.gen_audio_ops.audio_spectrogram).
  AudioSpectrogramNode *createAudioSpectrogram(llvm::StringRef name,
                                               NodeValue input,
                                               int64_t windowSize,
                                               int64_t windowStride,
                                               bool magnitudeSquared = true);

  /// Create as constants the Mel weights \p melWeights and ranges \p melRanges
  /// required for the MFCC (Mel Frequency Cepstral Coefficient) transform for a
  /// spectrogram of length \p spectrogramLength (which must be of the form
  /// 2 ^ N + 1) obtained for an audio signal with the given \p sampleRate
  /// (in Hertz) by mapping the spectrogram coefficients in \p filterBankCount
  /// bins on a Mel scale between \p lowerFrequency and \p upperFrequency
  /// (in Hertz) using a filterbank of triangular windows. The constant nodes
  /// will be named using \p prefix.
  void createMelWeights(llvm::StringRef prefix, dim_t spectrogramLength,
                        float sampleRate, float lowerFrequency,
                        float upperFrequency, dim_t filterBankCount,
                        Constant *&melWeights, Constant *&melRanges);

  /// Create the DCT-II transform matrix coefficients as a constant defined as:
  /// d[k][n] = sqrt(2 / N) * cos(pi / N * (n + 1/2) * k) with n = 0 .. N - 1
  /// and k = 0 .. K - 1 where \p N is the input data length and \p K is the
  /// output data length. The common case is that for which the input length
  /// \p N is equal to the output length \p K but a separate output length
  /// argument \p K <= \p N allows creating a partial DCT matrix used to compute
  /// only the first \p K results from the full DCT-II transform. The DCT matrix
  /// size will be \p K x \p N.  The node name will be \p name.
  Constant *createDCTMat(llvm::StringRef name, dim_t N, dim_t K);

  /// Computes the MFCC (Mel Frequency Cepstral Coefficient) for the given
  /// \p spectrogram and is commonly used as feature extractor for voice/speech
  /// audio data in voice command or keyword spotting applications. The input
  /// \p spectrogram is a power spectrogram and not a magnitude (computed using
  /// the 'AudioSpectrogram' node with the 'magnitudeSquared' flag set to True).
  /// The MFCC transform is computed using the given \p sampleRate (in Hertz)
  /// by mapping the spectrogram coefficients in \p filterBankCount bins on a
  /// Mel scale between \p lowerFrequency and \p upperFrequency (in Hertz) using
  /// a filterbank of triangular windows, taking the natural logarithm and then
  /// keeping the first \p numCoefficients from the DCT-II transform. If the
  /// input \p spectrogram size is [windowCount, spectrogramLen] then the output
  /// node size will be [windowCount, numCoefficients] since the MFCC transform
  /// is performed separately for each window of [spectrogramLen] input samples
  /// by yielding \p numCoefficients output samples. This node is inspired from
  /// TensorFlow (tensorflow.python.ops.gen_audio_ops.mfcc).
  MFCCNode *createMFCC(llvm::StringRef name, NodeValue spectrogram,
                       float sampleRate, float lowerFrequency,
                       float upperFrequency, int64_t filterBankCount,
                       int64_t numCoefficients);

  /// Performs the ROIAlign operation given the \p featureMap and the \p boxes.
  /// ROIAlign is similar to crop and resize followed by pooling. The
  /// co-ordinates to extract the crops are specified in \p boxes. Each cropped
  /// image has to be resized to have the shape specified by \p outputHeight and
  /// \p outputWidth. The \p samplingRatio specifies the number of samples to
  /// take from each bin (along both the dimensions) for the purpose of pooling.
  /// This node is defined in:
  /// (https://github.com/onnx/onnx/blob/master/docs/Operators.md#RoiAlign).
  /// \p aligned flag is an addition to Onnx definition to indicate if box
  /// coordinates are aligned to the center of a pixel (VS top-left corner).
  ROIAlignNode *createROIAlign(llvm::StringRef name, NodeValue featureMap,
                               NodeValue boxes, NodeValue batchIndices,
                               uint32_t outputHeight, uint32_t outputWidth,
                               uint32_t samplingRatio, float spatialScale,
                               bool aligned, bool rotated = false,
                               PoolingMode mode = PoolingMode::AVG);

  /// Transform proposal bounding boxes to target bounding box using bounding
  /// box regression deltas.
  /// Inputs:
  /// \p rois - Bounding box proposals in pixel coordinates.
  ///   Size (M, 4), format [x1, y1, x2, y2], or
  ///   Size (M, 5), format [batch_index, x1, y1, x2, y2].
  ///   If proposals from multiple images in a batch are present, they
  ///   should be grouped sequentially and in incremental order.
  ///   For rotated boxes, this would have an additional angle (in degrees)
  ///   in the format [<optionaal_batch_id>, ctr_x, ctr_y, w, h, angle].
  /// \p deltas - bounding box translations and scales,
  ///   size (M, 4*K), format [dx, dy, dw, dh], K = # classes.
  ///   For rotated boxes, size (M, 5*K, format [dx, dy, dw, dh, da].)
  /// \p imInfo - Image dimensions, size (batch_size, 3),
  ///   format [img_height, img_width, img_scale]
  /// Arguments:
  /// \p weights - vector<float> weights [wx, wy, ww, wh] for the deltas
  /// \p applyScale - transform the boxes to the scaled image space after
  /// applying the bbox deltas. Set to false to match the detectron code, set to
  /// true for keypoint models and for backward compatibility rotated - If true,
  /// then boxes (rois and deltas) include angle info to handle rotation. The
  /// format will be [ctr_x, ctr_y, width, height, angle (in degrees)].
  /// \p angleBoundOn - If set, for rotated boxes, angle is normalized to be
  /// within [angle_bound_lo, angle_bound_hi].
  /// \p angleBoundLo - If set, for rotated boxes, angle is normalized to be
  /// within [angle_bound_lo, angle_bound_hi].
  /// \p angleBoundHi - If set, for rotated boxes, angle is normalized to be
  /// within [angle_bound_lo, angle_bound_hi].
  /// \p clipAngleThresh - For RRPN, clip almost horizontal boxes within this
  /// threshold of tolerance for backward compatibility. Set to negative value
  /// for no clipping.
  /// Outputs:
  /// boxOut - Pixel coordinates of the transformed bounding boxes,
  /// Size (M, 4*K), format [x1, y1, x2, y2]. For rotated boxes, size (M, 5*K),
  /// format [ctr_x, ctr_y, w, h, angle].
  /// roiBatchSplits - Tensor of shape (batch_size) with each element
  /// denoting the number of RoIs belonging to the corresponding image in batch
  /// See definition:
  /// https://github.com/pytorch/pytorch/blob/master/caffe2/operators/bbox_transform_op.cc#L10
  BBoxTransformNode *
  createBBoxTransform(llvm::StringRef name, NodeValue rois, NodeValue deltas,
                      NodeValue imInfo, llvm::ArrayRef<float> weights,
                      bool applyScale, bool rotated, bool angleBoundOn,
                      int64_t angleBoundLo, int64_t angleBoundHi,
                      float clipAngleThresh, bool legacyPlusOne);

  /// Create an ExternFunctionCall node. \p funcImpl will contain body
  /// of or reference to the function which can be invoked.
  /// \p funcKind contains the type of function. The type of function  could be
  /// source code, like OpenCL, CUDA, or could be a binary or
  /// a handle to an external function.
  ExternalFunctionCallNode *
  createExternalFunctionCall(llvm::StringRef name, TypeRef outTy,
                             llvm::ArrayRef<glow::NodeValue> inputs,
                             llvm::StringRef funcName, llvm::StringRef funcImpl,
                             llvm::StringRef funcKind);

  /// Erase the node \p N from the Function.
  void eraseNode(Node *N);

  /// Erase the node \p I from the Function.
  void eraseNode(NodesList::iterator I);

  /// Clone the current function into a new function with the name \p newName in
  /// the same module. If \p map is non-null then the procedure records the
  /// mapping between the old node to the new node in \p map. If \p currToNewMap
  /// is non-null it is used as the initial state of the currToNew map inside
  /// the cloner.
  /// \returns a new function that is a copy of the current function.
  Function *clone(llvm::StringRef newName,
                  llvm::DenseMap<const Node *, Node *> *map = nullptr,
                  llvm::DenseMap<const Node *, Node *> *currToNewMap = nullptr);

  /// Clone the current function into a user-provided function \p newF. The
  /// function \p newF is not automatically added to a module by the clone call.
  /// If \p map is non-null then the procedure records the mapping between the
  /// old node to the new node in \p map. If \p currToNewMap is non-null it is
  /// used as the initial state of the currToNew map inside the cloner. \returns
  /// a user-provided function \p newF that now contains a clone of the current
  /// function.
  Function *
  clone(Function *newF, llvm::DenseMap<const Node *, Node *> *map = nullptr,
        llvm::DenseMap<const Node *, Node *> *currToNewMap = nullptr) const;

  /// Verify the correctness of the Function. If \p backend is provided, checks
  /// backend-specific layout requirements. Else checks the requirements based
  /// on Glow's "canonical" layout. \returns true when the function is valid.
  /// False otherwise.
  bool verify(const Backend *backend = nullptr) const;

  /// Dump a textual representation of the Function into provided output stream.
  void dump() const;

  /// Dump a textual representation of the Function to std::string. If
  /// \p skipUsersForStorage then user counts for Storage will not be dumped.
  /// If \p skipName then the name of the Function will not be dumped.
  std::string toString(bool skipUsersForStorage = false,
                       bool skipName = false) const;

  /// \returns a hash code of the function.
  llvm::hash_code getHash() const;

  /// Dump a textual representation of the Function into default output stream.
  /// If \p skipUsersForStorage then user counts for Storage will not be dumped.
  /// If \p skipName then the name of the Function will not be dumped.
  void dump(llvm::raw_ostream &os, bool skipUsersForStorage = false,
            bool skipName = false) const;

  /// Dump a dotty graph that depicts the function into a file.
  /// \returns full path to the file.
  std::string dumpDAG();

  /// Dump a dotty graph that depicts the function.
  void dumpDAG(llvm::StringRef dotFilename);

  /// Dump a dotty graph that depicts the function.
  void dumpDAG(const char *dotFilename);

  /// \returns the list of nodes that the Function owns.
  NodesList &getNodes() { return nodes_; }

  const NodesList &getNodes() const { return nodes_; }

  /// \returns a node with the name \p name or nullptr if no node was found.
  Node *getNodeByName(llvm::StringRef name);

  /// \returns a node value using the \p name which has the same format as the
  /// one used by the \ref NodeValue::generateNodeOutputName which is
  /// "nodeName:outputNumber". The returned node value has a nullptr for the
  /// node if not found in the Function or if the node has no outputs (for
  /// example SaveNode). The searched node value can be one of a graph node,
  /// constant or placeholder.
  NodeValue getNodeValueByName(llvm::StringRef name);

  /// \returns pointer to the class member for the nodes list.
  static NodesList Function::*getNodesMemberPtr() { return &Function::nodes_; }

  /// Randomize all of the Constants in the function. If a Constant with users
  /// in this Function also has users in other Functions then this will result
  /// in a FATAL. \p ignoredConstants is a map Kinds of nodes to the input
  /// indices for that node that should be ignored (not randomized).
  void randomizeConstants(
      const std::map<Kinded::Kind, std::set<unsigned>> &ignoredConstants = {});
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

/// Clone \p node and its sources into \p newF using old-to-new mapping \p
/// currToNew.
Node *recursiveClone(Function *newF, Node *node, NodeMap &currToNew);

/// If \p PH is an output placeholder in the Function \p F,
/// \returns true.
/// This is determined by checking if the PH has a user which uses the PH as an
/// overwritten input.
bool isOutput(const Placeholder *PH, const Function &F);

/// If \p PH is an input placeholderin the Function \p F,
/// \returns true.
/// This is determined by checking if the PH is the input to a saveNode or is
/// used by a non saveNode.
bool isInput(const Placeholder *PH, const Function &F);

/// Helper vectors for common transpose shuffles.
#define NCHW2NHWC                                                              \
  { 0u, 2u, 3u, 1u }
#define NCTHW2NTHWC                                                            \
  { 0u, 2u, 3u, 4u, 1u }
#define NHWC2NCHW                                                              \
  { 0u, 3u, 1u, 2u }
#define NTHWC2NCTHW                                                            \
  { 0u, 4u, 1u, 2u, 3u }
#define HWCN2NHWC                                                              \
  { 3u, 0u, 1u, 2u }
#define NHWC2HWNC                                                              \
  { 1u, 2u, 0u, 3u }
#define CNHW2NHWC                                                              \
  { 1u, 2u, 3u, 0u }
#define NHWC2CHWN                                                              \
  { 3u, 1u, 2u, 0u }
#define CHWN2NHWC                                                              \
  { 3u, 1u, 2u, 0u }
#define D2S_DCR                                                                \
  { 0u, 1u, 3u, 2u, 4u, 5u }
#define D2S_CRD                                                                \
  { 0u, 1u, 4u, 2u, 5u, 3u }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Module &mod);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Module *mod);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Function &F);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Function *F);

/// \returns whether the Convolution node \p node is equivalent with a
/// FullyConnected node. This happens for a 2D NHWC Convolution with 1x1 filter
/// with strides 1, pads 0, group 1 and dilations 1.
bool isConvolutionSameAsFullyConnected(const ConvolutionNode *node,
                                       bool enfoceInput1x1 = false);

/// \returns whether the Gemm node \p node is equivalent with a FullyConnected
/// node. This happens when alpha and beta are 1.0 and the C operand is 1D.
bool isGemmSameAsFullyConnected(const GemmNode *node);

} // namespace glow

#endif // GLOW_GRAPH_GRAPH_H
