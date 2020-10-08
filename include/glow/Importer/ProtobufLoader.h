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

#ifndef GLOW_IMPORTER_PROTOBUFLOADER_H
#define GLOW_IMPORTER_PROTOBUFLOADER_H

#include "glow/Base/Tensor.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Support/Error.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <google/protobuf/text_format.h>

#include <memory>
#include <string>
#include <vector>

/// This is the maximum allowed protobuf size (2GB).
#define MAX_PROTO_SIZE 0x7FFFFFFF

namespace glow {

/// Some model formats (Caffe2, PyTorch, TensorFlowLite) allow defining weights
/// and activations in UINT8 format. Since Glow supports only INT8 weights and
/// activations we do a transformation from UINT8 to INT8 quantized data by
/// subtracting the value 128 from both the quantized values and the offset:
///   val(int8) = val(uint8) - 128
///   scale(int8) = scale(uint8) (scale value is preserved)
///   offset(int8) = scale(uint8) - 128
/// The constant definition below defines the value used for subtraction.
constexpr int32_t UINT8_TO_INT8_SHIFT = 128;

/// Enables or disables constant-folding of Loader Ops with \p flag.
void setConstantFoldLoaderOpsFlag(bool flag);

/// Returns true if constant-folding for loader Ops is enabled.
bool getConstantFoldLoaderOpsFlag();

/// Returns true iff all elements of \p a are the same.
bool isArrayConstant(const llvm::ArrayRef<size_t> a);

/// Prints a single serialized protocol buffer node. This method is useful for
/// debugging the network and printing errors.
template <typename T>
std::string unexpectedNodeErrorMessage(const T &node, llvm::StringRef message) {
  std::string str;
  google::protobuf::TextFormat::PrintToString(node, &str);
  return llvm::formatv("{0}\n{1}", message, str);
}

/// Reads a single integer.
template <typename T> static Expected<int> loadInt(const T *arg) {
  RETURN_ERR_IF_NOT(arg->has_i(), "Node has no Int value");
  return arg->i();
}

/// Reads a single float.
template <typename T> static Expected<float> loadFloat(const T *arg) {
  RETURN_ERR_IF_NOT(arg->has_f(), "Node has no float value");
  return arg->f();
}

/// Reads a single string.
template <typename T> static Expected<std::string> loadStr(const T *arg) {
  RETURN_ERR_IF_NOT(arg->has_s(), "Node has no str value");
  return arg->s();
}

/// Load the 'shape' record from \p arg into a vector of sizes. \returns the
/// vector if there is no error, or an error otherwise. If \p allowEmptyShape
/// then no error will be returned if the vector is empty.
template <typename ElemTy, typename AttrType>
Expected<std::vector<ElemTy>> getShape(const AttrType *arg,
                                       bool allowEmptyShape = false) {
  RETURN_ERR_IF_NOT(arg, "Node has no ints attribute with this name");
  if (!allowEmptyShape) {
    RETURN_ERR_IF_NOT(arg->ints_size() > 0, "Node has no ints values");
  }
  std::vector<ElemTy> dim;
  for (auto i : arg->ints()) {
    dim.push_back(i);
  }
  return dim;
}

/// Load a floating record vector from \p arg into a vector. \returns the vector
/// if there is no error, or an error otherwise.
template <typename AttrType>
Expected<std::vector<float>> getFloats(const AttrType *arg) {
  RETURN_ERR_IF_NOT(arg, "Node has no floats attribute with this name");
  std::vector<float> dim;
  for (auto i : arg->floats()) {
    dim.push_back(i);
  }
  return dim;
}

/// Load a string record vector from \p arg into a vector. \returns the vector
/// if there is no error, or an error otherwise.
template <typename AttrType>
Expected<std::vector<std::string>> getStrings(const AttrType *arg) {
  RETURN_ERR_IF_NOT(arg, "Node has no strings attribute with this name");
  RETURN_ERR_IF_NOT(arg->strings_size() > 0, "Node has no strings values");
  std::vector<std::string> strs;
  for (const auto &s : arg->strings()) {
    strs.push_back(s);
  }
  return strs;
}

/// Returns canonical name for a given operator: either \p name() from proto,
/// or its type name.
template <typename T> std::string loadOperatorName(const T &op) {
  if (op.name().length()) {
    return op.name();
  }
  if (op.output_size() > 0) {
    return op.op_type() + "_" + op.output(0);
  }
  return op.op_type();
}

/// \returns the positive value (modulo) of \p axis given the \p rank (number
/// of dimensions) of the tensor it refers to. The proto format allows negative
/// axis in which case the axis is used for counting dimensions from the back.
/// Since Glow cannot use a negative axis we use this converter utility. For
/// example an axis value of -1 for a tensor with 3 dimensions (rank 3) is
/// converted to 2. A good definition of the axis value requires to be in the
/// range [rank, rank-1].
template <typename T> Expected<T> getPositiveAxis(int axis, int rank) {
  RETURN_ERR_IF_NOT(
      (-rank <= axis) && (axis < rank),
      strFormat("Axis value %d is invalid! Should be in the range [%d, %d]!",
                axis, -rank, rank - 1));
  int axisPos = (axis < 0) ? axis + rank : axis;
  return static_cast<T>(axisPos);
}

/// \returns the positive value of \p axis given the rank of the value \p val.
template <typename T> Expected<T> getPositiveAxis(int axis, NodeValue val) {
  return getPositiveAxis<T>(axis, val.dims().size());
}

/// Reads a single axis parameter which is wrapped if negative using \p rank
/// based on the logic of \ref getPositiveAxis.
template <typename ElemTy, typename T>
static Expected<ElemTy> loadAxis(const T *arg, int rank) {
  int axis;
  ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(arg));
  ASSIGN_VALUE_OR_RETURN_ERR(axis, getPositiveAxis<int>(axis, rank));
  return static_cast<ElemTy>(axis);
}

/// Reads multiple axes as an array which are wrapped if negative using \p rank
/// based on the logic of \ref getPositiveAxis.
template <typename ElemTy, typename T>
static Expected<std::vector<ElemTy>> loadAxes(const T *arg, int rank) {
  std::vector<int> axes;
  ASSIGN_VALUE_OR_RETURN_ERR(axes, getShape<int>(arg));
  std::vector<ElemTy> axesPos;
  for (int axis : axes) {
    int axisPos;
    ASSIGN_VALUE_OR_RETURN_ERR(axisPos, getPositiveAxis<int>(axis, rank));
    axesPos.push_back(static_cast<ElemTy>(axisPos));
  }
  return axesPos;
}

/// Loads model: graph and weights.
class ProtobufLoader {
protected:
  /// The graph that we are constructing.
  Function *G_;
  /// The module containing the graph(s) that we are constructing.
  Module &mod_;
  /// A map from partition names to Function for it.
  llvm::StringMap<Function *> partNameToFun_;
  /// Saves network nodes by name.
  llvm::StringMap<NodeValue> nodeValueByName_;
  /// Saves intermediate PHs by name, i.e. those that are used to communicate
  /// between partitions.
  llvm::StringMap<Placeholder *> intermediatePHsByName_;
  /// A map from names of the external outputs of the network to Variables.
  llvm::StringMap<Placeholder *> outputVarsByName_;
  /// A map from names of the external inputs of the network to Variables.
  llvm::StringMap<Placeholder *> inputVarsByName_;
  /// A vector of input names ordered by their position of inference interface
  std::vector<std::string> positionalInputNames_;
  /// A vector of output names ordered by their position of inference interface
  std::vector<std::string> positionalOutputNames_;
  /// Whether to try constant folding as we load each op from a protobuf.
  bool constFoldInLoader_{true};
  /// Whether to load the proto into the existing module. All Functions and
  /// Storage should already exist for the proto; the Functions should be empty
  /// and will be filled with Nodes from the proto connected to Storage.
  bool loadIntoExistingModule_{false};
  /// Uniqued offset value used when saving the mapping from quant params to
  /// loader names.
  int32_t currUniqueOffset_{0};
  /// An optional mapping from the names of ops and inputs that are quantized to
  /// the TQP that it came with.
  OriginNameToTQPMap *originNameToTQPMap_{nullptr};
  /// Whether to load uniqued dummy quantization params instead of the actual
  /// quantization params in the model. \ref originNameToTQPMap_ must be
  /// non-null when this is true.
  bool loadUniquedDummyQParams_{false};
  /// New TQP to use, indexed by the unique dummy offset it is mapped to.
  std::vector<TensorQuantizationParams> updatedTQPs_;
  /// Whether to try to replace dummy TQPs found during loading with real
  /// updated ones in \ref updatedTQPs_.
  bool replaceDummyTQPs_{false};

  // Delete all Constants that have no users. This is useful because some
  // Constants may have been copied and modified during loading instead of used
  // directly so they may be unused.
  void deleteUnusedConstants();

  /// Create a new constant that's initialized with \p tensor, and register it
  /// under the name \p name. If an existing Placeholder is already registered
  /// under the same name then the tensor is thrown out and no new Constant
  /// is created. The Constant will have Layout \p layout.
  Error createAndRegisterConstant(llvm::StringRef name, Tensor &&tensor,
                                  const std::string &layout = ANY_LAYOUT);

  /// Create a new Placeholder of type \p T, and register it
  /// under the name \p name. If \p isStatic is true register the Placeholder as
  /// a static placeholder. \p isTrainable and \p layout are set in the
  /// Placeholder accoringly. \returns The newly created placeholder.
  Expected<Placeholder *>
  createAndRegisterPlaceholder(llvm::StringRef name, TypeRef T,
                               bool isStatic = false, bool isTrainable = false,
                               const std::string &layout = ANY_LAYOUT);

  /// \returns the NodeValue that was registered with the name \p name or
  /// a nullptr wrapped in a NodeValue if no node has been registered with this
  /// name. Storage NodeValues are always returned if found. Otherwise, if
  /// \ref G_ is the same as the parent of the NodeValue (or if
  /// \p ignoreSrcFun), then the direct mapping is returned from
  /// \ref nodeValueByName_. Otherwise, a SaveNode will be created and output to
  /// a Placeholder, which will be added to \ref nodeValueByName_ and returned.
  NodeValue getNodeValueByNameOrNullNodeValue(llvm::StringRef name,
                                              bool ignoreSrcFun = false);

  Placeholder *getStaticPlaceholderByNameOrNull(llvm::StringRef name) const;

  /// \returns the Constant registered with the given \p name and nullptr if
  /// no Constant has been registered with this name.
  Constant *getConstantByNameOrNull(llvm::StringRef name) const;

  /// \returns an Expected of the Constant registered with the given \p
  /// name and returns and Error if no Constant has been registered with this
  /// name.
  Expected<Constant *> getConstantByName(llvm::StringRef name) const;

  /// \returns whether or not a Constant has been registered with the given \p
  /// name.
  bool hasConstantByName(llvm::StringRef name) const;

  /// Sets up a new Loader based on \p tensorNames, \p types, and sets any error
  /// in \p errPtr.
  void setupLoader(llvm::ArrayRef<const char *> tensorNames,
                   llvm::ArrayRef<TypeRef> types, Error *errPtr);

  /// \returns the TQP located at \p uniqueOffsetIdx in \ref updatedTQPs_.
  Expected<TensorQuantizationParams> getUpdatedTQP(int32_t uniqueOffsetIdx);

  /// \returns a quantized type with loader name \p name given \p k, \p dims,
  /// \p scale, and \p offset. If \p shiftUInt8ToInt8 and \p k is Int8QTy, then
  /// the offset is shifted by UINT8_TO_INT8_SHIFT to Int8.
  Expected<TypeRef> loadQuantTy(const std::string &name, ElemKind k,
                                llvm::ArrayRef<dim_t> dims, float scale,
                                int32_t offset, bool shiftUInt8ToInt8 = true);

public:
  /// \returns the NodeValue that was registered with the name \p name.  If
  /// looking up a NodeValue which is produced from a different Function (and if
  /// not \p ignoreSrcFun) then this returns (and creates if doesn't yet exist)
  /// a new SaveNode and Placeholder to connect the two Nodes from different
  /// Functions.
  /// \pre hasNodeByName(name)
  Expected<NodeValue> getNodeValueByName(llvm::StringRef name,
                                         bool ignoreSrcFun = false);

  /// \returns True if the node that's registered using \p name exists.
  bool hasNodeByName(llvm::StringRef name) const;

  /// Constructs new ProtobufLoader object. It will populate the network into
  /// \p F. The list \p types and \p names are used to initialize the inputs
  /// of the model with specific names and types. If \p errPtr is not null then
  /// if an error occurs it will get assigned there otherwise if an error
  /// occurs it will abort. If \p loadIntoExistingModule then all Functions and
  /// Storage is expected to already exist, so they will be searched for
  /// according to the proto being loaded instead of created as usual.
  ProtobufLoader(llvm::ArrayRef<const char *> tensorNames,
                 llvm::ArrayRef<TypeRef> types, Function *F,
                 Error *errPtr = nullptr, bool loadIntoExistingModule = false,
                 OriginNameToTQPMap *originNameToTQPMap = nullptr,
                 bool loadUniquedDummyQParams = false,
                 bool replaceDummyTQPs = false);

  /// Constructs new ProtobufLoader object. It will populate the network into
  /// \p mod. The list \p types and \p names are used to initialize the inputs
  /// of the model with specific names and types. If \p errPtr is not null then
  /// if an error occurs it will get assigned there otherwise if an error
  /// occurs it will abort. If \p loadIntoExistingModule then all Functions and
  /// Storage is expected to already exist, so they will be searched for
  /// according to the proto being loaded instead of created as usual.
  ProtobufLoader(llvm::ArrayRef<const char *> tensorNames,
                 llvm::ArrayRef<TypeRef> types, Module &mod,
                 Error *errPtr = nullptr, bool loadIntoExistingModule = false,
                 OriginNameToTQPMap *originNameToTQPMap = nullptr,
                 bool loadUniquedDummyQParams = false,
                 bool replaceDummyTQPs = false);

  ProtobufLoader(const ProtobufLoader &other) = delete;
  ProtobufLoader &operator=(const ProtobufLoader &) = delete;
  virtual ~ProtobufLoader() = default;

  /// \returns mapping between external names and actual Glow output nodes.
  const llvm::StringMap<Placeholder *> &getOutputVarsMapping() const {
    return outputVarsByName_;
  }

  /// \returns mapping between external names and actual Glow input nodes.
  const llvm::StringMap<Placeholder *> &getInputVarsMapping() const {
    return inputVarsByName_;
  }

  /// \returns vector of primary input names based on their position
  const std::vector<std::string> &getPositionalInputNames() const {
    return positionalInputNames_;
  }

  /// \returns vector of primary output names based on their position
  const std::vector<std::string> &getPositionalOutputNames() const {
    return positionalOutputNames_;
  }

  /// \returns the single final output of the network. The function assumes
  /// that there is only one output, returns Error otherwise. For image
  /// classification, this single final output is usually the result of the
  /// last softmax or regression layer.
  Expected<Placeholder *> getSingleOutput() const;

  /// \returns the single input of the network. The function assumes that there
  /// is only one input, returns Error otherwise. For most of the models the
  /// single input is usually an image tensor.
  Expected<Placeholder *> getSingleInput() const;

  /// \returns the Placeholder for the external output with \p name.
  /// \pre outputVarsByName_.find(name) != outputVarsByName_.end()
  Expected<Placeholder *> getOutputByName(llvm::StringRef name) const;

  /// \returns the Placeholder for the external input with \p name.
  /// \pre inputVarsByName_.find(name) != inputVarsByName_.end()
  Expected<Placeholder *> getInputByName(llvm::StringRef name) const;

  /// \returns True if the operator with name \p typeName having input node
  /// list as \p inputs is constant foldable.
  bool isConstantFoldable(llvm::ArrayRef<NodeValue> inputs,
                          std::string typeName) const;
};

/// \returns success if the folding of operator \p op in the loader
/// \p loader is successful. The folding utility uses temporary
/// loader \p tmpLoader, and associated temporary function \p F.
template <class LoaderType, class OpType>
Error constantFoldInLoader(Function *F, LoaderType &tmpLoader,
                           LoaderType *loader, const OpType &op) {
  PlaceholderBindings bindings;
  std::vector<Tensor *> outTensors;

  // Register the constant inputs to the current op with the constant folding
  // loader.
  for (unsigned i = 0; i < (dim_t)op.input_size(); i++) {
    Constant *tmpConst = loader->getConstantByNameOrNull(op.input(i));
    RETURN_ERR_IF_NOT(tmpConst, "No constant found");
    tmpLoader.nodeValueByName_[op.input(i)] = tmpConst->getOutput();
  }

  // Using the loader to load the current operator.
  RETURN_IF_ERR(tmpLoader.loadOperator(op));

  // To collect the folded outputs allocate and add save nodes to the folding
  // function.
  llvm::SmallVector<Placeholder *, 2> tmpPHs;
  for (int i = 0; i < op.output_size(); i++) {
    const auto &outputName = op.output(i);
    NodeValue r;
    ASSIGN_VALUE_OR_RETURN_ERR(r, tmpLoader.getNodeValueByName(outputName));
    Placeholder *PH = F->getParent()->createPlaceholder(
        r.getType(), "__CONSTFOLD__TMP__" + outputName, false);
    SaveNode *SN = F->createSave("save_" + PH->getName().str(), r, PH);
    auto *result = bindings.allocate(SN->getPlaceholder());
    outTensors.push_back(result);
    tmpPHs.push_back(PH);
  }

  // Cleanup to remove the temporary Placeholders we created.
  ScopeGuard cleanup([&]() {
    auto &mod = *F->getParent();
    auto &modPHs = mod.getPlaceholders();
    for (Placeholder *tmpPH : tmpPHs) {
      mod.erasePlaceholder(std::find(modPHs.begin(), modPHs.end(), tmpPH));
    }
  });

  // Evaluate the constant outputs using interpreter backend.
  std::unique_ptr<Backend> backend(createBackend("Interpreter"));
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  cctx.optimizationOpts.enableConstantFolding = false;
  cctx.optimizationOpts.enableConstantDeduplication = false;
  cctx.backendOpts.collectConstants = true;
  // Do not print out compilation errors encountered, as constant folding is a
  // best effort; simply silently give up and continue with compilation.
  cctx.verboseCompile = false;
  RETURN_IF_ERR(executeConstantFunction(*backend, *F, bindings, cctx));

  // Using the graph output, place constant nodes in the original graph.
  for (int i = 0; i < op.output_size(); i++) {
    RETURN_IF_ERR(loader->createAndRegisterConstant(op.output(i),
                                                    std::move(*outTensors[i])));
  }

  return Error::success();
}

} // namespace glow

#endif // GLOW_IMPORTER_PROTOBUFLOADER_H
