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

#ifndef GLOW_IMPORTER_ONNXMODELLOADER_H
#define GLOW_IMPORTER_ONNXMODELLOADER_H

#include "glow/Graph/Graph.h"
#include "glow/Importer/CommonOperatorLoader.h"

#include "onnx/onnx_pb.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <unordered_set>

namespace ONNX_NAMESPACE {
class AttributeProto;
class NodeProto;
class GraphProto;
class ModelProto;
class TensorProto;
} // namespace ONNX_NAMESPACE

namespace glow {

/// Loads tensor \p T from the input \p in.
Error loadTensor(const ONNX_NAMESPACE::TensorProto &in, Tensor *T);

/// Define undefined symbols to \p str loaded from an ONNX proto. See
/// onnxDefineSymbolOpt in ONNXModelLoader.cpp.
void setOnnxDefineSymbol(const std::vector<std::string> &lst);

/// Loads ONNX models.
class ONNXModelLoader
    : public CommonOperatorLoader<ONNX_NAMESPACE::NodeProto,
                                  ONNX_NAMESPACE::AttributeProto> {
  /// \returns True if the operator has broadcasting activated.
  Expected<bool> getBroadcast(const ArgumentDictionaryTy &dict) override;

  /// \returns True if the operator with the name \p typeName has support for
  /// multidirectional broadcasting.
  bool hasMultidirectionalBroadcast(const llvm::StringRef typeName) override;

  /// Converts a ONNX TensorProto DataType enum to the Glow element type.
  /// Supports only non quantized and signed types.
  Expected<ElemKind>
  convertTensorProtoDataType(ONNX_NAMESPACE::TensorProto_DataType t);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network. \returns Error if operator \p op cannot be loaded.
  Error loadOperator(const ONNX_NAMESPACE::NodeProto &op);

  /// \returns True if the operator\ op is successfully folded.
  Expected<bool> foldOperator(const ONNX_NAMESPACE::NodeProto &op);

  /// ONNX model ir_version;
  size_t irVersion_;

  /// ONNX model op_version;
  size_t opsetVersion_;

  /// A set of inputs which will be static placeholders.
  std::unordered_set<std::string> staticInputs_;

  /// Load Constant ONNX operator.
  Error loadConstant(const ONNX_NAMESPACE::NodeProto &op,
                     const ArgumentDictionaryTy &dict);

  /// Load Slice ONNX operator.
  Error loadSlice(const ONNX_NAMESPACE::NodeProto &op,
                  const ArgumentDictionaryTy &dict);

  /// Load Conv ONNX operator.
  Error loadConv(const ONNX_NAMESPACE::NodeProto &op,
                 const ArgumentDictionaryTy &dict);

  /// Load ChannelwiseQuantizedConvolution Glow operator.
  Error loadChannelwiseQuantizedConvolution(const ONNX_NAMESPACE::NodeProto &op,
                                            const ArgumentDictionaryTy &dict);

  /// Load Glow conv operator with quantized inputs. Since this isn't a normal
  /// part of the ops supported by Glow, the assumption is that this op was
  /// produced by Glow's on ONNXModelWriter and thus has NHWC layout for inputs.
  Error loadTensorwiseQuantizedConvolution(const ONNX_NAMESPACE::NodeProto &op,
                                           const ArgumentDictionaryTy &dict);

  /// Load MaxPool or AveragePool ONNX operator. \p typeName is the name of the
  /// ONNX operator being loaded, either MaxPool or AveragePool.
  Error loadPool(const ONNX_NAMESPACE::NodeProto &op,
                 const ArgumentDictionaryTy &dict, llvm::StringRef typeName);

  /// Load GlobalAveragePool ONNX operator.
  Error loadGlobalAveragePool(const ONNX_NAMESPACE::NodeProto &op,
                              const ArgumentDictionaryTy &dict);

  /// Load Squeeze ONNX operator.
  Error loadSqueeze(const ONNX_NAMESPACE::NodeProto &op,
                    const ArgumentDictionaryTy &dict);

  /// Load Unsqueeze ONNX operator.
  Error loadUnsqueeze(const ONNX_NAMESPACE::NodeProto &op,
                      const ArgumentDictionaryTy &dict);

  /// Load ArgMax ONNX operator.
  Error loadArgMax(const ONNX_NAMESPACE::NodeProto &op,
                   const ArgumentDictionaryTy &dict);

  /// Load BatchNormalization ONNX operator.
  Error loadBatchNormalization(const ONNX_NAMESPACE::NodeProto &op,
                               const ArgumentDictionaryTy &dict);

  /// Load Concat ONNX operator.
  Error loadConcat(const ONNX_NAMESPACE::NodeProto &op,
                   const ArgumentDictionaryTy &dict);

  /// Load FCTransposed ONNX operator.
  Error loadFCTransposed(const ONNX_NAMESPACE::NodeProto &op,
                         const ArgumentDictionaryTy &dict);

  /// Load Gemm ONNX operator.
  Error loadGemm(const ONNX_NAMESPACE::NodeProto &op,
                 const ArgumentDictionaryTy &dict);

  /// Load MatMul ONNX operator.
  Error loadMatMul(const ONNX_NAMESPACE::NodeProto &op,
                   const ArgumentDictionaryTy &dict);

  /// Load Pad ONNX operator.
  Error loadPad(const ONNX_NAMESPACE::NodeProto &op,
                const ArgumentDictionaryTy &dict);

  /// Load Cast ONNX operator.
  Error loadCast(const ONNX_NAMESPACE::NodeProto &op,
                 const ArgumentDictionaryTy &dict);

  /// Load LeakyRelu ONNX operator.
  Error loadLeakyRelu(const ONNX_NAMESPACE::NodeProto &op,
                      const ArgumentDictionaryTy &dict);

  /// Load SpaceToDepth ONNX operator.
  Error loadSpaceToDepth(const ONNX_NAMESPACE::NodeProto &op,
                         const ArgumentDictionaryTy &dict);

  /// Load ConstantOfShape ONNX operator.
  Error loadConstantOfShape(const ONNX_NAMESPACE::NodeProto &op,
                            const ArgumentDictionaryTy &dict, bool isSplat);

  /// Load Tile ONNX operator.
  Error loadTile(const ONNX_NAMESPACE::NodeProto &op,
                 const ArgumentDictionaryTy &dict);

  /// Load Where ONNX operator.
  Error loadWhere(const ONNX_NAMESPACE::NodeProto &op,
                  const ArgumentDictionaryTy &dict);

  /// Load RNN ONNX operator.
  Error loadRNN(const ONNX_NAMESPACE::NodeProto &op,
                const ArgumentDictionaryTy &dict);

  /// Load GRU ONNX operator.
  Error loadGRU(const ONNX_NAMESPACE::NodeProto &op,
                const ArgumentDictionaryTy &dict);

  /// Load LSTM ONNX operator.
  Error loadLSTM(const ONNX_NAMESPACE::NodeProto &op,
                 const ArgumentDictionaryTy &dict);

  /// Load Glow specific operators, not defined in ONNX format
  /// Load Glow CmpEQ operator.
  Error loadCmpEQ(const ONNX_NAMESPACE::NodeProto &op,
                  const ArgumentDictionaryTy &dict);

  /// Load Glow CmpLTE operator.
  Error loadCmpLTE(const ONNX_NAMESPACE::NodeProto &op,
                   const ArgumentDictionaryTy &dict);

  /// Load Glow Select operator.
  Error loadSelect(const ONNX_NAMESPACE::NodeProto &op,
                   const ArgumentDictionaryTy &dict);

  /// Load Glow Quantize operator.
  Error loadQuantize(const ONNX_NAMESPACE::NodeProto &op,
                     const ArgumentDictionaryTy &dict);

  /// Load Glow ConvertTo operator.
  Error loadConvertTo(const ONNX_NAMESPACE::NodeProto &op,
                      const ArgumentDictionaryTy &dict);

  /// Load Glow Dequantize operator.
  Error loadDequantize(const ONNX_NAMESPACE::NodeProto &op,
                       const ArgumentDictionaryTy &dict);

  /// Load Glow Regression operator.
  Error loadRegression(const ONNX_NAMESPACE::NodeProto &op,
                       const ArgumentDictionaryTy &dict);

  /// Load Glow BatchedAdd operator.
  Error loadBatchedAdd(const ONNX_NAMESPACE::NodeProto &op,
                       const ArgumentDictionaryTy &dict);

  /// Load Glow CumSum operator.
  Error loadCumSum(const ONNX_NAMESPACE::NodeProto &op,
                   const ArgumentDictionaryTy &dict);

  /// Load Glow ScatterAssign operator.
  Error loadScatterAssign(const ONNX_NAMESPACE::NodeProto &op,
                          const ArgumentDictionaryTy &dict);

  /// Load Glow IntLookupTable operator.
  Error loadIntLookupTable(const ONNX_NAMESPACE::NodeProto &op,
                           const ArgumentDictionaryTy &dict);

  /// Load Glow LengthsRangeFill operator.
  Error loadLengthsRangeFill(const ONNX_NAMESPACE::NodeProto &op,
                             const ArgumentDictionaryTy &dict);

  /// Load Glow RescaleQuantized operator.
  Error loadRescaleQuantized(const ONNX_NAMESPACE::NodeProto &op,
                             const ArgumentDictionaryTy &dict);

  /// Load Glow RowwiseQuantizedSparseLengthsWeightedSum operator.
  Error loadRowwiseQuantizedSparseLengthsWeightedSum(
      const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict);

  /// Load Glow FusedRowwiseQuantizedSparseLengthsWeightedSum operator.
  Error loadFusedRowwiseQuantizedSparseLengthsWeightedSum(
      const ONNX_NAMESPACE::NodeProto &op, const ArgumentDictionaryTy &dict);

  /// Load Glow FusedRowwiseQuantizedSparseLengthsSum operator.
  Error
  loadFusedRowwiseQuantizedSparseLengthsSum(const ONNX_NAMESPACE::NodeProto &op,
                                            const ArgumentDictionaryTy &dict);

  /// Load Glow RowwiseQuantizedFullyConnected operator.
  Error loadRowwiseQuantizedFullyConnected(const ONNX_NAMESPACE::NodeProto &op,
                                           const ArgumentDictionaryTy &dict);

  /// Load Glow FullyConnected operator.
  Error loadFullyConnected(const ONNX_NAMESPACE::NodeProto &op,
                           const ArgumentDictionaryTy &dict);

  /// Load ONNX Identity operator.
  Error loadIdentity(const ONNX_NAMESPACE::NodeProto &op,
                     const ArgumentDictionaryTy &dict);

  /// Load Glow Splat operator.
  Error loadSplat(const ONNX_NAMESPACE::NodeProto &op,
                  const ArgumentDictionaryTy &dict);

  /// Load NonMaxSuppression ONNX and TF NMSv4 operator.
  /// The \p isV4 indicates whether this is ONNX or custom NMSv4 operator.
  Error loadNonMaxSuppression(const ONNX_NAMESPACE::NodeProto &op,
                              const ArgumentDictionaryTy &dict, bool isV4);

  /// Load Glow InsertTensor operator.
  Error loadInsertTensor(const ONNX_NAMESPACE::NodeProto &op,
                         const ArgumentDictionaryTy &dict);

  /// Load AdaptiveAvgPool Glow operator.
  Error loadAdaptiveAvgPool(const ONNX_NAMESPACE::NodeProto &op,
                            const ArgumentDictionaryTy &dict);

  /// Load Flip Glow operator.
  Error loadFlip(const ONNX_NAMESPACE::NodeProto &op,
                 const ArgumentDictionaryTy &dict);

protected:
  /// Load the network operators from the GraphProto.
  /// \returns Error if network cannot be loaded.
  Error loadNetwork(ONNX_NAMESPACE::GraphProto &net);

  /// Set the output nodes of the network \p net. Initializes the map from the
  /// names of the outputs to the save nodes that save each output.
  /// \returns Error if network cannot be loaded.
  Error setOutputNodes(ONNX_NAMESPACE::GraphProto &net);

  /// Set ir verion and op version.
  Error setVersion(ONNX_NAMESPACE::ModelProto MP);

  /// \returns Expected<ModelProto> if a ModelProto can be loaded from the
  /// stream \p iStream.
  static Expected<ONNX_NAMESPACE::ModelProto>
  loadProto(google::protobuf::io::ZeroCopyInputStream &iStream);

  /// Load the network initializers from the GraphProto.
  Error loadInitializers(ONNX_NAMESPACE::GraphProto &net);

  /// Load the inputs from the GraphProto. If \p loadInputsAsPlaceholders is
  /// true then this will load each graph input as a placeholder otherwise it
  /// will create an empty tensor for each input.
  Error loadInputs(ONNX_NAMESPACE::GraphProto &net,
                   bool loadInputsAsPlaceholders);

  /// \returns Expected<ModelProto> if a ModelProto can be constructed from the
  /// contents of the file \p filename and Error otherwise.
  /// Loads ModelProto from the file containing serialized protobuf.
  static Expected<ONNX_NAMESPACE::ModelProto>
  loadProto(const std::string &filename);

  /// \returns Expected<ModelProto> if a ModelProto can be constructed from the
  /// in-memory serialized protobuf.
  /// Loads ModelProto from the in-memory serialized protobuf \p
  /// onnxModel with the model size \p onnxModelSize.
  static Expected<ONNX_NAMESPACE::ModelProto> loadProto(const void *onnxModel,
                                                        size_t onnxModelSize);

  /// Checks that the inputs tensors are compatible with the inputs declared in
  /// the ONNX model. The input types in \p types match the list of names
  /// \p tensorNames.
  Error checkInputs(ONNX_NAMESPACE::GraphProto &net,
                    llvm::ArrayRef<const char *> tensorNames,
                    llvm::ArrayRef<TypeRef> types);

  /// Go through the ValueInfoProto of the inputs of the \p net and collect
  /// static placeholders if it's marked in the ValueInfoProto.
  Error collectStaticInputs(ONNX_NAMESPACE::GraphProto &net);

  /// Creates a ONNX model loader to build \p F.
  /// Loads the ONNIXFI \p model from memory of \p modelSize size,
  /// and \p weightsCount, and \p onnxTensorDescriptorV1 correspondent
  /// descriptors. Converts inputs into placeholder if requested \p
  /// loadInputsAsPlaceholders. Reports success/failure through optional
  /// parameter \p errPtr. This constructor always overrides the default
  /// constant folding in loader flag with \p constFoldInLoader.
  ONNXModelLoader(const void *model, uint32_t modelSize, uint32_t weightsCount,
                  const onnxTensorDescriptorV1 *weightDescriptors, Function &F,
                  bool loadInputsAsPlaceholders, Error *errPtr = nullptr,
                  bool constFoldInLoader = true);

  friend class ONNXIFIModelLoader;

  /// \returns success if the folding of operator \p op in the loader
  /// \p loader is successful. The folding utility uses temporary
  /// loader \p tmpLoader, and associated temporary function \p F.
  template <class LoaderType, class OpType>
  friend Error constantFoldInLoader(Function *F, LoaderType &tmpLoader,
                                    LoaderType *loader, const OpType &op);

public:
  /// \returns ONNX model ir_version;
  size_t getIrVersion() const { return irVersion_; };

  /// \returns ONNX model op_version;
  size_t getOpSetVersion() const { return opsetVersion_; };

  /// Creates a ONNX model loader to build \p F.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  ONNXModelLoader(Function &F, Error *errPtr = nullptr);

  /// Loads the ONNX model that's represented by a model description file,
  /// serialized in \p modelDescFilename and populates the network into \p F.
  /// The types in \p types match the list of names \p tensorNames and used as
  /// inputs to the network.
  /// If \p names and \p types are empty loader fills inputs automatically.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  ONNXModelLoader(const std::string &modelDescFilename,
                  llvm::ArrayRef<const char *> tensorNames,
                  llvm::ArrayRef<TypeRef> types, Function &F,
                  Error *errPtr = nullptr, bool zipMode = false);
};

} // namespace glow

#endif // GLOW_IMPORTER_ONNXMODELLOADER_H
