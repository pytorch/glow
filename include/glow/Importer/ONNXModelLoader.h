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

#ifndef GLOW_IMPORTER_ONNXMODELLOADER_H
#define GLOW_IMPORTER_ONNXMODELLOADER_H

#include "glow/Graph/Graph.h"
#include "glow/Importer/CommonOperatorLoader.h"

#include "onnx/onnx_pb.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace ONNX_NAMESPACE {
class AttributeProto;
class NodeProto;
class GraphProto;
class ModelProto;
} // namespace ONNX_NAMESPACE

namespace glow {

/// Loads ONNX models.
class ONNXModelLoader
    : public CommonOperatorLoader<ONNX_NAMESPACE::NodeProto,
                                  ONNX_NAMESPACE::AttributeProto> {
  /// \returns True if the operator has broadcasting activated.
  llvm::Expected<bool> getBroadcast(const ArgumentDictionaryTy &dict) override;

  /// \returns True if the operator with the name \p typeName has support for
  /// multidirectional broadcasting.
  bool hasMultidirectionalBroadcast(const llvm::StringRef typeName) override;

  /// Converts a ONNX TensorProto DataType enum to the Glow element type.
  /// Supports only non quantized and signed types.
  llvm::Expected<ElemKind>
  convertTensorProtoDataType(ONNX_NAMESPACE::TensorProto_DataType t);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network. \returns Error if operator \p op cannot be loaded.
  llvm::Error loadOperator(const ONNX_NAMESPACE::NodeProto &op);

  /// ONNX model ir_version;
  size_t irVersion_;

  /// ONNX model op_version;
  size_t opsetVersion_;

  /// Load Constant ONNX operator.
  llvm::Error loadConstant(const ONNX_NAMESPACE::NodeProto &op,
                           const ArgumentDictionaryTy &dict);

  /// Load Slice ONNX operator.
  llvm::Error loadSlice(const ONNX_NAMESPACE::NodeProto &op,
                        const ArgumentDictionaryTy &dict);

  /// Load Conv ONNX operator.
  llvm::Error loadConv(const ONNX_NAMESPACE::NodeProto &op,
                       const ArgumentDictionaryTy &dict);

  /// Load MaxPool or AveragePool ONNX operator. \p typeName is the name of the
  /// ONNX operator being loaded, either MaxPool or AveragePool.
  llvm::Error loadPool(const ONNX_NAMESPACE::NodeProto &op,
                       const ArgumentDictionaryTy &dict,
                       llvm::StringRef typeName);

  /// Load GlobalAveragePool ONNX operator.
  llvm::Error loadGlobalAveragePool(const ONNX_NAMESPACE::NodeProto &op,
                                    const ArgumentDictionaryTy &dict);

  /// Load Squeeze ONNX operator.
  llvm::Error loadSqueeze(const ONNX_NAMESPACE::NodeProto &op,
                          const ArgumentDictionaryTy &dict);

  /// Load Unsqueeze ONNX operator.
  llvm::Error loadUnsqueeze(const ONNX_NAMESPACE::NodeProto &op,
                            const ArgumentDictionaryTy &dict);

  /// Load BatchNormalization ONNX operator.
  llvm::Error loadBatchNormalization(const ONNX_NAMESPACE::NodeProto &op,
                                     const ArgumentDictionaryTy &dict);

  /// Load Concat ONNX operator.
  llvm::Error loadConcat(const ONNX_NAMESPACE::NodeProto &op,
                         const ArgumentDictionaryTy &dict);

  /// Load FCTransposed ONNX operator.
  llvm::Error loadFCTransposed(const ONNX_NAMESPACE::NodeProto &op,
                               const ArgumentDictionaryTy &dict);

  /// Load Gemm ONNX operator.
  llvm::Error loadGemm(const ONNX_NAMESPACE::NodeProto &op,
                       const ArgumentDictionaryTy &dict);

  /// Load MatMul ONNX operator.
  llvm::Error loadMatMul(const ONNX_NAMESPACE::NodeProto &op,
                         const ArgumentDictionaryTy &dict);

  /// Load Pad ONNX operator.
  llvm::Error loadPad(const ONNX_NAMESPACE::NodeProto &op,
                      const ArgumentDictionaryTy &dict);

  /// Load Cast ONNX operator.
  llvm::Error loadCast(const ONNX_NAMESPACE::NodeProto &op,
                       const ArgumentDictionaryTy &dict);

  /// Load LeakyRelu ONNX operator.
  llvm::Error loadLeakyRelu(const ONNX_NAMESPACE::NodeProto &op,
                            const ArgumentDictionaryTy &dict);

  /// Load SpaceToDepth ONNX operator.
  llvm::Error loadSpaceToDepth(const ONNX_NAMESPACE::NodeProto &op,
                               const ArgumentDictionaryTy &dict);

  /// Load ConstantOfShape ONNX operator.
  llvm::Error loadConstantOfShape(const ONNX_NAMESPACE::NodeProto &op,
                                  const ArgumentDictionaryTy &dict);

  /// Load Tile ONNX operator.
  llvm::Error loadTile(const ONNX_NAMESPACE::NodeProto &op,
                       const ArgumentDictionaryTy &dict);

protected:
  /// Load the network operators from the GraphProto.
  /// \returns Error if network cannot be loaded.
  llvm::Error loadNetwork(ONNX_NAMESPACE::GraphProto &net);

  /// Set the output nodes of the network \p net. Initializes the map from the
  /// names of the outputs to the save nodes that save each output.
  /// \returns Error if network cannot be loaded.
  llvm::Error setOutputNodes(ONNX_NAMESPACE::GraphProto &net);

  /// Set ir verion and op version.
  llvm::Error setVersion(ONNX_NAMESPACE::ModelProto MP);

  /// \returns Expected<ModelProto> if a ModelProto can be loaded from the
  /// stream \p iStream.
  static llvm::Expected<ONNX_NAMESPACE::ModelProto>
  loadProto(google::protobuf::io::ZeroCopyInputStream &iStream);

  /// Load the network initializers from the GraphProto.
  llvm::Error loadInitializers(ONNX_NAMESPACE::GraphProto &net);

  /// Load the inputs from the GraphProto. If \p loadInputsAsPlaceholders is
  /// true then this will load each graph input as a placeholder otherwise it
  /// will create an empty tensor for each input.
  llvm::Error loadInputs(ONNX_NAMESPACE::GraphProto &net,
                         bool loadInputsAsPlaceholders);

  /// \returns Expected<ModelProto> if a ModelProto can be constructed from the
  /// contents of the file \p filename and Error otherwise.
  /// Loads ModelProto from the file containing serialized protobuf.
  static llvm::Expected<ONNX_NAMESPACE::ModelProto>
  loadProto(const std::string &filename);

  /// \returns Expected<ModelProto> if a ModelProto can be constructed from the
  /// in-memory serialized protobuf.
  /// Loads ModelProto from the in-memory serialized protobuf \p
  /// onnxModel with the model size \p onnxModelSize.
  static llvm::Expected<ONNX_NAMESPACE::ModelProto>
  loadProto(const void *onnxModel, size_t onnxModelSize);

  /// Checks that the inputs tensors are compatible with the inputs declared in
  /// the ONNX model. The input types in \p types match the list of names
  /// \p tensorNames.
  llvm::Error checkInputs(ONNX_NAMESPACE::GraphProto &net,
                          llvm::ArrayRef<const char *> tensorNames,
                          llvm::ArrayRef<TypeRef> types);

  /// Creates a ONNX model loader to build \p F.
  /// Loads the ONNIXFI \p model from memory of \p modelSize size,
  /// and \p weightsCount, and \p onnxTensorDescriptorV1 correspondent
  /// descriptors. Converts inputs into placeholder if requested \p
  /// loadInputsAsPlaceholders. Reports success/failure through optional
  /// parameter \p errPtr.
  ONNXModelLoader(const void *model, uint32_t modelSize, uint32_t weightsCount,
                  const onnxTensorDescriptorV1 *weightDescriptors, Function &F,
                  bool loadInputsAsPlaceholders, llvm::Error *errPtr = nullptr);

  friend class ONNXIFIModelLoader;

public:
  /// Creates a ONNX model loader to build \p F.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  ONNXModelLoader(Function &F, llvm::Error *errPtr = nullptr);

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
                  llvm::Error *errPtr = nullptr);
};

} // namespace glow

#endif // GLOW_IMPORTER_ONNXMODELLOADER_H
