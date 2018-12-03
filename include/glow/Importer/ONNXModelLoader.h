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

  /// Load the network initializers from the GraphProto.
  llvm::Error loadInitializers(ONNX_NAMESPACE::GraphProto &net);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network. \returns Error if operator \p op cannot be loaded.
  llvm::Error loadOperator(const ONNX_NAMESPACE::NodeProto &op);

  /// ONNX model ir_version;
  size_t irVersion_;

  /// ONNX model op_version;
  size_t opsetVersion_;

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

public:
  /// Creates a ONNX model loader to build \p F.
  ONNXModelLoader(Function &F);

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

  /// Loads the ONNX model that's represented by a model description file,
  /// serialized in \p modelDescFilename and populates the network into \p F.
  /// The types in \p types match the list of names \p tensorNames and used as
  /// inputs to the network.
  ONNXModelLoader(const std::string &modelDescFilename,
                  llvm::ArrayRef<const char *> tensorNames,
                  llvm::ArrayRef<TypeRef> types, Function &F);
};

} // namespace glow

#endif // GLOW_IMPORTER_ONNXMODELLOADER_H
