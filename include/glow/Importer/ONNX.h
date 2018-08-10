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

#ifndef GLOW_IMPORTER_ONNX_H
#define GLOW_IMPORTER_ONNX_H

#include "glow/Graph/Graph.h"
#include "glow/Importer/CommonOperatorLoader.h"

#include "llvm/ADT/ArrayRef.h"

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
    : public CommonOperatorLoader<ONNX_NAMESPACE::NodeProto, ONNX_NAMESPACE::AttributeProto> {
  /// Get the broadcast attribute based on different ONNX op versions.
  bool getBroadcast(const ArgumentDictionaryTy &dict) override;

  /// Set ir verion and op version.
  void setVersion(ONNX_NAMESPACE::ModelProto MP);

  /// Load the network initializers from the GraphProto.
  void loadInitializers(ONNX_NAMESPACE::GraphProto &net);

  /// \returns true if operator \p op can be loaded.
  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network.
  bool loadOperator(const ONNX_NAMESPACE::NodeProto &op);

  /// \returns true if \p net can be constructed from the content of the
  /// file \p filename.
  /// Loads GraphProto \p net from the file containing serialized protobuf.
  bool loadProto(ONNX_NAMESPACE::GraphProto &net, const std::string &filename);

  /// \returns true if GraphProto \p net can be loaded from the stream \p
  /// iStream.
  bool loadProto(ONNX_NAMESPACE::GraphProto &net,
                 google::protobuf::io::ZeroCopyInputStream &iStream);

  /// ONNX model ir_version;
  size_t irVersion_;

  /// ONNX model op_version;
  size_t opsetVersion_;

protected:
  /// Creates a ONNX model loader to build \p F.
  ONNXModelLoader(Function &F);

  /// Load the network operators from the GraphProto.
  /// \returns true if network can be loaded.
  bool loadNetwork(ONNX_NAMESPACE::GraphProto &net);

  /// \returns true if \p net can be constructed from the in-memory
  /// serialized protobuf.
  /// Loads GraphProto \p net from the in-memory serialized protobuf \p
  /// onnxModel with the model size \p onnxModelSize.
  bool loadProto(ONNX_NAMESPACE::GraphProto &net, const void *onnxModel,
                 size_t onnxModelSize);

  /// Set the output nodes of the network \p net. Initializes the map from the
  /// names of the outputs to the save nodes that save each output.
  /// \returns false if no output nodes are found in the model.
  bool setOutputNodes(ONNX_NAMESPACE::GraphProto &net);

public:
  /// Loads the ONNX model that's represented by a model description file,
  /// serialized in \p modelDescFilename and populates the network into \p F.
  /// The tensors in \p tensors are stored with the names in the list of names
  /// \p tensorNames and used as inputs to the network.
  ONNXModelLoader(const std::string &modelDescFilename,
                  llvm::ArrayRef<const char *> tensorNames,
                  llvm::ArrayRef<Tensor *> tensors, Function &F);
};

} // namespace glow

#endif // GLOW_IMPORTER_ONNX_H
