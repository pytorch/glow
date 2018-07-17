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

namespace onnx {
class AttributeProto;
class NodeProto;
class GraphProto;
class ModelProto;
} // namespace onnx

namespace glow {

/// Loads ONNX models.
class ONNXModelLoader
    : public CommonOperatorLoader<onnx::NodeProto, onnx::AttributeProto> {
  /// Get the broadcast attribute based on different ONNX op versions.
  bool getBroadcast(const ArgumentDictionaryTy &dict) override;

  /// Set ir verion and op version.
  void setVersion(onnx::ModelProto MP);

  /// Load the network operators from the GraphProto.
  void loadNetwork(onnx::GraphProto &net);

  /// Load the network initializers from the GraphProto.
  void loadInitializers(onnx::GraphProto &net);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network.
  void loadOperator(const onnx::NodeProto &op);

  /// Reads a network (weights or structure) from the serialized protocol buffer
  /// file.
  bool loadProtoFile(onnx::GraphProto &net, const std::string &filename);

  /// ONNX model ir_version;
  size_t irVersion_;

  /// ONNX model op_version;
  size_t opsetVersion_;

public:
  /// Loads the ONNX model that's represented by a model description file,
  /// serialized in \p modelDescFilename and populates the network into \p F.
  /// The tensors in \p tensors are stored with the names in the list of names
  /// \p names and used as inputs to the network.
  ONNXModelLoader(const std::string &modelDescFilename,
                  llvm::ArrayRef<const char *> names,
                  llvm::ArrayRef<Tensor *> tensors, Function &F);
};

} // namespace glow

#endif // GLOW_IMPORTER_ONNX_H
