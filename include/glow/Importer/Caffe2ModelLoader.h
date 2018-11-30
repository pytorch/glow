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

#ifndef GLOW_IMPORTER_CAFFE2MODELLOADER_H
#define GLOW_IMPORTER_CAFFE2MODELLOADER_H

#include "glow/Graph/Graph.h"
#include "glow/Importer/CommonOperatorLoader.h"

#include "llvm/ADT/ArrayRef.h"

#include <string>

namespace caffe2 {
class Argument;
class OperatorDef;
class NetDef;
} // namespace caffe2

namespace glow {

class Tensor;
class Value;

/// Loads caffe2 models.
class Caffe2ModelLoader
    : public CommonOperatorLoader<caffe2::OperatorDef, caffe2::Argument> {
  /// Get the broadcast attribute.
  llvm::Expected<bool> getBroadcast(const ArgumentDictionaryTy &dict) override;

  /// Load the weight tensors from the 'init' file and register them in the map
  /// \p tensors.
  llvm::Error loadWeights(caffe2::NetDef &net);

  /// Loads an individual weight \p op.
  llvm::Error loadWeight(const caffe2::OperatorDef &op);

  /// Load the structure of the network from the 'net' file.
  llvm::Error loadNetwork(caffe2::NetDef &net);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network.
  llvm::Error loadOperator(const caffe2::OperatorDef &op);

  /// Reads a network (weights or structure) from the serialized protocol buffer
  /// file.
  llvm::Expected<caffe2::NetDef> loadProtoFile(const std::string &filename);

public:
  /// Loads the caffe2 model that's represented by a network description file,
  /// serialized in \p netDescFilename, and weights file, serialized in
  /// \p netWeightFilename, and populates the network in \p F.
  /// The list \p types and \p names are used to initialized the inputs and
  /// outputs with specific names and types.
  Caffe2ModelLoader(const std::string &netDescFilename,
                    const std::string &netWeightFilename,
                    llvm::ArrayRef<const char *> names,
                    llvm::ArrayRef<TypeRef> types, Function &F);
};

} // namespace glow

#endif // GLOW_IMPORTER_CAFFE2MODELLOADER_H
