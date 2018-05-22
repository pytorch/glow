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

#ifndef GLOW_IMPORTER_CAFFE2_H
#define GLOW_IMPORTER_CAFFE2_H

#include "glow/Graph/Graph.h"
#include "glow/Importer/ProtobufLoader.h"

#include "llvm/ADT/ArrayRef.h"

#include <string>

namespace caffe2 {
class OperatorDef;
class NetDef;
} // namespace caffe2

namespace glow {

class Tensor;
class Value;

/// Loads caffe2 models.
class caffe2ModelLoader : public ProtobufLoader {
  /// Load the weight tensors from the 'init' file and register them in the map
  /// \p tensors.
  void loadWeights(caffe2::NetDef &net);

  /// Load the structure of the network from the 'net' file.
  void loadNetwork(caffe2::NetDef &net);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network.
  void loadOperator(const caffe2::OperatorDef &op);

  /// Reads a network (weights or structure) from the serialized protocol buffer
  /// file.
  bool loadProtoFile(caffe2::NetDef &net, const std::string &filename);

public:
  /// Loads the caffe2 model that's represented by a network description file,
  /// serialized in \p netDescFilename, and weights file, serialized in
  /// \p netWeightFilename, and populates the network in \p F.
  /// The tensors in \p tensors are stored with the names in the list of names
  /// \p names and used as inputs to the network.
  caffe2ModelLoader(const std::string &netDescFilename,
                    const std::string &netWeightFilename,
                    llvm::ArrayRef<const char *> names,
                    llvm::ArrayRef<Tensor *> tensors, Function &F);
};

} // namespace glow

#endif // GLOW_IMPORTER_CAFFE2_H
