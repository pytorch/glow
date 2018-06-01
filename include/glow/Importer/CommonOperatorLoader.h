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

#ifndef GLOW_IMPORTER_COMMONOPERATORLOADER_H
#define GLOW_IMPORTER_COMMONOPERATORLOADER_H

#include "glow/Importer/ProtobufLoader.h"

#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"

#include "llvm/ADT/ArrayRef.h"

#include <functional>
#include <string>
#include <unordered_map>

namespace glow {

/// Contains loaders for operators, which are common to ONNX and Caffe2 formats.
/// Every loader method adds necessary nodes to property G_, which is inherited
/// from ProtobufLoader class, therefore modifying the class instance itself.
template <typename OpType, typename AttrType>
class CommonOperatorLoader : public ProtobufLoader {
protected:
  using ArgumentDictionaryTy =
      std::unordered_map<std::string, const AttrType *>;

  void addNodeAsOutput(const OpType &op, Node *R) {
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = R;
    }
  }

  /// Loads RELU operator, given its protobuf representation and parsed args.
  void loadRelu(const OpType &op, const ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto *in = getOrCreateNodeByName(op.input(0));
    auto *R = G_.createRELU(opName, in);
    addNodeAsOutput(op, R);
  }
  // TODO: move more common operators here (from ONNX.cpp and Caffe2.cpp)

  using ProtobufLoader::ProtobufLoader;

  /// If operator type is supported, returns true and creates new operator.
  /// Otherwise returns false.
  bool tryLoadCommonOperator(llvm::StringRef typeName, const OpType &op,
                             const ArgumentDictionaryTy &dict) {
    if (typeName == "Relu") {
      loadRelu(op, dict);
      return true;
    }
    return false;
  }
};

} // namespace glow

#endif // GLOW_IMPORTER_COMMONOPERATORLOADER_H
