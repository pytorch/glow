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
#include "llvm/ADT/StringRef.h"

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
  /// \returns True if the operator has broadcasting activated.
  llvm::Expected<bool> getBroadcast(const ArgumentDictionaryTy &dict) override;

  /// \returns True if the operator with the name \p typeName has support for
  /// multidirectional broadcasting.
  bool hasMultidirectionalBroadcast(const llvm::StringRef typeName) override;

  /// Load the weight tensors from the 'init' file and register them in the map
  /// \p tensors.
  llvm::Error loadWeightsFromNet(caffe2::NetDef &net);

  /// Loads an individual weight \p op.
  llvm::Error loadWeight(const caffe2::OperatorDef &op);

  /// Load the structure of the network from the 'net' file.
  llvm::Error loadNetwork(caffe2::NetDef &net);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network.
  llvm::Error loadOperator(const caffe2::OperatorDef &op);

  /// Load the Conv or ConvRelu operators.
  llvm::Error loadConv(const caffe2::OperatorDef &op,
                       ArgumentDictionaryTy &dict);

  /// Load the Int8Conv or Int8ConvRelu operators.
  llvm::Error loadConvQuantized(const caffe2::OperatorDef &op,
                                ArgumentDictionaryTy &dict);

  /// Reads a network (weights or structure) from the serialized protocol
  /// buffer file.
  llvm::Expected<caffe2::NetDef> loadProtoFile(const std::string &filename);

  /// loadInputs calls this function for each member in its target arguments.
  /// Currently we are supporting two tensorprototypes:
  /// caffe2::TensorProto, caffe2::QTensorProto
  template <class TensorProtoType>
  llvm::Error loadInputsWithTensorProtoType(const caffe2::NetDef &net,
                                            bool loadInputsAsPlaceholders,
                                            const TensorProtoType &in);

  /// Load the inputs from the NetDef. If \p loadInputsAsPlaceholders is
  /// true then this will load each graph input as a placeholder otherwise it
  /// will create an empty tensor for each input.
  llvm::Error loadInputs(const caffe2::NetDef &net,
                         bool loadInputsAsPlaceholders);

  /// \returns Expected<NetDef> if a NetDef can be constructed from the
  /// in-memory serialized protobuf.
  /// Loads ModelProto from the in-memory serialized protobuf \p
  /// c2Model with the model size \p c2ModelSize.
  static llvm::Expected<caffe2::NetDef> loadProto(const void *c2Model,
                                                  size_t c2ModelSize);

  /// Creates a Caffe2 model loader to build \p F.
  /// Loads the ONNIXFI \p model from memory of \p modelSize size,
  /// and \p weightsCount, and \p onnxTensorDescriptorV1 correspondent
  /// descriptors. Converts inputs into placeholder if requested \p
  /// loadInputsAsPlaceholders. Reports success/failure through optional
  /// parameter \p errPtr.
  Caffe2ModelLoader(const void *model, uint32_t modelSize,
                    uint32_t weightsCount,
                    const onnxTensorDescriptorV1 *weightDescriptors,
                    Function &F, bool loadInputsAsPlaceholders,
                    llvm::Error *errPtr = nullptr);

  friend class ONNXIFIModelLoader;

public:
  /// Loads the caffe2 model that's represented by a network description file,
  /// serialized in \p netDescFilename, and weights file, serialized in
  /// \p netWeightFilename, and populates the network in \p F.
  /// The list \p types and \p names are used to initialized the inputs and
  /// outputs with specific names and types.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  Caffe2ModelLoader(const std::string &netDescFilename,
                    const std::string &netWeightFilename,
                    llvm::ArrayRef<const char *> names,
                    llvm::ArrayRef<TypeRef> types, Function &F,
                    llvm::Error *errPtr = nullptr);

  /// Creates a Caffe2 model loader to build \p F.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  Caffe2ModelLoader(Function &F, llvm::Error *errPtr);
};

} // namespace glow

#endif // GLOW_IMPORTER_CAFFE2MODELLOADER_H
