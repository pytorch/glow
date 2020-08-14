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

#ifndef GLOW_IMPORTER_CAFFE2MODELLOADER_H
#define GLOW_IMPORTER_CAFFE2MODELLOADER_H

#include "glow/Graph/Graph.h"
#include "glow/Importer/CommonOperatorLoader.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "caffe2/proto/caffe2.pb.h"

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
  Expected<bool> getBroadcast(ArgumentDictionaryTy &dict) override;

  /// \returns True if the operator with the name \p typeName has support for
  /// multidirectional broadcasting.
  bool hasMultidirectionalBroadcast(const llvm::StringRef typeName) override;

  /// Load the weight tensors from the 'init' file and register them in the map
  /// \p tensors.
  Error loadWeightsFromNet(caffe2::NetDef &net);

  /// Loads an individual weight \p op.
  Error loadWeight(const caffe2::OperatorDef &op);

  /// Load the structure of the network from the 'net' file.
  Error loadNetwork(caffe2::NetDef &net);

  /// Load the operator \p op into the network. This creates one or more nodes
  /// in the network.
  Error loadOperator(const caffe2::OperatorDef &op);

  /// \returns True if the operator \p op is successfully folded.
  Expected<bool> foldOperator(const caffe2::OperatorDef &op);

  /// Helper function to print better log information for operator failure cases
  const std::string opErrMsg(const caffe2::OperatorDef &op,
                             const std::string &errMsg);
  /// Load the Conv or ConvRelu operators.
  Error loadConv(const caffe2::OperatorDef &op, ArgumentDictionaryTy &dict);

  /// Load the ConvTranspose operator.
  Error loadConvTranspose(const caffe2::OperatorDef &op,
                          ArgumentDictionaryTy &dict);

  /// Load the Int8Conv or Int8ConvRelu operators.
  Error loadConvQuantized(const caffe2::OperatorDef &op,
                          ArgumentDictionaryTy &dict);

  /// Load LayerNorm Caffe2 operator \p op given \p dict.
  Error loadLayerNorm(const caffe2::OperatorDef &op,
                      ArgumentDictionaryTy &dict);

  /// Reads a network (weights or structure) from the serialized protocol
  /// buffer file.
  Expected<caffe2::NetDef> loadProtoFile(const std::string &filename);

  /// loadInputs calls this function for each member in its target arguments.
  /// Currently we are supporting two tensorprototypes:
  /// caffe2::TensorProto, caffe2::QTensorProto
  template <class TensorProtoType>
  Error loadInputsWithTensorProtoType(
      const caffe2::NetDef &net,
      const std::unordered_set<std::string> &initializers,
      const TensorProtoType &in);

  /// Creates tensor \p T from the input \p in. Note, there is no data
  /// associated with the Tensor. This method makes sure that the tensor is
  /// created with the proper shape and element type.
  Expected<LoadWeightResult>
  createAndSetTensorType(const caffe2::TensorProto &in);

  /// Creates quantized tensor \p T from the input \p in. Note, there is no data
  /// associated with the Tensor. This method makes sure that the tensor is
  /// created with the proper shape and element type.
  Expected<LoadWeightResult>
  createAndSetTensorType(const caffe2::QTensorProto &in);

  /// Load the inputs from the NetDef. \p initializers is the set of tensors
  /// that should be loaded as empty Constants in the graph for the purposes of
  /// onnxifi compatibility checks, any other inputs will be loaded as
  /// placeholders.
  Error loadInputs(const caffe2::NetDef &net,
                   const std::unordered_set<std::string> &initializers);

  /// \returns Expected<NetDef> if a NetDef can be constructed from the
  /// in-memory serialized protobuf.
  /// Loads ModelProto from the in-memory serialized protobuf \p
  /// c2Model with the model size \p c2ModelSize.
  static Expected<caffe2::NetDef> loadProto(const void *c2Model,
                                            size_t c2ModelSize);

  /// Creates a Caffe2 model loader to build one or more Functions in \p mod.
  /// Loads the ONNIXFI \p model from memory of \p modelSize size,
  /// and \p weightsCount, and \p onnxTensorDescriptorV1 correspondent
  /// descriptors. Reports success/failure through optional parameter \p errPtr.
  /// This constructor always overrides the default constant folding in loader
  /// flag with \p constFoldInLoader. If the model is pre-partitioned, then \p
  /// PPC will be filled with relevant configuration for partitioning, and all
  /// Functions created will be named with prefix /p funNamePrefix. Otherwise \p
  /// PPC is ignored, and \p funNamePrefix is used as the name of the single
  /// Function that is created inside \p mod. If \p originNameToTQPMap is
  /// non-null then names of ops and inputs that are quantized will be mapped to
  /// the TQP that it came with. If \p loadUniquedDummyQParams then the actual
  /// quant params in the model will be discarded and unique dummies will be
  /// used instead.
  Caffe2ModelLoader(const void *model, uint32_t modelSize,
                    uint32_t weightsCount,
                    const onnxTensorDescriptorV1 *weightDescriptors,
                    Module &mod, llvm::StringRef funNamePrefix,
                    runtime::PrePartitionedConfig *PPC, Error *errPtr = nullptr,
                    bool constFoldInLoader = true,
                    OriginNameToTQPMap *originNameToTQPMap = nullptr,
                    bool loadUniquedDummyQParams = false);

  friend class ONNXIFIModelLoader;

  /// Complete initialization when loading a module, including loading
  /// pre-partitioned models, given \p networkDef loaded from caller, as well as
  /// \p funNamePrefix, and \p PPC forwarded from caller.
  Error initWithModule(caffe2::NetDef &networkDef,
                       llvm::StringRef funNamePrefix,
                       runtime::PrePartitionedConfig *PPC);

  /// \returns success if the folding of operator \p op in the loader
  /// \p loader is successful. The folding utility uses temporary
  /// loader \p tmpLoader, and associated temporary function \p F.
  template <class LoaderType, class OpType>
  friend Error constantFoldInLoader(Function *F, LoaderType &tmpLoader,
                                    LoaderType *loader, const OpType &op);

public:
  /// Loads the caffe2 model that's represented by a network description file,
  /// serialized in \p netDescFilename, and weights file, serialized in
  /// \p netWeightFilename, and populates the network in \p F.
  /// The list \p types and \p names are used to initialized the inputs and
  /// outputs with specific names and types.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  /// If \p originNameToTQPMap is non-null then names of ops and inputs that are
  /// quantized will be mapped to the TQP that it came with.
  /// If \p loadUniquedDummyQParams then the actual quant params in the model
  /// will be discarded and unique dummies will be used instead.
  Caffe2ModelLoader(const std::string &netDescFilename,
                    const std::string &netWeightFilename,
                    llvm::ArrayRef<const char *> names,
                    llvm::ArrayRef<TypeRef> types, Function &F,
                    Error *errPtr = nullptr,
                    OriginNameToTQPMap *originNameToTQPMap = nullptr,
                    bool loadUniquedDummyQParams = false);

  /// Loads the caffe2 model that's represented by a network description file,
  /// serialized in \p netDescFilename, and weights file, serialized in
  /// \p netWeightFilename, and populates the network in \p mod.
  /// Any Functions created in \p mod will have name (or prefixed name for
  /// pre-partitioned protos) \p funNamePrefix.  \p PPC is used to store the
  /// pre-partitioned config for the model if relevant.
  /// The list \p types and \p names are used to initialized the inputs and
  /// outputs with specific names and types.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  Caffe2ModelLoader(const std::string &netDescFilename,
                    const std::string &netWeightFilename,
                    llvm::ArrayRef<const char *> names,
                    llvm::ArrayRef<TypeRef> types, Module &mod,
                    llvm::StringRef funNamePrefix,
                    runtime::PrePartitionedConfig *PPC = nullptr,
                    Error *errPtr = nullptr);

  /// Creates a Caffe2 model loader to build \p F.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  Caffe2ModelLoader(Function &F, Error *errPtr);

  /// Creates a Caffe2 model loader that builds into what is intended to be a
  /// dummy Module in \p dummyMod, in order to fill in \p originNameToTQPMap
  /// with a map from C2 op names to TQPs that they were loaded with in model
  /// \p modelStr given \p weightsCount and \p weightDescriptors. Returns any
  /// errors into \p errPtr.
  Caffe2ModelLoader(const std::string &modelStr, uint32_t weightsCount,
                    const onnxTensorDescriptorV1 *weightDescriptors,
                    Module &dummyMod, Error *errPtr,
                    OriginNameToTQPMap *originNameToTQPMap);
};

} // namespace glow

#endif // GLOW_IMPORTER_CAFFE2MODELLOADER_H
