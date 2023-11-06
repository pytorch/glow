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

#include "glow/Importer/ProtobufLoader.h"
#include "llvm/Support/CommandLine.h"
#include <string>

namespace glow {

llvm::cl::OptionCategory loaderOptCat("Model Loader Options");

static llvm::cl::opt<bool> isConstFoldLoaderOps(
    "const-fold-ops",
    llvm::cl::desc(
        "Performs constant folding on ONNX and Caffe Operators while loading."),
    llvm::cl::init(true), llvm::cl::cat(loaderOptCat));

bool isArrayConstant(llvm::ArrayRef<size_t> a) {
  for (size_t i = 1; i < a.size(); i++)
    if (a[0] != a[i])
      return false;
  return true;
}

void setConstantFoldLoaderOpsFlag(bool flag) { isConstFoldLoaderOps = flag; }

bool getConstantFoldLoaderOpsFlag() { return isConstFoldLoaderOps; }

bool ProtobufLoader::isConstantFoldable(llvm::ArrayRef<NodeValue> inputs,
                                        std::string typeName) const {
  int numInputs = inputs.size();
  if (!getConstantFoldLoaderOpsFlag()) {
    return false;
  }
  // foldUnsupportedTypes: List of typenames unsupported for folding.
  std::string foldUnsupportedTypes[] = {"Constant", "Loop", "If"};
  std::string *findType = std::find(std::begin(foldUnsupportedTypes),
                                    std::end(foldUnsupportedTypes), typeName);
  // Early exit if folding is not supported for current operator.
  if (findType != std::end(foldUnsupportedTypes)) {
    return false;
  }

  // If all the inputs to the operator are constant this op can be folded.
  for (int i = 0; i < numInputs; i++) {
    if (inputs[i].getNode()->getKind() != Kinded::Kind::ConstantKind) {
      return false;
    }
  }
  return true;
}

Placeholder *
ProtobufLoader::getStaticPlaceholderByNameOrNull(llvm::StringRef name) const {
  auto it = nodeValueByName_.find(name);
  if (it == nodeValueByName_.end()) {
    return nullptr;
  }
  auto *res = llvm::dyn_cast<Placeholder>(it->second.getNode());
  return (res && res->isStatic()) ? res : nullptr;
}

Constant *ProtobufLoader::getConstantByNameOrNull(llvm::StringRef name) const {
  auto it = nodeValueByName_.find(name);
  if (it == nodeValueByName_.end()) {
    return nullptr;
  }
  auto *res = llvm::dyn_cast<Constant>(it->second.getNode());
  return res ? res : nullptr;
}

Expected<Constant *>
ProtobufLoader::getConstantByName(llvm::StringRef name) const {
  auto *ptr = getConstantByNameOrNull(name);
  RETURN_ERR_IF_NOT(
      ptr, strFormat("could not find constant with name %s", name.data()));
  return ptr;
}

bool ProtobufLoader::hasConstantByName(llvm::StringRef name) const {
  return getConstantByNameOrNull(name) != nullptr;
}

Expected<Placeholder *> ProtobufLoader::getSingleOutput() const {
  RETURN_ERR_IF_NOT(outputVarsByName_.size() == 1,
                    "There must be only one output.");
  return outputVarsByName_.begin()->second;
}

Expected<Placeholder *> ProtobufLoader::getSingleInput() const {
  RETURN_ERR_IF_NOT(inputVarsByName_.size() == 1,
                    "There must be only one input.");
  return inputVarsByName_.begin()->second;
}

Expected<Placeholder *>
ProtobufLoader::getOutputByName(llvm::StringRef name) const {
  auto it = outputVarsByName_.find(name);
  RETURN_ERR_IF_NOT(
      it != outputVarsByName_.end(),
      llvm::Twine("No external output Variable was registered with name ", name)
          .str());
  return it->second;
}

Expected<Placeholder *>
ProtobufLoader::getInputByName(llvm::StringRef name) const {
  auto it = inputVarsByName_.find(name);
  RETURN_ERR_IF_NOT(
      it != inputVarsByName_.end(),
      llvm::Twine("No external input Variable was registered with name ", name)
          .str());
  return it->second;
}

NodeValue
ProtobufLoader::getNodeValueByNameOrNullNodeValue(llvm::StringRef name,
                                                  bool ignoreSrcFun) {
  auto it = nodeValueByName_.find(name);
  if (it == nodeValueByName_.end()) {
    return NodeValue(nullptr);
  }

  // Always return the NV of a storage Node since Storage lives in the Module
  // and is accessible to any Node.
  NodeValue NV = it->second;
  if (llvm::isa<Storage>(NV)) {
    return NV;
  }

  // Check if the current Function G_ we are loading into is the same as the
  // Function of the NV we found; if so then return it.
  Function *srcF = NV.getNode()->getParent();
  if (srcF == G_ || ignoreSrcFun) {
    return NV;
  }

  // Otherwise we must be looking up a NV from a different Function in the
  // Module, so look for an intermediate Placeholder linking the two if it
  // exists, or otherwise create one and remember it.
  assert(partNameToFun_.size() > 0 &&
         "Must be loading a pre-partitioned model.");
  auto itPH = intermediatePHsByName_.find(name);
  Placeholder *intermedPH = nullptr;
  // Create the intermediate PH and SaveNode if it does not yet exist. Note that
  // we store these intermediate PHs separately from nodeValueByName_ because we
  // want future users from the same Function as the NV to still use the Node
  // directly through nodeValueByName_.
  if (itPH == intermediatePHsByName_.end()) {
    auto *save = srcF->createSave("tmp_" + NV.getNode()->getName().str(), NV);
    intermedPH = save->getPlaceholder();
    intermediatePHsByName_[name] = intermedPH;
  } else {
    intermedPH = itPH->second;
  }
  return intermedPH->getOutput();
}

Expected<NodeValue> ProtobufLoader::getNodeValueByName(llvm::StringRef name,
                                                       bool ignoreSrcFun) {
  RETURN_ERR_IF_NOT(hasNodeByName(name),
                    llvm::Twine("No node under name ", name).str());
  auto node = getNodeValueByNameOrNullNodeValue(name, ignoreSrcFun);
  RETURN_ERR_IF_NOT(node.getNode(), "Null is under that name??");
  return node;
}

Error ProtobufLoader::createAndRegisterConstant(llvm::StringRef name,
                                                Tensor &&tensor,
                                                const std::string &layout) {
  auto it = nodeValueByName_.find(name);
  if (it != nodeValueByName_.end()) {
    if (llvm::dyn_cast<Placeholder>(it->second.getNode())) {
      // Placeholders take precedence over Constants.
      return Error::success();
    }
  }
  // Note: We do not support training from models loaded from protos, so
  // trainable is always set to false here.
  Constant *node = mod_.createConstant(name, std::move(tensor), layout);
  nodeValueByName_[name] = node->getOutput();
  return Error::success();
}

void ProtobufLoader::deleteUnusedConstants() {
  std::vector<std::string> nodeValuesToRemove;
  // Note that it's possible a constant is referred by more than one names
  // (e.g., via Identity operator). Therefore, we maintain a set of constants to
  // erase separately from the list for names.
  std::unordered_set<Constant *> constantToRemove;

  for (auto &kv : nodeValueByName_) {
    auto *node = kv.second.getNode();
    if (auto *c = llvm::dyn_cast<Constant>(node)) {
      if (!c->hasUsers()) {
        nodeValuesToRemove.push_back(kv.getKey().str());
        constantToRemove.insert(c);
      }
    }
  }

  for (auto &name : nodeValuesToRemove) {
    auto it = nodeValueByName_.find(name);
    DCHECK(llvm::isa<Constant>(it->second.getNode()))
        << "NodeValue with name " << name
        << " was expected to have been a Constant";
    nodeValueByName_.erase(it);
  }

  for (auto *c : constantToRemove) {
    G_->getParent()->eraseConstant(c);
  }
}

Expected<Placeholder *>
ProtobufLoader::createAndRegisterPlaceholder(llvm::StringRef name, TypeRef T,
                                             bool isStatic, bool isTrainable,
                                             const std::string &layout) {
  RETURN_ERR_IF_NOT(
      !hasNodeByName(name),
      llvm::Twine("Creating an already existing node ", name).str());
  RETURN_ERR_IF_NOT(!mod_.hasStorageName(name),
                    strFormat("A Placeholder was already registered by name %s",
                              name.data()));

  Placeholder *node = mod_.createPlaceholder(T, name, isTrainable, layout);
  node->setStatic(isStatic);
  nodeValueByName_[name] = node->getOutput();
  return node;
}

bool ProtobufLoader::hasNodeByName(llvm::StringRef name) const {
  return nodeValueByName_.find(name) != nodeValueByName_.end();
}

ProtobufLoader::ProtobufLoader(llvm::ArrayRef<const char *> tensorNames,
                               llvm::ArrayRef<TypeRef> types, Module &mod,
                               Error *errPtr, bool loadIntoExistingModule,
                               OriginNameToTQPMap *originNameToTQPMap,
                               bool loadUniquedDummyQParams,
                               bool replaceDummyTQPs, bool zeroScaleFP16Clip,
                               bool clipQuantRangeToFP16)
    : G_(nullptr), mod_(mod), loadIntoExistingModule_(loadIntoExistingModule),
      originNameToTQPMap_(originNameToTQPMap),
      loadUniquedDummyQParams_(loadUniquedDummyQParams),
      replaceDummyTQPs_(replaceDummyTQPs),
      zeroScaleFP16Clip_(zeroScaleFP16Clip),
      clipQuantRangeToFP16_(clipQuantRangeToFP16) {
  setupLoader(tensorNames, types, errPtr);
}

ProtobufLoader::ProtobufLoader(llvm::ArrayRef<const char *> tensorNames,
                               llvm::ArrayRef<TypeRef> types, Function *F,
                               Error *errPtr, bool loadIntoExistingModule,
                               OriginNameToTQPMap *originNameToTQPMap,
                               bool loadUniquedDummyQParams,
                               bool replaceDummyTQPs, bool zeroScaleFP16Clip,
                               bool clipQuantRangeToFP16)
    : G_(F), mod_(*F->getParent()),
      loadIntoExistingModule_(loadIntoExistingModule),
      originNameToTQPMap_(originNameToTQPMap),
      loadUniquedDummyQParams_(loadUniquedDummyQParams),
      replaceDummyTQPs_(replaceDummyTQPs),
      zeroScaleFP16Clip_(zeroScaleFP16Clip),
      clipQuantRangeToFP16_(clipQuantRangeToFP16) {
  setupLoader(tensorNames, types, errPtr);
}

void ProtobufLoader::setupLoader(llvm::ArrayRef<const char *> tensorNames,
                                 llvm::ArrayRef<TypeRef> types, Error *errPtr) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // if errPtr already contains an error then don't continue with constructor
  if (errPtr && *errPtr) {
    return;
  }

  // Use the global flag as default. This may be overridden by instantiations of
  // the loader later on.
  constFoldInLoader_ = getConstantFoldLoaderOpsFlag();

  // Lambda to setup the ProtobufLoader and return any Errors that were
  // raised.
  auto setup = [&]() -> Error {
    RETURN_ERR_IF_NOT(tensorNames.size() == types.size(),
                      "Invalid initialization list");
    for (size_t i = 0, e = tensorNames.size(); i < e; i++) {
      RETURN_ERR_IF_NOT(!hasNodeByName(tensorNames[i]),
                        "Input names have duplicate");
      TypeRef T = types[i];
      if (T->isQuantizedType() && !T->isFusedQuantizedType()) {
        RETURN_ERR_IF_NOT(!clipQuantRangeToFP16_,
                          strFormat("Do not support clipQuantRangeToFP16 with "
                                    "unfused quantized input Placeholders: %s",
                                    tensorNames[i]));
        // Note: Never shift here, because these are the types that were already
        // imported/defined based on Glow.
        ASSIGN_VALUE_OR_RETURN_ERR(
            T, loadQuantTy(tensorNames[i], T->getElementType(), T->dims(),
                           T->getScale(), T->getOffset(),
                           /* shiftUInt8ToInt8 */ false));
      }
      Placeholder *placeholder;
      ASSIGN_VALUE_OR_RETURN_ERR(
          placeholder, createAndRegisterPlaceholder(tensorNames[i], T));
      inputVarsByName_.try_emplace(tensorNames[i], placeholder);
    }
    return Error::success();
  };

  if (errPtr) {
    *errPtr = setup();
  } else {
    EXIT_ON_ERR(setup());
  }
}

Expected<TensorQuantizationParams>
ProtobufLoader::getUpdatedTQP(int32_t uniqueOffsetIdx) {
  RETURN_ERR_IF_NOT(replaceDummyTQPs_, "replaceDummyTQPs_ was not enabled");
  RETURN_ERR_IF_NOT(
      uniqueOffsetIdx < int32_t(updatedTQPs_.size()),
      strFormat("Unexpected size of updated TQPs %lu vs. dummy offset %d",
                updatedTQPs_.size(), uniqueOffsetIdx));
  return updatedTQPs_[uniqueOffsetIdx];
}

Expected<TypeRef> ProtobufLoader::loadQuantTy(const std::string &name,
                                              ElemKind k,
                                              llvm::ArrayRef<dim_t> dims,
                                              float scale, int32_t offset,
                                              bool shiftUInt8ToInt8,
                                              bool skipClipQuantRangeToFP16) {
  // If we have Int8QTy, we may have loaded as UInt8, and so will need to shift
  // to align to Glow's Int8QTy.
  if (k == ElemKind::Int8QTy && shiftUInt8ToInt8) {
    offset -= UINT8_TO_INT8_SHIFT;
  }

  // If we don't have a map to track dummy unique offsets to loader names, then
  // just load as normal with the actual scale/offset we loaded.
  if (!loadUniquedDummyQParams_) {
    // If clipping qparams to fp16 range then do so here.
    if (clipQuantRangeToFP16_ && !skipClipQuantRangeToFP16) {
      const auto qMinMax = getQuantizedValueRange(scale, offset, k);
      const float newMin = std::max(qMinMax.first, kMinFP16);
      const float newMax = std::min(qMinMax.second, kMaxFP16);
      const TensorQuantizationParams newQParams = chooseQuantizationParams(
          {newMin, newMax}, quantization::Asymmetric, k);
      scale = newQParams.scale;
      offset = newQParams.offset;
    }
    // If we are clipping qparam scales below the kMinScaleFP16 threshold to
    // kMinScaleFP16 then do so here.
    if (zeroScaleFP16Clip_ && scale < kMinScaleFP16) {
      scale = kMinScaleFP16;
    }

    if (originNameToTQPMap_) {
      bool inserted =
          originNameToTQPMap_
              ->emplace(name, TensorQuantizationParams{scale, offset})
              .second;
      RETURN_ERR_IF_NOT(inserted, "Already inserted TQP for " + name);
    }
    return mod_.uniqueType(k, dims, scale, offset);
  }

  RETURN_ERR_IF_NOT(originNameToTQPMap_,
                    "Must have valid originNameToTQPMap_ when loading "
                    "uniqued dummy qparams.");

  // We use dummyScale to represent a dummy scale/offset pair. Make sure the
  // original model did not have dummyScale, since we will use it later on to
  // verify all qparams are now dummies.
  RETURN_ERR_IF_NOT(scale != dummyScale, "Found dummy scale for " + name);

  // For uniqued scale/offset, ignore the actual loaded values. Instead use
  // dummyScale to signal these quant params are dummies, and then a uniqued
  // incremented offset to represent this unique quant param pair. Save the name
  // of the C2 edge that we loaded to use these quant params in the cctx so we
  // can ue it in the future. The index the name is at represents which unique
  // index it is mapped to.
  RETURN_ERR_IF_NOT(int32_t(originNameToTQPMap_->size()) == currUniqueOffset_,
                    "Unexpected size encountered for qparam origin tracking");
  const int32_t thisUniqueOffset = currUniqueOffset_++;
  bool inserted =
      originNameToTQPMap_
          ->emplace(name,
                    TensorQuantizationParams{dummyScale, thisUniqueOffset})
          .second;
  RETURN_ERR_IF_NOT(inserted, "Already inserted TQP for " + name);
  return mod_.uniqueType(k, dims, dummyScale, thisUniqueOffset);
}

}; // namespace glow
