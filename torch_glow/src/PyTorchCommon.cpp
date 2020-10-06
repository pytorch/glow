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

#include <fstream>
#include <string>

#include "PyTorchCommon.h"

#include "GlowFuser.h"
#include "PyTorchModelLoader.h"
#include "Registration.h"
#include "ShapeInferenceEngine.h"

#include "torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h"
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator_options.h>

#include "torch/csrc/jit/ir/irparser.h"

#include <torch/script.h>

DEFINE_string(torch_glow_backend, "Interpreter",
              "Glow backend used for torchifi");
DEFINE_int32(torch_glow_num_devices, -1, "Number of devices for Glow backend");

DEFINE_bool(saturateHost, false, "See PyTorchLoaderSettings");

DEFINE_int32(torch_glow_min_fusion_group_size, 1,
             "Minimum number of nodes in the glow fusion group");
DEFINE_bool(printJITIndex, false, "Enable printing of jit node indexes");
DEFINE_bool(dumpGlowDag, false, "See PyTorchLoaderSettings");
DEFINE_bool(jitVsGlowCompare, false, "Enable per-group error check");
DEFINE_bool(dumpFinalGlowGraph, false, "See PyTorchLoaderSettings");
DEFINE_bool(enableGlowTracing, false, "See PyTorchLoaderSettings");
DEFINE_int32(numTracesPerDump, 1, "See PyTorchLoaderSettings");

// settings for model precision conversion
DEFINE_bool(convertToFP16, false, "See PyTorchLoaderSettings");
DEFINE_bool(convertFusedToFP16, false, "See PyTorchLoaderSettings");
DEFINE_bool(clipFP16, false, "See PyTorchLoaderSettings");
DEFINE_bool(clipFP16SkipInputs, true, "See PyTorchLoaderSettings");
DEFINE_bool(convertPlaceholdersToFP16, true, "See PyTorchLoaderSettings");
DEFINE_bool(convertConstantsToFP16, true, "See PyTorchLoaderSettings");
DEFINE_bool(forceFP16AccumSLS, true, "See PyTorchLoaderSettings");

DEFINE_string(opBlacklist, "", "See PyTorchLoaderSettings");
DEFINE_int32(replicationCount, 1, "Number of replications on each device");
DEFINE_bool(writeToOnnx, false, "See PyTorchLoaderSettings");
DEFINE_bool(onnxZipMode, false, "See PyTorchLoaderSettings");
DEFINE_int32(maxActiveRequests, 250,
             "Max number of active requests before HostManager starts queuing");
DEFINE_bool(randomizeConstants, false, "See PyTorchLoaderSettings");
DEFINE_bool(runShapeInference, false, "See PyTorchLoaderSettings");
DEFINE_int32(fusionStartIndex, -1, "See PyTorchLoaderSettings");
DEFINE_int32(fusionEndIndex, -1, "See PyTorchLoaderSettings");
DEFINE_bool(setIncludeLastOffsets, true, "See PyTorchLoaderSettings");
DEFINE_bool(inferShapeForCompilation, false,
            "Infer shape for the entire model for compilation");
DEFINE_bool(enableRemoveMutation, true, "See PyTorchLoaderSettings");

namespace glow {

namespace {

static int setGraphExecutorToLegacy() {
  // use legacy GraphExecutor for Glow
  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  return 0;
}

static const int USE_LEGACY_GE = setGraphExecutorToLegacy();

} // namespace

std::shared_ptr<runtime::HostManager>
getHostManager(const std::string &backendName, int32_t numDevices) {
  static std::mutex m_;
  std::unique_lock<std::mutex> lock(m_);
  static std::unordered_map<std::string, std::weak_ptr<runtime::HostManager>>
      map_;

  std::shared_ptr<runtime::HostManager> hostManager;
  auto it = map_.find(backendName);
  if (it != map_.end()) {
    hostManager = it->second.lock();
  }

  // If HostManager was found, check that it's valid, otherwise create a new
  // HostManager
  if (hostManager) {
    if (numDevices != -1) {
      CHECK_EQ(hostManager->numDevices(), numDevices)
          << "Tried to create a new HostManager for backend \"" << backendName
          << "\" but there is already an existing HostManager in use for that "
             "Backend but with a different number of devices";
    }
  } else {
    // If number of devices isn't specified then just use 1 device.
    if (numDevices < 0) {
      numDevices = 1;
    }
    std::vector<std::unique_ptr<runtime::DeviceConfig>> deviceConfigs;
    for (int i = 0; i < numDevices; i++) {
      auto config = std::make_unique<runtime::DeviceConfig>(backendName);
      config->deviceID = i;
      deviceConfigs.push_back(std::move(config));
    }

    glow::runtime::HostConfig hostConfig;
    hostConfig.maxActiveRequests = FLAGS_maxActiveRequests;

    hostManager = std::make_shared<runtime::HostManager>(
        std::move(deviceConfigs), hostConfig);

    map_[backendName] = hostManager;
  }
  return hostManager;
}

std::shared_ptr<runtime::HostManager> getHostManager() {
  auto &settings = getPyTorchLoaderSettings();
  return getHostManager(settings.backendName, settings.numDevices);
}

/// Given a Glow ElemKind \p ty, \returns a matching PyTorch ScalarType.
c10::ScalarType elemKindToScalarType(glow::ElemKind ty) {
  switch (ty) {
  case ElemKind::FloatTy:
    return at::kFloat;
  case ElemKind::Float16Ty:
    return at::kHalf;
  case ElemKind::BFloat16Ty:
    return at::kBFloat16;
  case ElemKind::Int32ITy:
    return at::kInt;
  case ElemKind::Int64ITy:
    return at::kLong;
  case ElemKind::BoolTy:
    return at::kBool;
  case ElemKind::Int8QTy:
    return at::kQInt8;
  case ElemKind::UInt8FusedQTy:
  case ElemKind::UInt8FusedFP16QTy:
  case ElemKind::UInt4FusedFP16QTy:
  case ElemKind::UInt4FusedQTy:
  case ElemKind::UInt8QTy:
  case ElemKind::Int16QTy:
  case ElemKind::Int32QTy:
    LOG(DFATAL) << "Not supported yet.";
    return at::kLong;
  }
  LOG(DFATAL) << "Cannot reach here.";
}

/// Given a PyTorch ScalarType \p ty, \returns a matching Glow ElemKind.
glow::ElemKind scalarTypeToElemKind(c10::ScalarType ty) {
  if (ty == at::kFloat) {
    return ElemKind::FloatTy;
  } else if (ty == at::kHalf) {
    return ElemKind::Float16Ty;
  } else if (ty == at::kInt) {
    return ElemKind::Int32ITy;
  } else if (ty == at::kLong) {
    return ElemKind::Int64ITy;
  } else if (ty == at::kBool) {
    return ElemKind::BoolTy;
  } else if (ty == at::kByte) {
    // We should have an 8-byte non-quantized integer type eventually
    // Currently usage of Bool is fine
    return ElemKind::BoolTy;
  } else if (ty == at::kQInt8) {
    return ElemKind::Int8QTy;
  } else if (ty == at::kQUInt8) {
    return ElemKind::UInt8QTy;
  } else {
    LOG(DFATAL) << "ScalarType " << static_cast<int>(ty)
                << " not supported yet.";
    return ElemKind::Int64ITy;
  }
}

/// Given a c10 typekind \p ty, \returns a matching Glow ElemKind.
ElemKind typeKindToElemKind(c10::TypeKind ty) {
  if (ty == c10::TypeKind::FloatType) {
    return ElemKind::FloatTy;
  } else if (ty == c10::TypeKind::IntType) {
    return ElemKind::Int32ITy;
  } else if (ty == c10::TypeKind::BoolType) {
    return ElemKind::BoolTy;
  } else {
    LOG(DFATAL) << "Not supported yet.";
    return ElemKind::Int64ITy;
  }
}

/// Split string \p s on character \p k and eliminate spaces.
static std::vector<std::string> splitString(const std::string &s,
                                            const char k = ',') {
  std::vector<std::string> substrings;
  size_t start = 0;
  bool lastWasSplit = true;
  for (size_t i = 0; i < s.size(); i++) {
    if (lastWasSplit && s[i] == ' ') {
      start = i + 1;
      continue;
    }
    lastWasSplit = false;
    if (s[i] == k) {
      substrings.push_back(s.substr(start, i - start));
      start = i + 1;
      lastWasSplit = true;
    }
  }

  if (start < s.size() - 1) {
    substrings.push_back(s.substr(start, s.size() - start));
  }

  return substrings;
}

void PyTorchLoaderSettings::initSettings() {
  minFusionGroupSize = FLAGS_torch_glow_min_fusion_group_size;
  dumpGlowDag = FLAGS_dumpGlowDag;
  jitVsGlowCompare = FLAGS_jitVsGlowCompare;
  printJITIndex = FLAGS_printJITIndex;
  dumpFinalGlowGraph = FLAGS_dumpFinalGlowGraph;
  enableGlowTracing = FLAGS_enableGlowTracing;
  numTracesPerDump = FLAGS_numTracesPerDump;
  saturateHost = FLAGS_saturateHost;
  convertToFP16 = FLAGS_convertToFP16;
  convertFusedToFP16 = FLAGS_convertFusedToFP16;
  clipFP16 = FLAGS_clipFP16;
  clipFP16SkipInputs = FLAGS_clipFP16SkipInputs;
  convertPlaceholdersToFP16 = FLAGS_convertPlaceholdersToFP16;
  convertConstantsToFP16 = FLAGS_convertConstantsToFP16;
  forceFP16AccumSLS = FLAGS_forceFP16AccumSLS;

  replicationCount = FLAGS_replicationCount;
  writeToOnnx = FLAGS_writeToOnnx;
  onnxZipMode = FLAGS_onnxZipMode;
  randomizeConstants = FLAGS_randomizeConstants;
  backendName = FLAGS_torch_glow_backend;
  numDevices = FLAGS_torch_glow_num_devices;
  runShapeInference = FLAGS_runShapeInference;
  fusionStartIndex = FLAGS_fusionStartIndex;
  fusionEndIndex = FLAGS_fusionEndIndex;
  setIncludeLastOffsets = FLAGS_setIncludeLastOffsets;
  inferShapeForCompilation = FLAGS_inferShapeForCompilation;
  enableRemoveMutation = FLAGS_enableRemoveMutation;

  if (!FLAGS_opBlacklist.empty()) {
    auto kindStrings = splitString(FLAGS_opBlacklist);
    for (const auto &kindString : kindStrings) {
      opBlacklist.insert(torch::jit::Symbol::fromQualString(kindString));
    }
  }
}

PyTorchLoaderSettings::PyTorchLoaderSettings() { initSettings(); }

namespace {
Expected<bool> strToBool(const std::string &str) {
  static std::unordered_set<std::string> trueVals = {"True", "true", "1"};
  static std::unordered_set<std::string> falseVals = {"False", "false", "0"};
  if (trueVals.count(str) > 0) {
    return true;
  }
  if (falseVals.count(str) > 0) {
    return false;
  }
  return MAKE_ERR(strFormat("Invalid truth value: %s", str.c_str()));
}
} // namespace

PyTorchLoaderSettings &getPyTorchLoaderSettings() {
  static PyTorchLoaderSettings settings;
  return settings;
}

#define TRY_LOAD_BOOL_FROM_DICT(KEY_, DICT_)                                   \
  if (DICT_.contains(#KEY_)) {                                                 \
    auto res = strToBool(DICT_.at(#KEY_));                                     \
    if (res) {                                                                 \
      KEY_ = res.get();                                                        \
    } else {                                                                   \
      auto err = res.takeError();                                              \
      LOG(FATAL) << "could not find bool value for key: " << #KEY_ << ". "     \
                 << ERR_TO_STRING(std::move(err)) << std::endl;                \
    }                                                                          \
  }

#define TRY_LOAD_INT_FROM_DICT(KEY_, DICT_)                                    \
  if (DICT_.contains(#KEY_)) {                                                 \
    KEY_ = std::stoi((DICT_.at(#KEY_)));                                       \
  }

#define TRY_LOAD_STR_FROM_DICT(KEY_, DICT_)                                    \
  if (DICT_.contains(#KEY_)) {                                                 \
    KEY_ = DICT_.at(#KEY_);                                                    \
  }

PyTorchLoaderSettings::PyTorchLoaderSettings(
    torch::Dict<std::string, std::string> dict) {
  initSettings();
  TRY_LOAD_BOOL_FROM_DICT(convertToFP16, dict);
  TRY_LOAD_BOOL_FROM_DICT(convertFusedToFP16, dict);
  TRY_LOAD_BOOL_FROM_DICT(clipFP16, dict);
  TRY_LOAD_BOOL_FROM_DICT(clipFP16SkipInputs, dict);
  TRY_LOAD_BOOL_FROM_DICT(convertPlaceholdersToFP16, dict);
  TRY_LOAD_BOOL_FROM_DICT(convertConstantsToFP16, dict);
  TRY_LOAD_BOOL_FROM_DICT(forceFP16AccumSLS, dict);
  TRY_LOAD_BOOL_FROM_DICT(saturateHost, dict);
  TRY_LOAD_BOOL_FROM_DICT(randomizeConstants, dict);
  TRY_LOAD_STR_FROM_DICT(backendOptionsFile, dict);
  TRY_LOAD_INT_FROM_DICT(replicationCount, dict);
  TRY_LOAD_BOOL_FROM_DICT(preCompilePyTorchModule, dict);
  TRY_LOAD_BOOL_FROM_DICT(fusionPassEnabled, dict);
  TRY_LOAD_BOOL_FROM_DICT(dumpGlowDag, dict);
  TRY_LOAD_INT_FROM_DICT(minFusionGroupSize, dict);
  TRY_LOAD_INT_FROM_DICT(maxFusionMergeSize, dict);
  TRY_LOAD_INT_FROM_DICT(fusionStartIndex, dict);
  TRY_LOAD_INT_FROM_DICT(fusionEndIndex, dict);
  TRY_LOAD_BOOL_FROM_DICT(dumpFinalGlowGraph, dict);
  TRY_LOAD_BOOL_FROM_DICT(enableGlowTracing, dict);
  TRY_LOAD_BOOL_FROM_DICT(enableRemoveMutation, dict);
  TRY_LOAD_INT_FROM_DICT(numTracesPerDump, dict);
  TRY_LOAD_BOOL_FROM_DICT(writeToOnnx, dict);
  TRY_LOAD_BOOL_FROM_DICT(onnxZipMode, dict);
  TRY_LOAD_BOOL_FROM_DICT(jitVsGlowCompare, dict);
  TRY_LOAD_BOOL_FROM_DICT(randomizeConstants, dict);
  TRY_LOAD_STR_FROM_DICT(backendName, dict);
  TRY_LOAD_INT_FROM_DICT(numDevices, dict);
  TRY_LOAD_BOOL_FROM_DICT(runShapeInference, dict);
  TRY_LOAD_BOOL_FROM_DICT(setIncludeLastOffsets, dict);
  TRY_LOAD_BOOL_FROM_DICT(inferShapeForCompilation, dict);
  TRY_LOAD_BOOL_FROM_DICT(enableDebugFuser, dict);
  if (dict.contains("opBlacklist")) {
    std::string commaSepOpsList = dict.at("opBlacklist");
    if (!commaSepOpsList.empty()) {
      auto kindStrings = splitString(commaSepOpsList);
      for (const auto &kindString : kindStrings) {
        opBlacklist.insert(torch::jit::Symbol::fromQualString(kindString));
      }
    }
  }
  if (dict.contains("backendSpecificOpts")) {
    std::string commaSepOptsList = dict.at("backendSpecificOpts");
    if (!commaSepOptsList.empty()) {
      auto optsStrings = splitString(commaSepOptsList);
      CHECK(optsStrings.size() % 2 == 0)
          << "Found " << optsStrings.size()
          << " elements. backendSpecificOpts must have equal number of keys "
             "and values.";
      for (auto it = optsStrings.begin(); it != optsStrings.end(); it++) {
        std::string key = *it;
        it++;
        std::string val = *it;
        backendSpecificOpts[key] = val;
      }
    }
  }
}

#define INSERT_BOOL_TO_DICT(KEY_, DICT_)                                       \
  DICT_.insert(#KEY_, KEY_ ? "true" : "false");

#define INSERT_STR_TO_DICT(KEY_, DICT_) DICT_.insert(#KEY_, KEY_);

#define INSERT_INT_TO_DICT(KEY_, DICT_)                                        \
  DICT_.insert(#KEY_, std::to_string(KEY_));

torch::Dict<std::string, std::string>
PyTorchLoaderSettings::serializeToDict() const {
  torch::Dict<std::string, std::string> dict;
  INSERT_BOOL_TO_DICT(convertToFP16, dict);
  INSERT_BOOL_TO_DICT(convertFusedToFP16, dict);
  INSERT_BOOL_TO_DICT(clipFP16, dict);
  INSERT_BOOL_TO_DICT(clipFP16SkipInputs, dict);
  INSERT_BOOL_TO_DICT(convertPlaceholdersToFP16, dict);
  INSERT_BOOL_TO_DICT(convertConstantsToFP16, dict);
  INSERT_BOOL_TO_DICT(forceFP16AccumSLS, dict);
  INSERT_BOOL_TO_DICT(saturateHost, dict);
  INSERT_BOOL_TO_DICT(randomizeConstants, dict);
  INSERT_STR_TO_DICT(backendOptionsFile, dict);
  INSERT_INT_TO_DICT(replicationCount, dict);
  INSERT_BOOL_TO_DICT(preCompilePyTorchModule, dict);
  INSERT_BOOL_TO_DICT(fusionPassEnabled, dict);
  INSERT_BOOL_TO_DICT(dumpGlowDag, dict);
  INSERT_INT_TO_DICT(minFusionGroupSize, dict);
  INSERT_INT_TO_DICT(maxFusionMergeSize, dict);
  INSERT_INT_TO_DICT(fusionStartIndex, dict);
  INSERT_BOOL_TO_DICT(enableRemoveMutation, dict);
  INSERT_INT_TO_DICT(fusionEndIndex, dict);
  INSERT_BOOL_TO_DICT(dumpFinalGlowGraph, dict);
  INSERT_BOOL_TO_DICT(enableGlowTracing, dict);
  INSERT_INT_TO_DICT(numTracesPerDump, dict);
  INSERT_BOOL_TO_DICT(writeToOnnx, dict);
  INSERT_BOOL_TO_DICT(onnxZipMode, dict);
  INSERT_BOOL_TO_DICT(jitVsGlowCompare, dict);
  INSERT_BOOL_TO_DICT(randomizeConstants, dict);
  INSERT_STR_TO_DICT(backendName, dict);
  INSERT_INT_TO_DICT(numDevices, dict);
  INSERT_BOOL_TO_DICT(runShapeInference, dict);
  INSERT_BOOL_TO_DICT(setIncludeLastOffsets, dict);
  INSERT_BOOL_TO_DICT(enableDebugFuser, dict);
  if (opBlacklist.size() > 0) {
    std::stringstream commaSepOpsList;
    for (const auto &op : opBlacklist) {
      commaSepOpsList << op.toQualString() << ",";
    }
    dict.insert("opBlacklist", commaSepOpsList.str());
  }
  if (backendSpecificOpts.size() > 0) {
    std::stringstream commaSepOptsList;
    for (const auto &opt : backendSpecificOpts) {
      commaSepOptsList << opt.first << "," << opt.second << ",";
    }
    dict.insert("backendSpecificOpts", commaSepOptsList.str());
  }
  return dict;
}

std::string PyTorchLoaderSettings::toString() const {
  auto dict = serializeToDict();
  std::stringstream s;
  for (const auto &item : dict) {
    s << item.key() << " : " << item.value() << std::endl;
  }
  return s.str();
}

const c10::Symbol &getGlowSymbol() {
  static c10::Symbol glowSymbol =
      at::Symbol::fromQualString("glow::FusionGroup");
  return glowSymbol;
}

c10::Symbol getGlowSymbol(std::shared_ptr<torch::jit::Graph> g) {
  std::string symbol = "glow::FusionGroup";
  if (g) {
    symbol += strFormat("_%lu", reinterpret_cast<uint64_t>(g.get()));
  }
  return at::Symbol::fromQualString(symbol);
}

glow::Type ptTypeToGlowType(const c10::TensorType &ptType) {
  DCHECK(ptType.scalarType().has_value())
      << "TensorType has no associated scalar type.";
  const auto concreteSizes = ptType.sizes().concrete_sizes().value();
  std::vector<glow::dim_t> dims;
  for (const auto &size : concreteSizes) {
    dims.push_back(static_cast<glow::dim_t>(size));
  }

  auto scalarType = ptType.scalarType().value();
  return glow::Type(scalarTypeToElemKind(scalarType), dims);
}

glow::Type ptTypeToGlowType(const c10::TensorType &ptType, float scale,
                            int32_t zero_point) {
  DCHECK(ptType.scalarType().has_value())
      << "TensorType has no associated scalar type.";
  const auto concreteSizes = ptType.sizes().concrete_sizes().value();
  std::vector<glow::dim_t> dims;
  for (const auto &size : concreteSizes) {
    dims.push_back(static_cast<glow::dim_t>(size));
  }

  auto scalarType = ptType.scalarType().value();
  return glow::Type(scalarTypeToElemKind(scalarType), dims, scale, zero_point);
}

at::Tensor convertQuantizedToDtype(at::Tensor ptTensor, c10::ScalarType dtype) {
  if (dtype != at::kQInt8 && dtype != at::kQUInt8) {
    LOG(DFATAL) << "Can only convert to int8 or uint8";
  }

  if (!ptTensor.is_quantized()) {
    LOG(DFATAL) << "Only support perform convert in quantized tensor.";
  }

  if (ptTensor.qscheme() != at::kPerTensorAffine) {
    LOG(DFATAL)
        << "Only support perform convert for per tensor quantized tensor.";
  }

  // dtype is ptTensor type, do nothing
  if (ptTensor.scalar_type() == dtype) {
    return ptTensor;
  }

  int offsetShift = 0;
  c10::ScalarType targetDQType;

  // We need to manually cast ptTensor to targetDQType, then make it quantized
  // tensor. In PyTorch, int8 is char and uint8 is byte.
  if (dtype == at::kQUInt8 && ptTensor.scalar_type() == at::kQInt8) {
    offsetShift = UINT8_TO_INT8_SHIFT;
    targetDQType = at::kByte;
  } else if (dtype == at::kQInt8 && ptTensor.scalar_type() == at::kQUInt8) {
    offsetShift = -UINT8_TO_INT8_SHIFT;
    targetDQType = at::kChar;
  } else {
    LOG(FATAL) << "Can not reach here.";
  }

  float scale = static_cast<float>(ptTensor.q_scale());
  int32_t offset = static_cast<int32_t>(ptTensor.q_zero_point());
  auto ptNewTensor = ptTensor.int_repr().to(targetDQType).add(offsetShift);
  auto ptNewQTensor = at::_make_per_tensor_quantized_tensor(
      ptNewTensor, scale, offset + offsetShift);
  return ptNewQTensor;
}

at::Tensor glowTypeToEmptyPTTensor(const glow::Type &glowType) {
  std::vector<int64_t> sizes;
  for (const auto dim : glowType.dims()) {
    sizes.push_back(dim);
  }
  if (glowType.isQuantizedType()) {
    auto scale = glowType.getScale();
    auto offset = glowType.getOffset();
    return at::_empty_affine_quantized(
        sizes,
        at::TensorOptions().dtype(
            elemKindToScalarType(glowType.getElementType())),
        scale, offset);
  } else {
    return at::empty(sizes, at::TensorOptions().dtype(elemKindToScalarType(
                                glowType.getElementType())));
  }
}

glow::Tensor ptTensorToGlowTensor(const at::Tensor &ptTensor) {
  CHECK(ptTensor.is_contiguous());
  if (ptTensor.is_quantized()) {
    float scale = 1.0;
    int32_t offset = 0;
    if (ptTensor.qscheme() == at::kPerChannelAffine) {
      // If it is channel wise quantized, which means
      // this tensor is the weight of quantized linear or conv
      // Then we dont deal with the qparams here,
      // and only set up soome dummy scale & offset by using the first
      // elements's scale & offset.
      scale = ptTensor.q_per_channel_scales()[0].item<float>();
      offset = ptTensor.q_per_channel_zero_points()[0].item<int32_t>();
    } else if (ptTensor.qscheme() == at::kPerTensorAffine) {
      scale = static_cast<float>(ptTensor.q_scale());
      offset = static_cast<int32_t>(ptTensor.q_zero_point());
    } else {
      LOG(DFATAL)
          << "PyTorch tensor with unsupported quantization scheme detected.";
    }
    auto glowType =
        ptTypeToGlowType(*c10::TensorType::create(ptTensor), scale, offset);
    return glow::Tensor(ptTensor.data_ptr(), &glowType);
  } else if (ptTensor.scalar_type() == at::kDouble) {
    at::Tensor atTensor = ptTensor.to(at::kFloat);
    auto glowType = ptTypeToGlowType(*c10::TensorType::create(atTensor));
    return glow::Tensor(atTensor.data_ptr(), &glowType).clone();
  } else {
    auto glowType = ptTypeToGlowType(*c10::TensorType::create(ptTensor));
    return glow::Tensor(ptTensor.data_ptr(), &glowType);
  }
}

std::vector<glow::InputMeta> loadInputMeta(const std::string &raw_data) {
  if (raw_data.empty()) {
    return {};
  }
  auto inputMeta = std::vector<glow::InputMeta>();
  std::stringstream ss_raw(raw_data);

  std::string line;
  while (std::getline(ss_raw, line)) {
    std::vector<glow::sdim_t> dims;
    std::stringstream ss(line);
    ss.ignore();
    for (int i; ss >> i;) {
      dims.push_back(i);
      if (ss.peek() == ',' || ss.peek() == '[' || ss.peek() == ']') {
        ss.ignore();
      }
    }
    std::getline(ss_raw, line);
    c10::ScalarType t = static_cast<c10::ScalarType>(std::stoi(line));
    inputMeta.emplace_back(t, std::move(dims));
  }
  return inputMeta;
}

// Similar to glowAOTFusion() however supports multiple Glow subgraphs and
// runners. We'd still need both since in some cases we may not be able to infer
// the entire model and would leverage glowAOTFusion() to run the partially
// lowered model.
void glowAOTFusionWithShapeInference(
    torch::jit::Module &model, const std::vector<glow::InputMeta> &inputMeta) {
  glow::PyTorchLoaderSettings &glowLoaderSettings =
      glow::getPyTorchLoaderSettings();
  glowLoaderSettings.preCompilePyTorchModule = true;

  auto graph = model.get_method("forward").function().graph();

  // fuse ListUnpack and Chunk into ConstantChunk. Put it here to work around
  // some JIT serialization/deserialization problem.
  torch::jit::CanonicalizeOps(graph);

  // create some fake inputs to run shape inference.
  // Usually users provide one set of inputs for the entire
  // model and expect the model can be lowered. However there
  // are cases where we cannot lower the entire model.
  // There could be multiple fused graphs and the inputs to
  // each fused graph could be different from the inputMeta user
  // provided. Therefore we leverage shape inference to populate
  // shape and type information over the entire model so we
  // could lower whatever we want.
  std::vector<torch::jit::IValue> inputs;
  for (const auto &i : inputMeta) {
    inputs.push_back(
        torch::empty(i.dims, torch::TensorOptions().dtype(i.type)));
  }

  const at::ArrayRef<torch::jit::IValue> inputRefs(inputs);

  // The base symbol of all.
  std::string baseSymbol = glow::getGlowSymbol(nullptr).toQualString();

  // There could be multiple glow fusion nodes created.
  glow::glowCustomFuse(graph);

  ShapeInferenceEngine shapeInf(*graph, inputRefs, baseSymbol);
  auto e = shapeInf.run();
  if (e) {
    LOG(ERROR) << ERR_TO_STRING(std::move(e));
  }

  const auto &shapeMap = shapeInf.getVariableMap();

  // this is a fuser subgraph to lower
  std::shared_ptr<torch::jit::Graph> subgraph;

  // Create one cachingGraphRunner for each fused graph.
  for (auto *node : graph->nodes()) {
    std::string kind = node->kind().toQualString();

    if (kind == baseSymbol) { // Found a match
      assert(node->hasAttribute(torch::jit::attr::Subgraph));
      subgraph = node->g(torch::jit::attr::Subgraph);
      // Find the index of this fusion node
      int idx = findIndex(node);

      // create the graph runner and warm its cache, this graph runner will be
      // picked up during operator registration
      // All Glow fusion nodes would have the same kind and there isn't a good
      // native way to differentiate them at runtime. Therefore we scan the
      // graph containing Glow fusion nodes and index each of them. The index
      // would be used as part of the key to find corresponding
      // cachingGraphRunner.
      auto runner =
          glow::setGraphRunnerForKey(kind + std::to_string(idx), [subgraph] {
            return std::make_unique<glow::CachingGraphRunner>(
                subgraph, glow::getHostManager(),
                glow::getPyTorchLoaderSettings());
          });

      std::vector<glow::InputMeta> perGraphInputMeta;
      auto graphInputValues = subgraph->inputs();

      for (size_t i = 0; i < graphInputValues.size(); ++i) {
        const torch::jit::Value *inputValue = graphInputValues[i];
        auto itr = shapeMap.find(inputValue);
        if (itr == shapeMap.end()) {
          LOG(ERROR) << "Node " << node->kind().toQualString() << " input " << i
                     << " Not found in the shape map!";
        }
        // Only support tensor input for now
        // TODO Add support for other input types, e.g., tensor[]
        perGraphInputMeta.emplace_back(itr->second.dtype,
                                       itr->second.shape<TensorShape>());
      }

      e = runner->warmCache(perGraphInputMeta, runner->getSettings(),
                            /*useMaxSizeCompilation*/ true);
      if (e) {
        // If the graph is already compiled previously, warmCache() will report
        // an error but it is fine with our execution. So here we extract the
        // error only.
        LOG(ERROR) << ERR_TO_STRING(std::move(e));
      }
    }
  }
  if (!subgraph) {
    // at least one
    LOG(ERROR) << "Cannot create a Glow fusion subgraph";
  }
}

void glowAOTFusion(torch::jit::Module &model, const std::string &inputMetaStr) {
  auto inputMeta = glow::loadInputMeta(inputMetaStr);

  if (FLAGS_inferShapeForCompilation) {
    return glowAOTFusionWithShapeInference(model, inputMeta);
  }

  glow::PyTorchLoaderSettings &glowLoaderSettings =
      glow::getPyTorchLoaderSettings();
  glowLoaderSettings.preCompilePyTorchModule = true;

  // We assume the model is flattened and only one graph will be lowered. In the
  // future we may need to support multiple graphs.
  auto graph = model.get_method("forward").function().graph();

  // fuse ListUnpack and Chunk into ConstantChunk. Put it here to work around
  // some JIT serialization/deserialization problem.
  torch::jit::CanonicalizeOps(graph);

  c10::Symbol symbol = glow::getGlowSymbol(graph);
  glow::registerGlowOp(symbol);
  glow::glowCustomFuse(graph, symbol);

  // this is the fuser subgraph to lower
  std::shared_ptr<torch::jit::Graph> subgraph;
  for (auto *node : graph->nodes()) {
    if (node->kind().toQualString() == symbol.toQualString()) {
      assert(node->hasAttribute(torch::jit::attr::Subgraph));
      subgraph = node->g(torch::jit::attr::Subgraph);
      break;
    }
  }
  if (!subgraph) {
    MAKE_ERR("Cannot create a Glow fusion subgraph");
  }

  // create the graph runner and warm its cache, this graph runner will be
  // picked up during operator registration
  auto runner = glow::setGraphRunnerForKey(symbol.toQualString(), [subgraph] {
    return std::make_unique<glow::CachingGraphRunner>(
        subgraph, glow::getHostManager(), glow::getPyTorchLoaderSettings());
  });

  auto e = runner->warmCache(inputMeta, runner->getSettings(),
                             /*useMaxSizeCompilation*/ true);
  if (e) {
    // If the graph is already compiled previously, warmCache() will report
    // an error but it is fine with our execution. So here we extract the
    // error only.
    LOG(ERROR) << ERR_TO_STRING(std::move(e));
  }
}

static bool &_signalHandlerOverridesEnabled() {
  static bool enabled = false;
  return enabled;
}

void enableSignalHandlerOverrides(bool enable) {
  _signalHandlerOverridesEnabled() = enable;
}

bool signalHandlerOverridesEnabled() {
  return _signalHandlerOverridesEnabled();
}

} // namespace glow
