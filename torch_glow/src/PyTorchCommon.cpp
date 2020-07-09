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

#include "torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h"
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator_options.h>

DEFINE_string(torch_glow_backend, "Interpreter",
              "Glow backend used for torchifi");
DEFINE_int32(torch_glow_num_devices, -1, "Number of devices for Glow backend");
DEFINE_int32(torch_glow_min_fusion_group_size, 1,
             "Minimum number of nodes in the glow fusion group");
DEFINE_bool(dumpGlowDag, false, "See PyTorchLoaderSettings");
DEFINE_bool(jitVsGlowCompare, false, "Enable per-group error check");
DEFINE_bool(dumpFinalGlowGraph, false, "See PyTorchLoaderSettings");
DEFINE_bool(enableGlowTracing, false, "See PyTorchLoaderSettings");
DEFINE_int32(numTracesPerDump, 1, "See PyTorchLoaderSettings");
DEFINE_bool(saturateHost, false, "See PyTorchLoaderSettings");
DEFINE_bool(convertToFP16, false, "See PyTorchLoaderSettings");
DEFINE_bool(convertFusedToFP16, false, "See PyTorchLoaderSettings");
DEFINE_string(opBlacklist, "", "See PyTorchLoaderSettings");
DEFINE_int32(replicationCount, 1, "Number of replications on each device");
DEFINE_bool(writeToOnnx, false, "See PyTorchLoaderSettings");
DEFINE_int32(maxActiveRequests, 250,
             "Max number of active requests before HostManager starts queuing");
DEFINE_bool(randomizeConstants, false, "See PyTorchLoaderSettings");
DEFINE_bool(runShapeInference, false, "See PyTorchLoaderSettings");

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

static PyTorchLoaderSettings getInitialSettings() {
  PyTorchLoaderSettings settings;
  settings.minFusionGroupSize = FLAGS_torch_glow_min_fusion_group_size;
  settings.dumpGlowDag = FLAGS_dumpGlowDag;
  settings.jitVsGlowCompare = FLAGS_jitVsGlowCompare;
  settings.dumpFinalGlowGraph = FLAGS_dumpFinalGlowGraph;
  settings.enableGlowTracing = FLAGS_enableGlowTracing;
  settings.numTracesPerDump = FLAGS_numTracesPerDump;
  settings.saturateHost = FLAGS_saturateHost;
  settings.convertToFP16 = FLAGS_convertToFP16;
  settings.convertFusedToFP16 = FLAGS_convertFusedToFP16;
  settings.replicationCount = FLAGS_replicationCount;
  settings.writeToOnnx = FLAGS_writeToOnnx;
  settings.randomizeConstants = FLAGS_randomizeConstants;
  settings.backendName = FLAGS_torch_glow_backend;
  settings.numDevices = FLAGS_torch_glow_num_devices;
  settings.runShapeInference = FLAGS_runShapeInference;

  if (!FLAGS_opBlacklist.empty()) {
    auto kindStrings = splitString(FLAGS_opBlacklist);
    for (const auto &kindString : kindStrings) {
      settings.opBlacklist.insert(
          torch::jit::Symbol::fromQualString(kindString));
    }
  }

  return settings;
}

PyTorchLoaderSettings &getPyTorchLoaderSettings() {
  static PyTorchLoaderSettings settings = getInitialSettings();
  return settings;
}

c10::Symbol getGlowSymbol(std::shared_ptr<torch::jit::Graph> g) {
  if (g) {
    return at::Symbol::fromQualString(strFormat(
        "glow::FusionGroup_%lu", reinterpret_cast<uint64_t>(g.get())));
  } else {
    return at::Symbol::fromQualString("glow::FusionGroup");
  }
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

std::shared_ptr<std::vector<glow::InputMeta>>
loadInputMeta(const std::string &raw_data) {
  if (raw_data.empty()) {
    return nullptr;
  }
  auto inputMeta = std::make_shared<std::vector<glow::InputMeta>>();
  std::stringstream ss_raw(raw_data);

  std::string line;
  while (std::getline(ss_raw, line)) {
    std::vector<size_t> dims;
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
    inputMeta->emplace_back(t, std::move(dims));
  }
  return inputMeta;
}

void glowAOTFusion(torch::jit::Module &model, const std::string &inputMetaStr) {
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

  // create the graph runner and warm its cache, this graph runner will be picked
  // up during operator registration
  auto runner = glow::setGraphRunnerForKey(symbol.toQualString(), [subgraph] {
    return std::make_unique<glow::CachingGraphRunner>(
        subgraph, glow::getHostManager(), glow::getPyTorchLoaderSettings());
  });

  auto inputMeta = glow::loadInputMeta(inputMetaStr);

  auto e = runner->warmCache(*inputMeta);
  if (e) {
    // If the graph is already compiled previously, warmCache() will report
    // an error but it is fine with our execution. So here we extract the
    // error only.
    ERR_TO_STRING(std::move(e));
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
