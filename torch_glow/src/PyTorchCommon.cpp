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

#include "PyTorchCommon.h"

#include "GlowFuser.h"
#include "PyTorchModelLoader.h"

#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator_options.h>

DEFINE_string(torch_glow_backend, "Interpreter",
              "Glow backend used for torchifi");
DEFINE_int32(torch_glow_num_devices, 1, "Number of devices for Glow backend");
DEFINE_int32(torch_glow_min_fusion_group_size, 1,
             "Number of devices for Glow backend");
DEFINE_bool(dumpGlowDag, false, "See PyTorchLoaderSettings");
DEFINE_bool(dumpFinalGlowGraph, false, "See PyTorchLoaderSettings");
DEFINE_bool(enableGlowTracing, false, "See PyTorchLoaderSettings");
DEFINE_int32(numTracesPerDump, 1, "See PyTorchLoaderSettings");
DEFINE_bool(saturateHost, false, "See PyTorchLoaderSettings");
DEFINE_bool(convertToFP16, false, "See PyTorchLoaderSettings");
DEFINE_string(opBlacklist, "", "See PyTorchLoaderSettings");
DEFINE_int32(replicationCount, 1, "Number of replications on each device");

namespace glow {

namespace {

static int setGraphExecutorToLegacy() {
  // use legacy GraphExecutor for Glow
  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  return 0;
}

static const int USE_LEGACY_GE = setGraphExecutorToLegacy();

/// GlowBackendState stores the currently active Glow HostManager that will
/// be used to run the subgraphs lowered to Glow. It also contains information
/// about the number and type of backend devices owned by the HostManager.
struct GlowBackendState {
  std::shared_ptr<runtime::HostManager> hostManager;
  std::string backendName;
  size_t numDevices = 0;
};

/// Meyers singleton for GlowBackendState.
GlowBackendState *getGlowBackendState() {
  static GlowBackendState state_;
  return &state_;
}

} // namespace

std::shared_ptr<runtime::HostManager> getHostManager() {
  auto hostManager = getGlowBackendState()->hostManager;
  // If no HostManager has been set, use Glow's Interpreter.
  if (!hostManager) {
    setHostManager(FLAGS_torch_glow_backend, FLAGS_torch_glow_num_devices);
    hostManager = getGlowBackendState()->hostManager;
  }
  return hostManager;
}

const std::string &getBackendName() {
  return getGlowBackendState()->backendName;
}

size_t getBackendNumDevices() { return getGlowBackendState()->numDevices; }

void setHostManager(const std::string &backendName, size_t numDevices) {
  auto *state = getGlowBackendState();

  // Don't create a new identical HostManager.
  if (state->backendName == backendName && state->numDevices == numDevices) {
    return;
  }

  state->backendName = backendName;
  state->numDevices = numDevices;

  std::vector<std::unique_ptr<runtime::DeviceConfig>> deviceConfigs;
  for (int i = 0; i < numDevices; i++) {
    auto config = llvm::make_unique<runtime::DeviceConfig>(backendName);
    config->deviceID = i;
    deviceConfigs.push_back(std::move(config));
  }

  state->hostManager =
      std::make_shared<runtime::HostManager>(std::move(deviceConfigs));
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
  settings.dumpFinalGlowGraph = FLAGS_dumpFinalGlowGraph;
  settings.enableGlowTracing = FLAGS_enableGlowTracing;
  settings.numTracesPerDump = FLAGS_numTracesPerDump;
  settings.saturateHost = FLAGS_saturateHost;
  settings.convertToFP16 = FLAGS_convertToFP16;
  settings.replicationCount = FLAGS_replicationCount;

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

const c10::Symbol &getGlowSymbol() {
  static c10::Symbol glowSymbol =
      at::Symbol::fromQualString("glow::FusionGroup");
  return glowSymbol;
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
    offsetShift = OFFSETSHIFT;
    targetDQType = at::kByte;
  } else if (dtype == at::kQInt8 && ptTensor.scalar_type() == at::kQUInt8) {
    offsetShift = -OFFSETSHIFT;
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

} // namespace glow
