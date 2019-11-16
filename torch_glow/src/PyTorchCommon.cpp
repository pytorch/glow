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

#include "CachingGraphRunner.h"
#include "GlowFuser.h"
#include "PyTorchModelLoader.h"

#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace glow {

bool GlowCompilePyTorchModule = false;

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
    setHostManager("Interpreter");
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
    deviceConfigs.push_back(
        llvm::make_unique<runtime::DeviceConfig>(backendName));
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
  case ElemKind::UInt8FusedQTy:
  case ElemKind::UInt8FusedFP16QTy:
  case ElemKind::UInt4FusedFP16QTy:
  case ElemKind::Int8QTy:
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

PyTorchLoaderSettings &getPyTorchLoaderSettings() {
  static PyTorchLoaderSettings settings;
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
  std::vector<size_t> dims;
  for (const auto &size : concreteSizes) {
    dims.push_back(static_cast<size_t>(size));
  }

  auto scalarType = ptType.scalarType().value();
  return glow::Type(scalarTypeToElemKind(scalarType), dims);
}

at::Tensor glowTypeToEmptyPTTensor(const glow::Type &glowType) {
  std::vector<int64_t> sizes;
  for (const auto dim : glowType.dims()) {
    sizes.push_back(dim);
  }

  return at::empty(sizes, at::TensorOptions().dtype(
                              elemKindToScalarType(glowType.getElementType())));
}

glow::Tensor ptTensorToGlowTensor(const at::Tensor &ptTensor) {
  auto glowType = ptTypeToGlowType(*c10::TensorType::create(ptTensor));
  return glow::Tensor(ptTensor.data_ptr(), &glowType);
}
} // namespace glow
