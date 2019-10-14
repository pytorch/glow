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
#include "FusePrepack.h"
#include "GlowFuser.h"
#include "PyTorchModelLoader.h"

#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace glow {

bool GlowCompilePyTorchModule = false;

namespace {
/// Builds and \returns a HostManager instance.
std::unique_ptr<runtime::HostManager> buildHostManager() {
  constexpr size_t numGlowDevices = 1;

  std::vector<std::unique_ptr<runtime::DeviceConfig>> deviceConfigs;
  for (int i = 0; i < numGlowDevices; i++) {
    deviceConfigs.push_back(llvm::make_unique<runtime::DeviceConfig>(
        getPyTorchLoaderSettings().glowBackendName));
  }

  return llvm::make_unique<runtime::HostManager>(std::move(deviceConfigs));
}

} // namespace

/// \returns the HostManager singleton used to run all PyTorch graphs in Glow.
runtime::HostManager *getHostManager() {
  static std::unique_ptr<runtime::HostManager> hostManager = buildHostManager();
  return hostManager.get();
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
    LOG(DFATAL) << "Not supported yet.";
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

void glowCustomFuse(std::shared_ptr<torch::jit::Graph> &g,
                    at::Symbol fuseSymbol) {
  // Fuse all linear operators
  // Currently PyTorch does not have good support for aten:addmm when fusing
  // Therefore we use some pattern to translate all aten::addmm to
  // aten::linear before we fuse the whole graph.
  FuseKnownPatterns(g);

  GlowCustomFuse(g, PyTorchModelLoader::isNodeSupported, fuseSymbol);
}

void registerGlowOp(const c10::Symbol &symbol) {
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(at::AliasAnalysisKind::PURE_FUNCTION);

  torch::jit::RegisterOperators op({torch::jit::Operator(
      symbol,
      [](const torch::jit::Node *node) -> torch::jit::Operation {
        if (GlowCompilePyTorchModule) {
          std::string key = node->kind().toQualString();
          auto graphRunner = glow::CachingGraphRunner::getCachingGraphRunner();

          return [graphRunner, key](torch::jit::Stack &stack) {
            Error err = graphRunner->run(key, stack);

            if (static_cast<bool>(err)) {
              // PyTorch framework expects an exception been thrown here.
              throw std::invalid_argument(ERR_TO_STRING(std::move(err)));
            }
            return 0;
          };
        } else {
          std::shared_ptr<torch::jit::Graph> graph =
              node->g(at::attr::Subgraph);
          auto graphRunner = std::make_shared<CachingGraphRunner>(
              graph.get(), getHostManager());

          return [graphRunner](torch::jit::Stack &stack) {
            Error err = graphRunner->run(stack);

            if (static_cast<bool>(err)) {
              // PyTorch framework expects an exception been thrown here.
              throw std::invalid_argument(ERR_TO_STRING(std::move(err)));
            }
            return 0;
          };
        }
      },
      options)});
}

void registerGlowFusionPass(std::function<bool()> enablePassFn) {
  torch::jit::RegisterPass pass([enablePassFn = std::move(enablePassFn)](
                                    std::shared_ptr<torch::jit::Graph> &g) {
    if (enablePassFn()) {
      glow::glowCustomFuse(g, getGlowSymbol());
    }
  });
}

void registerGlowFusionOpAndPass(std::function<bool()> enablePassFn) {
  registerGlowOp(getGlowSymbol());
  registerGlowFusionPass(std::move(enablePassFn));
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

void FuseKnownPatterns(std::shared_ptr<torch::jit::Graph> &graph) {
  FuseConvPrepack(graph);

  std::string originalPat = R"IR(
graph(%input):
  %res1 = prim::NumToTensor(%input)
  %res2 = aten::Int(%res1)
  return (%res2))IR";

  std::string replacementPat = R"IR(
graph(%input):
  return (%input))IR";

  torch::jit::SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(originalPat, replacementPat);
  rewriter.runOnGraph(graph);
}

} // namespace glow
