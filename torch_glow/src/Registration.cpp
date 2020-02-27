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

#include "CachingGraphRunner.h"
#include "GlowFuser.h"
#include "PyTorchCommon.h"
#include "glow/Support/Error.h"

#include <glog/logging.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace glow {
std::unordered_map<std::string, std::unique_ptr<CachingGraphRunner>> &
getPreloadedRunnerMap() {
  static std::unordered_map<std::string, std::unique_ptr<CachingGraphRunner>>
      preloadedGraphRunners_;
  return preloadedGraphRunners_;
}

std::unique_ptr<CachingGraphRunner>
getGraphRunnerForKey(const std::string &key) {
  auto &preloadedRunners = getPreloadedRunnerMap();
  auto it = preloadedRunners.find(key);
  if (it == preloadedRunners.end()) {
    return nullptr;
  } else {
    auto graphRunner = std::move(it->second);
    preloadedRunners.erase(it);
    return graphRunner;
  }
}

void setGraphRunnerForKey(const std::string &key,
                          std::unique_ptr<CachingGraphRunner> graphRunner) {
  auto &preloadedRunners = getPreloadedRunnerMap();
  DCHECK_EQ(preloadedRunners.count(key), 0);
  preloadedRunners[key] = std::move(graphRunner);
}

void registerGlowOp(const c10::Symbol &symbol) {
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(at::AliasAnalysisKind::PURE_FUNCTION);

  torch::jit::RegisterOperators op({torch::jit::Operator(
      symbol,
      [](const torch::jit::Node *node) -> torch::jit::Operation {
        std::string key = node->kind().toQualString();

        std::shared_ptr<CachingGraphRunner> graphRunner =
            getGraphRunnerForKey(key);

        // If no preloaded graph runner was created for this node, create a new
        // empty one.
        if (!graphRunner) {
          graphRunner = std::make_shared<CachingGraphRunner>(
              node->g(at::attr::Subgraph), getHostManager(),
              getPyTorchLoaderSettings());
        }

        return [graphRunner](torch::jit::Stack &stack) {
          Error err = Error::empty();
          if (graphRunner->getSettings().preCompilePyTorchModule) {
            err = graphRunner->runOnly(stack);
          } else {
            err = graphRunner->run(stack);
          }

          if (static_cast<bool>(err)) {
            // PyTorch framework expects an exception been thrown here.
            throw std::invalid_argument(ERR_TO_STRING(std::move(err)));
          }
          return 0;
        };
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
} // namespace glow
