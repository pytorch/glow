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

#include <c10/util/hash.h>
#include <glog/logging.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include <mutex>
#include <shared_mutex>
#include <signal.h>

namespace glow {

namespace {
/// Lock to protect the global graph runner map.
std::shared_timed_mutex runnerMapMutex;

std::unordered_map<std::string, std::unique_ptr<CachingGraphRunner>> &
getPreloadedRunnerMap() {
  static std::unordered_map<std::string, std::unique_ptr<CachingGraphRunner>>
      preloadedGraphRunners_;
  return preloadedGraphRunners_;
}
} // namespace

std::unique_ptr<CachingGraphRunner>
getGraphRunnerForKey(const std::string &key) {
  auto &preloadedRunners = getPreloadedRunnerMap();
  std::unique_lock<std::shared_timed_mutex> wlock(runnerMapMutex);
  auto it = preloadedRunners.find(key);
  if (it == preloadedRunners.end()) {
    return nullptr;
  } else {
    auto res = std::move(it->second);
    preloadedRunners.erase(it);
    return res;
  }
}

CachingGraphRunner *
setGraphRunnerForKey(const std::string &key,
                     std::function<std::unique_ptr<CachingGraphRunner>(void)>
                         graphRunnerBuilder) {
  auto &preloadedRunners = getPreloadedRunnerMap();
  std::unique_lock<std::shared_timed_mutex> wlock(runnerMapMutex);
  const auto it = preloadedRunners.find(key);
  if (it != preloadedRunners.end()) {
    return it->second.get();
  }

  auto runner = graphRunnerBuilder();
  auto *runnerRaw = runner.get();
  auto ret = preloadedRunners.emplace(key, std::move(runner));
  CHECK(ret.second);
  return runnerRaw;
}

bool removeGraphRunnerForKey(const std::string &key) {
  auto &preloadedRunners = getPreloadedRunnerMap();
  std::unique_lock<std::shared_timed_mutex> wlock(runnerMapMutex);
  const auto it = preloadedRunners.find(key);
  if (it == preloadedRunners.end()) {
    return false;
  }
  preloadedRunners.erase(key);
  return true;
}

int findIndex(const torch::jit::Node *node) {
  auto g = node->owningGraph();
  auto kind = node->kind();
  int index = 0;
  for (auto n : g->nodes()) {
    if (n->kind() == kind) {
      if (n == node) {
        return index;
      }
      ++index;
    }
  }
  CHECK(0); // should never reach this line
  return index;
}

void registerGlowOp(const c10::Symbol &symbol) {
  torch::jit::RegisterOperators op({torch::jit::Operator(
      symbol,
      [](const torch::jit::Node *node) -> torch::jit::Operation {
        std::string key = node->kind().toQualString();
        if (getPyTorchLoaderSettings().inferShapeForCompilation) {
          // All Glow fusion nodes would have the same kind and there isn't a
          // good native way to differentiate them at runtime. Therefore we scan
          // the graph containing Glow fusion nodes and index each of them. The
          // index would be used as part of the key to find corresponding
          // cachingGraphRunner.
          int idx = findIndex(node);
          key += std::to_string(idx);
        }

        // How to find a graphRunner:
        // 1. See if a key based on fusion node symbol string has been
        // registered, which is usually done in AOT fashion
        // 2. If not, create a graphRunner with graph hash as a key
        std::shared_ptr<CachingGraphRunner> graphRunner;
        if (!graphRunner) {
          graphRunner = getGraphRunnerForKey(key);
        }

        // If no preloaded graph runner was created for this node, create a new
        // empty one.
        if (!graphRunner) {
          graphRunner = std::make_unique<CachingGraphRunner>(
              node->g(at::attr::Subgraph), getHostManager(),
              getPyTorchLoaderSettings());
        }

        return [graphRunner =
                    std::move(graphRunner)](torch::jit::Stack *stack) {
          Error err = Error::empty();
          // Store old Python signal handlers and install standard signal
          // handlers, so that it is possible to kill/interrupt the process if
          // needed.
          typedef void (*sighandler_t)(int);
          sighandler_t oldSigIntHandler = nullptr;
          sighandler_t oldSigTermHandler = nullptr;

          if (signalHandlerOverridesEnabled()) {
            oldSigIntHandler = signal(SIGINT, SIG_DFL);
            oldSigTermHandler = signal(SIGTERM, SIG_DFL);
          }

          if (graphRunner->getSettings().preCompilePyTorchModule) {
            err = graphRunner->runOnly(*stack);
          } else {
            err = graphRunner->run(*stack);
          }

          // Restore old signal handlers.
          if (oldSigIntHandler != nullptr && oldSigIntHandler != SIG_ERR &&
              oldSigIntHandler != SIG_DFL) {
            signal(SIGINT, oldSigIntHandler);
          }
          if (oldSigTermHandler != nullptr && oldSigTermHandler != SIG_ERR &&
              oldSigTermHandler != SIG_DFL) {
            signal(SIGTERM, oldSigTermHandler);
          }

          if (static_cast<bool>(err)) {
            // PyTorch framework expects an exception been thrown here.
            throw std::invalid_argument(ERR_TO_STRING(std::move(err)));
          }
        };
      },
      at::AliasAnalysisKind::PURE_FUNCTION)});
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
