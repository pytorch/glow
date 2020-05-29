/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#ifndef GLOW_BACKENDS_NNPI_INFERENCEPOOL_H
#define GLOW_BACKENDS_NNPI_INFERENCEPOOL_H

#include "InferenceContext.h"
#include "NNPICompiledFunction.h"
#include "NNPITracing.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "nnpi_inference.h"
#include "nnpi_transformer.h"
#include <atomic>
#include <map>
#include <unordered_map>
#include <vector>

namespace glow {
namespace runtime {
class NNPIDeviceBindings;
class InferencePoolEnv {
  std::unique_ptr<folly::CPUThreadPoolExecutor> workersPool_;
  std::vector<InferenceContext> inferenceContexts_;
  std::vector<InferenceContext *> freeContexts_;
  std::mutex freeContextsLock_;
  NNPIDeviceNetwork deviceNetwork_;
  std::shared_ptr<NNPIDeviceOptions> deviceOptions_;
  unsigned deviceId_;
  ResourceDescVec inputDesc_;
  ResourceDescVec outputDesc_;
  NNPIAdapter adapter_;
  NNPIDeviceContext device_;
  NNPICompiledFunction *nnpiCompiledFunction_;
  StaticPlaceholderMap *staticPlaceholderMap_;
  std::string functionName_;

public:
  InferencePoolEnv();
  ~InferencePoolEnv();
  Error init(NNPIAdapter adapter, NNPIDeviceContext device,
             CompiledFunction *compiledFunction,
             StaticPlaceholderMap *staticPlaceholderMap,
             std::shared_ptr<NNPIDeviceOptions> deviceOptions,
             const std::string &functionName, unsigned deviceId);
  InferenceContext *
  createDetachedInferenceContext(PlaceholderUsageMap &phUsage);
  void stop(bool block);
  void execute(RunIdentifierTy runId, std::unique_ptr<ExecutionContext> ctx,
               runtime::ResultCBTy resultCB);
};

using InferencePoolMap = std::unordered_map<std::string, InferencePoolEnv>;

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_NNPI_INFERENCEPOOL_H
