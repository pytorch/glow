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

#ifndef GLOW_BACKENDS_NNPI_INFERENCECONTEXT_H
#define GLOW_BACKENDS_NNPI_INFERENCECONTEXT_H

#include "NNPICompiledFunction.h"
#include "NNPIResource.h"
#include "NNPITracing.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "nnpi_inference.h"
#include "nnpi_transformer.h"
#include <unordered_map>
#include <vector>

namespace glow {
namespace runtime {
class NNPIDeviceManager;

using StaticPlaceholderMap =
    std::unordered_map<const Placeholder *, std::weak_ptr<NNPIResource>>;

class InferenceContext {
private:
  NNPINetwork nnpiNetwork_;                 // For ice-ref path only.
  NNPICompilationConfig compilationConfig_; // For ice-ref path only.

  NNPIDeviceContext device_; // For queuing purposes (is created/destroyed in
                             // the DM ctor/dtor).
  NNPIInferCommand inferCmd_;
  NNPICommandList commandList_;
  std::vector<NNPICommandConfig> cmdConfigs_;
  std::vector<NNPICommandListError> cmdListErrors_;

  /// Set of inputs that can be partial tensors.
  const std::unordered_set<const Placeholder *> *partialInputs_;

  /// Set of inputs that are static tensors.
  std::unordered_set<const Placeholder *> staticInputs_;

  /// Device tracing handler.
  std::shared_ptr<NNPIDeviceTracing> deviceTracing_;

  /// NNPI Device configuration.
  std::shared_ptr<NNPIDeviceOptions> deviceOptions_;

  /// NNPI Device id.
  unsigned deviceId_;

  /// NNPI Resources.
  std::vector<std::shared_ptr<NNPIResource>> inputResources_;
  std::vector<std::shared_ptr<NNPIResource>> outputResources_;

  /// Vector of placeholders. The order of the non-static placeholders match
  /// with the inputResources_ and outputResources_.
  std::vector<Placeholder *> netInputPlaceholders_;
  std::vector<Placeholder *> netOutputPlaceholders_;

  // Name for the function that we are executing.
  std::string functionName_;

  /// Trace context names.
  std::string traceBackendExecuteContextName_;
  std::string tracePreProcessContextName_;
  std::string traceInferenceContextName_;
  std::string tracePostProcessContextName_;

public:
  InferenceContext();
  ~InferenceContext();
  void execute(RunIdentifierTy runId, std::unique_ptr<ExecutionContext> ctx,
               runtime::ResultCBTy resultCB);
  bool init(
      // For ICE-Ref path.
      NNPINetwork network, NNPICompilationConfig config,
      // For ICE-T path.
      NNPIHostNetwork hostNetwork, NNPIDeviceNetwork deviceNetwork,
      NNPIAdapter adapter, NNPIDeviceContext device,
      const std::unordered_set<const Placeholder *> &partialInputs,
      const std::unordered_set<const Placeholder *> &staticInputs,
      std::shared_ptr<NNPIDeviceTracing> deviceTracing,
      StaticPlaceholderMap *staticPlaceholderMap,
      std::shared_ptr<NNPIDeviceOptions> deviceOptions,
      const std::string &functionName, unsigned deviceId);
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_NNPI_INFERENCECONTEXT_H
