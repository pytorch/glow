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

#include "NNPICompiledFunction.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/ThreadPool.h"
#include "nnpi_inference.h"
#include "nnpi_transformer.h"
#include <atomic>
#include <map>
#include <unordered_map>
#include <vector>

namespace glow {
namespace runtime {

class InferenceThreadEnv {
  NNPINetwork nnpiNetwork_;                 // For ice-ref path only.
  NNPICompilationConfig compilationConfig_; // For ice-ref path only.

  NNPIDeviceContext device_; // For queuing purposes (is created/destroyed in
                             // the DM ctor/dtor).
  NNPIInferCommand inferCmd_;

  std::vector<std::pair<std::string, NNPITensorDesc>> netInputs_;
  std::vector<std::pair<std::string, NNPITensorDesc>> netOutputs_;
  std::map<std::string, Tensor *> ioTensors_;
  struct NamedResource {
    NNPIObjectName name;
    NNPIResourceDesc desc;
    NNPIHandle handle;
  };
  std::vector<NamedResource> hostInputs_, hostOutputs_, deviceInputs_,
      deviceOutputs_;
  std::vector<NNPICopyCommand> inputCopyCmds_, outputCopyCmds_;
  std::vector<void *> rawInputs_, rawOutputs_;
  std::set<int32_t *> tmpBuffers_; // Used for int64 tensors.
public:
  InferenceThreadEnv();
  ~InferenceThreadEnv();
  bool execute(RunIdentifierTy runId, std::unique_ptr<ExecutionContext> ctx,
               runtime::ResultCBTy resultCB);
  bool init(
      // For ICE-Ref path.
      NNPINetwork network, NNPICompilationConfig config,
      // For ICE-T path.
      NNPIHostNetwork hostNetwork, NNPIDeviceNetwork deviceNetwork,
      NNPIAdapter adapter, NNPIDeviceContext device);
};

class InferencePoolEnv {
  unsigned numWorkers_;
  std::atomic<unsigned> workerIndex_;
  std::unique_ptr<ThreadPool> workersPool_;
  std::vector<InferenceThreadEnv> threadEnvs_;
  NNPIHostNetwork hostNetwork_;
  NNPIDeviceNetwork deviceNetwork_;

public:
  InferencePoolEnv();
  ~InferencePoolEnv();
  Error init(unsigned numWorkers, NNPIAdapter adapter, NNPIDeviceContext device,
             CompiledFunction *compiledFunction);
  void stop(bool block);
  void execute(RunIdentifierTy runId, std::unique_ptr<ExecutionContext> ctx,
               runtime::ResultCBTy resultCB);
};

using InferencePoolMap = std::unordered_map<std::string, InferencePoolEnv>;

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_NNPI_INFERENCEPOOL_H
