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

#ifndef GLOW_NNPI_FUNCTION_H
#define GLOW_NNPI_FUNCTION_H

#include "BlockStream.h"
#include "NNPIOptions.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Backends/BackendOptions.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "nnpi_transformer.h"
#include <map>
#include <memory>
#include <mutex>

namespace glow {

/// Function "compiled" for execution by the NNPI backend.
class NNPICompiledFunction final : public CompiledFunction {
public:
  NNPICompiledFunction(Function *F)
      : CompiledFunction(runtime::RuntimeBundle::create(*F)),
        compilationOptions_({}){};

  /// \name CompiledFunction interface.
  ///@{
  ~NNPICompiledFunction() override;

  /// Execute the network and allocate Placeholder memory with given
  /// \p ctx providing mapping between Placeholder and populated tensor.
  virtual Error execute(ExecutionContext *ctx) override {
    return MAKE_ERR(
        "Don't execute NNPI functions directly, use the device manager!");
  }

  /// \returns the Kind of Backend used to compile this function.
  virtual std::string getCompileBackendName() const override { return "NNPI"; }

  /// \returns the compiled network handle.
  NNPINetwork getCompiledNetworkHandle() const { return network_; }

  /// \returns the compilation config object.
  NNPICompilationConfig getCompilationConfig() const { return config_; }

  /// \returns a reference to the set of Placeholders supporting partial inputs.
  const std::unordered_set<const Placeholder *> &getPartialInputs() const {
    return partialInputs_;
  }

  /// \returns a reference to the set of static Placeholders.
  const std::unordered_set<const Placeholder *> &getStaticInputs() const {
    return staticInputs_;
  }

  /// Locks the output stream.
  BlockStream &lockCompiledStream();
  /// Unlocks the output stream.
  void unlockCompiledStream();
  virtual void freeCompilationResources() override;

  virtual Error compile(Function *F, const BackendOptions &opts);

  NNPICompilationOptions getCompilationOptions() const {
    return compilationOptions_;
  }

  const std::string &getCompilationFilename() const {
    return compilationFileName_;
  }

private:
  NNPINetwork network_;
  NNPICompilationConfig config_;
  BlockStream compiledStream_;
  std::mutex compiledStreamMutex_;
  std::unordered_set<const Placeholder *> partialInputs_;
  std::unordered_set<const Placeholder *> staticInputs_;
  NNPICompilationOptions compilationOptions_;
  std::string compilationFileName_;

  Error updateCompilationConfigFromOptions(
      NNPICompilationOptions &compilationOptions);

  /// Setup compilation hints for \p F given \p backendSpecificNodeInfo.
  /// \returns an error if some validation issue is found given expected format.
  Error
  setupCompilationHints(const Function *F,
                        const BackendSpecificNodeInfo &backendSpecificNodeInfo);
  ///@}
};
} // end namespace glow
#endif // GLOW_NNPI_FUNCTION_H
