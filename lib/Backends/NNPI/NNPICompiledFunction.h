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
#include "glow/Backend/BlockStreamBase.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Backends/BackendOptions.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "nnpi_inference_types.h"
#include "nnpi_transformer.h"
#include <folly/dynamic.h>
#include <map>
#include <memory>
#include <mutex>

namespace glow {

/// Update device network config from the compilation config
NNPIDeviceNetworkConfig parseDeviceNetworkConfig(
    const glow::NNPICompilationOptions &compilationOptions);

/// Struct containing details exported for a compiled tensor.
struct NNPICompiledTensor {
  std::string name;
  std::string type;
  std::vector<uint32_t> shape;
  NNPI_ALLOCATION_TYPE allocType;
  std::string dump() const;
  std::vector<NNPI_ALLOCATION_TYPE> possibleAlloc;
};

/// Struct containing details exported for a compiled operator.
struct NNPICompiledOp {
  std::string name;
  std::string type;
  NNPI_EXECUTION_TYPE execType;
  int32_t coreIndex;
  int32_t iceBo;
  std::vector<NNPICompiledTensor> inputs;
  std::vector<NNPICompiledTensor> outputs;
  std::string dump() const;
};

/// Collection of exported details for compiled functions.
struct NNPICompilationInfo {
  std::map<std::string, NNPICompiledOp> ops;
  std::vector<std::pair<std::string, std::string>> opDependencies;
  std::string dump(const std::string &functionName) const;
  void clear() {
    ops.clear();
    opDependencies.clear();
  }
};

/// Function "compiled" for execution by the NNPI backend.
class NNPICompiledFunction final : public CompiledFunction {
public:
  /// \name CompiledFunction interface.
  ///@{
  NNPICompiledFunction(Function *F);

#if FACEBOOK_INTERNAL
  NNPICompiledFunction(const folly::dynamic &FXIR, const std::string &submod,
                       const llvm::StringMap<const void *> &constants,
                       Module *glowModule);
#endif

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

  /// \returns a reference to the set of Placeholders that needed to be padded.
  /// This set is introduced because we need to pad some tensors with its last
  /// element instead of zeros. In the future, we may introduce a flag to
  /// determine other padding schemes.
  const std::unordered_set<const Placeholder *> &getPaddedInputs() const {
    return paddedInputs_;
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

#if FACEBOOK_INTERNAL
  Error compileFX(const folly::dynamic &FXIR, const std::string &submod,
                  const llvm::StringMap<const void *> &constants,
                  const BackendOptions &opts, Module *glowModule);
#endif

  NNPICompilationOptions getCompilationOptions() const {
    return compilationOptions_;
  }

  const std::string &getCompilationFilename() const {
    return compilationFileName_;
  }

  const std::vector<std::string> &getInputNames() const { return inputNames_; }

  const std::vector<std::string> &getOutputNames() const {
    return outputNames_;
  }

  NNPIDeviceNetworkConfig getDeviceNetworkConfig() const {
    return devNetConfig_;
  }

  const std::vector<std::string> &getIAExtensionPaths() const {
    return iaExtensionPaths_;
  }

  const std::vector<std::pair<std::string, std::vector<char>>> &
  getIAExtensionLibs() const {
    return iaExtensionLibs_;
  }

  const NNPICompilationInfo &getCompilationInfo() const {
    return compilationInfo_;
  }

  const std::string toJSON() const override;

  std::unique_ptr<BlockStreamBase> serialize() override;

  Error deserialize(const std::vector<char> &serializedData) override;

private:
  NNPINetwork network_;
  NNPICompilationConfig config_;
  BlockStream compiledStream_;
  std::mutex compiledStreamMutex_;
  std::unordered_set<const Placeholder *> partialInputs_;
  std::unordered_set<const Placeholder *> paddedInputs_;
  std::unordered_set<const Placeholder *> staticInputs_;
  NNPICompilationOptions compilationOptions_;
  std::string compilationFileName_;
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;
  NNPIDeviceNetworkConfig devNetConfig_;
  std::vector<std::string> iaExtensionPaths_;
  std::vector<std::pair<std::string, std::vector<char>>> iaExtensionLibs_;
  NNPICompilationInfo compilationInfo_;

  Error
  updateCompilationConfigFromOptions(NNPICompilationOptions &compilationOptions,
                                     bool requiresDSPKernels);

  /// Setup compilation hints for \p F given \p backendSpecificNodeInfo.
  /// \returns an error if some validation issue is found given expected format.
  Error
  setupCompilationHints(const Function *F,
                        const BackendSpecificNodeInfo &backendSpecificNodeInfo);

  /// Update the internal compilation info object. Return true iff successful.
  bool updateCompilationInfo();
  ///@}
};
} // end namespace glow
#endif // GLOW_NNPI_FUNCTION_H
