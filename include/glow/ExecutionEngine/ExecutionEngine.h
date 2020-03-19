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
#ifndef GLOW_EXECUTIONENGINE_EXECUTIONENGINE_H
#define GLOW_EXECUTIONENGINE_EXECUTIONENGINE_H

#include "glow/Backend/Backend.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Base/Train.h"
#include "glow/Base/Traits.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Runtime/HostManager/HostManager.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <unordered_map>

namespace glow {

/// This is the ExecutionEngine. It encapsulates the Glow Runtime.  It handles
/// compilation and execution of a network.
class ExecutionEngine final {
  /// Module containing the function and supporting information. This is reset
  /// if the backend type is changed.
  std::unique_ptr<Module> module_;

  /// Raw pointer to module_ this is to support module access after the module
  /// has been added to hostManager_.
  Module *rawModule_;

  /// Name of the backend being used for compilation and execution. Changing
  /// this resets the ExecutionEngine.
  std::string backendName_ = "";

  /// Size of device memory in bytes, if 0 device default is used.
  uint64_t deviceMemory_{0};

  /// Whether to ignore the user-specified DeviceConfig.
  bool ignoreUserDeviceConfig_{false};

  /// The HostManager for executing the compiled functions.
  std::unique_ptr<runtime::HostManager> hostManager_;

  /// Glow functions compiled for this ExecutionEngine's backend.
  std::set<std::string> compiledFunctions_;

  /// Whether to move all Device Resident Tensors on to the host at the end of
  /// the run.
  bool ensureOutputsOnHost_{true};

  /// Whether to override the cctx's skipModuleStrip setting and skip stripping
  /// the module. Used for testing purposes.
  bool skipModuleStrip_{false};

  /// Whether to allow multiple functions when running. This is usually due to
  /// running a pre-partitioned model.
  bool allowMultiFunction_{false};

  /// Single execution of the given function, \p name with the given context
  /// \bindings.
  void runInternal(ExecutionContext &context, llvm::StringRef name);

public:
  /// Constructor for an ExecutionEngine with \p backend and memory \p
  /// deviceMemory in bytes. If \p ignoreUserDeviceConfig then user device
  /// configs will be ignored. \p numDevices controls how many devices to create
  /// for the EE.
  ExecutionEngine(llvm::StringRef backend = "Interpreter",
                  uint64_t deviceMemory = 0,
                  bool ignoreUserDeviceConfig = false, unsigned numDevices = 1);

  ~ExecutionEngine();

  /// Set the code generator to \p backend. New code will be generated
  /// using this backend. This clears all previously loaded functions and resets
  /// the Module. \p numDevices controls how many devices to create for the EE.
  void setBackendName(llvm::StringRef backend, size_t numDevices = 1);

  /// Set the device memory to \p mem. This will reset the existing device,
  /// clearing all existing functions and resetting the module.
  void setDeviceMemory(uint64_t mem) {
    deviceMemory_ = mem;
    setBackendName(backendName_);
  }

  // Set whether or not to ensure outputs are in host memory.
  void ensureOutputsOnHost(bool should) { ensureOutputsOnHost_ = should; }

  /// Get the name of the current backend in use.
  llvm::StringRef getBackendName() const;

  /// \returns the internal graph. Note: After compilation the contents of the
  /// module will have been altered and raw pointers to elements of the graph
  /// may no longer be valid.
  Module &getModule() const { return *rawModule_; }

  /// Clears the ExecutionEngine and all CompiledFunctions.
  void clear();

  /// \returns the DAG for the specified \p network.
  Expected<runtime::DAG *> getDAG(llvm::StringRef network) {
    return hostManager_->getNetworkDAG(network);
  }

  /// Compiles all functions in the Module with the given \p cctx.  This method
  /// should be invoked before the run method and can only be called once
  /// without resetting the backend.
  void compile(CompilationContext &cctx);

  /// A convenience function for the most common type of compile. Can only be
  /// called once without resetting the backend.
  void compile(CompilationMode mode);

  /// Context aware single execution of a function. If more than one
  /// function has been compiled by this ExecutionEngine then a name must be
  /// supplied to specify which function to run.
  void run(ExecutionContext &context);

  /// Context aware single execution of a function with the given \p
  /// name.
  void run(ExecutionContext &context, llvm::StringRef name);

  /// Context aware single execution of a function. If more than one
  /// function has been compiled by this ExecutionEngine then a name must be
  /// supplied to specify which function to run.
  void run(PlaceholderBindings &bindings);

  /// Context aware single execution of a function with the given \p
  /// name.
  void run(PlaceholderBindings &bindings, llvm::StringRef name);

  /// \returns a reference to the backend with name \p backendName owned by the
  /// Provisioner inside of \ref hostManager_.
  Backend &getBackend(llvm::StringRef backendName) const;

  /// \returns a reference to the backend with name of the current backend in
  /// use by the EE.
  Backend &getBackend() const;

  /// \returns the single Function contained in this Module.
  /// \pre Must be a single Function in the Module.
  Function *getSingleFunctionFromModule() const;

  /// Setter for \ref skipModuleStrip_ to \p b.
  void setSkipModuleStrip(bool b) { skipModuleStrip_ = b; }
};

//===----------------------------------------------------------------------===//
//         Helper methods for running the execution engine.
//===----------------------------------------------------------------------===//

/// This method updates the placeholders in \p ph with the tensor content
/// values \p inputs, in \p bindings.
void updateInputPlaceholders(PlaceholderBindings &bindings,
                             llvm::ArrayRef<Placeholder *> ph,
                             llvm::ArrayRef<Tensor *> inputs);

/// This method updates the placeholders in the module. The placeholders are
/// found by name
///  in \p ph with the tensor content values \p inputs.
void updateInputPlaceholdersByName(PlaceholderBindings &bindings, Module *mod,
                                   llvm::ArrayRef<llvm::StringRef> ph,
                                   llvm::ArrayRef<Tensor *> inputs);

/// Runs \p iterations iterations of the compiled function. The method updates a
/// global counter and future invocations of this method continue running
/// iterations of the batch at the next available slice.
///
/// The method updates the placeholder in \p ph with the tensors \p inputs. The
/// shape of the slice has to be identical to the shape of slices in the batch.
/// All dimensions, except for the first (batch) dimension must be identical.
///
/// The variable \p sampleCounter is consumed and updated by the function. This
/// variable records the number of samples that were consumed by the network in
/// previous iterations. The next input to be loaded is
/// (sampleCounter % batchsize). If there is more than one compiledFunction \p
/// name must be provided to specify the desired function.
void runBatch(ExecutionEngine &EE, PlaceholderBindings &bindings,
              size_t iterations, size_t &sampleCounter,
              llvm::ArrayRef<Placeholder *> ph, llvm::ArrayRef<Tensor *> inputs,
              llvm::StringRef name = "");

/// Runs \p numMinibatchRuns iterations of the compiled function called \p name.
/// The method updates a global counter and future invocations of this method
/// continue running iterations of the batch at the next available slice.
/// The provided callback function \p cb is invoked on each sample.
void evalBatch(
    ExecutionEngine &EE, PlaceholderBindings &bindings, size_t numMinibatchRuns,
    size_t &sampleCounter, Placeholder *inputPH, Placeholder *outputPH,
    Tensor &samplesInput, Tensor &labelsInput, llvm::StringRef name,
    std::function<void(const Tensor &sampleIn, const Tensor &sampleOut,
                       const Tensor &label, size_t sampleIndex)> &&cb);
} // namespace glow

#endif // GLOW_EXECUTIONENGINE_EXECUTIONENGINE_H
