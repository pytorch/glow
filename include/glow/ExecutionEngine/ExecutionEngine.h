/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <unordered_map>

namespace glow {

/// This is the ExecutionEngine. It owns the Graph, the backend, and the
/// compiled function.  The Graph, etc in this class are defined as pointers, in
/// order to erase the type and prevent the internal types from leaking out to
/// the users of this class.
class ExecutionEngine final {
  /// The Module that represents the high-level program.
  Module M_;

  /// The network execution backend.
  Backend *backend_ = nullptr;

  /// Whether or not the ExecutionEngine owns the backend or is just using
  /// a backend provided from elsewhere. If ownsBackend is true,
  /// ~ExecutionEngine will delete the backend_.
  bool ownsBackend_ = false;

  /// The device manager for executing compiled funtions.
  std::unique_ptr<runtime::DeviceManager> device_;

  /// Glow functions compiled for this ExecutionEngine's backend.
  llvm::StringMap<std::unique_ptr<CompiledFunction>> compiledFunctions_;

  /// Single execution of the given \compiledFunction with the given context
  /// \bindings.
  void runInternal(ExecutionContext &context, llvm::StringRef name,
                   CompiledFunction &compiledFunction);

public:
  ExecutionEngine(llvm::StringRef backend = "Interpreter");

  ~ExecutionEngine();

  /// Set the code generator to \p backend. New code will be generated
  /// using this backend.
  void setBackend(llvm::StringRef backend);

  /// Set the code generator to a custom \p backend. If \p ownsBackend is false
  /// then ExecutionEngine will use the given backend without owning it which
  /// means that ~ExecutionEngine will not delete it.
  void setBackend(Backend *backend, bool ownsBackend = true);

  /// Get a pointer to the backend.
  const Backend *getBackend() const;

  /// \returns the internal graph.
  Module &getModule() { return M_; }

  /// Clears the DeviceManager and all CompiledFunctions.
  void clear();

  /// \returns the compiled function. If more than one function
  /// has been compiled by this ExecutionEngine then a name must be supplied
  /// to specify which function to return.
  CompiledFunction &getCompiledFunction();

  /// \returns the compiled function with the given \p name.
  CompiledFunction &getCompiledFunction(llvm::StringRef name);

  /// Stores \p func in the CompiledFunction map, enabling it to be run.
  void insertCompiledFunction(llvm::StringRef name,
                              std::unique_ptr<CompiledFunction> func);

  /// \returns whether a node with the provided \p NI is supported by the
  /// underlying backend.
  bool isOpSupported(const NodeInfo &NI) const {
    return backend_->isOpSupported(NI);
  }

  /// Optimize the Function \p f and pass it to the backend to compile it for a
  /// specific target, all given \p cctx. If \p clearOtherFunctions is false
  /// then the function will be added to the collection of previously compiled
  /// functions otherwise any previously compiled functions will be removed
  /// first. This method should be invoked before the run method.
  void compile(Function *F, CompilationContext &cctx,
               bool clearOtherFunctions = true);

  /// A convenience function for the most common type of compile.
  void compile(CompilationMode mode, Function *F,
               bool clearOtherFunctions = true);

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
/// (sampleCounter % batchsize).
void runBatch(ExecutionEngine &EE, PlaceholderBindings &bindings,
              size_t iterations, size_t &sampleCounter,
              llvm::ArrayRef<Placeholder *> ph,
              llvm::ArrayRef<Tensor *> inputs);

} // namespace glow

#endif // GLOW_EXECUTIONENGINE_EXECUTIONENGINE_H
