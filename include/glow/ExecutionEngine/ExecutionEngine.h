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

#include "glow/Backends/Backend.h"
#include "glow/Backends/CompiledFunction.h"
#include "glow/Base/Train.h"
#include "glow/Base/Traits.h"
#include "glow/Graph/Graph.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <unordered_map>

namespace glow {

/// This is the ExecutionEngine. It owns the Graph, the IR, and the backends.
/// The Graph, IR, etc in this class are defined as pointers, in order to
/// erase the type and prevent the internal types from leaking out to the
/// users of this class.
class ExecutionEngine final {
  /// The Module that represents the high-level program.
  Module M_;
  /// The network execution backend.
  std::unique_ptr<Backend> backend_;
  /// A glow function compiled for this ExecutionEngine's backend.
  std::unique_ptr<CompiledFunction> function_;

  /// Optimize the graph, generate IR, and optimize the IR.
  std::unique_ptr<IRFunction> generateIR(CompilationMode mode, Function *F);

public:
  ExecutionEngine(BackendKind backendKind = BackendKind::Interpreter);

  ~ExecutionEngine();

  /// Set the code generator kind to \p backendKind. New code will be generated
  /// using this backend.
  void setBackend(BackendKind backendKind);

  /// Set the code generator to a custom \p backend.
  void setBackend(Backend *backend);

  /// \returns the internal graph.
  Module &getModule() { return M_; }

  /// \returns whether operation is supported by the underlying backend.
  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const {
    return backend_->isOpSupported(opKind, elementTy);
  }

  /// Optimize the graph, generate IR, optimize IR and compile it for a
  /// specific target. This method should be invoked before the run method.
  void compile(CompilationMode mode, Function *F);

  /// Save a bundle for a standalone execution. This method takes care of
  /// everything when preparing the bundle for saving. There is no need to
  /// invoke the compile method before it.
  /// Make \p networkName the function name for
  /// the entry point of the network and prepend all generated
  /// files with this name.
  void save(CompilationMode mode, Function *F, llvm::StringRef outputDir,
            llvm::StringRef networkName);

  /// This method updates the variables in \p nodes with the tensor content
  /// values \p inputs.
  void updateVariables(llvm::ArrayRef<Variable *> vars,
                       llvm::ArrayRef<Tensor *> inputs);

  /// Runs a single execution of the function.
  void run();

  /// Runs \p iterations iterations of the function. The method updates a local
  /// counter and future invocations of this method continue running iterations
  /// of the batch at the next available slice.
  /// The method updates the variables in \p vars with the tensors \p inputs.
  void runBatch(size_t iterations, llvm::ArrayRef<Variable *> vars,
                llvm::ArrayRef<Tensor *> inputs);

  // Update the content of the tensor \p v with \p input.
  void loadValueFromTensor(Variable *v, Tensor *input);
};

} // namespace glow

#endif // GLOW_EXECUTIONENGINE_EXECUTIONENGINE_H
