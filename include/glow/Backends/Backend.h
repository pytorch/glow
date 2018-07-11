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
#ifndef GLOW_BACKENDS_BACKEND_H
#define GLOW_BACKENDS_BACKEND_H

#include "glow/Backends/CompiledFunction.h"
#include "glow/Base/Traits.h"
#include "glow/Optimizer/Optimizer.h"

#include <llvm/ADT/StringRef.h>

namespace glow {

class Context;
class IRFunction;
class Value;
class Tensor;
class Variable;
class Function;
class Node;

enum class BackendKind {
  Interpreter, // Execute the network with the built-in interpreter.
  OpenCL,      // Run the code on an OpenCL device.
  CPU,         // Compile and run the code on the host.
};

// This is the interface that glow backends need to implement.
class Backend {
public:
  /// Dtor.
  virtual ~Backend() = default;

  /// Generate code for input function \param IR.
  virtual std::unique_ptr<CompiledFunction>
  compile(std::unique_ptr<IRFunction> IR) const = 0;

  /// Save the bundle for a later standalone execution.
  virtual void save(std::unique_ptr<IRFunction> IR, llvm::StringRef outputDir) {
    GLOW_UNREACHABLE("Saving a bundle is not supported by the backend");
  }

  /// @name Backend transform methods for different phases.
  /// These methods are called by the compiler before code generation and gives
  /// the backend an opportunity to transform the graph before IRGen. The
  /// backend may insert target specific nodes. The backend is responsible for
  /// cleaning up after itself.
  /// \returns True if the graph was modified.
  ///@{
  virtual bool transformPreLowering(Function *F, CompilationMode mode) {
    return false;
  }
  virtual bool transformPostLowering(Function *F, CompilationMode mode) {
    return false;
  }
  /// @}

  /// \returns true if backend supports given kind of operation with
  /// the given \p elementTy element type.
  virtual bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const = 0;

  /// \returns true if the supplied Node \N should be lowered. By default, all
  /// Nodes are candidates for lowering.
  virtual bool shouldLower(const Node *N) const { return true; }

  /// \returns true if the Backend wants the buffer sharing optimization
  /// performed.
  virtual bool shouldShareBuffers() const { return true; }
};

/// Create a backend of kind \p kind.
Backend *createBackend(BackendKind backendKind);

} // namespace glow

#endif // GLOW_BACKENDS_BACKEND_H
