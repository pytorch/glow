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

  /// Wipe out the state of the backend.
  virtual void clear() = 0;

  /// Prepare the interpreter for execution of new code.
  virtual void init() = 0;

  /// Save the bundle for a later standalone execution.
  virtual void save(llvm::StringRef outputDir);

  /// Perform a single forward scan of the network, interpreting all of the
  /// instructions.
  virtual void doForwardPass() = 0;

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
  virtual bool shouldLower(Node *N) { return true; }
};

/// Create a backend of kind \p kind, to run the IR function \p M.
Backend *createBackend(BackendKind backendKind, IRFunction *M);

} // namespace glow

#endif // GLOW_BACKENDS_BACKEND_H
