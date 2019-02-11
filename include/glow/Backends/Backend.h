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

class IRFunction;
class Node;
class Context;
class IRGenVisitor;

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

  /// Generate code for input function \param F.
  virtual std::unique_ptr<CompiledFunction> compile(Function *F) const = 0;

  /// Generate instrumented (for tracing) code for input function \param F.
  /// When writing a new backend you must override this function to support auto
  /// instrumentation.
  virtual std::unique_ptr<CompiledFunction>
  instrumentAndCompile(Function *F) const {
    GLOW_UNREACHABLE("Auto instrumentation is not supported by the backend");
    return compile(F);
  }

  /// Generate code for input function \param F but do not collect constants.
  virtual std::unique_ptr<CompiledFunction>
  compileWithoutConstants(Function *F) const = 0;

  /// Save the bundle for \p F for a later standalone execution
  /// in \p outputDir. Make \p networkName the function name for
  /// the entry point of the network and prepend all generated
  /// files with this name.
  virtual void save(Function *F, llvm::StringRef outputDir,
                    llvm::StringRef networkName) const {
    GLOW_UNREACHABLE("Saving a bundle is not supported by the backend");
  }

  /// Used by the compiler during graph optimization and before code generation,
  /// giving the backend an opportunity to transform the graph before IRGen. The
  /// backend may insert backend-specific nodes. The backend is responsible for
  /// cleaning up after itself.
  /// \returns True if the graph was modified.
  virtual bool transformPostLowering(Function *F, CompilationMode mode) const {
    return false;
  }

  /// \returns true if backend supports given kind of operation with
  /// the given \p elementTy element type.
  virtual bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const = 0;

  /// \returns true if the supplied Node \N should be lowered. By default, all
  /// Nodes are candidates for lowering.
  virtual bool shouldLower(const Node *N) const { return true; }

  /// \returns true if the Backend wants the buffer sharing optimization
  /// performed.
  virtual bool shouldShareBuffers() const { return true; }

  /// Optimize the Function \p F given compilation mode \p mode.
  void optimizeFunction(CompilationMode mode, Function *F);

  /// \returns true if Backend generated Instruction for Node \p N,
  /// using IRGenVisitor \p irgen.
  virtual bool generateInst(Node *N, IRGenVisitor &irgen) const {
    return false;
  }

protected:
  /// Inserts a TraceEventInst between every instruction, the most basic form of
  /// auto instrumentation. Necessary only if the Backend doesn't provide
  /// profiling/tracing in another way.
  /// Modifies \p IR and returns a new TraceInfo map.
  /// \p traceEventDataSize is the size of metrics collected for  a single
  /// TraceEvent, and should be equal to the Backend's getTraceEventDataSize()
  /// method.
  TraceInfo autoInstrument(IRFunction *IR, size_t traceEventDataSize) const;
};

/// Create a backend of kind \p kind.
Backend *createBackend(BackendKind backendKind);

// Backends that use Glow low-level IR should inherit from this class. It allows
// for unit tests to create low-level IR to compile and run.
class BackendUsingGlowIR : public Backend {
public:
  /// Generate code for input IR function \param IR. \p ctx is the context that
  /// maps the graph to the concrete execution environment for a specific
  /// function. This is used only for unit testing.
  virtual std::unique_ptr<CompiledFunction>
  compileIR(std::unique_ptr<IRFunction> IR) const = 0;
};

} // namespace glow

#endif // GLOW_BACKENDS_BACKEND_H
