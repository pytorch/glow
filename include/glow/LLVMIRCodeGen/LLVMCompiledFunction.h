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
#ifndef GLOW_LLVMIRCODEGEN_LLVMCOMPILEDFUNCTION_H
#define GLOW_LLVMIRCODEGEN_LLVMCOMPILEDFUNCTION_H

#include "glow/LLVMIRCodeGen/GlowJIT.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"

namespace glow {
/// A Glow IR function compiled using LLVM.
class LLVMCompiledFunction : public CompiledFunction {
public:
  LLVMCompiledFunction(std::unique_ptr<GlowJIT> JIT,
                       runtime::RuntimeBundle &&runtimeBundle);

  /// \name CompiledFunction interface
  ///@{
  virtual Error execute(ExecutionContext *context) override;

  virtual void collectConstants(const Module *module) override;

  /// Read trace events out of this func and write them into /p bindings
  virtual void translateTraceEvents(ExecutionContext *context) const override;
  ///@}
  //

protected:
  /// Load constant tensors from \p bindings into \p weightsAddress, as defined
  /// by the RuntimeBundle (pre-run).
  virtual void loadPlaceholders(PlaceholderBindings *bindings,
                                uint8_t *weightsAddress);

  /// Load weights from \p weightsAddress into applicable backing tensors in
  /// \p bindings, as defined by the RuntimeBundle (post-run).
  virtual void updatePlaceholders(PlaceholderBindings *bindings,
                                  uint8_t *weightsAddress);

  /// The LLVM JIT engine. The jit must be initialized after the ctor
  /// initializes the LLVM backends.
  std::unique_ptr<GlowJIT> JIT_;

  /// The JIT can be accessed from multiple threads but is not thread safe,
  /// JITLock_ protects it.
  std::mutex JITLock_;
};
} // end namespace glow

#endif // GLOW_LLVMIRCODEGEN_LLVMCOMPILEDFUNCTION_H
