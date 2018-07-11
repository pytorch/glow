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
#ifndef GLOW_BACKENDS_CPU_CPUFUNCTION_H
#define GLOW_BACKENDS_CPU_CPUFUNCTION_H

#include "GlowJIT.h"

#include "glow/Backends/CompiledFunction.h"

namespace glow {

/// A Glow IR function compiled for the CPU using LLVM.
class CPUFunction final : public CompiledFunction {
  /// The LLVM JIT engine. The jit must be initialized after the ctor
  /// initializes the LLVM backends.
  std::unique_ptr<llvm::orc::GlowJIT> JIT_;
  /// This represents the heap, that stores the activations at runtime.
  void *heap_;

public:
  /// Ctor.
  CPUFunction(std::unique_ptr<llvm::orc::GlowJIT> JIT, void *heap);

  /// \name CompiledFunction interface
  ///@{
  ~CPUFunction() override;

  void doForwardPass() override;
  ///@}
};

} // end namespace glow

#endif // GLOW_BACKENDS_CPU_CPUFUNCTION_H
