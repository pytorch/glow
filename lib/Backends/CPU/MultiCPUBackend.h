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
#ifndef GLOW_BACKENDS_CPU_MULTICPUBACKEND_H
#define GLOW_BACKENDS_CPU_MULTICPUBACKEND_H

#include "CPUBackend.h"
#include "glow/IR/IR.h"

namespace glow {

class MultiCPUBackend final : public CPUBackend {
  using CPUBackend::compile;

public:
  /// Ctor.
  MultiCPUBackend() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~MultiCPUBackend() override = default;

  std::unique_ptr<CompiledFunction>
  compile(std::vector<std::unique_ptr<IRFunction>> IR,
          FunctionGraph &G) const override;

  FunctionGraph partition(Function *F) const override;

  bool supportsPartitioning() const override { return true; }
  /// @}
};

} // namespace glow

#endif // GLOW_BACKENDS_CPU_MULTICPUBACKEND_H
