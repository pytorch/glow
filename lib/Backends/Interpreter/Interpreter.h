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
#ifndef GLOW_BACKENDS_INTERPRETER_INTERPRETER_H
#define GLOW_BACKENDS_INTERPRETER_INTERPRETER_H

#include "InterpreterFunction.h"

#include "glow/Backend/Backend.h"

namespace glow {

/// This is the IR-interpreter. It owns the IR, and the heap, and is able to
/// execute the instructions one at a time.
class Interpreter final : public BackendUsingGlowIR {
public:
  /// Ctor.
  Interpreter() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~Interpreter() override = default;

  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "Interpreter"; }

  std::unique_ptr<CompiledFunction>
  compileIR(std::unique_ptr<IRFunction> IR) const override;

  std::unique_ptr<CompiledFunction>
  compileIRWithoutConstants(std::unique_ptr<IRFunction> IR) const;

  llvm::Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

  bool isOpSupported(const NodeInfo &NI) const override;

  bool shouldLower(const Node *N) const override;

  /// @}
  //
  /// \returns the size of metrics collected for a single TraceEvent.
  static size_t getTraceEventDataSizeStatic() { return sizeof(uint64_t); }
  size_t getTraceEventDataSize() const override {
    return Interpreter::getTraceEventDataSizeStatic();
  }
};

} // namespace glow

#endif // GLOW_BACKENDS_INTERPRETER_INTERPRETER_H
