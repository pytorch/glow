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

#include "glow/Backends/Backend.h"

namespace glow {

/// This is the IR-interpreter. It owns the IR, and the heap, and is able to
/// execute the instructions one at a time.
class Interpreter final : public Backend {
  /// State necessary to execute a function using the interpreter.
  std::unique_ptr<CompiledFunction> function_;

public:
  /// Ctor.
  Interpreter() = default;

  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  ~Interpreter() override = default;

  void init(std::unique_ptr<IRFunction> IR) override;

  void doForwardPass() override;

  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override;

  bool shouldLower(const Node *N) const override;
  /// @}
};

} // namespace glow

#endif // GLOW_BACKENDS_INTERPRETER_INTERPRETER_H
