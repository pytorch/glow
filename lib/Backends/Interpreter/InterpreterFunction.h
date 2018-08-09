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
#ifndef GLOW_BACKENDS_INTERPRETER_INTERPRETERFUNCTION_H
#define GLOW_BACKENDS_INTERPRETER_INTERPRETERFUNCTION_H

#include "glow/Backends/CompiledFunction.h"
#include "glow/Base/Tensor.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <unordered_map>

namespace glow {

class Context;
class IRFunction;
class Value;
class Tensor;
class Variable;

// Forward declare all of the classes.
#define DEF_VALUE(CLASS, NAME) class CLASS;
#define DEF_INSTR(CLASS, NAME) class CLASS;
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
#include "glow/AutoGenInstr.def"

/// Function "compiled" for execution by the interpreter.
class InterpreterFunction final : public CompiledFunction {
  /// The IR to be executed.
  std::unique_ptr<IRFunction> F_;
  /// Maps values to Tensors, that are owned by this class.
  std::unordered_map<const Value *, Tensor *> tensors_;
  /// Maps values to Tensors, that are *not* owned by this class.
  std::unordered_map<const Value *, Tensor *> externalTensors_;

public:
  InterpreterFunction(std::unique_ptr<IRFunction> F);

  /// \name CompiledFunction interface
  ///@{
  ~InterpreterFunction() override;

  void execute() override;
  ///@}

private:
  /// \returns a pointer to the tensor that is saved under \p v.
  Tensor *getTensor(const Value *v) const;

  /// Allocate a tensor to back the value \p v. Do not allocate anything if a
  /// tensor is already allocated for \p v.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateTensor(const Value *v);

  /// Allocate an unowned tensor to back the value \p v. The source tensor of
  /// the unowned tensor is provided by \p src.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateUnownedTensor(const Value *v, const Value *src,
                                   llvm::ArrayRef<uint64_t> offsets);

  /// If a tensor is allocated for \p v then delete it.
  void deleteTensor(const Value *v);

  /// \returns a typed handle to the tensor that is stored at \p v.
  template <class ElemTy = float>
  Handle<ElemTy> getWeightHandle(Value *v) const {
    return getTensor(v)->getHandle<ElemTy>();
  }

  /// @name Interpreter methods. This is a list of method declerations that are
  /// used by the interpreter to dispatch different instructions.
  ///@{

#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME) void fwd##CLASS(const CLASS *I);
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
#include "glow/AutoGenInstr.def"

  void fwdConvolutionInst_I8Impl(Value *inV, Value *outV, Value *filterV,
                                 Value *biasV,
                                 llvm::ArrayRef<uint64_t> filterSizes,
                                 llvm::ArrayRef<uint64_t> strides,
                                 llvm::ArrayRef<uint64_t> pads, uint64_t group);
  void fwdConvolutionInst_FloatImpl(Value *inV, Value *outV, Value *filterV,
                                    Value *biasV,
                                    llvm::ArrayRef<uint64_t> filterSizes,
                                    llvm::ArrayRef<uint64_t> strides,
                                    llvm::ArrayRef<uint64_t> pads,
                                    uint64_t group);
  ///@}
};

} // end namespace glow

#endif // GLOW_BACKENDS_INTERPRETER_INTERPRETERFUNCTION_H
