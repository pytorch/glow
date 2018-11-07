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
#include "glow/Graph/Context.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <unordered_map>

namespace glow {

class Context;
class IRFunction;
class Value;
class Tensor;
class Constant;

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
  InterpreterFunction(std::unique_ptr<IRFunction> F, const Context &ctx);

  /// \name CompiledFunction interface
  ///@{
  ~InterpreterFunction() override;

  void execute(Context &ctx) override;
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
                                   llvm::ArrayRef<size_t> offsets);

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
                                 llvm::ArrayRef<unsigned_t> kernelSizes,
                                 llvm::ArrayRef<unsigned_t> strides,
                                 llvm::ArrayRef<unsigned_t> pads, size_t group);

  template <typename ElemTy = float>
  void fwdConvolutionInst_FloatImpl(Value *inV, Value *outV, Value *filterV,
                                    Value *biasV,
                                    llvm::ArrayRef<unsigned_t> kernelSizes,
                                    llvm::ArrayRef<unsigned_t> strides,
                                    llvm::ArrayRef<unsigned_t> pads,
                                    size_t group);

  void fwdAvgPoolInst_I8Impl(const AvgPoolInst *I);
  template <typename ElemTy>
  void fwdAvgPoolInst_FloatImpl(const AvgPoolInst *I);
  template <typename ElemTy> void fwdSoftMaxInst_Impl(const SoftMaxInst *I);

  void fwdMatMulInst_I8Impl(const glow::MatMulInst *I);
  template <typename ElemTy>
  void fwdMatMulInst_FloatImpl(const glow::MatMulInst *I);

  void fwdElementAddInst_I8Impl(const ElementAddInst *I);
  template <typename ElemTy>
  void fwdElementAddInst_FloatImpl(const ElementAddInst *I);

  void fwdElementMaxInst_I8Impl(const ElementMaxInst *I);
  template <typename ElemTy>
  void fwdElementMaxInst_FloatImpl(const ElementMaxInst *I);

  void fwdBatchedAddInst_I8Impl(const BatchedAddInst *I);
  template <typename ElemTy>
  void fwdBatchedAddInst_FloatImpl(const BatchedAddInst *I);

  template <typename ElemTy>
  void fwdElementCmpEQInstImpl(const glow::ElementCmpEQInst *I);

  template <typename ElemTy>
  void fwdBatchOneHotImpl(const glow::BatchOneHotInst *I);

  template <typename ElemTy>
  void fwdSigmoidInst_FloatImpl(const SigmoidInst *I);

  template <typename ElemTy> void fwdTanhInst_FloatImpl(const TanhInst *I);

  template <typename ElemTy>
  void fwdCrossEntropyLossInst_FloatImpl(const CrossEntropyLossInst *I);

  template <typename ElemTy>
  void fwdLocalResponseNormalizationInst_FloatImpl(
      const glow::LocalResponseNormalizationInst *I);

  template <typename ElemTy>
  void fwdElementSubInst_FloatImpl(const ElementSubInst *I);

  template <typename ElemTy>
  void fwdElementMulInst_FloatImpl(const ElementMulInst *I);

  template <typename ElemTy>
  void fwdElementMinInst_FloatImpl(const ElementMinInst *I);

  template <typename ElemTy>
  void fwdElementCmpLTEInst_FloatImpl(const ElementCmpLTEInst *I);

  template <typename ElemTy>
  void fwdElementPowInst_FloatImpl(const ElementPowInst *I);

  template <typename ElemTy>
  void fwdElementIsNaNInst_FloatImpl(const ElementIsNaNInst *I);

  template <typename ElemTy>
  void fwdElementLogInst_FloatImpl(const ElementLogInst *I);

  template <typename ElemTy>
  void fwdElementSelectInst_FloatImpl(const ElementSelectInst *I);

  template <typename ElemTy>
  void fwdBatchedReduceAddInst_FloatImpl(Value *batch, Value *dest,
                                         unsigned_t axis,
                                         const ShapeVector &eBatchDims,
                                         const ShapeVector &eDestDims);

  template <typename ElemTy>
  void fwdLengthsSumInst_FloatImpl(const LengthsSumInst *I);

  void
  fwdSparseLengthsWeightedSumInst_I8Impl(const SparseLengthsWeightedSumInst *I);
  template <typename ElemTy>
  void fwdSparseLengthsWeightedSumInst_FloatImpl(
      const SparseLengthsWeightedSumInst *I);

  template <typename ElemTy>
  void fwdSparseToDenseInst_FloatImpl(const SparseToDenseInst *I);

  template <typename ElemTy>
  void fwdDequantizeInst_Impl(const DequantizeInst *I);
  ///@}
};

} // end namespace glow

#endif // GLOW_BACKENDS_INTERPRETER_INTERPRETERFUNCTION_H
