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
#ifndef GLOW_BACKENDS_INTERPRETER_INTERPRETERFUNCTION_H
#define GLOW_BACKENDS_INTERPRETER_INTERPRETERFUNCTION_H

#include "glow/Backend/Backend.h"
#include "glow/Backend/BackendUtils.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Base/Tensor.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Quantization/Base/Base.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <unordered_map>

namespace glow {

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
class InterpreterFunction final : public CompiledFunction,
                                  public IRInstructionProcessingHandler {
  /// The IR to be executed.
  std::unique_ptr<IRFunction> F_;

  /// Maps Value.name to tensors for constants.
  std::unordered_map<std::string, Tensor *> constants_;

public:
  InterpreterFunction(std::unique_ptr<IRFunction> F,
                      runtime::RuntimeBundle &&bundle);

  /// \name CompiledFunction interface
  ///@{
  ~InterpreterFunction() override;

  Error execute(ExecutionContext *context) override;

  /// Collects constants for runtime.
  void collectConstants(const Module *module) override;

  /// Add a constant to the function, this is used for loading static
  /// placeholders.
  void addConstant(std::string name, Tensor *T);

  /// Get reference to IR function.
  IRFunction *getIR() { return F_.get(); }

  /// Read trace events out of this func and write them into /p context
  void translateTraceEvents(ExecutionContext *context) const override;

  /// \returns the backend used to compile this function.
  virtual std::string getCompileBackendName() const override {
    return "Interpreter";
  }
  ///@}
};

/// An InterpreterFunction bound to a specific invocation.
class BoundInterpreterFunction : public IRInstructionProcessingHandler {
  /// Maps values to Tensors, that are owned by this class.
  std::unordered_map<const Value *, Tensor *> tensors_;

  /// Maps values to Tensors, that are *not* owned by this class.
  std::unordered_map<const Value *, Tensor *> externalTensors_;

  /// A reference to the constant map from the owning InterpreterFunction.
  const std::unordered_map<std::string, Tensor *> &constants_;

public:
  explicit BoundInterpreterFunction(
      const std::unordered_map<std::string, Tensor *> &constants)
      : constants_(constants) {}

  ~BoundInterpreterFunction();

  Error execute(IRFunction *F, ExecutionContext *context);

  /// \returns a pointer to the tensor that is saved under \p v.
  Tensor *getTensor(const Value *v) const;

  /// \returns a typed handle to the tensor that is stored at \p v.
  template <class ElemTy = float>
  Handle<ElemTy> getWeightHandle(Value *v) const {
    return getTensor(v)->getHandle<ElemTy>();
  }

private:
  /// Allocate a tensor to back the value \p v. Do not allocate anything if a
  /// tensor is already allocated for \p v.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateTensor(const Value *v);

  /// Allocate an unowned tensor to back the value \p v. The source tensor of
  /// the unowned tensor is provided by \p src.
  /// \returns a tensor for \p v.
  Tensor *getOrCreateUnownedTensor(const Value *v, const Value *src,
                                   llvm::ArrayRef<dim_t> offsets);

  /// If a tensor is allocated for \p v then delete it.
  void deleteTensor(const Value *v);

  /// @name BoundInterpreterFunction methods. This is a list of method
  /// declerations that are used by the interpreter to dispatch different
  /// instructions.
  ///@{

#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME) void fwd##CLASS(const CLASS *I);
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
#include "glow/AutoGenInstr.def"

  template <typename ElemTy, typename AccumulatorTy,
            typename BiasElemTy = int32_t>
  void fwdConvolutionInstQuantizedImpl(Value *inV, Value *outV, Value *filterV,
                                       Value *biasV,
                                       llvm::ArrayRef<unsigned_t> kernelSizes,
                                       llvm::ArrayRef<unsigned_t> strides,
                                       llvm::ArrayRef<unsigned_t> pads,
                                       size_t group,
                                       llvm::ArrayRef<unsigned_t> dilation);

  template <typename ElemTy = float>
  void fwdConvolutionInstFloatImpl(Value *inV, Value *outV, Value *filterV,
                                   Value *biasV,
                                   llvm::ArrayRef<unsigned_t> kernelSizes,
                                   llvm::ArrayRef<unsigned_t> strides,
                                   llvm::ArrayRef<unsigned_t> pads,
                                   size_t group,
                                   llvm::ArrayRef<unsigned_t> dilation);

  template <typename ElemTy, typename AccumulatorTy,
            typename BiasElemTy = int32_t>
  void fwdConvolution3DInstQuantizedImpl(Value *inV, Value *outV,
                                         Value *filterV, Value *biasV,
                                         llvm::ArrayRef<unsigned_t> kernelSizes,
                                         llvm::ArrayRef<unsigned_t> strides,
                                         llvm::ArrayRef<unsigned_t> pads,
                                         size_t group);

  template <typename ElemTy = float>
  void fwdConvolution3DInstFloatImpl(Value *inV, Value *outV, Value *filterV,
                                     Value *biasV,
                                     llvm::ArrayRef<unsigned_t> kernelSizes,
                                     llvm::ArrayRef<unsigned_t> strides,
                                     llvm::ArrayRef<unsigned_t> pads,
                                     size_t group);

  template <typename ElemTy = float>
  void fwdConvTransposeInstFloatImpl(Value *inV, Value *outV, Value *filterV,
                                     Value *biasV,
                                     llvm::ArrayRef<unsigned_t> kernelSizes,
                                     llvm::ArrayRef<unsigned_t> strides,
                                     llvm::ArrayRef<unsigned_t> pads,
                                     size_t group,
                                     llvm::ArrayRef<unsigned_t> dilation);

  template <typename ElemTy = float>
  void fwdBatchNormalizationFloatImpl(const BatchNormalizationInst *I,
                                      int numDims);
  template <typename ParamTy = float16_t>
  void fwdBatchNormalizationI8Impl(const BatchNormalizationInst *I,
                                   int numDims);

  void fwdAvgPoolInstI8Impl(const AvgPoolInst *I);
  template <typename ElemTy> void fwdAvgPoolInstFloatImpl(const AvgPoolInst *I);

  void fwdAvgPool3DInstI8Impl(const AvgPoolInst *I);
  template <typename ElemTy>
  void fwdAvgPool3DInstFloatImpl(const AvgPoolInst *I);

  void fwdAdaptiveAvgPoolInstI8Impl(const AdaptiveAvgPoolInst *I);
  template <typename ElemTy>
  void fwdAdaptiveAvgPoolInstFloatImpl(const AdaptiveAvgPoolInst *I);

  template <typename ElemTy> void fwdSoftMaxInstImpl(const SoftMaxInst *I);

  template <typename ElemTy, typename AccumulatorTy>
  void fwdMatMulInstQuantizedImpl(const MatMulInst *I);
  template <typename ElemTy> void fwdMatMulInstFloatImpl(const MatMulInst *I);

  template <typename ElemTy, typename AccumulatorTy,
            typename BiasElemTy = int32_t>
  void fwdFullyConnectedInstQuantizedImpl(const FullyConnectedInst *I);
  template <typename ElemTy>
  void fwdFullyConnectedInstFloatImpl(const FullyConnectedInst *I);

  template <typename ElemTy, typename AccumulatorTy,
            typename BiasElemTy = int32_t>
  void fwdRowwiseQuantizedFullyConnectedInstImpl(Value *inV, Value *outV,
                                                 Value *weightsV, Value *biasV,
                                                 Value *scalesV,
                                                 Value *offsetsV);

  template <typename ElemTy, typename AccumulatorTy,
            typename BiasElemTy = int32_t>
  void fwdChannelwiseQuantizedConv2DInstImpl(
      const ChannelwiseQuantizedConvolutionInst *I);

  template <typename ElemTy, typename AccumulatorTy,
            typename BiasElemTy = int32_t>
  void fwdChannelwiseQuantizedConv3DInstImpl(
      const ChannelwiseQuantizedConvolutionInst *I);

  void fwdElementAddInstI8Impl(const ElementAddInst *I);
  template <typename ElemTy>
  void fwdElementAddInstArithmeticImpl(const ElementAddInst *I);

  void fwdElementMaxInstI8Impl(const ElementMaxInst *I);
  template <typename ElemTy>
  void fwdElementMaxInstArithmeticImpl(const ElementMaxInst *I);

  template <typename ElemTy>
  void fwdBatchedAddInstFloatImpl(const BatchedAddInst *I);

  template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
            typename CmpTy = ElemTy>
  void fwdElementCmpEQInstImpl(const ElementCmpEQInst *I);

  template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
            typename CmpTy = ElemTy>
  void fwdElementCmpNEQInstImpl(const ElementCmpNEQInst *I);

  template <typename ElemTy>
  void fwdBatchOneHotImpl(const glow::BatchOneHotInst *I);

  template <typename ElemTy>
  void fwdSpaceToDepthInstImpl(const glow::SpaceToDepthInst *I);

  template <typename ElemTy>
  void fwdResizeNearestInstImpl(const ResizeNearestInst *I);

  template <typename ElemTy>
  void fwdResizeBilinearInstImpl(const ResizeBilinearInst *I);

  template <typename ElemTy> void fwdSigmoidInstFloatImpl(const SigmoidInst *I);

  template <typename ElemTy> void fwdTanhInstFloatImpl(const TanhInst *I);

  template <typename ElemTy>
  void fwdCrossEntropyLossInstFloatImpl(const CrossEntropyLossInst *I);

  template <typename ElemTy>
  void fwdLocalResponseNormalizationInstFloatImpl(
      const glow::LocalResponseNormalizationInst *I);

  template <typename ElemTy>
  void fwdElementSubInstArithmeticImpl(const ElementSubInst *I);

  template <typename ElemTy>
  void fwdElementMulInstArithmeticImpl(const ElementMulInst *I);

  template <typename ElemTy>
  void fwdElementMinInstArithmeticImpl(const ElementMinInst *I);

  template <typename ElemTy, typename InstKind>
  void fwdUnaryArithmeticImpl(const InstKind *I,
                              std::function<float(float)> func);

  template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
            typename CmpTy = ElemTy>
  void fwdElementCmpLTEInstImpl(const ElementCmpLTEInst *I);

  template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
            typename CmpTy = ElemTy>
  void fwdElementCmpLTInstImpl(const ElementCmpLTInst *I);

  template <typename ElemTy, typename ElemOffsetTy, typename ElemScaleTy,
            typename CmpTy, typename InstCmpKind>
  void
  fwdElementCmpHelperImpl(const InstCmpKind *I,
                          std::function<bool(CmpTy LHS, CmpTy RHS)> cmpHelper);

  template <typename ElemTy>
  void fwdElementPowInstFloatImpl(const ElementPowInst *I);

  template <typename ElemTy>
  void fwdElementIsNaNInstFloatImpl(const ElementIsNaNInst *I);

  template <typename ElemTy>
  void fwdElementLogInstFloatImpl(const ElementLogInst *I);

  template <typename ElemTy>
  void fwdElementExpInstFloatImpl(const ElementExpInst *I);

  template <typename ElemTy>
  void fwdElementSignInstFloatImpl(const ElementSignInst *I);

  template <typename ElemTy>
  void fwdElementSelectInstFloatImpl(const ElementSelectInst *I);

  template <typename ElemTy, typename InstKind>
  void fwdUnaryTrigonometricImpl(const InstKind *I,
                                 std::function<float(float)> func);
  template <typename ElemTy>
  void fwdBatchedReduceAddInstImpl(Value *batch, Value *dest, unsigned_t axis,
                                   const ShapeVector &eBatchDims,
                                   const ShapeVector &eDestDims);

  template <typename ElemTy>
  void fwdBatchedReduceMinInstImpl(Value *batch, Value *dest,
                                   const ShapeVector &eBatchDims,
                                   const ShapeVector &eDestDims, ElemTy max);

  template <typename ElemTy>
  void fwdBatchedReduceMaxInstImpl(Value *batch, Value *dest,
                                   const ShapeVector &eBatchDims,
                                   const ShapeVector &eDestDims, ElemTy min);

  template <typename ElemTy>
  void fwdBatchedReduceProdInstFloatImpl(Value *batch, Value *dest,
                                         unsigned_t axis,
                                         const ShapeVector &eBatchDims,
                                         const ShapeVector &eDestDims);

  template <typename ElemTy>
  void fwdCumSumInstImpl(Value *input, Value *dest, bool exclusive,
                         bool reverse);

  template <typename ElemTy>
  void fwdLengthsSumInstFloatImpl(const LengthsSumInst *I);

  template <typename ElemTy> void fwdGatherInstImpl(const GatherInst *I);
  template <typename ElemTy> void fwdGatherNDInstImpl(const GatherNDInst *I);
  template <typename ElemTy>
  void fwdGatherRangesInstImpl(const GatherRangesInst *I);
  template <typename ElemTy>
  void fwdScatterDataInstCopyImpl(const ScatterDataInst *I);
  template <typename ElemTy>
  void fwdScatterDataInstAddFloatImpl(const ScatterDataInst *I);
  template <typename ElemTy>
  void fwdScatterDataInstAddQuantizedImpl(const ScatterDataInst *I);

  template <typename ElemTy>
  void fwdSparseLengthsSumInstI8Impl(const SparseLengthsSumInst *I);
  template <typename ElemTy, typename TI>
  void fwdSparseLengthsSumInstFloatImpl(const SparseLengthsSumInst *I);

  template <typename ElemTy>
  void
  fwdSparseLengthsWeightedSumInstI8Impl(const SparseLengthsWeightedSumInst *I);
  template <typename ElemTy, typename TI>
  void fwdSparseLengthsWeightedSumInstFloatImpl(
      const SparseLengthsWeightedSumInst *I);

  template <typename ElemTy>
  void fwdEmbeddingInstImpl(Tensor *wtT, Tensor *indT, Tensor *outT,
                            int64_t padIdx, bool sparse, bool scale,
                            dim_t embedding_dim);

  template <typename ElemTy>
  void fwdEmbeddingBagInstFloatImpl(const EmbeddingBagInst *I);

  template <typename ElemTy>
  void fwdSparseToDenseInstImpl(const SparseToDenseInst *I);

  template <class eTy>
  void fwdRescaleQuantizedInstImpl(Value *src, Value *dest,
                                   TensorQuantizationParams &srcQ,
                                   TensorQuantizationParams &destQ);

  template <typename ElemTy> void fwdModuloInstImpl(glow::ModuloInst const *I);

  template <typename T, typename AccumT, typename TI>
  void fwdRowwiseQuantizedSparseLengthsWeightedSumImpl(
      const RowwiseQuantizedSparseLengthsWeightedSumInst *I);

  template <typename T, typename AccumT, typename TI>
  void fwdFusedRowwiseQuantizedSparseLengthsWeightedSumImpl(
      const FusedRowwiseQuantizedSparseLengthsWeightedSumInst *I);

  template <typename T>
  void fwdNonMaxSuppressionInstImpl(glow::NonMaxSuppressionInst const *I);

  void fwdAudioSpectrogramInstFloatImpl(glow::AudioSpectrogramInst const *I);

  void fwdMFCCInstFloatImpl(glow::MFCCInst const *I);

  template <typename T>
  void fwdROIAlignInstFloatImpl(glow::ROIAlignInst const *I);

  template <typename T>
  void fwdBBoxTransformInstFloatImpl(glow::BBoxTransformInst const *I);

  template <typename T, typename AccumT>
  void fwdEmbeddingBagByteRowwiseOffsetsImpl(
      const EmbeddingBagByteRowwiseOffsetsInst *I);

  template <typename ElemTy> void fwdFlipInstImpl(const FlipInst *I);

  template <typename ElemTy>
  void fwdBatchedPairwiseDotProductInstImpl(
      const glow::BatchedPairwiseDotProductInst *I);

  template <typename ElemTy>
  void fwdBatchedPairwiseDotProductGradInstImpl(
      const glow::BatchedPairwiseDotProductGradInst *I);
  void fwdAvgPool2DGradInst(const AvgPoolGradInst *I);
  void fwdAvgPool3DGradInst(const AvgPoolGradInst *I);

  ///@}
};

} // end namespace glow

#endif // GLOW_BACKENDS_INTERPRETER_INTERPRETERFUNCTION_H
