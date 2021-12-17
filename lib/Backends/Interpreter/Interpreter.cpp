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

#include "glow/Backends/Interpreter/Interpreter.h"
#include "glow/Backends/Interpreter/InterpreterFunction.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Flags/Flags.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"

using namespace glow;

Expected<std::unique_ptr<CompiledFunction>>
Interpreter::compile(Function *F, const BackendOptions &opts) const {
  TraceInfo traceInfo = buildManualTraceInfo(F);
  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

  if (!opts.backendSpecificOpts.empty()) {
    parseBackendSpecificOptions(opts);
  }

  if (opts.autoInstrument) {
    autoInstrument(traceInfo, IR.get());
  }

  std::unique_ptr<CompiledFunction> compiledFunc;
  if (opts.collectConstants) {
    compiledFunc = compileIR(std::move(IR));
  } else {
    compiledFunc = compileIRWithoutConstants(std::move(IR));
  }

  compiledFunc->setTraceInfo(std::move(traceInfo));
  return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}

std::unique_ptr<CompiledFunction>
Interpreter::compileIR(std::unique_ptr<IRFunction> IR) const {
  auto *mod = IR->getParent();
  auto function = compileIRWithoutConstants(std::move(IR));
  auto IFunction = static_cast<InterpreterFunction *>(function.get());
  IFunction->collectConstants(mod);
  return function;
}

std::unique_ptr<CompiledFunction>
Interpreter::compileIRWithoutConstants(std::unique_ptr<IRFunction> IR) const {
  MemoryAllocator constantWeightsAllocator("ConstantWeights", 0);
  MemoryAllocator placeholderWeightsAllocator("PlaceholderWeights", 0);
  MemoryAllocator activationsAllocator("Activations", 0);
  runtime::RuntimeBundle bundle = runtime::RuntimeBundle::create(
      *IR, constantWeightsAllocator, placeholderWeightsAllocator,
      activationsAllocator);
  auto compiledFunction =
      glow::make_unique<InterpreterFunction>(std::move(IR), std::move(bundle));
  compiledFunction->setIRInstructionProcessingHandler(
      getIRInstructionProcessingHandler());
  return compiledFunction;
}

bool Interpreter::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {
  case Kinded::Kind::BatchedReduceProdNodeKind:
  case Kinded::Kind::FmodNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::DivNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MinNodeKind:
  case Kinded::Kind::ClipNodeKind:
  case Kinded::Kind::BatchedReduceMinNodeKind:
  case Kinded::Kind::BatchedReduceMaxNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy, ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::ResizeNearestNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy, ElemKind::Int16QTy, ElemKind::Int32QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});
  case Kinded::Kind::ResizeBilinearNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy, ElemKind::Int16QTy, ElemKind::Int32QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::BatchNormalizationNodeKind: {
    auto elemType = NI.getInElemTy(BatchNormalizationNode::InputIdx);

    // input can be int8, float16 or float32
    bool isNodePrecisionSupported =
        (elemType == ElemKind::Int8QTy || elemType == ElemKind::FloatTy ||
         elemType == ElemKind::Float16Ty);

    // parameters have to be float16 or float
    isNodePrecisionSupported = isNodePrecisionSupported &&
                               NI.allInputsAndOutputsHaveSameElemKind(
                                   {ElemKind::FloatTy, ElemKind::Float16Ty},
                                   {BatchNormalizationNode::InputIdx},
                                   {BatchNormalizationNode::ResultIdx});

    // input and output element types have to match
    isNodePrecisionSupported =
        isNodePrecisionSupported &&
        NI.allInputsAndOutputsHaveSameElemKind(
            {elemType},
            {BatchNormalizationNode::ScaleIdx, BatchNormalizationNode::BiasIdx,
             BatchNormalizationNode::MeanIdx, BatchNormalizationNode::VarIdx});
    return isNodePrecisionSupported;
  }

  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::AdaptiveAvgPoolNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int32ITy, ElemKind::FloatTy, ElemKind::Float16Ty,
         ElemKind::BFloat16Ty, ElemKind::Int8QTy});

  case Kinded::Kind::DynamicQuantizedFullyConnectedNodeKind:
    return (NI.getInElemTy(DynamicQuantizedFullyConnectedNode::InputIdx) ==
                ElemKind::Float16Ty ||
            NI.getInElemTy(DynamicQuantizedFullyConnectedNode::InputIdx) ==
                ElemKind::FloatTy) &&
           NI.getInElemTy(DynamicQuantizedFullyConnectedNode::WeightsIdx) ==
               ElemKind::Int8QTy &&
           NI.getInElemTy(DynamicQuantizedFullyConnectedNode::BiasIdx) ==
               ElemKind::FloatTy;

  case Kinded::Kind::DynamicRowwiseQuantizedFullyConnectedNodeKind:
    return (NI.getInElemTy(
                DynamicRowwiseQuantizedFullyConnectedNode::InputIdx) ==
                ElemKind::Float16Ty ||
            NI.getInElemTy(
                DynamicRowwiseQuantizedFullyConnectedNode::InputIdx) ==
                ElemKind::FloatTy) &&
           NI.getInElemTy(
               DynamicRowwiseQuantizedFullyConnectedNode::WeightsIdx) ==
               ElemKind::Int8QTy &&
           NI.getInElemTy(DynamicRowwiseQuantizedFullyConnectedNode::BiasIdx) ==
               ElemKind::FloatTy &&
           NI.getInElemTy(
               DynamicRowwiseQuantizedFullyConnectedNode::ScalesIdx) ==
               ElemKind::FloatTy &&
           NI.getInElemTy(
               DynamicRowwiseQuantizedFullyConnectedNode::OffsetsIdx) ==
               ElemKind::Int32ITy;

  case Kinded::Kind::MatMulNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy, ElemKind::Int16QTy});

  case Kinded::Kind::BatchMatMulNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});

  case Kinded::Kind::FullyConnectedNodeKind:
    if (!NI.getInTy(FullyConnectedNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});
    }
    return (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int8QTy}, {FullyConnectedNode::BiasIdx}) &&
            (NI.getInElemTy(FullyConnectedNode::BiasIdx) == ElemKind::Int8QTy ||
             NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                 ElemKind::Int32QTy ||
             NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                 ElemKind::FloatTy)) ||
           (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int16QTy}, {FullyConnectedNode::BiasIdx}) &&
            (NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                 ElemKind::Int16QTy ||
             NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
                 ElemKind::Int32QTy));

  case Kinded::Kind::MaxPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
                ElemKind::Int8QTy},
               {}, {MaxPoolNode::ArgmaxIdx}) &&
           (NI.getOutElemTy(MaxPoolNode::ArgmaxIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::ArgMaxNodeKind:
  case Kinded::Kind::ArgMinNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
               {ArgMaxNode::ResultIdx}) &&
           (NI.getOutElemTy(ArgMaxNode::ResultIdx) == ElemKind::Int64ITy ||
            NI.getOutElemTy(ArgMaxNode::ResultIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::AcosNodeKind:
  case Kinded::Kind::AsinNodeKind:
  case Kinded::Kind::AtanNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});

  case Kinded::Kind::PowNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy});
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
  case Kinded::Kind::LayerNormalizationNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::ExpNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::SoftPlusNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});
  case Kinded::Kind::SliceNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy, ElemKind::Int32QTy, ElemKind::Int64ITy,
         ElemKind::Int32ITy, ElemKind::BoolTy});
  case Kinded::Kind::SpaceToDepthNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy, ElemKind::Int64ITy});

  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::TouchNodeKind:
  case Kinded::Kind::InsertTensorNodeKind:
  case Kinded::Kind::ConcatNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int8QTy, ElemKind::Int16QTy, ElemKind::Int32ITy,
         ElemKind::Int64ITy, ElemKind::BoolTy});
  case Kinded::Kind::NonZeroNodeKind:
    return NI.getInElemTy(NonZeroNode::CondIdx) == ElemKind::BoolTy &&
           NI.getOutElemTy(NonZeroNode::ResultIdx) == ElemKind::Int32ITy;
  case Kinded::Kind::SelectNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
                ElemKind::Int8QTy},
               {SelectNode::CondIdx}) &&
           (NI.getInElemTy(SelectNode::CondIdx) == ElemKind::BoolTy);

  case Kinded::Kind::NotNodeKind:
  case Kinded::Kind::AndNodeKind:
  case Kinded::Kind::OrNodeKind:
  case Kinded::Kind::XorNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::BoolTy});

  case Kinded::Kind::SignNodeKind:
  case Kinded::Kind::CeilNodeKind:
  case Kinded::Kind::RoundNodeKind:
  case Kinded::Kind::SqrtNodeKind:
  case Kinded::Kind::RsqrtNodeKind:
  case Kinded::Kind::ReciprocalNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy});

  case Kinded::Kind::SinNodeKind:
  case Kinded::Kind::CosNodeKind:
  case Kinded::Kind::ErfNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});

  case Kinded::Kind::AbsNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});

  case Kinded::Kind::NegNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int32ITy, ElemKind::Int8QTy});

  case Kinded::Kind::FloorNodeKind:
  case Kinded::Kind::TruncateNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});

  case Kinded::Kind::CmpEQNodeKind:
  case Kinded::Kind::CmpNEQNodeKind:
  case Kinded::Kind::CmpLTNodeKind:
  case Kinded::Kind::CmpLTENodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
                ElemKind::Int8QTy, ElemKind::Int16QTy, ElemKind::Int32ITy,
                ElemKind::Int64ITy},
               {}, {CmpEQNode::ResultIdx}) &&
           (NI.getOutElemTy(CmpEQNode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::IsNaNNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
               {}, {IsNaNNode::ResultIdx}) &&
           (NI.getOutElemTy(IsNaNNode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::BitwiseAndNodeKind:
  case Kinded::Kind::BitwiseOrNodeKind:
  case Kinded::Kind::BitwiseXorNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::BoolTy, ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::BitwiseNotNodeKind:
  case Kinded::Kind::ModuloNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::ConvolutionNodeKind:
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});
    }
    return (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int8QTy}, {ConvolutionNode::BiasIdx}) &&
            (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int8QTy ||
             NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy)) ||
           (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int16QTy}, {ConvolutionNode::BiasIdx}) &&
            (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int16QTy ||
             NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind:
    return (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::InputIdx) ==
            ElemKind::Int8QTy) &&
           (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::FilterIdx) ==
            ElemKind::Int8QTy) &&
           ((NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::BiasIdx) ==
             ElemKind::Int8QTy) ||
            (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::BiasIdx) ==
             ElemKind::Int32QTy)) &&
           (NI.getInElemTy(
                ChannelwiseQuantizedConvolutionNode::FilterScalesIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(
                ChannelwiseQuantizedConvolutionNode::FilterOffsetsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getInElemTy(
                ChannelwiseQuantizedConvolutionNode::BiasScalesIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(
                ChannelwiseQuantizedConvolutionNode::BiasOffsetsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getOutElemTy(ChannelwiseQuantizedConvolutionNode::ResultIdx) ==
            ElemKind::Int8QTy);

  case Kinded::Kind::Convolution3DNodeKind:
    if (!NI.getInTy(Convolution3DNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});
    }
    return (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int8QTy}, {Convolution3DNode::BiasIdx}) &&
            (NI.getInElemTy(Convolution3DNode::BiasIdx) == ElemKind::Int8QTy ||
             NI.getInElemTy(Convolution3DNode::BiasIdx) ==
                 ElemKind::Int32QTy)) ||
           (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int16QTy}, {Convolution3DNode::BiasIdx}) &&
            (NI.getInElemTy(Convolution3DNode::BiasIdx) == ElemKind::Int16QTy ||
             NI.getInElemTy(Convolution3DNode::BiasIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::ConvTransposeNodeKind:
    // TODO - support other types.
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::BatchedAddNodeKind:
    if (!NI.getInTy(BatchedAddNode::BatchIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});
    }
    return (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int8QTy}, {BatchedAddNode::SliceIdx}) &&
            (NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int8QTy ||
             NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int32QTy)) ||
           (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::Int16QTy}, {BatchedAddNode::SliceIdx}) &&
            (NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int16QTy ||
             NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind:
    return (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::InputIdx) ==
            ElemKind::Int8QTy) &&
           (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::WeightsIdx) ==
            ElemKind::Int8QTy) &&
           (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::ScalesIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::OffsetsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::BiasIdx) ==
                ElemKind::Int8QTy ||
            NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::BiasIdx) ==
                ElemKind::Int32QTy) &&
           (NI.getOutElemTy(RowwiseQuantizedFullyConnectedNode::ResultIdx) ==
            ElemKind::Int8QTy);

  case Kinded::Kind::SparseLengthsSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
                ElemKind::Int8QTy},
               {SparseLengthsSumNode::IndicesIdx,
                SparseLengthsSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsSumNode::IndicesIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(SparseLengthsSumNode::IndicesIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(SparseLengthsSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
                ElemKind::Int8QTy},
               {SparseLengthsWeightedSumNode::IndicesIdx,
                SparseLengthsWeightedSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::EmbeddingNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
               {EmbeddingNode::IndicesIdx}) &&
           (NI.getInElemTy(EmbeddingNode::IndicesIdx) == ElemKind::Int64ITy ||
            NI.getInElemTy(EmbeddingNode::IndicesIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::EmbeddingBagNodeKind:
    return (NI.allInputsAndOutputsHaveSameElemKind(
                {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
                {EmbeddingBagNode::IndicesIdx, EmbeddingBagNode::OffsetsIdx}) &&
            (((NI.getInElemTy(EmbeddingBagNode::IndicesIdx) ==
               ElemKind::Int64ITy) &&
              (NI.getInElemTy(EmbeddingBagNode::OffsetsIdx) ==
               ElemKind::Int64ITy)) ||
             ((NI.getInElemTy(EmbeddingBagNode::IndicesIdx) ==
               ElemKind::Int32ITy) &&
              (NI.getInElemTy(EmbeddingBagNode::OffsetsIdx) ==
               ElemKind::Int32ITy))));

  case Kinded::Kind::SparseLengthsWeightedSumGradNodeKind:
    // GradOfInputNamedIndicesIdx and GradOfInputNamedLengthsIdx do not need to
    // be checked because they are not used.
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
               {SparseLengthsWeightedSumGradNode::IndicesIdx,
                SparseLengthsWeightedSumGradNode::LengthsIdx},
               {SparseLengthsWeightedSumGradNode::GradOfInputNamedIndicesIdx,
                SparseLengthsWeightedSumGradNode::
                    GradOfInputNamedLengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsWeightedSumGradNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(SparseLengthsWeightedSumGradNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
               {RowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx,
                RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx,
                RowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx}) &&
           (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx) ==
            ElemKind::UInt8QTy) &&
           (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind: {
    if (!((NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::IndicesIdx) ==
               ElemKind::Int32ITy &&
           NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::OffsetsIdx) ==
               ElemKind::Int32ITy) ||
          (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::IndicesIdx) ==
               ElemKind::Int64ITy &&
           NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::OffsetsIdx) ==
               ElemKind::Int64ITy))) {
      return false;
    }

    switch (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::DataIdx)) {
    case ElemKind::UInt4FusedFP16QTy:
    case ElemKind::UInt8FusedFP16QTy:
      return (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::WeightsIdx) ==
              ElemKind::Float16Ty) &&
             (NI.getOutElemTy(EmbeddingBagByteRowwiseOffsetsNode::ResultIdx) ==
              ElemKind::Float16Ty);
    case ElemKind::UInt8FusedQTy:
      return (
          (((NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                WeightsIdx) == ElemKind::FloatTy) &&
            (NI.getOutElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                 ResultIdx) == ElemKind::FloatTy))) ||
          ((NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               WeightsIdx) == ElemKind::Float16Ty) &&
           (NI.getOutElemTy(
                FusedRowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx) ==
            ElemKind::Float16Ty)));
    default:
      return false;
    }
  }

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind: {
    if ((NI.getInElemTy(
             FusedRowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) !=
             ElemKind::Int64ITy &&
         NI.getInElemTy(
             FusedRowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) !=
             ElemKind::Int32ITy) ||
        NI.getInElemTy(
            FusedRowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx) !=
            ElemKind::Int32ITy) {
      return false;
    }

    switch (NI.getInElemTy(
        FusedRowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx)) {
    case ElemKind::UInt4FusedFP16QTy:
    case ElemKind::UInt8FusedFP16QTy:
      return (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                 WeightsIdx) == ElemKind::Float16Ty ||
              NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                 WeightsIdx) == ElemKind::FloatTy) &&
             (NI.getOutElemTy(
                  FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                      ResultIdx) == ElemKind::Float16Ty);
    case ElemKind::UInt4FusedQTy:
    case ElemKind::UInt8FusedQTy:
      if ((NI.getInElemTy(
               FusedRowwiseQuantizedSparseLengthsWeightedSumNode::WeightsIdx) ==
           ElemKind::FloatTy) &&
          (NI.getOutElemTy(
               FusedRowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx) ==
           ElemKind::FloatTy)) {
        return true;
      }
      return (
          (NI.getInElemTy(
               FusedRowwiseQuantizedSparseLengthsWeightedSumNode::WeightsIdx) ==
           ElemKind::Float16Ty) &&
          (NI.getOutElemTy(
               FusedRowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx) ==
           ElemKind::Float16Ty));
    default:
      return false;
    }
  }

  case Kinded::Kind::LengthsRangeFillNodeKind:
  case Kinded::Kind::LengthsToRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});

  case Kinded::Kind::GatherNodeKind:
    // Note: Data and Result can be any data type, but must match.
    return (NI.getInElemTy(GatherNode::DataIdx) ==
            NI.getOutElemTy(GatherNode::ResultIdx)) &&
           ((NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int32ITy) ||
            (NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int64ITy));

  case Kinded::Kind::GatherNDNodeKind:
    // Note: Data and Result can be any data type, but must match.
    return ((NI.getInElemTy(GatherNDNode::IndicesIdx) == ElemKind::Int32ITy) ||
            (NI.getInElemTy(GatherNDNode::IndicesIdx) == ElemKind::Int64ITy));

  case Kinded::Kind::GatherElementsNodeKind:
    // Note: Data and Result can be any data type, but must match.
    return (NI.getInElemTy(GatherNode::DataIdx) ==
            NI.getOutElemTy(GatherNode::ResultIdx)) &&
           ((NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int32ITy) ||
            (NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int64ITy));

  case Kinded::Kind::BatchOneHotNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
                ElemKind::Int8QTy, ElemKind::Int32ITy, ElemKind::Int64ITy},
               {BatchOneHotNode::LengthsIdx}) &&
           (NI.getInElemTy(BatchOneHotNode::LengthsIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::QuantizationProfileNodeKind:
  case Kinded::Kind::AvgPoolGradNodeKind:
  case Kinded::Kind::AdaptiveAvgPoolGradNodeKind:
  case Kinded::Kind::LocalResponseNormalizationGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::MaxPoolGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
               {MaxPoolGradNode::OriginalOutputForArgmaxIdx,
                MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx}) &&
           (NI.getInElemTy(MaxPoolGradNode::OriginalOutputForArgmaxIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(
                MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx) ==
            ElemKind::Int64ITy);

  case Kinded::Kind::QuantizeNodeKind:
    return ((NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::FloatTy) ||
            (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::Float16Ty) ||
            (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::BFloat16Ty)) &&
           ((NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int8QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::UInt8QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int16QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::DequantizeNodeKind:
    return ((NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int8QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::UInt8QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int16QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int32QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) ==
             ElemKind::UInt8FusedQTy)) &&
           ((NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::FloatTy) ||
            (NI.getOutElemTy(DequantizeNode::ResultIdx) ==
             ElemKind::Float16Ty) ||
            (NI.getOutElemTy(DequantizeNode::ResultIdx) ==
             ElemKind::BFloat16Ty));

  case Kinded::Kind::RescaleQuantizedNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int8QTy, ElemKind::Int16QTy, ElemKind::Int32QTy});

  case Kinded::Kind::IntLookupTableNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int8QTy, ElemKind::Int16QTy});

  case Kinded::Kind::ConvertToNodeKind: {
    auto isConversionSupportedFor = [](ElemKind kind) {
      switch (kind) {
      case ElemKind::Float16Ty:
      case ElemKind::BFloat16Ty:
      case ElemKind::FloatTy:
      case ElemKind::Int32ITy:
      case ElemKind::Int64ITy:
      case ElemKind::BoolTy:
        return true;
      default:
        return false;
      }
    };
    return (isConversionSupportedFor(NI.getInElemTy(ConvertToNode::InputIdx)) &&
            isConversionSupportedFor(
                NI.getOutElemTy(ConvertToNode::ResultIdx))) ||
           (NI.getInElemTy(ConvertToNode::InputIdx) ==
                ElemKind::UInt8FusedQTy &&
            NI.getOutElemTy(ConvertToNode::ResultIdx) ==
                ElemKind::UInt8FusedFP16QTy);
  }

  case Kinded::Kind::TopKNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
                ElemKind::Int8QTy},
               {}, {TopKNode::IndicesIdx}) &&
           ((NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int64ITy) ||
            (NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int32ITy));

  case Kinded::Kind::ScatterDataNodeKind:
    return ((NI.getInElemTy(ScatterDataNode::IndicesIdx) ==
             ElemKind::Int32ITy) ||
            (NI.getInElemTy(ScatterDataNode::IndicesIdx) ==
             ElemKind::Int64ITy)) &&
           (NI.getOutElemTy(ScatterDataNode::ResultIdx) ==
            NI.getInElemTy(ScatterDataNode::DataIdx)) &&
           (NI.getOutElemTy(ScatterDataNode::ResultIdx) ==
            NI.getInElemTy(ScatterDataNode::SlicesIdx));

  // We just clip 64 to 32 SelectedIdx silently with the SoftMax
  // SelectedIdx in case dim_t is 32b.
  case Kinded::Kind::SoftMaxNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
               {SoftMaxNode::SelectedIdx}) &&
           (NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int32ITy ||
            NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::LogSoftMaxNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
               {LogSoftMaxNode::SelectedIdx}) &&
           (NI.getInElemTy(LogSoftMaxNode::SelectedIdx) == ElemKind::Int32ITy ||
            NI.getInElemTy(LogSoftMaxNode::SelectedIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::GatherRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::Int32ITy, ElemKind::Int64ITy},
               {GatherRangesNode::DataIdx}, {GatherRangesNode::OutputIdx}) &&
           (NI.getOutElemTy(GatherRangesNode::OutputIdx) ==
            NI.getInElemTy(GatherRangesNode::DataIdx));

  case Kinded::Kind::CrossEntropyLossNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
               {CrossEntropyLossNode::LabelsIdx}) &&
           (NI.getInElemTy(CrossEntropyLossNode::LabelsIdx) ==
            ElemKind::Int64ITy);

  case Kinded::Kind::CumSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty,
         ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::LengthsSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
               {LengthsSumNode::LengthsIdx}) &&
           (NI.getInElemTy(LengthsSumNode::LengthsIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::BatchSparseToDenseNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
               {BatchSparseToDenseNode::LengthsIdx,
                BatchSparseToDenseNode::IndicesIdx}) &&
           (NI.getInElemTy(BatchSparseToDenseNode::LengthsIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(BatchSparseToDenseNode::LengthsIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(BatchSparseToDenseNode::IndicesIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(BatchSparseToDenseNode::IndicesIdx) ==
                ElemKind::Int32ITy);

  case Kinded::Kind::FillExamplesWithIndicatorNodeKind:
    return (NI.getInElemTy(FillExamplesWithIndicatorNode::DataIdx) ==
            NI.getOutElemTy(FillExamplesWithIndicatorNode::ResultIdx)) &&
           ((NI.getInElemTy(FillExamplesWithIndicatorNode::IndicatorIdx) ==
             ElemKind::Int32ITy) ||
            (NI.getInElemTy(FillExamplesWithIndicatorNode::IndicatorIdx) ==
             ElemKind::Int64ITy) ||
            (NI.getInElemTy(FillExamplesWithIndicatorNode::IndicatorIdx) ==
             ElemKind::BoolTy));

  case Kinded::Kind::SparseToDenseMaskNodeKind:
    return (NI.getInElemTy(SparseToDenseMaskNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(SparseToDenseMaskNode::LengthsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getInElemTy(SparseToDenseMaskNode::ValuesIdx) ==
            NI.getInElemTy(SparseToDenseMaskNode::DefaultValueIdx)) &&
           (NI.getInElemTy(SparseToDenseMaskNode::ValuesIdx) ==
            NI.getOutElemTy(SparseToDenseMaskNode::ResultIdx));

  case Kinded::Kind::SparseLabelSplitNodeKind:
    return (NI.getInElemTy(SparseLabelSplitNode::LengthsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getInElemTy(SparseLabelSplitNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(SparseLabelSplitNode::ValuesIdx) ==
            NI.getOutElemTy(SparseLabelSplitNode::LabelValuesIdx)) &&
           (NI.getOutElemTy(SparseLabelSplitNode::ExampleIdsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getOutElemTy(SparseLabelSplitNode::GradientOffsetMapIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::TraceEventNodeKind:
    return NI.getInElemTy(TraceEventNode::DataIdx) == ElemKind::Int64ITy;

  case Kinded::Kind::TransposeNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
  case Kinded::Kind::SaveNodeKind:
  case Kinded::Kind::FlipNodeKind:
    // These work regardless of the underlying type.
    return true;

  case Kinded::Kind::GaussianFillNodeKind:
    return NI.getOutElemTy(GaussianFillNode::ResultIdx) == ElemKind::Float16Ty;

  case Kinded::Kind::NonMaxSuppressionNodeKind:
    return NI.getInElemTy(NonMaxSuppressionNode::BoxesIdx) ==
               ElemKind::FloatTy &&
           NI.getInElemTy(NonMaxSuppressionNode::ScoresIdx) ==
               ElemKind::FloatTy &&
           (NI.getOutElemTy(NonMaxSuppressionNode::IndicesIdx) ==
                ElemKind::Int32ITy ||
            NI.getOutElemTy(NonMaxSuppressionNode::IndicesIdx) ==
                ElemKind::Int64ITy) &&
           (NI.getOutElemTy(
                NonMaxSuppressionNode::NumberOfSelectedIndicesIdx) ==
                ElemKind::Int32ITy ||
            NI.getOutElemTy(
                NonMaxSuppressionNode::NumberOfSelectedIndicesIdx) ==
                ElemKind::Int64ITy) &&
           (NI.getOutElemTy(
                NonMaxSuppressionNode::NumberOfSelectedIndicesIdx) ==
            NI.getOutElemTy(NonMaxSuppressionNode::IndicesIdx));

  case Kinded::Kind::TFLiteDetectionPostProcessNodeKind:
    return NI.getInElemTy(TFLiteDetectionPostProcessNode::BoxesIdx) ==
               ElemKind::FloatTy &&
           NI.getInElemTy(TFLiteDetectionPostProcessNode::ScoresIdx) ==
               ElemKind::FloatTy &&
           NI.getInElemTy(TFLiteDetectionPostProcessNode::AnchorsIdx) ==
               ElemKind::FloatTy &&
           NI.getOutElemTy(TFLiteDetectionPostProcessNode::DetectionBoxesIdx) ==
               ElemKind::FloatTy &&
           NI.getOutElemTy(
               TFLiteDetectionPostProcessNode::DetectionClassesIdx) ==
               ElemKind::Int32ITy &&
           NI.getOutElemTy(
               TFLiteDetectionPostProcessNode::DetectionScoresIdx) ==
               ElemKind::FloatTy &&
           NI.getOutElemTy(TFLiteDetectionPostProcessNode::NumDetectionsIdx) ==
               ElemKind::Int32ITy;

  case Kinded::Kind::AudioSpectrogramNodeKind:
    return NI.getInElemTy(AudioSpectrogramNode::InputIdx) ==
               ElemKind::FloatTy &&
           NI.getOutElemTy(AudioSpectrogramNode::SpectrogramIdx) ==
               ElemKind::FloatTy;

  case Kinded::Kind::MFCCNodeKind:
    return NI.getInElemTy(MFCCNode::SpectrogramIdx) == ElemKind::FloatTy &&
           NI.getOutElemTy(MFCCNode::CoefficientsIdx) == ElemKind::FloatTy;

  case Kinded::Kind::ROIAlignNodeKind:
    return (NI.getInElemTy(ROIAlignNode::BatchIndicesIdx) ==
                ElemKind::Int32ITy ||
            NI.getInElemTy(ROIAlignNode::BatchIndicesIdx) ==
                ElemKind::Int64ITy) &&
           NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty},
               /*ignoreIn*/ {ROIAlignNode::BatchIndicesIdx});

  case Kinded::Kind::CollectRpnProposalsNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty});

  case Kinded::Kind::BBoxTransformNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty});

  case Kinded::Kind::SoftMaxGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {SoftMaxGradNode::SelectedIdx},
               {SoftMaxGradNode::GradOfInputNamedSelectedIdx}) &&
           (NI.getInElemTy(SoftMaxGradNode::SelectedIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::ConvolutionGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy}, {},
        {ConvolutionGradNode::GradOfInputNamedInputIdx});

  case Kinded::Kind::CrossEntropyLossGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {CrossEntropyLossGradNode::LabelsIdx},
               {CrossEntropyLossGradNode::GradOfInputNamedLabelsIdx}) &&
           (NI.getInElemTy(CrossEntropyLossGradNode::LabelsIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getOutElemTy(
                CrossEntropyLossGradNode::GradOfInputNamedLabelsIdx) ==
            ElemKind::Int64ITy);

  case Kinded::Kind::BatchedPairwiseDotProductNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::BatchedPairwiseDotProductGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::BucketizeNodeKind:
    return NI.getInElemTy(BucketizeNode::InputIdx) == ElemKind::FloatTy &&
           NI.getOutElemTy(BucketizeNode::ResultIdx) == ElemKind::Int32ITy;

  case Kinded::Kind::BatchedUnaryEmbeddingsBagsNodeKind:
    return (
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::BFloat16Ty},
            {BatchedUnaryEmbeddingsBagsNode::TableOffsetsIdx,
             BatchedUnaryEmbeddingsBagsNode::IndicesIdx,
             BatchedUnaryEmbeddingsBagsNode::OffsetsIdx}) &&
        (((NI.getInElemTy(BatchedUnaryEmbeddingsBagsNode::TableOffsetsIdx) ==
           ElemKind::Int64ITy) &&
          (NI.getInElemTy(BatchedUnaryEmbeddingsBagsNode::IndicesIdx) ==
           ElemKind::Int64ITy) &&
          (NI.getInElemTy(BatchedUnaryEmbeddingsBagsNode::OffsetsIdx) ==
           ElemKind::Int64ITy)) ||
         ((NI.getInElemTy(BatchedUnaryEmbeddingsBagsNode::TableOffsetsIdx) ==
           ElemKind::Int32ITy) &&
          (NI.getInElemTy(BatchedUnaryEmbeddingsBagsNode::IndicesIdx) ==
           ElemKind::Int32ITy) &&
          (NI.getInElemTy(BatchedUnaryEmbeddingsBagsNode::OffsetsIdx) ==
           ElemKind::Int32ITy))));

  case Kinded::Kind::IntNBitSplitEmbeddingBagsNodeKind:
    return (((NI.getInElemTy(IntNBitSplitEmbeddingBagsNode::IndicesIdx) ==
              ElemKind::Int64ITy) &&
             (NI.getInElemTy(IntNBitSplitEmbeddingBagsNode::OffsetsIdx) ==
              ElemKind::Int64ITy)) ||
            ((NI.getInElemTy(IntNBitSplitEmbeddingBagsNode::IndicesIdx) ==
              ElemKind::Int32ITy) &&
             (NI.getInElemTy(IntNBitSplitEmbeddingBagsNode::OffsetsIdx) ==
              ElemKind::Int32ITy))) &&
           NI.getInElemTy(IntNBitSplitEmbeddingBagsNode::DevWeightsIdx) ==
               ElemKind::UInt8ITy &&
           NI.getInElemTy(IntNBitSplitEmbeddingBagsNode::UvmWeightsIdx) ==
               ElemKind::UInt8ITy &&
           NI.getInElemTy(IntNBitSplitEmbeddingBagsNode::WeightsTysIdx) ==
               ElemKind::UInt8ITy &&
           NI.getInElemTy(
               IntNBitSplitEmbeddingBagsNode::WeightsPlacementsIdx) ==
               ElemKind::Int32ITy &&
           NI.getInElemTy(IntNBitSplitEmbeddingBagsNode::WeightsOffsetsIdx) ==
               ElemKind::Int32ITy &&
           NI.getInElemTy(IntNBitSplitEmbeddingBagsNode::DimOffsetsIdx) ==
               ElemKind::Int32ITy;

  case Kinded::Kind::IntNBitSplitEmbeddingWeightedBagsNodeKind:
    return (((NI.getInElemTy(
                  IntNBitSplitEmbeddingWeightedBagsNode::IndicesIdx) ==
              ElemKind::Int64ITy) &&
             (NI.getInElemTy(
                  IntNBitSplitEmbeddingWeightedBagsNode::OffsetsIdx) ==
              ElemKind::Int64ITy)) ||
            ((NI.getInElemTy(
                  IntNBitSplitEmbeddingWeightedBagsNode::IndicesIdx) ==
              ElemKind::Int32ITy) &&
             (NI.getInElemTy(
                  IntNBitSplitEmbeddingWeightedBagsNode::OffsetsIdx) ==
              ElemKind::Int32ITy))) &&
           NI.getInElemTy(
               IntNBitSplitEmbeddingWeightedBagsNode::DevWeightsIdx) ==
               ElemKind::UInt8ITy &&
           NI.getInElemTy(
               IntNBitSplitEmbeddingWeightedBagsNode::UvmWeightsIdx) ==
               ElemKind::UInt8ITy &&
           NI.getInElemTy(
               IntNBitSplitEmbeddingWeightedBagsNode::WeightsTysIdx) ==
               ElemKind::UInt8ITy &&
           NI.getInElemTy(
               IntNBitSplitEmbeddingWeightedBagsNode::WeightsPlacementsIdx) ==
               ElemKind::Int32ITy &&
           NI.getInElemTy(
               IntNBitSplitEmbeddingWeightedBagsNode::WeightsOffsetsIdx) ==
               ElemKind::Int32ITy &&
           NI.getInElemTy(
               IntNBitSplitEmbeddingWeightedBagsNode::DimOffsetsIdx) ==
               ElemKind::Int32ITy &&
           NI.getInElemTy(
               IntNBitSplitEmbeddingWeightedBagsNode::IndiceWeightIdx) ==
               ElemKind::FloatTy;

  default:
    return false;
  }
}

/// Use template meta-programming to check if typename ClassName contains
/// has_getLayout() method. Below generates a struct named has_getLayout that
/// looks for said method.
CLASS_CONTAINS_METHOD(getLayout)

template <typename T, std::enable_if_t<
                          !has_getLayout<T, ConvolutionLayout>::value, int> = 0>
static bool checkLayout(const T &I) {
  (void)I;
  return true;
}

template <typename T,
          std::enable_if_t<has_getLayout<T, ConvolutionLayout>::value, int> = 0>
static bool checkLayout(const T &I) {
  if (I.getLayout() != NHWC && I.getLayout() != NTHWC) {
    report("Glow Interpreter supports only NHWC and NTHWC");
    return false;
  }
  return true;
}

static bool checkLayoutForNode(const Node &N) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case Kinded::Kind::CLASS##Kind: {                                            \
    const CLASS *CI = llvm::cast<CLASS>(&N);                                   \
    return checkLayout(*CI);                                                   \
    break;                                                                     \
  }
  switch (N.getKind()) {
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Invalid instruction.");
  }
  return true;
}

bool Interpreter::verify(const Function &F, bool verbose) const {
  if (!F.verify(this)) {
    return false;
  }
  if (!checkAllNodesSupported(F, verbose)) {
    return false;
  }
  for (const Node &N : F.getNodes()) {
    if (!checkLayoutForNode(N)) {
      return false;
    }
    if (!checkNoFusionForNode(N)) {
      return false;
    }
  }
  return true;
}

static bool checkLayoutForInstr(const Instruction &I) {
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    const CLASS *CI = llvm::cast<CLASS>(&I);                                   \
    return checkLayout(*CI);                                                   \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)
  switch (I.getKind()) {
#include "glow/AutoGenInstr.def"
  default:
    llvm_unreachable("Invalid instruction.");
  }
  return true;
}

bool Interpreter::verify(const IRFunction &IR) const {
  for (const auto &I : IR.getInstrs()) {
    if (!checkNoFusionForInstr(I)) {
      return false;
    }
    if (!checkLayoutForInstr(I)) {
      return false;
    }
  }
  return true;
}

bool Interpreter::shouldLower(const Node *N) const {
  switch (N->getKind()) {
  case Kinded::Kind::ConvolutionNodeKind:
  case Kinded::Kind::Convolution3DNodeKind:
  case Kinded::Kind::SparseLengthsSumNodeKind:
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::BatchNormalizationNodeKind:
  case Kinded::Kind::BucketizeNodeKind:
    return false;
  case Kinded::Kind::LayerNormalizationNodeKind:
    return interpreter::flags::LowerLayerNormalization;
  case Kinded::Kind::BatchMatMulNodeKind:
    return interpreter::flags::LowerBatchMatMul;
  default:
    return true;
  }
}

/// Quantize the float \p bias for the given FullyConnectedNode as int32 using
/// \p inputScale, weight \p scales and \p offset=0. \returns false if the bias
/// was already quantized and thus no change was made and true otherwise.
static bool quantizeFCFloatBias(Function *F,
                                FullyConnectedNode &fullyConnected) {
  if (fullyConnected.getBias().getType()->isQuantizedType() ||
      (!fullyConnected.getWeights().getType()->isQuantizedType())) {
    return false;
  }
  assert(fullyConnected.getBias().getElementType() == ElemKind::FloatTy &&
         "Bias type must be a float in order to quantize it.");
  Constant *biasC =
      llvm::dyn_cast<Constant>(fullyConnected.getBias().getNode());
  assert(biasC && "bias input to FullyConnectedNode must be Constant in order "
                  "to quantize the bias");
  const auto &biasUnquantizedH = biasC->getPayload().getHandle<float>();
  // biasQuantizedT is Int32QTy
  const float inputScale = fullyConnected.getInput().getType()->getScale();
  const float weigthScale = fullyConnected.getWeights().getType()->getScale();
  const float scale = inputScale * weigthScale;
  auto biasQuantizedT = Tensor(ElemKind::Int32QTy, biasUnquantizedH.dims(),
                               /* scale */ scale, /* offset */ 0);
  auto biasQuantizedH = biasQuantizedT.getHandle<int32_t>();
  TensorQuantizationParams tqp;
  tqp.scale = scale;
  tqp.offset = 0;
  for (dim_t i = 0; i < biasQuantizedH.size(); ++i) {
    biasQuantizedH.raw(i) =
        quantization::quantize<int32_t>(biasUnquantizedH.raw(i), tqp);
  }
  auto biasQuantizedC = F->getParent()->createConstant(
      biasC->getName(), std::move(biasQuantizedT));
  auto newFullyConnectedNode = F->createFullyConnected(
      fullyConnected.getName(), fullyConnected.getInput(),
      fullyConnected.getWeights(), biasQuantizedC,
      fullyConnected.getResult().getType(), /* axis doens't matter */ 1);
  fullyConnected.getResult().replaceAllUsesOfWith(newFullyConnectedNode);
  return true;
}

/// Quantize the float \p bias for the given RowwiseQuantizedFullyConnectedNode
/// as int32. \returns false if the bias was already quantized and thus no
/// change was made and true otherwise.
static bool quantizeRQFCFloatBias(Function *F,
                                  RowwiseQuantizedFullyConnectedNode &rqfc) {
  if (rqfc.getBias().getType()->isQuantizedType() ||
      (!rqfc.getWeights().getType()->isQuantizedType())) {
    return false;
  }
  assert(rqfc.getBias().getElementType() == ElemKind::FloatTy &&
         "Bias type must be a float in order to quantize it.");
  Constant *biasC = llvm::dyn_cast<Constant>(rqfc.getBias().getNode());
  assert(biasC && "bias input to RowwiseQuantizedFullyConnectedNode must be a "
                  "Constant in order to quantize the bias");

  auto TQPs = getTensorQuantizationParams(
      biasC->getPayload(), quantization::Schema::Asymmetric, ElemKind::Int32QTy,
      0, biasC->dims()[0]);

  DCHECK_EQ(TQPs.size(), 1) << "Should only be one dimension to quantize on";

  auto biasQuantizedT = quantization::quantizeTensor(
      biasC->getPayload(), TQPs[0], ElemKind::Int32QTy);

  auto biasQuantizedC = F->getParent()->createConstant(
      biasC->getName(), std::move(biasQuantizedT));

  Constant *weights = llvm::dyn_cast<Constant>(rqfc.getWeights().getNode());
  Constant *scales = llvm::dyn_cast<Constant>(rqfc.getScales().getNode());
  Constant *offsets = llvm::dyn_cast<Constant>(rqfc.getOffsets().getNode());

  DCHECK(weights);
  DCHECK(scales);
  DCHECK(offsets);

  auto newRQFC = F->createRowwiseQuantizedFullyConnected(
      rqfc.getName(), rqfc.getInput(), weights, scales, offsets, biasQuantizedC,
      rqfc.getResult().getType());
  rqfc.getResult().replaceAllUsesOfWith(newRQFC);
  return true;
}

/// This function performs the channelwise quantization for the bias operand of
/// a ChannelwiseQuantizedConvolutionNode \p channelwiseConv from function \p F.
/// The quantization is done only if the bias is float. \returns false if the
/// bias was already quantized and thus no change was made and true otherwise.
static bool channelwiseQuantizeFloatBias(
    Function *F, ChannelwiseQuantizedConvolutionNode &channelwiseConv) {

  // If bias is already quantized then quit.
  if (channelwiseConv.getBias().getType()->isQuantizedType()) {
    return false;
  }

  DCHECK(channelwiseConv.getBias().getElementType() == ElemKind::FloatTy)
      << "Bias type must be a float in order to quantize it!";

  Constant *biasC =
      llvm::dyn_cast<Constant>(channelwiseConv.getBias().getNode());
  DCHECK(biasC)
      << "Bias input to ChannelwiseQuantizedConvolutionNode must be a Constant "
         "in order to quantize the bias!";

  Constant *filterScalesC =
      llvm::dyn_cast<Constant>(channelwiseConv.getFilterScales().getNode());
  DCHECK(filterScalesC)
      << "Filter scales input to ChannelwiseQuantizedConvolutionNode must be a "
         "Constant in order to quantize the bias!";

  // Create new constants for Bias, BiasScales and BiasOffsets operands.
  Constant *biasCQ = F->getParent()->createConstant(
      ElemKind::Int32QTy, biasC->getType()->dims(), 1.0, 0, biasC->getName());
  Constant *biasScalesC = F->getParent()->createConstant(
      ElemKind::FloatTy, biasC->getType()->dims(), "biasScales");
  Constant *biasOffsetsC = F->getParent()->createConstant(
      ElemKind::Int32ITy, biasC->getType()->dims(), "biasOffsets");

  // Quantize the bias operand manually from FloatTy to Int32QTy using the
  // quantization parameters biasScales[i] = inputScale * filterScales[i] and
  // biasOffsets[i] = 0.
  float inputScale = channelwiseConv.getInput().getType()->getScale();
  const auto &filterScalesH = filterScalesC->getPayload().getHandle<float>();
  const auto &biasH = biasC->getPayload().getHandle<float>();
  auto biasQH = biasCQ->getPayload().getHandle<int32_t>();
  auto biasScalesH = biasScalesC->getPayload().getHandle<float>();
  auto biasOffsetsH = biasOffsetsC->getPayload().getHandle<int32_t>();
  for (dim_t idx = 0, idxEnd = biasC->getType()->size(); idx < idxEnd; ++idx) {
    TensorQuantizationParams biasTQP;
    biasTQP.scale = inputScale * filterScalesH.raw(idx);
    biasTQP.offset = 0;
    biasQH.raw(idx) = quantization::quantize<int32_t>(biasH.raw(idx), biasTQP);
    biasScalesH.raw(idx) = biasTQP.scale;
    biasOffsetsH.raw(idx) = biasTQP.offset;
  }

  // Create new ChannelwiseQuantizedConvolutionNode with quantized bias
  // and explicit bias scales and offsets.
  auto *newChannelwiseConv = F->createChannelwiseQuantizedConv(
      channelwiseConv.getName(), channelwiseConv.getInput(),
      channelwiseConv.getFilter(), biasCQ, channelwiseConv.getFilterScales(),
      channelwiseConv.getFilterOffsets(), biasScalesC, biasOffsetsC,
      channelwiseConv.getResult().getType(), channelwiseConv.getKernels(),
      channelwiseConv.getStrides(), channelwiseConv.getPads(),
      channelwiseConv.getGroup(), channelwiseConv.getDilation());
  channelwiseConv.getResult().replaceAllUsesOfWith(newChannelwiseConv);
  return true;
}

Expected<bool> Interpreter::transformPostLowering(
    Function *F, CompilationContext &cctx,
    const glow::runtime::DeviceInfo *devInfo) const {
  LOG_SCOPE(F->getLogContext(), "Interpreter::transformPostLowering")

  bool changed = false;
  for (auto &node : F->getNodes()) {
    if (auto *channelwiseConv =
            llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(&node)) {
      changed |= channelwiseQuantizeFloatBias(F, *channelwiseConv);
    } else if (auto *fullyConnected =
                   llvm::dyn_cast<FullyConnectedNode>(&node)) {
      changed |= quantizeFCFloatBias(F, *fullyConnected);
    } else if (auto *rowwiseFC =
                   llvm::dyn_cast<RowwiseQuantizedFullyConnectedNode>(&node)) {
      changed |= quantizeRQFCFloatBias(F, *rowwiseFC);
    }
  }
  return changed;
}

void Interpreter::parseBackendSpecificOptions(
    const BackendOptions &opts) const {}

Expected<double> Interpreter::estimateNodeCost(const glow::Node *node) const {
  // Using default cost from Partitioner which is 1.
  return 1.0;
}
