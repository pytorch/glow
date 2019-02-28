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

#include "Interpreter.h"
#include "InterpreterFunction.h"

#include "glow/Backends/BackendUtils.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"

using namespace glow;

std::unique_ptr<CompiledFunction>
Interpreter::compile(Function *F, const CompilationOptions &opts) const {
  TraceInfo traceInfo = buildManualTraceInfo(F);
  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

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
  return compiledFunc;
}

std::unique_ptr<CompiledFunction>
Interpreter::compileIR(std::unique_ptr<IRFunction> IR) const {
  auto *mod = IR->getGraph()->getParent();
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
  runtime::RuntimeBundle bundle =
      generateRuntimeBundle(*IR, constantWeightsAllocator,
                            placeholderWeightsAllocator, activationsAllocator);
  return llvm::make_unique<InterpreterFunction>(std::move(IR), bundle);
}

bool Interpreter::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {
  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MinNodeKind:
  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::MaxPoolNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});

  case Kinded::Kind::PowNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::TanhNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty});

  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::InsertTensorNodeKind:
  case Kinded::Kind::DivNodeKind:
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::SliceNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int64ITy});

  case Kinded::Kind::SelectNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
               {SelectNode::CondIdx}) &&
           (NI.getInElemTy(SelectNode::CondIdx) == ElemKind::BoolTy);

  case Kinded::Kind::CmpLTENodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
               {CmpLTENode::ResultIdx}) &&
           (NI.getOutElemTy(CmpLTENode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::IsNaNNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty}, {},
               {CmpLTENode::ResultIdx}) &&
           (NI.getOutElemTy(CmpLTENode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::CmpEQNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int64ITy}, {},
               {CmpEQNode::ResultIdx}) &&
           (NI.getOutElemTy(CmpEQNode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::ModuloNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::ConvolutionNodeKind:
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    }
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::Int8QTy, ElemKind::Int16QTy},
               {ConvolutionNode::BiasIdx}) &&
           (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy);

  case Kinded::Kind::FullyConnectedNodeKind:
    if (!NI.getInTy(FullyConnectedNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    }
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::Int8QTy}, {FullyConnectedNode::BiasIdx}) &&
           (NI.getInElemTy(FullyConnectedNode::BiasIdx) == ElemKind::Int32QTy);

  case Kinded::Kind::BatchedAddNodeKind:
    if (!NI.getInTy(BatchedAddNode::BatchIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    }
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy},
                                                  {BatchedAddNode::SliceIdx}) &&
           ((NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int8QTy) ||
            (NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int32QTy));

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
            ElemKind::Int32QTy) &&
           (NI.getOutElemTy(RowwiseQuantizedFullyConnectedNode::ResultIdx) ==
            ElemKind::Int8QTy);

  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
               {SparseLengthsWeightedSumNode::IndicesIdx,
                SparseLengthsWeightedSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    return (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx) ==
            ElemKind::Int8QTy) &&
           (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::ScalesIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::OffsetsIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::WeightsIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getOutElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx) ==
            ElemKind::FloatTy);

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    return (NI.getInElemTy(
                FusedRowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx) ==
            ElemKind::UInt8FusedQTy) &&
           (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               WeightsIdx) == ElemKind::FloatTy) &&
           (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               IndicesIdx) == ElemKind::Int64ITy) &&
           (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               LengthsIdx) == ElemKind::Int32ITy) &&
           (NI.getOutElemTy(
                FusedRowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx) ==
            ElemKind::FloatTy);

  case Kinded::Kind::LengthsToRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});

  case Kinded::Kind::GatherNodeKind:
    // Note: Data and Result can be any data type, but must match.
    return (NI.getInElemTy(GatherNode::DataIdx) ==
            NI.getOutElemTy(GatherNode::ResultIdx)) &&
           ((NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int32ITy) ||
            (NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int64ITy));

  case Kinded::Kind::BatchOneHotNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int32ITy,
                ElemKind::Int64ITy},
               {BatchOneHotNode::LengthsIdx}) &&
           (NI.getInElemTy(BatchOneHotNode::LengthsIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::QuantizationProfileNodeKind:
  case Kinded::Kind::AvgPoolGradNodeKind:
  case Kinded::Kind::MaxPoolGradNodeKind:
  case Kinded::Kind::LocalResponseNormalizationGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::QuantizeNodeKind:
    return ((NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::FloatTy) ||
            (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::Float16Ty)) &&
           ((NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int8QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int16QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::DequantizeNodeKind:
    return ((NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int8QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int16QTy) ||
            (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int32QTy)) &&
           ((NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::FloatTy) ||
            (NI.getOutElemTy(DequantizeNode::ResultIdx) ==
             ElemKind::Float16Ty));

  case Kinded::Kind::RescaleQuantizedNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int8QTy, ElemKind::Int16QTy, ElemKind::Int32QTy});

  case Kinded::Kind::IntLookupTableNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy});

  case Kinded::Kind::ConvertToNodeKind:
    return ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::FloatTy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) ==
             ElemKind::Float16Ty)) ||
           ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::Float16Ty) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) == ElemKind::FloatTy));

  case Kinded::Kind::TopKNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
               {TopKNode::IndicesIdx}) &&
           (NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::ScatterAssignNodeKind:
    return (NI.getInElemTy(ScatterAssignNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getOutElemTy(ScatterAssignNode::ResultIdx) ==
            NI.getInElemTy(ScatterAssignNode::DataIdx)) &&
           (NI.getOutElemTy(ScatterAssignNode::ResultIdx) ==
            NI.getInElemTy(ScatterAssignNode::SlicesIdx));

  case Kinded::Kind::SoftMaxNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty},
               {SoftMaxNode::SelectedIdx}) &&
           (NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::GatherRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::Int32ITy, ElemKind::Int64ITy},
               {GatherRangesNode::DataIdx}, {GatherRangesNode::OutputIdx}) &&
           (NI.getOutElemTy(GatherRangesNode::OutputIdx) ==
            NI.getInElemTy(GatherRangesNode::DataIdx));

  case Kinded::Kind::CrossEntropyLossNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty},
               {CrossEntropyLossNode::LabelsIdx}) &&
           (NI.getInElemTy(CrossEntropyLossNode::LabelsIdx) ==
            ElemKind::Int64ITy);

  case Kinded::Kind::LengthsSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty},
               {LengthsSumNode::LengthsIdx}) &&
           (NI.getInElemTy(LengthsSumNode::LengthsIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::SparseToDenseNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty},
               {SparseToDenseNode::IndicesIdx}) &&
           (NI.getInElemTy(SparseToDenseNode::IndicesIdx) ==
            ElemKind::Int64ITy);

  case Kinded::Kind::SparseToDenseMaskNodeKind:
    return (NI.getInElemTy(SparseToDenseMaskNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(SparseToDenseMaskNode::LengthsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getInElemTy(SparseToDenseMaskNode::ValuesIdx) ==
            NI.getInElemTy(SparseToDenseMaskNode::DefaultValueIdx)) &&
           (NI.getInElemTy(SparseToDenseMaskNode::ValuesIdx) ==
            NI.getOutElemTy(SparseToDenseMaskNode::ResultIdx));

  case Kinded::Kind::TraceEventNodeKind:
    return NI.getInElemTy(TraceEventNode::DataIdx) == ElemKind::Int64ITy;

  case Kinded::Kind::TransposeNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
  case Kinded::Kind::SaveNodeKind:
    // These work regardless of the underlying type.
    return true;

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

  default:
    return false;
  }
}

bool Interpreter::shouldLower(const Node *N) const {
  if (N->getKind() == Kinded::Kind::ConvolutionNodeKind)
    return false;
  return true;
}

namespace glow {
Backend *createInterpreter() { return new Interpreter(); }
} // namespace glow
