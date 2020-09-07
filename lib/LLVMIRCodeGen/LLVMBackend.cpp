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

#include "glow/LLVMIRCodeGen/LLVMBackend.h"
#include "glow/LLVMIRCodeGen/BundleSaver.h"
#include "glow/LLVMIRCodeGen/CommandLine.h"
#include "glow/LLVMIRCodeGen/LLVMCompiledFunction.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

using namespace glow;

namespace {

//===----------------------------------------------------------------------===//
//                   Functions for executing code using JIT
//===----------------------------------------------------------------------===//

/// Perform memory allocation for a JIT execution.
void allocateJITMemory(const IRFunction *F, AllocationsInfo &allocationsInfo) {
  allocationsInfo.numberValues(F);
  allocationsInfo.allocateActivations(F);
  allocationsInfo.allocateWeightVars(F);
  allocationsInfo.allocateTensorViews(F);
}

} // end namespace

bool LLVMBackend::isOpSupported(const NodeInfo &NI) const {
  // Note: For brevity below, "X ==> Y, Z" signifes that Node X is IRGen'd into
  // Instructions Y and Z.
  switch (NI.getKind()) {
  case Kinded::Kind::BatchedReduceMaxNodeKind:
  case Kinded::Kind::BatchedReduceMinNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::BatchedReduceProdNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::MulNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int32ITy,
         ElemKind::Int64ITy});

  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ClipNodeKind:
  case Kinded::Kind::LeakyReluNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MinNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::AvgPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy});

  case Kinded::Kind::AdaptiveAvgPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::MaxPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
               {MaxPoolNode::ArgmaxIdx}) &&
           (NI.getOutElemTy(MaxPoolNode::ArgmaxIdx) == ElemKind::Int64ITy ||
            NI.getOutElemTy(MaxPoolNode::ArgmaxIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::ArgMaxNodeKind:
  case Kinded::Kind::ArgMinNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
               {ArgMaxNode::ResultIdx}) &&
           (NI.getOutElemTy(ArgMaxNode::ResultIdx) == ElemKind::Int64ITy ||
            NI.getOutElemTy(ArgMaxNode::ResultIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::ResizeNearestNodeKind:
  case Kinded::Kind::ResizeBilinearNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int32QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::SaveNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
    // These are implemented via a Copy Instruction.
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int32QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy, ElemKind::BoolTy});

    // InsertTensor ==> Copy + InsertTensor. Copy supports everything
    // ReshapeNode above supports, so InsertTensor is the limiting factor.
  case Kinded::Kind::InsertTensorNodeKind:
    // Concat ==> Splat + Insert. Both only support the following.
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::TouchNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy,
         ElemKind::Int32ITy, ElemKind::BoolTy});
  case Kinded::Kind::SliceNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int32QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});
  case Kinded::Kind::SpaceToDepthNodeKind:
  case Kinded::Kind::DivNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy,
         ElemKind::Int32ITy});

  case Kinded::Kind::TransposeNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy,
         ElemKind::BoolTy});

  case Kinded::Kind::FlipNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int16QTy,
         ElemKind::Int32QTy, ElemKind::Int32ITy, ElemKind::Int64ITy,
         ElemKind::BoolTy});

  case Kinded::Kind::SparseLengthsSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {SparseLengthsSumNode::IndicesIdx,
                                     SparseLengthsSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsSumNode::IndicesIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(SparseLengthsSumNode::IndicesIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(SparseLengthsSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
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
               {ElemKind::FloatTy}, {EmbeddingNode::IndicesIdx}) &&
           (NI.getInElemTy(EmbeddingNode::IndicesIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::EmbeddingBagNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
               {EmbeddingBagNode::IndicesIdx, EmbeddingBagNode::OffsetsIdx}) &&
           (NI.getInElemTy(EmbeddingBagNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(EmbeddingBagNode::OffsetsIdx) == ElemKind::Int64ITy);

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
                ElemKind::Int64ITy ||
            NI.getInElemTy(SparseLengthsWeightedSumGradNode::IndicesIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(SparseLengthsWeightedSumGradNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    return (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx) ==
            ElemKind::UInt8QTy) &&
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
                ElemKind::Int64ITy ||
            NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getOutElemTy(
                RowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx) ==
            ElemKind::FloatTy);

  case Kinded::Kind::LengthsRangeFillNodeKind:
  case Kinded::Kind::LengthsToRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});

  case Kinded::Kind::IntLookupTableNodeKind:
  case Kinded::Kind::RescaleQuantizedNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy});

  case Kinded::Kind::PowNodeKind:
  case Kinded::Kind::AvgPoolGradNodeKind:
  case Kinded::Kind::QuantizationProfileNodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
  case Kinded::Kind::LocalResponseNormalizationGradNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::ExpNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::ModuloNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::MaxPoolGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
               {MaxPoolGradNode::OriginalOutputForArgmaxIdx,
                MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx}) &&
           (NI.getInElemTy(MaxPoolGradNode::OriginalOutputForArgmaxIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(MaxPoolGradNode::OriginalOutputForArgmaxIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(
                MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(
                MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx) ==
                ElemKind::Int32ITy);

  case Kinded::Kind::ConvolutionNodeKind:
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});
    }

    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy},
                                                  {ConvolutionNode::BiasIdx}) &&
           (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int8QTy ||
            NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy);

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

  case Kinded::Kind::ConvTransposeNodeKind:
    // TODO - not quantized support yet in libjit.
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::BatchedAddNodeKind:
    if (!NI.getInTy(BatchedAddNode::BatchIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});
    }
    // Allow for Int8QTy or Int32QTy for the Slice input.
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy},
                                                  {BatchedAddNode::SliceIdx}) &&
           ((NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int8QTy) ||
            (NI.getInElemTy(BatchedAddNode::SliceIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::GatherNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy,
                ElemKind::Int32ITy},
               {GatherNode::IndicesIdx}) &&
           ((NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int32ITy) ||
            (NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int64ITy));

  case Kinded::Kind::GatherRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy,
                ElemKind::Int32ITy},
               {GatherRangesNode::RangesIdx}, {GatherRangesNode::LengthsIdx}) &&
           ((NI.getInElemTy(GatherRangesNode::RangesIdx) ==
             NI.getOutElemTy(GatherRangesNode::LengthsIdx)) &&
            ((NI.getOutElemTy(GatherRangesNode::LengthsIdx) ==
              ElemKind::Int32ITy) ||
             (NI.getOutElemTy(GatherRangesNode::LengthsIdx) ==
              ElemKind::Int64ITy)));

  case Kinded::Kind::ScatterDataNodeKind:
    // ScatterData ==> Copy + ScatterData. Copy supports everything
    // ReshapeNode above supports, however ScatterData only supports the
    // following.
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy},
               {ScatterDataNode::IndicesIdx}) &&
           (NI.getInElemTy(ScatterDataNode::IndicesIdx) == ElemKind::Int64ITy ||
            NI.getInElemTy(ScatterDataNode::IndicesIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::SelectNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int32ITy},
               {SelectNode::CondIdx}) &&
           ((NI.getInElemTy(SelectNode::CondIdx) == ElemKind::BoolTy));

  case Kinded::Kind::NotNodeKind:
  case Kinded::Kind::AndNodeKind:
  case Kinded::Kind::OrNodeKind:
  case Kinded::Kind::XorNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::BoolTy});

  case Kinded::Kind::AbsNodeKind:
  case Kinded::Kind::NegNodeKind:
  case Kinded::Kind::FloorNodeKind:
  case Kinded::Kind::CeilNodeKind:
  case Kinded::Kind::RoundNodeKind:
  case Kinded::Kind::SqrtNodeKind:
  case Kinded::Kind::RsqrtNodeKind:
  case Kinded::Kind::ReciprocalNodeKind:
  case Kinded::Kind::SinNodeKind:
  case Kinded::Kind::CosNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::CmpEQNodeKind:
  case Kinded::Kind::CmpNEQNodeKind:
  case Kinded::Kind::CmpLTNodeKind:
  case Kinded::Kind::CmpLTENodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int32ITy,
                ElemKind::Int64ITy},
               {}, {CmpEQNode::ResultIdx}) &&
           (NI.getOutElemTy(CmpEQNode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::IsNaNNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy}, {},
                                                  {IsNaNNode::ResultIdx}) &&
           (NI.getOutElemTy(IsNaNNode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::TopKNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
               {TopKNode::IndicesIdx}) &&
           (NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int64ITy ||
            NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::QuantizeNodeKind:
    return (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::FloatTy) &&
           ((NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int8QTy) ||
            (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int32QTy));

  case Kinded::Kind::DequantizeNodeKind:
    return (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int8QTy) &&
           (NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::FloatTy);

  case Kinded::Kind::SoftMaxNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy},
                                                  {SoftMaxNode::SelectedIdx}) &&
           (NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int64ITy ||
            NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::CrossEntropyLossNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {CrossEntropyLossNode::LabelsIdx}) &&
           (NI.getInElemTy(CrossEntropyLossNode::LabelsIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(CrossEntropyLossNode::LabelsIdx) ==
                ElemKind::Int32ITy);

  case Kinded::Kind::LengthsSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {LengthsSumNode::LengthsIdx}) &&
           (NI.getInElemTy(LengthsSumNode::LengthsIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind:
    return (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::DataIdx) ==
            ElemKind::UInt8FusedQTy) &&
           (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::WeightsIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::OffsetsIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getOutElemTy(EmbeddingBagByteRowwiseOffsetsNode::ResultIdx) ==
            ElemKind::FloatTy);

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    return (NI.getInElemTy(
                FusedRowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx) ==
            ElemKind::UInt8FusedQTy) &&
           (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               WeightsIdx) == ElemKind::FloatTy) &&
           ((NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                IndicesIdx) == ElemKind::Int64ITy ||
             NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                IndicesIdx) == ElemKind::Int32ITy)) &&
           (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               LengthsIdx) == ElemKind::Int32ITy) &&
           (NI.getOutElemTy(
                FusedRowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx) ==
            ElemKind::FloatTy);

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

  case Kinded::Kind::SparseToDenseNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {SparseToDenseNode::IndicesIdx}) &&
           (NI.getInElemTy(SparseToDenseNode::IndicesIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(SparseToDenseNode::IndicesIdx) ==
                ElemKind::Int32ITy);

  case Kinded::Kind::SoftMaxGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {SoftMaxGradNode::SelectedIdx},
               {SoftMaxGradNode::GradOfInputNamedSelectedIdx}) &&
           (NI.getInElemTy(SoftMaxGradNode::SelectedIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(SoftMaxGradNode::SelectedIdx) == ElemKind::Int32ITy);

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

  case Kinded::Kind::TraceEventNodeKind:
    return NI.getInElemTy(TraceEventNode::DataIdx) == ElemKind::Int64ITy;

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
                ElemKind::Int64ITy);

  case Kinded::Kind::AudioSpectrogramNodeKind:
    return NI.getInElemTy(AudioSpectrogramNode::InputIdx) ==
               ElemKind::FloatTy &&
           NI.getOutElemTy(AudioSpectrogramNode::SpectrogramIdx) ==
               ElemKind::FloatTy;

  case Kinded::Kind::MFCCNodeKind:
    return NI.getInElemTy(MFCCNode::SpectrogramIdx) == ElemKind::FloatTy &&
           NI.getOutElemTy(MFCCNode::CoefficientsIdx) == ElemKind::FloatTy;

  case Kinded::Kind::ConvertToNodeKind:
    return ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::Int32ITy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) == ElemKind::FloatTy)) ||
           ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::BoolTy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) == ElemKind::FloatTy)) ||
           ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::Int64ITy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) ==
             ElemKind::Int32ITy)) ||
           ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::Int32ITy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) ==
             ElemKind::Int64ITy)) ||
           ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::FloatTy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) == ElemKind::BoolTy)) ||
           ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::BoolTy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) == ElemKind::Int32ITy));

  default:
    return false;
  }
}

LLVMBackendOptions::LLVMBackendOptions() {
  // Initialize using command-line options by default.
  arch_ = llvmArch;
  target_ = llvmTarget;
  cpu_ = llvmCPU;
  abi_ = llvmABI;
  floatABI_ = floatABI;
  codeModel_ = llvmCodeModel;
  bundleCodeModel_ = llvmBundleCodeModel;
  relocModel_ = llvmRelocModel;
  bundleAPI_ = bundleAPI;
  targetFeatures_.append(llvmTargetFeatures.begin(), llvmTargetFeatures.end());
}

LLVMBackend::LLVMBackend() {}

std::string LLVMBackend::getHostTarget() {
  return llvm::sys::getDefaultTargetTriple();
}

std::string LLVMBackend::getHostCPU() {
  auto cpu_name = llvm::sys::getHostCPUName();
  // Skip avx512 because LLVM does not support it well.
  cpu_name.consume_back("-avx512");
  return cpu_name.str();
}

llvm::SmallVector<std::string, 0> LLVMBackend::getHostFeatures() {
  llvm::SmallVector<std::string, 0> result;
  llvm::StringMap<bool> hostFeatures;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    for (auto &feature : hostFeatures) {
      if (feature.second) {
        llvm::StringRef fn = feature.first();
        // Skip avx512 because LLVM does not support it well.
        if (fn.startswith("avx512")) {
          continue;
        }
        result.push_back(fn);
      }
    }
  }
  return result;
}

/// Emit the entry point for JIT called "jitmain".
/// Function has the following API:
/// int jitmain(uint8_t *baseConstantWeightVars,
///             uint8_t *baseInOutWeightVars,
///             uint8_t *baseActivations);
void LLVMBackend::emitJitMain(LLVMIRGen &irgen) const {
  AllocationsInfo &allocationsInfo = irgen.getAllocationsInfo();
  auto int8PtrTy = llvm::Type::getInt8PtrTy(irgen.getLLVMContext());
  llvm::Type *retTy =
      llvm::Type::getIntNTy(irgen.getLLVMContext(), irgen.getLibjitIntWidth());
  llvm::FunctionType *jitFuncTy =
      llvm::FunctionType::get(retTy, {int8PtrTy, int8PtrTy, int8PtrTy}, false);
  auto *func =
      llvm::Function::Create(jitFuncTy, llvm::Function::ExternalLinkage,
                             "jitmain", &irgen.getModule());
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(irgen.getLLVMContext(), "entry", func);
  llvm::IRBuilder<> builder(entry_bb);
  // Add a provisional terminator to make the function well-formed.
  auto *zero = builder.getIntN(irgen.getLibjitIntWidth(), 0);
  auto *ret = builder.CreateRet(zero);
  builder.SetInsertPoint(ret);

  // Prepare arguments for the "main" function.
  llvm::SmallVector<llvm::Value *, 4> initFunctionCallArgs;
  initFunctionCallArgs.push_back(func->args().begin());
  initFunctionCallArgs.push_back(func->args().begin() + 1);
  initFunctionCallArgs.push_back(func->args().begin() + 2);
  // Now form the offsets array and pass it as the last argument.
  auto offsetsArray =
      irgen.emitConstOffsetsArray(irgen.getBuilder(), allocationsInfo);
  initFunctionCallArgs.push_back(offsetsArray);
  // Invoke the main entry with constant arguments and let LLVM optimizer make
  // use of it.
  auto *entryF = irgen.getModule().getFunction(irgen.getMainEntryName());
  entryF->setLinkage(llvm::Function::InternalLinkage);
  auto *result = irgen.createCall(builder, entryF, initFunctionCallArgs);
  // Terminate the function.
  builder.CreateRet(result);
  // Remove the provisional terminator.
  ret->eraseFromParent();
  // Emit JIT file printer.
  irgen.generateJITFileWriter();
  // Create the debug info for the entry point function.
  irgen.generateFunctionDebugInfo(func);
}

std::unique_ptr<CompiledFunction>
LLVMBackend::compileIR(std::unique_ptr<IRFunction> IR) const {
  auto function = compileIRWithoutConstants(IR.get());
  static_cast<LLVMCompiledFunction *>(function.get())
      ->getRuntimeBundle()
      .collectConstants(IR.get());
  return function;
}

std::unique_ptr<CompiledFunction>
LLVMBackend::compileIRWithoutConstants(IRFunction *IR) const {
  AllocationsInfo allocationsInfo;
  std::unique_ptr<LLVMIRGen> irgen = createIRGen(IR, allocationsInfo);
  llvm::SmallVector<std::string, 8> targetFeatures(llvmTargetFeatures.begin(),
                                                   llvmTargetFeatures.end());
  irgen->initTargetMachine(getOptions());
  irgen->initCodeGen();
  irgen->setIRFunction(IR);
  // Perform the address assignment for activations and WeightVars.
  allocateJITMemory(IR, irgen->getAllocationsInfo());
  // Emit the code for the body of the entry function.
  irgen->performCodeGen();
  // Create the jitmain function to be invoked by JIT.
  emitJitMain(*irgen);
  irgen->finishCodeGen();
  // Hand over the module to JIT for the machine code generation.
  auto JIT = glow::make_unique<llvm::orc::GlowJIT>(irgen->getTargetMachine());
  JIT->addModule(irgen->borrowModule());
  // Build runtimeBundle object containing offsets and allocation sizes.
  MemoryAllocator constantAllocator("ConstantWeights", 0);
  MemoryAllocator placeholderAllocator("Placeholders", 0);
  MemoryAllocator activationsAllocator("Activations", 0);
  auto runtimeInfo = runtime::RuntimeBundle::create(
      *IR, constantAllocator, placeholderAllocator, activationsAllocator);
  return createCompiledFunction(std::move(JIT), std::move(runtimeInfo));
}

Expected<std::unique_ptr<CompiledFunction>>
LLVMBackend::compile(Function *F, const BackendOptions &opts) const {
  TraceInfo traceInfo = buildManualTraceInfo(F);
  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());

  if (opts.autoInstrument) {
    autoInstrument(traceInfo, IR.get());
  }

  std::unique_ptr<CompiledFunction> compiledFunc;
  if (opts.collectConstants) {
    compiledFunc = compileIR(std::move(IR));
  } else {
    compiledFunc = compileIRWithoutConstants(IR.get());
  }

  compiledFunc->setTraceInfo(std::move(traceInfo));
  return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}

void LLVMBackend::save(Function *F, llvm::StringRef outputDir,
                       llvm::StringRef bundleName,
                       llvm::StringRef mainEntryName) const {
  llvm::SmallVector<std::string, 8> targetFeatures(llvmTargetFeatures.begin(),
                                                   llvmTargetFeatures.end());
  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());
  auto bundleSaver = createBundleSaver(*this, outputDir, bundleName);
  bundleSaver->save(mainEntryName, IR.get());
  bundleSaver->produceBundle();
}

void LLVMBackend::saveFunctions(llvm::ArrayRef<BundleEntry> entries,
                                llvm::StringRef outputDir,
                                llvm::StringRef bundleName) const {
  auto bundleSaver = createBundleSaver(*this, outputDir, bundleName);
  std::vector<std::unique_ptr<glow::IRFunction>> irFunctions;
  for (auto &entry : entries) {
    auto IR = generateAndOptimizeIR(entry.func, *this, shouldShareBuffers());
    bundleSaver->save(entry.name, IR.get());
    irFunctions.emplace_back(std::move(IR));
  }
  bundleSaver->produceBundle();
}

std::unique_ptr<BundleSaver>
LLVMBackend::createBundleSaver(const LLVMBackend &llvmBackend,
                               llvm::StringRef outputDir,
                               llvm::StringRef bundleName) const {
  return glow::make_unique<BundleSaver>(llvmBackend, outputDir, bundleName);
}
