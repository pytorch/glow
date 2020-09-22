/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "NNPI.h"
#include "DebugMacros.h"
#include "Importer.h"
#include "InferenceContext.h"
#include "NNPICompiledFunction.h"
#include "NNPIDeviceManager.h"
#include "NNPIUtils.h"
#include "glow/Flags/Flags.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPassPipeline.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Optimizer/Lower/Lower.h"

#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <unordered_map>
#include <unordered_set>

using namespace glow;

NNPIBackendOptions NNPIBackend::backendOptions_;
NNPIAdapterContainer NNPIBackend::adapter_;

unsigned NNPIBackend::numDevices() {
  if (!backendOptions_.inferOnDevice) {
    // Will return 1 device (for ICE-Ref)
    return 1;
  }
  NNPIAdapter adapter = NNPI_INVALID_NNPIHANDLE;
  NNPIAdapterInfo adapterInfo;
  memset(&adapterInfo, 0, sizeof(adapterInfo));
  LOG_AND_RETURN_IF_NOT(
      ERROR, nnpiAdapterCreate(nullptr, &adapter) == NNPI_INF_NO_ERROR,
      "Failed to create NNPI Adapter.", 0);
  LOG_AND_RETURN_IF_NOT(
      ERROR, nnpiAdapterGetInfo(adapter, &adapterInfo) == NNPI_INF_NO_ERROR,
      "Failed get device info.", 0);
  LOG_NNPI_INF_IF_ERROR(nnpiAdapterDestroy(adapter),
                        "Failed to destroy NNPI Adapter");
  return adapterInfo.numDevices;
}

/// \returns whether \p type is 2 dimensional and unary. Usually the data input
/// of SparseLengths(Weighted)Sum is passed in here.
static bool isUnaryLookup(TypeRef type) {
  if (type->dims().size() != 2) {
    return false;
  }
  return type->dims()[1] == 1;
}

bool NNPIBackend::acceptForExecution(const NodeInfo &NI) const {
  if (!isOpSupported(NI)) {
    return false;
  }

  // For performance reasons, only accept for execution SLS/SLWS with non-unary
  // data inputs.
  switch (NI.getKind()) {
  case Kinded::Kind::SparseLengthsSumNodeKind:
    return GlowNNPIAcceptUnarySLS ||
           !isUnaryLookup(NI.getInTy(SparseLengthsSumNode::DataIdx));
  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return GlowNNPIAcceptUnarySLS ||
           !isUnaryLookup(NI.getInTy(SparseLengthsWeightedSumNode::DataIdx));

  default:
    return true;
  }
}

/// \returns whether SLS indices type is valid for NNPI.
static bool isSLSIndicesValid(TypeRef type) {
  // Don't support more than 64k indices.
  return type->dims().size() == 1 && type->dims()[0] < (1 << 16);
}

bool NNPIBackend::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {
  // General math fp32/fp16/i8.
  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MinNodeKind:
  case Kinded::Kind::PowNodeKind:
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ReplaceNaNNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
  case Kinded::Kind::BatchedReduceMeanNodeKind:
  case Kinded::Kind::BatchedReduceMinNodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
  case Kinded::Kind::BatchedAddNodeKind:
  case Kinded::Kind::BatchedMulNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::ExpNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});
  case Kinded::Kind::ModuloNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::LayerNormalizationNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});
  case Kinded::Kind::GeluNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty});
  case Kinded::Kind::BatchNormalizationNodeKind: {
    auto elemType = NI.getInElemTy(BatchNormalizationNode::InputIdx);
    bool isSup =
        (elemType == ElemKind::Int8QTy || elemType == ElemKind::FloatTy ||
         elemType == ElemKind::Float16Ty);

    isSup = isSup && NI.allInputsAndOutputsHaveSameElemKind(
                         {ElemKind::FloatTy, ElemKind::Float16Ty},
                         {BatchNormalizationNode::InputIdx});

    return isSup;
  }

  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::AdaptiveAvgPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});

  case Kinded::Kind::SwishNodeKind:
  case Kinded::Kind::BatchMatMulNodeKind:
  case Kinded::Kind::PReluNodeKind:
  case Kinded::Kind::ClipNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int8QTy, ElemKind::Float16Ty});

  case Kinded::Kind::DivNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int64ITy});

  // Data transfer fp32/fp16/i8/i32/i64/bool.
  case Kinded::Kind::SaveNodeKind:
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::TileNodeKind:
  case Kinded::Kind::TransposeNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy, ElemKind::BoolTy});

  case Kinded::Kind::ConvolutionNodeKind:
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    }
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy},
                                                  {ConvolutionNode::BiasIdx}) &&
           ((NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy) ||
            (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::FloatTy));

  case Kinded::Kind::Convolution3DNodeKind:
    if (!NI.getInTy(Convolution3DNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    }
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::Int8QTy}, {Convolution3DNode::BiasIdx}) &&
           ((NI.getInElemTy(Convolution3DNode::BiasIdx) ==
             ElemKind::Int32QTy) ||
            (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::FloatTy));

  case Kinded::Kind::QuantizeNodeKind:
    return (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::FloatTy ||
            NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::Float16Ty) &&
           (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int8QTy);

  case Kinded::Kind::DequantizeNodeKind:
    return (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int8QTy) &&
           (NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::FloatTy ||
            NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::Float16Ty);

  case Kinded::Kind::RescaleQuantizedNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy});

  case Kinded::Kind::ConvertToNodeKind: {
    auto isConversionSupportedFor = [](ElemKind kind) {
      switch (kind) {
      case ElemKind::FloatTy:
      case ElemKind::Float16Ty:
      case ElemKind::Int32ITy:
      case ElemKind::Int64ITy:
        return true;
      default:
        return false;
      }
    };
    return isConversionSupportedFor(NI.getInElemTy(ConvertToNode::InputIdx)) &&
           isConversionSupportedFor(NI.getOutElemTy(ConvertToNode::ResultIdx));
  }

  case Kinded::Kind::FullyConnectedNodeKind:
    if (!NI.getInTy(FullyConnectedNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    }
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::Int8QTy}, {FullyConnectedNode::BiasIdx}) &&
           ((NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
             ElemKind::Int32QTy) ||
            (NI.getInElemTy(FullyConnectedNode::BiasIdx) == ElemKind::FloatTy));

  case Kinded::Kind::MaxPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
               {MaxPoolNode::ArgmaxIdx}) &&
           (NI.getOutElemTy(MaxPoolNode::ArgmaxIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::TopKNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
               {TopKNode::IndicesIdx}) &&
           (NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::GatherNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int64ITy,
                ElemKind::Int8QTy},
               {GatherNode::IndicesIdx}) &&
           ((NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int32ITy) ||
            (NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int64ITy));

  case Kinded::Kind::GatherRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::Int32ITy, ElemKind::Int64ITy},
               {GatherRangesNode::DataIdx}, {GatherRangesNode::OutputIdx}) &&
           ((NI.getInElemTy(GatherRangesNode::DataIdx) == ElemKind::FloatTy) ||
            (NI.getInElemTy(GatherRangesNode::DataIdx) ==
             ElemKind::Float16Ty) ||
            (NI.getInElemTy(GatherRangesNode::DataIdx) == ElemKind::Int8QTy) ||
            (NI.getInElemTy(GatherRangesNode::DataIdx) == ElemKind::Int32ITy) ||
            (NI.getInElemTy(GatherRangesNode::DataIdx) ==
             ElemKind::Int64ITy)) &&
           (NI.getOutElemTy(GatherRangesNode::OutputIdx) ==
            NI.getInElemTy(GatherRangesNode::DataIdx));

  case Kinded::Kind::SliceNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int64ITy});

  case Kinded::Kind::CmpLTENodeKind:
  case Kinded::Kind::CmpEQNodeKind:
  case Kinded::Kind::CmpLTNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                ElemKind::Int32ITy, ElemKind::Int64ITy},
               {}, {CmpEQNode::ResultIdx}) &&
           (NI.getOutElemTy(CmpEQNode::ResultIdx) == ElemKind::BoolTy);
  case Kinded::Kind::SelectNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
               {SelectNode::CondIdx}) &&
           (NI.getInElemTy(SelectNode::CondIdx) == ElemKind::BoolTy);

  case Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind:
    return (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::InputIdx) ==
            ElemKind::Int8QTy) &&
           (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::WeightsIdx) ==
            ElemKind::Int8QTy) &&
           (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::ScalesIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::OffsetsIdx) ==
            ElemKind::Int32ITy) &&
           ((NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::BiasIdx) ==
             ElemKind::Int32QTy) ||
            (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::BiasIdx) ==
             ElemKind::FloatTy)) &&
           (NI.getOutElemTy(RowwiseQuantizedFullyConnectedNode::ResultIdx) ==
            ElemKind::Int8QTy);

  case Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind:
    return (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::InputIdx) ==
            ElemKind::Int8QTy) &&
           (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::FilterIdx) ==
            ElemKind::Int8QTy) &&
           ((NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::BiasIdx) ==
             ElemKind::Int32QTy) ||
            (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::BiasIdx) ==
             ElemKind::FloatTy)) &&
           (NI.getInElemTy(
                ChannelwiseQuantizedConvolutionNode::FilterScalesIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(
                ChannelwiseQuantizedConvolutionNode::FilterOffsetsIdx) ==

            ElemKind::Int32ITy) &&
           (NI.getOutElemTy(ChannelwiseQuantizedConvolutionNode::ResultIdx) ==
            ElemKind::Int8QTy);

  case Kinded::Kind::SparseLengthsSumNodeKind:
    return isSLSIndicesValid(NI.getInTy(SparseLengthsSumNode::IndicesIdx)) &&
           NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
               {SparseLengthsSumNode::IndicesIdx,
                SparseLengthsSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsSumNode::IndicesIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(SparseLengthsSumNode::IndicesIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(SparseLengthsSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);
  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return isSLSIndicesValid(
               NI.getInTy(SparseLengthsWeightedSumNode::IndicesIdx)) &&
           NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
               {SparseLengthsWeightedSumNode::IndicesIdx,
                SparseLengthsWeightedSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
                ElemKind::Int64ITy ||
            NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
                ElemKind::Int32ITy) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::EmbeddingBagNodeKind:
    return isSLSIndicesValid(NI.getInTy(EmbeddingBagNode::IndicesIdx)) &&
           NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
               {EmbeddingBagNode::IndicesIdx, EmbeddingBagNode::OffsetsIdx}) &&
           (NI.getInElemTy(EmbeddingBagNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(EmbeddingBagNode::OffsetsIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind: {
    auto dataK = NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::DataIdx);
    auto offsetsK =
        NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::OffsetsIdx);
    auto indicesK =
        NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::IndicesIdx);
    auto resultK =
        NI.getOutElemTy(EmbeddingBagByteRowwiseOffsetsNode::ResultIdx);
    return isSLSIndicesValid(
               NI.getInTy(EmbeddingBagByteRowwiseOffsetsNode::IndicesIdx)) &&
           (dataK == ElemKind::UInt8FusedQTy ||
            dataK == ElemKind::UInt8FusedFP16QTy ||
            dataK == ElemKind::UInt4FusedFP16QTy) &&
           (resultK == ElemKind::FloatTy || resultK == ElemKind::Float16Ty) &&
           (indicesK == ElemKind::Int64ITy) && (offsetsK == ElemKind::Int64ITy);
  }

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind: {
    auto dataK =
        NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsSumNode::DataIdx);
    auto lengthsK =
        NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsSumNode::LengthsIdx);
    auto indicesK =
        NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsSumNode::IndicesIdx);
    auto resultK =
        NI.getOutElemTy(FusedRowwiseQuantizedSparseLengthsSumNode::ResultIdx);
    return isSLSIndicesValid(NI.getInTy(
               FusedRowwiseQuantizedSparseLengthsSumNode::IndicesIdx)) &&
           (dataK == ElemKind::UInt8FusedQTy ||
            dataK == ElemKind::UInt8FusedFP16QTy ||
            dataK == ElemKind::UInt4FusedFP16QTy) &&
           (resultK == ElemKind::FloatTy || resultK == ElemKind::Float16Ty) &&
           (indicesK == ElemKind::Int64ITy || indicesK == ElemKind::Int32ITy) &&
           (lengthsK == ElemKind::Int32ITy);
  }

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind: {
    auto dataK = NI.getInElemTy(
        FusedRowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx);
    auto weightsK = NI.getInElemTy(
        FusedRowwiseQuantizedSparseLengthsWeightedSumNode::WeightsIdx);
    auto lengthsK = NI.getInElemTy(
        FusedRowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx);
    auto indicesK = NI.getInElemTy(
        FusedRowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx);
    auto resultK = NI.getOutElemTy(
        FusedRowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx);
    return isSLSIndicesValid(
               NI.getInTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                              IndicesIdx)) &&
           (dataK == ElemKind::UInt8FusedQTy ||
            dataK == ElemKind::UInt8FusedFP16QTy ||
            dataK == ElemKind::UInt4FusedFP16QTy) &&
           (weightsK == ElemKind::FloatTy || weightsK == ElemKind::Float16Ty) &&
           (resultK == ElemKind::FloatTy || resultK == ElemKind::Float16Ty) &&
           (indicesK == ElemKind::Int64ITy || indicesK == ElemKind::Int32ITy) &&
           (lengthsK == ElemKind::Int32ITy);
  }

  case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    return isSLSIndicesValid(NI.getInTy(
               RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx)) &&
           NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty},
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

  case Kinded::Kind::SparseToDenseNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {SparseToDenseNode::IndicesIdx}) &&
           (NI.getInElemTy(SparseToDenseNode::IndicesIdx) ==
            ElemKind::Int64ITy);

  case Kinded::Kind::SoftMaxNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
               {SoftMaxNode::SelectedIdx}) &&
           (NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::LengthsRangeFillNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});

  case Kinded::Kind::BatchOneHotNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
                ElemKind::Int32ITy, ElemKind::Int64ITy},
               {BatchOneHotNode::LengthsIdx}) &&
           (NI.getInElemTy(BatchOneHotNode::LengthsIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::NNPICustomDSPNodeKind:
  case Kinded::Kind::NNPICustomIANodeKind:
    return true;

  case Kinded::Kind::SpaceToDepthNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::ArgMaxNodeKind:
    return (NI.getOutElemTy(ArgMaxNode::ResultIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::LogitNodeKind: {
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty});
  }

  default:
    llvm::outs() << "Unsupported op:\n" << NI.getDebugDesc() << "\n";
    return false;
  }

  return false;
}

bool NNPIBackend::shouldLower(const Node *N) const {
  switch (N->getKind()) {
  case Kinded::Kind::ConvolutionNodeKind:
    return isConvolutionSameAsFullyConnected(llvm::cast<ConvolutionNode>(N),
                                             /* enforceInput1x1*/ true);
  case Kinded::Kind::ClipNodeKind:
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::SoftMaxNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::SwishNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::Convolution3DNodeKind:
  case Kinded::Kind::TileNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::ReplaceNaNNodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
  case Kinded::Kind::BatchedReduceMeanNodeKind:
  case Kinded::Kind::BatchedReduceMinNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::BatchMatMulNodeKind:
  case Kinded::Kind::BatchNormalizationNodeKind:
  case Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind:
  case Kinded::Kind::EmbeddingBagNodeKind:
  case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind:
  case Kinded::Kind::LayerNormalizationNodeKind:
  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind:
  case Kinded::Kind::PReluNodeKind:
  case Kinded::Kind::GeluNodeKind:
  case Kinded::Kind::LogitNodeKind:
  case Kinded::Kind::SparseLengthsSumNodeKind:
  case Kinded::Kind::BatchedMulNodeKind:
    return false;
  default:
    return true;
  }
  return true;
}

runtime::DeviceManager *
NNPIBackend::createDeviceManager(const runtime::DeviceConfig &deviceConfig) {
  return createNNPIDeviceManager(deviceConfig, &adapter_);
}

/// Setup basic parallelization in \p numChunks and \p parOpts for \p F, where
/// every node may be split \p numParallelChunks times.
static void setupBasicParallelizationConfigs(
    Function *F, llvm::DenseMap<Node *, size_t> &numChunks,
    llvm::DenseMap<Node *, ParallelTransformKind> &parOpts,
    int32_t numParallelChunks) {
  // Process nodes PostOrder so we always process inputs before outputs of any
  // Node, so parallelization can be based on if a parent is parallelized.
  GraphPostOrderVisitor visitor(*F);
  for (auto *node : visitor.getPostOrder()) {
    // Find all FC layers to split
    if (auto *FC = llvm::dyn_cast<FullyConnectedNode>(node)) {
      size_t K = FC->getWeights().dims()[1];
      if (K >= 512) {
        parOpts[FC] = ParallelTransformKind::Model;
        numChunks[FC] =
            std::min((size_t)numParallelChunks, FC->getResult().dims()[1]);
        continue;
      }
      size_t M = FC->getInput().dims()[0];
      if (M >= 256) {
        parOpts[FC] = ParallelTransformKind::Data;
        numChunks[FC] =
            std::min((size_t)numParallelChunks, FC->getResult().dims()[0]);
        continue;
      }
    }

    // Relu parallelization.
    // If a Relu follows FC, mirror FC split so that they fuse.
    // Otherwise, use data parallelism.
    if (auto *R = llvm::dyn_cast<ReluNode>(node)) {
      // For Relus that arent preceded by FC, do data parallelism if the input
      // was parallelized.
      Node *inputNode = R->getInput().getNode();
      auto FC = llvm::dyn_cast<FullyConnectedNode>(inputNode);
      if (!FC) {
        if (numChunks.find(inputNode) != numChunks.end() &&
            parOpts.find(inputNode) != parOpts.end()) {
          parOpts[R] = ParallelTransformKind::Data;
          numChunks[R] =
              std::min((size_t)numParallelChunks, R->getResult().dims()[0]);
        }
        continue;
      }

      // Otherwise, mirror FC split.
      if (R->getInput().dims().size() < 2) {
        continue;
      }
      size_t K = R->getInput().dims()[1];
      if (K >= 512) {
        parOpts[R] = ParallelTransformKind::Model;
        numChunks[R] =
            std::min((size_t)numParallelChunks, R->getResult().dims()[1]);
        continue;
      }
      size_t M = R->getInput().dims()[0];
      if (M >= 256) {
        parOpts[R] = ParallelTransformKind::Data;
        numChunks[R] =
            std::min((size_t)numParallelChunks, R->getResult().dims()[0]);
        continue;
      }
    }

    // Split Gelu layers in data parallel fashion
    if (auto *GL = llvm::dyn_cast<GeluNode>(node)) {
      size_t M = GL->getInput().dims()[0];
      if (M >= numParallelChunks) {
        parOpts[GL] = ParallelTransformKind::Data;
        numChunks[GL] = numParallelChunks;
        continue;
      }
    }

    // Split transpose layers in data parallel fashion
    if (auto *TP = llvm::dyn_cast<TransposeNode>(node)) {
      parOpts[TP] = ParallelTransformKind::Data;
      numChunks[TP] =
          std::min((size_t)numParallelChunks, TP->getResult().dims()[0]);
      continue;
    }

    // Split Quantize layers in data parallel fashion
    if (auto *QN = llvm::dyn_cast<QuantizeNode>(node)) {
      parOpts[QN] = ParallelTransformKind::Data;
      numChunks[QN] =
          std::min((size_t)numParallelChunks, QN->getResult().dims()[0]);
      continue;
    }

    // Split Dequantize layers in data parallel fashion
    if (auto *DQN = llvm::dyn_cast<DequantizeNode>(node)) {
      parOpts[DQN] = ParallelTransformKind::Data;
      numChunks[DQN] =
          std::min((size_t)numParallelChunks, DQN->getResult().dims()[0]);
      continue;
    }

    // Split Tile layers
    if (auto *TN = llvm::dyn_cast<TileNode>(node)) {
      if (TN->getAxis() == 0) {
        if (TN->getInput().dims().size() < 2) {
          continue;
        }
        size_t N = TN->getInput().dims()[1];
        if (N < 256) {
          continue;
        }
        parOpts[TN] = ParallelTransformKind::Model;
        numChunks[TN] =
            std::min((size_t)numParallelChunks, TN->getResult().dims()[1]);
      } else if (TN->getAxis() == 1) {
        if (TN->getInput().dims().size() < 2) {
          continue;
        }
        size_t M = TN->getInput().dims()[0];
        if (M < 256) {
          continue;
        }
        parOpts[TN] = ParallelTransformKind::Data;
        numChunks[TN] =
            std::min((size_t)numParallelChunks, TN->getResult().dims()[0]);
      }
      continue;
    }

    // Split BatchedReduceAdd layers
    if (auto *BR = llvm::dyn_cast<BatchedReduceAddNode>(node)) {
      size_t N = BR->getResult().dims()[0];
      if (N < 64) {
        continue;
      }
      parOpts[BR] = ParallelTransformKind::Data;
      numChunks[BR] =
          std::min((size_t)numParallelChunks, BR->getResult().dims()[0]);
      continue;
    }

    // Split LayerNorm layers in data parallel fashion
    if (auto *LN = llvm::dyn_cast<LayerNormalizationNode>(node)) {
      if (LN->getInput().dims().size() < 2) {
        continue;
      }
      size_t NIdx = getMaxDimOtherThanBatch(LN->getInput().dims());
      size_t N = LN->getInput().dims()[NIdx];
      if (N < 1024) {
        continue;
      }
      parOpts[LN] = ParallelTransformKind::Data;
      numChunks[LN] =
          std::min((size_t)numParallelChunks, LN->getResult().dims()[0]);
      continue;
    }

    // Split BMM layers in data parallel fashion
    if (auto *BMM = llvm::dyn_cast<BatchMatMulNode>(node)) {
      parOpts[BMM] = ParallelTransformKind::Data;
      numChunks[BMM] =
          std::min((size_t)numParallelChunks, BMM->getResult().dims()[0]);
      continue;
    }

    // Split MatMul layers in Model parallel fashion
    if (auto *MM = llvm::dyn_cast<MatMulNode>(node)) {
      parOpts[MM] = ParallelTransformKind::Model;
      numChunks[MM] =
          std::min((size_t)numParallelChunks, MM->getResult().dims()[1]);
      continue;
    }

    // Split Tanh layers in data parallel fashion
    if (auto *TH = llvm::dyn_cast<TanhNode>(node)) {
      if (TH->getInput().dims().size() < 2) {
        continue;
      }
      if (TH->getInput().dims().size() == 2) {
        size_t N = TH->getInput().dims()[1];
        if (N < 1792) {
          continue;
        }
        parOpts[TH] = ParallelTransformKind::Data;
        numChunks[TH] =
            std::min((size_t)numParallelChunks, TH->getResult().dims()[0]);
        continue;
      } else if (TH->getInput().dims().size() == 3) {
        size_t N = TH->getInput().dims()[1];
        size_t K = TH->getInput().dims()[2];
        if (N * K < 2048) {
          continue;
        }
        parOpts[TH] = ParallelTransformKind::Data;
        numChunks[TH] =
            std::min((size_t)numParallelChunks, TH->getResult().dims()[0]);
        continue;
      }
    }

    // Split Add layers in data parallel fashion
    if (auto *AD = llvm::dyn_cast<AddNode>(node)) {
      if (AD->getLHS().dims().size() < 2) {
        continue;
      }
      if (AD->getLHS().dims().size() == 2) {
        size_t N = AD->getLHS().dims()[1];
        if (N < 1792) {
          continue;
        }
        parOpts[AD] = ParallelTransformKind::Data;
        numChunks[AD] =
            std::min((size_t)numParallelChunks, AD->getResult().dims()[0]);
        continue;
      } else if (AD->getLHS().dims().size() == 3) {
        size_t N = AD->getLHS().dims()[1];
        size_t K = AD->getLHS().dims()[2];
        if (N * K < 2048) {
          continue;
        }
        parOpts[AD] = ParallelTransformKind::Data;
        numChunks[AD] =
            std::min((size_t)numParallelChunks, AD->getResult().dims()[0]);
        continue;
      }
    }

    // Split Swish layers in data parallel fashion
    if (auto *SW = llvm::dyn_cast<SwishNode>(node)) {
      if (SW->getInput().dims().size() < 2) {
        continue;
      }
      size_t N = SW->getInput().dims()[1];
      if (N < 512) {
        continue;
      }
      parOpts[SW] = ParallelTransformKind::Data;
      numChunks[SW] =
          std::min((size_t)numParallelChunks, SW->getResult().dims()[0]);
      continue;
    }

    // Split Mul layers in data parallel fashion
    if (auto *M = llvm::dyn_cast<MulNode>(node)) {
      if (M->getLHS().dims().size() < 2) {
        continue;
      }
      size_t N = M->getLHS().dims()[1];
      if (N < 512) {
        continue;
      }
      parOpts[M] = ParallelTransformKind::Data;
      numChunks[M] =
          std::min((size_t)numParallelChunks, M->getResult().dims()[0]);
      continue;
    }

    // Split Sigmoid layers in data parallel fashion
    if (auto *S = llvm::dyn_cast<SigmoidNode>(node)) {
      if (S->getInput().dims().size() < 2) {
        continue;
      }
      size_t N = S->getInput().dims()[1];
      if (N < 512) {
        continue;
      }
      parOpts[S] = ParallelTransformKind::Data;
      numChunks[S] =
          std::min((size_t)numParallelChunks, S->getResult().dims()[0]);
      continue;
    }

    // Split Softmax layers in data parallel fashion
    if (auto *SM = llvm::dyn_cast<SoftMaxNode>(node)) {
      if (SM->getInput().dims().size() < 2) {
        continue;
      }
      size_t M = SM->getInput().dims()[0];
      size_t N = SM->getInput().dims()[1];
      if (N < 32 || M < 128) {
        continue;
      }
      parOpts[SM] = ParallelTransformKind::Data;
      numChunks[SM] =
          std::min((size_t)numParallelChunks, SM->getResult().dims()[0]);
      continue;
    }

    // Clip parallelization.
    // If a Clip follows a parallel op, mirror that.
    if (auto *C = llvm::dyn_cast<ClipNode>(node)) {
      Node *inputNode = C->getInput().getNode();
      if (numChunks.find(inputNode) != numChunks.end() &&
          parOpts.find(inputNode) != parOpts.end()) {
        parOpts[C] = parOpts[inputNode];
        if (parOpts[C] == ParallelTransformKind::Data) {
          numChunks[C] =
              std::min((size_t)numChunks[inputNode], C->getResult().dims()[0]);
        } else {
          numChunks[C] =
              std::min((size_t)numChunks[inputNode], C->getResult().dims()[1]);
        }
      }
      continue;
    }
  }
}

/// If we've done some paralleization specified in \p replacedMap then
/// validate that the parallelization matches with the specified previous
/// NodeInfo. \returns whether any validation error is found.
static Error validateBackendSpecificNodeInfo(
    Function *F, const std::unordered_map<Node *, ConcatNode *> &replacedMap,
    BackendSpecificNodeInfo &backendSpecificNodeInfo) {
  // Build a map from replaced names of a Node to the ConcatNode that replaced
  // it. Used later for cleaning up extraEdges of split Nodes.
  llvm::StringMap<const ConcatNode *> nameToReplacementMap;

  auto funNodeInfoIt = backendSpecificNodeInfo.find(F);
  RETURN_ERR_IF_NOT(funNodeInfoIt != backendSpecificNodeInfo.end(),
                    "Must have backend-specific info for this Function.");
  auto &currFunInfo = funNodeInfoIt->second;

  for (const auto &replacedPair : replacedMap) {
    const Node *replacedNode = replacedPair.first;
    nameToReplacementMap[replacedNode->getName().str()] = replacedPair.second;

    RETURN_ERR_IF_NOT(
        replacedNode->getNumUsers() == 0,
        "Replaced Node should no longer be used in the Function.");

    auto curNodeInfoIt = currFunInfo.find(replacedNode);
    RETURN_ERR_IF_NOT(
        curNodeInfoIt != currFunInfo.end(),
        "Only should have parallelized if backendSpecificNodeInfo said so.");
    auto &nodeInfo = curNodeInfoIt->second;

    // Validate that the number of nodes concatenated together is equal to the
    // parallelization factor specified in numParallelChunks.
    const ConcatNode *CN = replacedPair.second;
    auto numParChunksIt = nodeInfo.find(numParallelChunksKey);
    RETURN_ERR_IF_NOT(numParChunksIt != nodeInfo.end(),
                      "Must have corresponding " +
                          std::string(numParallelChunksKey) +
                          " for any Node that was parallelized.");
    RETURN_ERR_IF_NOT(numParChunksIt->second.size() == 1,
                      "Expected a single value for numParallelChunks");
    int numParChunksVal;
    ASSIGN_VALUE_OR_RETURN_ERR(numParChunksVal,
                               getIntFromStr(numParChunksIt->second.front()));
    // It is possible that the number of inputs is less than numParChunksVal
    // due to alignment
    RETURN_ERR_IF_NOT(numParChunksVal >= CN->getInputs().size(),
                      "Node not split the expected number of times.");

    // Now we can erase this Node's info from currFunInfo because it has been
    // replaced and will be DCE'd soon.
    currFunInfo.erase(curNodeInfoIt);
  }

  // No parallelization or placement hints should be present at this point
  for (auto &node : F->getNodes()) {
    auto curNodeInfoIt = currFunInfo.find(&node);
    if (curNodeInfoIt == currFunInfo.end()) {
      continue;
    }
    auto &nodeInfo = curNodeInfoIt->second;

    RETURN_ERR_IF_NOT(!nodeInfo.count(parallelTransformKindKey),
                      strFormat("Node %s should not have a "
                                "parallelTransformKind after parallelization",
                                node.getName().str().c_str()));

    RETURN_ERR_IF_NOT(
        !nodeInfo.count(numParallelChunksKey),
        strFormat(
            "Node %s should not have a numParallelChunks after parallelization",
            node.getName().str().c_str()));

    RETURN_ERR_IF_NOT(
        !nodeInfo.count(coreAssignmentsKey),
        strFormat(
            "Node %s should not have a coreAssignments prior to placement",
            node.getName().str().c_str()));

    RETURN_ERR_IF_NOT(!nodeInfo.count(coreAssignmentsSuffixKey),
                      strFormat("Node %s should not have a "
                                "coreAssignmentsSuffix prior to placement",
                                node.getName().str().c_str()));

    RETURN_ERR_IF_NOT(
        !nodeInfo.count(extraEdgesTargetNameKey),
        strFormat(
            "Node %s should not have a extraEdgesTargetName prior to placement",
            node.getName().str().c_str()));

    RETURN_ERR_IF_NOT(!nodeInfo.count(extraEdgesTargetSuffixKey),
                      strFormat("Node %s should not have a "
                                "extraEdgesTargetSuffix prior to placement",
                                node.getName().str().c_str()));

    RETURN_ERR_IF_NOT(!nodeInfo.count(extraEdgesSourceSuffixKey),
                      strFormat("Node %s should not have a "
                                "extraEdgesSourceSuffix prior to placement",
                                node.getName().str().c_str()));
  }
  return Error::success();
}

/// Sets up \p partOpts and \p numChunks based on the spec found in \p
/// setupPerOpParallelizationConfigs for all Nodes in \p F. \returns if there
/// was an error while parsing \p backendSpecificNodeInfo.
static Error setupPerNodeParallelizationConfigs(
    Function *F, llvm::DenseMap<Node *, size_t> &numOfChunks,
    llvm::DenseMap<Node *, ParallelTransformKind> &parOpts,
    const BackendSpecificNodeInfo &backendSpecificNodeInfo) {
  auto funNodeInfoIt = backendSpecificNodeInfo.find(F);
  RETURN_ERR_IF_NOT(funNodeInfoIt != backendSpecificNodeInfo.end(),
                    "Must have backend-specific info for this Function.");
  auto &currFunInfo = funNodeInfoIt->second;

  for (auto &node : F->getNodes()) {
    auto curNodeInfoIt = currFunInfo.find(&node);
    if (curNodeInfoIt == currFunInfo.end()) {
      continue;
    }
    auto &nodeInfo = curNodeInfoIt->second;

    // Setup parallelTransformKind. It can be specified without
    // numParallelChunks only if it is set to "None".
    auto parTransformKindIt = nodeInfo.find(parallelTransformKindKey);
    if (parTransformKindIt == nodeInfo.end()) {
      continue;
    }
    RETURN_ERR_IF_NOT(parTransformKindIt->second.size() == 1,
                      "Expected single value for " +
                          std::string(parallelTransformKindKey));
    const std::string &pKindStr = parTransformKindIt->second.front();
    ParallelTransformKind pKind;
    if (pKindStr == "Data") {
      pKind = ParallelTransformKind::Data;
    } else if (pKindStr == "Model") {
      pKind = ParallelTransformKind::Model;
    } else if (pKindStr == "None") {
      pKind = ParallelTransformKind::None;
    } else {
      return MAKE_ERR(std::string(parallelTransformKindKey) + " " + pKindStr +
                      " not supported.");
    }
    if (pKind == ParallelTransformKind::None) {
      continue;
    }

    // Setup numParallelChunks. It must be specified at this point, as we have a
    // valid parallelTransformKind found above.
    auto numParChunksIt = nodeInfo.find(numParallelChunksKey);
    RETURN_ERR_IF_NOT(numParChunksIt != nodeInfo.end(),
                      std::string(numParallelChunksKey) + " and " +
                          std::string(parallelTransformKindKey) +
                          " must be specified together.");
    RETURN_ERR_IF_NOT(numParChunksIt->second.size() == 1,
                      "Expected single value for " +
                          std::string(numParallelChunksKey));

    int numChunks;
    ASSIGN_VALUE_OR_RETURN_ERR(numChunks,
                               getIntFromStr(numParChunksIt->second.front()));
    RETURN_ERR_IF_NOT(numChunks > 1, "numChunks must be > 1.");
    numOfChunks[&node] = numChunks;
    parOpts[&node] = pKind;
  }

  return Error::success();
}

/// Parallelize \p F. If this Function has backendSpecificNodeInfo in \p opts
/// then this parallelization is done based on that. Else perform basic
/// parallelization according to either GlowNNPINumParallelChunks, or if not
/// specified then NNPINumParallelChunks found in
/// backendOpts.backendSpecificOpts from \p opts. \returns whether \p F was
/// modified.
static Expected<bool> parallelizeFunction(Function *F, BackendOptions &opts) {
  // Split FC layers in model/data parallel fashion
  llvm::DenseMap<Node *, size_t> numChunks;
  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;

  int32_t defaultNumParallelChunks = 1;

  const bool usePerNodeParallelizationSpec =
      opts.backendSpecificNodeInfo.find(F) !=
      opts.backendSpecificNodeInfo.end();
  if (usePerNodeParallelizationSpec) {
    // Only parallelize based on what is explicitly specified.
    RETURN_IF_ERR(setupPerNodeParallelizationConfigs(
        F, numChunks, parOpts, opts.backendSpecificNodeInfo));
  } else {
    // Check for basic parallelization based on specified degree of parallelism.
    defaultNumParallelChunks = glow::onnxifi::GlowNNPINumParallelChunks;

    // GlowNNPINumParallelChunks set via flags takes precedence over backend
    // options in cctx.
    if (!defaultNumParallelChunks) {
      auto it =
          opts.backendSpecificOpts.find(std::string("NNPINumParallelChunks"));
      if (it != opts.backendSpecificOpts.end()) {
        ASSIGN_VALUE_OR_RETURN_ERR(defaultNumParallelChunks,
                                   getIntFromStr(it->second));
      }
    }

    // If there's no parallelization to perform then exit early.
    if (defaultNumParallelChunks <= 1) {
      return false;
    }
    setupBasicParallelizationConfigs(F, numChunks, parOpts,
                                     defaultNumParallelChunks);
  }

  RETURN_ERR_IF_NOT(numChunks.size() == parOpts.size(),
                    "Require that numChunks and parOpts have same size.");

  // No parallelization to do, so return early.
  if (numChunks.size() == 0) {
    return false;
  }

  int32_t defaultModelParallelSplitAlignment =
      glow::onnxifi::GlowNNPIModelParallelSplitAlignment;

  // GlowNNPIModelParallelSplitAlignment set via flags takes precedence over
  // backend options in cctx.
  if (defaultModelParallelSplitAlignment == 1) {
    auto it = opts.backendSpecificOpts.find(
        std::string("NNPIModelParallelSplitAlignment"));
    if (it != opts.backendSpecificOpts.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(defaultModelParallelSplitAlignment,
                                 getIntFromStr(it->second));
    }
  }

  // Now actually do the parallelization.
  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_RETURN_ERR(
      replacedMap,
      parallelizeOps(F, numChunks, parOpts, defaultNumParallelChunks,
                     defaultModelParallelSplitAlignment));

  RETURN_ERR_IF_NOT(numChunks.size() == replacedMap.size(),
                    "Expected that numChunks and replacedMap have same size.");

  if (usePerNodeParallelizationSpec) {
    // If parallelization was based on backend-specific node info then
    // validate the new nodes that were added.
    RETURN_IF_ERR(validateBackendSpecificNodeInfo(
        F, replacedMap, opts.backendSpecificNodeInfo));
  }

  return true;
}

Expected<std::unique_ptr<CompiledFunction>>
NNPIBackend::compile(Function *F, const BackendOptions &opts) const {
  std::unique_ptr<NNPICompiledFunction> compiledFunc =
      glow::make_unique<NNPICompiledFunction>(F);
  auto compileHasError = compiledFunc->compile(F, opts);
  if (compileHasError) {
    return std::move(compileHasError);
  }

  return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}

std::unique_ptr<FunctionPassPipeline>
NNPIBackend::getOptimizationPipeline() const {
  // We temporarily need to disable FoldTileAddIntoBatchedAdd, as it is causing
  // issues for NNPI.
  auto pipeline = createDefaultGraphOptimizationPassPipeline();
  pipeline->removeAllInstancesOfPass(FunctionPassID::FoldTileAddIntoBatchedAdd);

  // Disable SinkCode, as NNPI does data parallel transformations and so we do
  // not want to undo that by sinking Nodes back together.
  pipeline->removeAllInstancesOfPass(FunctionPassID::SinkCode);

  // Quantize Swish when wrapped inS Quantize/Dequantize.
  pipeline->pushBack(FunctionPassID::QuantizeSwish);

  // Raise Clips above Shape Nodes (e.g. Reshape) to try to ensure fusion
  // occurs. Note that we do this last as it may counteract some earlier
  // optimizations that push Clips down to try to eliminate them.
  pipeline->pushBack(FunctionPassID::RaiseClipsAboveShapeNodes);

  // Optimize away intermediate conversions, e.g. Quantize(ConvertTo(Node)) ->
  // Quantize(Node).
  pipeline->pushBack(FunctionPassID::OptimizeOutIntermediateConversions);

  // Now that we've raised clips up try to optimize quantize-clip combos again.
  pipeline->pushBack(FunctionPassID::OptimizeQuantizeClip);

  // Now try to eliminate any redundant Clips.
  pipeline->pushBack(FunctionPassID::OptimizeClips);

  // Look for float Relus that we can fuse up into quantized FCs.
  pipeline->pushBack(FunctionPassID::OptimizeQuantFCFloatRelu);

  // Optimize concats and quantized/dequantize patterns.
  pipeline->pushBack(FunctionPassID::OptimizeConcatQuantization);

  // Optimize quantization now that we've optimized some other quant nodes.
  pipeline->pushBack(
      {FunctionPassID::OptimizeQuantization, ConvergenceMode::UntilFixedPoint});

  // Now try to sink conversions below concats.
  pipeline->pushBack(FunctionPassID::SinkConversions);

  // Now that things have been sunk try to get rid of unnecessary concats.
  pipeline->pushBack(FunctionPassID::OptimizeConcatNodes);

  // Now try to get rid of unnecessary splits right before concats.
  pipeline->pushBack(FunctionPassID::EliminateSliceConcat);

  // Look for float Relus that we can fuse up into quantized FCs.
  pipeline->pushBack(FunctionPassID::OptimizeQuantFCFloatRelu);

  // Optimize concats and quantized/dequantize patterns.
  pipeline->pushBack(FunctionPassID::OptimizeConcatQuantization);

  // Sink concats below quantizes in order to try to eliminate unnecessary
  // quantizes above the concat.
  pipeline->pushBack(FunctionPassID::SinkConcatBelowQuantize);

  // Optimize quantization now that we've optimized some other quant nodes.
  pipeline->pushBack(
      {FunctionPassID::OptimizeQuantization, ConvergenceMode::UntilFixedPoint});

  // Now try to also optimize clips next to quantizes since we raised quantizes
  // above concats.
  pipeline->pushBack(FunctionPassID::OptimizeQuantizeClip);

  // Now try to sink conversions below concats again in case the concat quantize
  // sinking didn't help.
  pipeline->pushBack(FunctionPassID::SinkConversions);

  // Cleanup everything now.
  pipeline->pushBack(getDCEPassConfig());

  return pipeline;
}

bool NNPIBackend::lowerRequiredNodes(Function *F,
                                     CompilationContext &cctx) const {
  bool changed = false;
  for (auto &N : F->getNodes()) {
    NodeInfo NI(N);
    switch (N.getKind()) {
    case Kinded::Kind::BatchMatMulNodeKind:
      if (!GlowNNPILowerAllBatchMatMul &&
          !NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy})) {
        continue;
      }
      changed |= lowerNode(F, &N, cctx);
      continue;

    case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind: {
      if (NI.getOutElemTy(
              FusedRowwiseQuantizedSparseLengthsSumNode::ResultIdx) ==
          ElemKind::Float16Ty) {
        continue;
      }
      changed |= lowerNode(F, &N, cctx);
      continue;
    }

    case Kinded::Kind::PReluNodeKind:
    case Kinded::Kind::LogitNodeKind:
      if (NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Float16Ty})) {
        continue;
      }
      changed |= lowerNode(F, &N, cctx);
      continue;

    case Kinded::Kind::ConvertToNodeKind: {
      ConvertToNode *CT = llvm::cast<ConvertToNode>(&N);
      // Handle bool->float conversion
      if (((CT->getResult().getElementType() == ElemKind::FloatTy) ||
           (CT->getResult().getElementType() == ElemKind::Float16Ty)) &&
          CT->getInput().getElementType() == ElemKind::BoolTy) {
        auto outputType = CT->getResult().getType();
        auto ctName = CT->getName().str();
        auto *s0 = F->createSplat(ctName + "_s0", outputType, 0.0f);
        auto *s1 = F->createSplat(ctName + "_s1", outputType, 1.0f);
        auto *sel = F->createSelect(ctName + "_sel", CT->getInput(), s1, s0);
        CT->getResult().replaceAllUsesOfWith(sel);
        changed = true;
      }
      continue;
    }

    case Kinded::Kind::ReshapeNodeKind: {
      ReshapeNode *RS = llvm::cast<ReshapeNode>(&N);
      // Handle bool reshape by converting to Float16, reshaping, and
      // comparing >0
      if ((RS->getResult().getElementType() == ElemKind::BoolTy) &&
          (RS->getInput().getElementType() == ElemKind::BoolTy)) {
        auto rsName = RS->getName().str();
        auto *inputType = RS->getInput().getType();
        auto *outputType = RS->getResult().getType();
        auto *fp16Type =
            F->getParent()->uniqueType(ElemKind::Float16Ty, inputType->dims());
        auto *s0 = F->createSplat(rsName + "_s0", fp16Type, 0.0f);
        auto *s1 = F->createSplat(rsName + "_s1", fp16Type, 1.0f);
        auto *sel = F->createSelect(rsName + "_sel", RS->getInput(), s1, s0);
        auto *fp16ResType =
            F->getParent()->uniqueType(ElemKind::Float16Ty, outputType->dims());
        auto *res = F->createReshape(rsName + "_res", sel, fp16ResType->dims());
        auto *c0 = F->createSplat(rsName + "_c0", fp16ResType, 0.0f);
        auto *bres = F->createCmpGT(rsName + "_cmpGT", res, c0);
        RS->getResult().replaceAllUsesOfWith(bres);
        changed = true;
      }
      continue;
    }

    // Explicitly lower clips as they may be introduced during lowering other
    // ops above, e.g. Logit.
    case Kinded::Kind::ClipNodeKind:
    case Kinded::Kind::SwishNodeKind:
      if (NI.allInputsAndOutputsHaveSameElemKind(
              {ElemKind::Float16Ty, ElemKind::Int8QTy})) {
        continue;
      }
      changed |= lowerNode(F, &N, cctx);
      continue;

    case Kinded::Kind::SparseLengthsSumNodeKind: {
      const ElemKind k = NI.getOutElemTy(SparseLengthsSumNode::ResultIdx);
      // WA - lower until ICE-T implements it.
      if ((NNPIBackend::backendOptions_.useIceT ||
           NNPIBackend::backendOptions_.inferOnDevice) &&
          (k == ElemKind::FloatTy || k == ElemKind::Int8QTy)) {
        changed |= lowerNode(F, &N, cctx);
      }
      continue;
    }

    default:
      continue;
    }
  }
  return changed;
}

/// All activations have a single input and output.
static constexpr unsigned ActivationIOIdx = 0;
static_assert(ActivationIOIdx == ReluNode::InputIdx, "Format incorrect");
static_assert(ActivationIOIdx == ReluNode::ResultIdx, "Format incorrect");
static_assert(ActivationIOIdx == SigmoidNode::InputIdx, "Format incorrect");
static_assert(ActivationIOIdx == SigmoidNode::ResultIdx, "Format incorrect");
static_assert(ActivationIOIdx == TanhNode::InputIdx, "Format incorrect");
static_assert(ActivationIOIdx == TanhNode::ResultIdx, "Format incorrect");

/// Helper which looks for FC -> Clip -> Activation -> Clip, and removes the
/// Clip between the FC and Activation. These activations block FC-Activation
/// fusion from occurring.
static bool removeClipsBlockingFusion(Function *F) {
  bool changed = false;
  for (auto &N : F->getNodes()) {
    auto *clipActivation = llvm::dyn_cast<ClipNode>(&N);
    if (!clipActivation) {
      continue;
    }
    Node *activation = clipActivation->getInput().getNode();
    NodeValue activationInput;
    NodeValue activationResult;
    switch (activation->getKind()) {
    case Kinded::Kind::ReluNodeKind:
    case Kinded::Kind::SigmoidNodeKind:
    case Kinded::Kind::TanhNodeKind:
      activationInput = activation->getNthInput(ActivationIOIdx);
      activationResult = activation->getNthResult(ActivationIOIdx);
      break;
    default:
      continue;
    }
    auto *clipFC = llvm::dyn_cast<ClipNode>(activationInput);
    if (!clipFC) {
      continue;
    }
    if (clipFC->getMin() != clipActivation->getMin() ||
        clipFC->getMax() != clipActivation->getMax()) {
      continue;
    }
    auto *FC = llvm::dyn_cast<FullyConnectedNode>(clipFC->getInput());
    if (!FC) {
      continue;
    }
    clipFC->getResult().replaceAllUsesOfWith(FC->getResult());
    changed = true;
  }
  return changed;
}

Expected<bool>
NNPIBackend::transformPostOptPipeline(Function *F,
                                      CompilationContext &cctx) const {
  bool parallelized;
  ASSIGN_VALUE_OR_RETURN_ERR(parallelized,
                             parallelizeFunction(F, cctx.backendOpts));
  if (parallelized) {
    // Use the normal NNPI-specific optimization pipeline, but without sinking
    // conversions, because we just parallelized and so don't want to undo any
    // parallelization we performed on quantizes/dequantizes.
    auto P = getOptimizationPipeline();
    P->removeAllInstancesOfPass(FunctionPassID::SinkConversions);

    // Do not re-merge ConcatNodes, as we may be parallelizing them.
    const bool restoreMerge = cctx.optimizationOpts.skipConcatMerging;
    cctx.optimizationOpts.skipConcatMerging = true;

    FunctionPassManager("NNPI_transformPostOptPipeline", std::move(P), this)
        .run(F, cctx);

    cctx.optimizationOpts.skipConcatMerging = restoreMerge;
  }
  return parallelized;
}

Expected<bool> NNPIBackend::transformPostLowering(
    Function *F, CompilationContext &cctx,
    const glow::runtime::DeviceInfo *devInfo) const {
  LOG_SCOPE(F->getLogContext(), "NNPIBackend::transformPostLowering");

  // Signal to ConstantFolding to materialize those Splats which we require to
  // be Constants when importing later on.
  auto &kindSet = cctx.optimizationOpts.materializeSplatsUsedBySet;
  kindSet.insert(Kinded::Kind::ConvolutionNodeKind);
  kindSet.insert(Kinded::Kind::Convolution3DNodeKind);
  kindSet.insert(Kinded::Kind::FullyConnectedNodeKind);
  kindSet.insert(Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind);
  kindSet.insert(Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind);

  if (glow::onnxifi::GlowDisableNNPITransforms) {
    return false;
  }

  bool changed = removeClipsBlockingFusion(F);
  changed |= lowerRequiredNodes(F, cctx);

#if FACEBOOK_INTERNAL
  if (glow::onnxifi::GlowDisableNNPIPrivateTransforms) {
    return changed;
  }
  changed |= transformPrivate(F, cctx);
#endif /* FACEBOOK_INTERNAL */

  return changed;
}

// Traverse the DAG and collect nodes in post order.
static void
traversePostOrder(const runtime::DAGNode *root,
                  std::unordered_set<const runtime::DAGNode *> &visited,
                  std::vector<const runtime::DAGNode *> &postOrder) {
  if (root == nullptr) {
    return;
  }
  visited.insert(root);
  for (auto &c : root->children) {
    if (visited.count(c) == 0) {
      traversePostOrder(c, visited, postOrder);
    }
  }
  postOrder.push_back(root);
}

Error NNPIBackend::bindContexts(
    llvm::ArrayRef<runtime::ContextBinding> bindings,
    const runtime::DAGNode *root, bool enableP2P, bool enableDRT) {
  if (backendOptions_.dumpRuntime) {
    DotWriter::clear();
    DotWriter::addSubGraph("Host", "Host");
  }

  // Need post order to ensure p2p dest resources are created before their
  // source (since source will handle the copy command).
  std::unordered_set<const runtime::DAGNode *> visited;
  std::vector<const runtime::DAGNode *> postOrder;
  traversePostOrder(root, visited, postOrder);
  runtime::PlaceholderUsageMap phUsage;
  // Collect placeholders usage count.
  for (const auto &cb : bindings) {
    runtime::NNPIDeviceManager *nnpiDM =
        dynamic_cast<runtime::NNPIDeviceManager *>(cb.device);
    LOG_IF_NOT_RETURN_LLVMERROR(nnpiDM, "Invalid device manager");
    nnpiDM->addPlaceholderUsageCount(cb.networkName, phUsage);
  }

  for (auto &usage : phUsage) {
    LOG_IF_NOT_RETURN_LLVMERROR(
        usage.second.numWriters < 2,
        "Multiple writes to the same placeholder not suported");
    usage.second.disableP2P = !enableP2P;
    usage.second.disableDRT = !enableDRT;
  }

  for (auto *dagNode : postOrder) {
    if (dagNode->backendName != "NNPI") {
      continue;
    }

    // Find the contextbinding for this node (assuming there's only one).
    ExecutionContext *ctx = nullptr;
    runtime::DeviceManager *devMgr = nullptr;
    for (auto &cb : bindings) {
      if (cb.networkName == dagNode->name) {
        ctx = cb.context;
        devMgr = cb.device;
        break;
      }
    }
    if (ctx && devMgr) {
      // Update the tensors bound to placeholders.
      auto *phBindings = ctx->getPlaceholderBindings();
      for (auto &usage : phUsage) {
        const auto &phName = usage.first;
        auto *ph = phBindings->getPlaceholderByNameSlow(phName);
        usage.second.tensor = phBindings->get(ph);
      }

      runtime::NNPIDeviceManager *nnpiDM =
          dynamic_cast<runtime::NNPIDeviceManager *>(devMgr);
      LOG_IF_NOT_RETURN_LLVMERROR(nnpiDM, "Invalid device manager bound");
      LOG_IF_NOT_RETURN_LLVMERROR(
          !nnpiDM->bindContext(dagNode->name, ctx, phUsage),
          "Failed to bind context");
    }
  }

  if (backendOptions_.dumpRuntime) {
    DotWriter::writeToFile(root->name);
  }

  return Error::success();
}

/// Partial update of the NNPITensorDesc. Some members are ignored as they're
/// not used for estimation.
static bool updateDescForEstimate(NNPITensorDesc &desc,
                                  const glow::TypeRef ty) {
  LOG_AND_RETURN_IF(ERROR, ty == nullptr, "Invalid type", false);

  // Update dims and layout.
  NNPIImporter::updateDescDimsFromGlow(ty->dims(), desc);

  // Update Quantization.
  switch (ty->getElementType()) {
  case glow::ElemKind::FloatTy:
    desc.quantParams.precision = NNPI_PRECISION_FLOAT32;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case glow::ElemKind::Float16Ty:
    desc.quantParams.precision = NNPI_PRECISION_FLOAT16;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case glow::ElemKind::Int8QTy:
    desc.quantParams.precision = NNPI_PRECISION_INT8;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP;
    break;
  case glow::ElemKind::UInt8QTy:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP;
    break;
  case glow::ElemKind::Int32ITy:
    desc.quantParams.precision =
        NNPI_PRECISION_INT32; // The backend will convert to Int32 when
                              // compiling.
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case glow::ElemKind::Int64ITy:
    desc.quantParams.precision =
        NNPI_PRECISION_INT32; // The backend will convert to Int32 when
                              // compiling.
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case glow::ElemKind::Int32QTy:
    desc.quantParams.precision = NNPI_PRECISION_INT32;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP;
    break;
  case glow::ElemKind::UInt8FusedQTy:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ_FUSED;
    break;
  case glow::ElemKind::UInt8FusedFP16QTy:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ_FUSED_FP16;
    break;
  case glow::ElemKind::UInt4FusedFP16QTy:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ_4BIT_FUSED_FP16;
    break;
  case glow::ElemKind::BoolTy:
    desc.quantParams.precision = NNPI_PRECISION_BOOLEAN;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;

  default:
    LOG_AND_RETURN_IF(ERROR, true, "Invalid type", false);
    break;
  }
  memset(&desc.quantParams.params, 0,
         sizeof(desc.quantParams.params)); // Actual values are not needed here.

  desc.attributes.value = 0; // No attributes needed here.

  return true;
}

/// Prepare the list of NNPITensorDesc for the estimate call.
static bool updateDescListForEstimate(std::vector<NNPITensorDesc> &descs,
                                      const std::vector<glow::TypeRef> types) {
  if (descs.size() != types.size()) {
    return false;
  }
  bool retVal = true;
  for (size_t i = 0; i < descs.size(); i++) {
    if (types.at(i) != nullptr) {
      retVal &= updateDescForEstimate(descs.at(i), types.at(i));
    }
  }
  return retVal;
}

double NNPIBackend::estimateEmbeddingNode(const glow::NodeInfo &NI,
                                          bool fp32Accumulation,
                                          glow::LengthsMode lengthsMode,
                                          float averageLength) const {
  if (!isOpSupported(NI)) {
    // Op isn't supported.
    return -1.0;
  }
  NNPI_LENGTH_TYPE lengthType = NNPI_LENGTH_VARIABLE;
  LOG_AND_RETURN_IF(ERROR,
                    NNPIImporter::convertLengthsModeToLengthType(
                        lengthsMode, lengthType) != NNPI_NO_ERROR,
                    "Failed to convert LengthsMode", -1.0);

  enum DescIndex {
    Input = 0,
    Output = 1,
    Weight = 2,
    Index = 3,
    Length = 4,

    // Keep this last.
    NumIndices = 5,
  };
  std::vector<NNPITensorDesc> descs(NumIndices);

  bool validWeight = false;
  bool useLengthAsOffset = false;
  glow::TypeRef tr(nullptr);
  switch (NI.getKind()) {

  case Kinded::Kind::SparseLengthsSumNodeKind:
    LOG_AND_RETURN_IF(ERROR,
                      !updateDescListForEstimate(
                          descs,
                          {
                              NI.getInTy(SparseLengthsSumNode::DataIdx),
                              NI.getOutTy(SparseLengthsSumNode::ResultIdx),
                              nullptr,
                              NI.getInTy(SparseLengthsSumNode::IndicesIdx),
                              NI.getInTy(SparseLengthsSumNode::LengthsIdx),
                          }),
                      "Failed to update NNPITensorDesc", -1.0);
    break;

  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    validWeight = true;
    LOG_AND_RETURN_IF(
        ERROR,
        !updateDescListForEstimate(
            descs,
            {
                NI.getInTy(SparseLengthsWeightedSumNode::DataIdx),
                NI.getOutTy(SparseLengthsWeightedSumNode::ResultIdx),
                NI.getInTy(SparseLengthsWeightedSumNode::WeightsIdx),
                NI.getInTy(SparseLengthsWeightedSumNode::IndicesIdx),
                NI.getInTy(SparseLengthsWeightedSumNode::LengthsIdx),
            }),
        "Failed to update NNPITensorDesc", -1.0);
    break;

  case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    validWeight = true;
    LOG_AND_RETURN_IF(
        ERROR,
        !updateDescListForEstimate(
            descs,
            {
                NI.getInTy(
                    RowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx),
                NI.getOutTy(
                    RowwiseQuantizedSparseLengthsWeightedSumNode::ResultIdx),
                NI.getInTy(
                    RowwiseQuantizedSparseLengthsWeightedSumNode::WeightsIdx),
                NI.getInTy(
                    RowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx),
                NI.getInTy(
                    RowwiseQuantizedSparseLengthsWeightedSumNode::LengthsIdx),
            }),
        "Failed to update NNPITensorDesc", -1.0);
    break;

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind:
    LOG_AND_RETURN_IF(
        ERROR,
        !updateDescListForEstimate(
            descs,
            {
                NI.getInTy(FusedRowwiseQuantizedSparseLengthsSumNode::DataIdx),
                NI.getOutTy(
                    FusedRowwiseQuantizedSparseLengthsSumNode::ResultIdx),
                nullptr,
                NI.getInTy(
                    FusedRowwiseQuantizedSparseLengthsSumNode::IndicesIdx),
                NI.getInTy(
                    FusedRowwiseQuantizedSparseLengthsSumNode::LengthsIdx),
            }),
        "Failed to update NNPITensorDesc", -1.0);
    break;

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    validWeight = true;
    LOG_AND_RETURN_IF(
        ERROR,
        !updateDescListForEstimate(
            descs,
            {
                NI.getInTy(
                    FusedRowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx),
                NI.getOutTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                                ResultIdx),
                NI.getInTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               WeightsIdx),
                NI.getInTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               IndicesIdx),
                NI.getInTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               LengthsIdx),
            }),
        "Failed to update NNPITensorDesc", -1.0);
    break;

  case Kinded::Kind::EmbeddingBagNodeKind:
    validWeight = true;
    useLengthAsOffset = true;
    LOG_AND_RETURN_IF(
        ERROR,
        !updateDescListForEstimate(descs,
                                   {
                                       NI.getInTy(EmbeddingBagNode::DataIdx),
                                       NI.getOutTy(EmbeddingBagNode::ResultIdx),
                                       NI.getInTy(EmbeddingBagNode::WeightsIdx),
                                       NI.getInTy(EmbeddingBagNode::IndicesIdx),
                                       NI.getInTy(EmbeddingBagNode::OffsetsIdx),
                                   }),
        "Failed to update NNPITensorDesc", -1.0);
    break;

  case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind:
    validWeight = true;
    useLengthAsOffset = true;
    LOG_AND_RETURN_IF(
        ERROR,
        !updateDescListForEstimate(
            descs,
            {
                NI.getInTy(EmbeddingBagByteRowwiseOffsetsNode::DataIdx),
                NI.getOutTy(EmbeddingBagByteRowwiseOffsetsNode::ResultIdx),
                NI.getInTy(EmbeddingBagByteRowwiseOffsetsNode::WeightsIdx),
                NI.getInTy(EmbeddingBagByteRowwiseOffsetsNode::IndicesIdx),
                NI.getInTy(EmbeddingBagByteRowwiseOffsetsNode::OffsetsIdx),
            }),
        "Failed to update NNPITensorDesc", -1.0);
    break;

  default:
    return -1.0;
  }

  double estimate = -1.0;
  LOG_NNPI_IF_ERROR(nnpiEstimateSparseLengthsWeightedSumOp(
                        &(descs.at(Input)), &(descs.at(Output)),
                        validWeight ? &(descs.at(Weight)) : nullptr,
                        &(descs.at(Index)), &(descs.at(Length)),
                        fp32Accumulation, useLengthAsOffset, averageLength,
                        lengthType, &estimate),
                    "Failed to estimate SLS op.");

  return estimate;
}

Expected<double> NNPIBackend::estimateNodeCost(const glow::Node *node) const {
  double returnCost = -1.0;
  switch (node->getKind()) {
  case Kinded::Kind::SparseLengthsSumNodeKind: {
    const SparseLengthsSumNode *SLS = llvm::cast<SparseLengthsSumNode>(node);
    returnCost =
        estimateEmbeddingNode(glow::NodeInfo(*SLS), false,
                              SLS->getLengthsMode(), SLS->getAvgLength());
    break;
  }
  case Kinded::Kind::SparseLengthsWeightedSumNodeKind: {
    const SparseLengthsWeightedSumNode *SLWS =
        llvm::cast<SparseLengthsWeightedSumNode>(node);
    returnCost =
        estimateEmbeddingNode(glow::NodeInfo(*SLWS), false,
                              SLWS->getLengthsMode(), SLWS->getAvgLength());
    break;
  }
  case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind: {
    const RowwiseQuantizedSparseLengthsWeightedSumNode *RQSLWS =
        llvm::cast<RowwiseQuantizedSparseLengthsWeightedSumNode>(node);
    returnCost =
        estimateEmbeddingNode(glow::NodeInfo(*RQSLWS), false,
                              RQSLWS->getLengthsMode(), RQSLWS->getAvgLength());
    break;
  }
  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind: {
    const FusedRowwiseQuantizedSparseLengthsSumNode *FRQSLS =
        llvm::cast<FusedRowwiseQuantizedSparseLengthsSumNode>(node);
    returnCost =
        estimateEmbeddingNode(glow::NodeInfo(*FRQSLS), false,
                              FRQSLS->getLengthsMode(), FRQSLS->getAvgLength());
    break;
  }
  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind: {
    const FusedRowwiseQuantizedSparseLengthsWeightedSumNode *FRQSLWS =
        llvm::cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(node);
    returnCost = estimateEmbeddingNode(glow::NodeInfo(*FRQSLWS), false,
                                       FRQSLWS->getLengthsMode(),
                                       FRQSLWS->getAvgLength());
    break;
  }
  case Kinded::Kind::EmbeddingBagNodeKind: {
    const EmbeddingBagNode *EB = llvm::cast<EmbeddingBagNode>(node);
    returnCost = estimateEmbeddingNode(
        glow::NodeInfo(*EB), false, EB->getLengthsMode(), EB->getAvgLength());
    break;
  }
  case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind: {
    const EmbeddingBagByteRowwiseOffsetsNode *EBBRO =
        llvm::cast<EmbeddingBagByteRowwiseOffsetsNode>(node);
    returnCost =
        estimateEmbeddingNode(glow::NodeInfo(*EBBRO), false,
                              EBBRO->getLengthsMode(), EBBRO->getAvgLength());
    break;
  }
  default:
    break;
  }
  RETURN_ERR_IF_NOT(returnCost >= 0.0,
                    strFormat("Estimate not supported for Node kind %s",
                              Kinded::getKindName(node->getKind())));
  return returnCost;
}
