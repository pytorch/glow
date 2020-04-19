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
#include "NNPICompiledFunction.h"
#include "NNPIDeviceManager.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPassPipeline.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Optimizer/Lower/Lower.h"

#include "llvm/Support/CommandLine.h"

#include <fstream>

using namespace glow;

namespace glow {
llvm::cl::OptionCategory optionsForNNPI("NNPI Backend Options");

bool GlowNNPILowerAllBatchMatMul = false;
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    GlowNNPILowerAllBatchMatMulOpt(
        "glow_nnpi_lower_all_batch_matmul",
        llvm::cl::desc("Whether to override default "
                       "lowering for NNPI and "
                       "always lower BatchMatMul to a "
                       "series of MatMuls."),
        llvm::cl::location(GlowNNPILowerAllBatchMatMul), llvm::cl::Optional,
        llvm::cl::init(false), llvm::cl::cat(optionsForNNPI));

bool GlowNNPIAcceptUnarySLS = false;
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    GlowNNPIAcceptUnarySLSOpt(
        "glow_nnpi_accept_unary_sls",
        llvm::cl::desc(
            "Whether to accept unary SLS ops during ONNXIFI loading."),
        llvm::cl::location(GlowNNPIAcceptUnarySLS), llvm::cl::Optional,
        llvm::cl::init(false), llvm::cl::cat(optionsForNNPI));

namespace onnxifi {

bool GlowDumpNNPICompilerData = false;
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    GlowDumpNNPICompilerDataOpt("glow_dump_nnpi_compiler_data",
                                llvm::cl::desc("Whether to dump NNPI compiler"
                                               "data to a file"),
                                llvm::cl::location(GlowDumpNNPICompilerData),
                                llvm::cl::Optional, llvm::cl::init(false),
                                llvm::cl::cat(optionsForNNPI));

bool GlowUsePerPartitionIcetConfig = true;
static llvm::cl::opt<bool, /* ExternalStorage */ true>
    GlowUsePerPartitionIcetConfigOpt(
        "glow_use_per_partition_icet_config",
        llvm::cl::desc("Whether to load an"
                       "icet_config.json file"
                       "for each partition"),
        llvm::cl::location(GlowUsePerPartitionIcetConfig), llvm::cl::Optional,
        llvm::cl::init(false), llvm::cl::cat(optionsForNNPI));

bool GlowDisableNNPITransforms = false;
bool GlowDisableNNPIPrivateTransforms = false;
int32_t GlowNNPINumParallelChunks = 0;

} // namespace onnxifi
} // namespace glow

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

bool NNPIBackend::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {
  // General math fp32/fp16/i8.
  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MulNodeKind:
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
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::ExpNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::LayerNormalizationNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});

  case Kinded::Kind::BatchNormalizationNodeKind:
  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::AdaptiveAvgPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});

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
           (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy);

  case Kinded::Kind::Convolution3DNodeKind:
    if (!NI.getInTy(Convolution3DNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    }
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::Int8QTy}, {Convolution3DNode::BiasIdx}) &&
           (NI.getInElemTy(Convolution3DNode::BiasIdx) == ElemKind::Int32QTy);
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
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    }
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::Int8QTy}, {FullyConnectedNode::BiasIdx}) &&
           (NI.getInElemTy(FullyConnectedNode::BiasIdx) == ElemKind::Int32QTy);

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
           (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::BiasIdx) ==
            ElemKind::Int32QTy) &&
           (NI.getOutElemTy(RowwiseQuantizedFullyConnectedNode::ResultIdx) ==
            ElemKind::Int8QTy);

  case Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind:
    return (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::InputIdx) ==
            ElemKind::Int8QTy) &&
           (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::FilterIdx) ==
            ElemKind::Int8QTy) &&
           (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::BiasIdx) ==
            ElemKind::Int32QTy) &&
           (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::ScalesIdx) ==
            ElemKind::FloatTy) &&
           (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::OffsetsIdx) ==
            ElemKind::Int32ITy) &&
           (NI.getOutElemTy(ChannelwiseQuantizedConvolutionNode::ResultIdx) ==
            ElemKind::Int8QTy);

  case Kinded::Kind::SparseLengthsSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
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
    return NI.allInputsAndOutputsHaveSameElemKind(
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
    return NI.allInputsAndOutputsHaveSameElemKind(
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
    return (dataK == ElemKind::UInt8FusedQTy ||
            dataK == ElemKind::UInt8FusedFP16QTy) &&
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
    return (dataK == ElemKind::UInt8FusedQTy ||
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
    return (dataK == ElemKind::UInt8FusedQTy ||
            dataK == ElemKind::UInt8FusedFP16QTy ||
            dataK == ElemKind::UInt4FusedFP16QTy) &&
           (weightsK == ElemKind::FloatTy || weightsK == ElemKind::Float16Ty) &&
           (resultK == ElemKind::FloatTy || resultK == ElemKind::Float16Ty) &&
           (indicesK == ElemKind::Int64ITy || indicesK == ElemKind::Int32ITy) &&
           (lengthsK == ElemKind::Int32ITy);
  }

  case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
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
               {ElemKind::FloatTy, ElemKind::Float16Ty},
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
    return true;

  case Kinded::Kind::SpaceToDepthNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::ArgMaxNodeKind:
    return (NI.getOutElemTy(ArgMaxNode::ArgmaxIdx) == ElemKind::Int64ITy);

  default:
    llvm::outs() << "Unsupported op:\n" << NI.getDebugDesc() << "\n";
    return false;
  }

  return false;
}

bool NNPIBackend::shouldLower(const Node *N) const {
  switch (N->getKind()) {
  case Kinded::Kind::ClipNodeKind: {
    const ClipNode *CN = llvm::cast<ClipNode>(N);
    if (CN->getResult().getElementType() != ElemKind::Float16Ty &&
        CN->getResult().getElementType() != ElemKind::Int8QTy) {
      return true;
    }
    return false;
  }
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ConvolutionNodeKind:
  case Kinded::Kind::TileNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::ReplaceNaNNodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
  case Kinded::Kind::BatchedReduceMeanNodeKind:
  case Kinded::Kind::BatchedReduceMinNodeKind:
  case Kinded::Kind::BatchMatMulNodeKind:
  case Kinded::Kind::BatchNormalizationNodeKind:
  case Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind:
  case Kinded::Kind::AdaptiveAvgPoolNodeKind:
  case Kinded::Kind::EmbeddingBagNodeKind:
  case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind:
    return false;
  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind: {
    const FusedRowwiseQuantizedSparseLengthsSumNode *SLSN =
        llvm::cast<FusedRowwiseQuantizedSparseLengthsSumNode>(N);
    if (SLSN->getResult().getElementType() == ElemKind::Float16Ty) {
      return false; // Don't lower == keep without weights
    } else {
      return true;
    }
  }
  case Kinded::Kind::LayerNormalizationNodeKind:
  case Kinded::Kind::SparseLengthsSumNodeKind:
    // WA - lower until ICE-T implements it.
    if (NNPIBackend::backendOptions_.useIceT ||
        NNPIBackend::backendOptions_.inferOnDevice) {
      return true;
    }
    return false;
  case Kinded::Kind::PReluNodeKind: {
    NodeInfo NI(*N);
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});
  }
  default:
    return true;
    break;
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
        numChunks[FC] = numParallelChunks;
        continue;
      }
      size_t M = FC->getInput().dims()[0];
      if (M >= 256) {
        parOpts[FC] = ParallelTransformKind::Data;
        numChunks[FC] = numParallelChunks;
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
          numChunks[R] = numParallelChunks;
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
        numChunks[R] = numParallelChunks;
        continue;
      }
      size_t M = R->getInput().dims()[0];
      if (M >= 256) {
        parOpts[R] = ParallelTransformKind::Data;
        numChunks[R] = numParallelChunks;
        continue;
      }
    }

    // Split transpose layers in data parallel fashion
    if (auto *TP = llvm::dyn_cast<TransposeNode>(node)) {
      parOpts[TP] = ParallelTransformKind::Data;
      numChunks[TP] = numParallelChunks;
    }

    // Split Quantize layers in data parallel fashion
    if (auto *QN = llvm::dyn_cast<QuantizeNode>(node)) {
      parOpts[QN] = ParallelTransformKind::Data;
      numChunks[QN] = numParallelChunks;
    }

    // Split Dequantize layers in data parallel fashion
    if (auto *DQN = llvm::dyn_cast<DequantizeNode>(node)) {
      parOpts[DQN] = ParallelTransformKind::Data;
      numChunks[DQN] = numParallelChunks;
    }

    // Split BMM layers in data parallel fashion
    if (auto *BMM = llvm::dyn_cast<BatchMatMulNode>(node)) {
      parOpts[BMM] = ParallelTransformKind::Data;
      numChunks[BMM] = numParallelChunks;
    }

    // Split Tanh layers in data parallel fashion
    if (auto *TH = llvm::dyn_cast<TanhNode>(node)) {
      if (TH->getInput().dims().size() < 2) {
        continue;
      }
      size_t N = TH->getInput().dims()[1];
      if (N < 4096) {
        continue;
      }
      parOpts[TH] = ParallelTransformKind::Data;
      numChunks[TH] = numParallelChunks;
    }

    // Split Mul layers in data parallel fashion
    if (auto *M = llvm::dyn_cast<MulNode>(node)) {
      if (M->getLHS().dims().size() < 2) {
        continue;
      }
      size_t N = M->getLHS().dims()[1];
      if (N < 4096) {
        continue;
      }
      parOpts[M] = ParallelTransformKind::Data;
      numChunks[M] = numParallelChunks;
    }

    // Clip parallelization.
    // If a Clip follows a parallel op, mirror that.
    if (auto *C = llvm::dyn_cast<ClipNode>(node)) {
      Node *inputNode = C->getInput().getNode();
      if (numChunks.find(inputNode) != numChunks.end() &&
          parOpts.find(inputNode) != parOpts.end()) {
        parOpts[C] = parOpts[inputNode];
        numChunks[C] = numChunks[inputNode];
      }
    }
  }
}

/// If we've done some paralleization specified in \p replacedMap then propagate
/// any NodeInfo from original nodes to the newly created Nodes in
/// \p backendSpecificNodeInfo. Additionally, validate that the parallelization
/// matches with the specified previous NodeInfo. \returns whether any
/// validation error is found.
static Error propagateBackendSpecificNodeInfo(
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
    RETURN_ERR_IF_NOT(numParChunksVal == CN->getInputs().size(),
                      "Node not split the expected number of times.");

    // Look for coreAssignments and propagate them into each Node.
    auto coreAssignmentsIt = nodeInfo.find(coreAssignmentsKey);
    if (coreAssignmentsIt != nodeInfo.end()) {
      RETURN_ERR_IF_NOT(coreAssignmentsIt->second.size() ==
                            CN->getInputs().size(),
                        "Require same number of assignments as split factor");
      for (size_t i = 0, e = CN->getInputs().size(); i < e; i++) {
        Node *inputCN = CN->getInputs()[i].getNode();
        auto &newCoreAssignments = currFunInfo[inputCN][coreAssignmentsKey];
        RETURN_ERR_IF_NOT(newCoreAssignments.size() == 0,
                          std::string(coreAssignmentsKey) +
                              " should have been empty.");
        newCoreAssignments.push_back(coreAssignmentsIt->second[i]);
      }
    }

    // Look for NNPI_extraEdges and propagate them into each Node.
    auto extraEdgesIt = nodeInfo.find(extraEdgesKey);
    if (extraEdgesIt != nodeInfo.end()) {
      for (const NodeValue &inputCNNV : CN->getInputs()) {
        auto &newExtraEdges = currFunInfo[inputCNNV.getNode()][extraEdgesKey];
        RETURN_ERR_IF_NOT(newExtraEdges.size() == 0,
                          std::string(extraEdgesKey) +
                              " should have been empty.");
        for (const std::string &edge : extraEdgesIt->second) {
          newExtraEdges.push_back(edge);
        }
      }
    }

    // Now we can erase this Node's info from currFunInfo because it has been
    // replaced and will be DCE'd soon.
    currFunInfo.erase(curNodeInfoIt);
  }

  // Now we need to look through all extraEdges and clean them up so they point
  // to parallelized names of opts. They should be formatted like "nodeName@#",
  // where '#' is an int representing which parallel chunk edge should be used.
  for (auto &nodeInfoPair : currFunInfo) {
    for (auto &keyOptsPair : nodeInfoPair.second) {
      const llvm::StringRef &key = keyOptsPair.getKey();
      std::vector<std::string> &opts = keyOptsPair.getValue();

      // Look for any extraEdges options.
      if (key != extraEdgesKey) {
        continue;
      }

      for (std::string &edge : opts) {
        // Only process edges that were expected to be split.
        llvm::StringRef edgeRef(edge);
        if (!edgeRef.contains('@')) {
          continue;
        }

        auto splitPair = edgeRef.split('@');
        RETURN_ERR_IF_NOT(splitPair.second != "",
                          "Edge must have an integer value after @");

        int splitNum;
        ASSIGN_VALUE_OR_RETURN_ERR(splitNum, getIntFromStr(splitPair.second));

        auto it = nameToReplacementMap.find(splitPair.first);
        RETURN_ERR_IF_NOT(
            it != nameToReplacementMap.end(),
            "Must have a replacement Concat for a parallelized edge.");

        const ConcatNode *replaceCN = it->second;
        RETURN_ERR_IF_NOT(splitNum < replaceCN->getInputs().size(),
                          "splitNum for edge exceeded size of the split.");

        // Finally, replace the name of the old edge (containing '@') with the
        // name of the new edge created during the parallelization pass.
        edge = replaceCN->getInputs()[splitNum].getNode()->getName().str();
      }
    }
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

/// Parallelize \p F. If \p usePerNodeParallelizationSpec then this
/// parallelization is done based on the spec found in backendSpecificNodeInfo
/// in \p opts. Else perform basic parallelization according to either
/// GlowNNPINumParallelChunks, or if not specified then NNPINumParallelChunks
/// found in backendOpts.backendSpecificOpts from \p opts. \returns whether \p F
/// was modified.
static Expected<bool> parallelizeFunction(Function *F, BackendOptions &opts,
                                          bool usePerNodeParallelizationSpec) {
  // Split FC layers in model/data parallel fashion
  llvm::DenseMap<Node *, size_t> numChunks;
  llvm::DenseMap<Node *, ParallelTransformKind> parOpts;

  int32_t defaultNumParallelChunks = 1;
  if (usePerNodeParallelizationSpec) {
    // If we don't have any info for this function then return early.
    if (opts.backendSpecificNodeInfo.find(F) ==
        opts.backendSpecificNodeInfo.end()) {
      return false;
    }

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

  // Now actually do the parallelization.
  std::unordered_map<Node *, ConcatNode *> replacedMap;
  ASSIGN_VALUE_OR_RETURN_ERR(
      replacedMap,
      parallelizeOps(F, numChunks, parOpts, defaultNumParallelChunks));

  RETURN_ERR_IF_NOT(numChunks.size() == replacedMap.size(),
                    "Expected that numChunks and replacedMap have same size.");

  if (usePerNodeParallelizationSpec) {
    // If parallelization was based on backend-specific node info then propagate
    // it to new nodes that were added.
    RETURN_IF_ERR(propagateBackendSpecificNodeInfo(
        F, replacedMap, opts.backendSpecificNodeInfo));
  }

  return true;
}

Expected<std::unique_ptr<CompiledFunction>>
NNPIBackend::compile(Function *F, const BackendOptions &opts) const {
  BackendOptions newOpts = opts;

  // Perform parallelization based on any node options found in opts.
  bool parallelized;
  ASSIGN_VALUE_OR_RETURN_ERR(
      parallelized, parallelizeFunction(
                        F, newOpts, /* usePerNodeParallelizationSpec */ true));
  if (parallelized) {
    // If we parallelized then we want to run very specific optimizations to
    // clean up the now-parallelized graph while preserving the Nodes in the
    // Function so we don't mess up the placement info map. Specifically, we
    // eliminate Concat-Slice patterns which are created during parallelization.
    // This does not create any new nodes (it only removes Concat-Slice
    // patterns, replacing uses of Concat with the input of Slice). Then we DCE
    // away the now-dead Concats/Slices.
    FunctionPassManager FPM("FinalizeFPM",
                            {
                                FunctionPassID::EliminateConcatSlice,
                                FunctionPassID::FoldSlicesIntoConstants,
                                getDCEPassConfig(),
                            });
    FPM.run(F, CompilationContext());
  }

  std::unique_ptr<NNPICompiledFunction> compiledFunc =
      glow::make_unique<NNPICompiledFunction>(F);
  auto compileHasError = compiledFunc->compile(F, newOpts);
  if (compileHasError) {
    return std::move(compileHasError);
  }

  return Expected<std::unique_ptr<CompiledFunction>>(std::move(compiledFunc));
}

FunctionPassPipeline NNPIBackend::getOptimizationPipeline() const {
  // We temporarily need to disable FoldTileAddIntoBatchedAdd, as it is causing
  // issues for NNPI.
  auto pipeline = createDefaultGraphOptimizationPassPipeline();
  pipeline.removeAllInstancesOfPass(FunctionPassID::FoldTileAddIntoBatchedAdd);

  // Disable SinkCode, as NNPI does data parallel transformations and so we do
  // not want to undo that by sinking Nodes back together.
  pipeline.removeAllInstancesOfPass(FunctionPassID::SinkCode);

  // Raise Clips above Shape Nodes (e.g. Reshape) to try to ensure fusion
  // occurs. Note that we do this last as it may counteract some earlier
  // optimizations that push Clips down to try to eliminate them.
  pipeline.pushBack(FunctionPassID::RaiseClipsAboveShapeNodes);

  // Optimize away intermediate conversions, e.g. Quantize(ConvertTo(Node)) ->
  // Quantize(Node).
  pipeline.pushBack(FunctionPassID::OptimizeOutIntermediateConversions);

  // Now that we've raised clips up try to optimize quantize-clip combos again.
  pipeline.pushBack(FunctionPassID::OptimizeQuantizeClip);

  // Now try to eliminate any redundant Clips.
  pipeline.pushBack(FunctionPassID::OptimizeClips);

  // Look for float Relus that we can fuse up into quantized FCs.
  pipeline.pushBack(FunctionPassID::OptimizeQuantFCFloatRelu);

  // Optimize concats and quantized/dequantize patterns.
  pipeline.pushBack(FunctionPassID::OptimizeConcatQuantization);

  // Optimize quantization now that we've optimized some other quant nodes.
  pipeline.pushBack(FunctionPassID::OptimizeQuantization);

  // Now try to sink conversions below concats.
  pipeline.pushBack(FunctionPassID::SinkConversions);

  // Now that things have been sunk try to get rid of unnecessary concats.
  pipeline.pushBack(FunctionPassID::OptimizeConcatNodes);

  // Look for float Relus that we can fuse up into quantized FCs.
  pipeline.pushBack(FunctionPassID::OptimizeQuantFCFloatRelu);

  // Optimize concats and quantized/dequantize patterns.
  pipeline.pushBack(FunctionPassID::OptimizeConcatQuantization);

  // Cleanup everything now.
  pipeline.pushBack(getDCEPassConfig());

  return pipeline;
}

/// Helper to lower nodes which need further lowering. \returns whether \p F was
/// modified.
static bool lowerRequiredNodes(Function *F, CompilationContext &cctx) {
  bool changed = false;
  for (auto &N : F->getNodes()) {
    BatchMatMulNode *BMMN = llvm::dyn_cast<BatchMatMulNode>(&N);
    if (!BMMN) {
      continue;
    }

    if (!GlowNNPILowerAllBatchMatMul &&
        !NodeInfo(*BMMN).allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy})) {
      continue;
    }

    lowerNode(F, BMMN, cctx);
    changed = true;
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

Expected<bool> NNPIBackend::transformPostLowering(
    Function *F, CompilationContext &cctx,
    const glow::runtime::DeviceInfo *devInfo) const {
  LOG_SCOPE(F->getLogContext(), "NNPIBackend::transformPostLowering");

  if (glow::onnxifi::GlowDisableNNPITransforms) {
    return false;
  }

  bool changed = removeClipsBlockingFusion(F);
  changed |= lowerRequiredNodes(F, cctx);
  bool parallelized;
  ASSIGN_VALUE_OR_RETURN_ERR(
      parallelized,
      parallelizeFunction(F, cctx.backendOpts,
                          /* usePerNodeParallelizationSpec */ false));
  changed |= parallelized;

#if FACEBOOK_INTERNAL
  if (glow::onnxifi::GlowDisableNNPIPrivateTransforms) {
    return changed;
  }
  changed |= transformPrivate(F, cctx);
#endif /* FACEBOOK_INTERNAL */

  return changed;
}
