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
#include "CustomKernels/DSPInjectors/DSPInjectors.h"
#include "CustomKernels/IAInjectors/IAInjectors.h"
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

#include <cstdio>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

std::vector<unsigned> NNPIBackend::scanDeviceIDs() {
  std::vector<unsigned> devices;
  for (int i = 0; i < NNPIBackend::numDevices(); ++i) {
    std::string devPath = "/dev/nnpi" + std::to_string(i);
    if (FILE *devFile = fopen(devPath.c_str(), "r")) {
      fclose(devFile);
      LOG(INFO) << "Scan NNPI device found: " << i;
      devices.push_back(i);
    } else {
      continue;
    }
  }
  return devices;
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
    return nnpi::flags::AcceptUnarySLS ||
           !isUnaryLookup(NI.getInTy(SparseLengthsSumNode::DataIdx));
  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return nnpi::flags::AcceptUnarySLS ||
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

enum NodeSupportLevels {
  PRECISION_SUPPORTED,
  SUPPORTED,
  NOT_SUPPORTED,
};

static NodeSupportLevels isNodeSupported(const NodeInfo &NI) {
  bool isNodePrecisionSupported = false;
  bool isNodeHasAnySupport = true;
  switch (NI.getKind()) {
  // General math fp32/fp16/i8/int32.
  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MinNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int32ITy, ElemKind::Float16Ty,
         ElemKind::Int8QTy, ElemKind::Int64ITy});
    break;
  case Kinded::Kind::DivNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int64ITy, ElemKind::Int32ITy});
    break;

  // General math fp32/fp16/i8.
  case Kinded::Kind::PowNodeKind:
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ReplaceNaNNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
  case Kinded::Kind::BatchedReduceMeanNodeKind:
  case Kinded::Kind::BatchedAddNodeKind:
  case Kinded::Kind::BatchedMulNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::NegNodeKind:
  case Kinded::Kind::AbsNodeKind:
  case Kinded::Kind::ExpNodeKind:
  case Kinded::Kind::SoftPlusNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy});
    break;
  case Kinded::Kind::BatchedReduceMinNodeKind:
  case Kinded::Kind::BatchedReduceMaxNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy});
    break;
  case Kinded::Kind::SplatNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});
    break;
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Float16Ty, ElemKind::Int8QTy});
    break;
  case Kinded::Kind::ModuloNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});
    break;
#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 1
  case Kinded::Kind::NNPILookupTableNodeKind:
  case Kinded::Kind::IntLookupTableNodeKind:
    isNodePrecisionSupported = true;
    break;
  case Kinded::Kind::BBoxTransformNodeKind:
    // RoiBatchSplits output should be FP16 in the Glow node and get
    // converted explicitly to FP32 in NNPI importer.
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Float16Ty});
    break;
  case Kinded::Kind::ROIAlignNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::Float16Ty}, {ROIAlignNode::BatchIndicesIdx}) &&
        (NI.getInElemTy(ROIAlignNode::BatchIndicesIdx) == ElemKind::Int32ITy ||
         NI.getInElemTy(ROIAlignNode::BatchIndicesIdx) == ElemKind::Int64ITy);
    break;
  case Kinded::Kind::LSTMUnitNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Float16Ty});
    break;
  case Kinded::Kind::ResizeNearestNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int32QTy,
         ElemKind::Int8QTy, ElemKind::UInt8QTy});
    break;
  case Kinded::Kind::SparseLabelSplitNodeKind: {
    auto valuesIdxDataType = NI.getInElemTy(SparseLabelSplitNode::ValuesIdx);
    isNodePrecisionSupported =
        (NI.getInElemTy(SparseLabelSplitNode::LengthsIdx) ==
         ElemKind::Int32ITy) &&
        (NI.getInElemTy(SparseLabelSplitNode::IndicesIdx) ==
         ElemKind::Int64ITy) &&
        (NI.getInElemTy(SparseLabelSplitNode::ValuesIdx) ==
         NI.getOutElemTy(SparseLabelSplitNode::LabelValuesIdx)) &&
        (NI.getOutElemTy(SparseLabelSplitNode::ExampleIdsIdx) ==
         ElemKind::Int32ITy) &&
        (NI.getOutElemTy(SparseLabelSplitNode::GradientOffsetMapIdx) ==
         ElemKind::Int32ITy) &&
        (valuesIdxDataType == ElemKind::FloatTy ||
         valuesIdxDataType == ElemKind::Float16Ty ||
         valuesIdxDataType == ElemKind::Int8QTy ||
         valuesIdxDataType == ElemKind::UInt8QTy);
    break;
  }
#endif // NNPI > 1.1
  case Kinded::Kind::LayerNormalizationNodeKind: {
    auto scaleType = NI.getInElemTy(LayerNormalizationNode::ScaleIdx);
    auto biasType = NI.getInElemTy(LayerNormalizationNode::BiasIdx);
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::Float16Ty, ElemKind::Int8QTy},
            {LayerNormalizationNode::ScaleIdx,
             LayerNormalizationNode::BiasIdx}) &&
        scaleType == biasType &&
        (scaleType == ElemKind::Float16Ty || scaleType == ElemKind::Int8QTy);
    break;
  }
  case Kinded::Kind::SwishNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Float16Ty, ElemKind::Int8QTy, ElemKind::UInt8QTy});
    break;
  case Kinded::Kind::GeluNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Float16Ty});
    break;
  case Kinded::Kind::BatchNormalizationNodeKind: {
    auto elemType = NI.getInElemTy(BatchNormalizationNode::InputIdx);
    isNodePrecisionSupported =
        (elemType == ElemKind::Int8QTy || elemType == ElemKind::FloatTy ||
         elemType == ElemKind::Float16Ty);

    isNodePrecisionSupported = isNodePrecisionSupported &&
                               NI.allInputsAndOutputsHaveSameElemKind(
                                   {ElemKind::FloatTy, ElemKind::Float16Ty},
                                   {BatchNormalizationNode::InputIdx},
                                   {BatchNormalizationNode::ResultIdx});

    isNodePrecisionSupported =
        isNodePrecisionSupported &&
        NI.allInputsAndOutputsHaveSameElemKind(
            {elemType},
            {BatchNormalizationNode::ScaleIdx, BatchNormalizationNode::BiasIdx,
             BatchNormalizationNode::MeanIdx, BatchNormalizationNode::VarIdx});
    break;
  }
  case Kinded::Kind::VectorNormNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Float16Ty, ElemKind::Int8QTy, ElemKind::UInt8QTy});
    break;
  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::AdaptiveAvgPoolNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy});
    break;
  case Kinded::Kind::BatchMatMulNodeKind:
  case Kinded::Kind::PReluNodeKind:
  case Kinded::Kind::ClipNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int8QTy, ElemKind::Float16Ty});
    break;
  case Kinded::Kind::FmodNodeKind:
#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 7
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Float16Ty});
#else
    // Supporting these two for now because for fp inputs NNPI returns result
    // with the same sign as the divisor instead of the dividend.
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int64ITy, ElemKind::Int32ITy});
#endif // NNPI >= 1.7
    break;
  // Data transfer fp32/fp16/i8/i32/i64/bool.
  case Kinded::Kind::SaveNodeKind:
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::TileNodeKind:
  case Kinded::Kind::TransposeNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy, ElemKind::BoolTy});
    break;
  case Kinded::Kind::ConvolutionNodeKind: {
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    } else {
      isNodePrecisionSupported =
          NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy},
                                                 {ConvolutionNode::BiasIdx}) &&
          ((NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy) ||
           (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::FloatTy));
    }
    break;
  }
  case Kinded::Kind::Convolution3DNodeKind:
    if (!NI.getInTy(Convolution3DNode::InputIdx)->isQuantizedType()) {
      isNodePrecisionSupported =
          NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Float16Ty});
    } else {
      isNodePrecisionSupported =
          NI.allInputsAndOutputsHaveSameElemKind(
              {ElemKind::Int8QTy}, {Convolution3DNode::BiasIdx}) &&
          ((NI.getInElemTy(Convolution3DNode::BiasIdx) == ElemKind::Int32QTy) ||
           (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::FloatTy));
    }
    break;
  case Kinded::Kind::QuantizeNodeKind:
    isNodePrecisionSupported =
        (NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::FloatTy ||
         NI.getInElemTy(QuantizeNode::InputIdx) == ElemKind::Float16Ty) &&
        (NI.getOutElemTy(QuantizeNode::ResultIdx) == ElemKind::Int8QTy);
    break;
  case Kinded::Kind::DequantizeNodeKind:
    isNodePrecisionSupported =
        (NI.getInElemTy(DequantizeNode::InputIdx) == ElemKind::Int8QTy) &&
        (NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::FloatTy ||
         NI.getOutElemTy(DequantizeNode::ResultIdx) == ElemKind::Float16Ty);
    break;
  case Kinded::Kind::RescaleQuantizedNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy});
    break;
  case Kinded::Kind::ConvertToNodeKind: {
    auto isConversionSupportedFor = [](ElemKind kindFrom, ElemKind kindTo) {
      switch (kindFrom) {
      case ElemKind::Float16Ty:
        switch (kindTo) {
        case ElemKind::FloatTy:
        case ElemKind::Int8QTy:
        case ElemKind::UInt8QTy:
        case ElemKind::BoolTy:
          return true;
        case ElemKind::Int32ITy:
#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 7
          return true;
#else
          return glow::nnpi::flags::EnableCustomIAKernels;
#endif // NNPI >= 1.7
        default:
          return false;
        }
        return false;

      case ElemKind::FloatTy:
        switch (kindTo) {
        case ElemKind::Float16Ty:
        case ElemKind::Int8QTy:
        case ElemKind::UInt8QTy:
        case ElemKind::BoolTy:
          return true;
        case ElemKind::Int32ITy:
          return glow::nnpi::flags::EnableCustomIAKernels;
        default:
          return false;
        }
        return false;

      case ElemKind::Int64ITy:
        switch (kindTo) {
        case ElemKind::Int32ITy:
        case ElemKind::FloatTy:
        case ElemKind::Int8QTy:
          return true;
        default:
          return false;
        }
        return false;

      // NOTE: this is supported by a custom kernel
      case ElemKind::BoolTy:
        switch (kindTo) {
        case ElemKind::Int32ITy:
          return true;
        default:
          return false;
        }
        return false;

      case ElemKind::Int32ITy:
        switch (kindTo) {
        case ElemKind::Int64ITy:
        case ElemKind::Float16Ty:
        case ElemKind::FloatTy:
        case ElemKind::Int8QTy:
          return true;
        case ElemKind::BoolTy:
          return glow::nnpi::flags::EnableCustomIAKernels;
        default:
          return false;
        }
        return false;

      case ElemKind::Int32QTy:
        switch (kindTo) {
        case ElemKind::Float16Ty:
          return true;
        default:
          return false;
        }
        return false;

      case ElemKind::UInt8QTy:
      case ElemKind::Int8QTy:
        return true;

      case ElemKind::UInt8FusedQTy:
        return (kindTo == ElemKind::Float16Ty ||
                kindTo == ElemKind::UInt8FusedFP16QTy);
      case ElemKind::UInt8FusedFP16QTy:
        return (kindTo == ElemKind::Float16Ty);
      default:
        return false;
      }
      return false;
    };
    isNodePrecisionSupported =
        isConversionSupportedFor(NI.getInElemTy(ConvertToNode::InputIdx),
                                 NI.getOutElemTy(ConvertToNode::ResultIdx));
    break;
  }

  case Kinded::Kind::DynamicQuantizedFullyConnectedNodeKind:
    isNodePrecisionSupported =
        (NI.getInElemTy(DynamicQuantizedFullyConnectedNode::InputIdx) ==
             ElemKind::Float16Ty ||
         NI.getInElemTy(DynamicQuantizedFullyConnectedNode::InputIdx) ==
             ElemKind::FloatTy) &&
        NI.getInElemTy(DynamicQuantizedFullyConnectedNode::WeightsIdx) ==
            ElemKind::Int8QTy &&
        NI.getInElemTy(DynamicQuantizedFullyConnectedNode::BiasIdx) ==
            ElemKind::FloatTy;
    break;

  case Kinded::Kind::DynamicRowwiseQuantizedFullyConnectedNodeKind:
    isNodePrecisionSupported =
        (NI.getInElemTy(DynamicRowwiseQuantizedFullyConnectedNode::InputIdx) ==
             ElemKind::Float16Ty ||
         NI.getInElemTy(DynamicRowwiseQuantizedFullyConnectedNode::InputIdx) ==
             ElemKind::FloatTy) &&
        NI.getInElemTy(DynamicRowwiseQuantizedFullyConnectedNode::WeightsIdx) ==
            ElemKind::Int8QTy &&
        NI.getInElemTy(DynamicRowwiseQuantizedFullyConnectedNode::BiasIdx) ==
            ElemKind::FloatTy &&
        NI.getInElemTy(DynamicRowwiseQuantizedFullyConnectedNode::ScalesIdx) ==
            ElemKind::FloatTy &&
        NI.getInElemTy(DynamicRowwiseQuantizedFullyConnectedNode::OffsetsIdx) ==
            ElemKind::Int32ITy;
    break;
  case Kinded::Kind::FullyConnectedNodeKind:
    if (!NI.getInTy(FullyConnectedNode::InputIdx)->isQuantizedType()) {
      isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::FloatTy, ElemKind::Float16Ty});
    } else {
      isNodePrecisionSupported =
          NI.allInputsAndOutputsHaveSameElemKind(
              {ElemKind::Int8QTy}, {FullyConnectedNode::BiasIdx}) &&
          ((NI.getInElemTy(FullyConnectedNode::BiasIdx) ==
            ElemKind::Int32QTy) ||
           (NI.getInElemTy(FullyConnectedNode::BiasIdx) == ElemKind::FloatTy));
    }
    break;
  case Kinded::Kind::MaxPoolNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
            {MaxPoolNode::ArgmaxIdx}) &&
        (NI.getOutElemTy(MaxPoolNode::ArgmaxIdx) == ElemKind::Int64ITy);
    break;
  case Kinded::Kind::TopKNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy}, {},
            {TopKNode::IndicesIdx}) &&
        (NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int64ITy ||
         NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int32ITy);
    break;
  case Kinded::Kind::GatherNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int64ITy,
             ElemKind::Int8QTy},
            {GatherNode::IndicesIdx}) &&
        ((NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int32ITy) ||
         (NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int64ITy));
    break;
  case Kinded::Kind::GatherRangesNodeKind:

    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::Int32ITy, ElemKind::Int64ITy},
            {GatherRangesNode::DataIdx}, {GatherRangesNode::OutputIdx}) &&
        ((NI.getInElemTy(GatherRangesNode::DataIdx) == ElemKind::FloatTy) ||
         (NI.getInElemTy(GatherRangesNode::DataIdx) == ElemKind::Float16Ty) ||
         (NI.getInElemTy(GatherRangesNode::DataIdx) == ElemKind::Int8QTy) ||
         (NI.getInElemTy(GatherRangesNode::DataIdx) == ElemKind::Int32ITy) ||
         (NI.getInElemTy(GatherRangesNode::DataIdx) == ElemKind::Int64ITy)) &&
        (NI.getOutElemTy(GatherRangesNode::OutputIdx) ==
         NI.getInElemTy(GatherRangesNode::DataIdx));
    break;
  case Kinded::Kind::SliceNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy, ElemKind::BoolTy});
    break;
  case Kinded::Kind::ReshapeNodeKind:

    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});
    break;
  case Kinded::Kind::CmpLTENodeKind:
  case Kinded::Kind::CmpLTNodeKind:
  case Kinded::Kind::CmpEQNodeKind:
  case Kinded::Kind::CmpNEQNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
             ElemKind::Int32ITy},
            {}, {CmpEQNode::ResultIdx}) &&
        (NI.getOutElemTy(CmpEQNode::ResultIdx) == ElemKind::BoolTy);
    break;
  case Kinded::Kind::NonZeroNodeKind:
    isNodePrecisionSupported =
        (NI.getOutElemTy(CmpEQNode::ResultIdx) == ElemKind::Int32ITy);
    break;
  case Kinded::Kind::SelectNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
            {SelectNode::CondIdx}) &&
        (NI.getInElemTy(SelectNode::CondIdx) == ElemKind::BoolTy);
    break;
  case Kinded::Kind::GaussianFillNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
             ElemKind::Int32ITy, ElemKind::Int64ITy},
            {}, {GaussianFillNode::ResultIdx}) &&
        (NI.getOutElemTy(GaussianFillNode::ResultIdx)) == ElemKind::Float16Ty;
    break;
  case Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind:
    isNodePrecisionSupported =
        (NI.getInElemTy(RowwiseQuantizedFullyConnectedNode::InputIdx) ==
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
    break;
  case Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind:
    isNodePrecisionSupported =
        (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::InputIdx) ==
         ElemKind::Int8QTy) &&
        (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::FilterIdx) ==
         ElemKind::Int8QTy) &&
        ((NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::BiasIdx) ==
          ElemKind::Int32QTy) ||
         (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::BiasIdx) ==
          ElemKind::FloatTy)) &&
        (NI.getInElemTy(ChannelwiseQuantizedConvolutionNode::FilterScalesIdx) ==
         ElemKind::FloatTy) &&
        (NI.getInElemTy(
             ChannelwiseQuantizedConvolutionNode::FilterOffsetsIdx) ==

         ElemKind::Int32ITy) &&
        (NI.getOutElemTy(ChannelwiseQuantizedConvolutionNode::ResultIdx) ==
         ElemKind::Int8QTy);
    break;
  case Kinded::Kind::SparseLengthsSumNodeKind:
    isNodePrecisionSupported =
        isSLSIndicesValid(NI.getInTy(SparseLengthsSumNode::IndicesIdx)) &&
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
    break;
  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    isNodePrecisionSupported =
        isSLSIndicesValid(
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
    break;
  case Kinded::Kind::EmbeddingNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
            {EmbeddingNode::IndicesIdx}) &&
        (NI.getInElemTy(EmbeddingNode::IndicesIdx) == ElemKind::Int64ITy ||
         NI.getInElemTy(EmbeddingNode::IndicesIdx) == ElemKind::Int32ITy);
    break;
  case Kinded::Kind::EmbeddingBagNodeKind:
    isNodePrecisionSupported =
        isSLSIndicesValid(NI.getInTy(EmbeddingBagNode::IndicesIdx)) &&
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
            {EmbeddingBagNode::IndicesIdx, EmbeddingBagNode::OffsetsIdx}) &&
        (NI.getInElemTy(EmbeddingBagNode::IndicesIdx) == ElemKind::Int64ITy ||
         NI.getInElemTy(EmbeddingBagNode::IndicesIdx) == ElemKind::Int32ITy) &&
        (NI.getInElemTy(EmbeddingBagNode::OffsetsIdx) == ElemKind::Int64ITy ||
         NI.getInElemTy(EmbeddingBagNode::OffsetsIdx) == ElemKind::Int32ITy);
    break;
  case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsNodeKind: {
    auto dataK = NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::DataIdx);
    auto offsetsK =
        NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::OffsetsIdx);
    auto indicesK =
        NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::IndicesIdx);
    auto resultK =
        NI.getOutElemTy(EmbeddingBagByteRowwiseOffsetsNode::ResultIdx);
    isNodePrecisionSupported =
        isSLSIndicesValid(
            NI.getInTy(EmbeddingBagByteRowwiseOffsetsNode::IndicesIdx)) &&
        (dataK == ElemKind::UInt8FusedQTy ||
         dataK == ElemKind::UInt8FusedFP16QTy ||
         dataK == ElemKind::UInt4FusedFP16QTy) &&
        (resultK == ElemKind::FloatTy || resultK == ElemKind::Float16Ty) &&
        (offsetsK == ElemKind::Int64ITy || offsetsK == ElemKind::Int32ITy) &&
        (indicesK == ElemKind::Int64ITy || indicesK == ElemKind::Int32ITy);

    break;
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
    isNodePrecisionSupported =
        isSLSIndicesValid(NI.getInTy(
            FusedRowwiseQuantizedSparseLengthsSumNode::IndicesIdx)) &&
        (dataK == ElemKind::UInt8FusedQTy ||
         dataK == ElemKind::UInt8FusedFP16QTy ||
         dataK == ElemKind::UInt4FusedFP16QTy) &&
        (resultK == ElemKind::FloatTy || resultK == ElemKind::Float16Ty) &&
        (indicesK == ElemKind::Int64ITy || indicesK == ElemKind::Int32ITy) &&
        (lengthsK == ElemKind::Int32ITy);
    break;
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
    isNodePrecisionSupported =
        isSLSIndicesValid(NI.getInTy(
            FusedRowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx)) &&
        (dataK == ElemKind::UInt8FusedQTy ||
         dataK == ElemKind::UInt8FusedFP16QTy ||
         dataK == ElemKind::UInt4FusedFP16QTy) &&
        (weightsK == ElemKind::FloatTy || weightsK == ElemKind::Float16Ty) &&
        (resultK == ElemKind::FloatTy || resultK == ElemKind::Float16Ty) &&
        (indicesK == ElemKind::Int64ITy || indicesK == ElemKind::Int32ITy) &&
        (lengthsK == ElemKind::Int32ITy);
  } break;
  case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    isNodePrecisionSupported =
        isSLSIndicesValid(NI.getInTy(
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
    break;
  case Kinded::Kind::ScatterDataNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
             ElemKind::UInt8QTy},
            {ScatterDataNode::IndicesIdx}) &&
        (NI.getInElemTy(ScatterDataNode::IndicesIdx) == ElemKind::Int32ITy ||
         NI.getInElemTy(ScatterDataNode::IndicesIdx) == ElemKind::Int64ITy);
    break;
  case Kinded::Kind::BucketizeNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::Float16Ty, ElemKind::Int8QTy, ElemKind::UInt8QTy}, {},
            {BucketizeNode::ResultIdx}) &&
        (NI.getOutElemTy(BucketizeNode::ResultIdx) == ElemKind::Int32ITy);
    break;
  case Kinded::Kind::SoftMaxNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
            {SoftMaxNode::SelectedIdx}) &&
        (NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int64ITy ||
         NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int32ITy);
    break;
  case Kinded::Kind::LengthsRangeFillNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});
    break;
  case Kinded::Kind::BatchOneHotNodeKind:

    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
             ElemKind::Int32ITy, ElemKind::Int64ITy},
            {BatchOneHotNode::LengthsIdx}) &&
        (NI.getInElemTy(BatchOneHotNode::LengthsIdx) == ElemKind::Int32ITy);
    break;
  case Kinded::Kind::NNPICustomDSPNodeKind:
  case Kinded::Kind::NNPICustomIANodeKind:
    isNodePrecisionSupported = true;
    break;
  case Kinded::Kind::SpaceToDepthNodeKind:
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});
    break;
  case Kinded::Kind::ArgMaxNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::Float16Ty, ElemKind::Int8QTy, ElemKind::Int32ITy,
             ElemKind::Int64ITy, ElemKind::BoolTy},
            {}, {ArgMaxNode::ResultIdx}) &&
        (NI.getOutElemTy(ArgMaxNode::ResultIdx) == ElemKind::Int64ITy ||
         NI.getOutElemTy(ArgMinNode::ResultIdx) == ElemKind::Int32ITy);
    break;
  case Kinded::Kind::ArgMinNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::Float16Ty, ElemKind::FloatTy, ElemKind::Int8QTy,
             ElemKind::Int32ITy, ElemKind::Int64ITy, ElemKind::BoolTy},
            {}, {ArgMinNode::ResultIdx}) &&
        (NI.getOutElemTy(ArgMinNode::ResultIdx) == ElemKind::Int64ITy ||
         NI.getOutElemTy(ArgMinNode::ResultIdx) == ElemKind::Int32ITy);
    break;
  case Kinded::Kind::LogitNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Float16Ty});
    break;
  case Kinded::Kind::CumSumNodeKind:
#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 7
    isNodePrecisionSupported = NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Int32ITy, ElemKind::Int8QTy, ElemKind::UInt8QTy});
#else
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});
#endif // NNPI >= 1.7
    break;
#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 9
  case Kinded::Kind::BatchSparseToDenseNodeKind:
    isNodePrecisionSupported =
        NI.allInputsAndOutputsHaveSameElemKind(
            {ElemKind::Float16Ty, ElemKind::UInt8QTy, ElemKind::Int8QTy},
            {BatchSparseToDenseNode::LengthsIdx,
             BatchSparseToDenseNode::IndicesIdx}) &&
        ((NI.getInElemTy(BatchSparseToDenseNode::LengthsIdx) ==
              ElemKind::Int64ITy ||
          NI.getInElemTy(BatchSparseToDenseNode::LengthsIdx) ==
              ElemKind::Int32ITy)) &&
        ((NI.getInElemTy(BatchSparseToDenseNode::IndicesIdx) ==
              ElemKind::Int64ITy ||
          NI.getInElemTy(BatchSparseToDenseNode::IndicesIdx) ==
              ElemKind::Int32ITy));
    break;
  case Kinded::Kind::FillExamplesWithIndicatorNodeKind:
    isNodePrecisionSupported =
        (NI.getInElemTy(FillExamplesWithIndicatorNode::DataIdx) ==
         NI.getOutElemTy(FillExamplesWithIndicatorNode::ResultIdx)) &&
        ((NI.getInElemTy(FillExamplesWithIndicatorNode::IndicatorIdx) ==
          ElemKind::Int32ITy) ||
         (NI.getInElemTy(FillExamplesWithIndicatorNode::IndicatorIdx) ==
          ElemKind::Int64ITy));
    break;
#endif // NNPI >= 1.9
  default:
    isNodeHasAnySupport = false;
    isNodePrecisionSupported = false;
  }

  if (isNodePrecisionSupported) {
    return NodeSupportLevels::PRECISION_SUPPORTED;
  } else if (isNodeHasAnySupport) {
    return NodeSupportLevels::SUPPORTED;
  }
  return NodeSupportLevels::NOT_SUPPORTED;
}

bool NNPIBackend::isOpSupported(const NodeInfo &NI) const {
  if (isNodeSupported(NI) != NodeSupportLevels::PRECISION_SUPPORTED) {
    LOG(ERROR) << "Unsupported op:\n" << NI.getDebugDesc();
    return false;
  }
  return true;
}

bool NNPIBackend::shouldLower(const Node *N) const {
  switch (N->getKind()) {
  case Kinded::Kind::ConvolutionNodeKind: {
    bool isDilated = false;
    const ConvolutionNode *convNode = llvm::dyn_cast<ConvolutionNode>(N);
    if (convNode && std::any_of(convNode->getDilation().begin(),
                                convNode->getDilation().end(),
                                [](unsigned_t i) { return i > 1; })) {
      isDilated = true;
    }
    return isDilated ||
           isConvolutionSameAsFullyConnected(llvm::cast<ConvolutionNode>(N),
                                             /* enforceInput1x1*/ true);
  } break;
  case Kinded::Kind::SparseLengthsSumNodeKind:
  case Kinded::Kind::BatchedMulNodeKind:
  case Kinded::Kind::BucketizeNodeKind:
    return false;
  default:
    break;
  }

  NodeInfo dummyNodeInfo(*N);
  return isNodeSupported(dummyNodeInfo) == NodeSupportLevels::NOT_SUPPORTED;
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

    if (auto *R = llvm::dyn_cast<RescaleQuantizedNode>(node)) {
      // For Rescales that are preceded by FC or Relu, mirror their
      // parallelization.
      Node *inputNode = R->getInput().getNode();
      if (!llvm::isa<FullyConnectedNode>(inputNode) &&
          !llvm::isa<ReluNode>(inputNode)) {
        continue;
      }
      auto numChunksIt = numChunks.find(inputNode);
      auto parOptsIt = parOpts.find(inputNode);
      if (numChunksIt == numChunks.end() || parOptsIt == parOpts.end()) {
        continue;
      }
      parOpts[R] = parOptsIt->second;
      numChunks[R] = numChunksIt->second;
      continue;
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

    // If we find parallelization info here then it means the node was not
    // parallelized; log and continue.
    bool skippedPar = nodeInfo.erase(parallelTransformKindKey);
    skippedPar |= nodeInfo.erase(numParallelChunksKey);
    LOG_IF(WARNING, skippedPar)
        << "Parallelization was skipped for Node: " << node.getDebugDesc();

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
    } else if (pKindStr == "Model_Axis1") {
      pKind = ParallelTransformKind::Model_Axis1;
    } else if (pKindStr == "Model_Axis2") {
      pKind = ParallelTransformKind::Model_Axis2;
    } else if (pKindStr == "Model_Axis3") {
      pKind = ParallelTransformKind::Model_Axis3;
    } else if (pKindStr == "Model_Axis4") {
      pKind = ParallelTransformKind::Model_Axis4;
    } else if (pKindStr == "Model_Axis5") {
      pKind = ParallelTransformKind::Model_Axis5;
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
    defaultNumParallelChunks = glow::nnpi::flags::NumParallelChunks;

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

    // Override basic parallelization for everything but Gelu
    auto it =
        opts.backendSpecificOpts.find(std::string("NNPIOnlyParallelizeGelu"));
    if (it != opts.backendSpecificOpts.end()) {
      for (const auto &pair : parOpts) {
        Node *N = pair.first;
        if (N->getKind() != Kinded::Kind::GeluNodeKind) {
          numChunks[N] = 1;
        }
      }
    }
  }

  RETURN_ERR_IF_NOT(numChunks.size() == parOpts.size(),
                    "Require that numChunks and parOpts have same size.");

  // No parallelization to do, so return early.
  if (numChunks.size() == 0) {
    return false;
  }

  int32_t defaultModelParallelSplitAlignment =
      glow::nnpi::flags::ModelParallelSplitAlignment;

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

  if (usePerNodeParallelizationSpec) {
    // If parallelization was based on backend-specific node info then
    // validate the new nodes that were added.
    RETURN_IF_ERR(validateBackendSpecificNodeInfo(
        F, replacedMap, opts.backendSpecificNodeInfo));
  } else if (numChunks.size() != replacedMap.size()) {
    for (const auto &pair : numChunks) {
      Node *N = pair.first;
      LOG_IF(WARNING, !replacedMap.count(N))
          << "Parallelization was skipped for Node: " << N->getDebugDesc();
    }
  }

  return true;
}

Expected<std::unique_ptr<CompiledFunction>>
NNPIBackend::compile(Function *F, const BackendOptions &opts) const {
  // Do some verification prior to final compilation. Check that for all
  // non-fused qparams, scales are not zero after FP16 conversion.
  for (const Node &N : F->getNodes()) {
    for (size_t i = 0, e = N.getNumResults(); i < e; i++) {
      const TypeRef resTy = N.getNthResult(i).getType();
      if (resTy->isQuantizedType() && !resTy->isFusedQuantizedType()) {
        RETURN_ERR_IF_NOT(float(float16_t(resTy->getScale())) != 0.f,
                          "Quantized type in node has zero FP16 scale: " +
                              N.getDebugDesc());
      }
    }
  }

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

  // Quantize Swish when wrapped in Quantize/Dequantize.
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
  pipeline->pushBack(FunctionPassID::EliminateClipsOutsideFP16Range);

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
  pipeline->pushBack(FunctionPassID::EliminateClipsOutsideFP16Range);

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
    bool shouldLowerNode = isNodeSupported(NI) == NodeSupportLevels::SUPPORTED;
    switch (N.getKind()) {
    case Kinded::Kind::BatchMatMulNodeKind:
      shouldLowerNode |= nnpi::flags::LowerAllBatchMatMul;
      break;
    case Kinded::Kind::ConvertToNodeKind: {
      shouldLowerNode = false;
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
      break;
    }
    case Kinded::Kind::GeluNodeKind: {
      auto it = cctx.backendOpts.backendSpecificOpts.find(
          std::string("NNPILowerAllGelu"));
      if (it != cctx.backendOpts.backendSpecificOpts.end()) {
        shouldLowerNode |= (it->second == "true");
      }
      break;
    }
    case Kinded::Kind::ReshapeNodeKind: {
      shouldLowerNode = false;
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
    case Kinded::Kind::SparseLengthsSumNodeKind: {
      shouldLowerNode = false;
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
      break;
    }
    if (shouldLowerNode) {
      changed |= lowerNode(F, &N, cctx);
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

/// Looks for LayernormNode that has int8 input with fp scale and bias and
/// quantize the scale and bias. This pass is a temporary workaround which
/// should be diabled after NNPI supports fp scale and bias.
bool quantizeLayernormScaleAndBias(Function *F) {
  bool changed = false;

  for (auto &N : F->getNodes()) {
    auto *LN = llvm::dyn_cast<LayerNormalizationNode>(&N);
    if (!LN) {
      continue;
    }

    auto in = LN->getInput();
    auto gamma = LN->getScale();
    auto beta = LN->getBias();

    // Skip if input is not quantized type or gamma is quantized type.
    if (!in.getType()->isQuantizedType() ||
        gamma.getType()->isQuantizedType()) {
      continue;
    }

    auto *gammaC = llvm::dyn_cast<Constant>(gamma);
    auto *betaC = llvm::dyn_cast<Constant>(beta);
    if (!gammaC || !betaC) {
      continue;
    }

    std::vector<TensorQuantizationParams> gammaTQP;
    std::vector<TensorQuantizationParams> betaTQP;
    if (gammaC->getElementType() == ElemKind::FloatTy) {
      gammaTQP = quantization::getTensorQuantizationParams(
          gammaC->getPayload(), quantization::Schema::Asymmetric,
          ElemKind::Int8QTy,
          /* qDim */ 0, /* qStep */ gamma.dims()[0]);
      betaTQP = quantization::getTensorQuantizationParams(
          betaC->getPayload(), quantization::Schema::Asymmetric,
          ElemKind::Int8QTy,
          /* qDim */ 0, /* qStep */ beta.dims()[0]);
    } else {
      gammaTQP = quantization::getTensorQuantizationParams(
          gammaC->getPayload().getCopyConvertedToType(ElemKind::FloatTy),
          quantization::Schema::Asymmetric, ElemKind::Int8QTy,
          /* qDim */ 0, /* qStep */ gamma.dims()[0]);
      betaTQP = quantization::getTensorQuantizationParams(
          betaC->getPayload().getCopyConvertedToType(ElemKind::FloatTy),
          quantization::Schema::Asymmetric, ElemKind::Int8QTy,
          /* qDim */ 0, /* qStep */ beta.dims()[0]);
    }

    auto *gammaQ = F->createQuantize(
        "layernorm_scale_quant", gamma,
        F->getParent()->uniqueType(ElemKind::Int8QTy, gamma.dims(),
                                   gammaTQP[0].scale, gammaTQP[0].offset));
    auto *betaQ = F->createQuantize(
        "layernorm_bias_quant", beta,
        F->getParent()->uniqueType(ElemKind::Int8QTy, beta.dims(),
                                   betaTQP[0].scale, betaTQP[0].offset));
    auto *QLN =
        F->createLayerNormalization("layernorm", LN->getResult().getType(), in,
                                    gammaQ, betaQ, LN->getEpsilon());
    LN->getResult().replaceAllUsesOfWith(QLN->getResult());

    changed = true;
  }

  return changed;
}

template <typename T>
void zeroOutEmbeddingTable(Tensor &tensor, const int32_t &padIdx) {
  auto handle = tensor.getHandle<T>();
  size_t base = handle.getElementPtr({static_cast<unsigned long>(padIdx)});
  for (unsigned i = 0; i < tensor.dims()[1]; i++) {
    handle.raw(base + i) = 0;
  }
}

/// Helper which looks for EmbeddingNode that has padIdx specified and lower it
/// to GatherNode. Since Gather doesn't support padIdx so we'll need to zero out
/// those index before creating GatherNode.
bool lowerEmbeddingToGather(Function *F) {
  bool changed = false;

  for (auto &N : F->getNodes()) {
    auto *EN = llvm::dyn_cast<EmbeddingNode>(&N);
    if (!EN) {
      continue;
    }

    auto weightsNV = EN->getWeights();
    auto padIdx = EN->getPadIdx();

    DCHECK(weightsNV.dims().size() == 2)
        << "Expect [Embedding] weight dimensions be 2, but got "
        << weightsNV.dims().size();
    DCHECK(!EN->getScale()) << "[Embedding] scale must be false";
    DCHECK(!EN->getSparse()) << "[Embedding] sparse must be false";

    auto *weightsConstant = llvm::dyn_cast<glow::Constant>(weightsNV.getNode());

    // If weightsConstant is not available, probably means we are doing AOT
    if (!weightsConstant) {
      continue;
    }

    // Zero out embedding table if padIdx is not -1
    if (padIdx != -1) {
      // If embedding table only has one user, we don't make additional copies
      if (weightsConstant->hasOneUse()) {
        auto &weightsTensorNew = weightsConstant->getPayloadMutable();

        if (weightsTensorNew.getElementType() == ElemKind::FloatTy) {
          zeroOutEmbeddingTable<float>(weightsTensorNew, padIdx);
        } else if (weightsTensorNew.getElementType() == ElemKind::Float16Ty) {
          zeroOutEmbeddingTable<float16_t>(weightsTensorNew, padIdx);
        } else {
          LOG(ERROR)
              << "Unsupported Embedding weight Elemtype for transformation: "
              << Type::getElementName(weightsTensorNew.getElementType()).str();
        }
      } else { // Embedding table has more than one user
        auto weightsTensorNew = weightsConstant->getPayload().clone();

        if (weightsTensorNew.getElementType() == ElemKind::FloatTy) {
          zeroOutEmbeddingTable<float>(weightsTensorNew, padIdx);
        } else if (weightsTensorNew.getElementType() == ElemKind::Float16Ty) {
          zeroOutEmbeddingTable<float16_t>(weightsTensorNew, padIdx);
        } else {
          LOG(ERROR)
              << "Unsupported Embedding weight Elemtype for transformation: "
              << Type::getElementName(weightsTensorNew.getElementType()).str();
        }

        weightsConstant = F->getParent()->createConstant(
            "PaddedEmbeddingWeights", std::move(weightsTensorNew));
      }
    }

    auto *GN = F->createGather("embedding", weightsConstant, EN->getIndices());
    EN->getResult().replaceAllUsesOfWith(GN);
    changed = true;
  }

  return changed;
}

/// Helper which looks for Quantized Conv whose kernel is smaller than stride in
/// one or more dimension. These kernels will be padded to at least stride size.
static bool padKernelToStride(Function *F) {
  bool changed = false;
  for (auto &N : F->getNodes()) {
    if (N.getKind() == Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind) {
      auto *glowChannelwiseQuantizedConv =
          llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(&N);
      auto filterNodeValue = glowChannelwiseQuantizedConv->getFilter();
      auto filterOffsetNodeValue =
          glowChannelwiseQuantizedConv->getFilterOffsets();

      auto *filterConstant =
          llvm::dyn_cast<glow::Constant>(filterNodeValue.getNode());

      // Show whether there is kernel < stride in any dimension of N.
      bool isKernelPadded = false;

      // Quantized Conv's attributes.
      const uint32_t SPATIAL_DIMS2 = 2;
      uint32_t kernel[SPATIAL_DIMS2] = {
          glowChannelwiseQuantizedConv->getKernels()[0],
          glowChannelwiseQuantizedConv->getKernels()[1]};
      // Kernel size before padding.
      uint32_t kernelOrig[SPATIAL_DIMS2] = {kernel[0], kernel[1]};
      uint32_t paddingStart[SPATIAL_DIMS2] = {
          glowChannelwiseQuantizedConv->getPads()[0],
          glowChannelwiseQuantizedConv->getPads()[1]};
      uint32_t paddingEnd[SPATIAL_DIMS2] = {
          glowChannelwiseQuantizedConv->getPads()[2],
          glowChannelwiseQuantizedConv->getPads()[3]};
      uint32_t stride[SPATIAL_DIMS2] = {
          glowChannelwiseQuantizedConv->getStrides()[0],
          glowChannelwiseQuantizedConv->getStrides()[1]};

      bool is1x1s2Case = true;
      // This is for special case of 1x1 stride 2.
      for (int i = 0; i < SPATIAL_DIMS2; i++) {
        is1x1s2Case &= (kernel[i] == 1 && stride[i] == 2);
      }

      if (is1x1s2Case) {
        continue;
      }

      for (int i = 0; i < SPATIAL_DIMS2; i++) {
        if (kernel[i] < stride[i]) {
          isKernelPadded = true;
          // Pad the kernel to make it not smaller than stride.

          // First we need to make sure inputSize[i + 1] % stride[i] == 0
          // inputSize[i + 1] is for NHWC layout.
          // If it is not, we maybe need to pad the input.
          uint32_t inputSize =
              glowChannelwiseQuantizedConv->getInput().getType()->dims()[i + 1];
          uint32_t paddedInputSize =
              inputSize + paddingStart[i] + paddingEnd[i];
          // inputSize[i + 1] % stride[i] need to be greater than original
          // kernel size to generate one more output pixel. In this case, we pad
          // to make sure output size is still correct.
          if (paddedInputSize % stride[i] >= kernel[i]) {
            // We pad at the end.
            paddingEnd[i] += stride[i] - (paddedInputSize % stride[i]);
          }

          // Then we expand the kernel to at least stride size.
          kernel[i] = stride[i];
        }
      }

      // Create a new filter tensor that overwrites the old one in
      // filterConstant.
      if (isKernelPadded) {
        changed = true;
        // Set new pad size and kernel size.
        glowChannelwiseQuantizedConv->setPads(
            {paddingStart[0], paddingStart[1], paddingEnd[0], paddingEnd[1]});
        glowChannelwiseQuantizedConv->setKernels({kernel[0], kernel[1]});

        glow::Tensor filterTensorOrig = filterConstant->getPayload().clone();
        glow::Tensor filterOffsetTensor =
            llvm::dyn_cast<glow::Constant>(filterOffsetNodeValue.getNode())
                ->getPayload()
                .clone();
        glow::Tensor filterTensorNew(glow::ElemKind::Int8QTy,
                                     {filterTensorOrig.dims()[0], kernel[0],
                                      kernel[1], filterTensorOrig.dims()[3]},
                                     1.0, 0);
        filterTensorNew.zero();

        for (unsigned nn = 0; nn < filterTensorNew.dims()[0]; nn++) {
          // Currently we only support padding weights offset are all zeros on
          // card.
          int32_t offset = filterOffsetTensor.getHandle<int32_t>().at({nn});
          LOG_AND_RETURN_IF(ERROR, offset != 0,
                            "Weight offset should be all zeros", false);
          // Also, we will not be able to pad the kernel if offset is too big,
          // even if weights offset is supported in the future.
          LOG_AND_RETURN_IF(
              ERROR, offset < INT8_MIN || offset > INT8_MAX,
              "Fatal error: offset exceed int8 limit while padding "
              "kernel. Please contact Glow team to solve.",
              false);
          for (unsigned hh = 0; hh < filterTensorNew.dims()[1]; hh++) {
            for (unsigned ww = 0; ww < filterTensorNew.dims()[2]; ww++) {
              for (unsigned cc = 0; cc < filterTensorNew.dims()[3]; cc++) {
                // If the content is in original area, we just simply copy it.
                // Or else we pad it with the value of offset.
                if (hh < kernelOrig[0] && ww < kernelOrig[1]) {
                  filterTensorNew.getHandle<int8_t>().at({nn, hh, ww, cc}) =
                      filterTensorOrig.getHandle<int8_t>().at({nn, hh, ww, cc});
                } else {
                  filterTensorNew.getHandle<int8_t>().at({nn, hh, ww, cc}) =
                      static_cast<int8_t>(offset);
                }
              }
            }
          }
        }
        auto filterConstantNew = F->getParent()->createConstant(
            "StridePaddedFilter", std::move(filterTensorNew));
        // Set filter.
        N.setNthInput(ChannelwiseQuantizedConvolutionNode::FilterIdx,
                      filterConstantNew->getOutput());
      }
    }
  }
  return changed;
}

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

// Replace inefficient Concat Nodes with Reshape->Concat->Transpose.
// For Concat Nodes concatenating inputs on the innermost dimension, if the
// innermost dimension size of the inputs is small (e.g., 1), the corresponding
// memory access is inefficient. Therefore, those Concat Nodes need to be
// replaced by Reshape->Concat->Transpose.
// Example:
//   Original: Inputs[4096x1]->Concat[4096x50]
//   New:      Inputs[4096x1]->Reshape[1x4096]->Concat[50x4096]
//                           ->Transpose[4096x50]
static bool replaceInefficientConcat(Function *F) {
  bool changed = false;
  // This optimization is applied conservatively to avoid causing potential
  // perf regression on other models.
  const int targetInputNumDims = 2;
  const int targetInputLastDim = 1;
  const int targetInputOtherDim = 4096;
  for (auto &N : F->getNodes()) {
    auto *CN = llvm::dyn_cast<ConcatNode>(&N);
    if (!CN) {
      continue;
    }
    bool isTargetConcat = true;
    auto origInputs = CN->getInputs();
    // Check whether the Concat Node is a target
    if ((origInputs[0].dims().size() != targetInputNumDims) ||
        (CN->getDim() != (targetInputNumDims - 1)) ||
        (origInputs[0].dims()[0] < targetInputOtherDim)) {
      isTargetConcat = false;
    } else {
      for (auto &input : origInputs) {
        const auto &origInputDims = input.dims();
        if (origInputDims[origInputDims.size() - 1] != targetInputLastDim) {
          isTargetConcat = false;
          break;
        }
      }
    }
    if (!isTargetConcat) {
      continue;
    }
    // Insert Reshape Nodes
    std::vector<NodeValue> reshapes(origInputs.size());
    for (int idx = 0; idx < origInputs.size(); ++idx) {
      const std::vector<dim_t> newInputDims = {origInputs[idx].dims()[1],
                                               origInputs[idx].dims()[0]};
      DCHECK(idx < reshapes.size()) << "out-of-range idx";
      reshapes[idx] = F->createReshape(
          origInputs[idx].getNode()->getName().str() + "_reshaped",
          origInputs[idx].getNode(), newInputDims);
    }
    // Create new Concat Node
    auto *newCN =
        F->createConcat(CN->getName().str() + "_for_tranpose", reshapes, 0);
    // Insert Transpose Node
    auto *newTN = F->createTranspose(CN->getName().str(), newCN, {1, 0});
    CN->getResult().replaceAllUsesOfWith(newTN->getResult());
    changed = true;
  }
  return changed;
}

Expected<bool>
NNPIBackend::transformPostOptPipeline(Function *F,
                                      CompilationContext &cctx) const {
  bool changed;
  ASSIGN_VALUE_OR_RETURN_ERR(changed, parallelizeFunction(F, cctx.backendOpts));
  if (changed) {
    // Use the normal NNPI-specific optimization pipeline, but without sinking
    // conversions, because we just parallelized and so don't want to undo any
    // parallelization we performed on quantizes/dequantizes.
    auto P = getOptimizationPipeline();
    P->removeAllInstancesOfPass(FunctionPassID::SinkConversions);
    P->removeAllInstancesOfPass(FunctionPassID::SinkConcatBelowQuantize);
    P->removeAllInstancesOfPass(FunctionPassID::MergeMatMulOnLHS);
    P->removeAllInstancesOfPass(FunctionPassID::MergeMatMulOnRHS);
    // Do not re-merge ConcatNodes, as we may be parallelizing them.
    const bool restoreMerge = cctx.optimizationOpts.skipConcatMerging;
    cctx.optimizationOpts.skipConcatMerging = true;

    FunctionPassManager("NNPI_transformPostOptPipeline", std::move(P), this)
        .run(F, cctx);

    cctx.optimizationOpts.skipConcatMerging = restoreMerge;
  }

  // Swap existing nodes for custom NNPI-specific LUT nodes here. This way,
  // the parallelization can occur on Glow nodes prior to the swap.
  bool changedFromSwap;
  ASSIGN_VALUE_OR_RETURN_ERR(changedFromSwap, swapInSpecializedLUT(F, cctx));
  changed = changed | changedFromSwap;
  return changed;
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

  if (glow::nnpi::flags::DisableTransforms) {
    return false;
  }

  bool changed = false;
  changed |= removeClipsBlockingFusion(F);
  changed |= padKernelToStride(F);
  changed |= lowerEmbeddingToGather(F);

// NNPI support fp16 scale and bias after 1.5
#if NNPI_MAJOR_VERSION == 1 && NNPI_MINOR_VERSION < 5
  changed |= quantizeLayernormScaleAndBias(F);
#endif

  changed |= replaceInefficientConcat(F);
  auto it =
      cctx.backendOpts.backendSpecificOpts.find("NNPI_ZeroScaleFP16Replace");
  if (it != cctx.backendOpts.backendSpecificOpts.end() && it->second == "1") {
    FunctionPassManager FPM(
        "NNPI_ZeroScaleFP16Replace",
        {FunctionPassID::ReplaceZeroScaleFP16QuantNodes, getDCEPassConfig()},
        this);
    changed |= FPM.run(F, cctx);
  }
  changed |= lowerRequiredNodes(F, cctx);

#if FACEBOOK_INTERNAL
  if (!(glow::nnpi::flags::EnableCustomDSPKernels ||
        glow::nnpi::flags::EnableCustomIAKernels)) {
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

unsigned NNPIBackend::getContextCount(CompilationContext &cctx) const {
  if (cctx.enableP2P || cctx.enableDRT) {
    return cctx.maxActiveRequestsPerInstance;
  } else {
    auto opts = NNPICompilationOptions(cctx.backendOpts.backendSpecificOpts);
    return opts.numWorkers;
  }
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
      RETURN_IF_ERR(nnpiDM->bindContext(dagNode->name, ctx, phUsage));
    }
  }

  if (backendOptions_.dumpRuntime) {
    DotWriter::writeToFile(root->name);
  }

  return Error::success();
}

/// Partial update of the NNPITensorDesc. Some members are ignored as they're
/// not used for estimation.
static bool updateDescForEstimate(NNPITensorDesc &desc, const glow::TypeRef ty,
                                  bool alternativeLayout) {
  LOG_AND_RETURN_IF(ERROR, ty == nullptr, "Invalid type", false);

  // Update dims and layout.
  NNPIImporter::updateDescDimsFromGlow(ty->dims(), desc, alternativeLayout);

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
                                      const std::vector<glow::TypeRef> types,
                                      bool alternativeLayout = false) {
  if (descs.size() != types.size()) {
    return false;
  }
  bool retVal = true;
  for (size_t i = 0; i < descs.size(); i++) {
    if (types.at(i) != nullptr) {
      retVal &=
          updateDescForEstimate(descs.at(i), types.at(i), alternativeLayout);
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

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 7
static bool isBatchNormUsingAlternativeLayout(const size_t channelIdx,
                                              const size_t numDims) {
  // Handle NNPI_LAYOUT_NDHWC and NNPI_LAYOUT_NHWC layout.
  if ((numDims == 4 || numDims == 5) && (channelIdx == numDims - 1)) {
    return true;
  }

  // Handle NNPI_LAYOUT_CN layout.
  if (numDims == 2 && channelIdx == 0) {
    return true;
  }

  return false;
}

double NNPIBackend::estimateBatchNormalizationNode(
    const BatchNormalizationNode *BN) const {

  NodeInfo NI(*BN);
  if (!isOpSupported(NI)) {
    return -1.0;
  }

  auto inputDims = BN->getInput().getType()->dims().size();
  auto alternativeLayout =
      isBatchNormUsingAlternativeLayout(BN->getChannelIdx(), inputDims);

  std::vector<NNPITensorDesc> descs(1); // only input for this node type

  LOG_AND_RETURN_IF(ERROR,
                    !updateDescListForEstimate(
                        descs, {NI.getInTy(BatchNormalizationNode::InputIdx)},
                        alternativeLayout),
                    "Failed to update NNPITensorDesc", -1.0);

  double estimate = -1.0;
  const NNPITensorDesc *td = &(descs.at(0));

  LOG_NNPI_IF_ERROR(
      nnpiEstimateOp(
          (BN->getName().str()).c_str(),
          NNPI_COST_MODEL_OP_TYPE::NNPI_COST_MODEL_BATCH_NORMALIZATION,
          0, /* subType is don't care for this node */
          &td, 1, &estimate),
      "Failed to estimate BatchNormalization op.");

  return estimate;
}

double NNPIBackend::estimateAvgPoolNode(const AvgPoolNode *avgPoolNode) const {
  NodeInfo NI(*avgPoolNode);
  if (!isOpSupported(NI)) {
    return -1.0;
  }

  // Update kernel Descriptors.
  // These are partially filled with only 'numDims' and 'dims[]'.
  NNPITensorDesc kernelDesc, strideDesc, padStartDesc, padEndDesc;
  auto numKernelDims = avgPoolNode->getKernels().size();
  kernelDesc.numDims = numKernelDims;
  strideDesc.numDims = numKernelDims;
  padStartDesc.numDims = numKernelDims;
  padEndDesc.numDims = numKernelDims;

  // Layout of kernels/stride etc. seems to be CHW/HW.
  // This is what is used by NNPI API.
  for (size_t i = 0; i < numKernelDims; i++) {
    kernelDesc.dims[i] = avgPoolNode->getKernels()[i];
    strideDesc.dims[i] = avgPoolNode->getStrides()[i];

    if (numKernelDims == 2) {
      padStartDesc.dims[i] = avgPoolNode->getPads()[i];
      padEndDesc.dims[i] = avgPoolNode->getPads()[numKernelDims + i];
    } else {
      padStartDesc.dims[i] = avgPoolNode->getPads()[i * 2];
      padEndDesc.dims[i] = avgPoolNode->getPads()[i * 2 + 1];
    }
  }

  // Update IO Descriptors.
  std::vector<NNPITensorDesc> ioDescs(2);
  LOG_AND_RETURN_IF(
      ERROR,
      !updateDescListForEstimate(ioDescs,
                                 {NI.getInTy(AvgPoolNode::InputIdx),
                                  NI.getOutTy(AvgPoolNode::ResultIdx)},
                                 true /* alternativeLayout */),
      "Failed to update NNPITensorDesc", -1.0);

  // The order of init in this array is _important_. This is
  // the order the NNPI API expects the values for this Op.
  // Any change here needs an update there also.
  const uint64_t numTotalDescs = 6;
  const NNPITensorDesc *td[numTotalDescs] = {&(ioDescs.at(0)), &(ioDescs.at(1)),
                                             &kernelDesc,      &strideDesc,
                                             &padStartDesc,    &padEndDesc};

  double estimate = -1.0;
  nnpiEstimateOp((avgPoolNode->getName().str()).c_str(),
                 NNPI_COST_MODEL_OP_TYPE::NNPI_COST_MODEL_AVG_POOL,
                 0, /* subType is don't care for this node */
                 td, numTotalDescs, &estimate);

  return estimate;
}

#endif // NNPI >= 1.7

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 8
double NNPIBackend::estimateLayerNormalizationNode(
    const LayerNormalizationNode *LN) const {

  NodeInfo NI(*LN);
  if (!isOpSupported(NI)) {
    return -1.0;
  }

  std::vector<NNPITensorDesc> descs(3); // only input for this node type

  LOG_AND_RETURN_IF(ERROR,
                    !updateDescListForEstimate(
                        descs,
                        {NI.getInTy(LayerNormalizationNode::InputIdx),
                         NI.getOutTy(LayerNormalizationNode::ResultIdx),
                         NI.getInTy(LayerNormalizationNode::ScaleIdx)},
                        false /* alternativeLayout */),
                    "Failed to update NNPITensorDesc", -1.0);

  const uint64_t numTotalDescs = 3;
  const NNPITensorDesc *td[numTotalDescs] = {&(descs.at(0)), &(descs.at(1)),
                                             &(descs.at(2))};

  double estimate = -1.0;
  LOG_NNPI_IF_ERROR(
      nnpiEstimateOp(
          (LN->getName().str()).c_str(),
          NNPI_COST_MODEL_OP_TYPE::NNPI_COST_MODEL_LAYER_NORMALIZATION,
          0, /* subType is don't care for this node */
          td, 3, &estimate),
      "Failed to estimate LayerNormalization op.");

  return estimate;
}

template <class EltwiseNodeType, NNPI_ELTWISE_TYPE eltwiseType>
double NNPIBackend::estimateBinaryEltwiseNode(const glow::Node *n) const {

  auto *glowEltwise = llvm::dyn_cast<EltwiseNodeType>(n);

  NodeInfo NI(*glowEltwise);
  if (!isOpSupported(NI)) {
    return -1.0;
  }

  std::vector<NNPITensorDesc> descs(3); // 2 inputs and 1 output

  LOG_AND_RETURN_IF(
      ERROR,
      !updateDescListForEstimate(descs,
                                 {NI.getInTy(EltwiseNodeType::RHSIdx),
                                  NI.getInTy(EltwiseNodeType::LHSIdx),
                                  NI.getOutTy(EltwiseNodeType::ResultIdx)},
                                 false /* alternativeLayout */),
      "Failed to update NNPITensorDesc", -1.0);

  const uint64_t numTotalDescs = 3;
  const NNPITensorDesc *td[numTotalDescs] = {&(descs.at(0)), &(descs.at(1)),
                                             &(descs.at(2))};

  double estimate = -1.0;
  LOG_NNPI_IF_ERROR(
      nnpiEstimateOp((glowEltwise->getName().str()).c_str(),
                     NNPI_COST_MODEL_OP_TYPE::NNPI_COST_MODEL_ELTWISE_OPS,
                     eltwiseType, td, 3, &estimate),
      "Failed to estimate Binary Eltwise op.");

  return estimate;
}

template <class EltwiseNodeType, NNPI_ELTWISE_TYPE eltwiseType>
double NNPIBackend::estimateUnaryEltwiseNode(const glow::Node *n) const {

  auto *glowEltwise = llvm::dyn_cast<EltwiseNodeType>(n);

  NodeInfo NI(*glowEltwise);
  if (!isOpSupported(NI)) {
    return -1.0;
  }

  std::vector<NNPITensorDesc> descs(2); // 1 input and 1 output

  LOG_AND_RETURN_IF(
      ERROR,
      !updateDescListForEstimate(descs,
                                 {NI.getInTy(EltwiseNodeType::InputIdx),
                                  NI.getOutTy(EltwiseNodeType::ResultIdx)},
                                 false /* alternativeLayout */),
      "Failed to update NNPITensorDesc", -1.0);

  const uint64_t numTotalDescs = 2;
  const NNPITensorDesc *td[numTotalDescs] = {&(descs.at(0)), &(descs.at(1))};

  double estimate = -1.0;
  LOG_NNPI_IF_ERROR(
      nnpiEstimateOp((glowEltwise->getName().str()).c_str(),
                     NNPI_COST_MODEL_OP_TYPE::NNPI_COST_MODEL_ELTWISE_OPS,
                     eltwiseType, td, 2, &estimate),
      "Failed to estimate Unary Eltwise op.");
  return estimate;
}

#endif // NNPI >= 1.8

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
#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 7
  case Kinded::Kind::BatchNormalizationNodeKind: {
    const BatchNormalizationNode *BN = llvm::cast<BatchNormalizationNode>(node);
    returnCost = estimateBatchNormalizationNode(BN);
    break;
  }
  case Kinded::Kind::AvgPoolNodeKind: {
    const AvgPoolNode *avgPoolNode = llvm::cast<AvgPoolNode>(node);
    returnCost = estimateAvgPoolNode(avgPoolNode);
    break;
  }
#endif // NNPI >= 1.7

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 8
  case Kinded::Kind::LayerNormalizationNodeKind: {
    const LayerNormalizationNode *LN = llvm::cast<LayerNormalizationNode>(node);
    returnCost = estimateLayerNormalizationNode(LN);
    break;
  }
  case Kinded::Kind::AddNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::AddNode, NNPI_ELTWISE_ADD>(node);
    break;
  }
  case Kinded::Kind::MulNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::MulNode, NNPI_ELTWISE_MUL>(node);
    break;
  }
  case Kinded::Kind::DivNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::DivNode, NNPI_ELTWISE_DIV>(node);
    break;
  }
  case Kinded::Kind::SubNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::SubNode, NNPI_ELTWISE_SUB>(node);
    break;
  }
  case Kinded::Kind::PowNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::PowNode, NNPI_ELTWISE_POW>(node);
    break;
  }
  case Kinded::Kind::FmodNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::FmodNode, NNPI_ELTWISE_MODULO>(node);
    break;
  }
  case Kinded::Kind::CmpEQNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::CmpEQNode, NNPI_ELTWISE_EQ>(node);
    break;
  }
  case Kinded::Kind::CmpLTENodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::CmpLTENode, NNPI_ELTWISE_LTE>(node);
    break;
  }
  case Kinded::Kind::CmpLTNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::CmpLTNode, NNPI_ELTWISE_LESS>(node);
    break;
  }
  case Kinded::Kind::MaxNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::MaxNode, NNPI_ELTWISE_MAX>(node);
    break;
  }
  case Kinded::Kind::MinNodeKind: {
    returnCost =
        estimateBinaryEltwiseNode<glow::MinNode, NNPI_ELTWISE_MIN>(node);
    break;
  }
  case Kinded::Kind::ExpNodeKind: {
    returnCost =
        estimateUnaryEltwiseNode<glow::ExpNode, NNPI_ELTWISE_EXP>(node);
    break;
  }
  case Kinded::Kind::AbsNodeKind: {
    returnCost =
        estimateUnaryEltwiseNode<glow::AbsNode, NNPI_ELTWISE_ABS>(node);
    break;
  }
  case Kinded::Kind::NegNodeKind: {
    returnCost =
        estimateUnaryEltwiseNode<glow::NegNode, NNPI_ELTWISE_NEG>(node);
    break;
  }
#endif // NNPI >= 1.8

  default:
    break;
  }
  RETURN_ERR_IF_NOT(returnCost >= 0.0,
                    strFormat("Estimate not supported for Node kind %s",
                              Kinded::getKindName(node->getKind())));
  return returnCost;
}
