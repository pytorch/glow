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
#include "NNPICompiledFunction.h"
#include "NNPIDeviceManager.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/GraphOptimizerPipeline/Pipeline.h"
#include "glow/Optimizer/Lower/Lower.h"
#include "llvm/Support/CommandLine.h"

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

namespace onnxifi {

bool GlowDumpGraph = false;
bool GlowDisableNNPITransforms = false;
bool GlowDisableNNPIPrivateTransforms = false;
int32_t GlowNNPINumParallelChunks = 1;

} // namespace onnxifi
} // namespace glow

NNPIBackendOptions NNPIBackend::backendOptions_;

bool NNPIBackend::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {
  case Kinded::Kind::ClipNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::Float16Ty, ElemKind::Int8QTy});

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
  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::BatchedAddNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::ExpNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::BatchMatMulNodeKind:
  case Kinded::Kind::PReluNodeKind:
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

  case Kinded::Kind::SparseLengthsSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
               {SparseLengthsSumNode::IndicesIdx,
                SparseLengthsSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsSumNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(SparseLengthsSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);
  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Float16Ty, ElemKind::Int8QTy},
               {SparseLengthsWeightedSumNode::IndicesIdx,
                SparseLengthsWeightedSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
            ElemKind::Int64ITy) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

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
            dataK == ElemKind::UInt8FusedFP16QTy) &&
           (resultK == ElemKind::FloatTy || resultK == ElemKind::Float16Ty) &&
           (indicesK == ElemKind::Int64ITy) && (lengthsK == ElemKind::Int32ITy);
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
           (weightsK == resultK) &&
           (resultK == ElemKind::FloatTy || resultK == ElemKind::Float16Ty) &&
           (indicesK == ElemKind::Int64ITy) && (lengthsK == ElemKind::Int32ITy);
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
            ElemKind::Int64ITy) &&
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
    return false;
  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind: {
    const FusedRowwiseQuantizedSparseLengthsSumNode *SLSN =
        llvm::cast<FusedRowwiseQuantizedSparseLengthsSumNode>(N);
    if ((backendOptions_.useIceT || backendOptions_.inferOnDevice) &&
        (SLSN->getResult().getElementType() == ElemKind::Float16Ty)) {
      return false; // Don't lower == keep without weights
    } else {
      return true;
    }
  }
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
  return createNNPIDeviceManager(deviceConfig);
}

Expected<std::unique_ptr<CompiledFunction>>
NNPIBackend::compile(Function *F, const BackendOptions &opts) const {
  if (glow::onnxifi::GlowDumpGraph) {
    std::string fname = "Graph_" + F->getName().str() + ".dot";
    LOG(INFO) << "Dumping net to " << fname;
    F->dumpDAG(fname);
  }
  std::unique_ptr<NNPICompiledFunction> compiledFunc =
      glow::make_unique<NNPICompiledFunction>(F);
  auto compileHasError = compiledFunc->compile(F, opts);
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

bool NNPIBackend::transformPostLowering(Function *F,
                                        CompilationContext &cctx) const {
  LOG_SCOPE(F->getLogContext(), "NNPIBackend::transformPostLowering");

  if (glow::onnxifi::GlowDisableNNPITransforms) {
    return false;
  }

  bool changed = removeClipsBlockingFusion(F);
  changed |= lowerRequiredNodes(F, cctx);

#if FACEBOOK_INTERNAL
  if (glow::onnxifi::GlowDisableNNPIPrivateTransforms) {
    return changed;
  }

  if (glow::onnxifi::GlowNNPINumParallelChunks > 1) {
    cctx.backendOpts.backendSpecificOpts["NNPINumParallelChunks"] =
        std::to_string(glow::onnxifi::GlowNNPINumParallelChunks);
  }
  changed |= transformPrivate(F, cctx);
#endif /* FACEBOOK_INTERNAL */

  return changed;
}
