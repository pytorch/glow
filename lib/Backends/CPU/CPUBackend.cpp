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

#include "CPUBackend.h"
#include "CPUFunction.h"
#include "CPULLVMIRGen.h"

#include "glow/Backend/BackendUtils.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

using namespace glow;

/// We compile the standard library (libjit) to LLVM bitcode, and then convert
/// that binary data to an include file using an external utility (include-bin).
/// The resulting file is included here to compile the bitcode image into our
/// library.
static const unsigned char libjit_bc[] = {
#include "glow/CPU/libjit_bc.inc"
};
static const size_t libjit_bc_size = sizeof(libjit_bc);

bool CPUBackend::isOpSupported(const NodeInfo &NI) const {
  // Note: For brevity below, "X ==> Y, Z" signifes that Node X is IRGen'd into
  // Instructions Y and Z.
  switch (NI.getKind()) {
  case Kinded::Kind::BatchedReduceMinNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int32ITy, ElemKind::Int64ITy});

  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MinNodeKind:
  case Kinded::Kind::CPUMaxSplatNodeKind:
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
           (NI.getOutElemTy(MaxPoolNode::ArgmaxIdx) == IndexElemKind);

  case Kinded::Kind::ArgMaxNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
               {ArgMaxNode::ArgmaxIdx}) &&
           (NI.getOutElemTy(ArgMaxNode::ArgmaxIdx) == IndexElemKind);

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
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, IndexElemKind,
         ElemKind::Int64ITy, ElemKind::BoolTy});

  case Kinded::Kind::DivNodeKind:
  case Kinded::Kind::SliceNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int32QTy,
         ElemKind::Int64ITy});
  case Kinded::Kind::SpaceToDepthNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy});

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
            IndexElemKind) &&
           (NI.getInElemTy(SparseLengthsSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
               {SparseLengthsWeightedSumNode::IndicesIdx,
                SparseLengthsWeightedSumNode::LengthsIdx}) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::IndicesIdx) ==
            IndexElemKind) &&
           (NI.getInElemTy(SparseLengthsWeightedSumNode::LengthsIdx) ==
            ElemKind::Int32ITy);

  case Kinded::Kind::EmbeddingBagNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
               {EmbeddingBagNode::IndicesIdx, EmbeddingBagNode::OffsetsIdx}) &&
           (NI.getInElemTy(EmbeddingBagNode::IndicesIdx) == IndexElemKind) &&
           (NI.getInElemTy(EmbeddingBagNode::OffsetsIdx) == IndexElemKind);

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
            IndexElemKind) &&
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
            IndexElemKind) &&
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
  case Kinded::Kind::CPUConvDKKC8NodeKind:
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
            IndexElemKind) &&
           (NI.getInElemTy(
                MaxPoolGradNode::GradOfOriginalOutputNamedArgmaxIdx) ==
            IndexElemKind);

  case Kinded::Kind::ConvolutionNodeKind:
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});
    }

    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy},
                                                  {ConvolutionNode::BiasIdx}) &&
           (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int8QTy ||
            NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy);

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
               {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy},
               {GatherNode::IndicesIdx}) &&
           ((NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int32ITy) ||
            (NI.getInElemTy(GatherNode::IndicesIdx) == IndexElemKind));

  case Kinded::Kind::GatherRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy},
               {GatherRangesNode::RangesIdx}, {GatherRangesNode::LengthsIdx}) &&
           ((NI.getInElemTy(GatherRangesNode::RangesIdx) ==
             NI.getOutElemTy(GatherRangesNode::LengthsIdx)) &&
            ((NI.getOutElemTy(GatherRangesNode::LengthsIdx) ==
              ElemKind::Int32ITy) ||
             (NI.getOutElemTy(GatherRangesNode::LengthsIdx) == IndexElemKind)));

  case Kinded::Kind::ScatterDataNodeKind:
    // ScatterData ==> Copy + ScatterData. Copy supports everything
    // ReshapeNode above supports, however ScatterData only supports the
    // following.
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy},
               {ScatterDataNode::IndicesIdx}) &&
           (NI.getInElemTy(ScatterDataNode::IndicesIdx) == IndexElemKind);

  case Kinded::Kind::SelectNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {SelectNode::CondIdx}) &&
           (NI.getInElemTy(SelectNode::CondIdx) == ElemKind::BoolTy);

  case Kinded::Kind::CmpLTENodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
               {CmpLTENode::ResultIdx}) &&
           (NI.getOutElemTy(CmpLTENode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::CmpLTNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int32ITy}, {},
               {CmpLTNode::ResultIdx}) &&
           (NI.getOutElemTy(CmpLTNode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::IsNaNNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy}, {},
                                                  {CmpLTENode::ResultIdx}) &&
           (NI.getOutElemTy(CmpLTENode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::CmpEQNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int64ITy}, {},
                                                  {CmpEQNode::ResultIdx}) &&
           (NI.getOutElemTy(CmpEQNode::ResultIdx) == ElemKind::BoolTy);

  case Kinded::Kind::TopKNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
               {TopKNode::IndicesIdx}) &&
           (NI.getOutElemTy(TopKNode::IndicesIdx) == IndexElemKind);

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
           (NI.getInElemTy(SoftMaxNode::SelectedIdx) == IndexElemKind);

  case Kinded::Kind::CrossEntropyLossNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {CrossEntropyLossNode::LabelsIdx}) &&
           (NI.getInElemTy(CrossEntropyLossNode::LabelsIdx) == IndexElemKind);

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
            IndexElemKind) &&
           (NI.getInElemTy(EmbeddingBagByteRowwiseOffsetsNode::OffsetsIdx) ==
            IndexElemKind) &&
           (NI.getOutElemTy(EmbeddingBagByteRowwiseOffsetsNode::ResultIdx) ==
            ElemKind::FloatTy);

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    return (NI.getInElemTy(
                FusedRowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx) ==
            ElemKind::UInt8FusedQTy) &&
           (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               WeightsIdx) == ElemKind::FloatTy) &&
           (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               IndicesIdx) == IndexElemKind) &&
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
           (NI.getInElemTy(SparseToDenseNode::IndicesIdx) == IndexElemKind);

  case Kinded::Kind::SoftMaxGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {SoftMaxGradNode::SelectedIdx},
               {SoftMaxGradNode::GradOfInputNamedSelectedIdx}) &&
           (NI.getInElemTy(SoftMaxGradNode::SelectedIdx) == IndexElemKind);

  case Kinded::Kind::ConvolutionGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy}, {},
        {ConvolutionGradNode::GradOfInputNamedInputIdx});

  case Kinded::Kind::CrossEntropyLossGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {CrossEntropyLossGradNode::LabelsIdx},
               {CrossEntropyLossGradNode::GradOfInputNamedLabelsIdx}) &&
           (NI.getInElemTy(CrossEntropyLossGradNode::LabelsIdx) ==
            IndexElemKind) &&
           (NI.getOutElemTy(
                CrossEntropyLossGradNode::GradOfInputNamedLabelsIdx) ==
            IndexElemKind);

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
                ElemKind::Int64ITy) &&
           (NI.getOutElemTy(
                NonMaxSuppressionNode::NumberOfSelectedIndicesIdx) ==
            NI.getOutElemTy(NonMaxSuppressionNode::IndicesIdx));

  case Kinded::Kind::ConvertToNodeKind:
    return ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::Int32ITy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) == ElemKind::FloatTy)) ||
           ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::Int64ITy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) ==
             ElemKind::Int32ITy)) ||
           ((NI.getInElemTy(ConvertToNode::InputIdx) == ElemKind::Int32ITy) &&
            (NI.getOutElemTy(ConvertToNode::ResultIdx) == ElemKind::Int64ITy));

  default:
    return false;
  }
}

bool CPUBackend::shouldLower(const Node *N) const {
  switch (N->getKind()) {
  case Kinded::Kind::ConvolutionNodeKind:
  case Kinded::Kind::SparseLengthsSumNodeKind:
    return false;
  default:
    return true;
  }
}

unsigned CPUBackend::numDevices() {
  return std::thread::hardware_concurrency();
}

std::unique_ptr<CompiledFunction> CPUBackend::createCompiledFunction(
    std::unique_ptr<llvm::orc::GlowJIT> JIT,
    runtime::RuntimeBundle &&runtimeBundle) const {
  return glow::make_unique<CPUFunction>(std::move(JIT),
                                        std::move(runtimeBundle));
}

std::unique_ptr<LLVMIRGen>
CPUBackend::createIRGen(const IRFunction *IR,
                        AllocationsInfo &allocationsInfo) const {
  CPULLVMIRGen *irgen =
      new CPULLVMIRGen(IR, allocationsInfo, "", getLibjitBitcode());
  return std::unique_ptr<CPULLVMIRGen>(irgen);
}

llvm::StringRef CPUBackend::getLibjitBitcode() const {
  return llvm::StringRef(reinterpret_cast<const char *>(libjit_bc),
                         libjit_bc_size);
}
