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

#include "CPUBackend.h"
#include "BundleSaver.h"
#include "CPUFunction.h"

#include "glow/Backends/BackendUtils.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

using namespace glow;

static llvm::cl::opt<std::string> target("target", llvm::cl::desc("target"));

namespace glow {
Backend *createCPUBackend() { return new CPUBackend(); }
} // namespace glow

namespace {

//===----------------------------------------------------------------------===//
//                   Functions for executing code using JIT
//===----------------------------------------------------------------------===//

/// Emit the entry point for JIT called "jitmain".
/// Function has the following API:
///   void jitmain(uint8_t *baseConstantWeightVars,
///                uint8_t *baseInOutWeightVars,
///                nuint8_t *baseActivations);
static void emitJitMain(LLVMIRGen &irgen) {
  AllocationsInfo &allocationsInfo = irgen.getAllocationsInfo();
  llvm::Type *voidTy = llvm::Type::getVoidTy(irgen.getLLVMContext());
  auto int8PtrTy = llvm::Type::getInt8PtrTy(irgen.getLLVMContext());
  llvm::FunctionType *jitFuncTy =
      llvm::FunctionType::get(voidTy, {int8PtrTy, int8PtrTy, int8PtrTy}, false);
  auto *func =
      llvm::Function::Create(jitFuncTy, llvm::Function::ExternalLinkage,
                             "jitmain", &irgen.getModule());
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(irgen.getLLVMContext(), "entry", func);
  llvm::IRBuilder<> builder(entry_bb);

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
  createCall(builder, entryF, initFunctionCallArgs);
  // Terminate the function.
  builder.CreateRetVoid();
  // Create the debug info for the entry point function.
  irgen.generateFunctionDebugInfo(func);
}

/// Perform memory allocation for a JIT execution.
void allocateJITMemory(const IRFunction *F, AllocationsInfo &allocationsInfo) {
  allocationsInfo.numberValues(F);
  allocationsInfo.allocateActivations(F);
  allocationsInfo.allocateWeightVars(F);
  allocationsInfo.allocateTensorViews(F);
}

} // end namespace

std::unique_ptr<LLVMIRGen>
CPUBackend::createIRGen(IRFunction *IR,
                        AllocationsInfo &allocationsInfo) const {
  LLVMIRGen *irgen = new LLVMIRGen(IR, allocationsInfo, "");
  return std::unique_ptr<LLVMIRGen>(irgen);
}

std::unique_ptr<CompiledFunction>
CPUBackend::compileIR(std::unique_ptr<IRFunction> IR) const {
  auto function = compileIRWithoutConstants(IR.get());
  static_cast<CPUFunction *>(function.get())
      ->getRuntimeBundle()
      .collectConstants(IR.get());
  return function;
}

std::unique_ptr<CompiledFunction>
CPUBackend::compileIRWithoutConstants(IRFunction *IR) const {
  AllocationsInfo allocationsInfo;
  std::unique_ptr<LLVMIRGen> irgen = createIRGen(IR, allocationsInfo);
  irgen->initTargetMachine(target.empty() ? "" : target.getValue(),
                           llvm::CodeModel::Model::Large);
  irgen->initCodeGen();
  // Perform the address assignment for activations and WeightVars.

  allocateJITMemory(IR, irgen->getAllocationsInfo());
  // Create the jitmain function to be invoked by JIT.
  emitJitMain(*irgen);
  // Emit the code for the body of the entry function.
  irgen->performCodeGen();
  // Hand over the module to JIT for the machine code generation.
  auto JIT = llvm::make_unique<llvm::orc::GlowJIT>(irgen->getTargetMachine());
  JIT->addModule(irgen->borrowModule());
  // Build runtimeBundle object containing offsets and allocation sizes.
  MemoryAllocator constantAllocator("ConstantWeights", 0);
  MemoryAllocator placeholderAllocator("Placeholders", 0);
  MemoryAllocator activationsAllocator("Activations", 0);
  runtime::RuntimeBundle runtimeInfo = generateRuntimeBundle(
      *IR, constantAllocator, placeholderAllocator, activationsAllocator);
  return llvm::make_unique<CPUFunction>(std::move(JIT), runtimeInfo);
}

std::unique_ptr<CompiledFunction>
CPUBackend::compile(Function *F, const CompilationOptions &opts) const {
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
  return compiledFunc;
}

void CPUBackend::save(Function *F, llvm::StringRef outputDir,
                      llvm::StringRef networkName) const {
  std::string tgt = target.empty() ? "" : target.getValue();
  auto IR = generateAndOptimizeIR(F, *this, shouldShareBuffers());
  BundleSaver(IR.get()).save(tgt, outputDir, networkName);
}

bool CPUBackend::isOpSupported(const NodeInfo &NI) const {
  // Note: For brevity below, "X ==> Y, Z" signifes that Node X is IRGen'd into
  // Instructions Y and Z.

  switch (NI.getKind()) {
  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MinNodeKind:
    // Note: Int8QTy Log, Tanh, and Sigmoid are lowered into a lookup
    // table. However, we do not lower them until after they're quantized. So we
    // need to return here that they are supported as Int8QTy.
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::CPUMaxSplatNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::MaxPoolNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy});

  case Kinded::Kind::SaveNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
    // These are implemented via a Copy Instruction.
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int32QTy,
         ElemKind::Int32ITy, ElemKind::Int64ITy, ElemKind::BoolTy});

  case Kinded::Kind::DivNodeKind:
    // InsertTensor ==> Copy + InsertTensor. Copy supports everything
    // ReshapeNode above supports, so InsertTensor is the limiting factor.
  case Kinded::Kind::InsertTensorNodeKind:
    // Concat ==> Splat + Insert. Both only support the following.
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::TransposeNodeKind:
  case Kinded::Kind::SliceNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
        {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy});

  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy},
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
            ElemKind::Int32ITy) &&
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

  case Kinded::Kind::LengthsToRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});

  case Kinded::Kind::IntLookupTableNodeKind:
  case Kinded::Kind::RescaleQuantizedNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy});

  case Kinded::Kind::PowNodeKind:
  case Kinded::Kind::AvgPoolGradNodeKind:
  case Kinded::Kind::MaxPoolGradNodeKind:
  case Kinded::Kind::QuantizationProfileNodeKind:
  case Kinded::Kind::CPUConvDKKC8NodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
  case Kinded::Kind::LocalResponseNormalizationGradNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});

  case Kinded::Kind::ConvolutionNodeKind:
    if (!NI.getInTy(ConvolutionNode::InputIdx)->isQuantizedType()) {
      return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::FloatTy});
    }

    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int8QTy},
                                                  {ConvolutionNode::BiasIdx}) &&
           (NI.getInElemTy(ConvolutionNode::BiasIdx) == ElemKind::Int32QTy);

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
            (NI.getInElemTy(GatherNode::IndicesIdx) == ElemKind::Int64ITy));

  case Kinded::Kind::GatherRangesNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy, ElemKind::Int64ITy},
               {GatherRangesNode::RangesIdx}, {GatherRangesNode::LengthsIdx}) &&
           ((NI.getInElemTy(GatherRangesNode::RangesIdx) ==
             NI.getOutElemTy(GatherRangesNode::LengthsIdx)) &&
            ((NI.getOutElemTy(GatherRangesNode::LengthsIdx) ==
              ElemKind::Int32ITy) ||
             (NI.getOutElemTy(GatherRangesNode::LengthsIdx) ==
              ElemKind::Int64ITy)));

  case Kinded::Kind::ScatterAssignNodeKind:
    // ScatterAssign ==> Copy + ScatterAssign. Copy supports everything
    // ReshapeNode above supports, however ScatterAssign only supports the
    // following.
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy},
               {ScatterAssignNode::IndicesIdx}) &&
           (NI.getInElemTy(ScatterAssignNode::IndicesIdx) ==
            ElemKind::Int64ITy);

  case Kinded::Kind::SelectNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {SelectNode::CondIdx}) &&
           (NI.getInElemTy(SelectNode::CondIdx) == ElemKind::BoolTy);

  case Kinded::Kind::CmpLTENodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy, ElemKind::Int8QTy}, {},
               {CmpLTENode::ResultIdx}) &&
           (NI.getOutElemTy(CmpLTENode::ResultIdx) == ElemKind::BoolTy);

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
           (NI.getOutElemTy(TopKNode::IndicesIdx) == ElemKind::Int64ITy);

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
           (NI.getInElemTy(SoftMaxNode::SelectedIdx) == ElemKind::Int64ITy);

  case Kinded::Kind::CrossEntropyLossNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {CrossEntropyLossNode::LabelsIdx}) &&
           (NI.getInElemTy(CrossEntropyLossNode::LabelsIdx) ==
            ElemKind::Int64ITy);

  case Kinded::Kind::LengthsSumNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {LengthsSumNode::LengthsIdx}) &&
           (NI.getInElemTy(LengthsSumNode::LengthsIdx) == ElemKind::Int32ITy);

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind:
    return (NI.getInElemTy(
                FusedRowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx) ==
            ElemKind::Int8FusedQTy) &&
           (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               WeightsIdx) == ElemKind::FloatTy) &&
           (NI.getInElemTy(FusedRowwiseQuantizedSparseLengthsWeightedSumNode::
                               IndicesIdx) == ElemKind::Int64ITy) &&
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
            ElemKind::Int32QTy) &&
           (NI.getOutElemTy(RowwiseQuantizedFullyConnectedNode::ResultIdx) ==
            ElemKind::Int8QTy);

  case Kinded::Kind::SparseToDenseNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind(
               {ElemKind::FloatTy}, {SparseToDenseNode::IndicesIdx}) &&
           (NI.getInElemTy(SparseToDenseNode::IndicesIdx) ==
            ElemKind::Int64ITy);

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

  case Kinded::Kind::TraceEventNodeKind:
    return NI.getInElemTy(TraceEventNode::DataIdx) == ElemKind::Int64ITy;

  default:
    return false;
  }
}

bool CPUBackend::shouldLower(const Node *N) const {
  if (N->getKind() == Kinded::Kind::ConvolutionNodeKind)
    return false;
  return true;
}

llvm::CallInst *glow::createCall(llvm::IRBuilder<> &builder,
                                 llvm::Function *callee,
                                 llvm::ArrayRef<llvm::Value *> args) {
#ifndef NDEBUG
  llvm::FunctionType *FTy = callee->getFunctionType();
  assert((args.size() == FTy->getNumParams() ||
          (FTy->isVarArg() && args.size() > FTy->getNumParams())) &&
         "Calling a function with bad signature: wrong number of arguments.");

  for (unsigned i = 0; i != args.size(); ++i)
    assert((i >= FTy->getNumParams() ||
            FTy->getParamType(i) == args[i]->getType()) &&
           "Calling a function with a bad signature: argument type mismatch.");
#endif
  return builder.CreateCall(callee, args);
}
