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

bool CPUBackend::isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const {
  // Check for quantization support.
  if (elementTy == ElemKind::Int8QTy) {
    switch (opKind) {
    case Kinded::Kind::AddNodeKind:
    case Kinded::Kind::BatchedAddNodeKind:
    case Kinded::Kind::BatchedReduceAddNodeKind:
    case Kinded::Kind::CmpLTENodeKind:
    case Kinded::Kind::ConcatNodeKind:
    case Kinded::Kind::ConvolutionNodeKind:
    case Kinded::Kind::DequantizeNodeKind:
    case Kinded::Kind::DivNodeKind:
    case Kinded::Kind::FullyConnectedNodeKind:
    case Kinded::Kind::GatherNodeKind:
    case Kinded::Kind::LogNodeKind:
    case Kinded::Kind::MatMulNodeKind:
    case Kinded::Kind::MaxNodeKind:
    case Kinded::Kind::MinNodeKind:
    case Kinded::Kind::MulNodeKind:
    case Kinded::Kind::AvgPoolNodeKind:
    case Kinded::Kind::MaxPoolNodeKind:
    case Kinded::Kind::QuantizeNodeKind:
    case Kinded::Kind::ReluNodeKind:
    case Kinded::Kind::RescaleQuantizedNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::SelectNodeKind:
    case Kinded::Kind::SliceNodeKind:
    case Kinded::Kind::SigmoidNodeKind:
    case Kinded::Kind::SubNodeKind:
    case Kinded::Kind::TanhNodeKind:
    case Kinded::Kind::TileNodeKind:
    case Kinded::Kind::TopKNodeKind:
    case Kinded::Kind::TransposeNodeKind:
      return true;
    default:
      return false;
    }
  }

  if (elementTy == ElemKind::Float16Ty) {
    return false;
  }

  if (elementTy == ElemKind::Int16QTy) {
    return false;
  }

  return true;
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
