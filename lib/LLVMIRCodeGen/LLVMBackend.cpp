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
#include "BundleSaver.h"
#include "CommandLine.h"
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

/// Emit the entry point for JIT called "jitmain".
/// Function has the following API:
///   void jitmain(uint8_t *baseConstantWeightVars,
///                uint8_t *baseInOutWeightVars,
///                uint8_t *baseActivations);
void LLVMBackend::emitJitMain(LLVMIRGen &irgen) const {
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
  irgen.createCall(builder, entryF, initFunctionCallArgs);
  // Terminate the function.
  builder.CreateRetVoid();
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
  BundleSaver bundleSaver(*this, outputDir, bundleName);
  bundleSaver.save(mainEntryName, IR.get());
  bundleSaver.produceBundle();
}

void LLVMBackend::saveFunctions(llvm::ArrayRef<BundleEntry> entries,
                                llvm::StringRef outputDir,
                                llvm::StringRef bundleName) const {
  BundleSaver bundleSaver(*this, outputDir, bundleName);
  std::vector<std::unique_ptr<glow::IRFunction>> irFunctions;
  for (auto &entry : entries) {
    auto IR = generateAndOptimizeIR(entry.func, *this, shouldShareBuffers());
    bundleSaver.save(entry.name, IR.get());
    irFunctions.emplace_back(std::move(IR));
  }
  bundleSaver.produceBundle();
}
