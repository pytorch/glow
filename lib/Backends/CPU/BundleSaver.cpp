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
#define DEBUG_TYPE "jit"

#include "BundleSaver.h"

#include "CPUBackend.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

BundleSaver::BundleSaver(const IRFunction *F)
    : F_(F), irgen_(F_, allocationsInfo_, "") {}

void BundleSaver::saveWeights(llvm::StringRef weightsFileName) {
  std::error_code EC;
  llvm::raw_fd_ostream weightsFile(weightsFileName, EC, llvm::sys::fs::F_None);
  GLOW_ASSERT(!EC &&
              "Could not open the output file for saving the bundle weights");
  // Serialize only constant weights.
  // Do not serialize mutable weights representing inputs and outputs, because
  // it should be configurable and set by the client.
  size_t pos = 0;
  size_t maxPos = 0;
  for (auto &v : F_->getGraph()->getParent()->getVars()) {
    auto *w = cast<WeightVar>(F_->getWeightForNode(v));
    if (v->getVisibilityKind() == VisibilityKind::Public)
      continue;
    auto numBytes = w->getSizeInBytes();
    auto payload = v->getPayload().getUnsafePtr();
    auto addr = allocationsInfo_.allocatedAddressed_[w];
    if (addr < pos) {
      // The payload was written already. It aliases something we have seen
      // already.
      continue;
    }
    weightsFile.seek(addr);
    weightsFile.write(payload, numBytes);
    pos = addr + numBytes;
    maxPos = std::max(pos, maxPos);
  }
  // Make sure that the file is as long as the constantWeightVarsMemSize_.
  // This is needed to properly handle alignments.
  weightsFile.seek(maxPos);
  for (size_t endPos = irgen_.getAllocationsInfo().constantWeightVarsMemSize_;
       maxPos < endPos; maxPos++) {
    weightsFile.write(0);
  }
  weightsFile.close();
}

void BundleSaver::emitSymbolTable() {
  // Define a struct for symbol table entries:
  // struct SymbolTableEntry {
  //  const char *name;
  //  size_t offset;
  //  size_t size;
  //  char kind;
  // };
  auto *charTy = llvm::Type::getInt8Ty(irgen_.getLLVMContext());
  auto *sizeTTy =
      llvm::Type::getIntNTy(irgen_.getLLVMContext(), sizeof(size_t) * 8);
  auto symbolTableEntryTy =
      llvm::StructType::get(irgen_.getLLVMContext(),
                            {charTy->getPointerTo(), sizeTTy, sizeTTy, charTy});
  // Set of entries in the symbol table.
  llvm::SmallVector<llvm::Constant *, 128> entries;
  // Iterate over all weights and record information about their names, offset,
  // size and kind.
  for (auto &v : F_->getGraph()->getParent()->getVars()) {
    auto *w = cast<WeightVar>(F_->getWeightForNode(v));
    bool isConstWeight = v->getVisibilityKind() != VisibilityKind::Public;
    auto size = w->getType()->size();
    auto addr = allocationsInfo_.allocatedAddressed_[w];
    // Create an SymbolTableEntry.
    auto *entry = llvm::ConstantStruct::get(
        symbolTableEntryTy,
        {// name.
         dyn_cast<llvm::Constant>(irgen_.getBuilder().CreateBitCast(
             irgen_.emitStringConst(irgen_.getBuilder(), w->getName()),
             charTy->getPointerTo())),
         // offset.
         llvm::ConstantInt::get(sizeTTy, addr),
         // size.
         llvm::ConstantInt::get(sizeTTy, size),
         // kind.
         llvm::ConstantInt::get(charTy, isConstWeight ? 0 : 1)});
    entries.push_back(entry);
  }

  // Create a constant array with these entries.
  auto *arr = llvm::ConstantArray::get(
      llvm::ArrayType::get(symbolTableEntryTy, entries.size()), entries);
  // Create a global variable and initialize it with the constructed array.
  new llvm::GlobalVariable(irgen_.getModule(), arr->getType(), true,
                           llvm::GlobalValue::InternalLinkage, arr,
                           irgen_.getMainEntryName() + "SymbolTable");
}

void BundleSaver::produceBundle(llvm::StringRef outputDir) {
  // Emit the symbol table for weight variables.
  emitSymbolTable();
  // Emit the config for the bundle.
  emitBundleConfig();

  auto &M = irgen_.getModule();
  auto bundleName = irgen_.getMainEntryName();
  auto bundleCodeOutput = (outputDir + "/" + bundleName + ".o").str();
  auto bundleWeightsOutput = (outputDir + "/" + bundleName + ".weights").str();
  DEBUG_GLOW(llvm::outs() << "Producing a bundle:\n"
                          << "bundle name: " << bundleName << "\n"
                          << "bundle code: " << bundleCodeOutput << "\n"
                          << "bundle weights:" << bundleWeightsOutput << "\n");
  llvm::StringRef fileName = bundleCodeOutput;
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(fileName, EC, llvm::sys::fs::F_None);
  GLOW_ASSERT(!EC &&
              "Could not open the output file for saving the bundle code");
  if (fileName.endswith(".bc")) {
    // Emit the bitcode file.
#if LLVM_VERSION_MAJOR > 6
    llvm::WriteBitcodeToFile(M, outputFile);
#else
    llvm::WriteBitcodeToFile(&M, outputFile);
#endif
  } else if (fileName.endswith(".o")) {
    // Emit the object file.
    llvm::legacy::PassManager PM;
    auto &TM = irgen_.getTargetMachine();
    TM.addPassesToEmitFile(
        PM, outputFile, llvm::TargetMachine::CodeGenFileType::CGFT_ObjectFile);
    PM.run(M);
  }
  outputFile.close();
  // Output weights.
  saveWeights(bundleWeightsOutput);
}

/// Emit the entry function for the bundle. It simply calls the main entry of
/// the module and forwards its arguments to it. As the last argument it
/// provides the constant array of offsets. Since these offsets are constants,
/// the LLVM optimizer will constant propagate them into relative addressing
/// computations and the like and produce a very efficient code that uses
/// absolute addressing whenever possible.
void BundleSaver::emitBundleEntryFunction() {
  // The bundle entry point has the following API:
  // void entry(uint8_t *baseConstantWeightVars, uint8_t *baseInoutWeightVars,
  // uint8_t *baseActivations);
  llvm::Type *voidTy = llvm::Type::getVoidTy(irgen_.getLLVMContext());
  auto int8PtrTy = llvm::Type::getInt8PtrTy(irgen_.getLLVMContext());
  llvm::FunctionType *bundleFuncTy =
      llvm::FunctionType::get(voidTy, {int8PtrTy, int8PtrTy, int8PtrTy}, false);
  auto *func =
      llvm::Function::Create(bundleFuncTy, llvm::Function::ExternalLinkage,
                             irgen_.getMainEntryName(), &irgen_.getModule());
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(irgen_.getLLVMContext(), "entry", func);
  llvm::IRBuilder<> builder(entry_bb);

  // Prepare arguments for the "main" function.
  llvm::SmallVector<llvm::Value *, 4> initFunctionCallArgs;
  initFunctionCallArgs.push_back(func->args().begin());
  initFunctionCallArgs.push_back(func->args().begin() + 1);
  initFunctionCallArgs.push_back(func->args().begin() + 2);
  // Now form the offsets array and pass it as the last argument.
  auto offsetsArray = irgen_.emitConstOffsetsArray(builder, allocationsInfo_);
  initFunctionCallArgs.push_back(offsetsArray);
  // Invoke the main entry with constant arguments and let LLVM optimizer make
  // use of it.
  auto *entryF = irgen_.getModule().getFunction("main");
  entryF->setLinkage(llvm::Function::InternalLinkage);
  createCall(builder, entryF, initFunctionCallArgs);
  // Terminate the function.
  builder.CreateRetVoid();
  // Create the debug info for the bundle entry point function.
  irgen_.generateFunctionDebugInfo(func);
}

// Create a config for this network. It will be exposed to the clients,
// so that they know how much memory they need to allocate, etc.
// Config consists of the following fields:
// struct BundleConfig {
//   size_t constantWeightVarsMemSize;
//   size_t mutableWeightVarsMemSize;
//   size_t activationsMemSize;
//   size_t alignment;
//   size_t numSymbols;
//   SymbolTableEntry *symbolTable;
// };
void BundleSaver::emitBundleConfig() {
  auto symbolTable = irgen_.getModule().getGlobalVariable(
      irgen_.getMainEntryName() + "SymbolTable", true);
  GLOW_ASSERT(symbolTable &&
              "Expected to find a symbol table for the AOT bundle");
  // Get the integer type having the same size in bits as size_t.
  auto *SizeTType = irgen_.getBuilder().getIntNTy(sizeof(size_t) * 8);
  auto symbolTableEntryTy = symbolTable->getType()->getPointerElementType();
  auto *bundleConfigTy = llvm::StructType::get(
      irgen_.getLLVMContext(), {SizeTType, SizeTType, SizeTType, SizeTType,
                                SizeTType, symbolTableEntryTy->getPointerTo()});
  auto config = new llvm::GlobalVariable(
      irgen_.getModule(), bundleConfigTy, /* isConst */ true,
      llvm::GlobalValue::LinkageTypes::ExternalLinkage, nullptr,
      irgen_.getMainEntryName() + "_config");
  config->setInitializer(llvm::ConstantStruct::get(
      bundleConfigTy,
      llvm::ConstantInt::get(
          SizeTType, irgen_.getAllocationsInfo().constantWeightVarsMemSize_),
      llvm::ConstantInt::get(
          SizeTType, irgen_.getAllocationsInfo().mutableWeightVarsMemSize_),
      llvm::ConstantInt::get(SizeTType,
                             irgen_.getAllocationsInfo().activationsMemSize_),
      llvm::ConstantInt::get(SizeTType, TensorAlignment),
      llvm::ConstantInt::get(SizeTType,
                             F_->getGraph()->getParent()->getVars().size()),
      symbolTable));
}

void BundleSaver::performBundleMemoryAllocation() {
  allocationsInfo_.numberValues(F_);
  allocationsInfo_.allocateActivations(F_);
  // Tell the allocateWeightVars to not reuse any existing addresses for weights
  // and to assign new ones.
  allocationsInfo_.allocateWeightVars(F_, false);
  allocationsInfo_.allocateTensorViews(F_);
}

void BundleSaver::save(llvm::StringRef target, llvm::StringRef outputDir) {
  // Object files generation works properly only in small mode.
  irgen_.initTargetMachine(target, llvm::CodeModel::Model::Small);
  irgen_.setMainEntryName(F_->getGraph()->getName());
  irgen_.setOutputDir(outputDir);
  irgen_.initCodeGen();
  // Perform the address assignment for activations and WeightVars.
  performBundleMemoryAllocation();
  // Create the bundle entry function.
  emitBundleEntryFunction();
  // Emit the code for the body of the entry function.
  irgen_.performCodeGen();
  // Produce the bundle.
  produceBundle(outputDir);
}
