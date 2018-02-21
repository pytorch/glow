// Copyright 2017 Facebook Inc.  All Rights Reserved.

#define DEBUG_TYPE "jit"
#include "JIT.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::StringRef;
using llvm::dyn_cast;
using llvm::isa;

JITBackend::JITBackend(IRFunction *F)
    : F_(F), irgen_(F_, allocationsInfo_, "") {}

JITBackend::~JITBackend() { clear(); }

void JITBackend::clear() { F_->clear(); }

//===----------------------------------------------------------------------===//
//                   Functions for executing code using JIT
//===----------------------------------------------------------------------===//

/// Emit the entry point for JIT called "jitmain". It simply calls the main
/// entry of the module with the constant concrete addresses of all the memory
/// areas. Since these addresses are constants, the LLVM optimizer will constant
/// propagate them into relative addressing computations and the like and
/// produce a very efficient code that uses absolute addressing whenever
/// possible.
void JITBackend::emitJitMain() {
  llvm::Type *voidTy = llvm::Type::getVoidTy(irgen_.getLLVMContext());
  llvm::FunctionType *jitFuncTy = llvm::FunctionType::get(voidTy, {}, false);
  auto *func =
      llvm::Function::Create(jitFuncTy, llvm::Function::ExternalLinkage,
                             "jitmain", &irgen_.getModule());
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(irgen_.getLLVMContext(), "entry", func);
  llvm::IRBuilder<> builder(entry_bb);

  // Prepare arguments for the "main" function.
  llvm::SmallVector<llvm::Value *, 4> initFunctionCallArgs;
  auto *sizeTType = builder.getIntNTy(sizeof(size_t) * 8);
  auto int8PtrTy = llvm::Type::getInt8PtrTy(irgen_.getLLVMContext());

  initFunctionCallArgs.push_back(builder.CreateIntToPtr(
      llvm::ConstantInt::get(
          sizeTType, reinterpret_cast<size_t>(
                         allocationsInfo_.baseConstantWeightVarsAddress_)),
      int8PtrTy));
  initFunctionCallArgs.push_back(builder.CreateIntToPtr(
      llvm::ConstantInt::get(
          sizeTType, reinterpret_cast<size_t>(
                         allocationsInfo_.baseMutableWeightVarsAddress_)),
      int8PtrTy));
  initFunctionCallArgs.push_back(builder.CreateIntToPtr(
      llvm::ConstantInt::get(
          sizeTType,
          reinterpret_cast<size_t>(allocationsInfo_.baseActivationsAddress_)),
      int8PtrTy));
  // Now form the offsets array and pass it as the last argument.
  auto offsetsArray =
      irgen_.emitConstOffsetsArray(irgen_.getBuilder(), allocationsInfo_);
  initFunctionCallArgs.push_back(offsetsArray);
  // Invoke the main entry with constant arguments and let LLVM optimizer make
  // use of it.
  auto *entryF = irgen_.getModule().getFunction(irgen_.getMainEntryName());
  entryF->setLinkage(llvm::Function::InternalLinkage);
  builder.CreateCall(entryF, initFunctionCallArgs);
  // Terminate the function.
  builder.CreateRetVoid();
}

void JITBackend::performJITMemoryAllocation() {
  allocationsInfo_.clear();
  allocationsInfo_.numberValues(F_);
  allocationsInfo_.allocateActivations(F_);
  // Tell the allocateWeightVars to reuse existing addresses for weights.
  allocationsInfo_.allocateWeightVars(F_, true);

  // Allocate the heap to match the max memory usage for activations.
  if (allocationsInfo_.activationsMemSize_ > 0) {
    heap_.resize(allocationsInfo_.activationsMemSize_);
    allocationsInfo_.baseActivationsAddress_ = &heap_[0];
  }
}

void JITBackend::init() {
  irgen_.initTargetMachine(llvm::CodeModel::Model::Large);
  JIT_ = llvm::make_unique<llvm::orc::GlowJIT>(irgen_.getTargetMachine());
  irgen_.initCodeGen();
  // Perform the address assignment for activations and WeightVars.
  performJITMemoryAllocation();
  // Create the jitmain function to be invoked by JIT.
  emitJitMain();
  // Emit the code for the body of the entry function.
  irgen_.performCodeGen();
  // Hand over the module to JIT for the machine code generation.
  JIT_->addModule(irgen_.borrowModule());
}

void JITBackend::doForwardPass(bool isTrain) {
  auto sym = JIT_->findSymbol("jitmain");
  assert(sym && "Unable to JIT the code!");
  using JitFuncType = void (*)(void);
  auto address = sym.getAddress();
  if (address) {
    JitFuncType funcPtr = reinterpret_cast<JitFuncType>(address.get());
    funcPtr();
  } else {
    GLOW_ASSERT(false && "Error getting address.");
  }
}
