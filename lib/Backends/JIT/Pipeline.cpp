// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "JIT.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Vectorize.h"

using namespace glow;
using llvm::StringRef;
using llvm::dyn_cast;
using llvm::isa;

void JITBackend::optimizeLLVMModule(llvm::Function *F,
                                    llvm::TargetMachine &TM) {
  auto *M = F->getParent();

  // Make all of the functions except for 'main' internal and optimizable.
  auto preserveMain = [=](const llvm::GlobalValue &GV) {
    return GV.getName() == "main";
  };
  llvm::internalizeModule(*M, preserveMain);

  // Perform specialization of functions for constant arguments before anything
  // else.
  performSpecialization();

  llvm::PassManagerBuilder PMB;
  PMB.OptLevel = 2;
  PMB.SizeLevel = 0;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  PMB.Inliner = llvm::createFunctionInliningPass();

  M->setTargetTriple(TM.getTargetTriple().normalize());
  M->setDataLayout(TM.createDataLayout());

  // Replace the target-specific machine code attributes that were attached by
  // the frontend.
  llvm::AttributeList AL;
  for (auto &FF : *M) {
    // Check for no-inline attribute.
    bool dontInline = FF.hasFnAttribute(llvm::Attribute::AttrKind::NoInline);
    // Clear all attributes.
    FF.setAttributes(AL);
    // Force inline all non-no-inline functions.
    if (!dontInline) {
      FF.addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
    }
    if (dontInline) {
      FF.addFnAttr(llvm::Attribute::AttrKind::NoInline);
    }
  }

  llvm::legacy::FunctionPassManager FPM(F->getParent());
  llvm::legacy::PassManager PM;

  // Add internal analysis passes from the target machine.
  PM.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
  FPM.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));

  PMB.populateFunctionPassManager(FPM);
  PMB.populateModulePassManager(PM);
  FPM.doInitialization();
  PM.run(*F->getParent());
  for (auto &FF : *M) {
    FPM.run(FF);
  }
  FPM.doFinalization();
  PM.run(*F->getParent());
}
