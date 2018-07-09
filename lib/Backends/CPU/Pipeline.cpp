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
using llvm::dyn_cast;
using llvm::isa;

void LLVMIRGen::optimizeLLVMModule(llvm::Function *F, llvm::TargetMachine &TM) {
  auto *M = F->getParent();

  // Make all of the definitions from libjit and unnamed symbols internal and
  // optimizable. Everything else should be preserved as is.
  auto preserveSymbols = [=](const llvm::GlobalValue &GV) {
    auto name = GV.getName();
    // Do not internalize declarations.
    if (GV.isDeclaration())
      return true;
    // Do not preserve any internal symbols, which typically have no name or
    // start with jit_
    if (name.empty() || name.startswith("libjit_"))
      return false;
    return true;
  };

  // Internalize functions libjit. In this part of the code we change the
  // visibility of the symbols in the module and make 'main' the only visibile
  // function.
  llvm::internalizeModule(*M, preserveSymbols);

  // Next, we remove all of the 'no-inline' attributes that clang in -O0 adds to
  // all functions.
  for (auto &FF : *M) {
    FF.removeFnAttr(llvm::Attribute::AttrKind::NoInline);
  }

  // Perform specialization of functions for constant arguments before anything
  // else.
  performSpecialization();

  llvm::PassManagerBuilder PMB;
  PMB.OptLevel = 2;
  PMB.SizeLevel = 0;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = false;
  PMB.Inliner = llvm::createFunctionInliningPass();

  M->setTargetTriple(TM.getTargetTriple().normalize());
  M->setDataLayout(TM.createDataLayout());

  // Replace the target-specific machine code attributes that were attached by
  // the frontend.
  llvm::AttributeList AL;
  for (auto &FF : *M) {
    if (FF.isDeclaration()) {
      continue;
    }
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
    // Add no-frame-pointer-elim=true attribute. It helps with profiling and
    // debugging the produced code.
    FF.addFnAttr("no-frame-pointer-elim", "true");
  }

  // The "main" function is parameterized by the base addresses of memory areas
  // and it is always invoked from either the "jitmain" function or the AOT
  // entry point. To enable better LLVM optimizations "main" should always be
  // inlined.
  M->getFunction("main")->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);

  llvm::legacy::FunctionPassManager FPM(M);
  llvm::legacy::PassManager PM;

  // Add internal analysis passes from the target machine.
  PM.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
  FPM.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));

  PMB.populateFunctionPassManager(FPM);
  PMB.populateModulePassManager(PM);
  FPM.doInitialization();
  PM.run(*M);
  for (auto &FF : *M) {
    FPM.run(FF);
  }
  FPM.doFinalization();
  PM.run(*M);
}
