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
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"

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
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Vectorize.h"

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

bool LLVMIRGen::preserveSymbol(const llvm::GlobalValue &GV) {
  auto name = GV.getName();
  // Do not preserve any internal symbols, which typically have no name or
  // start with libjit_.
  if (name.empty() || name.startswith("libjit_"))
    return false;
  return true;
}

llvm::Attribute::AttrKind
LLVMIRGen::getInlinineAttr(const llvm::Function *F) const {
  return llvm::Attribute::AttrKind::None;
}

void LLVMIRGen::populatePassManagerBuilderOptions(
    llvm::PassManagerBuilder &PMB) {
  PMB.OptLevel = 2;
  PMB.SizeLevel = 0;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = false;
  PMB.Inliner = llvm::createFunctionInliningPass();
}

void LLVMIRGen::updateInlineAttributes(llvm::Module *M) {
  for (auto &FF : *M) {
    if (FF.isDeclaration()) {
      continue;
    }
    // Check for no-inline attribute.
    bool dontInline = FF.hasFnAttribute(llvm::Attribute::AttrKind::NoInline);
    bool alwaysInline =
        FF.hasFnAttribute(llvm::Attribute::AttrKind::AlwaysInline);
    bool optnone = FF.hasFnAttribute(llvm::Attribute::AttrKind::OptimizeNone);

    bool hasOmitFramePointer = FF.hasFnAttribute("omit-frame-pointer");
    llvm::Attribute omitFramePointerAttr;
    if (hasOmitFramePointer) {
      omitFramePointerAttr = FF.getFnAttribute("omit-frame-pointer");
    }

    bool hasFramePointer = FF.hasFnAttribute("frame-pointer");
    llvm::Attribute framePointerAttr;
    if (hasFramePointer) {
      framePointerAttr = FF.getFnAttribute("frame-pointer");
    }

    bool hasNoFramePointerElim = FF.hasFnAttribute("no-frame-pointer-elim");
    llvm::Attribute noFramePointerElimAttr;
    if (hasNoFramePointerElim) {
      noFramePointerElimAttr = FF.getFnAttribute("no-frame-pointer-elim");
    }

    auto inlineAttr = getInlinineAttr(&FF);
    if (inlineAttr != llvm::Attribute::AttrKind::None) {
      DCHECK(inlineAttr == llvm::Attribute::AttrKind::AlwaysInline ||
             inlineAttr == llvm::Attribute::AttrKind::NoInline)
          << "Unknown inlining attribute returned by getInlinineAttr";
      dontInline = (inlineAttr == llvm::Attribute::AttrKind::NoInline);
    }
    // Replace the target-specific machine code function attributes that were
    // attached by the frontend. Keep return and parameter attributes, e.g.,
    // noalias.
    FF.setAttributes(FF.getAttributes().removeAttributes(
        M->getContext(), llvm::AttributeList::FunctionIndex));
    if (hasOmitFramePointer) {
      FF.addFnAttr("omit-frame-pointer",
                   omitFramePointerAttr.getValueAsString());
    }
    if (hasFramePointer) {
      FF.addFnAttr("frame-pointer", framePointerAttr.getValueAsString());
    }
    if (hasNoFramePointerElim) {
      FF.addFnAttr("no-frame-pointer-elim",
                   noFramePointerElimAttr.getValueAsString());
    }
    // Force inline all non-no-inline functions.
    if (!dontInline || alwaysInline) {
      FF.addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
      continue;
    }
    if (dontInline || optnone) {
      FF.addFnAttr(llvm::Attribute::AttrKind::NoInline);
      continue;
    }
  }
}

void LLVMIRGen::optimizeLLVMModule(llvm::Module *M, llvm::TargetMachine &TM) {
  // Make all of the definitions from libjit and unnamed symbols internal and
  // optimizable. Everything else should be preserved as is.
  auto preserveSymbolCallback = [&](const llvm::GlobalValue &GV) -> bool {
    // Do not internalize declarations.
    if (GV.isDeclaration()) {
      return true;
    }
    return preserveSymbol(GV);
  };

  // Internalize functions in the module using a backend-specific logic.
  // Typically only the entry point would be preserved.
  llvm::internalizeModule(*M, preserveSymbolCallback);

  // Next, we remove all of the 'no-inline' attributes that clang in -O0 adds to
  // all functions.
  for (auto &FF : *M) {
    // For libjit functions that are marked as dllimport or dllexport
    // This sets them as regular functions.
    // This allows LLVM to eliminate unused functions,
    // and speeds up compilation.
    if (FF.getDLLStorageClass() !=
        llvm::GlobalValue::DLLStorageClassTypes::DefaultStorageClass) {
      FF.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
      FF.setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
    }

    // Remove NoInline attribute.
    FF.removeFnAttr(llvm::Attribute::AttrKind::NoInline);

    // LinkOnce linkage seems to cause problems to OrcJIT on some OS platforms.
    // In particular, ORCJit doesn't like linkonce_odr linkage which is used for
    // almost all templatized C++ functions in the LLVM module.
    if (!FF.isDeclaration() && FF.isLinkOnceLinkage(FF.getLinkage())) {
      FF.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
    }
  }

  // Perform specialization of functions for constant arguments before anything
  // else.
  performSpecialization();

  // Add instrumentation into the code for better debugging experience.
  performDebugInstrumentation();

  M->setTargetTriple(TM.getTargetTriple().normalize());
  M->setDataLayout(TM.createDataLayout());

  // Properly set inline attributes.
  updateInlineAttributes(M);

  // Add no-frame-pointer-elim=true attribute. It helps with profiling and
  // debugging the produced code.
  for (auto &FF : *M) {
    if (FF.isDeclaration()) {
      continue;
    }
    if (FF.hasFnAttribute("no-frame-pointer-elim") ||
        FF.hasFnAttribute("frame-pointer") ||
        FF.hasFnAttribute("omit-frame-pointer")) {
      continue;
    }
    FF.addFnAttr("no-frame-pointer-elim", "true");
  }

  // The "main" function is parameterized by the base addresses of memory areas
  // and it is always invoked from either the "jitmain" function or the AOT
  // entry point. To enable better LLVM optimizations "main" should always be
  // inlined.
  getLLVMFunction()->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(M->getTargetTriple()));
  // Disable optimizations of some builtin functions. They cause issues on some
  // targets.
  llvm::LibFunc libFunc;
  if (TLII.getLibFunc(llvm::StringRef("printf"), libFunc)) {
    TLII.setUnavailable(libFunc);
  }

  auto *TLIWP = new llvm::TargetLibraryInfoWrapperPass(TLII);

  llvm::legacy::FunctionPassManager FPM(M);
  llvm::legacy::PassManager PM;
  llvm::PassManagerBuilder PMB;
  populatePassManagerBuilderOptions(PMB);

  PM.add(TLIWP);

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
