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

#include "glow/LLVMIRCodeGen/AllocationsInfo.h"
#include "glow/LLVMIRCodeGen/CommandLine.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <fstream>

using namespace glow;
using llvm::cast;
using llvm::DISubprogram;
using llvm::dyn_cast;
using llvm::isa;

extern llvm::cl::opt<bool> emitDebugInfo;

void LLVMIRGen::setCurrentDebugLocation(llvm::IRBuilder<> &builder,
                                        const glow::Instruction *I) {
  if (!emitDebugInfo)
    return;
  auto instrNum = instrNumbering_->getInstrNumber(I);
  // Get current function.
  llvm::Function *F = builder.GetInsertPoint()->getParent()->getParent();
  // Get a debug scope for the current function.
  auto *DIFunction = getOrCreateFunctionDebugInfo(F, dbgInfo_.mainFile_,
                                                  dbgInfo_.mainFile_, 0);
  auto DILoc = llvm::DILocation::get(
      getLLVMContext(), dbgInfo_.mainFileFirstInstrLineNo_ + instrNum, 0,
      DIFunction);
  llvm::DebugLoc loc(DILoc);
  builder.SetCurrentDebugLocation(loc);
}

llvm::DIType *LLVMIRGen::getDebugType(llvm::IRBuilder<> &builder,
                                      llvm::Type *ty) {
  // Check if the debug info for the type is in the cache and use it, if it is
  // available.
  if (dbgInfo_.DITypes_.count(ty))
    return dbgInfo_.DITypes_[ty];
  llvm::DIType *DITy{nullptr};
  if (ty == builder.getVoidTy()) {
    DITy = nullptr;
  } else if (ty == builder.getFloatTy()) {
    DITy = DIBuilder_->createBasicType("float", sizeof(float) * 8,
                                       llvm::dwarf::DW_ATE_float);
  } else if (ty == builder.getIntNTy(sizeof(size_t) * 8)) {
    DITy = DIBuilder_->createBasicType("size_t", sizeof(size_t) * 8,
                                       llvm::dwarf::DW_ATE_unsigned);
  } else if (auto *intTy = dyn_cast<llvm::IntegerType>(ty)) {
    std::string tyName = "int" + std::to_string(intTy->getBitWidth());
    DITy = DIBuilder_->createBasicType(tyName, intTy->getBitWidth(),
                                       llvm::dwarf::DW_ATE_unsigned);
  } else if (ty->isPointerTy()) {
    std::string tyName = "ptr" + std::to_string(dbgInfo_.DITypes_.size());
    DITy = DIBuilder_->createPointerType(
        getDebugType(builder, ty->getPointerElementType()), sizeof(void *) * 8);
  } else {
    llvm_unreachable("Cannot create DWARF debug type for an LLVM type");
  }
  dbgInfo_.DITypes_[ty] = DITy;
  return DITy;
}

static llvm::DebugLoc getDebugLoc(unsigned Line, unsigned Col,
                                  const llvm::MDNode *Scope) {

#if LLVM_VERSION_MAJOR < 12
  return llvm::DebugLoc::get(Line, Col, Scope);
#else
  // If no scope is available, this is an unknown location.
  if (!Scope)
    return llvm::DebugLoc();

  return llvm::DILocation::get(Scope->getContext(), Line, Col,
                               const_cast<llvm::MDNode *>(Scope), nullptr,
                               false);
#endif
}

void LLVMIRGen::generateFunctionDebugInfo(llvm::Function *F) {
  if (!emitDebugInfo)
    return;
  // First, generate a DISubprogram for the function.
  auto *DIFunction = getOrCreateFunctionDebugInfo(F, dbgInfo_.mainFile_,
                                                  dbgInfo_.mainFile_, 0);
  size_t lineNo = 0;
  auto file = dbgInfo_.mainFile_;
  auto *currentScope = DIFunction;
  lineNo = dbgInfo_.mainFileFirstInstrLineNo_;
  // Find the insertion poisition for debug instructions.
  llvm::IRBuilder<> builder(&F->getEntryBlock());
  if (!F->getEntryBlock().empty()) {
    llvm::DebugLoc debugLoc;
    // Find first instruction with non-empty debug loc.
    for (const auto &BB : *F) {
      for (const auto &I : BB) {
        if (I.getDebugLoc()) {
          debugLoc = I.getDebugLoc();
          break;
        }
      }
    }
    // Insert before the first instruction in the entry block.
    builder.SetInsertPoint(&F->getEntryBlock().front());
    if (!debugLoc) {
      debugLoc = getDebugLoc(lineNo, 0, currentScope);
    }
    builder.SetCurrentDebugLocation(debugLoc);
  }
  // Create debug information for the arguments, so that a debugger can expect
  // their values.
  for (unsigned i = 0, e = F->arg_size(); i != e; ++i) {
    // Create an alloca for storing a shadow of the function argument. The
    // parameter value will be copied there to make it easier for debugger to
    // inspect it.
    auto *paramAlloca =
        builder.CreateAlloca(F->getFunctionType()->getParamType(i));
    // Create a debug descriptor for the function argument.
    // TODO: Try to produce semantically meaningful parameter names, e.g. by
    // analyzing the debug information of the libjit.
    std::string paramName = "arg" + std::to_string(i + 1);
    auto param = DIBuilder_->createParameterVariable(
        currentScope, paramName, i + 1, file, lineNo,
        getDebugType(builder, F->getFunctionType()->getParamType(i)),
        /* alwaysPreserve */ true);
    // Store the initial value into the alloca, so that the debugger can show
    // it.
    auto *store = builder.CreateStore(F->arg_begin() + i, paramAlloca);
    DIBuilder_->insertDeclare(paramAlloca, param,
                              DIBuilder_->createExpression(),
                              getDebugLoc(lineNo, 0, currentScope), store);
  }
  DIBuilder_->finalizeSubprogram(F->getSubprogram());
  llvm::DIScope *scope = F->getSubprogram();
  if (!scope) {
    return;
  }
  // Add debug locations to all instructions inside the functions which have
  // debug information. This is required for the proper emission of the debug
  // information into object files. If debug locations are missing, LLVM would
  // not emit such information like e.g. types of function parameters, etc.
  llvm::DebugLoc debugLoc;
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (I.getDebugLoc()) {
        debugLoc = I.getDebugLoc();
        continue;
      }
      // Use last seen debug location.
      I.setDebugLoc(debugLoc);
    }
  }
}

llvm::DISubprogram *
LLVMIRGen::getOrCreateFunctionDebugInfo(llvm::Function *F, llvm::DIScope *scope,
                                        llvm::DIFile *file, unsigned lineNo) {
  // Do not emit any function debug information for LLVM internal functions.
  if (F->getName().empty() || F->getName().startswith("llvm."))
    return nullptr;
  auto *DIFunction = F->getSubprogram();
  if (!DIFunction) {
    // Create a function type. The result type should be stored in the first
    // element.
    llvm::SmallVector<llvm::Metadata *, 8> paramTys;

    // Add the result type.
    llvm::DIType *returnTy = getDebugType(*builder_, F->getReturnType());
    paramTys.push_back(returnTy);

    // Add the argument types.
    for (unsigned i = 0, e = F->arg_size(); i != e; ++i) {
      paramTys.push_back(
          getDebugType(*builder_, F->getFunctionType()->getParamType(i)));
    }
    // Create a function type.
    auto *DIFunctionTy = DIBuilder_->createSubroutineType(
        DIBuilder_->getOrCreateTypeArray(paramTys));
    // Create a debug information for the current function.
#if LLVM_VERSION_MAJOR == 7 || (LLVM_VERSION_MAJOR <= 8 && FACEBOOK_INTERNAL)
    DIFunction = DIBuilder_->createFunction(
        scope, F->getName(), "", file, lineNo, DIFunctionTy,
        false /* internal linkage */, true /* definition */, lineNo,
        llvm::DINode::FlagPrototyped, true /* isOptimized */);
#else
    DIFunction = DIBuilder_->createFunction(
        scope, F->getName(), "", file, lineNo, DIFunctionTy, lineNo,
        llvm::DINode::FlagPrototyped,
        DISubprogram::SPFlagLocalToUnit | DISubprogram::SPFlagDefinition |
            DISubprogram::SPFlagOptimized);
#endif

    assert(DIFunction);
    F->setSubprogram(DIFunction);
  }

  assert(F->getSubprogram() == DIFunction &&
         "Function has been assigned wrong debug information");
  return DIFunction;
}

/// Create and initialize global variables holding the bases addresses of
/// different memory areas.
static void initBaseAddressesOfMemoryAreas(DebugInfo &dbgInfo,
                                           llvm::IRBuilder<> &builder,
                                           llvm::Module &M,
                                           llvm::Function *mainF) {
  if (!mainF) {
    return;
  }
  auto *main = mainF;
  // Initialize the names of base address variables.
  // Only 3 memory areas are currently supported: constant weights, mutable
  // weights and activations. If more memory areas are introduced, the assert
  // and the initialization code below need to be adjusted.
  constexpr unsigned expectedNumMemoryAreas = 3;
  assert(MemoryAreaKind::LastMemoryArea == expectedNumMemoryAreas &&
         "Expected only 3 memory areas");
  dbgInfo.baseAddressesVariablesNames_.resize(expectedNumMemoryAreas);
  dbgInfo.baseAddressesVariablesNames_[MemoryAreaKind::ConstWeightsMemoryArea] =
      "constWeightsBaseAddress";
  dbgInfo
      .baseAddressesVariablesNames_[MemoryAreaKind::MutableWeightsMemoryArea] =
      "mutableWeightsBaseAddress";
  dbgInfo.baseAddressesVariablesNames_[MemoryAreaKind::ActivationsMemoryArea] =
      "activationsBaseAddress";

  dbgInfo.baseAddressesVariables_.resize(expectedNumMemoryAreas);
  // Create global variables to hold base addresses of different memory areas.
  for (unsigned idx = 0, e = dbgInfo.baseAddressesVariablesNames_.size();
       idx != e; ++idx) {
    auto name = dbgInfo.baseAddressesVariablesNames_[idx];
    // Create a global variable to hold a base address. Use CommonLinkage to
    // make sure it is not removed by optimizations.
    auto baseAddressVar = new llvm::GlobalVariable(
        M, builder.getInt8PtrTy(), /* isConst */ false,
        llvm::GlobalValue::ExternalLinkage, nullptr, name);
    baseAddressVar->setInitializer(
        llvm::ConstantPointerNull::get(builder.getInt8PtrTy()));
    // Initialize the variable by the corresponding base address passed to
    // "main" as a parameter.
    builder.CreateStore(main->args().begin() + idx, baseAddressVar);
    dbgInfo.baseAddressesVariables_[idx] = baseAddressVar;
  }
}

void LLVMIRGen::initDebugInfo() {
  if (!emitDebugInfo) {
    // No debug information is going to be emitted for the code generated from
    // the Glow IR. But any debug information from the Glow's library (e.g.
    // libjit) should be stripped as well. Let's strip the debug info as early
    // as possible, so that the LLVM's optimizer does not need to spend any time
    // to preserve the debug info during optimizations.
    // The debug info stripping is enabled only for LLVM >= 6.0.0, because
    // earlier versions of LLVM had a bug in this function which resulted in
    // compiler crashes.
    llvm::StripDebugInfo(getModule());
    return;
  }
  // Remove any existing debug info version flags from the module to
  // avoid possible conflicts, which may happen if libjit was compiled
  // using an older version of Clang which uses the old debug info format.
  llvm::NamedMDNode *NMD = llmodule_->getModuleFlagsMetadata();
  if (NMD) {
    NMD->eraseFromParent();
  }
  // Add the current debug info version into the module.
  llmodule_->addModuleFlag(llvm::Module::Override, "Debug Info Version",
                           llvm::DEBUG_METADATA_VERSION);
  llmodule_->addModuleFlag(llvm::Module::Override, "Dwarf Version", 4);
  llmodule_->addModuleFlag(llvm::Module::Override, "PIC Level", 2);

  // Construct the DIBuilder.
  DIBuilder_ = glow::make_unique<llvm::DIBuilder>(getModule());

  // Remove the old content of the Glow IR file.
  // The name of the file for the IR, without a path.
  auto irfileName = getBundleName().str() + ".glow";
  // Use the absolute path, so that a debugger can always find a file.
  llvm::SmallVector<char, 128> path(getOutputDir().begin(),
                                    getOutputDir().end());
  std::error_code EC = llvm::sys::fs::make_absolute(path);
  assert(!EC && "Could not create absolute path for a file");
  auto irfileFullPath = (path + "/" + irfileName).str();
  EC = llvm::sys::fs::remove(irfileFullPath);
  assert(!EC && "Could not remove the Glow IR file");

  // Create the debug information for the current file. It does not create a
  // real file. It is just a file name and path used for the debug locations.
  dbgInfo_.mainFile_ = DIBuilder_->createFile(
      irfileName, llvm::StringRef(path.data(), path.size()));

  // Create the compile unit for the module.
  dbgInfo_.compilationUnit_ = DIBuilder_->createCompileUnit(
      llvm::dwarf::DW_LANG_C, dbgInfo_.mainFile_, "Glow Compiler", 0, "", 0, "",
      llvm::DICompileUnit::DebugEmissionKind::FullDebug,
      /* SplitDebugInlining */ true,
      /* DebugInfoForProfiling */ true);
}

void LLVMIRGen::generateFunctionDebugInfo() {
  if (!emitDebugInfo) {
    return;
  }
  // Init global variables holding base address of different memory areas.
  initBaseAddressesOfMemoryAreas(dbgInfo_, *builder_, getModule(),
                                 getLLVMFunction());

  // Create a textual representation of the IR for the main function.
  // First store the textual IR into a string.
  std::string irContent;
  llvm::raw_string_ostream irfileContent(irContent);
  F_->dump(irfileContent);
  irfileContent.str();

  // Write the IR into a file.
  std::error_code EC;
  // The name of the file for the IR, without a path.
  auto irfileName = getBundleName().str() + ".glow";
  // Use the absolute path, so that a debugger can always find a file.
  llvm::SmallVector<char, 128> path(getOutputDir().begin(),
                                    getOutputDir().end());
  EC = llvm::sys::fs::make_absolute(path);
  assert(!EC && "Could not create absolute path for a file");
  auto irfileFullPath = (path + "/" + irfileName).str();
  llvm::raw_fd_ostream irfile(irfileFullPath, EC,
                              llvm::sys::fs::OpenFlags::F_Text |
                                  llvm::sys::fs::OpenFlags::F_Append);
  assert(!EC && "Error opening output file");
  irfile << irContent;
  irfile.close();

  // Find out the line number of the first IR instruction. It is required to
  // enable stepping in the debugger.
  std::ifstream in(irfileFullPath);
  std::string s;
  size_t lineNo = 0;
  // Find the last code section, because this is the section for the last bundle
  // entry.
  while (getline(in, s)) {
    lineNo++;
    // The first IR instruction comes right after the line "code {".
    if (s.substr(0, 6) == "code {") {
      dbgInfo_.mainFileFirstInstrLineNo_ = lineNo + 1;
    }
  }
  assert(dbgInfo_.mainFileFirstInstrLineNo_ &&
         "No IR code was found in the textual IR representation");

  // Create the debug info for the main function.
  auto *main = getLLVMFunction();
  dbgInfo_.mainF_ = main ? getOrCreateFunctionDebugInfo(
                               main, dbgInfo_.mainFile_, dbgInfo_.mainFile_,
                               dbgInfo_.mainFileFirstInstrLineNo_)
                         : nullptr;
}

void LLVMIRGen::emitDebugGlobalVariableForValue(const Value *val) {
  auto name = val->getName();
  // Create a proper type for the variable.
  // Represent Glow's N-dimensional tensors as N-dimensional C arrays in the
  // debug information. This allows for inspecting them in the debugger using a
  // natural array notation, i.e. tensor[idx1][idx2]...[idxN].
  auto *ty = val->getType();
  auto dims = ty->dims();
  auto dbgElemTy = getDebugType(*builder_, getElementType(*builder_, val));
  llvm::SmallVector<llvm::Metadata *, 8> subranges;
  for (auto dim : dims) {
    subranges.push_back(llvm::DISubrange::get(getLLVMContext(), dim));
  }
  auto subscripts = llvm::MDTuple::get(getLLVMContext(), subranges);
  auto dbgArrayTy = DIBuilder_->createArrayType(
      ty->getSizeInBytes() * 8, sizeof(float), dbgElemTy, subscripts);

  // Create a debug info for the logical global variable representing a weight
  // or an activation. This allows for inspecting the values of weights and
  // activations when using a debugger. The address of this logical global
  // variable is computed as (base address of the memory area + offset) using
  // the information from the AllocationsInfo.
  llvm::GlobalVariable *baseAddress{nullptr};

  MemoryAreaKind memoryAreaKind = MemoryAreaKind::LastMemoryArea;

  switch (allocationsInfo_.valueNumbers_[val].first) {
  case AllocationsInfo::ValueKind::Activation: {
    memoryAreaKind = MemoryAreaKind::ActivationsMemoryArea;
    break;
  }
  case AllocationsInfo::ValueKind::ConstantWeight: {
    memoryAreaKind = MemoryAreaKind::ConstWeightsMemoryArea;
    break;
  }
  case AllocationsInfo::ValueKind::MutableWeight: {
    memoryAreaKind = MemoryAreaKind::MutableWeightsMemoryArea;
    break;
  default:
    LOG(FATAL) << "Unknown memory area kind";
  }
  }

  baseAddress = dbgInfo_.baseAddressesVariables_[memoryAreaKind];

  // DWARF operations to be performed with the base address to compute the
  // address of the logical global variable.
  llvm::SmallVector<uint64_t, 4> ops;
  assert(allocationsInfo_.allocatedAddress_.count(val) &&
         "The weight should be in the map");
  auto offset = allocationsInfo_.allocatedAddress_[val];
  // Get the value of the global var.
  ops.push_back(llvm::dwarf::DW_OP_deref);
  // Add the offset to the value of the global var to get the address of the
  // logical debug variable being created.
  ops.push_back(llvm::dwarf::DW_OP_constu);
  ops.push_back(offset);
  ops.push_back(llvm::dwarf::DW_OP_plus);
  llvm::DIExpression *DIexpr{nullptr};
  DIexpr = DIBuilder_->createExpression(ops);
  auto *DIgv = DIBuilder_->createGlobalVariableExpression(
      dbgInfo_.compilationUnit_, name, "", dbgInfo_.mainFile_, 0, dbgArrayTy,
      /* isLocalToUnit */ false, DIexpr);
  baseAddress->addDebugInfo(DIgv);
}

void LLVMIRGen::generateModuleDebugInfo() {
  if (!emitDebugInfo)
    return;

  // Check that global variables representing base-addresses are not eliminated
  // e.g. by optimization passes. These variables are needed for emitting the
  // debug info for weights and activations, because it uses relative addressing
  // based on these variables.
  for (auto name : dbgInfo_.baseAddressesVariablesNames_) {
    (void)name;
    assert(getModule().getGlobalVariable(name,
                                         /* allowInternal */ true) &&
           "Base address variable should be present in the LLVM module");
  }

  // Now iterate over the module and add debug locations to all instructions
  // inside the functions which have debug information. This is required for the
  // proper emission of the debug information into object files. If debug
  // locations are missing, LLVM would not emit such information like e.g. types
  // of function parameters, etc.
  for (auto &F : getModule()) {
    if (F.isDeclaration())
      continue;
    // Bail if the function has no debug information.
    llvm::DIScope *scope = F.getSubprogram();
    if (!scope)
      continue;
    size_t lineNo = dbgInfo_.mainFileFirstInstrLineNo_;
    llvm::DebugLoc debugLoc(
        llvm::DILocation::get(getLLVMContext(), lineNo, 0, scope));
    for (auto &BB : F) {
      for (auto &I : BB) {
        // Do not update debug locations that are not belonging to the current
        // scope.
        if (I.getDebugLoc() &&
            I.getDebugLoc()->getScope()->getName() != F.getName())
          continue;
        // Do not update existing debug information in the current scope.
        if (I.getDebugLoc()) {
          // Use the last seen debug location in the current scope.
          debugLoc = I.getDebugLoc();
          continue;
        }
        I.setDebugLoc(debugLoc);
      }
    }
  }

  // Emit the debug info for weight variables and activations variables used by
  // the Glow IR. Represent those variables as global variables.
  for (auto &v : F_->findConstants()) {
    auto *w = cast<WeightVar>(F_->getWeightForNode(v));
    emitDebugGlobalVariableForValue(w);
  }

  for (const auto &I : F_->getInstrs()) {
    if (!isa<AllocActivationInst>(&I) && !isa<TensorViewInst>(&I))
      continue;
    emitDebugGlobalVariableForValue(&I);
  }

  // Finalize the debug info.
  DIBuilder_->finalize();

  // Fix function attributes related issues.
  for (auto &FF : getModule()) {
    // Optnone requires NoInline.
    if (FF.hasFnAttribute(llvm::Attribute::AttrKind::OptimizeNone)) {
      FF.addFnAttr(llvm::Attribute::AttrKind::NoInline);
    }
  }

  // Verify the module to see if there are any errors due to the debug
  // information.
  bool brokenDebugInfo = false;
  (void)brokenDebugInfo;
  // Pass brokenDebugInfo as a reference to the verifyModule.
  assert(!llvm::verifyModule(getModule(), &llvm::errs(), &brokenDebugInfo) &&
         "LLVM module verification error");
  assert(!brokenDebugInfo && "Debug information is broken");
}
