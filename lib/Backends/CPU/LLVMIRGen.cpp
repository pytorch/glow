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

#include "LLVMIRGen.h"

#include "CPUBackend.h"
#include "CommandLine.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Quantization.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::StringRef;

llvm::cl::OptionCategory CPUBackendCat("Glow CPU Backend Options");

static llvm::cl::opt<bool>
    dumpIR("dump-llvm-ir",
           llvm::cl::desc("Dump the LLVM-IR of the jitted code"),
           llvm::cl::init(false), llvm::cl::cat(CPUBackendCat));

static llvm::cl::opt<bool>
    dumpJitAsm("dump-llvm-asm",
               llvm::cl::desc("Dump the textual assembly of the jitted code"),
               llvm::cl::init(false), llvm::cl::cat(CPUBackendCat));

llvm::cl::opt<bool>
    emitDebugInfo("g", llvm::cl::desc("Emit debug information for debuggers"),
                  llvm::cl::init(false), llvm::cl::cat(CPUBackendCat));

/// Generate the LLVM MAttr list of attributes.
static llvm::SmallVector<std::string, 0> getMachineAttributes() {
  llvm::SmallVector<std::string, 0> result;
  llvm::StringMap<bool> hostFeatures;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    for (auto &feature : hostFeatures) {
      if (feature.second) {
        llvm::StringRef fn = feature.first();
        // Skip avx512 because LLVM does not support it well.
        if (fn.startswith("avx512")) {
          continue;
        }
        result.push_back(fn);
      }
    }
  }
  return result;
}

/// Returns the CPU hostname.
static llvm::StringRef getHostCpuName() {
  auto cpu_name = llvm::sys::getHostCPUName();
  // Skip avx512 because LLVM does not support it well.
  cpu_name.consume_back("-avx512");
  return cpu_name;
}

LLVMIRGen::LLVMIRGen(IRFunction *F, AllocationsInfo &allocationsInfo,
                     std::string mainEntryName)
    : F_(F), allocationsInfo_(allocationsInfo), mainEntryName_(mainEntryName) {}

void LLVMIRGen::initTargetMachine(StringRef T,
                                  llvm::CodeModel::Model codeModel) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  if (T.empty())
    TM_.reset(llvm::EngineBuilder().setCodeModel(codeModel).selectTarget(
        llvm::Triple(), "", getHostCpuName(), getMachineAttributes()));
  else
    TM_.reset(llvm::EngineBuilder().setCodeModel(codeModel).selectTarget(
        llvm::Triple(T), "", "", llvm::SmallVector<std::string, 0>()));
}

std::string LLVMIRGen::getMainEntryName() const {
  StringRef name = mainEntryName_.empty() ? "main" : F_->getGraph()->getName();
  auto delimPos = name.rfind('/');
  if (delimPos != StringRef::npos)
    name = name.substr(delimPos + 1);
  return name;
}

void LLVMIRGen::setMainEntryName(std::string name) { mainEntryName_ = name; }

/// Load base addresses of different memory areas so that they can be easily
/// reused during codegen.
void LLVMIRGen::loadBaseAddresses(llvm::IRBuilder<> &builder) {
  auto *F = builder.GetInsertBlock()->getParent();

  // Load the base addresses at the beginning of the entry function once they
  // are set. They won't change after this point and all relative addressing
  // computations will simply use them.
  baseActivationsAddr_ = builder.CreatePtrToInt(F->args().begin() + 2,
                                                llvm::Type::getInt64Ty(ctx_));
  baseConstantWeightVarsAddr_ =
      builder.CreatePtrToInt(F->args().begin(), llvm::Type::getInt64Ty(ctx_));
  baseMutableWeightVarsAddr_ = builder.CreatePtrToInt(
      F->args().begin() + 1, llvm::Type::getInt64Ty(ctx_));
  offsetsArray_ = F->args().begin() + 3;
}

// Search for the standard library bitcode file on disk and load it into an
// LLVM module. We search for the standard library around the current executable
// and also in the current directory.
static std::unique_ptr<llvm::Module> loadStandardLibrary(llvm::LLVMContext *ctx,
                                                         StringRef filename) {
  using llvm::sys::path::append;
  using llvm::sys::path::parent_path;

  llvm::SMDiagnostic error;
  // Figure out the location of the current executable.
  auto mainExec =
      llvm::sys::fs::getMainExecutable(nullptr, (void *)&loadStandardLibrary);
  StringRef basePath = parent_path(mainExec);

  // Search for the standard library starting at the location of the executable.
  // Go up the tree up to three levels (by removing the last directory name).
  for (int i = 0; i < 3; i++) {
    llvm::SmallString<256> libPath(basePath);
    append(libPath, filename);
    if (llvm::sys::fs::exists(libPath)) {
      auto res = llvm::parseIRFile(libPath, error, *ctx);

      // If we could not parse the bitcode file then print an error.
      if (!res.get()) {
        error.print(mainExec.c_str(), llvm::errs());
      }
      return res;
    }

    // Go up the filesystem tree.
    basePath = parent_path(basePath);
  }

  return llvm::parseIRFile(filename, error, *ctx);
}

/// Register a diagnostics handler that prevents the compiler from printing to
/// stdout.
static void registerEmptyDiagHandler(llvm::LLVMContext &ctx) {
#if LLVM_VERSION_MAJOR >= 6
  ctx.setDiagnosticHandlerCallBack(
      [](const llvm::DiagnosticInfo &DI, void *Context) {
        // Do not emit any warnings or diagnostics when JITting.
      });
#else
  ctx.setDiagnosticHandler([](const llvm::DiagnosticInfo &DI, void *Context) {
    // Do not emit any warnings or diagnostics when JITting.
  });
#endif
}

void LLVMIRGen::initCodeGen() {
  instrNumbering_.reset(new InstructionNumbering(*F_));
  // Load the jit library as a new module.
  llmodule_ = loadStandardLibrary(&ctx_, "libjit.bc");
  GLOW_ASSERT(llmodule_.get() && "Unable to load the JIT library.");

  // By default, LLVM would emit some diagnostics, remarks, etc. It is fine for
  // a static compiler, but not necessary for a JIT. Let's disable it by
  // providing a dummy diagnostics handler, that does not emit anything.
  // In particular, this allows us to get rid of the annoying "cannot vectorize"
  // warnings.
  registerEmptyDiagHandler(ctx_);

  // Assign the target information to the module.
  llmodule_->setDataLayout(getTargetMachine().createDataLayout());

  // Create the entry function into the LLVM module.
  auto int8PtrTy = llvm::Type::getInt8PtrTy(ctx_);
  auto sizeTPtrTy = llvm::Type::getIntNPtrTy(ctx_, sizeof(size_t) * 8);
  // The entry point has the following API:
  // void entry(uint8_t *baseConstantWeightVars, uint8_t
  // *baseInoutWeightVars, uint8_t *baseActivations, size_t *offsets);
  llvm::Type *voidTy = llvm::Type::getVoidTy(ctx_);
  llvm::FunctionType *jitFuncTy = llvm::FunctionType::get(
      voidTy, {int8PtrTy, int8PtrTy, int8PtrTy, sizeTPtrTy}, false);
  auto *func = llvm::Function::Create(
      jitFuncTy, llvm::Function::ExternalLinkage, "main", llmodule_.get());

  // Setup the entry basic block and initialize the IR builder.
  llvm::BasicBlock *entry_bb = llvm::BasicBlock::Create(ctx_, "entry", func);
  builder_ = llvm::make_unique<llvm::IRBuilder<>>(entry_bb);

  // Initialize the debug information emission.
  initDebugInfo();
}

/// \returns the LLVM type corresponding to the type of elements stored in \p
/// val.
llvm::Type *LLVMIRGen::getElementType(llvm::IRBuilder<> &builder, const Value *val) {
  switch (val->getElementType()) {
  case ElemKind::IndexTy:
    return builder.getIntNTy(sizeof(size_t) * 8);
  case ElemKind::FloatTy:
    return builder.getFloatTy();
  case ElemKind::Int8QTy:
    return builder.getInt8Ty();
  case ElemKind::Int32QTy:
    return builder.getInt32Ty();
  }
  return nullptr;
}

void LLVMIRGen::performCodeGen() {
  auto *func = builder_->GetInsertBlock()->getParent();
  loadBaseAddresses(*builder_);

  generateLLVMIRForModule(*builder_);

  // Terminate the function.
  builder_->CreateRetVoid();

  if (dumpIR) {
    llvm::outs() << "LLVM module before optimizations:\n";
    llmodule_->print(llvm::outs(), nullptr);
  }

  // Perform verification if no debug info is being emitted.
  // Otherwise, the verification is performed later by
  // generateDebugInfo, once the debug info emission is finalized.
  if (!emitDebugInfo) {
    // Perform verification, but ignore any debug info errors for now.
    // Debug info errors will be checked later by generateDebugInfo.
    bool brokenDebugInfo = false;
    (void)brokenDebugInfo;
    assert(!llvm::verifyModule(getModule(), &llvm::errs(), &brokenDebugInfo) &&
           "LLVM module verification error");
  }

  // Optimize the module.
  optimizeLLVMModule(func, getTargetMachine());

  // Generate debug information.
  generateDebugInfo();

  // And pass the ownership to the JIT.

  if (dumpIR) {
    llvm::outs() << "LLVM module after optimizations:\n";
    llmodule_->print(llvm::outs(), nullptr);
  }

  if (dumpJitAsm) {
    llvm::SmallVector<char, 0> asmBuffer;
    llvm::raw_svector_ostream asmStream(asmBuffer);
    llvm::legacy::PassManager PM;
    getTargetMachine().addPassesToEmitFile(
        PM, asmStream, llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
    PM.run(*llmodule_);
    llvm::outs() << asmStream.str();
  }
}

llvm::Value *LLVMIRGen::emitValueAddress(llvm::IRBuilder<> &builder,
                                         glow::Value *val) {
  assert(allocationsInfo_.allocatedAddressed_.count(val) &&
         "Value address was not allocated");
  auto sizeTTy = builder.getIntNTy(sizeof(size_t) * 8);
  llvm::Type *T = nullptr;

  switch (val->getElementType()) {
  case ElemKind::FloatTy:
    T = llvm::Type::getFloatPtrTy(ctx_);
    break;
  case ElemKind::Int8QTy:
    T = llvm::Type::getInt8PtrTy(ctx_);
    break;
  case ElemKind::IndexTy:
    T = sizeTTy->getPointerTo();
    break;
  default:
    llvm_unreachable("Unimplemented");
    break;
  }

  assert(allocationsInfo_.valueNumbers_.count(val));
  auto &kindAndValue = allocationsInfo_.valueNumbers_[val];

  // Get the required base address.
  llvm::Value *baseAddrValue = nullptr;
  switch (kindAndValue.first) {
  case AllocationsInfo::ValueKind::Activation:
    baseAddrValue = baseActivationsAddr_;
    break;
  case AllocationsInfo::ValueKind::ConstantWeight:
    baseAddrValue = baseConstantWeightVarsAddr_;
    break;
  case AllocationsInfo::ValueKind::MutableWeight:
    baseAddrValue = baseMutableWeightVarsAddr_;
    break;
  }

  // Use relative addressing.
  // Get offset.
  auto valueIdx = llvm::ConstantInt::get(sizeTTy, kindAndValue.second);
  auto offsetAddr = builder.CreateGEP(sizeTTy, offsetsArray_, valueIdx);
  auto offsetValue = builder.CreateLoad(sizeTTy, offsetAddr);
  // Add offset to the base address.
  llvm::Value *addr = builder.CreateAdd(baseAddrValue, offsetValue);
  return builder.CreateIntToPtr(addr, T);
}

llvm::Value *
LLVMIRGen::emitConstOffsetsArray(llvm::IRBuilder<> &builder,
                                 const AllocationsInfo &allocationsInfo) {
  auto sizeTType = builder.getIntNTy(sizeof(size_t) * 8);
  std::vector<llvm::Constant *> elems(allocationsInfo.valueNumbers_.size());
  for (auto &I : allocationsInfo.valueNumbers_) {
    auto *V = I.first;
    auto offset = I.second.second;
    elems[offset] = llvm::ConstantInt::get(
        sizeTType, allocationsInfo.allocatedAddressed_.lookup(V));
  }
  auto *arr = llvm::ConstantArray::get(
      llvm::ArrayType::get(sizeTType, elems.size()), elems);
  // Ensure that the same casted global variable is used for the equivalent
  // const arrays. This is important for the later function specialization pass.
  // LLVM does not do it automatically for this code pattern involving global
  // variables. It also reduces the number of variables.
  auto &constArrayVar = constArrayPtrs_[arr];
  if (constArrayVar && constArrayVar->getType() == sizeTType->getPointerTo())
    return constArrayVar;

  auto *M = builder.GetInsertBlock()->getModule();

  auto *G = new llvm::GlobalVariable(*M, arr->getType(), true,
                                     llvm::GlobalValue::InternalLinkage, arr);
  constArrayVar = builder.CreateBitCast(G, sizeTType->getPointerTo());
  return constArrayVar;
}

llvm::Value *LLVMIRGen::emitConstArray(llvm::IRBuilder<> &builder,
                                       llvm::ArrayRef<size_t> vals) {
  auto SizeTType = builder.getIntNTy(sizeof(size_t) * 8);
  std::vector<llvm::Constant *> elems;
  for (auto I : vals) {
    elems.push_back(llvm::ConstantInt::get(SizeTType, I));
  }
  return emitConstArray(builder, elems, SizeTType);
}

llvm::Value *LLVMIRGen::emitConstArray(llvm::IRBuilder<> &builder,
                                       llvm::ArrayRef<llvm::Constant *> vals,
                                       llvm::Type *elemTy) {
  std::vector<llvm::Constant *> elems;
  for (auto I : vals) {
    elems.push_back(cast<llvm::Constant>(builder.CreateBitCast(I, elemTy)));
  }
  auto *arr = llvm::ConstantArray::get(
      llvm::ArrayType::get(elemTy, elems.size()), elems);
  // Ensure that the same casted global variable is used for the equivalent
  // const arrays. This is important for the later function specialization pass.
  // LLVM does not do it automatically for this code pattern involving global
  // variables. It also reduces the number of variables.
  auto &constArrayVar = constArrayPtrs_[arr];
  if (constArrayVar && constArrayVar->getType() == elemTy->getPointerTo())
    return constArrayVar;

  auto *M = builder.GetInsertBlock()->getModule();

  auto *G = new llvm::GlobalVariable(*M, arr->getType(), true,
                                     llvm::GlobalValue::InternalLinkage, arr);
  constArrayVar = builder.CreateBitCast(G, elemTy->getPointerTo());
  return constArrayVar;
}

llvm::Value *LLVMIRGen::emitValueDims(llvm::IRBuilder<> &builder,
                                      glow::Value *val) {
  auto dims = val->dims();
  return emitConstArray(builder, dims);
}

llvm::Value *LLVMIRGen::emitValueSize(llvm::IRBuilder<> &builder,
                                      glow::Value *val) {
  return builder.getIntN(sizeof(size_t) * 8, val->size());
}

llvm::Value *LLVMIRGen::emitConstF32(llvm::IRBuilder<> &builder, float val) {
  return llvm::ConstantFP::get(llvm::Type::getFloatTy(ctx_), val);
}

llvm::Value *LLVMIRGen::emitConstI32(llvm::IRBuilder<> &builder, int32_t val) {
  return builder.getInt32(val);
}

llvm::Value *LLVMIRGen::emitConstI8(llvm::IRBuilder<> &builder, int8_t val) {
  return builder.getInt8(val);
}

llvm::Value *LLVMIRGen::emitConstSizeT(llvm::IRBuilder<> &builder, size_t val) {
  return builder.getIntN(sizeof(size_t) * 8, val);
}

llvm::Value *LLVMIRGen::emitConst(llvm::IRBuilder<> &builder, float val,
                                  glow::ElemKind kind) {
  switch (kind) {
  case ElemKind::FloatTy:
    return llvm::ConstantFP::get(llvm::Type::getFloatTy(ctx_), val);
  case ElemKind::IndexTy:
    return builder.getIntN(sizeof(size_t) * 8, static_cast<size_t>(val));
  case ElemKind::Int8QTy:
    return builder.getInt8(static_cast<int8_t>(val));
  case ElemKind::Int32QTy:
    return builder.getInt32(static_cast<int32_t>(val));
  }
  llvm_unreachable("Unknown element type");
}

llvm::Value *LLVMIRGen::emitStringConst(llvm::IRBuilder<> &builder,
                                        llvm::StringRef str) {
  llvm::Constant *constStrArray =
      llvm::ConstantDataArray::getString(ctx_, str, true);
  llvm::GlobalVariable *gvarStr = new llvm::GlobalVariable(
      *llmodule_, constStrArray->getType(), true,
      llvm::GlobalValue::PrivateLinkage, constStrArray, ".str");
  gvarStr->setAlignment(1);
  return builder.CreateBitCast(gvarStr, builder.getInt8PtrTy());
}

llvm::Function *LLVMIRGen::getFunction(const std::string &name) {
  auto *F = llmodule_->getFunction("libjit_" + name);
#ifndef NDEBUG
  if (!F) {
    llvm::errs() << "Unable to load the function: libjit_" << name << "\n";
  }
#endif
  GLOW_ASSERT(F && "Unable to load the function");
  return F;
}

llvm::Function *LLVMIRGen::getFunction(const std::string &name,
                                       ElemKind elemTy) {
  auto get = [this](llvm::StringRef funcName) {
    auto *F = llmodule_->getFunction(funcName);
#ifndef NDEBUG
    if (!F) {
      llvm::errs() << "Unable to load the function: " << funcName << "\n";
    }
#endif
    GLOW_ASSERT(F && "Unable to load the function");
    return F;
  };
  switch (elemTy) {
  case ElemKind::FloatTy:
    return get("libjit_" + name + "_f");
  case ElemKind::Int8QTy:
    return get("libjit_" + name + "_i8");
  case ElemKind::Int32QTy:
    return get("libjit_" + name + "_i32");
  case ElemKind::IndexTy:
    return get("libjit_" + name + "_u");
  default:
    GLOW_ASSERT("Unsupported element type");
  }
}

/// Create LLVM IR for the for loop with a loop count specified by the only
/// parameter of the enclosing function.
/// \returns a pair of basic blocks. The first BB is the BB of the loop body,
/// the second BB is the loop exit BB.
static std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
createLoop(llvm::IRBuilder<> &builder, llvm::LLVMContext &ctx,
           llvm::Value *numElements) {
  auto sizeTTy = builder.getIntNTy(sizeof(size_t) * 8);
  auto *initVal = llvm::ConstantInt::get(sizeTTy, 0);

  // Make the new basic block for the loop header. Insert it after current
  // block.
  llvm::Function *func = builder.GetInsertBlock()->getParent();
  auto *preheaderBB = builder.GetInsertBlock();
  auto *loopBB = llvm::BasicBlock::Create(ctx, "loop", func);

  // Insert a jump from the current block to the loopBB.
  builder.CreateBr(loopBB);

  // Start insertion in LoopBB.
  builder.SetInsertPoint(loopBB);

  // Create the PHI node with an entry for initial value.
  llvm::PHINode *var = builder.CreatePHI(sizeTTy, 2);
  var->addIncoming(initVal, preheaderBB);

  // Emit the step value.
  auto *stepVal = llvm::ConstantInt::get(sizeTTy, 1);
  auto *nextVal = builder.CreateAdd(var, stepVal, "nextvar", /* HasNUW */ true,
                                    /* HasNSW */ true);
  // Compute the end condition.
  auto *endCond = builder.CreateICmpULT(nextVal, numElements, "loopcond");

  // Create the "after loop" block and insert it.
  auto *afterBB = llvm::BasicBlock::Create(ctx, "afterloop", func);

  // Insert the conditional branch at the end of the loopBB.
  auto *backEdge = builder.CreateCondBr(endCond, loopBB, afterBB);
  // Add explicit loop llvm.loop.vectorize.enable metadata to the generated
  // loop to help the LLVM vectorizer. Without this metadata, LLVM loop
  // vectorizer bails on long data-parallel loops with a lot of operations. This
  // metadata forces it to vectorize them anyways.
  llvm::SmallVector<llvm::Metadata *, 4> args;
  // Reserve operand 0 for loop id self reference.
  //
  // Initialize it with a special temporary metadata node, which is typically
  // used to create cyclic metadata structures. tmpMD is a unique_ptr and thus
  // will be freed automatically when it goes out of scope.
  llvm::TempMDTuple tmpMD = llvm::MDNode::getTemporary(ctx, llvm::None);
  args.push_back(tmpMD.get());
  llvm::Metadata *Vals[] = {
      // Reserve operand 0 for loop id self reference.
      llvm::MDString::get(ctx, "llvm.loop.vectorize.enable"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt1Ty(ctx), true))};
  args.push_back(llvm::MDNode::get(ctx, Vals));
  auto *loopMD = llvm::MDNode::get(ctx, args);
  // Set the first operand to itself.
  loopMD->replaceOperandWith(0, loopMD);
  backEdge->setMetadata(llvm::LLVMContext::MD_loop, loopMD);
  // Add a new entry to the PHI node for the backedge.
  var->addIncoming(nextVal, loopBB);
  builder.SetInsertPoint(afterBB);
  return std::make_pair(loopBB, afterBB);
}

/// Emit the address of the buffer \p v inside a data-parallel kernel \p kernel
/// using the mapping provided by \p bufferToArgNum.
static llvm::Value *
emitBufferAddress(llvm::IRBuilder<> &builder, Value *val,
                  llvm::Function *kernel,
                  llvm::DenseMap<Value *, int> &bufferToArgNum) {
  assert(bufferToArgNum.count(val) && "Buffer should be in the map");
  return kernel->args().begin() + bufferToArgNum[val];
}

/// Emit the function that implements a data-parallel kernel and calls it.
///
/// The generated kernel functions get buffers as their parameters. The buffers
/// are uniqued, so that any buffer is passed as argument to the kernel function
/// only once. This allows us to mark all parameters of the generated kernel as
/// noalias. As a result, the LLVM optimizer makes use of the noalias attributes
/// and produces nicely vectorized code for the generated data-parallel kernels.
void LLVMIRGen::emitDataParallelKernel(llvm::IRBuilder<> &builder,
                                       llvm::ArrayRef<Instruction *> bundle) {
  if (bundle.empty())
    return;
  llvm::Type *voidTy = llvm::Type::getVoidTy(ctx_);
  // Types of arguments for the kernel function being generated.
  llvm::SmallVector<llvm::Type *, 32> argTypes;
  // Map each buffer used by the kernel to the argument number of the kernel
  // function. This ensures that same buffer is always mapped to the same
  // argument.
  llvm::DenseMap<Value *, int> bufferToArgNum;
  // Buffers to be passed to the kernel function as arguments.
  llvm::SmallVector<llvm::Value *, 32> buffers;
  // Collect unique buffers used by the instructions of the kernel.
  for (const auto I : bundle) {
    for (const auto &Op : I->getOperands()) {
      auto *buf = Op.first;
      if (!bufferToArgNum.count(buf)) {
        bufferToArgNum[buf] = argTypes.size();
        buffers.push_back(emitValueAddress(builder, buf));
        argTypes.push_back(getElementType(builder, buf)->getPointerTo());
      }
    }
  }

  // Create stacked kernel function type.
  llvm::FunctionType *kernelFuncTy =
      llvm::FunctionType::get(voidTy, argTypes, false);
  auto *kernelFunc =
      llvm::Function::Create(kernelFuncTy, llvm::Function::InternalLinkage,
                             "libjit_stacked_kernel", llmodule_.get());
  // Mark all kernel function buffer parameters as no-alias, because above
  // we ensured that they are uniqued.
  for (unsigned paramIdx = 0; paramIdx < bufferToArgNum.size(); ++paramIdx) {
    kernelFunc->addParamAttr(paramIdx, llvm::Attribute::AttrKind::NoAlias);
  }

  // Create the entry BB.
  llvm::BasicBlock *entryBB =
      llvm::BasicBlock::Create(ctx_, "entry", kernelFunc);
  llvm::IRBuilder<> kernelBuilder(entryBB);
  // Number of tensor elements.
  auto *numElements =
      emitValueSize(kernelBuilder, bundle[0]->getOperand(0).first);
  // Create a loop inside the stacked kernel function being generated.
  auto loopBBs = createLoop(kernelBuilder, ctx_, numElements);

  // Get the index parameter of the loop.
  // This is the PHI node of the BB.
  auto *kernelLoopIdx = dyn_cast<llvm::PHINode>(loopBBs.first->begin());
  assert(kernelLoopIdx && "Could not find the loop index");
  // Insert the body of the loop right after the PHI node.
  kernelBuilder.SetInsertPoint(loopBBs.first->getFirstNonPHIOrDbg());
  // Iterate over stacked instructions and create a kernel invocations per
  // instruction.
  for (auto &BI : bundle) {
    // Name of the stacked operation to be invoked.
    assert(BI->isDataParallel() && "Data parallel operation is expected");
    generateLLVMIRForDataParallelInstr(kernelBuilder, BI, kernelFunc,
                                       bufferToArgNum, kernelLoopIdx);
  }
  kernelBuilder.SetInsertPoint(loopBBs.second);
  // Add a return.
  kernelBuilder.CreateRetVoid();

  // Emit a call of the kernel.
  createCall(builder, kernelFunc, buffers);
  generateFunctionDebugInfo(kernelFunc);
}

/// Check if the provided operand overlaps with an operand of an instruction
/// already in the bundle, but is not exactly the same memory region.
/// Such memory regions cannot be considered data-parallel in the scope of the
/// same kernel.
///
/// \param allocationsInfo information about allocations
/// \param bundle current bundle of stacked instructions
/// \param buf the buffer operand to be checked for overlaps with the \p bundle.
static bool isOverlappingWithAnyBundleBufferOperands(
    AllocationsInfo &allocationsInfo,
    llvm::SmallVectorImpl<Instruction *> &bundle, Value *buf) {
  auto addr1 = allocationsInfo.allocatedAddressed_[buf];
  auto size1 = buf->getSizeInBytes();
  for (auto bi : bundle) {
    for (auto bop : bi->getOperands()) {
      auto buf2 = bop.first;
      auto addr2 = allocationsInfo.allocatedAddressed_[buf2];
      auto size2 = buf2->getSizeInBytes();
      // It is fine, if buffers of different data-parallel instructions are
      // allocated exactly the same memory region.
      if (addr1 == addr2 && size1 == size2) {
        continue;
      }
      if ((addr1 >= addr2 && addr1 < addr2 + size2) ||
          (addr2 >= addr1 && addr2 < addr1 + size1)) {
        // Two intervals overlap, but are not the same.
        return true;
      }
    }
  }
  return false;
}

void LLVMIRGen::generateLLVMIRForModule(llvm::IRBuilder<> &builder) {
  // Go over the instructions and try to group them into bundles.
  auto &instrs = F_->getInstrs();

  // Group instructions into bundles of shape compatible data parallel
  // instructions and emit them.
  llvm::SmallVector<Instruction *, 32> bundle;
  for (auto &I : instrs) {
    if (!I.isDataParallel()) {
      // Ignore memory management instructions as they are handled by the
      // MemoryManager and are NOPs for a JIT.
      if (isa<AllocActivationInst>(&I) || isa<DeallocActivationInst>(&I) ||
          isa<TensorViewInst>(&I))
        continue;
      emitDataParallelKernel(builder, bundle);
      bundle.clear();
      generateLLVMIRForInstr(builder, &I);
      continue;
    }

    // This is a data parallel instruction.

    // Check if the current instruction is shape compatible with the bundle.
    bool isBundleCompatible = true;
    if (!bundle.empty()) {
      auto val = I.getOperand(0).first;
      auto bundleVal = bundle.back()->getOperand(0).first;
      // Check if shapes have the same amount of elements.
      isBundleCompatible = val->size() == bundleVal->size();
    }

    // Check all mutated operands of the current instruction. Their memory
    // regions should not have a non-exact overlap with any operands of the
    // bundled instructions. In case this condition does not hold, the current
    // instruction cannot be included into the data-parallel bundle, because
    // overlapping operand buffers are not data parallel.
    for (auto op : I.getOperands()) {
      // Skip non-mutated operands.
      if (op.second == OperandKind::In)
        continue;
      // If the mutated operand buffer overlaps with any buffer already used by
      // the bundle, the current instruction cannot become a part of the bundle.
      if (isOverlappingWithAnyBundleBufferOperands(allocationsInfo_, bundle,
                                                   op.first)) {
        isBundleCompatible = false;
        break;
      }
    }

    // If the instruction cannot be added to the current bundle, emit the kernel
    // for the current bundle and start a new bundle.
    if (!isBundleCompatible) {
      emitDataParallelKernel(builder, bundle);
      bundle.clear();
    }
    // Add a data parallel instruction to the bundle.
    bundle.push_back(&I);
  }

  emitDataParallelKernel(builder, bundle);
}

void LLVMIRGen::generateLLVMIRForDataParallelInstr(
    llvm::IRBuilder<> &builder, glow::Instruction *I, llvm::Function *kernel,
    llvm::DenseMap<Value *, int> &bufferToArgNum, llvm::Value *loopCount) {
  setCurrentDebugLocation(builder, I);
  assert(I->isDataParallel() && "Expected a data parallel instruction");
  switch (I->getKind()) {

#define ARITHMETIC_UNARY_OP_WITH_IMM_CASE(INST_NAME_, FUN_NAME_, VALUE_)       \
  case Kinded::Kind::INST_NAME_##InstKind: {                                   \
    auto *AN = cast<INST_NAME_##Inst>(I);                                      \
    auto *dest = AN->getDest();                                                \
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);  \
    auto *elementTy = getElementType(builder, dest);                           \
    auto value = AN->get##VALUE_();                                            \
    auto *F = getFunction(FUN_NAME_ "_kernel", dest->getElementType());        \
    auto *pointerNull =                                                        \
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());             \
    if (dest->getType()->isQuantizedType()) {                                  \
      auto *destTy = dest->getType();                                          \
      /* Quantize value based on the output type. */                           \
      /* Perform this early and let jit library to work */                     \
      /* with quantized number. */                                             \
      TensorQuantizationParams TQP{destTy->getScale(), destTy->getOffset()};   \
      auto quantizedValue = quantization::quantize(value, TQP);                \
      auto *val = emitConstI8(builder, quantizedValue);                        \
      auto *stackedOpCall =                                                    \
          createCall(builder, F, {loopCount, val, pointerNull, pointerNull});  \
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,        \
                                         "buffer.element.addr");               \
      builder.CreateStore(stackedOpCall, destAddr);                            \
    } else {                                                                   \
      auto *val = emitConst(builder, value, dest->getElementType());           \
      auto *stackedOpCall =                                                    \
          createCall(builder, F, {loopCount, val, pointerNull, pointerNull});  \
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,        \
                                         "buffer.element.addr");               \
      builder.CreateStore(stackedOpCall, destAddr);                            \
    }                                                                          \
    break;                                                                     \
  }
    ARITHMETIC_UNARY_OP_WITH_IMM_CASE(Splat, "splat", Value);
#undef ARITHMETIC_UNARY_OP_WITH_IMM_CASE

  case Kinded::Kind::ElementSelectInstKind: {
    ElementSelectInst *ES = cast<ElementSelectInst>(I);
    auto *dest = ES->getDest();
    auto *cond = ES->getCond();
    auto *lhs = ES->getLHS();
    auto *rhs = ES->getRHS();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *condPtr = emitBufferAddress(builder, cond, kernel, bufferToArgNum);
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);

    // Need _kernel suffix since these operations are implemented as
    // "data-parallel" kernels in libjit.
    auto *F = getFunction("elementselect_kernel", dest->getElementType());

    if (lhs->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *lhsTy = lhs->getType();
      auto *rhsTy = rhs->getType();

      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *lhsOffset = emitConstI32(builder, lhsTy->getOffset());
      auto *rhsOffset = emitConstI32(builder, rhsTy->getOffset());

      // The selected value will be either lhs = s_l * (i_l - o_l) or
      // rhs = s_r * (i_r - o_r); the stored result that must be computed is
      // therefore one of:
      // (i)  i_d = (s_l / s_d) * (i_l - o_l) + o_d
      // (ii) i_d = (s_r / s_d) * (i_r - o_r) + o_d
      float destScale = destTy->getScale();
      auto lhsScaleParams = quantization::quantizeScaleOffset32To8(
          lhsTy->getScale() / destScale, lhsTy->getOffset());
      auto rhsScaleParams = quantization::quantizeScaleOffset32To8(
          rhsTy->getScale() / destScale, rhsTy->getOffset());

      auto *lhsPre = emitConstI32(builder, lhsScaleParams.pre_);
      auto *lhsPost = emitConstI32(builder, lhsScaleParams.post_);
      auto *lhsScale = emitConstI32(builder, lhsScaleParams.scale_);
      auto *rhsPre = emitConstI32(builder, rhsScaleParams.pre_);
      auto *rhsPost = emitConstI32(builder, rhsScaleParams.post_);
      auto *rhsScale = emitConstI32(builder, rhsScaleParams.scale_);

      auto *stackedOpCall = createCall(
          builder, F,
          {loopCount, condPtr, lhsPtr, rhsPtr, destOffset, lhsOffset, rhsOffset,
           lhsPre, lhsPost, lhsScale, rhsPre, rhsPost, rhsScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *stackedOpCall =
          createCall(builder, F, {loopCount, condPtr, lhsPtr, rhsPtr});
      auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }
    break;
  }
  case Kinded::Kind::IntLookupTableInstKind: {
    auto *lookupTable = cast<IntLookupTableInst>(I);
    auto *dest = lookupTable->getDest();
    auto *src = lookupTable->getSrc();
    auto *mapping = lookupTable->getMapping();

    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *mappingPtr =
        emitBufferAddress(builder, mapping, kernel, bufferToArgNum);

    auto *F = getFunction("intlookuptable_kernel", dest->getElementType());
    auto *stackedOpCall =
        builder.CreateCall(F, {loopCount, srcPtr, mappingPtr});
    auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr, loopCount,
                                       "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);

    break;
  }
#define ARITHMETIC_UNARY_OP_CASE(INST_NAME_, FUN_NAME_)                        \
  case Kinded::Kind::INST_NAME_##InstKind: {                                   \
    auto *AN = cast<INST_NAME_##Inst>(I);                                      \
    auto *dest = AN->getDest();                                                \
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);  \
    auto *srcPtr =                                                             \
        emitBufferAddress(builder, AN->getSrc(), kernel, bufferToArgNum);      \
    auto *F = getFunction(FUN_NAME_ "_kernel", dest->getElementType());        \
    auto *elementTy = getElementType(builder, dest);                           \
    auto *pointerNull =                                                        \
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());             \
    auto *stackedOpCall =                                                      \
        createCall(builder, F, {loopCount, srcPtr, pointerNull, pointerNull}); \
    auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,          \
                                       loopCount, "buffer.element.addr");      \
    builder.CreateStore(stackedOpCall, destAddr);                              \
    break;                                                                     \
  }

    ARITHMETIC_UNARY_OP_CASE(Sigmoid, "sigmoid");
    ARITHMETIC_UNARY_OP_CASE(Tanh, "tanh");
    ARITHMETIC_UNARY_OP_CASE(ElementLog, "element_log");
#undef ARITHMETIC_UNARY_OP_CASE

  case Kinded::Kind::CopyInstKind: {
    CopyInst *CI = cast<CopyInst>(I);
    auto *dest = CI->getDest();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *srcPtr =
        emitBufferAddress(builder, CI->getSrc(), kernel, bufferToArgNum);
    auto *F = getFunction("copy_kernel", dest->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *pointerNull =
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());
    auto *stackedOpCall =
        createCall(builder, F, {loopCount, srcPtr, pointerNull, pointerNull});
    auto *destAddr = builder.CreateGEP(getElementType(builder, dest), destPtr,
                                       loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }
  case Kinded::Kind::CPUMaxSplatInstKind: {
    auto *AN = cast<CPUMaxSplatInst>(I);
    auto *dest = AN->getDest();
    auto V = AN->getSplatValue();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhs = AN->getSrc();
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *F = getFunction("element_maxsplat_kernel", dest->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *pointerNull =
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());

    if (lhs->getType()->isQuantizedType()) {
      // Quantize value from the splat to the {S,O} of the lhs param.
      TensorQuantizationParams TQP{lhs->getType()->getScale(),
                                   lhs->getType()->getOffset()};
      auto quantizedValue = quantization::quantize(V, TQP);
      auto *val = emitConst(builder, quantizedValue, lhs->getElementType());
      auto *stackedOpCall =
          createCall(builder, F, {loopCount, val, lhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *val = emitConst(builder, V, lhs->getElementType());
      auto *stackedOpCall =
          createCall(builder, F, {loopCount, val, lhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }

    break;
  }

#undef ARITHMETIC_UNARY_OP_CASE

#define ARITHMETIC_BINARY_OP_CASE(INST_NAME_, FUN_NAME_)                       \
  case Kinded::Kind::INST_NAME_##InstKind: {                                   \
    auto *AN = cast<INST_NAME_##Inst>(I);                                      \
    auto *dest = AN->getDest();                                                \
    auto *lhs = AN->getLHS();                                                  \
    auto *rhs = AN->getRHS();                                                  \
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);  \
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);    \
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);    \
                                                                               \
    auto *F = getFunction(FUN_NAME_ "_kernel", dest->getElementType());        \
    auto *elementTy = getElementType(builder, dest);                           \
    auto *pointerNull =                                                        \
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());             \
                                                                               \
    if (lhs->getType()->isQuantizedType()) {                                   \
      auto *destTy = dest->getType();                                          \
      auto *lhsTy = lhs->getType();                                            \
      auto *rhsTy = rhs->getType();                                            \
                                                                               \
      auto *destOffset = emitConstI32(builder, destTy->getOffset());           \
      auto *lhsOffset = emitConstI32(builder, lhsTy->getOffset());             \
      auto *rhsOffset = emitConstI32(builder, rhsTy->getOffset());             \
                                                                               \
      float destScale = destTy->getScale();                                    \
                                                                               \
      auto lhsScaleParams = quantization::quantizeScaleOffset32To8(            \
          lhsTy->getScale() / destScale, lhsTy->getOffset());                  \
      auto rhsScaleParams = quantization::quantizeScaleOffset32To8(            \
          rhsTy->getScale() / destScale, rhsTy->getOffset());                  \
                                                                               \
      auto *lhsPre = emitConstI32(builder, lhsScaleParams.pre_);               \
      auto *lhsPost = emitConstI32(builder, lhsScaleParams.post_);             \
      auto *lhsScale = emitConstI32(builder, lhsScaleParams.scale_);           \
      auto *rhsPre = emitConstI32(builder, rhsScaleParams.pre_);               \
      auto *rhsPost = emitConstI32(builder, rhsScaleParams.post_);             \
      auto *rhsScale = emitConstI32(builder, rhsScaleParams.scale_);           \
                                                                               \
      auto *stackedOpCall = createCall(builder, F,                             \
                                       {loopCount, lhsPtr, rhsPtr, destOffset, \
                                        lhsOffset, rhsOffset, lhsPre, lhsPost, \
                                        lhsScale, rhsPre, rhsPost, rhsScale}); \
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,         \
                                         loopCount, "buffer.element.addr");    \
      builder.CreateStore(stackedOpCall, destAddr);                            \
    } else {                                                                   \
      auto *stackedOpCall =                                                    \
          createCall(builder, F, {loopCount, lhsPtr, rhsPtr, pointerNull});    \
      auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,        \
                                         loopCount, "buffer.element.addr");    \
      builder.CreateStore(stackedOpCall, destAddr);                            \
    }                                                                          \
    break;                                                                     \
  }
    ARITHMETIC_BINARY_OP_CASE(ElementAdd, "element_add");
    ARITHMETIC_BINARY_OP_CASE(ElementSub, "element_sub");
    ARITHMETIC_BINARY_OP_CASE(ElementMax, "elementmax");
    ARITHMETIC_BINARY_OP_CASE(ElementMin, "elementmin");
#undef ARITHMETIC_BINARY_OP_CASE

  case Kinded::Kind::ElementCmpLTEInstKind: {
    auto *CI = cast<ElementCmpLTEInst>(I);
    auto *dest = CI->getDest();
    auto *lhs = CI->getLHS();
    auto *rhs = CI->getRHS();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);

    // Need _kernel suffix since these operations are implemented as
    // "data-parallel" kernels in libjit.
    auto *F = getFunction("element_cmp_lte_kernel", dest->getElementType());
    auto *elementTy = getElementType(builder, dest);

    if (lhs->getType()->isQuantizedType()) {
      auto *lhsTy = lhs->getType();
      auto *rhsTy = rhs->getType();

      auto *lhsOffset = emitConstI32(builder, lhsTy->getOffset());
      auto *rhsOffset = emitConstI32(builder, rhsTy->getOffset());

      // We can divide both sides of the comparison by the rhs scale since it is
      // strictly positive; this saves one rescale within the backend. The
      // inequalities are:
      //     s_l * (i_l - o_l) <= s_r * (i_r - o_r)
      // <=> (s_l / s_r) * (i_l - o_l) <= i_r - o_r
      float scale = lhsTy->getScale() / rhsTy->getScale();
      auto scaleParams = quantization::quantizeScaleOffset32To8(scale, 0);
      auto *cmpPre = emitConstI32(builder, scaleParams.pre_);
      auto *cmpPost = emitConstI32(builder, scaleParams.post_);
      auto *cmpScale = emitConstI32(builder, scaleParams.scale_);

      auto *stackedOpCall = createCall(builder, F,
                                       {loopCount, lhsPtr, rhsPtr, lhsOffset,
                                        rhsOffset, cmpPre, cmpPost, cmpScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *pointerNull =
          llvm::ConstantPointerNull::get(elementTy->getPointerTo());
      auto *stackedOpCall =
          createCall(builder, F, {loopCount, lhsPtr, rhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }
    break;
  }

  case Kinded::Kind::ElementCmpEQInstKind: {
    auto *CI = cast<ElementCmpEQInst>(I);
    auto *dest = CI->getDest();

    auto *lhs = CI->getLHS();
    auto *rhs = CI->getRHS();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);

    // Need _kernel suffix since these operations are implemented as
    // "data-parallel" kernels in libjit.
    auto *F = getFunction("element_cmp_eq_kernel", dest->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *pointerNull =
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());
    auto *stackedOpCall =
        createCall(builder, F, {loopCount, lhsPtr, rhsPtr, pointerNull});
    auto *destAddr =
        builder.CreateGEP(elementTy, destPtr, loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::ElementMulInstKind: {
    auto *MI = cast<ElementMulInst>(I);
    auto *dest = MI->getDest();
    auto *lhs = MI->getLHS();
    auto *rhs = MI->getRHS();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);

    // Need _kernel suffix since these operations are implemented as
    // "data-parallel" kernels in libjit.
    auto *F = getFunction("element_mul_kernel", dest->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *pointerNull =
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());

    if (lhs->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *lhsTy = lhs->getType();
      auto *rhsTy = rhs->getType();

      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *lhsOffset = emitConstI32(builder, lhsTy->getOffset());
      auto *rhsOffset = emitConstI32(builder, rhsTy->getOffset());

      // The multiplicative scale factor is s_l * s_r / s_d due to the equation
      //    s_d * (i_d - o_d) = s_l * (i_l - o_l) * s_r * (i_r - o_r)
      // => i_d = (s_l * s_r / s_d) * (i_l - o_l) * (i_r - o_r) + o_d
      float scale = lhsTy->getScale() * rhsTy->getScale() / destTy->getScale();
      auto scaleParams = quantization::quantizeScaleOffset32To8(scale, 0);
      auto *mulPre = emitConstI32(builder, scaleParams.pre_);
      auto *mulPost = emitConstI32(builder, scaleParams.post_);
      auto *mulScale = emitConstI32(builder, scaleParams.scale_);

      auto *stackedOpCall =
          createCall(builder, F,
                     {loopCount, lhsPtr, rhsPtr, destOffset, lhsOffset,
                      rhsOffset, mulPre, mulPost, mulScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *stackedOpCall =
          createCall(builder, F, {loopCount, lhsPtr, rhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }
    break;
  }

  case Kinded::Kind::ElementDivInstKind: {
    auto *MI = cast<ElementDivInst>(I);
    auto *dest = MI->getDest();
    auto *lhs = MI->getLHS();
    auto *rhs = MI->getRHS();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);

    // Need _kernel suffix since these operations are implemented as
    // "data-parallel" kernels in libjit.
    auto *F = getFunction("element_div_kernel", dest->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *pointerNull =
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());

    if (lhs->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *lhsTy = lhs->getType();
      auto *rhsTy = rhs->getType();

      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *lhsOffset = emitConstI32(builder, lhsTy->getOffset());
      auto *rhsOffset = emitConstI32(builder, rhsTy->getOffset());

      // The division scale factor is s_l / (s_r * s_d) due to the equation
      //    s_d * (i_d - o_d) = (s_l * (i_l - o_l)) / (s_r * (i_r - o_r))
      // => i_d = (s_l / (s_r * s_d)) * ((i_l - o_l) / (i_r - o_r)) + o_d
      float scale =
          lhsTy->getScale() / (rhsTy->getScale() * destTy->getScale());
      auto scaleParams = quantization::quantizeScaleOffset32To8(scale, 0);
      auto *divPre = emitConstI32(builder, scaleParams.pre_);
      auto *divPost = emitConstI32(builder, scaleParams.post_);
      auto *divScale = emitConstI32(builder, scaleParams.scale_);

      auto *stackedOpCall =
          createCall(builder, F,
                     {loopCount, lhsPtr, rhsPtr, destOffset, lhsOffset,
                      rhsOffset, divPre, divPost, divScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *stackedOpCall =
          createCall(builder, F, {loopCount, lhsPtr, rhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }
    break;
  }

#define ARITHMETIC_BINARY_OP_WITH_IMM_CASE(INST_NAME_, FUN_NAME_, SRC_,        \
                                           VALUE_)                             \
  case Kinded::Kind::INST_NAME_##InstKind: {                                   \
    auto *AN = cast<INST_NAME_##Inst>(I);                                      \
    auto *dest = AN->getDest();                                                \
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);  \
    auto *val = emitConstF32(builder, AN->get##VALUE_());                      \
    auto *lhsPtr =                                                             \
        emitBufferAddress(builder, AN->get##SRC_(), kernel, bufferToArgNum);   \
    auto *F = getFunction(FUN_NAME_ "_kernel", dest->getElementType());        \
    auto *elementTy = getElementType(builder, dest);                           \
    auto *pointerNull =                                                        \
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());             \
    auto *stackedOpCall =                                                      \
        createCall(builder, F, {loopCount, val, lhsPtr, pointerNull});         \
    auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,          \
                                       loopCount, "buffer.element.addr");      \
    builder.CreateStore(stackedOpCall, destAddr);                              \
    break;                                                                     \
  }
    ARITHMETIC_BINARY_OP_WITH_IMM_CASE(ElementPow, "element_pow", Base, Exp);
#undef ARITHMETIC_BINARY_OP_WITH_IMM_CASE

  default:
#ifndef NDEBUG
    llvm::errs() << "Cannot select the instruction:\n";
    I->dump(llvm::errs());
    llvm::errs() << "\n";
#endif
    GLOW_UNREACHABLE("ERROR: Cannot select the instruction.");
  }
}

void LLVMIRGen::generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                       glow::Instruction *I) {
  setCurrentDebugLocation(builder, I);
  assert(!I->isDataParallel() &&
         "data parallel instructions are not handled here");
  switch (I->getKind()) {
  case Kinded::Kind::MatMulInstKind: {
    MatMulInst *MM = cast<MatMulInst>(I);
    auto *dest = MM->getDest();
    auto *lhs = MM->getLHS();
    auto *rhs = MM->getRHS();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *lhsPtr = emitValueAddress(builder, lhs);
    auto *rhsPtr = emitValueAddress(builder, rhs);

    auto *destDims = emitValueDims(builder, dest);
    auto *lhsDims = emitValueDims(builder, lhs);
    auto *rhsDims = emitValueDims(builder, rhs);

    auto *F = getFunction("matmul", dest->getElementType());

    if (lhs->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *lhsTy = lhs->getType();
      auto *rhsTy = rhs->getType();

      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *lhsOffset = emitConstI32(builder, lhsTy->getOffset());
      auto *rhsOffset = emitConstI32(builder, rhsTy->getOffset());

      auto outScaleParams = quantization::quantizeScaleOffset32To8(
          lhsTy->getScale() * rhsTy->getScale() / destTy->getScale(), 0);

      auto *outPre = emitConstI32(builder, outScaleParams.pre_);
      auto *outPost = emitConstI32(builder, outScaleParams.post_);
      auto *outScale = emitConstI32(builder, outScaleParams.scale_);

      createCall(builder, F,
                 {destPtr, lhsPtr, rhsPtr, destDims, lhsDims, rhsDims,
                  destOffset, lhsOffset, rhsOffset, outPre, outPost, outScale});
    } else {
      createCall(builder, F,
                 {destPtr, lhsPtr, rhsPtr, destDims, lhsDims, rhsDims});
    }
    break;
  }

  case Kinded::Kind::BatchedAddInstKind: {
    BatchedAddInst *BA = cast<BatchedAddInst>(I);
    auto *dest = BA->getDest();
    auto *batch = BA->getBatch();
    auto *slice = BA->getSlice();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *batchPtr = emitValueAddress(builder, batch);
    auto *slicePtr = emitValueAddress(builder, slice);

    auto bdim = flattenCdr(batch->dims());
    auto *numSlice = emitConstSizeT(builder, bdim.first);
    auto *sliceSize = emitConstSizeT(builder, bdim.second);

    auto *F = getFunction("batchedadd", dest->getElementType());

    if (batch->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *batchTy = batch->getType();
      auto *sliceTy = slice->getType();

      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *batchOffset = emitConstI32(builder, batchTy->getOffset());
      auto *sliceOffset = emitConstI32(builder, sliceTy->getOffset());

      float destScale = destTy->getScale();

      // Here, we select parameters for scaling both summands to the
      // destination scale.
      auto batchScaleParams = quantization::quantizeScaleOffset32To8(
          batchTy->getScale() / destScale, batchTy->getOffset());
      auto sliceScaleParams = quantization::quantizeScaleOffset32To8(
          sliceTy->getScale() / destScale, sliceTy->getOffset());

      auto *batchPre = emitConstI32(builder, batchScaleParams.pre_);
      auto *batchPost = emitConstI32(builder, batchScaleParams.post_);
      auto *batchScale = emitConstI32(builder, batchScaleParams.scale_);
      auto *slicePre = emitConstI32(builder, sliceScaleParams.pre_);
      auto *slicePost = emitConstI32(builder, sliceScaleParams.post_);
      auto *sliceScale = emitConstI32(builder, sliceScaleParams.scale_);

      createCall(builder, F,
                 {destPtr, batchPtr, slicePtr, numSlice, sliceSize, destOffset,
                  batchOffset, sliceOffset, batchPre, batchPost, batchScale,
                  slicePre, slicePost, sliceScale});
    } else {
      createCall(builder, F,
                 {destPtr, batchPtr, slicePtr, numSlice, sliceSize});
    }
    break;
  }

  case Kinded::Kind::BatchedReduceAddInstKind: {
    BatchedReduceAddInst *BR = cast<BatchedReduceAddInst>(I);
    auto *dest = BR->getDest();
    auto *batch = BR->getBatch();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *batchPtr = emitValueAddress(builder, batch);

    auto *destSize = emitConstSizeT(builder, dest->size());
    auto bdim = flattenCdr(batch->dims());
    auto *numSlice = emitConstSizeT(builder, bdim.first);
    auto *sliceSize = emitConstSizeT(builder, bdim.second);

    auto *F = getFunction("batchedreduceadd", dest->getElementType());

    if (batch->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *batchTy = batch->getType();

      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *batchOffset = emitConstI32(builder, batchTy->getOffset());

      // BatchedReduceAdd is an accumulation operation, with equations
      //    s_d * (i_d - o_d) = \sum s_b * (i_b - o_b)
      // => i_d - o_d = \sum (s_b / s_d) * (i_b - o_b)
      // => i_d = (s_b / s_d ) * [\sum (i_b - o_b)] + o_d
      auto batchScaleParams = quantization::quantizeScaleOffset32To8(
          batchTy->getScale() / destTy->getScale(), batchTy->getOffset());

      auto *batchPre = emitConstI32(builder, batchScaleParams.pre_);
      auto *batchPost = emitConstI32(builder, batchScaleParams.post_);
      auto *batchScale = emitConstI32(builder, batchScaleParams.scale_);

      createCall(builder, F,
                 {destPtr, batchPtr, destSize, numSlice, sliceSize, destOffset,
                  batchOffset, batchPre, batchPost, batchScale});
    } else {
      createCall(builder, F,
                 {destPtr, batchPtr, destSize, numSlice, sliceSize});
    }
    break;
  }

  case Kinded::Kind::ConvolutionInstKind: {
    ConvolutionInst *CI = cast<ConvolutionInst>(I);
    auto *dest = CI->getDest();
    auto *src = CI->getSrc();
    auto *filter = CI->getFilter();
    auto *bias = CI->getBias();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *filterPtr = emitValueAddress(builder, filter);
    auto *biasPtr = emitValueAddress(builder, bias);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);
    auto *filterDims = emitValueDims(builder, filter);
    auto *biasDims = emitValueDims(builder, bias);

    auto *kernel = emitConstSizeT(builder, CI->getKernel());
    auto *stride = emitConstSizeT(builder, CI->getStride());
    auto *pad = emitConstSizeT(builder, CI->getPad());
    auto *group = emitConstSizeT(builder, CI->getGroup());

    const char *kernelName = "convolution";

    auto destDepth = dest->dims()[3];

    // Try to 'block' the convolution on the 'depth' dimension. We will process
    // this number output slices each iteration.
    unsigned unrollDFactor = 1;

    // In libjit_convolution_f function, 'unrollDFactor' output
    // layers will be processed together. Therefore, the number of
    // output layers in each group should be divisible by 'unrollDFactor'
    bool groupDividedBy8 = ((destDepth / CI->getGroup()) % 8) == 0;
    if (groupDividedBy8) {
      unrollDFactor = 8;
    }

    auto *unrollD = emitConstI32(builder, unrollDFactor);

    auto *F = getFunction(kernelName, dest->getElementType());

    if (src->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *srcTy = src->getType();
      auto *filterTy = filter->getType();
      auto *biasTy = bias->getType();

      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *srcOffset = emitConstI32(builder, srcTy->getOffset());
      auto *filterOffset = emitConstI32(builder, filterTy->getOffset());
      auto *biasOffset = emitConstI32(builder, biasTy->getOffset());

      // Calculate the scale of the values that come out of the matrix
      // multiplication part of the calculation.
      float matMulScale = srcTy->getScale() * filterTy->getScale();

      // Calculate the sacling parameters for the bias and output.
      auto biasScaleParam = quantization::quantizeScaleOffset32To8(
          biasTy->getScale() / matMulScale, biasTy->getOffset());
      auto outScaleParam = quantization::quantizeScaleOffset32To8(
          matMulScale / destTy->getScale(), 0);

      // Pass the pre-shift, post-shift and integer scale parameters for the
      // bias and output calculation.
      auto *biasPre = emitConstI32(builder, biasScaleParam.pre_);
      auto *biasPost = emitConstI32(builder, biasScaleParam.post_);
      auto *biasScale = emitConstI32(builder, biasScaleParam.scale_);
      auto *outPre = emitConstI32(builder, outScaleParam.pre_);
      auto *outPost = emitConstI32(builder, outScaleParam.post_);
      auto *outScale = emitConstI32(builder, outScaleParam.scale_);

      createCall(builder, F,
                 {destPtr,    srcPtr,     filterPtr,  biasPtr,   destDims,
                  srcDims,    filterDims, biasDims,   kernel,    stride,
                  pad,        group,      destOffset, srcOffset, filterOffset,
                  biasOffset, biasPre,    biasPost,   biasScale, outPre,
                  outPost,    outScale,   unrollD});
    } else {
      createCall(builder, F,
                 {destPtr, srcPtr, filterPtr, biasPtr, destDims, srcDims,
                  filterDims, biasDims, kernel, stride, pad, group, unrollD});
    }
    break;
  }

  case Kinded::Kind::CPUConvDKKC8InstKind: {
    CPUConvDKKC8Inst *CI = cast<CPUConvDKKC8Inst>(I);
    auto *dest = CI->getDest();
    auto *src = CI->getSrc();
    auto *filter = CI->getFilter();
    auto *bias = CI->getBias();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *filterPtr = emitValueAddress(builder, filter);
    auto *biasPtr = emitValueAddress(builder, bias);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);
    auto *filterDims = emitValueDims(builder, filter);
    auto *biasDims = emitValueDims(builder, bias);

    auto *kernel = emitConstSizeT(builder, CI->getKernel());
    auto *stride = emitConstSizeT(builder, CI->getStride());
    auto *pad = emitConstSizeT(builder, CI->getPad());
    auto *group = emitConstSizeT(builder, CI->getGroup());

    size_t inChannels = src->dims()[3];
    size_t outChannels = src->dims()[3];

    // Select a method for iterating on the image in the pixel (filter-first, or
    // input-first). Perform convolutions with a high channel count by scanning
    // the input image multiple times, once for each filter entry. Scan images
    // with a low channel count by scanning the image once because the filter
    // scan will fall in the cache.
    bool pixelScanFirst = (inChannels < 16);

    // The number of float8 registers that we use to process the depth channel.
    unsigned numDepthRegs = (pixelScanFirst ? 8 : 2);
    // The number of y pixels to process at once.
    unsigned sizeGroupY = (pixelScanFirst ? 1 : 5);

    // When producing output pixels process this many times of depth-strips,
    // where each chunk is float8 * numDepthRegs. This is a form of tiling. It's
    // profitable to scan multiple depth-strips of the filter if the scanned
    // memory fits in the cahce and does not get evicted before the next
    // iteration. By increasing the number strips (and using more cache memory)
    // we reduce the number of times that we iterate over the input. However, we
    // also increase the pressure on the cache that has to store the filter so
    // we can't process too many strips at once.
    unsigned depthStrips = 1;
    unsigned stripSize = 8 * numDepthRegs * inChannels;
    unsigned tileSize = 16384;
    // Increase the number of strips until we reach the output-tensor depth size
    // or until we exceed some threashold.
    while (2 * depthStrips * stripSize <= tileSize &&
           2 * depthStrips * numDepthRegs * 8 <= outChannels &&
           depthStrips < 8) {
      depthStrips *= 2;
    }

    auto *pixelScanFirstVal = emitConstI32(builder, pixelScanFirst);
    auto *numDepthRegsVal = emitConstI32(builder, numDepthRegs);
    auto *sizeGroupYVal = emitConstI32(builder, sizeGroupY);
    auto *depthStripsVal = emitConstI32(builder, depthStrips);

    const char *kernelName = "convDKKC8";
    auto *F = getFunction(kernelName, dest->getElementType());

    createCall(builder, F,
               {destPtr, srcPtr, filterPtr, biasPtr, destDims, srcDims,
                filterDims, biasDims, kernel, stride, pad, group,
                pixelScanFirstVal, numDepthRegsVal, sizeGroupYVal,
                depthStripsVal});
    break;
  }

  case Kinded::Kind::ConvolutionGradInstKind: {
    ConvolutionGradInst *CG = cast<ConvolutionGradInst>(I);
    auto *srcGrad = CG->getSrcGrad();
    auto *destGrad = CG->getDestGrad();
    auto *src = CG->getSrc();
    auto *filterGrad = CG->getFilterGrad();
    auto *srcGradPtr = emitValueAddress(builder, srcGrad);
    auto *destGradPtr = emitValueAddress(builder, destGrad);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *filterGradPtr = emitValueAddress(builder, filterGrad);
    auto *biasGradPtr = emitValueAddress(builder, CG->getBiasGrad());
    auto *filterPtr = emitValueAddress(builder, CG->getFilter());

    auto *destGradDims = emitValueDims(builder, destGrad);
    auto *srcDims = emitValueDims(builder, src);
    auto *filterGradDims = emitValueDims(builder, filterGrad);

    auto *kernel = emitConstSizeT(builder, CG->getKernel());
    auto *stride = emitConstSizeT(builder, CG->getStride());
    auto *pad = emitConstSizeT(builder, CG->getPad());

    auto *F = getFunction("convolution_grad", srcGrad->getElementType());
    createCall(builder, F,
               {srcGradPtr, destGradPtr, srcPtr, filterGradPtr, biasGradPtr,
                filterPtr, destGradDims, srcDims, filterGradDims, kernel,
                stride, pad});
    break;
  }

  case Kinded::Kind::CrossEntropyLossInstKind: {
    auto *CI = cast<CrossEntropyLossInst>(I);
    auto *P = CI->getP();
    auto *labels = CI->getLabels();
    auto *CE = CI->getCE();

    auto *CEPtr = emitValueAddress(builder, CE);
    auto *PPtr = emitValueAddress(builder, P);
    auto *labelsPtr = emitValueAddress(builder, labels);
    auto *dims = emitValueDims(builder, P);

    auto *F = getFunction("cross_entropy_loss", CE->getElementType());
    createCall(builder, F, {CEPtr, PPtr, labelsPtr, dims});
    break;
  }

  case Kinded::Kind::LocalResponseNormalizationInstKind: {
    LocalResponseNormalizationInst *LRN =
        cast<LocalResponseNormalizationInst>(I);
    auto *dest = LRN->getDest();
    auto *src = LRN->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *scalePtr = emitValueAddress(builder, LRN->getScale());

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);
    auto *halfWindow = emitConstSizeT(builder, LRN->getHalfWindowSize());
    auto *alpha = emitConstF32(builder, LRN->getAlpha());
    auto *beta = emitConstF32(builder, LRN->getBeta());
    auto *k = emitConstF32(builder, LRN->getK());

    auto *F =
        getFunction("local_response_normalization", dest->getElementType());
    createCall(builder, F,
               {destPtr, srcPtr, scalePtr, destDims, srcDims, halfWindow, alpha,
                beta, k});
    break;
  }

  case Kinded::Kind::LocalResponseNormalizationGradInstKind: {
    LocalResponseNormalizationGradInst *LRNG =
        llvm::cast<LocalResponseNormalizationGradInst>(I);
    auto *srcGrad = LRNG->getSrcGrad();
    auto *dest = LRNG->getDest();
    auto *srcGradPtr = emitValueAddress(builder, srcGrad);
    auto *destGradPtr = emitValueAddress(builder, LRNG->getDestGrad());
    auto *srcPtr = emitValueAddress(builder, LRNG->getSrc());
    auto *destPtr = emitValueAddress(builder, dest);
    auto *scalePtr = emitValueAddress(builder, LRNG->getScale());

    auto *destDims = emitValueDims(builder, dest);

    auto *halfWindow = emitConstSizeT(builder, LRNG->getHalfWindowSize());
    auto *alpha = emitConstF32(builder, LRNG->getAlpha());
    auto *beta = emitConstF32(builder, LRNG->getBeta());

    auto *F = getFunction("local_response_normalization_grad",
                          srcGrad->getElementType());
    createCall(builder, F,
               {srcGradPtr, destGradPtr, srcPtr, destPtr, scalePtr, destDims,
                halfWindow, alpha, beta});
    break;
  }

  case Kinded::Kind::PoolMaxInstKind: {
    PoolMaxInst *PM = cast<PoolMaxInst>(I);
    auto *dest = PM->getDest();
    auto *src = PM->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *kernel = emitConstSizeT(builder, PM->getKernel());
    auto *stride = emitConstSizeT(builder, PM->getStride());
    auto *pad = emitConstSizeT(builder, PM->getPad());

    auto *F = getFunction("pool_max", dest->getElementType());
    createCall(builder, F,
               {srcPtr, destPtr, srcDims, destDims, kernel, stride, pad});
    break;
  }

  case Kinded::Kind::PoolMaxWithXYInstKind: {
    PoolMaxWithXYInst *PMXY = cast<PoolMaxWithXYInst>(I);
    auto *dest = PMXY->getDest();
    auto *src = PMXY->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *srcXYPtr = emitValueAddress(builder, PMXY->getSrcXY());

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *kernel = emitConstSizeT(builder, PMXY->getKernel());
    auto *stride = emitConstSizeT(builder, PMXY->getStride());
    auto *pad = emitConstSizeT(builder, PMXY->getPad());

    auto *F = getFunction("pool_max_xy", dest->getElementType());
    createCall(
        builder, F,
        {srcPtr, destPtr, srcXYPtr, srcDims, destDims, kernel, stride, pad});
    break;
  }

  case Kinded::Kind::PoolMaxWithXYGradInstKind: {
    PoolMaxWithXYGradInst *PMG = cast<PoolMaxWithXYGradInst>(I);
    auto *srcGrad = PMG->getSrcGrad();
    auto *srcGradPtr = emitValueAddress(builder, srcGrad);
    auto *destGradPtr = emitValueAddress(builder, PMG->getDestGrad());
    auto *srcXYPtr = emitValueAddress(builder, PMG->getSrcXY());

    auto *srcGradDims = emitValueDims(builder, srcGrad);
    auto *destDims = emitValueDims(builder, PMG->getDest());

    auto *F = getFunction("pool_max_xy_grad", srcGrad->getElementType());
    createCall(builder, F,
               {srcGradPtr, destGradPtr, srcXYPtr, srcGradDims, destDims});
    break;
  }

  case Kinded::Kind::PoolAvgInstKind: {
    PoolAvgInst *PA = cast<PoolAvgInst>(I);
    auto *dest = PA->getDest();
    auto *src = PA->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *kernel = emitConstSizeT(builder, PA->getKernel());
    auto *stride = emitConstSizeT(builder, PA->getStride());
    auto *pad = emitConstSizeT(builder, PA->getPad());

    if (src->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *srcTy = src->getType();
      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *srcOffset = emitConstI32(builder, srcTy->getOffset());
      // Reduce resulting scale by a factor of PA->getKernel() * PA->getKernel()
      // since each subtensor value is divided by the area of kernel.
      auto outScaleParam = quantization::quantizeScaleOffset32To8(
          srcTy->getScale() / destTy->getScale() /
              (PA->getKernel() * PA->getKernel()),
          destTy->getOffset());
      auto *outPre = emitConstI32(builder, outScaleParam.pre_);
      auto *outPost = emitConstI32(builder, outScaleParam.post_);
      auto *outScale = emitConstI32(builder, outScaleParam.scale_);

      auto *F = getFunction("pool_avg", dest->getElementType());
      createCall(builder, F,
                 {srcPtr, destPtr, srcDims, destDims, kernel, stride, pad,
                  destOffset, srcOffset, outPre, outPost, outScale});
      break;
    } else {
      auto *F = getFunction("pool_avg", dest->getElementType());
      createCall(builder, F,
                 {srcPtr, destPtr, srcDims, destDims, kernel, stride, pad});
      break;
    }
  }

  case Kinded::Kind::PoolAvgGradInstKind: {
    PoolAvgGradInst *PAG = cast<PoolAvgGradInst>(I);
    auto *srcGrad = PAG->getSrcGrad();
    auto *srcGradPtr = emitValueAddress(builder, srcGrad);
    auto *destGradPtr = emitValueAddress(builder, PAG->getDestGrad());

    auto *srcGradDims = emitValueDims(builder, srcGrad);
    auto *destDims = emitValueDims(builder, PAG->getDest());

    auto *kernel = emitConstSizeT(builder, PAG->getKernel());
    auto *stride = emitConstSizeT(builder, PAG->getStride());
    auto *pad = emitConstSizeT(builder, PAG->getPad());

    auto *F = getFunction("pool_avg_grad", srcGrad->getElementType());
    createCall(
        builder, F,
        {srcGradPtr, destGradPtr, srcGradDims, destDims, kernel, stride, pad});
    break;
  }

  case Kinded::Kind::QuantizeInstKind: {
    QuantizeInst *QI = cast<QuantizeInst>(I);
    auto *dest = QI->getDest();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, QI->getSrc());

    auto *destType = dest->getType();
    auto *numElem = emitConstSizeT(builder, destType->size());
    auto *scale = emitConstF32(builder, destType->getScale());
    auto *offset = emitConstI32(builder, destType->getOffset());

    auto *F = getFunction("quantize", dest->getElementType());
    createCall(builder, F, {destPtr, srcPtr, numElem, scale, offset});
    break;
  }

  case Kinded::Kind::DequantizeInstKind: {
    DequantizeInst *DQI = cast<DequantizeInst>(I);
    auto *dest = DQI->getDest();
    auto *src = DQI->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *srcType = src->getType();
    auto *numElem = emitConstSizeT(builder, dest->size());
    auto *scale = emitConstF32(builder, srcType->getScale());
    auto *offset = emitConstI32(builder, srcType->getOffset());

    auto *F = getFunction("dequantize", dest->getElementType());
    createCall(builder, F, {destPtr, srcPtr, numElem, scale, offset});
    break;
  }

  case Kinded::Kind::RescaleQuantizedInstKind: {
    RescaleQuantizedInst *RQI = cast<RescaleQuantizedInst>(I);
    auto *dest = RQI->getDest();
    auto *src = RQI->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destType = dest->getType();
    auto *srcType = src->getType();
    auto *numElem = emitConstSizeT(builder, destType->size());

    auto rescaleParams = quantization::quantizeScaleOffset32To8(
        srcType->getScale() / destType->getScale(), srcType->getOffset());

    auto *destOffset = emitConstI32(builder, destType->getOffset());
    auto *srcOffset = emitConstI32(builder, srcType->getOffset());
    auto *preShift = emitConstI32(builder, rescaleParams.pre_);
    auto *postShift = emitConstI32(builder, rescaleParams.post_);
    auto *scale = emitConstI32(builder, rescaleParams.scale_);

    auto *F = getFunction("rescale", dest->getElementType());
    createCall(builder, F,
               {destPtr, srcPtr, numElem, destOffset, srcOffset, preShift,
                postShift, scale});
    break;
  }

  case Kinded::Kind::SoftMaxInstKind: {
    SoftMaxInst *SM = cast<SoftMaxInst>(I);
    auto *dest = SM->getDest();
    auto *src = SM->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *F = getFunction("softmax", dest->getElementType());
    createCall(builder, F, {srcPtr, destPtr, srcDims, destDims});
    break;
  }

  case Kinded::Kind::SoftMaxGradInstKind: {
    SoftMaxGradInst *SMG = cast<SoftMaxGradInst>(I);
    auto *srcGrad = SMG->getSrcGrad();
    auto *selected = SMG->getSelected();
    auto *srcGradPtr = emitValueAddress(builder, srcGrad);
    auto *destPtr = emitValueAddress(builder, SMG->getOrigDest());
    auto *selectedPtr = emitValueAddress(builder, selected);

    auto *srcGradDims = emitValueDims(builder, srcGrad);
    auto *selectedDims = emitValueDims(builder, selected);

    auto *F = getFunction("softmax_grad", srcGrad->getElementType());
    createCall(builder, F,
               {srcGradPtr, destPtr, selectedPtr, srcGradDims, selectedDims});
    break;
  }

  case Kinded::Kind::TopKInstKind: {
    TopKInst *TI = cast<TopKInst>(I);
    auto *input = TI->getInput();
    auto *valuesPtr = emitValueAddress(builder, TI->getValues());
    auto *indicesPtr = emitValueAddress(builder, TI->getIndices());
    auto *inputPtr = emitValueAddress(builder, input);
    auto *scratchPtr = emitValueAddress(builder, TI->getScratch());

    auto *k = emitConstSizeT(builder, TI->getK());
    auto *n = emitConstSizeT(builder, input->dims().back());
    auto *size = emitConstSizeT(builder, input->size());

    auto *F = getFunction("topk", input->getElementType());
    createCall(builder, F,
               {valuesPtr, indicesPtr, inputPtr, scratchPtr, k, n, size});
    break;
  }

  case Kinded::Kind::TransposeInstKind: {
    TransposeInst *TI = cast<TransposeInst>(I);
    auto *dest = TI->getDest();
    auto *src = TI->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    // Convert the mask to size_t type.
    ShapeVector shuffSizeT;
    for (auto D : TI->getShuffle()) {
      shuffSizeT.push_back((size_t)D);
    }

    auto *shuffle = emitConstArray(builder, shuffSizeT);
    auto *len = emitConstSizeT(builder, TI->getShuffle().size());

    auto *F = getFunction("transpose", dest->getElementType());
    createCall(builder, F, {srcPtr, destPtr, srcDims, destDims, shuffle, len});
    break;
  }

    // Alloc and Dealloc instructions are handled by the memory allocator.
  case Kinded::Kind::AllocActivationInstKind:
  case Kinded::Kind::DeallocActivationInstKind:
  case Kinded::Kind::TensorViewInstKind:
    break;

  case Kinded::Kind::InsertTensorInstKind: {
    InsertTensorInst *ITI = llvm::cast<InsertTensorInst>(I);
    auto *dest = ITI->getDest();
    auto *src = ITI->getSrc();
    auto offsets = ITI->getOffsets();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *destDimsSize =
        emitConstSizeT(builder, dest->getType()->dims().size());
    auto *srcDimsSize = emitConstSizeT(builder, src->getType()->dims().size());
    auto *offsetsPtr = emitConstArray(builder, offsets);
    auto *offsetsArraySize = emitConstSizeT(builder, offsets.size());
    auto *count = emitConstSizeT(builder, ITI->getCount());
    auto *axis = emitConstSizeT(builder, ITI->getAxis());

    auto *F = getFunction("insert_tensor", dest->getElementType());
    createCall(builder, F,
               {destPtr, srcPtr, offsetsPtr, destDims, srcDims, destDimsSize,
                srcDimsSize, offsetsArraySize, count, axis});
    break;
  }

  case Kinded::Kind::ExtractTensorInstKind: {
    ExtractTensorInst *ITI = llvm::cast<ExtractTensorInst>(I);
    auto *dest = ITI->getDest();
    auto *src = ITI->getSrc();
    auto offsets = ITI->getOffsets();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *destDimsSize =
        emitConstSizeT(builder, dest->getType()->dims().size());
    auto *srcDimsSize = emitConstSizeT(builder, src->getType()->dims().size());
    auto *offsetsPtr = emitConstArray(builder, offsets);
    auto *offsetsArraySize = emitConstSizeT(builder, offsets.size());

    auto *F = getFunction("extract_tensor", dest->getElementType());
    createCall(builder, F,
               {srcPtr, destPtr, offsetsPtr, srcDims, destDims, srcDimsSize,
                destDimsSize, offsetsArraySize});
    break;
  }

  case Kinded::Kind::GatherInstKind: {
    GatherInst *GI = llvm::cast<GatherInst>(I);
    auto *dest = GI->getDest();
    auto *data = GI->getData();
    auto *indices = GI->getIndices();

    auto *destPtr = emitValueAddress(builder, dest);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *indicesPtr = emitValueAddress(builder, indices);

    auto *indicesSize = emitConstSizeT(builder, indices->size());

    auto *dataType = data->getType();
    auto *sliceSize =
        emitConstSizeT(builder, dataType->size() / dataType->dims()[0]);

    auto *F = getFunction("gather", dest->getElementType());
    createCall(builder, F,
               {destPtr, dataPtr, indicesPtr, indicesSize, sliceSize});
    break;
  }

  case Kinded::Kind::DebugPrintInstKind: {
    DebugPrintInst *DPI = llvm::cast<DebugPrintInst>(I);
    auto *src = DPI->getSrc();
    auto *srcPtr = emitValueAddress(builder, src);
    srcPtr = builder.CreateBitCast(srcPtr, builder.getInt8PtrTy());
    auto *srcDims = emitValueDims(builder, src);
    auto *srcDimsSize = emitConstSizeT(builder, src->getType()->dims().size());
    auto *srcElemKind =
        emitConstSizeT(builder, static_cast<size_t>(src->getElementType()));
    auto *name = emitStringConst(builder, I->getName());

    auto *F = getFunction("dump_tensor");
    createCall(builder, F, {srcPtr, srcDims, srcDimsSize, srcElemKind, name});
    break;
  }

  default:
#ifndef NDEBUG
    llvm::errs() << "Cannot select the instruction:\n";
    I->dump(llvm::errs());
    llvm::errs() << "\n";
#endif
    GLOW_UNREACHABLE("ERROR: Cannot select the instruction.");
  }
}
