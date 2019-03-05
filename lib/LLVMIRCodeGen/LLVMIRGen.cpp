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

#include "glow/LLVMIRCodeGen/LLVMIRGen.h"

#include "AllocationsInfo.h"
#include "CommandLine.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"

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

static llvm::cl::opt<bool>
    dumpIR("dump-llvm-ir",
           llvm::cl::desc("Dump the LLVM-IR of the jitted code"),
           llvm::cl::init(false), llvm::cl::cat(getLLVMBackendCat()));

static llvm::cl::opt<bool>
    dumpJitAsm("dump-llvm-asm",
               llvm::cl::desc("Dump the textual assembly of the jitted code"),
               llvm::cl::init(false), llvm::cl::cat(getLLVMBackendCat()));

llvm::cl::opt<bool>
    emitDebugInfo("g", llvm::cl::desc("Emit debug information for debuggers"),
                  llvm::cl::init(false), llvm::cl::cat(getLLVMBackendCat()));

llvm::cl::opt<llvm::Reloc::Model> relocModel(
    "relocation-model",
    llvm::cl::desc(
        "Specify which relocation model to use on the target machine"),
    llvm::cl::values(
        clEnumValN(llvm::Reloc::Static, "static", "Non-relocatable code"),
        clEnumValN(llvm::Reloc::PIC_, "pic", "Position independent code")),
    llvm::cl::init(llvm::Reloc::Static), llvm::cl::cat(getLLVMBackendCat()));

/// Limitation of number of arguments for `emitDataParallelKernel`.
constexpr static size_t kArgLimit = 64;

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

LLVMIRGen::LLVMIRGen(const IRFunction *F, AllocationsInfo &allocationsInfo,
                     std::string mainEntryName, llvm::StringRef libjitBC)
    : F_(F), allocationsInfo_(allocationsInfo), mainEntryName_(mainEntryName),
      libjitBC_(libjitBC) {}

void LLVMIRGen::initTargetMachine(llvm::StringRef T,
                                  llvm::CodeModel::Model codeModel) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  if (T.empty())
    TM_.reset(llvm::EngineBuilder()
                  .setCodeModel(codeModel)
                  .setRelocationModel(relocModel)
                  .selectTarget(llvm::Triple(), "", getHostCpuName(),
                                getMachineAttributes()));
  else
    TM_.reset(llvm::EngineBuilder()
                  .setCodeModel(codeModel)
                  .setRelocationModel(relocModel)
                  .selectTarget(llvm::Triple(T), "", "",
                                llvm::SmallVector<std::string, 0>()));
}

std::string LLVMIRGen::getMainEntryName() const {
  return mainEntryName_.empty() ? "main" : mainEntryName_;
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
static std::unique_ptr<llvm::Module>
loadStandardLibrary(llvm::LLVMContext *ctx, llvm::StringRef filename,
                    llvm::StringRef libjitBC) {
  using llvm::sys::path::append;
  using llvm::sys::path::parent_path;

  llvm::SMDiagnostic error;

  // Parse the compiled-in image of libjit and return the resulting Module.
  return llvm::parseIR(
      llvm::MemoryBufferRef(
          llvm::StringRef(reinterpret_cast<const char *>(libjitBC.data()),
                          libjitBC.size()),
          "libjit.bc"),
      error, *ctx);
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
  llmodule_ = loadStandardLibrary(&ctx_, "libjit.bc", libjitBC_);
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
llvm::Type *LLVMIRGen::getElementType(llvm::IRBuilder<> &builder,
                                      const Value *val) {
  switch (val->getElementType()) {
  case ElemKind::Int64ITy:
    return builder.getInt64Ty();
  case ElemKind::FloatTy:
    return builder.getFloatTy();
  case ElemKind::Float16Ty:
    llvm_unreachable("Not yet implemented");
  case ElemKind::Int8QTy:
    return builder.getInt8Ty();
  case ElemKind::Int16QTy:
    return builder.getInt16Ty();
  case ElemKind::Int32QTy:
    return builder.getInt32Ty();
  case ElemKind::Int32ITy:
    return builder.getInt32Ty();
  case ElemKind::UInt8FusedQTy:
    return builder.getInt8Ty();
  case ElemKind::BoolTy:
    static_assert(sizeof(bool) == sizeof(int8_t),
                  "Bool is expected to be the same size as int8.");
    return builder.getInt8Ty();
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
#if FACEBOOK_INTERNAL && LLVM_VERSION_PATCH < 20181009
    getTargetMachine().addPassesToEmitFile(
        PM, asmStream, llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
#elif LLVM_VERSION_MAJOR > 6
    getTargetMachine().addPassesToEmitFile(
        PM, asmStream, nullptr,
        llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
#else
    getTargetMachine().addPassesToEmitFile(
        PM, asmStream, llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
#endif

    PM.run(*llmodule_);
    llvm::outs() << asmStream.str();
  }
}

llvm::Value *LLVMIRGen::emitValueAddress(llvm::IRBuilder<> &builder,
                                         const glow::Value *val) {
  assert(allocationsInfo_.allocatedAddress_.count(val) &&
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
  case ElemKind::Int32QTy:
    T = llvm::Type::getInt32PtrTy(ctx_);
    break;
  case ElemKind::Int64ITy:
    T = llvm::Type::getInt64PtrTy(ctx_);
    break;
  case ElemKind::Int32ITy:
    T = llvm::Type::getInt32PtrTy(ctx_);
    break;
  case ElemKind::UInt8FusedQTy:
    T = llvm::Type::getInt8PtrTy(ctx_);
    break;
  case ElemKind::BoolTy:
    T = llvm::Type::getInt8PtrTy(ctx_);
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
        sizeTType, allocationsInfo.allocatedAddress_.lookup(V));
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

template <typename T>
llvm::Value *LLVMIRGen::emitConstSizeTArray(llvm::IRBuilder<> &builder,
                                            llvm::ArrayRef<T> vals) {
  assert(std::is_integral<T>() && "Can only convert integral type to size_t.");
  auto SizeTType = builder.getIntNTy(sizeof(size_t) * 8);
  std::vector<llvm::Constant *> elems;
  for (auto I : vals) {
    assert(I >= 0 && "Only allow casting positive values into size_t.");
    assert(I <= std::numeric_limits<size_t>::max() &&
           "Do not allow overflow of size_t.");
    elems.push_back(llvm::ConstantInt::get(SizeTType, (size_t)I));
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
                                      const glow::Value *val) {
  auto dims = val->dims();
  return emitConstSizeTArray(builder, dims);
}

llvm::Value *LLVMIRGen::emitValueSize(llvm::IRBuilder<> &builder,
                                      const glow::Value *val) {
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
  case ElemKind::Float16Ty:
    llvm_unreachable("No yet implemented");
  case ElemKind::Int64ITy:
    return builder.getInt64(static_cast<int64_t>(val));
  case ElemKind::Int8QTy:
    return builder.getInt8(static_cast<int8_t>(val));
  case ElemKind::Int16QTy:
    return builder.getInt16(static_cast<int16_t>(val));
  case ElemKind::Int32QTy:
    return builder.getInt32(static_cast<int32_t>(val));
  case ElemKind::Int32ITy:
    return builder.getInt32(static_cast<int32_t>(val));
  case ElemKind::UInt8FusedQTy:
    return builder.getInt8(static_cast<int8_t>(val));
  case ElemKind::BoolTy:
    return builder.getInt8(static_cast<int8_t>(val));
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

void LLVMIRGen::markArgAsUnspecialized(llvm::Value *val) {
  dontSpecializeArgsSet_.insert(val);
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
  case ElemKind::Int32ITy:
    return get("libjit_" + name + "_i32");
  case ElemKind::Int64ITy:
    return get("libjit_" + name + "_u");
  case ElemKind::BoolTy:
    return get("libjit_" + name + "_b");
  default:
    GLOW_UNREACHABLE("Unsupported element type");
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
llvm::Value *
LLVMIRGen::emitBufferAddress(llvm::IRBuilder<> &builder, Value *val,
                             llvm::Function *kernel,
                             llvm::DenseMap<Value *, int> &bufferToArgNum) {
  assert(bufferToArgNum.count(val) && "Buffer should be in the map");
  return kernel->args().begin() + bufferToArgNum[val];
}

/// Implementation of emitDataParallelKernel where we guarantee that the number
/// of arguments will be bound by 64.
void LLVMIRGen::emitDataParallelKernelImpl(
    llvm::IRBuilder<> &builder, llvm::ArrayRef<const Instruction *> bundle,
    llvm::ArrayRef<llvm::Type *> argTypes,
    llvm::DenseMap<Value *, int> &bufferToArgNum,
    llvm::ArrayRef<llvm::Value *> buffers) {
  if (bundle.empty()) {
    return;
  }
  // Create stacked kernel function type.
  llvm::Type *voidTy = llvm::Type::getVoidTy(ctx_);
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
}

/// Emit the function that implements a data-parallel kernel and calls it.
///
/// The generated kernel functions get buffers as their parameters. The buffers
/// are uniqued, so that any buffer is passed as argument to the kernel function
/// only once. This allows us to mark all parameters of the generated kernel as
/// noalias. As a result, the LLVM optimizer makes use of the noalias attributes
/// and produces nicely vectorized code for the generated data-parallel kernels.
/// Note that we will emit a kernel whenever the number of arguments (aka unique
/// buffers) exceeds `kArgLimit`.
void LLVMIRGen::emitDataParallelKernel(
    llvm::IRBuilder<> &builder, llvm::ArrayRef<const Instruction *> bundle) {
  if (bundle.empty())
    return;
  // Types of arguments for the kernel function being generated.
  llvm::SmallVector<llvm::Type *, 32> argTypes;
  // Map each buffer used by the kernel to the argument number of the kernel
  // function. This ensures that same buffer is always mapped to the same
  // argument.
  llvm::DenseMap<Value *, int> bufferToArgNum;
  // Buffers to be passed to the kernel function as arguments.
  llvm::SmallVector<llvm::Value *, 32> buffers;
  // Hold a group of instructions whose unique buffer size is no more than
  // `kArgLimit` and ship it for processing
  llvm::SmallVector<const Instruction *, 32> batchedBundle;
  // Collect unique buffers up to `kArgLimit` used by the instructions of the
  // kernel.
  for (const auto I : bundle) {
    // If adding the buffers of current instruction might make the total number
    // of unique buffer exceed `kArgLimit`, we need to emit the kernel and start
    // over. Note the "might" as this method is pessimistic, because number of
    // buffers from current instructure might not be unique. Trade-off here is
    // that the algorithm is cleaner and in practice, if we over-estimate the
    // argument size by several, it does not matter too much.
    if (argTypes.size() + I->getOperands().size() > kArgLimit) {
      emitDataParallelKernelImpl(builder, batchedBundle, argTypes,
                                 bufferToArgNum, buffers);
      batchedBundle.clear();
      argTypes.clear();
      bufferToArgNum.clear();
      buffers.clear();
    }

    // Add the instruction to the current bundle and process its operands
    batchedBundle.push_back(I);
    for (const auto &Op : I->getOperands()) {
      auto *buf = Op.first;
      if (!bufferToArgNum.count(buf)) {
        bufferToArgNum[buf] = argTypes.size();
        buffers.push_back(emitValueAddress(builder, buf));
        argTypes.push_back(getElementType(builder, buf)->getPointerTo());
      }
    }
  }
  emitDataParallelKernelImpl(builder, batchedBundle, argTypes, bufferToArgNum,
                             buffers);
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
    llvm::SmallVectorImpl<const Instruction *> &bundle, Value *buf) {
  auto addr1 = allocationsInfo.allocatedAddress_[buf];
  auto size1 = buf->getSizeInBytes();
  for (auto bi : bundle) {
    for (auto bop : bi->getOperands()) {
      auto buf2 = bop.first;
      auto addr2 = allocationsInfo.allocatedAddress_[buf2];
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
  llvm::SmallVector<const Instruction *, 32> bundle;
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
    llvm::IRBuilder<> &builder, const glow::Instruction *I,
    llvm::Function *kernel, llvm::DenseMap<Value *, int> &bufferToArgNum,
    llvm::Value *loopCount) {
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
    auto *ES = cast<ElementSelectInst>(I);
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
    auto *F = getFunction("elementselect_kernel", lhs->getElementType());

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

      auto *lhsPre = emitConstI32(builder, lhsScaleParams.pre);
      auto *lhsPost = emitConstI32(builder, lhsScaleParams.post);
      auto *lhsScale = emitConstI32(builder, lhsScaleParams.scale);
      auto *rhsPre = emitConstI32(builder, rhsScaleParams.pre);
      auto *rhsPost = emitConstI32(builder, rhsScaleParams.post);
      auto *rhsScale = emitConstI32(builder, rhsScaleParams.scale);

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

  case Kinded::Kind::ElementIsNaNInstKind: {
    auto *AN = cast<ElementIsNaNInst>(I);
    auto *src = AN->getSrc();
    auto *dest = AN->getDest();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *F = getFunction("element_is_nan_kernel", src->getElementType());
    auto *stackedOpCall = createCall(builder, F, {loopCount, srcPtr});
    auto *elementTy = getElementType(builder, dest);
    auto *destAddr =
        builder.CreateGEP(elementTy, destPtr, loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::QuantizeInstKind: {
    auto *QI = cast<QuantizeInst>(I);
    auto *src = QI->getSrc();
    auto *dest = QI->getDest();
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *destTy = dest->getType();
    auto *destScale = emitConstF32(builder, destTy->getScale());
    auto *destOffset = emitConstI32(builder, destTy->getOffset());
    auto *F = getFunction("element_quantize_kernel", dest->getElementType());

    auto *stackedOpCall =
        createCall(builder, F, {loopCount, srcPtr, destScale, destOffset});
    llvm::Value *destAddr = nullptr;
    if (dest->getElementType() == ElemKind::Int8QTy) {
      destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr, loopCount,
                                   "buffer.element.addr");
    } else if (dest->getElementType() == ElemKind::Int32QTy) {
      destAddr = builder.CreateGEP(builder.getInt32Ty(), destPtr, loopCount,
                                   "buffer.element.addr");
    } else {
      GLOW_UNREACHABLE("Type is not supported.");
    }

    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::DequantizeInstKind: {
    auto *DI = cast<DequantizeInst>(I);
    auto *src = DI->getSrc();
    auto *dest = DI->getDest();
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *srcTy = src->getType();
    auto *srcScale = emitConstF32(builder, srcTy->getScale());
    auto *srcOffset = emitConstI32(builder, srcTy->getOffset());
    auto *F = getFunction("element_dequantize_kernel", dest->getElementType());

    auto *stackedOpCall =
        createCall(builder, F, {loopCount, srcPtr, srcScale, srcOffset});
    auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr, loopCount,
                                       "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::RescaleQuantizedInstKind: {
    auto *RQI = cast<RescaleQuantizedInst>(I);
    auto *dest = RQI->getDest();
    auto *src = RQI->getSrc();
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);

    auto *destType = dest->getType();
    auto *srcType = src->getType();

    auto rescaleParams = quantization::quantizeScaleOffset32To8(
        srcType->getScale() / destType->getScale(), srcType->getOffset());

    auto *destOffset = emitConstI32(builder, destType->getOffset());
    auto *srcOffset = emitConstI32(builder, srcType->getOffset());
    auto *preShift = emitConstI32(builder, rescaleParams.pre);
    auto *postShift = emitConstI32(builder, rescaleParams.post);
    auto *scale = emitConstI32(builder, rescaleParams.scale);
    auto *F = getFunction("element_rescale_kernel", dest->getElementType());

    auto *stackedOpCall = createCall(
        builder, F,
        {loopCount, srcPtr, destOffset, srcOffset, preShift, postShift, scale});
    auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr, loopCount,
                                       "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::CopyInstKind: {
    auto *CI = cast<CopyInst>(I);
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
      auto *lhsPre = emitConstI32(builder, lhsScaleParams.pre);                \
      auto *lhsPost = emitConstI32(builder, lhsScaleParams.post);              \
      auto *lhsScale = emitConstI32(builder, lhsScaleParams.scale);            \
      auto *rhsPre = emitConstI32(builder, rhsScaleParams.pre);                \
      auto *rhsPost = emitConstI32(builder, rhsScaleParams.post);              \
      auto *rhsScale = emitConstI32(builder, rhsScaleParams.scale);            \
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
    ARITHMETIC_BINARY_OP_CASE(ElementPow, "element_pow");
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
    auto *F = getFunction("element_cmp_lte_kernel", lhs->getElementType());

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
      auto *cmpPre = emitConstI32(builder, scaleParams.pre);
      auto *cmpPost = emitConstI32(builder, scaleParams.post);
      auto *cmpScale = emitConstI32(builder, scaleParams.scale);

      auto *stackedOpCall = createCall(builder, F,
                                       {loopCount, lhsPtr, rhsPtr, lhsOffset,
                                        rhsOffset, cmpPre, cmpPost, cmpScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *stackedOpCall = createCall(builder, F, {loopCount, lhsPtr, rhsPtr});
      auto *elementTy = getElementType(builder, dest);
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,
                                         "buffer.element.addr");
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
    auto *F = getFunction("element_cmp_eq_kernel", lhs->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *stackedOpCall = createCall(builder, F, {loopCount, lhsPtr, rhsPtr});
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
      auto *mulPre = emitConstI32(builder, scaleParams.pre);
      auto *mulPost = emitConstI32(builder, scaleParams.post);
      auto *mulScale = emitConstI32(builder, scaleParams.scale);

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
      auto *divPre = emitConstI32(builder, scaleParams.pre);
      auto *divPost = emitConstI32(builder, scaleParams.post);
      auto *divScale = emitConstI32(builder, scaleParams.scale);

      auto *stackedOpCall =
          createCall(builder, F,
                     {loopCount, lhsPtr, rhsPtr, destOffset, lhsOffset,
                      rhsOffset, divPre, divPost, divScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *elementTy = getElementType(builder, dest);
      auto *stackedOpCall =
          createCall(builder, F, {loopCount, lhsPtr, rhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,
                                         "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }
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

void LLVMIRGen::generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                       const glow::Instruction *I) {
  setCurrentDebugLocation(builder, I);
  assert(!I->isDataParallel() &&
         "data parallel instructions are not handled here");
  switch (I->getKind()) {
  case Kinded::Kind::MatMulInstKind: {
    auto *MM = cast<MatMulInst>(I);
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

      auto *outPre = emitConstI32(builder, outScaleParams.pre);
      auto *outPost = emitConstI32(builder, outScaleParams.post);
      auto *outScale = emitConstI32(builder, outScaleParams.scale);

      createCall(builder, F,
                 {destPtr, lhsPtr, rhsPtr, destDims, lhsDims, rhsDims,
                  destOffset, lhsOffset, rhsOffset, outPre, outPost, outScale});
    } else {
      createCall(builder, F,
                 {destPtr, lhsPtr, rhsPtr, destDims, lhsDims, rhsDims});
    }
    break;
  }

  case Kinded::Kind::QuantizationProfileInstKind: {
    auto *QP = cast<QuantizationProfileInst>(I);
    auto *hist = QP->getHistogram();
    auto *compInfo = QP->getComputationInfo();
    auto *inputTensor = QP->getInputTensor();

    auto *histPtr = emitValueAddress(builder, hist);
    auto *compInfoPtr = emitValueAddress(builder, compInfo);
    auto *inputTensorInfoPtr = emitValueAddress(builder, inputTensor);

    auto *histDims = emitValueDims(builder, hist);
    auto *inputTensorDims = emitValueDims(builder, inputTensor);

    assert(inputTensor->getElementType() == ElemKind::FloatTy &&
           "None float Tensor type for Quantization Profile Instruction.");
    auto *srcDimsSize =
        emitConstSizeT(builder, inputTensor->getType()->dims().size());

    auto *F = getFunction("quantization_profile");
    createCall(builder, F,
               {inputTensorInfoPtr, inputTensorDims, srcDimsSize, compInfoPtr,
                histPtr, histDims});
    break;
  }

  case Kinded::Kind::RowwiseQuantizedFullyConnectedInstKind: {
    auto *RWQFC = cast<RowwiseQuantizedFullyConnectedInst>(I);
    // Since we can't get the variable from a glow::Value directly,
    // we need to traverse the var list and find the one matching the given
    // Value.
    Tensor scalesT;
    auto *F_ = getIRFunction();
    for (auto &v : F_->getGraph()->getParent()->getConstants()) {
      assert(isa<WeightVar>(F_->getWeightForNode(v)));
      auto *w = cast<glow::Value>(F_->getWeightForNode(v));
      if (w == RWQFC->getScales()) {
        scalesT.assign(&v->getPayload());
        break;
      }
    }
    GLOW_ASSERT(scalesT.getUnsafePtr() != nullptr &&
                "Can't find the variable.");

    auto scalesH = scalesT.getHandle();
    size_t rowNum = scalesH.dims()[0];
    float inputScale = RWQFC->getSrc()->getType()->getScale();

    float bScale = RWQFC->getBias()->getType()->getScale();
    int32_t bOffset = RWQFC->getBias()->getType()->getOffset();

    float outputScale = RWQFC->getDest()->getType()->getScale();

    std::vector<llvm::Constant *> biasPreV(rowNum);
    std::vector<llvm::Constant *> biasPostV(rowNum);
    std::vector<llvm::Constant *> biasScaleV(rowNum);
    std::vector<llvm::Constant *> outputPreV(rowNum);
    std::vector<llvm::Constant *> outputPostV(rowNum);
    std::vector<llvm::Constant *> outputScaleV(rowNum);

    for (size_t i = 0; i < rowNum; i++) {
      // Calculate the scale of the values that come out of the matrix
      // multiplication part of the calculation.
      float matMulScale = inputScale * scalesH.raw(i);

      // Calculate the scaling parameters for the bias and output.
      auto biasScaleParam =
          quantization::quantizeScaleOffset32To8(bScale / matMulScale, bOffset);
      auto outScaleParam =
          quantization::quantizeScaleOffset32To8(matMulScale / outputScale, 0);

      // Pass the pre-shift, post-shift and integer scale parameters for the
      // bias and output calculation.
      biasPreV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                           biasScaleParam.pre, true);
      biasPostV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                            biasScaleParam.post, true);
      biasScaleV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                             biasScaleParam.scale, true);
      outputPreV[i] =
          llvm::ConstantInt::get(builder.getInt32Ty(), outScaleParam.pre, true);
      outputPostV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                              outScaleParam.post, true);
      outputScaleV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                               outScaleParam.scale, true);
    }

    auto *dest = RWQFC->getDest();
    auto *src = RWQFC->getSrc();
    auto *weights = RWQFC->getWeights();
    auto *bias = RWQFC->getBias();
    auto *weightsOffsets = RWQFC->getOffsets();

    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *weightsPtr = emitValueAddress(builder, weights);
    auto *biasPtr = emitValueAddress(builder, bias);
    auto *weightsOffsetsPtr = emitValueAddress(builder, weightsOffsets);
    auto *biasPrePtr = emitConstArray(builder, biasPreV, builder.getInt32Ty());
    auto *biasPostPtr =
        emitConstArray(builder, biasPostV, builder.getInt32Ty());
    auto *biasScalePtr =
        emitConstArray(builder, biasScaleV, builder.getInt32Ty());
    auto *outputPrePtr =
        emitConstArray(builder, outputPreV, builder.getInt32Ty());
    auto *outputPostPtr =
        emitConstArray(builder, outputPostV, builder.getInt32Ty());
    auto *outputScalePtr =
        emitConstArray(builder, outputScaleV, builder.getInt32Ty());

    auto *srcDims = emitValueDims(builder, src);
    auto *weightsDims = emitValueDims(builder, weights);
    auto *destDims = emitValueDims(builder, dest);
    auto *biasDims = emitValueDims(builder, bias);
    auto *row = emitConstSizeT(builder, weightsOffsets->dims()[0]);

    auto *destOffset = emitConstI32(builder, dest->getType()->getOffset());
    auto *srcOffset = emitConstI32(builder, src->getType()->getOffset());
    auto *biasOffset = emitConstI32(builder, bOffset);

    auto *F = getFunction("rowwise_quantized_fc", dest->getElementType());

    createCall(builder, F,
               {destPtr, srcPtr, weightsPtr, biasPtr, weightsOffsetsPtr,
                biasPrePtr, biasPostPtr, biasScalePtr, outputPrePtr,
                outputPostPtr, outputScalePtr, destDims, srcDims, weightsDims,
                biasDims, row, destOffset, srcOffset, biasOffset});
    break;
  }

  case Kinded::Kind::BatchedAddInstKind: {
    auto *BA = cast<BatchedAddInst>(I);
    auto *dest = BA->getDest();
    auto *batch = BA->getBatch();
    auto *slice = BA->getSlice();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *batchPtr = emitValueAddress(builder, batch);
    auto *slicePtr = emitValueAddress(builder, slice);

    auto bdim = flattenCdr(batch->dims());
    auto *numSlice = emitConstSizeT(builder, bdim.first);
    auto *sliceSize = emitConstSizeT(builder, bdim.second);

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

      auto *batchPre = emitConstI32(builder, batchScaleParams.pre);
      auto *batchPost = emitConstI32(builder, batchScaleParams.post);
      auto *batchScale = emitConstI32(builder, batchScaleParams.scale);
      auto *slicePre = emitConstI32(builder, sliceScaleParams.pre);
      auto *slicePost = emitConstI32(builder, sliceScaleParams.post);
      auto *sliceScale = emitConstI32(builder, sliceScaleParams.scale);

      llvm::Function *F = nullptr;
      if (sliceTy->getElementType() == ElemKind::Int8QTy) {
        F = getFunction("batchedadd", dest->getElementType());
      } else if (sliceTy->getElementType() == ElemKind::Int32QTy) {
        F = getFunction("batchedadd_i32", dest->getElementType());
      } else {
        GLOW_UNREACHABLE("Type is not supported.");
      }
      createCall(builder, F,
                 {destPtr, batchPtr, slicePtr, numSlice, sliceSize, destOffset,
                  batchOffset, sliceOffset, batchPre, batchPost, batchScale,
                  slicePre, slicePost, sliceScale});
    } else {
      auto *F = getFunction("batchedadd", dest->getElementType());
      createCall(builder, F,
                 {destPtr, batchPtr, slicePtr, numSlice, sliceSize});
    }
    break;
  }

  case Kinded::Kind::BatchedReduceAddInstKind: {
    auto *BR = cast<BatchedReduceAddInst>(I);
    auto *dest = BR->getDest();
    auto *batch = BR->getBatch();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *batchPtr = emitValueAddress(builder, batch);
    auto *axis = emitConstSizeT(builder, BR->getAxis());

    ShapeVector eBatchDims = expandDimsToMax(batch->dims());
    ShapeVector eDestDims = eBatchDims;
    eDestDims[BR->getAxis()] = 1;

    auto *batchDims =
        emitConstSizeTArray(builder, llvm::makeArrayRef(eBatchDims));
    auto *destDims =
        emitConstSizeTArray(builder, llvm::makeArrayRef(eDestDims));

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

      auto *batchPre = emitConstI32(builder, batchScaleParams.pre);
      auto *batchPost = emitConstI32(builder, batchScaleParams.post);
      auto *batchScale = emitConstI32(builder, batchScaleParams.scale);

      createCall(builder, F,
                 {destPtr, batchPtr, destDims, batchDims, destOffset,
                  batchOffset, batchPre, batchPost, batchScale, axis});
    } else {
      auto *destSize = emitConstSizeT(builder, dest->size());

      createCall(builder, F,
                 {destPtr, batchPtr, destSize, destDims, batchDims, axis});
    }
    break;
  }

  case Kinded::Kind::ConvolutionInstKind: {
    auto *CI = cast<ConvolutionInst>(I);
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

    auto *kernels = emitConstSizeTArray(builder, CI->getKernels());
    auto *strides = emitConstSizeTArray(builder, CI->getStrides());
    auto *pads = emitConstSizeTArray(builder, CI->getPads());
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
      auto *biasPre = emitConstI32(builder, biasScaleParam.pre);
      auto *biasPost = emitConstI32(builder, biasScaleParam.post);
      auto *biasScale = emitConstI32(builder, biasScaleParam.scale);
      auto *outPre = emitConstI32(builder, outScaleParam.pre);
      auto *outPost = emitConstI32(builder, outScaleParam.post);
      auto *outScale = emitConstI32(builder, outScaleParam.scale);

      createCall(builder, F,
                 {destPtr,    srcPtr,     filterPtr,  biasPtr,   destDims,
                  srcDims,    filterDims, biasDims,   kernels,   strides,
                  pads,       group,      destOffset, srcOffset, filterOffset,
                  biasOffset, biasPre,    biasPost,   biasScale, outPre,
                  outPost,    outScale,   unrollD});
    } else {
      createCall(builder, F,
                 {destPtr, srcPtr, filterPtr, biasPtr, destDims, srcDims,
                  filterDims, biasDims, kernels, strides, pads, group,
                  unrollD});
    }
    break;
  }

  case Kinded::Kind::ConvolutionGradInstKind: {
    auto *CG = cast<ConvolutionGradInst>(I);
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

    auto *kernels = emitConstSizeTArray(builder, CG->getKernels());
    auto *strides = emitConstSizeTArray(builder, CG->getStrides());
    auto *pads = emitConstSizeTArray(builder, CG->getPads());
    auto *group = emitConstSizeT(builder, CG->getGroup());

    auto *F = getFunction("convolution_grad", srcGrad->getElementType());
    createCall(builder, F,
               {srcGradPtr, destGradPtr, srcPtr, filterGradPtr, biasGradPtr,
                filterPtr, destGradDims, srcDims, filterGradDims, kernels,
                strides, pads, group});
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

  case Kinded::Kind::LengthsToRangesInstKind: {
    auto *LTR = cast<LengthsToRangesInst>(I);
    auto *dest = LTR->getDest();
    auto *lengths = LTR->getLengths();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *lengthsPtr = emitValueAddress(builder, lengths);
    auto *size = emitConstSizeT(builder, lengths->dims()[0]);
    auto *F = getFunction("lengths_to_ranges", dest->getElementType());
    createCall(builder, F, {destPtr, lengthsPtr, size});
    break;
  }

  case Kinded::Kind::LengthsSumInstKind: {
    auto *LS = cast<LengthsSumInst>(I);
    auto *dest = LS->getDest();
    auto *data = LS->getData();
    auto *lengths = LS->getLengths();

    auto *destPtr = emitValueAddress(builder, dest);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *lengthsPtr = emitValueAddress(builder, lengths);

    auto *lengthsSize = emitConstSizeT(builder, lengths->size());
    auto *dataType = data->getType();
    auto *destSize = emitConstSizeT(builder, dest->size());
    auto *sliceSize =
        emitConstSizeT(builder, dataType->size() / dataType->dims()[0]);

    auto *F = getFunction("lengths_sum", data->getElementType());
    createCall(
        builder, F,
        {destPtr, dataPtr, lengthsPtr, destSize, lengthsSize, sliceSize});
    break;
  }

  case Kinded::Kind::LocalResponseNormalizationInstKind: {
    auto *LRN = cast<LocalResponseNormalizationInst>(I);
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
    auto *LRNG = llvm::cast<LocalResponseNormalizationGradInst>(I);
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

  case Kinded::Kind::MaxPoolInstKind: {
    auto *PM = cast<MaxPoolInst>(I);
    auto *dest = PM->getDest();
    auto *src = PM->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *kernels = emitConstSizeTArray(builder, PM->getKernels());
    auto *strides = emitConstSizeTArray(builder, PM->getStrides());
    auto *pads = emitConstSizeTArray(builder, PM->getPads());

    auto *F = getFunction("max_pool", dest->getElementType());
    createCall(builder, F,
               {srcPtr, destPtr, srcDims, destDims, kernels, strides, pads});
    break;
  }

  case Kinded::Kind::MaxPoolWithXYInstKind: {
    auto *PMXY = cast<MaxPoolWithXYInst>(I);
    auto *dest = PMXY->getDest();
    auto *src = PMXY->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *srcXYPtr = emitValueAddress(builder, PMXY->getSrcXY());

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *kernels = emitConstSizeTArray(builder, PMXY->getKernels());
    auto *strides = emitConstSizeTArray(builder, PMXY->getStrides());
    auto *pads = emitConstSizeTArray(builder, PMXY->getPads());

    auto *F = getFunction("max_pool_xy", dest->getElementType());
    createCall(
        builder, F,
        {srcPtr, destPtr, srcXYPtr, srcDims, destDims, kernels, strides, pads});
    break;
  }

  case Kinded::Kind::MaxPoolWithXYGradInstKind: {
    auto *PMG = cast<MaxPoolWithXYGradInst>(I);
    auto *srcGrad = PMG->getSrcGrad();
    auto *srcGradPtr = emitValueAddress(builder, srcGrad);
    auto *destGradPtr = emitValueAddress(builder, PMG->getDestGrad());
    auto *srcXYPtr = emitValueAddress(builder, PMG->getSrcXY());

    auto *srcGradDims = emitValueDims(builder, srcGrad);
    auto *destDims = emitValueDims(builder, PMG->getDest());

    auto *F = getFunction("max_pool_xy_grad", srcGrad->getElementType());
    createCall(builder, F,
               {srcGradPtr, destGradPtr, srcXYPtr, srcGradDims, destDims});
    break;
  }

  case Kinded::Kind::AvgPoolInstKind: {
    auto *PA = cast<AvgPoolInst>(I);
    auto *dest = PA->getDest();
    auto *src = PA->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *kernels = emitConstSizeTArray(builder, PA->getKernels());
    auto *strides = emitConstSizeTArray(builder, PA->getStrides());
    auto *pads = emitConstSizeTArray(builder, PA->getPads());

    if (src->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *srcTy = src->getType();
      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *srcOffset = emitConstI32(builder, srcTy->getOffset());
      // Reduce resulting scale by a factor of PA->getKernels()[0] *
      // PA->getKernels()[1] since each subtensor value is divided by the area
      // of kernel.
      auto outScaleParam = quantization::quantizeScaleOffset32To8(
          srcTy->getScale() / destTy->getScale() /
              (PA->getKernels()[0] * PA->getKernels()[1]),
          destTy->getOffset());
      auto *outPre = emitConstI32(builder, outScaleParam.pre);
      auto *outPost = emitConstI32(builder, outScaleParam.post);
      auto *outScale = emitConstI32(builder, outScaleParam.scale);

      auto *F = getFunction("avg_pool", dest->getElementType());
      createCall(builder, F,
                 {srcPtr, destPtr, srcDims, destDims, kernels, strides, pads,
                  destOffset, srcOffset, outPre, outPost, outScale});
      break;
    } else {
      auto *F = getFunction("avg_pool", dest->getElementType());
      createCall(builder, F,
                 {srcPtr, destPtr, srcDims, destDims, kernels, strides, pads});
      break;
    }
  }

  case Kinded::Kind::AvgPoolGradInstKind: {
    auto *PAG = cast<AvgPoolGradInst>(I);
    auto *srcGrad = PAG->getSrcGrad();
    auto *srcGradPtr = emitValueAddress(builder, srcGrad);
    auto *destGradPtr = emitValueAddress(builder, PAG->getDestGrad());

    auto *srcGradDims = emitValueDims(builder, srcGrad);
    auto *destDims = emitValueDims(builder, PAG->getDest());

    auto *kernels = emitConstSizeTArray(builder, PAG->getKernels());
    auto *strides = emitConstSizeTArray(builder, PAG->getStrides());
    auto *pads = emitConstSizeTArray(builder, PAG->getPads());

    auto *F = getFunction("avg_pool_grad", srcGrad->getElementType());
    createCall(builder, F,
               {srcGradPtr, destGradPtr, srcGradDims, destDims, kernels,
                strides, pads});
    break;
  }

  case Kinded::Kind::SoftMaxInstKind: {
    auto *SM = cast<SoftMaxInst>(I);
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
    auto *SMG = cast<SoftMaxGradInst>(I);
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
    auto *TI = cast<TopKInst>(I);
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
    auto *TI = cast<TransposeInst>(I);
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

    auto *shuffle =
        emitConstSizeTArray(builder, llvm::makeArrayRef(shuffSizeT));
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
    auto *ITI = llvm::cast<InsertTensorInst>(I);
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
    auto *offsetsPtr = emitConstSizeTArray(builder, offsets);
    auto *offsetsArraySize = emitConstSizeT(builder, offsets.size());
    auto *count = emitConstSizeT(builder, ITI->getCount());
    auto *axis = emitConstSizeT(builder, ITI->getAxis());

    // Don't specialize the offsetPtr because we typically generate lots of
    // extracts from different offsets and specializing on this argument does
    // not speed things up.
    markArgAsUnspecialized(offsetsPtr);

    auto *F = getFunction("insert_tensor", dest->getElementType());
    createCall(builder, F,
               {destPtr, srcPtr, offsetsPtr, destDims, srcDims, destDimsSize,
                srcDimsSize, offsetsArraySize, count, axis});
    break;
  }

  case Kinded::Kind::ExtractTensorInstKind: {
    auto *ITI = llvm::cast<ExtractTensorInst>(I);
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
    auto *offsetsPtr = emitConstSizeTArray(builder, offsets);
    auto *offsetsArraySize = emitConstSizeT(builder, offsets.size());

    // Don't specialize the offsetPtr because we typically generate lots of
    // extracts from different offsets and specializing on this argument does
    // not speed things up.
    markArgAsUnspecialized(offsetsPtr);

    auto *F = getFunction("extract_tensor", dest->getElementType());
    createCall(builder, F,
               {srcPtr, destPtr, offsetsPtr, srcDims, destDims, srcDimsSize,
                destDimsSize, offsetsArraySize});
    break;
  }

  case Kinded::Kind::GatherInstKind: {
    auto *GI = llvm::cast<GatherInst>(I);
    auto *dest = GI->getDest();
    auto *data = GI->getData();
    auto *indices = GI->getIndices();
    unsigned batchDims = GI->getBatchDims();

    auto *destPtr = emitValueAddress(builder, dest);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *indicesPtr = emitValueAddress(builder, indices);

    auto *indicesSize = emitConstSizeT(builder, indices->size());

    auto *dataType = data->getType();

    // The size of the sample in the batch.
    size_t sampleSize = dataType->getSliceSize(batchDims);
    // The size of the slices that we gather.
    size_t sliceSize = dataType->getSliceSize(batchDims + 1);
    // The size of each sample in the batch.
    size_t numSamples = dataType->size() / sampleSize;

    auto *sliceSizeVal = emitConstSizeT(builder, sliceSize);
    auto *numSamplesVal = emitConstSizeT(builder, numSamples);
    auto *sampleSizeVal = emitConstSizeT(builder, sampleSize);

    // Dispatching function depeending on the input type of Indices.
    llvm::Function *F = nullptr;
    if (indices->getElementType() == ElemKind::Int64ITy) {
      F = getFunction("gather64", dest->getElementType());
    } else if (indices->getElementType() == ElemKind::Int32ITy) {
      F = getFunction("gather32", dest->getElementType());
    }
    if (!F) {
      llvm_unreachable("Cannot get function for Gather. "
                       "Indices input of Gather has to be int32 or int64");
    }
    createCall(builder, F,
               {destPtr, dataPtr, indicesPtr, indicesSize, sliceSizeVal,
                numSamplesVal, sampleSizeVal});
    break;
  }

  case Kinded::Kind::GatherRangesInstKind: {
    auto *GRI = llvm::cast<GatherRangesInst>(I);
    auto *output = GRI->getOutput();
    auto *lengths = GRI->getLengths();
    auto *data = GRI->getData();
    auto *ranges = GRI->getRanges();

    auto *outputPtr = emitValueAddress(builder, output);
    auto *lengthsPtr = emitValueAddress(builder, lengths);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *rangesPtr = emitValueAddress(builder, ranges);

    auto rangesType = ranges->getType();

    // The number of examples in ranges.
    size_t numExamples = rangesType->dims()[0];
    // The number of range pairs in each example.
    size_t exampleSize = rangesType->dims()[1];

    auto *numExamplesVal = emitConstSizeT(builder, numExamples);
    auto *exampleSizeVal = emitConstSizeT(builder, exampleSize);

    // Dispatching function depending on the input type of Ranges.
    llvm::Function *F = nullptr;
    if (ranges->getElementType() == ElemKind::Int64ITy) {
      F = getFunction("gatherranges64", output->getElementType());
    } else if (ranges->getElementType() == ElemKind::Int32ITy) {
      F = getFunction("gatherranges32", output->getElementType());
    }
    if (!F) {
      llvm_unreachable("Cannot get function for GatherRanges. "
                       "Ranges input of GatherRanges has to be int32 or int64");
    }
    createCall(builder, F,
               {outputPtr, lengthsPtr, dataPtr, rangesPtr, numExamplesVal,
                exampleSizeVal});
    break;
  }

  case Kinded::Kind::ScatterAssignInstKind: {
    auto *SAI = llvm::cast<ScatterAssignInst>(I);
    auto *data = SAI->getData();
    auto *indices = SAI->getIndices();
    auto *slices = SAI->getSlices();

    auto *dataPtr = emitValueAddress(builder, data);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *slicesPtr = emitValueAddress(builder, slices);

    auto *indicesSize = emitConstSizeT(builder, indices->size());

    auto *dataType = data->getType();
    auto *sliceSize =
        emitConstSizeT(builder, dataType->size() / dataType->dims()[0]);

    auto *F = getFunction("scatterassign", data->getElementType());
    createCall(builder, F,
               {dataPtr, indicesPtr, slicesPtr, indicesSize, sliceSize});
    break;
  }

  case Kinded::Kind::SparseLengthsWeightedSumInstKind: {
    auto *SI = cast<SparseLengthsWeightedSumInst>(I);
    auto *dest = SI->getDest();
    auto *data = SI->getData();
    auto *weights = SI->getWeights();
    auto *indices = SI->getIndices();
    auto *lengths = SI->getLengths();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *weightsPtr = emitValueAddress(builder, weights);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *lengthsPtr = emitValueAddress(builder, lengths);
    auto *segments = emitConstSizeT(builder, lengths->dims()[0]);
    auto *lineSize = emitConstSizeT(builder, data->size() / data->dims()[0]);
    auto *F =
        getFunction("sparse_lengths_weighted_sum", dest->getElementType());
    createCall(builder, F,
               {destPtr, dataPtr, weightsPtr, indicesPtr, lengthsPtr, segments,
                lineSize});
    break;
  }

  case Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumInstKind: {
    auto *N = cast<RowwiseQuantizedSparseLengthsWeightedSumInst>(I);
    auto *dest = N->getDest();
    auto *data = N->getData();
    auto *scales = N->getScales();
    auto *offsets = N->getOffsets();
    auto *weights = N->getWeights();
    auto *indices = N->getIndices();
    auto *lengths = N->getLengths();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *scalesPtr = emitValueAddress(builder, scales);
    auto *offsetsPtr = emitValueAddress(builder, offsets);
    auto *weightsPtr = emitValueAddress(builder, weights);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *lengthsPtr = emitValueAddress(builder, lengths);
    auto *segments = emitConstSizeT(builder, lengths->dims()[0]);
    auto *lineSize = emitConstSizeT(builder, data->size() / data->dims()[0]);
    auto *F = getFunction("rowwise_quantized_sparse_lengths_weighted_sum",
                          dest->getElementType());
    createCall(builder, F,
               {destPtr, dataPtr, scalesPtr, offsetsPtr, weightsPtr, indicesPtr,
                lengthsPtr, segments, lineSize});
    break;
  }

  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumInstKind: {
    auto *N = cast<FusedRowwiseQuantizedSparseLengthsWeightedSumInst>(I);
    auto *dest = N->getDest();
    auto *data = N->getData();
    auto *weights = N->getWeights();
    auto *indices = N->getIndices();
    auto *lengths = N->getLengths();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *weightsPtr = emitValueAddress(builder, weights);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *lengthsPtr = emitValueAddress(builder, lengths);
    auto *segments = emitConstSizeT(builder, lengths->dims()[0]);
    auto *inLineSize = emitConstSizeT(builder, data->size() / data->dims()[0]);
    auto *outLineSize = emitConstSizeT(builder, dest->size() / dest->dims()[0]);
    auto *F = getFunction("fused_rowwise_quantized_sparse_lengths_weighted_sum",
                          dest->getElementType());
    createCall(builder, F,
               {destPtr, dataPtr, weightsPtr, indicesPtr, lengthsPtr, segments,
                inLineSize, outLineSize});
    break;
  }

  case Kinded::Kind::SparseToDenseInstKind: {
    auto *STDI = llvm::cast<SparseToDenseInst>(I);
    auto *indices = STDI->getIndices();
    auto *values = STDI->getValues();
    auto *dest = STDI->getDest();

    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *valuesPtr = emitValueAddress(builder, values);
    auto *destPtr = emitValueAddress(builder, dest);

    auto *indicesSize = emitConstSizeT(builder, indices->size());
    auto *destSize = emitConstSizeT(builder, dest->size());

    auto *valuesType = values->getType();
    auto *valueSize =
        emitConstSizeT(builder, valuesType->size() / valuesType->dims()[0]);

    auto *F = getFunction("sparse_to_dense", dest->getElementType());
    createCall(
        builder, F,
        {destPtr, indicesPtr, valuesPtr, indicesSize, destSize, valueSize});
    break;
  }

  case Kinded::Kind::DebugPrintInstKind: {
    auto *DPI = llvm::cast<DebugPrintInst>(I);
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

  case Kinded::Kind::TraceEventInstKind: {
    auto *TEI = llvm::cast<TraceEventInst>(I);
    auto *data = TEI->getData();
    auto *offset = emitConstSizeT(builder, TEI->getIndex());
    auto *dataPtr = emitValueAddress(builder, data);
    auto *F = getFunction("write_timestamp");
    createCall(builder, F, {dataPtr, offset});
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
