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

#include "glow/LLVMIRCodeGen/LLVMIRGen.h"
#include "glow/Base/DimType.h"

#include "glow/LLVMIRCodeGen/AllocationsInfo.h"
#include "glow/LLVMIRCodeGen/CommandLine.h"
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

/// Limitation of number of arguments for `emitDataParallelKernel`.
constexpr static size_t kArgLimit = 64;

/// Query the TargetMachine to get the pointer size in bits
static unsigned getPointerNumBits(const llvm::TargetMachine &TM) {
  return TM.getPointerSize(0) * 8;
}

LLVMIRGen::LLVMIRGen(const IRFunction *F, AllocationsInfo &allocationsInfo,
                     std::string mainEntryName, llvm::StringRef libjitBC)
    : F_(F), allocationsInfo_(allocationsInfo), libjitBC_(libjitBC) {
  // Legalize main entry name.
  setMainEntryName(mainEntryName);
}

/// Mutex to protect LLVM's TargetRegistry.
static std::mutex initTargetMutex;

void LLVMIRGen::initTargetMachine(const LLVMBackendOptions &opts) {
  // LLVM's TargetRegistry is not thread safe so we add a critical section.
  std::lock_guard<std::mutex> g(initTargetMutex);

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  llvm::TargetOptions targetOpts;
  if (opts.getFloatABI().hasValue()) {
    targetOpts.FloatABIType = opts.getFloatABI().getValue();
  }
  if (!opts.getABIName().empty()) {
    targetOpts.MCOptions.ABIName = opts.getABIName();
  }
  if (opts.getTarget().empty()) {
    TM_.reset(llvm::EngineBuilder()
                  .setCodeModel(opts.getCodeModel())
                  .setRelocationModel(opts.getRelocModel())
                  .setTargetOptions(targetOpts)
                  .selectTarget(llvm::Triple(), opts.getArch(),
                                LLVMBackend::getHostCPU(),
                                LLVMBackend::getHostFeatures()));
  } else {
    TM_.reset(llvm::EngineBuilder()
                  .setCodeModel(opts.getCodeModel())
                  .setRelocationModel(opts.getRelocModel())
                  .setTargetOptions(targetOpts)
                  .selectTarget(llvm::Triple(opts.getTarget()), opts.getArch(),
                                opts.getCPU(), opts.getTargetFeatures()));
  }
  assert(TM_ && "Could not initialize the target machine");
}

llvm::StringRef LLVMIRGen::getBundleName() const { return bundleName_; }

void LLVMIRGen::setBundleName(const std::string &name) {
  bundleName_ = name.empty() ? "bundle" : legalizeName(name);
}

std::string LLVMIRGen::getMainEntryName() const { return mainEntryName_; }

void LLVMIRGen::setMainEntryName(std::string name) {
  mainEntryName_ = name.empty() ? "main" : legalizeName(name);
}

/// Load base addresses of different memory areas so that they can be easily
/// reused during codegen.
void LLVMIRGen::loadBaseAddresses(llvm::IRBuilder<> &builder) {
  auto *F = builder.GetInsertBlock()->getParent();

  // Load the base addresses at the beginning of the entry function once they
  // are set. They won't change after this point and all relative addressing
  // computations will simply use them.
  auto sizeTTy = builder.getIntNTy(getLibjitSizeTWidth());
  baseActivationsAddr_ = builder.CreatePtrToInt(F->args().begin() + 2, sizeTTy);
  baseConstantWeightVarsAddr_ =
      builder.CreatePtrToInt(F->args().begin(), sizeTTy);
  baseMutableWeightVarsAddr_ =
      builder.CreatePtrToInt(F->args().begin() + 1, sizeTTy);
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
  // checking for and reporting errors from parseIR.

  auto mod = llvm::parseIR(
      llvm::MemoryBufferRef(
          llvm::StringRef(reinterpret_cast<const char *>(libjitBC.data()),
                          libjitBC.size()),
          "libjit.bc"),
      error, *ctx);

  if (!mod) {
    error.print("LLVMIRGen", llvm::errs());
  }
  return mod;
}

/// Register a diagnostics handler that prevents the compiler from printing to
/// stdout.
static void registerEmptyDiagHandler(llvm::LLVMContext &ctx) {
  ctx.setDiagnosticHandlerCallBack(
      [](const llvm::DiagnosticInfo &DI, void *Context) {
        // Do not emit any warnings or diagnostics when JITting.
      });
}

void LLVMIRGen::initCodeGen() {
  // Load the jit library as a new module.
  llmodule_ = loadStandardLibrary(&getLLVMContext(), "libjit.bc", libjitBC_);
  CHECK(llmodule_.get()) << "Unable to load the JIT library.";

  // By default, LLVM would emit some diagnostics, remarks, etc. It is fine for
  // a static compiler, but not necessary for a JIT. Let's disable it by
  // providing a dummy diagnostics handler, that does not emit anything.
  // In particular, this allows us to get rid of the annoying "cannot vectorize"
  // warnings.
  registerEmptyDiagHandler(getLLVMContext());

  // Assign the target information to the module.
  llmodule_->setDataLayout(getTargetMachine().createDataLayout());

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
    llvm_unreachable("Not implemented");
  case ElemKind::BFloat16Ty:
    llvm_unreachable("Not implemented");
  case ElemKind::Int8QTy:
    return builder.getInt8Ty();
  case ElemKind::UInt8QTy:
    llvm_unreachable("Not implemented");
  case ElemKind::Int16QTy:
    return builder.getInt16Ty();
  case ElemKind::Int32QTy:
    return builder.getInt32Ty();
  case ElemKind::Int32ITy:
    return builder.getInt32Ty();
  case ElemKind::UInt8FusedQTy:
    return builder.getInt8Ty();
  case ElemKind::UInt8FusedFP16QTy:
    return builder.getInt8Ty();
  case ElemKind::UInt4FusedFP16QTy:
    return builder.getInt8Ty();
  case ElemKind::UInt4FusedQTy:
    return builder.getInt8Ty();
  case ElemKind::BoolTy:
    static_assert(sizeof(bool) == sizeof(int8_t),
                  "Bool is expected to be the same size as int8.");
    return builder.getInt8Ty();
  }
  return nullptr;
}

void LLVMIRGen::performCodeGen() {
  // Create the entry function into the LLVM module.
  auto int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
  auto dimTPtrTy = llvm::Type::getIntNPtrTy(getLLVMContext(), DIM_T_BITWIDTH);
  // The entry point has the following API:
  // int entry(uint8_t *baseConstantWeightVars,
  //           uint8_t *baseInoutWeightVars,
  //           uint8_t *baseActivations,
  //           dim_t *offsets);
  llvm::Type *retTy =
      llvm::Type::getIntNTy(getLLVMContext(), getLibjitIntWidth());
  llvm::FunctionType *jitFuncTy = llvm::FunctionType::get(
      retTy, {int8PtrTy, int8PtrTy, int8PtrTy, dimTPtrTy}, false);
  llvmF_ = llvm::Function::Create(jitFuncTy, llvm::Function::ExternalLinkage,
                                  "main", llmodule_.get());
  emittedLLVMFunctions_.emplace_back(llvmF_);

  // Setup the entry basic block and initialize the IR builder.
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(getLLVMContext(), "entry", llvmF_);
  builder_ = glow::make_unique<llvm::IRBuilder<>>(entry_bb);
  // Terminate the function with a return instruction.
  auto zero = builder_->getIntN(getLibjitIntWidth(), 0);
  auto *ret = builder_->CreateRet(zero);
  // Emit all the code before the retrun instruction.
  builder_->SetInsertPoint(ret);

  instrNumbering_.reset(new InstructionNumbering(*F_));
  generateFunctionDebugInfo();
  loadBaseAddresses(*builder_);
  generateLLVMIRForModule(*builder_);
}

void LLVMIRGen::finishCodeGen() {
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
  optimizeLLVMModule(&getModule(), getTargetMachine());

  // Generate debug information.
  generateModuleDebugInfo();

  if (dumpIR) {
    llvm::outs() << "LLVM module after optimizations:\n";
    llmodule_->print(llvm::outs(), nullptr);
  }

  if (dumpJitAsm) {
    llvm::SmallVector<char, 0> asmBuffer;
    llvm::raw_svector_ostream asmStream(asmBuffer);
    llvm::legacy::PassManager PM;
#if FACEBOOK_INTERNAL && LLVM_VERSION_MAJOR < 8
    getTargetMachine().addPassesToEmitFile(
        PM, asmStream, llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
#elif LLVM_VERSION_MAJOR < 10
    getTargetMachine().addPassesToEmitFile(
        PM, asmStream, nullptr,
        llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
#else
    getTargetMachine().addPassesToEmitFile(PM, asmStream, nullptr,
                                           llvm::CGFT_AssemblyFile);
#endif

    PM.run(*llmodule_);
    llvm::outs() << asmStream.str();
  }
}

llvm::Value *LLVMIRGen::emitValueAddress(llvm::IRBuilder<> &builder,
                                         const glow::Value *val) {
  assert(allocationsInfo_.allocatedAddress_.count(val) &&
         "Value address was not allocated");
  llvm::Type *T = nullptr;

  switch (val->getElementType()) {
  case ElemKind::FloatTy:
    T = llvm::Type::getFloatPtrTy(getLLVMContext());
    break;
  case ElemKind::Float16Ty:
    T = llvm::Type::getInt16PtrTy(getLLVMContext());
    break;
  case ElemKind::BFloat16Ty:
    T = llvm::Type::getInt16PtrTy(getLLVMContext());
    break;
  case ElemKind::Int8QTy:
    T = llvm::Type::getInt8PtrTy(getLLVMContext());
    break;
  case ElemKind::UInt8QTy:
    T = llvm::Type::getInt8PtrTy(getLLVMContext());
    break;
  case ElemKind::Int16QTy:
    T = llvm::Type::getInt16PtrTy(getLLVMContext());
    break;
  case ElemKind::Int32QTy:
    T = llvm::Type::getInt32PtrTy(getLLVMContext());
    break;
  case ElemKind::Int64ITy:
    T = llvm::Type::getInt64PtrTy(getLLVMContext());
    break;
  case ElemKind::Int32ITy:
    T = llvm::Type::getInt32PtrTy(getLLVMContext());
    break;
  case ElemKind::UInt8FusedQTy:
    T = llvm::Type::getInt8PtrTy(getLLVMContext());
    break;
  case ElemKind::UInt8FusedFP16QTy:
    T = llvm::Type::getInt8PtrTy(getLLVMContext());
    break;
  case ElemKind::UInt4FusedFP16QTy:
    T = llvm::Type::getInt8PtrTy(getLLVMContext());
    break;
  case ElemKind::UInt4FusedQTy:
    T = llvm::Type::getInt8PtrTy(getLLVMContext());
    break;
  case ElemKind::BoolTy:
    T = llvm::Type::getInt8PtrTy(getLLVMContext());
    break;
  default:
    LOG(FATAL) << "Unsupported element type: "
               << Type::getElementName(val->getElementType()).str();
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
  auto sizeTTy = builder.getIntNTy(getLibjitSizeTWidth());
  auto dimTTy = builder.getIntNTy(DIM_T_BITWIDTH);

  auto valueIdx = llvm::ConstantInt::get(dimTTy, kindAndValue.second);
  auto offsetAddr = builder.CreateGEP(dimTTy, offsetsArray_, valueIdx);
  auto offsetValue = builder.CreateLoad(dimTTy, offsetAddr);
  // Add offset to the base address.
  llvm::Value *addr = builder.CreateAdd(
      baseAddrValue, builder.CreateZExt(offsetValue, sizeTTy));
  return builder.CreateIntToPtr(addr, T);
}

llvm::Value *
LLVMIRGen::emitConstOffsetsArray(llvm::IRBuilder<> &builder,
                                 const AllocationsInfo &allocationsInfo) {
  constexpr const char *offsetsArrayName = "offsetsArray";
  auto dimTType = builder.getIntNTy(DIM_T_BITWIDTH);
  std::vector<llvm::Constant *> elems(allocationsInfo.valueNumbers_.size());
  dim_t maxOffset = 0;
  for (auto &I : allocationsInfo.valueNumbers_) {
    auto *V = I.first;
    auto offset = I.second.second;
    elems[offset] = llvm::ConstantInt::get(
        dimTType, allocationsInfo.allocatedAddress_.lookup(V));
    maxOffset = std::max(maxOffset, (dim_t)offset);
  }
  elems.resize(maxOffset + 1);
  auto *arr = llvm::ConstantArray::get(
      llvm::ArrayType::get(dimTType, elems.size()), elems);
  // Ensure that the same casted global variable is used for the equivalent
  // const arrays. This is important for the later function specialization pass.
  // LLVM does not do it automatically for this code pattern involving global
  // variables. It also reduces the number of variables.
  auto &constArrayVar = constArrayPtrs_[arr];
  auto oldG =
      getModule().getGlobalVariable(offsetsArrayName, /* allowInternal */ true);
  if (constArrayVar && constArrayVar->getType() == dimTType->getPointerTo()) {
    return constArrayVar;
  }
  if (oldG) {
    oldG->setName("offsetsArrayOld");
  }
  auto *M = builder.GetInsertBlock()->getModule();
  auto *G = new llvm::GlobalVariable(*M, arr->getType(), true,
                                     llvm::GlobalValue::InternalLinkage, arr,
                                     offsetsArrayName);
  constArrayVar = builder.CreateBitCast(G, dimTType->getPointerTo());
  if (oldG) {
    // Replace the old offsetsArray by the new one and remove the old.
    oldG->replaceAllUsesWith(G);
    oldG->eraseFromParent();
  }
  return constArrayVar;
}

llvm::Value *LLVMIRGen::emitConstI32Array(llvm::IRBuilder<> &builder,
                                          llvm::ArrayRef<int32_t> vals) {
  std::vector<llvm::Constant *> elems;
  for (auto I : vals) {
    elems.push_back(builder.getInt32(I));
  }
  return emitConstArray(builder, elems, builder.getInt32Ty());
}

llvm::Value *LLVMIRGen::emitConstFloatArray(llvm::IRBuilder<> &builder,
                                            llvm::ArrayRef<float> vals) {
  std::vector<llvm::Constant *> elems;
  for (auto I : vals) {
    elems.push_back(
        llvm::ConstantFP::get(llvm::Type::getFloatTy(ctx_), (float)I));
  }
  return emitConstArray(builder, elems, llvm::Type::getFloatTy(ctx_));
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

void LLVMIRGen::emitArrayStore(llvm::IRBuilder<> &builder,
                               llvm::ArrayRef<llvm::Value *> vals,
                               llvm::Value *basePtr, unsigned baseIdx) {
  for (size_t idx = 0, end = vals.size(); idx < end; ++idx) {
    assert(vals[idx]->getType()->getPointerTo() == basePtr->getType() &&
           "Mismatch between pointer and value type!");
    auto *storeIdx = builder.getInt32(idx + baseIdx);
    auto *storeAddr = builder.CreateGEP(basePtr, storeIdx);
    builder.CreateStore(vals[idx], storeAddr);
  }
}

llvm::Value *LLVMIRGen::emitValueDims(llvm::IRBuilder<> &builder,
                                      const glow::Value *val) {
  auto dims = val->dims();
  return emitConstDimTArray(builder, dims);
}

template <class InstructionTy>
llvm::Value *LLVMIRGen::emitConstFloatActivationArgs(llvm::IRBuilder<> &builder,
                                                     const InstructionTy *I) {
  return emitConstFloatArray(builder, I->getFusedActivationArgs());
}

template <class InstructionTy>
llvm::Value *LLVMIRGen::emitConstQuantActivationArgs(llvm::IRBuilder<> &builder,
                                                     const InstructionTy *I) {
  auto actArgsF = I->getFusedActivationArgs();
  std::vector<int32_t> actArgsQ;
  auto *destTy = I->getDest()->getType();
  switch (I->getFusedActivation()) {
  case FusedActivation::NONE:
  case FusedActivation::RELU:
    assert(actArgsF.size() == 0 && "Invalid number of activation parameters!");
    break;
  case FusedActivation::CLIP: {
    // For Clip we quantize min/max using the output quantization params.
    assert(actArgsF.size() == 2 &&
           "Invalid number of parameters for fused Clip activation!");
    float minF = actArgsF[0];
    float maxF = actArgsF[1];
    TensorQuantizationParams TQP{destTy->getScale(), destTy->getOffset()};
    int32_t minQ = quantization::quantize<int32_t>(minF, TQP);
    int32_t maxQ = quantization::quantize<int32_t>(maxF, TQP);
    actArgsQ.push_back(minQ);
    actArgsQ.push_back(maxQ);
    break;
  }
  case FusedActivation::SIGMOID:
    LOG(FATAL) << "Fused Sigmoid for quantized type not supported!";
    break;
  case FusedActivation::TANH:
    LOG(FATAL) << "Fused Tanh for quantized type not supported!";
    break;
  case FusedActivation::LEAKY_RELU: {
    // For LeakyRelu we transform the alpha parameter into pre/post/scale.
    assert(actArgsF.size() == 1 &&
           "Invalid number of parameters for fused LeakyRelu activation!");
    float alpha = actArgsF[0];
    auto alphaScaleParam = quantization::quantizeScaleOffset32To8(alpha, 0);
    actArgsQ.push_back(alphaScaleParam.pre);
    actArgsQ.push_back(alphaScaleParam.post);
    actArgsQ.push_back(alphaScaleParam.scale);
    break;
  }
  default:
    LOG(FATAL) << "Unsupported fused activation type!";
  }
  return emitConstI32Array(builder, actArgsQ);
}

llvm::Value *LLVMIRGen::emitValueSize(llvm::IRBuilder<> &builder,
                                      const glow::Value *val) {
  return builder.getIntN(DIM_T_BITWIDTH, val->size());
}

llvm::Value *LLVMIRGen::emitConstF32(llvm::IRBuilder<> &builder, float val) {
  return llvm::ConstantFP::get(llvm::Type::getFloatTy(getLLVMContext()), val);
}

llvm::Value *LLVMIRGen::emitConstI32(llvm::IRBuilder<> &builder, int32_t val) {
  return builder.getInt32(val);
}

llvm::Value *LLVMIRGen::emitConstI8(llvm::IRBuilder<> &builder, int8_t val) {
  return builder.getInt8(val);
}

llvm::Value *LLVMIRGen::emitConstI1(llvm::IRBuilder<> &builder, bool val) {
  return builder.getInt1(val);
}

llvm::Value *LLVMIRGen::emitConstSizeT(llvm::IRBuilder<> &builder, size_t val) {
  return builder.getIntN(getLibjitSizeTWidth(), val);
}

llvm::Value *LLVMIRGen::emitConstDimT(llvm::IRBuilder<> &builder, dim_t val) {
  return builder.getIntN(sizeof(dim_t) * 8, val);
}

llvm::Value *LLVMIRGen::emitConst(llvm::IRBuilder<> &builder, float val,
                                  glow::ElemKind kind) {
  switch (kind) {
  case ElemKind::FloatTy:
    return llvm::ConstantFP::get(llvm::Type::getFloatTy(getLLVMContext()), val);
  case ElemKind::Float16Ty:
    llvm_unreachable("Not implemented");
  case ElemKind::BFloat16Ty:
    llvm_unreachable("Not implemented");
  case ElemKind::Int64ITy:
    return builder.getInt64(static_cast<int64_t>(val));
  case ElemKind::Int8QTy:
    return builder.getInt8(static_cast<int8_t>(val));
  case ElemKind::UInt8QTy:
    llvm_unreachable("Not implemented");
  case ElemKind::Int16QTy:
    return builder.getInt16(static_cast<int16_t>(val));
  case ElemKind::Int32QTy:
    return builder.getInt32(static_cast<int32_t>(val));
  case ElemKind::Int32ITy:
    return builder.getInt32(static_cast<int32_t>(val));
  case ElemKind::UInt8FusedQTy:
    return builder.getInt8(static_cast<int8_t>(val));
  case ElemKind::UInt8FusedFP16QTy:
    return builder.getInt8(static_cast<int8_t>(val));
  case ElemKind::UInt4FusedFP16QTy:
    return builder.getInt8(static_cast<int8_t>(val));
  case ElemKind::UInt4FusedQTy:
    return builder.getInt8(static_cast<int8_t>(val));
  case ElemKind::BoolTy:
    return builder.getInt8(static_cast<int8_t>(val));
  }
  llvm_unreachable("Unknown element type");
}

llvm::Value *LLVMIRGen::emitStringConst(llvm::IRBuilder<> &builder,
                                        llvm::StringRef str) {
  llvm::Constant *constStrArray =
      llvm::ConstantDataArray::getString(getLLVMContext(), str, true);
  llvm::GlobalVariable *gvarStr = new llvm::GlobalVariable(
      *llmodule_, constStrArray->getType(), true,
      llvm::GlobalValue::PrivateLinkage, constStrArray, ".str");
  gvarStr->setAlignment(1);
  return builder.CreateBitCast(gvarStr, builder.getInt8PtrTy());
}

void LLVMIRGen::markArgAsUnspecialized(llvm::Value *val) {
  dontSpecializeArgsSet_.insert(val);
}

static std::string createName(const std::string &name, ElemKind elemTy) {
  switch (elemTy) {
  case ElemKind::FloatTy:
    return name + "_f";
  case ElemKind::Float16Ty:
    return name + "_fp16";
  case ElemKind::BFloat16Ty:
    return name + "_bfloat16";
  case ElemKind::Int8QTy:
    return name + "_i8";
  case ElemKind::Int16QTy:
    return name + "_i16";
  case ElemKind::Int32QTy:
    return name + "_i32";
  case ElemKind::Int32ITy:
    return name + "_i32";
  case ElemKind::Int64ITy:
    return name + "_u";
  case ElemKind::BoolTy:
    return name + "_b";
  default:
    LOG(FATAL) << "Unsupported element type: "
               << Type::getElementName(elemTy).str();
  }
}

llvm::Function *
LLVMIRGen::getFunction(const std::string &name,
                       llvm::ArrayRef<glow::ElemKind> elemTyArray) {
  auto strName = "libjit_" + name;

  for (auto elTy : elemTyArray) {
    strName = createName(strName, elTy);
  }
  auto *F = llmodule_->getFunction(strName);
  CHECK(F) << "Unable to load the function: " << strName.c_str();
  return F;
}

llvm::Function *LLVMIRGen::getFunction(const std::string &name) {
  return getFunction(name, llvm::ArrayRef<ElemKind>{});
}

llvm::Function *LLVMIRGen::getFunction(const std::string &name,
                                       ElemKind elemTy) {
  return getFunction(name, llvm::ArrayRef<ElemKind>{elemTy});
}

llvm::Function *LLVMIRGen::getLLVMFunction() { return llvmF_; }

llvm::CallInst *LLVMIRGen::createCall(llvm::IRBuilder<> &builder,
                                      llvm::Function *callee,
                                      llvm::ArrayRef<llvm::Value *> args,
                                      bool checked) {
#ifndef NDEBUG
  llvm::FunctionType *FTy = callee->getFunctionType();
  assert((args.size() == FTy->getNumParams() ||
          (FTy->isVarArg() && args.size() > FTy->getNumParams())) &&
         "Calling a function with bad signature: wrong number of arguments.");

  for (unsigned i = 0; i != args.size(); ++i) {
    assert((i >= FTy->getNumParams() ||
            FTy->getParamType(i) == args[i]->getType()) &&
           "Calling a function with a bad signature: argument type mismatch.");
  }
#endif
  if (!checked || !callee->getReturnType()->isIntegerTy()) {
    return builder.CreateCall(callee, args);
  }
  // Check if callee returned an error, i.e. non-zero result.
  // Emit a return with this error code in this case.
  auto *result = builder.CreateCall(callee, args);
  auto *zero = builder.getIntN(result->getType()->getIntegerBitWidth(), 0);
  auto *cond = builder.CreateICmpNE(result, zero);
  auto insertionPoint = builder.GetInsertPoint();
  auto *currentBB = result->getParent();
  auto *falseBB =
      currentBB->splitBasicBlock(builder.GetInsertPoint(), "cont_bb");
  auto *trueBB = llvm::BasicBlock::Create(getLLVMContext(), "error_bb",
                                          result->getFunction());
  builder.SetInsertPoint(currentBB->getTerminator());
  builder.CreateCondBr(cond, trueBB, falseBB);
  currentBB->getTerminator()->eraseFromParent();
  builder.SetInsertPoint(trueBB);
  auto *castedResult =
      builder.CreateBitCast(result, builder.getIntNTy(getLibjitIntWidth()));
  builder.CreateRet(castedResult);
  builder.SetInsertPoint(falseBB, insertionPoint);
  builder.SetInsertPoint(falseBB->getTerminator());
  return result;
}

llvm::CallInst *
LLVMIRGen::createCheckedCall(llvm::IRBuilder<> &builder, llvm::Function *callee,
                             llvm::ArrayRef<llvm::Value *> args) {
  return createCall(builder, callee, args, /* checked */ true);
}

llvm::CallInst *
LLVMIRGen::createUncheckedCall(llvm::IRBuilder<> &builder,
                               llvm::Function *callee,
                               llvm::ArrayRef<llvm::Value *> args) {
  return createCall(builder, callee, args, /* checked */ false);
}

std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
LLVMIRGen::createLoop(llvm::IRBuilder<> &builder, llvm::LLVMContext &ctx,
                      llvm::Value *numElements) const {
  auto dimTTy = builder.getIntNTy(DIM_T_BITWIDTH);
  auto *initVal = llvm::ConstantInt::get(dimTTy, 0);

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
  llvm::PHINode *var = builder.CreatePHI(dimTTy, 2);
  var->addIncoming(initVal, preheaderBB);

  // Emit the step value.
  auto *stepVal = llvm::ConstantInt::get(dimTTy, 1);
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
  llvm::Type *voidTy = llvm::Type::getVoidTy(getLLVMContext());
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
      llvm::BasicBlock::Create(getLLVMContext(), "entry", kernelFunc);
  llvm::IRBuilder<> kernelBuilder(entryBB);
  // Number of tensor elements.
  auto *numElements =
      emitValueSize(kernelBuilder, bundle[0]->getOperand(0).first);
  // Create a loop inside the stacked kernel function being generated.
  auto loopBBs = createLoop(kernelBuilder, getLLVMContext(), numElements);

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
    assert(canBePartOfDataParallelKernel(BI) &&
           "Data parallel operation is expected");
    generateLLVMIRForDataParallelInstr(kernelBuilder, BI, kernelFunc,
                                       bufferToArgNum, kernelLoopIdx);
  }
  kernelBuilder.SetInsertPoint(loopBBs.second);
  // Add a return.
  kernelBuilder.CreateRetVoid();

  setCurrentDebugLocation(builder, *bundle.begin());
  // Emit a call of the kernel.
  createUncheckedCall(builder, kernelFunc, buffers);
  // Emit debug info for the generated data-parallel kernel.
  generateFunctionDebugInfo(kernelFunc);
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
    // buffers from current instruction might not be unique. Trade-off here is
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
      if (addr1 == addr2 && size1 == size2) {
        // The two buffers are the exact same memory region. The operations
        // cannot be within the same bundle because the buffer pointers are
        // "noalias" qualified, so the kernel operations can be reordered by
        // LLVM's optimizations.
        // TODO investigate if removing "noalias" can be used to create bigger
        // and faster bundles.
        return true;
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
    if (!canBePartOfDataParallelKernel(&I)) {
      // Ignore memory management instructions as they are handled by the
      // MemoryManager and are NOPs for a JIT.
      if (isa<AllocActivationInst>(&I) || isa<DeallocActivationInst>(&I) ||
          isa<TensorViewInst>(&I)) {
        generateLLVMIRForInstr(builder, &I);
        continue;
      }
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
  assert(canBePartOfDataParallelKernel(I) &&
         "Instruction cannot be part of a data parallel kernel");
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
      auto *stackedOpCall = createUncheckedCall(                               \
          builder, F, {loopCount, val, pointerNull, pointerNull});             \
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,        \
                                         "buffer.element.addr");               \
      builder.CreateStore(stackedOpCall, destAddr);                            \
    } else {                                                                   \
      auto *val = emitConst(builder, value, dest->getElementType());           \
      auto *stackedOpCall = createUncheckedCall(                               \
          builder, F, {loopCount, val, pointerNull, pointerNull});             \
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,        \
                                         "buffer.element.addr");               \
      builder.CreateStore(stackedOpCall, destAddr);                            \
    }                                                                          \
    break;                                                                     \
  }
    ARITHMETIC_UNARY_OP_WITH_IMM_CASE(Splat, "splat", Value);
#undef ARITHMETIC_UNARY_OP_WITH_IMM_CASE

  case Kinded::Kind::TouchInstKind:
    // do nothing;
    break;

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

      auto *stackedOpCall = createUncheckedCall(
          builder, F,
          {loopCount, condPtr, lhsPtr, rhsPtr, destOffset, lhsOffset, rhsOffset,
           lhsPre, lhsPost, lhsScale, rhsPre, rhsPost, rhsScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *stackedOpCall =
          createUncheckedCall(builder, F, {loopCount, condPtr, lhsPtr, rhsPtr});
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
    auto *stackedOpCall = createUncheckedCall(                                 \
        builder, F, {loopCount, srcPtr, pointerNull, pointerNull});            \
    auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,          \
                                       loopCount, "buffer.element.addr");      \
    builder.CreateStore(stackedOpCall, destAddr);                              \
    break;                                                                     \
  }

    ARITHMETIC_UNARY_OP_CASE(Sigmoid, "sigmoid");
    ARITHMETIC_UNARY_OP_CASE(Tanh, "tanh");
    ARITHMETIC_UNARY_OP_CASE(ElementLog, "element_log");
    ARITHMETIC_UNARY_OP_CASE(ElementExp, "element_exp");
    ARITHMETIC_UNARY_OP_CASE(ElementAbs, "element_abs");
    ARITHMETIC_UNARY_OP_CASE(ElementNeg, "element_neg");
    ARITHMETIC_UNARY_OP_CASE(ElementFloor, "element_floor");
    ARITHMETIC_UNARY_OP_CASE(ElementCeil, "element_ceil");
    ARITHMETIC_UNARY_OP_CASE(ElementRound, "element_round");
    ARITHMETIC_UNARY_OP_CASE(ElementSqrt, "element_sqrt");
    ARITHMETIC_UNARY_OP_CASE(ElementRsqrt, "element_rsqrt");
    ARITHMETIC_UNARY_OP_CASE(ElementReciprocal, "element_reciprocal");
    ARITHMETIC_UNARY_OP_CASE(ElementSin, "element_sin");
    ARITHMETIC_UNARY_OP_CASE(ElementCos, "element_cos");
#undef ARITHMETIC_UNARY_OP_CASE

  case Kinded::Kind::ReluInstKind: {
    auto *RI = cast<ReluInst>(I);
    auto *src = RI->getSrc();
    auto *dest = RI->getDest();
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto srcTy = src->getType();
    auto destTy = dest->getType();

    auto *F = getFunction("element_relu", dest->getElementType());
    llvm::CallInst *stackedOpCall = nullptr;
    if (dest->getElementType() == ElemKind::Int8QTy) {
      auto *srcOffset =
          emitConstI8(builder, static_cast<int8_t>(srcTy->getOffset()));
      auto *destOffset =
          emitConstI8(builder, static_cast<int8_t>(destTy->getOffset()));
      auto destScaleParams = quantization::quantizeScaleOffset32To8(
          srcTy->getScale() / destTy->getScale(), 0);
      auto *destPre = emitConstI32(builder, destScaleParams.pre);
      auto *destPost = emitConstI32(builder, destScaleParams.post);
      auto *destScale = emitConstI32(builder, destScaleParams.scale);
      stackedOpCall = createCall(builder, F,
                                 {loopCount, srcPtr, srcOffset, destOffset,
                                  destPre, destPost, destScale});
    } else if (dest->getElementType() == ElemKind::FloatTy) {
      stackedOpCall = createCall(builder, F, {loopCount, srcPtr});
    } else {
      LOG(FATAL) << "Type is not supported";
    }
    auto *elementTy = getElementType(builder, dest);
    auto *destAddr =
        builder.CreateGEP(elementTy, destPtr, loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::ClipInstKind: {
    auto *CI = cast<ClipInst>(I);
    auto *src = CI->getSrc();
    auto *dest = CI->getDest();
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto srcTy = src->getType();
    auto destTy = dest->getType();
    float clipMinF = CI->getMin();
    float clipMaxF = CI->getMax();

    auto *F = getFunction("element_clip", dest->getElementType());
    llvm::CallInst *stackedOpCall = nullptr;
    if (dest->getElementType() == ElemKind::Int8QTy) {
      TensorQuantizationParams srcTQP{src->getType()->getScale(),
                                      src->getType()->getOffset()};
      int8_t clipMinQ = quantization::quantize<int8_t>(clipMinF, srcTQP);
      int8_t clipMaxQ = quantization::quantize<int8_t>(clipMaxF, srcTQP);
      auto *clipMin = emitConstI8(builder, clipMinQ);
      auto *clipMax = emitConstI8(builder, clipMaxQ);
      auto *srcOffset =
          emitConstI8(builder, static_cast<int8_t>(srcTy->getOffset()));
      auto *destOffset =
          emitConstI8(builder, static_cast<int8_t>(destTy->getOffset()));
      auto destScaleParams = quantization::quantizeScaleOffset32To8(
          srcTy->getScale() / destTy->getScale(), 0);
      auto *destPre = emitConstI32(builder, destScaleParams.pre);
      auto *destPost = emitConstI32(builder, destScaleParams.post);
      auto *destScale = emitConstI32(builder, destScaleParams.scale);
      stackedOpCall =
          createCall(builder, F,
                     {loopCount, srcPtr, clipMin, clipMax, srcOffset,
                      destOffset, destPre, destPost, destScale});
    } else if (dest->getElementType() == ElemKind::FloatTy) {
      auto *clipMin = emitConstF32(builder, clipMinF);
      auto *clipMax = emitConstF32(builder, clipMaxF);
      stackedOpCall =
          createCall(builder, F, {loopCount, srcPtr, clipMin, clipMax});
    } else {
      LOG(FATAL) << "Type is not supported";
    }
    auto *elementTy = getElementType(builder, dest);
    auto *destAddr =
        builder.CreateGEP(elementTy, destPtr, loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::LeakyReluInstKind: {
    auto *LI = cast<LeakyReluInst>(I);
    auto *src = LI->getSrc();
    auto *dest = LI->getDest();
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto srcTy = src->getType();
    auto destTy = dest->getType();

    auto *F = getFunction("element_leaky_relu", dest->getElementType());
    llvm::CallInst *stackedOpCall = nullptr;
    if (dest->getElementType() == ElemKind::Int8QTy) {
      auto *srcOffset =
          emitConstI8(builder, static_cast<int8_t>(srcTy->getOffset()));
      auto *destOffset =
          emitConstI8(builder, static_cast<int8_t>(destTy->getOffset()));
      // Scale parameters for the positive input domain.
      auto posParams = quantization::quantizeScaleOffset32To8(
          srcTy->getScale() / destTy->getScale(), 0);
      auto *posPre = emitConstI32(builder, posParams.pre);
      auto *posPost = emitConstI32(builder, posParams.post);
      auto *posScale = emitConstI32(builder, posParams.scale);
      // Scale parameters for the negative input domain.
      auto negParams = quantization::quantizeScaleOffset32To8(
          srcTy->getScale() * LI->getAlpha() / destTy->getScale(), 0);
      auto *negPre = emitConstI32(builder, negParams.pre);
      auto *negPost = emitConstI32(builder, negParams.post);
      auto *negScale = emitConstI32(builder, negParams.scale);
      stackedOpCall =
          createCall(builder, F,
                     {loopCount, srcPtr, srcOffset, destOffset, posPre, posPost,
                      posScale, negPre, negPost, negScale});
    } else if (dest->getElementType() == ElemKind::FloatTy) {
      auto *alpha = emitConstF32(builder, LI->getAlpha());
      stackedOpCall = createCall(builder, F, {loopCount, srcPtr, alpha});
    } else {
      LOG(FATAL) << "Type is not supported";
    }
    auto *elementTy = getElementType(builder, dest);
    auto *destAddr =
        builder.CreateGEP(elementTy, destPtr, loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::ElementIsNaNInstKind: {
    auto *AN = cast<ElementIsNaNInst>(I);
    auto *src = AN->getSrc();
    auto *dest = AN->getDest();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *F = getFunction("element_is_nan_kernel", src->getElementType());
    auto *stackedOpCall = createUncheckedCall(builder, F, {loopCount, srcPtr});
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

    auto *stackedOpCall = createUncheckedCall(
        builder, F, {loopCount, srcPtr, destScale, destOffset});
    llvm::Value *destAddr = nullptr;
    if (dest->getElementType() == ElemKind::Int8QTy) {
      destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr, loopCount,
                                   "buffer.element.addr");
    } else if (dest->getElementType() == ElemKind::Int32QTy) {
      destAddr = builder.CreateGEP(builder.getInt32Ty(), destPtr, loopCount,
                                   "buffer.element.addr");
    } else {
      LOG(FATAL) << "Type is not supported";
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

    auto *stackedOpCall = createUncheckedCall(
        builder, F, {loopCount, srcPtr, srcScale, srcOffset});
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

    auto *stackedOpCall = createUncheckedCall(
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
    auto *stackedOpCall = createUncheckedCall(
        builder, F, {loopCount, srcPtr, pointerNull, pointerNull});
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
      auto *stackedOpCall = createUncheckedCall(                               \
          builder, F,                                                          \
          {loopCount, lhsPtr, rhsPtr, destOffset, lhsOffset, rhsOffset,        \
           lhsPre, lhsPost, lhsScale, rhsPre, rhsPost, rhsScale});             \
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,         \
                                         loopCount, "buffer.element.addr");    \
      builder.CreateStore(stackedOpCall, destAddr);                            \
    } else {                                                                   \
      auto *stackedOpCall = createUncheckedCall(                               \
          builder, F, {loopCount, lhsPtr, rhsPtr, pointerNull});               \
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,        \
                                         "buffer.element.addr");               \
      builder.CreateStore(stackedOpCall, destAddr);                            \
    }                                                                          \
    break;                                                                     \
  }
    ARITHMETIC_BINARY_OP_CASE(ElementAdd, "element_add");
    ARITHMETIC_BINARY_OP_CASE(ElementSub, "element_sub");
    ARITHMETIC_BINARY_OP_CASE(ElementMax, "element_max");
    ARITHMETIC_BINARY_OP_CASE(ElementMin, "element_min");
    ARITHMETIC_BINARY_OP_CASE(ElementPow, "element_pow");
#undef ARITHMETIC_BINARY_OP_CASE

  case Kinded::Kind::ElementNotInstKind: {
    auto *NI = cast<ElementNotInst>(I);
    auto *dest = NI->getDest();
    auto *src = NI->getSrc();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *F = getFunction("element_not_kernel", src->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *stackedOpCall = createUncheckedCall(builder, F, {loopCount, srcPtr});
    auto *destAddr =
        builder.CreateGEP(elementTy, destPtr, loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::ElementAndInstKind: {
    auto *AI = cast<ElementAndInst>(I);
    auto *dest = AI->getDest();
    auto *lhs = AI->getLHS();
    auto *rhs = AI->getRHS();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);
    auto *F = getFunction("element_and_kernel", lhs->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *stackedOpCall =
        createUncheckedCall(builder, F, {loopCount, lhsPtr, rhsPtr});
    auto *destAddr =
        builder.CreateGEP(elementTy, destPtr, loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::ElementOrInstKind: {
    auto *OI = cast<ElementOrInst>(I);
    auto *dest = OI->getDest();
    auto *lhs = OI->getLHS();
    auto *rhs = OI->getRHS();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);
    auto *F = getFunction("element_or_kernel", lhs->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *stackedOpCall =
        createUncheckedCall(builder, F, {loopCount, lhsPtr, rhsPtr});
    auto *destAddr =
        builder.CreateGEP(elementTy, destPtr, loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::ElementXorInstKind: {
    auto *XI = cast<ElementXorInst>(I);
    auto *dest = XI->getDest();
    auto *lhs = XI->getLHS();
    auto *rhs = XI->getRHS();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);
    auto *F = getFunction("element_xor_kernel", lhs->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *stackedOpCall =
        createUncheckedCall(builder, F, {loopCount, lhsPtr, rhsPtr});
    auto *destAddr =
        builder.CreateGEP(elementTy, destPtr, loopCount, "buffer.element.addr");
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  case Kinded::Kind::ElementCmpEQInstKind:
  case Kinded::Kind::ElementCmpNEQInstKind:
  case Kinded::Kind::ElementCmpLTInstKind:
  case Kinded::Kind::ElementCmpLTEInstKind: {
    Value *dest = nullptr;
    Value *lhs = nullptr;
    Value *rhs = nullptr;
    std::string kernelName;

    if (auto *CEQI = dyn_cast<ElementCmpEQInst>(I)) {
      dest = CEQI->getDest();
      lhs = CEQI->getLHS();
      rhs = CEQI->getRHS();
      kernelName = "element_cmp_eq_kernel";
    } else if (auto *CNEQI = dyn_cast<ElementCmpNEQInst>(I)) {
      dest = CNEQI->getDest();
      lhs = CNEQI->getLHS();
      rhs = CNEQI->getRHS();
      kernelName = "element_cmp_neq_kernel";
    } else if (auto *CLTEI = dyn_cast<ElementCmpLTEInst>(I)) {
      dest = CLTEI->getDest();
      lhs = CLTEI->getLHS();
      rhs = CLTEI->getRHS();
      kernelName = "element_cmp_lte_kernel";
    } else if (auto *CLTI = dyn_cast<ElementCmpLTInst>(I)) {
      dest = CLTI->getDest();
      lhs = CLTI->getLHS();
      rhs = CLTI->getRHS();
      kernelName = "element_cmp_lt_kernel";
    } else {
      llvm_unreachable(
          "Missmatch between Instruction Kind and instruction instance.");
    }

    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *rhsPtr = emitBufferAddress(builder, rhs, kernel, bufferToArgNum);

    // Need _kernel suffix since these operations are implemented as
    // "data-parallel" kernels in libjit.
    auto *F = getFunction(kernelName.c_str(), lhs->getElementType());

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

      auto *stackedOpCall =
          createUncheckedCall(builder, F,
                              {loopCount, lhsPtr, rhsPtr, lhsOffset, rhsOffset,
                               cmpPre, cmpPost, cmpScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *stackedOpCall =
          createUncheckedCall(builder, F, {loopCount, lhsPtr, rhsPtr});
      auto *elementTy = getElementType(builder, dest);
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,
                                         "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }
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
          createUncheckedCall(builder, F,
                              {loopCount, lhsPtr, rhsPtr, destOffset, lhsOffset,
                               rhsOffset, mulPre, mulPost, mulScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else if (lhs->getType()->getElementType() == ElemKind::Int64ITy ||
               lhs->getType()->getElementType() == ElemKind::Int32ITy ||
               lhs->getType()->getElementType() == ElemKind::FloatTy) {
      auto *stackedOpCall = createUncheckedCall(
          builder, F, {loopCount, lhsPtr, rhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,
                                         "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      LOG_ASSERT(false) << "Unsupported element type for Mul.";
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
          createUncheckedCall(builder, F,
                              {loopCount, lhsPtr, rhsPtr, destOffset, lhsOffset,
                               rhsOffset, divPre, divPost, divScale});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *elementTy = getElementType(builder, dest);
      auto *stackedOpCall = createUncheckedCall(
          builder, F, {loopCount, lhsPtr, rhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(elementTy, destPtr, loopCount,
                                         "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }
    break;
  }

  case Kinded::Kind::ModuloInstKind: {
    auto *MI = cast<ModuloInst>(I);
    auto *dest = MI->getDest();
    auto *src = MI->getSrc();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *srcPtr = emitBufferAddress(builder, src, kernel, bufferToArgNum);
    auto *divisor = emitConst(builder, MI->getDivisor(), ElemKind::Int64ITy);
    llvm::Function *F = nullptr;
    // Need _kernel suffix since these operations are implemented as
    // "data-parallel" kernels in libjit.
    if (MI->getSignFollowDivisor()) {
      F = getFunction("element_modulo_kernel_sign_follow",
                      dest->getElementType());
    } else {
      F = getFunction("element_modulo_kernel_no_sign_follow",
                      dest->getElementType());
    }
    auto *stackedOpCall =
        createUncheckedCall(builder, F, {loopCount, divisor, srcPtr});
    llvm::Value *destAddr = nullptr;
    if (dest->getElementType() == ElemKind::Int64ITy) {
      destAddr = builder.CreateGEP(builder.getInt64Ty(), destPtr, loopCount,
                                   "buffer.element.addr");
    } else {
      destAddr = builder.CreateGEP(builder.getInt32Ty(), destPtr, loopCount,
                                   "buffer.element.addr");
    }
    builder.CreateStore(stackedOpCall, destAddr);
    break;
  }

  default:
    std::string sBuf;
    llvm::raw_string_ostream s(sBuf);
    I->dump(s);
    LOG(FATAL) << "Cannot select the instruction: " << s.str();
  }
}

Tensor LLVMIRGen::getTensorForConstantValue(Value *value) {
  // Since we can't get the variable from a glow::Value directly,
  // we need to traverse the var list and find the one matching the given
  // Value.
  Tensor tensor;
  auto *F_ = getIRFunction();
  for (auto &v : F_->findConstants()) {
    assert(isa<WeightVar>(F_->getWeightForNode(v)));
    auto *w = cast<glow::Value>(F_->getWeightForNode(v));
    if (w == value) {
      tensor.assign(&v->getPayload());
      break;
    }
  }
  CHECK(tensor.getUnsafePtr()) << "Can't find the constant value!";
  return tensor;
}

void LLVMIRGen::generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                       const glow::Instruction *I) {
  setCurrentDebugLocation(builder, I);
  assert((!canBePartOfDataParallelKernel(I)) &&
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
    assert(inputTensor->getElementType() == ElemKind::FloatTy &&
           "None float Tensor type for Quantization Profile Instruction.");
    auto *tensorSize = emitConstDimT(builder, inputTensor->getType()->size());

    auto *F = getFunction("quantization_profile");
    createCall(
        builder, F,
        {inputTensorInfoPtr, tensorSize, compInfoPtr, histPtr, histDims});
    break;
  }

  case Kinded::Kind::RowwiseQuantizedFullyConnectedInstKind: {
    auto *RWQFC = cast<RowwiseQuantizedFullyConnectedInst>(I);

    auto scalesT = getTensorForConstantValue(RWQFC->getScales());
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
    auto *row = emitConstDimT(builder, weightsOffsets->dims()[0]);

    auto *destOffset = emitConstI32(builder, dest->getType()->getOffset());
    auto *srcOffset = emitConstI32(builder, src->getType()->getOffset());
    auto *biasOffset = emitConstI32(builder, bOffset);

    llvm::Function *F = nullptr;
    if ((dest->getElementType() == ElemKind::Int8QTy) &&
        (bias->getElementType() == ElemKind::Int8QTy)) {
      F = getFunction("rowwise_quantized_fc_i8_i8");
    } else if ((dest->getElementType() == ElemKind::Int8QTy) &&
               (bias->getElementType() == ElemKind::Int32QTy)) {
      F = getFunction("rowwise_quantized_fc_i8_i32");
    } else {
      LOG(FATAL) << "Unsupported element/bias type for "
                    "RowwiseQuantizedFullyConnectedInst";
    }

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
    auto *numSlice = emitConstDimT(builder, bdim.first);
    auto *sliceSize = emitConstDimT(builder, bdim.second);

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
        LOG(FATAL) << "Type is not supported: "
                   << Type::getElementName(sliceTy->getElementType()).str();
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
    auto *axis = emitConstDimT(builder, BR->getAxis());

    ShapeVector eBatchDims = expandDimsToMax(batch->dims());
    ShapeVector eDestDims = eBatchDims;
    eDestDims[BR->getAxis()] = 1;

    auto *batchDims =
        emitConstDimTArray(builder, llvm::makeArrayRef(eBatchDims));
    auto *destDims = emitConstDimTArray(builder, llvm::makeArrayRef(eDestDims));

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
      auto *destSize = emitConstDimT(builder, dest->size());

      createCall(builder, F,
                 {destPtr, batchPtr, destSize, destDims, batchDims, axis});
    }
    break;
  }

  case Kinded::Kind::BatchedReduceProdInstKind: {
    auto *BR = cast<BatchedReduceProdInst>(I);
    auto *dest = BR->getDest();
    auto *batch = BR->getBatch();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *batchPtr = emitValueAddress(builder, batch);
    auto *axis = emitConstDimT(builder, BR->getAxis());

    ShapeVector eBatchDims = expandDimsToMax(batch->dims());
    ShapeVector eDestDims = eBatchDims;
    eDestDims[BR->getAxis()] = 1;

    auto *batchDims =
        emitConstDimTArray(builder, llvm::makeArrayRef(eBatchDims));
    auto *destDims = emitConstDimTArray(builder, llvm::makeArrayRef(eDestDims));

    auto *F = getFunction("batchedreduceprod", dest->getElementType());

    assert(!batch->getType()->isQuantizedType() &&
           "Quantized implementation for ReduceProd not supported yet.");

    auto *destSize = emitConstDimT(builder, dest->size());

    createCall(builder, F,
               {destPtr, batchPtr, destSize, destDims, batchDims, axis});

    break;
  }

#define BATCHED_REDUCE_MINMAX_CASE(INST_NAME_, FUN_NAME_)                      \
  case Kinded::Kind::Batched##INST_NAME_##InstKind: {                          \
    auto *BR = cast<Batched##INST_NAME_##Inst>(I);                             \
    auto *dest = BR->getDest();                                                \
    auto *batch = BR->getBatch();                                              \
    auto axes = BR->getAxes();                                                 \
    auto *destPtr = emitValueAddress(builder, dest);                           \
    auto *batchPtr = emitValueAddress(builder, batch);                         \
                                                                               \
    ShapeVector eBatchDims = expandDimsToMax(batch->dims());                   \
    ShapeVector eDestDims = eBatchDims;                                        \
    for (dim_t i = 0; i < axes.size(); i++) {                                  \
      eDestDims[axes[i]] = 1;                                                  \
    }                                                                          \
                                                                               \
    auto *batchDims =                                                          \
        emitConstDimTArray(builder, llvm::makeArrayRef(eBatchDims));           \
    auto *destDims =                                                           \
        emitConstDimTArray(builder, llvm::makeArrayRef(eDestDims));            \
                                                                               \
    if (((batch->getElementType() != ElemKind::FloatTy) &&                     \
         (batch->getElementType() != ElemKind::Int32ITy) &&                    \
         (batch->getElementType() != ElemKind::Int64ITy)) ||                   \
        (batch->getElementType() != dest->getElementType())) {                 \
      std::string errStr = "Cannot get function for ";                         \
      std::string name = "INST_NAME_";                                         \
      errStr += name;                                                          \
      llvm_unreachable(errStr.c_str());                                        \
    }                                                                          \
                                                                               \
    llvm::Function *F = getFunction(FUN_NAME_, batch->getElementType());       \
    if (!batch->getType()->isQuantizedType()) {                                \
      auto *destSize = emitConstSizeT(builder, dest->size());                  \
                                                                               \
      createCall(builder, F,                                                   \
                 {destPtr, batchPtr, destSize, destDims, batchDims});          \
    }                                                                          \
    break;                                                                     \
  }
    BATCHED_REDUCE_MINMAX_CASE(ReduceMin, "reducemin")
    BATCHED_REDUCE_MINMAX_CASE(ReduceMax, "reducemax")
#undef BATCHED_REDUCE_MINMAX_CASE

  case Kinded::Kind::ConvolutionInstKind: {
    auto *CI = cast<ConvolutionInst>(I);
    assert(CI->getLayout() == NHWC &&
           "Glow CPU Backend supports only NHWC Convolutions");
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

    auto *kernels = emitConstDimTArray(builder, CI->getKernels());
    auto *strides = emitConstDimTArray(builder, CI->getStrides());
    auto *pads = emitConstDimTArray(builder, CI->getPads());
    auto *group = emitConstDimT(builder, CI->getGroup());
    auto *dilation = emitConstDimTArray(builder, CI->getDilation());

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

    auto *actType = emitConstI32(builder, CI->getFusedActivation());

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

      // Calculate the scaling parameters for the bias and output.
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

      // Emit parameters for fused activation.
      auto *actArgsQuant = emitConstQuantActivationArgs(builder, CI);

      auto *F = getFunction("conv2d",
                            {dest->getElementType(), bias->getElementType()});

      createCall(builder, F,
                 {destPtr,     srcPtr,     filterPtr,  biasPtr,   destDims,
                  srcDims,     filterDims, biasDims,   kernels,   strides,
                  pads,        group,      destOffset, srcOffset, filterOffset,
                  biasOffset,  biasPre,    biasPost,   biasScale, outPre,
                  outPost,     outScale,   unrollD,    dilation,  actType,
                  actArgsQuant});
    } else {

      // Emit parameters for fused activation.
      auto *actArgsFloat = emitConstFloatActivationArgs(builder, CI);

      auto *F = getFunction("conv2d", dest->getElementType());

      createCall(builder, F,
                 {destPtr, srcPtr, filterPtr, biasPtr, destDims, srcDims,
                  filterDims, biasDims, kernels, strides, pads, group, unrollD,
                  dilation, actType, actArgsFloat});
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

    auto *kernels = emitConstDimTArray(builder, CG->getKernels());
    auto *strides = emitConstDimTArray(builder, CG->getStrides());
    auto *pads = emitConstDimTArray(builder, CG->getPads());
    auto *group = emitConstDimT(builder, CG->getGroup());
    auto *dilation = emitConstDimTArray(builder, CG->getDilation());

    auto *F = getFunction("convolution_grad", srcGrad->getElementType());
    createCall(builder, F,
               {srcGradPtr, destGradPtr, srcPtr, filterGradPtr, biasGradPtr,
                filterPtr, destGradDims, srcDims, filterGradDims, kernels,
                strides, pads, group, dilation});
    break;
  }

  case Kinded::Kind::ConvTransposeInstKind: {
    auto *CI = cast<ConvTransposeInst>(I);
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

    auto *kernels = emitConstDimTArray(builder, CI->getKernels());
    auto *strides = emitConstDimTArray(builder, CI->getStrides());
    auto *pads = emitConstDimTArray(builder, CI->getPads());
    auto *group = emitConstDimT(builder, CI->getGroup());
    auto *dilation = emitConstDimTArray(builder, CI->getDilation());

    const char *kernelName = "conv_transpose";

    auto *F = getFunction(kernelName, dest->getElementType());

    if (src->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *srcTy = src->getType();
      auto *filterTy = filter->getType();

      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *srcOffset = emitConstI32(builder, srcTy->getOffset());
      auto *filterOffset = emitConstI32(builder, filterTy->getOffset());

      // Calculate the scale of the values that come out of the matrix
      // multiplication part of the calculation.
      float matMulScale = srcTy->getScale() * filterTy->getScale();

      // Calculate the scaling parameters for the bias and output.
      auto outScaleParam = quantization::quantizeScaleOffset32To8(
          matMulScale / destTy->getScale(), 0);

      // Pass the pre-shift, post-shift and integer scale parameters for the
      // output calculation.
      auto *outPre = emitConstI32(builder, outScaleParam.pre);
      auto *outPost = emitConstI32(builder, outScaleParam.post);
      auto *outScale = emitConstI32(builder, outScaleParam.scale);

      createCall(builder, F,
                 {destPtr, srcPtr, filterPtr, biasPtr, destDims, srcDims,
                  filterDims, biasDims, kernels, strides, pads, group,
                  destOffset, srcOffset, filterOffset, outPre, outPost,
                  outScale, dilation});
    } else {
      createCall(builder, F,
                 {destPtr, srcPtr, filterPtr, biasPtr, destDims, srcDims,
                  filterDims, biasDims, kernels, strides, pads, group,
                  dilation});
    }
    break;
  }

  case Kinded::Kind::ChannelwiseQuantizedConvolutionInstKind: {
    auto *CQCI = cast<ChannelwiseQuantizedConvolutionInst>(I);
    auto *dest = CQCI->getDest();
    auto *src = CQCI->getSrc();
    auto *filter = CQCI->getFilter();
    auto *bias = CQCI->getBias();
    auto *filterScales = CQCI->getFilterScales();
    auto *filterOffsets = CQCI->getFilterOffsets();
    auto *biasScales = CQCI->getBiasScales();
    auto *biasOffsets = CQCI->getBiasOffsets();

    auto *destTy = dest->getType();
    auto *srcTy = src->getType();

    auto filterScalesT = getTensorForConstantValue(filterScales);
    auto filterScalesH = filterScalesT.getHandle<float>();

    auto biasScalesT = getTensorForConstantValue(biasScales);
    auto biasScalesH = biasScalesT.getHandle<float>();

    // Compute quantization parameters for each channel.
    auto channelNum = dest->dims().back();
    std::vector<llvm::Constant *> biasPreV(channelNum);
    std::vector<llvm::Constant *> biasPostV(channelNum);
    std::vector<llvm::Constant *> biasScaleV(channelNum);
    std::vector<llvm::Constant *> outputPreV(channelNum);
    std::vector<llvm::Constant *> outputPostV(channelNum);
    std::vector<llvm::Constant *> outputScaleV(channelNum);
    for (size_t i = 0; i < channelNum; i++) {

      // Compute the scaling parameters for bias and output.
      float matMulScale = srcTy->getScale() * filterScalesH.raw(i);
      auto biasScaleParam = quantization::quantizeScaleOffset32To8(
          biasScalesH.raw(i) / matMulScale, 0);
      auto outScaleParam = quantization::quantizeScaleOffset32To8(
          matMulScale / destTy->getScale(), 0);

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

    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *filterPtr = emitValueAddress(builder, filter);
    auto *biasPtr = emitValueAddress(builder, bias);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);
    auto *filterDims = emitValueDims(builder, filter);
    auto *biasDims = emitValueDims(builder, bias);

    auto *kernels = emitConstDimTArray(builder, CQCI->getKernels());
    auto *strides = emitConstDimTArray(builder, CQCI->getStrides());
    auto *pads = emitConstDimTArray(builder, CQCI->getPads());
    auto *group = emitConstDimT(builder, CQCI->getGroup());
    auto *dilation = emitConstDimTArray(builder, CQCI->getDilation());

    auto *destOffset = emitConstI32(builder, destTy->getOffset());
    auto *srcOffset = emitConstI32(builder, srcTy->getOffset());
    auto *filterOffsetsPtr = emitValueAddress(builder, filterOffsets);
    auto *biasOffsetsPtr = emitValueAddress(builder, biasOffsets);

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

    bool isConv3D = (srcTy->dims().size() == 5);
    auto *F = getFunction(isConv3D ? "channelwise_quantized_conv3d"
                                   : "channelwise_quantized_conv2d",
                          {dest->getElementType(), bias->getElementType()});

    auto *actType = emitConstI32(builder, CQCI->getFusedActivation());
    auto *actArgsQuant = emitConstQuantActivationArgs(builder, CQCI);

    createCall(builder, F,
               {destPtr,        srcPtr,        filterPtr,      biasPtr,
                destDims,       srcDims,       filterDims,     biasDims,
                kernels,        strides,       pads,           group,
                dilation,       destOffset,    srcOffset,      filterOffsetsPtr,
                biasOffsetsPtr, biasPrePtr,    biasPostPtr,    biasScalePtr,
                outputPrePtr,   outputPostPtr, outputScalePtr, actType,
                actArgsQuant});
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

    auto *F = getFunction("cross_entropy_loss",
                          {CE->getElementType(), labels->getElementType()});
    createCall(builder, F, {CEPtr, PPtr, labelsPtr, dims});
    break;
  }

  case Kinded::Kind::LengthsToRangesInstKind: {
    auto *LTR = cast<LengthsToRangesInst>(I);
    auto *dest = LTR->getDest();
    auto *lengths = LTR->getLengths();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *lengthsPtr = emitValueAddress(builder, lengths);
    auto *size = emitConstDimT(builder, lengths->dims()[0]);
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

    auto *lengthsSize = emitConstDimT(builder, lengths->size());
    auto *dataType = data->getType();
    auto *destSize = emitConstDimT(builder, dest->size());
    auto *sliceSize =
        emitConstDimT(builder, dataType->size() / dataType->dims()[0]);

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
    auto *halfWindow = emitConstDimT(builder, LRN->getHalfWindowSize());
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

    auto *halfWindow = emitConstDimT(builder, LRNG->getHalfWindowSize());
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
    assert(PM->getLayout() == NHWC &&
           "Glow CPU Backend supports only NHWC Pools");
    auto *dest = PM->getDest();
    auto *src = PM->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *kernels = emitConstDimTArray(builder, PM->getKernels());
    auto *strides = emitConstDimTArray(builder, PM->getStrides());
    auto *pads = emitConstDimTArray(builder, PM->getPads());

    auto *F = getFunction("max_pool", dest->getElementType());

    if (src->getType()->isQuantizedType()) {
      auto *destOffset = emitConstI32(builder, dest->getType()->getOffset());
      createCall(builder, F,
                 {srcPtr, destPtr, srcDims, destDims, kernels, strides, pads,
                  destOffset});
    } else {
      createCall(builder, F,
                 {srcPtr, destPtr, srcDims, destDims, kernels, strides, pads});
    }
    break;
  }

  case Kinded::Kind::MaxPoolWithArgmaxInstKind: {
    auto *PMXY = cast<MaxPoolWithArgmaxInst>(I);
    assert(PMXY->getLayout() == NHWC &&
           "Glow CPU Backend supports only NHWC Pools");
    auto *dest = PMXY->getDest();
    auto *src = PMXY->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *argMax = PMXY->getArgmax();
    auto *argmaxPtr = emitValueAddress(builder, argMax);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *kernels = emitConstDimTArray(builder, PMXY->getKernels());
    auto *strides = emitConstDimTArray(builder, PMXY->getStrides());
    auto *pads = emitConstDimTArray(builder, PMXY->getPads());

    auto *F = getFunction("max_pool_argmax",
                          {dest->getElementType(), argMax->getElementType()});
    createCall(builder, F,
               {srcPtr, destPtr, argmaxPtr, srcDims, destDims, kernels, strides,
                pads});
    break;
  }

  case Kinded::Kind::MaxPoolWithArgmaxGradInstKind: {
    auto *PMG = cast<MaxPoolWithArgmaxGradInst>(I);
    auto *srcGrad = PMG->getSrcGrad();
    auto *srcGradPtr = emitValueAddress(builder, srcGrad);
    auto *destGradPtr = emitValueAddress(builder, PMG->getDestGrad());
    auto *argMax = PMG->getArgmax();
    auto *argmaxPtr = emitValueAddress(builder, argMax);

    auto *srcGradDims = emitValueDims(builder, srcGrad);
    auto *destDims = emitValueDims(builder, PMG->getDest());

    auto *F = getFunction("max_pool_argmax_grad", {srcGrad->getElementType(),
                                                   argMax->getElementType()});
    createCall(builder, F,
               {srcGradPtr, destGradPtr, argmaxPtr, srcGradDims, destDims});
    break;
  }

  case Kinded::Kind::ArgMaxInstKind: {
    auto *AM = cast<ArgMaxInst>(I);
    auto *dest = AM->getDest();
    auto *src = AM->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *srcDims = emitValueDims(builder, src);
    auto *srcNumDims = emitConstSizeT(builder, src->dims().size());
    auto *axis = emitConstSizeT(builder, AM->getAxis());
    auto *F =
        getFunction("arg_max", {src->getElementType(), dest->getElementType()});
    createCall(builder, F, {srcPtr, destPtr, srcDims, srcNumDims, axis});
    break;
  }

  case Kinded::Kind::ArgMinInstKind: {
    auto *AM = cast<ArgMinInst>(I);
    auto *dest = AM->getDest();
    auto *src = AM->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *srcDims = emitValueDims(builder, src);
    auto *srcNumDims = emitConstSizeT(builder, src->dims().size());
    auto *axis = emitConstSizeT(builder, AM->getAxis());
    auto *F =
        getFunction("arg_min", {src->getElementType(), dest->getElementType()});
    createCall(builder, F, {srcPtr, destPtr, srcDims, srcNumDims, axis});
    break;
  }

  case Kinded::Kind::AvgPoolInstKind: {
    auto *PA = cast<AvgPoolInst>(I);
    assert(PA->getLayout() == NHWC &&
           "Glow CPU Backend supports only NHWC Pools");
    auto *dest = PA->getDest();
    auto *src = PA->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *kernels = emitConstDimTArray(builder, PA->getKernels());
    auto *strides = emitConstDimTArray(builder, PA->getStrides());
    auto *pads = emitConstDimTArray(builder, PA->getPads());
    auto *countIncludePads = emitConstI1(builder, PA->getCountIncludePads());

    auto *F = getFunction("avg_pool", dest->getElementType());

    if (src->getType()->isQuantizedType()) {
      auto *destTy = dest->getType();
      auto *srcTy = src->getType();
      auto *destOffset = emitConstI32(builder, destTy->getOffset());
      auto *srcOffset = emitConstI32(builder, srcTy->getOffset());
      // When we count the padding pixels in the normalizing factor we include
      // the filter area in the scaling parameters since it is a constant.
      float scale = srcTy->getScale() / destTy->getScale();
      if (PA->getCountIncludePads()) {
        scale = scale / (PA->getKernels()[0] * PA->getKernels()[1]);
      }
      auto outScaleParam = quantization::quantizeScaleOffset32To8(scale, 0);
      auto *outPre = emitConstI32(builder, outScaleParam.pre);
      auto *outPost = emitConstI32(builder, outScaleParam.post);
      auto *outScale = emitConstI32(builder, outScaleParam.scale);
      createCall(builder, F,
                 {srcPtr, destPtr, srcDims, destDims, kernels, strides, pads,
                  countIncludePads, destOffset, srcOffset, outPre, outPost,
                  outScale});
    } else {
      createCall(builder, F,
                 {srcPtr, destPtr, srcDims, destDims, kernels, strides, pads,
                  countIncludePads});
    }
    break;
  }

  case Kinded::Kind::AdaptiveAvgPoolInstKind: {
    auto *PA = cast<AdaptiveAvgPoolInst>(I);

    auto *dest = PA->getDest();
    auto *src = PA->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    auto *F = getFunction("adaptive_avg_pool", dest->getElementType());
    createCall(builder, F, {srcPtr, destPtr, srcDims, destDims});
    break;
  }

  case Kinded::Kind::AvgPoolGradInstKind: {
    auto *PAG = cast<AvgPoolGradInst>(I);
    auto *srcGrad = PAG->getSrcGrad();
    auto *srcGradPtr = emitValueAddress(builder, srcGrad);
    auto *destGradPtr = emitValueAddress(builder, PAG->getDestGrad());

    auto *srcGradDims = emitValueDims(builder, srcGrad);
    auto *destDims = emitValueDims(builder, PAG->getDest());

    auto *kernels = emitConstDimTArray(builder, PAG->getKernels());
    auto *strides = emitConstDimTArray(builder, PAG->getStrides());
    auto *pads = emitConstDimTArray(builder, PAG->getPads());
    auto *countIncludePads = emitConstI1(builder, PAG->getCountIncludePads());

    auto *F = getFunction("avg_pool_grad", srcGrad->getElementType());
    createCall(builder, F,
               {srcGradPtr, destGradPtr, srcGradDims, destDims, kernels,
                strides, pads, countIncludePads});
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

    auto *F = getFunction("softmax_grad", {srcGrad->getElementType(),
                                           selected->getElementType()});
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

    auto *k = emitConstDimT(builder, TI->getK());
    auto *n = emitConstDimT(builder, input->dims().back());
    auto *size = emitConstDimT(builder, input->size());

    auto indicesTy = TI->getIndices()->getElementType();
    auto *F = getFunction("topk", {input->getElementType(), indicesTy});

    createCall(builder, F,
               {valuesPtr, indicesPtr, inputPtr, scratchPtr, k, n, size});
    break;
  }

  case Kinded::Kind::SpaceToDepthInstKind: {
    auto *SI = cast<SpaceToDepthInst>(I);
    auto *dest = SI->getDest();
    auto *src = SI->getSrc();

    auto *dstPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);

    auto *dstDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);

    unsigned blockSize = SI->getBlockSize();

    auto *F = getFunction("space_to_depth", src->getElementType());
    createCall(
        builder, F,
        {srcPtr, dstPtr, emitConstDimT(builder, blockSize), srcDims, dstDims});
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

    auto *shuffle = emitConstDimTArray(builder, llvm::makeArrayRef(shuffSizeT));
    auto *len = emitConstDimT(builder, TI->getShuffle().size());

    auto *F = getFunction("transpose", dest->getElementType());
    createCall(builder, F, {srcPtr, destPtr, srcDims, destDims, shuffle, len});
    break;
  }

  case Kinded::Kind::FlipInstKind: {
    auto *FI = cast<FlipInst>(I);
    auto *dest = FI->getDest();
    auto *src = FI->getSrc();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *dims = emitValueDims(builder, src);
    auto *axis = emitConstDimT(builder, FI->getAxis());
    auto *dimsSize = emitConstDimT(builder, src->getType()->dims().size());
    auto *F = getFunction("flip", src->getElementType());
    createCall(builder, F, {srcPtr, destPtr, dims, axis, dimsSize});
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

    auto *destDimsSize = emitConstDimT(builder, dest->getType()->dims().size());
    auto *srcDimsSize = emitConstDimT(builder, src->getType()->dims().size());
    auto *offsetsPtr = emitConstDimTArray(builder, offsets);
    auto *offsetsArraySize = emitConstDimT(builder, offsets.size());
    auto *count = emitConstDimT(builder, ITI->getCount());
    auto *axis = emitConstDimT(builder, ITI->getAxis());

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

    auto *destDimsSize = emitConstDimT(builder, dest->getType()->dims().size());
    auto *srcDimsSize = emitConstDimT(builder, src->getType()->dims().size());
    auto *offsetsPtr = emitConstDimTArray(builder, offsets);
    auto *offsetsArraySize = emitConstDimT(builder, offsets.size());

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

    auto *indicesSize = emitConstDimT(builder, indices->size());

    auto *dataType = data->getType();

    // The size of the sample in the batch.
    size_t sampleSize = dataType->getSliceSize(batchDims);
    // The size of the slices that we gather.
    size_t sliceSize = dataType->getSliceSize(batchDims + 1);
    // The size of each sample in the batch.
    size_t numSamples = dataType->size() / sampleSize;

    auto *sliceSizeVal = emitConstDimT(builder, sliceSize);
    auto *numSamplesVal = emitConstDimT(builder, numSamples);
    auto *sampleSizeVal = emitConstDimT(builder, sampleSize);

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

    auto *numExamplesVal = emitConstDimT(builder, numExamples);
    auto *exampleSizeVal = emitConstDimT(builder, exampleSize);

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

  case Kinded::Kind::LengthsRangeFillInstKind: {
    auto *LRFI = llvm::cast<LengthsRangeFillInst>(I);
    auto *dest = LRFI->getDest();
    auto *lengths = LRFI->getLengths();

    auto *destPtr = emitValueAddress(builder, dest);
    auto *lengthsPtr = emitValueAddress(builder, lengths);

    auto *lengthsSize = emitConstDimT(builder, lengths->size());

    // Dispatching function depending on the input type of Ranges.
    auto *F = getFunction("lengths_range_fill", dest->getElementType());
    createCall(builder, F, {lengthsPtr, destPtr, lengthsSize});
    break;
  }

  case Kinded::Kind::ScatterDataInstKind: {
    auto *SDI = llvm::cast<ScatterDataInst>(I);
    auto *data = SDI->getData();
    auto *indices = SDI->getIndices();
    auto *slices = SDI->getSlices();

    auto *dataPtr = emitValueAddress(builder, data);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *slicesPtr = emitValueAddress(builder, slices);
    auto *dataDims = emitValueDims(builder, data);

    auto *indicesCnt = emitConstDimT(builder, indices->getType()->dims()[0]);
    auto *indicesSize = emitConstDimT(builder, indices->getType()->dims()[1]);
    auto *slicesType = slices->getType();
    auto *sliceSize =
        emitConstDimT(builder, slicesType->size() / slicesType->dims()[0]);
    auto *isCumulative = emitConstI1(builder, SDI->getCumulative());
    auto *F = getFunction("scatterdata",
                          {data->getElementType(), indices->getElementType()});
    if (data->getType()->isQuantizedType()) {
      auto *dataScale = emitConstF32(builder, data->getType()->getScale());
      auto *dataOffset = emitConstI32(builder, data->getType()->getOffset());
      auto *sliceScale = emitConstF32(builder, slices->getType()->getScale());
      auto *sliceOffset = emitConstI32(builder, slices->getType()->getOffset());
      createCall(builder, F,
                 {dataPtr, dataDims, indicesPtr, slicesPtr, indicesCnt,
                  indicesSize, sliceSize, isCumulative, dataScale, dataOffset,
                  sliceScale, sliceOffset});
    } else {
      createCall(builder, F,
                 {dataPtr, dataDims, indicesPtr, slicesPtr, indicesCnt,
                  indicesSize, sliceSize, isCumulative});
    }
    break;
  }

  case Kinded::Kind::SparseLengthsSumInstKind: {
    auto *SI = cast<SparseLengthsSumInst>(I);
    auto *dest = SI->getDest();
    auto *data = SI->getData();
    auto *indices = SI->getIndices();
    auto *lengths = SI->getLengths();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *lengthsPtr = emitValueAddress(builder, lengths);
    auto *segments = emitConstDimT(builder, lengths->dims()[0]);
    auto *lineSize = emitConstDimT(builder, data->size() / data->dims()[0]);
    auto *F = getFunction("sparse_lengths_sum",
                          {dest->getElementType(), indices->getElementType()});
    createCall(builder, F,
               {destPtr, dataPtr, indicesPtr, lengthsPtr, segments, lineSize});
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
    auto *segments = emitConstDimT(builder, lengths->dims()[0]);
    auto *lineSize = emitConstDimT(builder, data->size() / data->dims()[0]);
    auto *F = getFunction("sparse_lengths_weighted_sum",
                          {dest->getElementType(), indices->getElementType()});
    createCall(builder, F,
               {destPtr, dataPtr, weightsPtr, indicesPtr, lengthsPtr, segments,
                lineSize});
    break;
  }

  case Kinded::Kind::EmbeddingInstKind: {
    auto *SI = cast<EmbeddingInst>(I);
    auto *dest = SI->getDest();
    auto *weights = SI->getWeights();
    auto *indices = SI->getIndices();
    auto *padIdx = emitConstSizeT(builder, SI->getPadIdx());
    auto *scale = emitConstI1(builder, SI->getScale());
    auto *sparse = emitConstI1(builder, SI->getSparse());
    auto *destPtr = emitValueAddress(builder, dest);
    auto *weightsPtr = emitValueAddress(builder, weights);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *indDims = emitValueDims(builder, indices);
    auto *indSize = emitConstDimT(builder, indices->dims().size());
    assert(weights->dims().size() == 2 && "weights must be 2-D");
    auto *numEmbedding = emitConstDimT(builder, weights->dims()[0]);
    auto *embeddingDim = emitConstDimT(builder, weights->dims()[1]);
    auto *F = getFunction("embedding", dest->getElementType());
    createCall(builder, F,
               {destPtr, weightsPtr, indicesPtr, indDims, indSize, numEmbedding,
                embeddingDim, padIdx, scale, sparse});
    break;
  }

  case Kinded::Kind::EmbeddingBagInstKind: {
    auto *SI = cast<EmbeddingBagInst>(I);
    auto *dest = SI->getDest();
    auto *data = SI->getData();
    auto *weights = SI->getWeights();
    auto *indices = SI->getIndices();
    auto *offsets = SI->getOffsets();
    auto *hasEndOffset = emitConstI1(builder, SI->getHasEndOffset());
    auto *destPtr = emitValueAddress(builder, dest);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *weightsPtr = emitValueAddress(builder, weights);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *offsetsPtr = emitValueAddress(builder, offsets);
    auto *segments = emitConstDimT(builder, offsets->dims()[0]);
    auto *totalLength = emitConstDimT(builder, indices->dims()[0]);
    auto *lineSize = emitConstDimT(builder, data->size() / data->dims()[0]);
    auto *F = getFunction("embedding_bag", dest->getElementType());
    createCall(builder, F,
               {destPtr, dataPtr, weightsPtr, indicesPtr, offsetsPtr, segments,
                lineSize, totalLength, hasEndOffset});
    break;
  }

  case Kinded::Kind::SparseLengthsWeightedSumGradInstKind: {
    auto *SI = cast<SparseLengthsWeightedSumGradInst>(I);
    auto *destGrad = SI->getDestGrad();
    auto *dataGrad = SI->getDataGrad();
    auto *weightsGrad = SI->getWeightsGrad();
    auto *data = SI->getData();
    auto *weights = SI->getWeights();
    auto *indices = SI->getIndices();
    auto *lengths = SI->getLengths();
    auto *destGradPtr = emitValueAddress(builder, destGrad);
    auto *dataGradPtr = emitValueAddress(builder, dataGrad);
    auto *weightsGradPtr = emitValueAddress(builder, weightsGrad);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *weightsPtr = emitValueAddress(builder, weights);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *lengthsPtr = emitValueAddress(builder, lengths);
    auto *segments = emitConstDimT(builder, lengths->dims()[0]);
    auto *dataGradRawSize =
        emitConstDimT(builder, dataGrad->size() * sizeof(float));
    auto *lineSize =
        emitConstDimT(builder, dataGrad->size() / dataGrad->dims()[0]);
    auto *F =
        getFunction("sparse_lengths_weighted_sum_grad",
                    {destGrad->getElementType(), indices->getElementType()});
    createCall(builder, F,
               {destGradPtr, dataGradPtr, weightsGradPtr, dataPtr, weightsPtr,
                indicesPtr, lengthsPtr, segments, lineSize, dataGradRawSize});
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
    auto *segments = emitConstDimT(builder, lengths->dims()[0]);
    auto *lineSize = emitConstDimT(builder, data->size() / data->dims()[0]);
    auto *F = getFunction("rowwise_quantized_sparse_lengths_weighted_sum",
                          {dest->getElementType(), indices->getElementType()});
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
    auto *segments = emitConstDimT(builder, lengths->dims()[0]);
    auto *inLineSize = emitConstDimT(builder, data->size() / data->dims()[0]);
    auto *outLineSize = emitConstDimT(builder, dest->size() / dest->dims()[0]);
    auto *F = getFunction("fused_rowwise_quantized_sparse_lengths_weighted_sum",
                          {dest->getElementType(), indices->getElementType()});
    createCall(builder, F,
               {destPtr, dataPtr, weightsPtr, indicesPtr, lengthsPtr, segments,
                inLineSize, outLineSize});
    break;
  }

  case Kinded::Kind::EmbeddingBagByteRowwiseOffsetsInstKind: {
    auto *N = cast<EmbeddingBagByteRowwiseOffsetsInst>(I);
    auto *dest = N->getDest();
    auto *data = N->getData();
    auto *weights = N->getWeights();
    auto *indices = N->getIndices();
    auto *offsets = N->getOffsets();
    auto *hasEndOffset = emitConstI1(builder, N->getHasEndOffset());
    auto *destPtr = emitValueAddress(builder, dest);
    auto *dataPtr = emitValueAddress(builder, data);
    auto *weightsPtr = emitValueAddress(builder, weights);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *offsetsPtr = emitValueAddress(builder, offsets);
    auto *segments = emitConstDimT(builder, offsets->dims()[0]);
    auto *numIndices = emitConstDimT(builder, indices->dims()[0]);
    auto *inLineSize = emitConstDimT(builder, data->size() / data->dims()[0]);
    auto *outLineSize = emitConstDimT(builder, dest->size() / dest->dims()[0]);
    auto *F = getFunction("embedding_bag_byte_rowwise_offsets",
                          dest->getElementType());
    createCall(builder, F,
               {destPtr, dataPtr, weightsPtr, indicesPtr, offsetsPtr, segments,
                numIndices, inLineSize, outLineSize, hasEndOffset});
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

    auto *indicesSize = emitConstDimT(builder, indices->size());
    auto *destSize = emitConstDimT(builder, dest->size());

    auto *valuesType = values->getType();
    auto *valueSize =
        emitConstDimT(builder, valuesType->size() / valuesType->dims()[0]);

    auto *F = getFunction("sparse_to_dense",
                          {dest->getElementType(), indices->getElementType()});
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
    auto *srcDimsSize = emitConstDimT(builder, src->getType()->dims().size());
    auto *srcSize = emitConstSizeT(builder, src->getType()->size());
    auto *srcSizeBytes =
        emitConstSizeT(builder, src->getType()->getSizeInBytes());
    auto *srcElemKind =
        emitConstDimT(builder, static_cast<size_t>(src->getElementType()));
    auto *name = emitStringConst(builder, I->getName());
    auto *filename = emitStringConst(builder, DPI->getFileName());
    auto srcTypeStr = src->getType()->toString();

    std::string format = DPI->getFormat();
    if (format == "console") {
      // Dump tensor in console.
      auto *F = getFunction("dump_tensor_console");
      createCall(builder, F, {srcPtr, srcDims, srcDimsSize, srcElemKind, name});

    } else if (format == "bin") {
      // Dump tensor in file in binary format.
      auto *F = getFunction("dump_tensor_bin");
      auto *header = emitStringConst(builder, srcTypeStr);
      createCall(builder, F, {srcPtr, srcSizeBytes, filename, header});

    } else if (format == "txt") {
      // Dump tensor in file in text format.
      auto *F = getFunction("dump_tensor_txt", src->getElementType());
      auto *header = emitStringConst(builder, srcTypeStr);
      createCall(builder, F, {srcPtr, srcSize, filename, header});

    } else if (format == "rawbin") {
      // Dump tensor in file in raw binary format.
      auto *F = getFunction("dump_tensor_bin");
      auto *header = emitStringConst(builder, "");
      createCall(builder, F, {srcPtr, srcSizeBytes, filename, header});

    } else if (format == "rawtxt") {
      // Dump tensor in file in raw text format.
      auto *F = getFunction("dump_tensor_txt", src->getElementType());
      auto *header = emitStringConst(builder, "");
      createCall(builder, F, {srcPtr, srcSize, filename, header});

    } else {
      LOG(FATAL) << "Invalid 'Format' attribute for DebugPrint instruction!";
    }
    break;
  }

  case Kinded::Kind::InstrumentInstKind: {
    auto *instrumentI = llvm::cast<InstrumentInst>(I);
    auto *opInfo = instrumentI->getOperandsInfo();

    // Instruction being instrumented.
    Instruction *instrRef = instrumentI->getInstrRef();

    // Emit instruction ID and instruction kind.
    llvm::Type *intTy =
        llvm::Type::getIntNTy(getLLVMContext(), getLibjitIntWidth());
    auto *ID = llvm::ConstantInt::get(intTy, instrumentI->getID());
    auto *kind = llvm::ConstantInt::get(intTy, (int)(instrRef->getKind()));

    // Emit number of input and output operands.
    auto inpNum = instrRef->getNumInputs();
    auto outNum = instrRef->getNumOutputs();
    auto opNum = inpNum + outNum;
    auto *opInp = llvm::ConstantInt::get(intTy, inpNum);
    auto *opOut = llvm::ConstantInt::get(intTy, outNum);

    // Emit opInfo address as uint8_t*.
    assert(opInfo->getType()->getSizeInBytes() >= 2 * sizeof(int64_t) &&
           "Not enough memory allocated for instrumentation!");
    auto *opInfoPtr = emitValueAddress(builder, opInfo);
    opInfoPtr = builder.CreateBitCast(opInfoPtr, builder.getInt8PtrTy());

    // Emit opAddr address as uint8_t** starting from offset 0.
    auto *opAddrPtr =
        builder.CreateGEP(opInfoPtr, llvm::ConstantInt::get(intTy, 0));
    opAddrPtr = builder.CreateBitCast(opAddrPtr,
                                      builder.getInt8PtrTy()->getPointerTo());

    // Emit opSize address as int* starting from offset opNum * sizeof(int64_t).
    auto *opSizePtr = builder.CreateGEP(
        opInfoPtr, llvm::ConstantInt::get(intTy, opNum * sizeof(int64_t)));
    opSizePtr = builder.CreateBitCast(opSizePtr, intTy->getPointerTo());

    // Generate instrumentation.
    auto instrumentKind = instrumentI->getInstrumentKind();
    if (instrumentKind == InstrumentKind::Before) {

      // Operands addresses and sizes.
      std::vector<llvm::Value *> opAddrArray;
      std::vector<llvm::Value *> opSizeArray;

      // Get addresses and sizes for the input operands.
      for (const auto &op : instrRef->getOperands()) {
        if (op.second == OperandKind::Out) {
          continue;
        }
        // Emit operand address as uint8_t* variable.
        auto *opAddr = emitValueAddress(builder, op.first);
        opAddr = builder.CreateBitCast(opAddr, builder.getInt8PtrTy());
        opAddrArray.push_back(opAddr);
        // Emit operand size in bytes as int constant.
        auto *opSize = llvm::ConstantInt::get(
            intTy, op.first->getType()->getSizeInBytes());
        opSizeArray.push_back(opSize);
      }
      assert(opAddrArray.size() == inpNum && "Inconsistent size!");

      // Get addresses and sizes for the output operands.
      for (const auto &op : instrRef->getOperands()) {
        if (op.second == OperandKind::In) {
          continue;
        }
        // Emit operand address as uint8_t* variable.
        auto *opAddr = emitValueAddress(builder, op.first);
        opAddr = builder.CreateBitCast(opAddr, builder.getInt8PtrTy());
        opAddrArray.push_back(opAddr);
        // Emit operand size in bytes as int constant.
        auto *opSize = llvm::ConstantInt::get(
            intTy, op.first->getType()->getSizeInBytes());
        opSizeArray.push_back(opSize);
      }
      assert(opAddrArray.size() == opNum && "Inconsistent size!");

      // Write the addresses of the operands in the opAddr.
      emitArrayStore(builder, opAddrArray, opAddrPtr);

      // Write the sizes of the operands in opSize.
      emitArrayStore(builder, opSizeArray, opSizePtr);

      // Create callback call.
      auto *F = getFunction("instrument_before");
      createCall(builder, F, {ID, kind, opInp, opOut, opAddrPtr, opSizePtr});

    } else if (instrumentKind == InstrumentKind::After) {

      // Create callback call.
      auto *F = getFunction("instrument_after");
      createCall(builder, F, {ID, kind, opInp, opOut, opAddrPtr, opSizePtr});

    } else {
      llvm_unreachable("Instrumentation kind not supported!");
    }
    // Print the IR instrumentation callback API.
    printInstrumentIR_ = true;
    break;
  }

  case Kinded::Kind::TraceEventInstKind: {
    auto *TEI = llvm::cast<TraceEventInst>(I);
    auto *data = TEI->getData();
    auto *offset = emitConstDimT(builder, TEI->getIndex());
    auto *dataPtr = emitValueAddress(builder, data);
    auto *F = getFunction("write_timestamp");
    createCall(builder, F, {dataPtr, offset});
    break;
  }

  case Kinded::Kind::ResizeNearestInstKind: {
    auto *RNI = llvm::cast<ResizeNearestInst>(I);
    auto *result = RNI->getDest();
    auto *input = RNI->getSrc();
    auto *resultPtr = emitValueAddress(builder, result);
    auto *inputPtr = emitValueAddress(builder, input);

    auto *scalePtr = emitConstFloatArray(builder, RNI->getScale());
    auto *destDims = emitValueDims(builder, result);
    auto *srcDims = emitValueDims(builder, input);
    auto *F = getFunction("resizenearest", input->getElementType());
    createCall(builder, F, {resultPtr, inputPtr, scalePtr, srcDims, destDims});
    break;
  }

  case Kinded::Kind::ResizeBilinearInstKind: {
    auto *RBI = llvm::cast<ResizeBilinearInst>(I);
    auto *result = RBI->getDest();
    auto *input = RBI->getSrc();
    auto *resultPtr = emitValueAddress(builder, result);
    auto *inputPtr = emitValueAddress(builder, input);

    CHECK_EQ(RBI->getScale()[0], 1.0) << "Scaling batch not supported.";
    CHECK_EQ(RBI->getScale()[3], 1.0) << "Scaling channel not supported.";

    auto *scalePtr = emitConstFloatArray(builder, RBI->getScale());
    auto *destDims = emitValueDims(builder, result);
    auto *srcDims = emitValueDims(builder, input);
    auto *F = getFunction("resizebilinear", input->getElementType());
    createCall(builder, F, {resultPtr, inputPtr, scalePtr, srcDims, destDims});
    break;
  }

  case Kinded::Kind::NonMaxSuppressionInstKind: {
    auto *NMSI = llvm::cast<NonMaxSuppressionInst>(I);
    auto boxes = NMSI->getBoxes();
    auto scores = NMSI->getScores();
    auto indices = NMSI->getIndices();
    auto numDetected = NMSI->getNumberOfSelectedIndices();
    float iouThreshold = NMSI->getIouThreshold();
    int64_t maxBoxesPerClass = NMSI->getMaxOutputBoxesPerClass();
    float scoreThreshold = NMSI->getScoreThreshold();
    int centerPointBox = NMSI->getCenterPointBox();
    bool isV4 = NMSI->getIsTFVersion4();

    auto *boxesPtr = emitValueAddress(builder, boxes);
    auto *scoresPtr = emitValueAddress(builder, scores);
    auto *indicesPtr = emitValueAddress(builder, indices);
    auto *numDetectedPtr = emitValueAddress(builder, numDetected);

    auto *maxBoxesPerClassVal = emitConstI32(builder, maxBoxesPerClass);
    auto *centerPointBoxVal = emitConstI32(builder, centerPointBox);
    auto *iouThresholdVal = emitConstF32(builder, iouThreshold);
    auto *scoreThresholdVal = emitConstF32(builder, scoreThreshold);

    auto *boxesDimVal = emitValueDims(builder, boxes);
    auto *scoreDimVal = emitValueDims(builder, scores);
    auto *indicesDimVal = emitValueDims(builder, indices);
    auto *boxesDimSizeVal = emitConstDimT(builder, boxes->dims().size());
    auto *scoresDimSizeVal = emitConstDimT(builder, scores->dims().size());
    auto *indicesDimSizeVal = emitConstDimT(builder, indices->dims().size());
    auto *isV4Val = emitConstI1(builder, isV4);

    auto *F = getFunction("nms", indices->getElementType());
    createCall(builder, F,
               {indicesPtr, numDetectedPtr, boxesPtr, boxesDimVal,
                boxesDimSizeVal, scoresPtr, scoreDimVal, scoresDimSizeVal,
                indicesDimVal, indicesDimSizeVal, centerPointBoxVal,
                maxBoxesPerClassVal, iouThresholdVal, scoreThresholdVal,
                isV4Val});
    break;
  }

  case Kinded::Kind::AudioSpectrogramInstKind: {
    auto *ASI = llvm::cast<AudioSpectrogramInst>(I);
    auto winOutScratch = ASI->getWinOutScratch();
    auto fftOutScratch = ASI->getFftOutScratch();
    auto spectrogram = ASI->getSpectrogram();
    auto input = ASI->getInput();
    auto window = ASI->getWindow();
    auto twiddleFactors = ASI->getTwiddleFactors();
    auto bitReverseIndices = ASI->getBitReverseIndices();
    auto complexToRealWeights = ASI->getComplexToRealWeights();
    int64_t windowSize = ASI->getWindowSize();
    int64_t windowStride = ASI->getWindowStride();
    bool magnitudeSquared = ASI->getMagnitudeSquared();

    auto *winOutScratchPtr = emitValueAddress(builder, winOutScratch);
    auto *fftOutScratchPtr = emitValueAddress(builder, fftOutScratch);
    auto *spectrogramPtr = emitValueAddress(builder, spectrogram);
    auto *inputPtr = emitValueAddress(builder, input);
    auto *windowPtr = emitValueAddress(builder, window);
    auto *twiddleFactorsPtr = emitValueAddress(builder, twiddleFactors);
    auto *bitReverseIndicesPtr = emitValueAddress(builder, bitReverseIndices);
    auto *complexToRealWeightsPtr =
        emitValueAddress(builder, complexToRealWeights);
    auto *spectrogramDimVal = emitValueDims(builder, spectrogram);
    auto *inputLengthVal = emitConstDimT(builder, input->size());
    auto *windowSizeVal = emitConstDimT(builder, windowSize);
    auto *windowStrideVal = emitConstDimT(builder, windowStride);
    auto *magnitudeSquaredVal = emitConstI1(builder, magnitudeSquared);

    auto *F = getFunction("audio_spectrogram", spectrogram->getElementType());
    createCall(builder, F,
               {winOutScratchPtr, fftOutScratchPtr, spectrogramPtr, inputPtr,
                windowPtr, twiddleFactorsPtr, bitReverseIndicesPtr,
                complexToRealWeightsPtr, spectrogramDimVal, inputLengthVal,
                windowSizeVal, windowStrideVal, magnitudeSquaredVal});
    break;
  }

  case Kinded::Kind::MFCCInstKind: {
    auto *MFCCI = llvm::cast<MFCCInst>(I);
    auto scratch = MFCCI->getScratch();
    auto coefficients = MFCCI->getCoefficients();
    auto spectrogram = MFCCI->getSpectrogram();
    auto melWeights = MFCCI->getMelWeights();
    auto melRanges = MFCCI->getMelRanges();
    auto dctMat = MFCCI->getDctMat();
    int64_t filterBankCount = MFCCI->getFilterBankCount();

    auto *scratchPtr = emitValueAddress(builder, scratch);
    auto *coefficientsPtr = emitValueAddress(builder, coefficients);
    auto *spectrogramPtr = emitValueAddress(builder, spectrogram);
    auto *melWeightsPtr = emitValueAddress(builder, melWeights);
    auto *melRangesPtr = emitValueAddress(builder, melRanges);
    auto *dctMatPtr = emitValueAddress(builder, dctMat);
    auto *coefficientsDimVal = emitValueDims(builder, coefficients);
    auto *spectrogramDimVal = emitValueDims(builder, spectrogram);
    auto *filterBankCountVal = emitConstDimT(builder, filterBankCount);

    auto *F = getFunction("mfcc", coefficients->getElementType());
    createCall(builder, F,
               {scratchPtr, coefficientsPtr, spectrogramPtr, melWeightsPtr,
                melRangesPtr, dctMatPtr, coefficientsDimVal, spectrogramDimVal,
                filterBankCountVal});
    break;
  }

  case Kinded::Kind::ConvertToInstKind: {
    auto *CTI = llvm::cast<ConvertToInst>(I);
    auto *input = CTI->getInput();
    auto *output = CTI->getResult();

    auto *inputVal = emitValueAddress(builder, input);
    auto *outptVal = emitValueAddress(builder, output);
    auto *dimsVal = emitValueDims(builder, output);
    auto *dimSizeVal = emitConstDimT(builder, output->dims().size());

    auto *F = getFunction("convertTo",
                          {output->getElementType(), input->getElementType()});

    createCall(builder, F, {outptVal, inputVal, dimsVal, dimSizeVal});
    break;
  }

  default:
    std::string sBuf;
    llvm::raw_string_ostream s(sBuf);
    I->dump(s);
    LOG(FATAL) << "Cannot select the instruction: " << s.str();
  }
}

unsigned LLVMIRGen::getTargetSizeTWidth() const {
  return getPointerNumBits(*TM_);
}

unsigned LLVMIRGen::getLibjitSizeTWidth() const {
  auto *sizeTVar = getModule().getGlobalVariable("libjit_sizeTVar",
                                                 /* allowInternal */ true);
  assert(sizeTVar && "libjit_sizeTVar is not found");
  return sizeTVar->getType()->getPointerElementType()->getIntegerBitWidth();
}

unsigned LLVMIRGen::getLibjitIntWidth() const {
  auto *intVar = getModule().getGlobalVariable("libjit_intVar",
                                               /* allowInternal */ true);
  assert(intVar && "libjit_intVar is not found");
  return intVar->getType()->getPointerElementType()->getIntegerBitWidth();
}

bool LLVMIRGen::isEligibleForSpecialization(const llvm::CallInst *call) {
  return true;
}

bool LLVMIRGen::canBePartOfDataParallelKernel(
    const glow::Instruction *I) const {
  return I->isDataParallel();
}

/// Extra bundle header file content with the IR instrumentation callback API.
static const char *instrumentIRApi =
    R"RAW(
// -----------------------------------------------------------------------------
// Callback function used for Glow IR instruction instrumentation:
// - This callback is called by the bundle BEFORE executing each instruction.
// - This callback must be defined by the bundle user application.
// ARGUMENTS:
//   id     - Instruction instance ID.
//   kind   - Instruction kind (type).
//   opInp  - Number of input operands.
//   opOut  - Number of output operands.
//   opAddr - Array with addresses for all operands. The addresses are listed
//            first for the input operands and then for the output operands.
//            The array contains opInp + opOut addresses.
//   opSize - Array with sizes (in bytes) for all operands. The sizes are listed
//            first for the input operands and then for the output operands.
//            The array contains opInp + opOut sizes.
// NOTES:
// - This callback should be used to dump only the input operands since the
//   output operands are not yet computed/written when this callback is used.
// - This callback uses C linkage therefore if the callback is implemented in a
//   .cpp file you must enclose the implementation in extern "C" {}.
// - Look in the metafile "instrument-ir.info" generated during compile-time
//   to see more information about the instrumented instructions.
// -----------------------------------------------------------------------------
void glow_instrument_before(int id, int kind, int opInp, int opOut, uint8_t **opAddr, int *opSize);

// -----------------------------------------------------------------------------
// Callback function used for Glow IR instruction instrumentation:
// - This callback is called by the bundle AFTER executing each instruction.
// - This callback must be defined by the bundle user application.
// ARGUMENTS:
//   id     - Instruction instance ID.
//   kind   - Instruction kind (type).
//   opInp  - Number of input operands.
//   opOut  - Number of output operands.
//   opAddr - Array with addresses for all operands. The addresses are listed
//            first for the input operands and then for the output operands.
//            The array contains opInp + opOut addresses.
//   opSize - Array with sizes (in bytes) for all operands. The sizes are listed
//            first for the input operands and then for the output operands.
//            The array contains opInp + opOut sizes.
// NOTES:
// - This callback should be used to dump only the output operands since some
//   of the input operands might have been overwritten for instructions which
//   perform in-place computation.
// - This callback uses C linkage therefore if the callback is implemented in a
//   .cpp file you must enclose the implementation in extern "C" {}.
// - Look in the metafile "instrument-ir.info" generated during compile-time
//   to see more information about the instrumented instructions.
// -----------------------------------------------------------------------------
void glow_instrument_after(int id, int kind, int opInp, int opOut, uint8_t **opAddr, int *opSize);
)RAW";

std::string LLVMIRGen::getBundleHeaderExtra() const {
  std::string headerExtra = "";
  // Print IR instrumentation callback API.
  if (printInstrumentIR_) {
    headerExtra += std::string(instrumentIRApi);
  }
  return headerExtra;
}
