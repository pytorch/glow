// Copyright 2017 Facebook Inc.  All Rights Reserved.

#define DEBUG_TYPE "jit"
#include "JIT.h"

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace glow;
using llvm::StringRef;
using llvm::dyn_cast;
using llvm::isa;

static llvm::cl::opt<bool>
    dumpIR("dump-llvm-ir",
           llvm::cl::desc("Dump the LLVM-IR of the jitted code"),
           llvm::cl::init(false));

static llvm::cl::opt<bool>
    dumpJitAsm("dump-llvm-asm",
               llvm::cl::desc("Dump the textual assembly of the jitted code"),
               llvm::cl::init(false));

JITBackend::JITBackend(IRFunction *M) : M_(M) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  JIT_ = llvm::make_unique<llvm::orc::GlowJIT>();
}

JITBackend::~JITBackend() { clear(); }

void JITBackend::clear() { M_->clear(); }

// Search for the standard library bitcode file on disk and load it into an
// LLVM module. We search for the standard library around the current executable
// and also in the current directory.
static std::unique_ptr<llvm::Module> loadStandardLibrary(llvm::LLVMContext *ctx,
                                                         StringRef filename) {
  using llvm::sys::path::append;
  using llvm::sys::path::parent_path;

  llvm::SMDiagnostic Err;
  auto mainExec =
      llvm::sys::fs::getMainExecutable(nullptr, (void *)&loadStandardLibrary);
  StringRef basePath = parent_path(mainExec);

  for (int i = 0; i < 3; i++) {
    llvm::SmallString<256> libPath(basePath);
    append(libPath, filename);
    if (llvm::sys::fs::exists(libPath)) {
      return llvm::parseIRFile(libPath, Err, *ctx);
    }

    basePath = parent_path(basePath);
  }

  return llvm::parseIRFile(filename, Err, *ctx);
}

void JITBackend::init() {
  // Load the jit library as a new module.
  llmodule_ = loadStandardLibrary(&ctx_, "libjit.bc");
  GLOW_ASSERT(llmodule_.get() && "Unable to load the JIT library.");

  // Assign the target information to the module.
  llmodule_->setDataLayout(JIT_->getTargetMachine().createDataLayout());

  // Create the 'main' function into the LLVM module.
  llvm::Type *void_type = llvm::Type::getVoidTy(ctx_);
  llvm::FunctionType *jit_func_type =
      llvm::FunctionType::get(void_type, {}, false);
  func_ = llvm::Function::Create(jit_func_type, llvm::Function::ExternalLinkage,
                                 "main", llmodule_.get());

  // Setup the entry basic block and initialize the IR builder.
  llvm::BasicBlock *entry_bb = llvm::BasicBlock::Create(ctx_, "entry", func_);
  llvm::IRBuilder<> builder(entry_bb);

  allocateActivationsAndWeights();

  // For each instruction in the module:
  for (auto &I : M_->getInstrs()) {
    generateLLVMIRForInstr(builder, I);
  }

  // Terminate the function.
  builder.CreateRetVoid();
  assert(!llvm::verifyFunction(*func_, &llvm::errs()) && "Verification failed");

  // Optimize the module.
  optimizeLLVMModule(func_, JIT_->getTargetMachine());
  // And pass the ownership to the JIT.

  if (dumpIR) {
    llmodule_->print(llvm::outs(), nullptr);
  }

  if (dumpJitAsm) {
    llvm::SmallVector<char, 0> asmBuffer;
    llvm::raw_svector_ostream asmStream(asmBuffer);
    llvm::legacy::PassManager PM;
    JIT_->getTargetMachine().addPassesToEmitFile(
        PM, asmStream, llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
    PM.run(*llmodule_);
    llvm::outs() << asmStream.str();
  }

  JIT_->addModule(std::move(llmodule_));
}

void JITBackend::doForwardPass(bool isTrain) {
  auto sym = JIT_->findSymbol("main");
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

void JITBackend::allocateActivationsAndWeights() {
  // Use a memory allocator with no upper bound on how much memory we can
  // allocate.
  MemoryAllocator allocator(0);
  allocatedAddressed_.clear();

  // Maps activations and views to some offset within the heap.
  llvm::DenseMap<Value *, size_t> activationAddr;

  // Register the addresses of the tensor payload.
  for (auto &v : M_->getGraph()->getParent().getVars()) {
    auto *w = M_->getWeightForNode(v);
    allocatedAddressed_[w] = v->getPayload().getUnsafePtr();
  }

  // Assign device-space addresses to the activations.
  for (auto &I : M_->getInstrs()) {
    if (auto *A = llvm::dyn_cast<AllocActivationInst>(I)) {
      auto numBytes = I->getType()->getSizeInBytes();
      size_t addr = allocator.allocate(numBytes);
      assert(!activationAddr.count(A) && "Allocation already made!");
      activationAddr[A] = addr;
      continue;
    }

    if (auto *TV = llvm::dyn_cast<TensorViewInst>(I)) {
      // If the source is heap allocated then add it to the 'heap' map.
      if (activationAddr.count(TV->getSrc())) {
        assert(!activationAddr.count(TV) && "Allocation already made!");
        assert(activationAddr.count(TV->getSrc()) && "Can't find TV source");
        activationAddr[TV] = activationAddr[TV->getSrc()];
      } else {
        assert(allocatedAddressed_.count(TV->getSrc()) &&
               "Can't find the Weight address");
        allocatedAddressed_[TV] = allocatedAddressed_[TV->getSrc()];
      }
      continue;
    }

    if (auto *D = llvm::dyn_cast<DeallocActivationInst>(I)) {
      auto *A = D->getAlloc();
      assert(activationAddr.count(A) && "Invalid deallocation!");
      allocator.deallocate(activationAddr[A]);
      continue;
    }
  }

  // Allocate the heap to match the max memory usage.
  heap_.resize(allocator.getMaxMemoryUsage());

  // Register specific addresses within the heap to activations.
  for (auto &A : activationAddr) {
    allocatedAddressed_[A.first] = &heap_[0] + A.second;
  }

  DEBUG(for (auto &A
             : allocatedAddressed_) {
    llvm::errs() << "Allocated " << A.first->getName()
                 << " size: " << A.first->getType()->getSizeInBytes()
                 << "  address range:  [" << allocatedAddressed_[A.first]
                 << ", "
                 << static_cast<void *>(
                        (static_cast<char *>(allocatedAddressed_[A.first]) +
                         A.first->getType()->getSizeInBytes()))
                 << "]\n";
  });
}
