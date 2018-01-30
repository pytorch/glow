// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "JIT.h"

#include "glow/IR/Instrs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace glow;
using llvm::StringRef;
using llvm::dyn_cast;
using llvm::isa;

namespace {
/// Specialize taking into account only dimensions, but not the buffer
/// addresses.
static llvm::cl::opt<bool>
    jitSpecializeDims("jit-specialize-dims",
                      llvm::cl::desc("Create specialized functions for "
                                     "operations with constant dimensions"),
                      llvm::cl::init(true));

/// Specialize taking into account the whole set of arguments, including buffer
/// addresses.
static llvm::cl::opt<bool>
    jitSpecialize("jit-specialize",
                  llvm::cl::desc("Create specialized functions for operations "
                                 "with constant arguments"),
                  llvm::cl::init(false));
} // namespace

llvm::Value *JITBackend::emitValueAddress(llvm::IRBuilder<> &builder,
                                          glow::Value *val, ElemKind ptrTy) {
  assert(allocatedAddressed_.count(val) && "Value address was not allocated");
  void *ptr = allocatedAddressed_[val];
  auto *offset = emitConst(builder, (size_t)ptr);

  llvm::Type *T = nullptr;

  switch (ptrTy) {
  case ElemKind::FloatTy:
    T = llvm::Type::getFloatTy(ctx_)->getPointerTo();
    break;
  case ElemKind::Int8QTy:
    T = llvm::Type::getInt8Ty(ctx_)->getPointerTo();
    break;
  case ElemKind::IndexTy:
    T = builder.getIntNTy(sizeof(size_t) * 8)->getPointerTo();
    break;
  default:
    llvm_unreachable("Unimplemented");
    break;
  }

  return builder.CreateIntToPtr(offset, T);
}

llvm::Value *JITBackend::emitConstArray(llvm::IRBuilder<> &builder,
                                        llvm::ArrayRef<size_t> vals) {
  auto SizeTType = builder.getIntNTy(sizeof(size_t) * 8);
  std::vector<llvm::Constant *> elems;
  for (auto I : vals) {
    elems.push_back(llvm::ConstantInt::get(SizeTType, I));
  }
  auto *arr = llvm::ConstantArray::get(
      llvm::ArrayType::get(SizeTType, elems.size()), elems);

  auto *M = builder.GetInsertBlock()->getModule();

  auto *G = new llvm::GlobalVariable(*M, arr->getType(), true,
                                     llvm::GlobalValue::CommonLinkage, arr);
  return builder.CreateBitCast(G, SizeTType->getPointerTo());
}

llvm::Value *JITBackend::emitValueDims(llvm::IRBuilder<> &builder,
                                       glow::Value *val) {
  auto dims = val->dims();
  return emitConstArray(builder, dims);
}

llvm::Value *JITBackend::emitValueSize(llvm::IRBuilder<> &builder,
                                       glow::Value *val) {
  return builder.getIntN(sizeof(size_t) * 8, val->getType()->size());
}

llvm::Value *JITBackend::emitConst(llvm::IRBuilder<> &builder, float val) {
  return llvm::ConstantFP::get(llvm::Type::getFloatTy(ctx_), val);
}

llvm::Value *JITBackend::emitConst(llvm::IRBuilder<> &builder, size_t val) {
  return builder.getIntN(sizeof(size_t) * 8, val);
}

llvm::Function *JITBackend::getFunction(const std::string &name) {
  return llmodule_->getFunction(name);
}

llvm::Function *JITBackend::getOrCreateSpecializedFunction(
    llvm::Function *F, llvm::SmallVectorImpl<size_t> &constParams,
    std::function<void(llvm::IRBuilder<> &builder,
                       llvm::SmallVectorImpl<llvm::Value *> &args,
                       bool emitAllArgs)>
        constArgsEmitter) {
  // Bail if there is nothing to do.
  if (!jitSpecialize && !jitSpecializeDims)
    return F;

  // Produce a name of the specialized function.
  std::string name;
  llvm::raw_string_ostream sb{name};
  sb << F->getName();
  for (auto param : constParams) {
    sb << "_" << param;
  }
  auto &specializedName = sb.str();

  // Check if a specialization for these parameters exists already.
  auto *specializedF = llmodule_->getFunction(specializedName);
  if (specializedF)
    return specializedF;

  // Create a specialized function.
  // This function should have exactly the same type as the original function.
  // It should forward all its parameters.
  // The specialized function should be marked as noinline, to avoid code bloat.
  // FF.hasFnAttribute(llvm::Attribute::AttrKind::NoInline);
  specializedF = dyn_cast<llvm::Function>(
      llmodule_->getOrInsertFunction(specializedName, F->getFunctionType()));
  assert(specializedF && "Could not create a specialized function");
  // Specialization thunks should not be inlined.
  specializedF->addFnAttr(llvm::Attribute::AttrKind::NoInline);

  // Generate the code to forward the invocation to the original function.

  // Setup the entry basic block and initialize the IR builder.
  llvm::BasicBlock *entry_bb =
      llvm::BasicBlock::Create(ctx_, "entry", specializedF);
  llvm::IRBuilder<> builder(entry_bb);
  llvm::SmallVector<llvm::Value *, 16> args;

  if (!jitSpecialize) {
    // Forward all non-const buffers arguments.
    for (auto &arg : specializedF->args()) {
      if (!arg.getType()->isPointerTy())
        break;
      auto *elementTy = arg.getType()->getPointerElementType();
      if (!elementTy->isFloatTy())
        break;
      args.push_back(&arg);
    }
  }

  // Emit the rest of arguments, namely all constant arguments.
  constArgsEmitter(builder, args, jitSpecialize);

  // Forward all parameters.
  builder.CreateCall(F, args);
  builder.CreateRetVoid();
  DEBUG(llvm::errs() << "\n\nCreated specialized function " << specializedName
                     << "\n";
        specializedF->print(llvm::errs(), nullptr));
  return specializedF;
}

void JITBackend::generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                        glow::Instruction *I) {
  switch (I->getKind()) {
  case Kinded::Kind::SplatInstKind: {
    SplatInst *SI = llvm::cast<SplatInst>(I);
    auto *addr = emitValueAddress(builder, SI->getDest(), ElemKind::FloatTy);
    auto cnt = emitValueSize(builder, SI->getDest());
    auto *val = emitConst(builder, SI->getValue());
    auto *F = getFunction("splat_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {addr, cnt, val});
    break;
  }

  case Kinded::Kind::ElementMaxInstKind: {
    ElementMaxInst *EM = llvm::cast<ElementMaxInst>(I);
    auto *destPtr = emitValueAddress(builder, EM->getDest(), ElemKind::FloatTy);
    auto *LHSPtr = emitValueAddress(builder, EM->getLHS(), ElemKind::FloatTy);
    auto *RHSPtr = emitValueAddress(builder, EM->getRHS(), ElemKind::FloatTy);
    auto cnt = emitValueSize(builder, EM->getDest());
    auto *F = getFunction("elementmax_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, LHSPtr, RHSPtr, cnt});
    break;
  }

  case Kinded::Kind::ElementMinInstKind: {
    ElementMinInst *EM = llvm::cast<ElementMinInst>(I);
    auto *destPtr = emitValueAddress(builder, EM->getDest(), ElemKind::FloatTy);
    auto *LHSPtr = emitValueAddress(builder, EM->getLHS(), ElemKind::FloatTy);
    auto *RHSPtr = emitValueAddress(builder, EM->getRHS(), ElemKind::FloatTy);
    auto cnt = emitValueSize(builder, EM->getDest());
    auto *F = getFunction("elementmin_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, LHSPtr, RHSPtr, cnt});
    break;
  }

  case Kinded::Kind::ElementSelectInstKind: {
    ElementSelectInst *ES = llvm::cast<ElementSelectInst>(I);
    auto *destPtr = emitValueAddress(builder, ES->getDest(), ElemKind::FloatTy);
    auto *condPtr = emitValueAddress(builder, ES->getCond(), ElemKind::FloatTy);
    auto *LHSPtr = emitValueAddress(builder, ES->getLHS(), ElemKind::FloatTy);
    auto *RHSPtr = emitValueAddress(builder, ES->getRHS(), ElemKind::FloatTy);
    auto cnt = emitValueSize(builder, ES->getDest());
    auto *F = getFunction("elementselect_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, condPtr, LHSPtr, RHSPtr, cnt});
    break;
  }

  case Kinded::Kind::BatchedMatMulInstKind: {
    BatchedMatMulInst *BMM = llvm::cast<BatchedMatMulInst>(I);
    auto constArgsEmitter = [&](llvm::IRBuilder<> &builder,
                                llvm::SmallVectorImpl<llvm::Value *> &args,
                                bool emitAllArgs) {
      if (emitAllArgs) {
        // Emit arguments for buffers.
        auto *destPtr =
            emitValueAddress(builder, BMM->getDest(), ElemKind::FloatTy);
        auto *LHSPtr =
            emitValueAddress(builder, BMM->getLHS(), ElemKind::FloatTy);
        auto *RHSPtr =
            emitValueAddress(builder, BMM->getRHS(), ElemKind::FloatTy);

        args.push_back(destPtr);
        args.push_back(LHSPtr);
        args.push_back(RHSPtr);
      }
      // Emit arguments for dimensions.
      auto *destDims = emitValueDims(builder, BMM->getDest());
      auto *LHSDims = emitValueDims(builder, BMM->getLHS());
      auto *RHSDims = emitValueDims(builder, BMM->getRHS());
      args.push_back(destDims);
      args.push_back(LHSDims);
      args.push_back(RHSDims);
    };

    auto *F = getFunction("batchedmatmul_f");
    assert(F && "Unable to load the function");

    // Set of constant arguments.
    llvm::SmallVector<size_t, 16> constArgs{};
    if (jitSpecialize) {
      // Buffers have constant addresses, take them into account.
      constArgs.push_back((size_t)allocatedAddressed_[BMM->getDest()]);
      constArgs.push_back((size_t)allocatedAddressed_[BMM->getLHS()]);
      constArgs.push_back((size_t)allocatedAddressed_[BMM->getRHS()]);
    }
    constArgs.append(BMM->getDest()->dims().begin(),
                     BMM->getDest()->dims().end());
    constArgs.append(BMM->getLHS()->dims().begin(),
                     BMM->getLHS()->dims().end());
    constArgs.append(BMM->getRHS()->dims().begin(),
                     BMM->getRHS()->dims().end());

    // Get the specialized function.
    auto specializedF =
        getOrCreateSpecializedFunction(F, constArgs, constArgsEmitter);

    llvm::SmallVector<llvm::Value *, 16> fnArgs{};
    constArgsEmitter(builder, fnArgs, /* emitAllArgs */ true);

    builder.CreateCall(specializedF, fnArgs);
    break;
  }

  case Kinded::Kind::CopyInstKind: {
    CopyInst *CI = llvm::cast<CopyInst>(I);
    auto *destPtr = emitValueAddress(builder, CI->getDest(), ElemKind::Int8QTy);
    auto *srcPtr = emitValueAddress(builder, CI->getSrc(), ElemKind::Int8QTy);
    auto sizeInBytes = CI->getDest()->getType()->getSizeInBytes();
    auto *bytes = emitConst(builder, sizeInBytes);

    auto *F = getFunction("copy_buffer");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, srcPtr, bytes});
    break;
  }

  case Kinded::Kind::BatchedAddInstKind: {
    BatchedAddInst *BA = llvm::cast<BatchedAddInst>(I);
    auto *destPtr = emitValueAddress(builder, BA->getDest(), ElemKind::FloatTy);
    auto *batchPtr =
        emitValueAddress(builder, BA->getBatch(), ElemKind::FloatTy);
    auto *slicePtr =
        emitValueAddress(builder, BA->getSlice(), ElemKind::FloatTy);

    auto bdim = flattenCdr(BA->getBatch()->dims());
    auto *numSlice = emitConst(builder, bdim.first);
    auto *sliceSize = emitConst(builder, bdim.second);

    auto *F = getFunction("batchedadd_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, batchPtr, slicePtr, numSlice, sliceSize});
    break;
  }

  case Kinded::Kind::BatchedReduceAddInstKind: {
    BatchedReduceAddInst *BR = llvm::cast<BatchedReduceAddInst>(I);
    auto *destPtr = emitValueAddress(builder, BR->getDest(), ElemKind::FloatTy);
    auto *batchPtr =
        emitValueAddress(builder, BR->getBatch(), ElemKind::FloatTy);

    auto *destSize = emitConst(builder, BR->getDest()->getType()->size());
    auto bdim = flattenCdr(BR->getBatch()->dims());
    auto *numSlice = emitConst(builder, bdim.first);
    auto *sliceSize = emitConst(builder, bdim.second);

    auto *F = getFunction("batchedreduceadd_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, batchPtr, destSize, numSlice, sliceSize});
    break;
  }

  case Kinded::Kind::ConvolutionInstKind: {
    ConvolutionInst *CI = llvm::cast<ConvolutionInst>(I);
    auto constArgsEmitter = [&](llvm::IRBuilder<> &builder,
                                llvm::SmallVectorImpl<llvm::Value *> &args,
                                bool emitAllArgs) {
      if (emitAllArgs) {
        // Emit arguments for buffers.
        auto *srcPtr =
            emitValueAddress(builder, CI->getSrc(), ElemKind::FloatTy);
        auto *destPtr =
            emitValueAddress(builder, CI->getDest(), ElemKind::FloatTy);
        auto *filterPtr =
            emitValueAddress(builder, CI->getFilter(), ElemKind::FloatTy);
        auto *biasPtr =
            emitValueAddress(builder, CI->getBias(), ElemKind::FloatTy);
        args.push_back(srcPtr);
        args.push_back(destPtr);
        args.push_back(filterPtr);
        args.push_back(biasPtr);
      }
      // Emit arguments for dimensions.
      auto *srcDims = emitValueDims(builder, CI->getSrc());
      auto *destDims = emitValueDims(builder, CI->getDest());
      auto *filterDims = emitValueDims(builder, CI->getFilter());
      auto *biasDims = emitValueDims(builder, CI->getBias());

      auto *kernel = emitConst(builder, CI->getKernel());
      auto *pad = emitConst(builder, CI->getPad());
      auto *stride = emitConst(builder, CI->getStride());
      args.push_back(srcDims);
      args.push_back(destDims);
      args.push_back(filterDims);
      args.push_back(biasDims);
      args.push_back(kernel);
      args.push_back(pad);
      args.push_back(stride);
    };

    const char *kernelName = "convolution_f";

    // Use a special version of the kernel for the case where K (the depth of
    // the convolution) is a multiple of 4.
    if ((CI->getDest()->dims()[3] % 4) == 0) {
      kernelName = "convolution_f_unroll_k4";
    }

    auto *F = getFunction(kernelName);
    assert(F && "Unable to load the function");

    // Set of constant arguments.
    llvm::SmallVector<size_t, 16> constArgs{};

    if (jitSpecialize) {
      // Buffers have constant addresses, take them into account.
      constArgs.push_back((size_t)allocatedAddressed_[CI->getSrc()]);
      constArgs.push_back((size_t)allocatedAddressed_[CI->getDest()]);
      constArgs.push_back((size_t)allocatedAddressed_[CI->getFilter()]);
      constArgs.push_back((size_t)allocatedAddressed_[CI->getBias()]);
    }

    constArgs.append(CI->getSrc()->dims().begin(), CI->getSrc()->dims().end());
    constArgs.append(CI->getDest()->dims().begin(),
                     CI->getDest()->dims().end());
    constArgs.append(CI->getFilter()->dims().begin(),
                     CI->getFilter()->dims().end());
    constArgs.append(CI->getBias()->dims().begin(),
                     CI->getBias()->dims().end());
    constArgs.push_back(CI->getKernel());
    constArgs.push_back(CI->getPad());
    constArgs.push_back(CI->getStride());

    // Get the specialized function.
    auto *specializedF =
        getOrCreateSpecializedFunction(F, constArgs, constArgsEmitter);
    llvm::SmallVector<llvm::Value *, 16> fnArgs{};
    constArgsEmitter(builder, fnArgs, /* emitAllArgs */ true);

    builder.CreateCall(specializedF, fnArgs);
    break;
  }

  case Kinded::Kind::PoolMaxInstKind: {
    PoolMaxInst *PM = llvm::cast<PoolMaxInst>(I);
    auto *destPtr = emitValueAddress(builder, PM->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, PM->getSrc(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, PM->getDest());
    auto *srcDims = emitValueDims(builder, PM->getSrc());

    auto *kernel = emitConst(builder, PM->getKernel());
    auto *stride = emitConst(builder, PM->getStride());
    auto *pad = emitConst(builder, PM->getPad());

    auto *F = getFunction("pool_max_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(
        F, {srcPtr, destPtr, srcDims, destDims, pad, kernel, stride});
    break;
  }

  case Kinded::Kind::PoolAvgInstKind: {
    PoolAvgInst *PM = llvm::cast<PoolAvgInst>(I);
    auto *destPtr = emitValueAddress(builder, PM->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, PM->getSrc(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, PM->getDest());
    auto *srcDims = emitValueDims(builder, PM->getSrc());

    auto *kernel = emitConst(builder, PM->getKernel());
    auto *stride = emitConst(builder, PM->getStride());
    auto *pad = emitConst(builder, PM->getPad());

    auto *F = getFunction("pool_avg_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(
        F, {srcPtr, destPtr, srcDims, destDims, pad, kernel, stride});
    break;
  }

  case Kinded::Kind::SoftMaxInstKind: {
    SoftMaxInst *SM = llvm::cast<SoftMaxInst>(I);
    auto *destPtr = emitValueAddress(builder, SM->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, SM->getSrc(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, SM->getDest());
    auto *srcDims = emitValueDims(builder, SM->getSrc());

    auto *F = getFunction("softmax_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcPtr, destPtr, srcDims, destDims});
    break;
  }

  case Kinded::Kind::SigmoidInstKind: {
    SigmoidInst *SI = llvm::cast<SigmoidInst>(I);
    auto *destPtr = emitValueAddress(builder, SI->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, SI->getSrc(), ElemKind::FloatTy);
    auto *numElemVal = emitConst(builder, SI->getDest()->getType()->size());
    auto *F = getFunction("sigmoid_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcPtr, destPtr, numElemVal});
    break;
  }

  case Kinded::Kind::TanhInstKind: {
    TanhInst *TI = llvm::cast<TanhInst>(I);
    auto *destPtr = emitValueAddress(builder, TI->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, TI->getSrc(), ElemKind::FloatTy);
    auto *numElemVal = emitConst(builder, TI->getDest()->getType()->size());
    auto *F = getFunction("tanh_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcPtr, destPtr, numElemVal});
    break;
  }

  case Kinded::Kind::TransposeInstKind: {
    TransposeInst *TI = llvm::cast<TransposeInst>(I);
    auto *destPtr = emitValueAddress(builder, TI->getDest(), ElemKind::FloatTy);
    auto *srcPtr = emitValueAddress(builder, TI->getSrc(), ElemKind::FloatTy);
    auto *destDims = emitValueDims(builder, TI->getDest());
    auto *srcDims = emitValueDims(builder, TI->getSrc());

    // Convert the mask to size_t type.
    llvm::SmallVector<size_t, 6> shuffSizeT;
    for (auto D : TI->getShuffle()) {
      shuffSizeT.push_back((size_t)D);
    }

    auto *shuffle = emitConstArray(builder, shuffSizeT);
    auto *len = emitConst(builder, TI->getShuffle().size());

    auto *F = getFunction("transpose_f");
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {srcPtr, destPtr, srcDims, destDims, shuffle, len});
    break;
  }

  case Kinded::Kind::IntrinsicInstKind: {
    IntrinsicInst *II = llvm::cast<IntrinsicInst>(I);
    if (II->getIdentifier().equals("jit.max0")) {
      auto *dest = II->getOperand(0).first;
      auto *src = II->getOperand(1).first;
      auto *destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
      auto *LHSPtr = emitValueAddress(builder, src, ElemKind::FloatTy);
      auto cnt = emitValueSize(builder, dest);
      auto *F = getFunction("elementmax0_f");
      assert(F && "Unable to load the function");
      builder.CreateCall(F, {destPtr, LHSPtr, cnt});
      break;
    }

    llvm_unreachable("Unknown intrinsic");
  }

  case Kinded::Kind::ElementDivInstKind:
  case Kinded::Kind::ElementMulInstKind:
  case Kinded::Kind::ElementAddInstKind:
  case Kinded::Kind::ElementSubInstKind:
  case Kinded::Kind::ElementCmpLTEInstKind: {
    // Generate code for the op parameters.
    auto *dest = I->getOperand(0).first;
    auto *lhs = I->getOperand(1).first;
    auto *rhs = I->getOperand(2).first;
    auto *destPtr = emitValueAddress(builder, dest, ElemKind::FloatTy);
    auto *lhsPtr = emitValueAddress(builder, lhs, ElemKind::FloatTy);
    auto *rhsPtr = emitValueAddress(builder, rhs, ElemKind::FloatTy);
    auto numElem = dest->getType()->size();
    auto *numElemVal = emitConst(builder, numElem);

    // Select the correct kernel from the library.
    const char *funcName = "";
    switch (I->getKind()) {
    case Kinded::Kind::ElementDivInstKind:
      funcName = "element_div_f";
      break;
    case Kinded::Kind::ElementMulInstKind:
      funcName = "element_mul_f";
      break;
    case Kinded::Kind::ElementAddInstKind:
      funcName = "element_add_f";
      break;
    case Kinded::Kind::ElementSubInstKind:
      funcName = "element_sub_f";
      break;
    case Kinded::Kind::ElementCmpLTEInstKind:
      funcName = "element_cmp_lte_f";
      break;
    default:
      llvm_unreachable("Invalid node kind");
    }

    auto *F = getFunction(funcName);
    assert(F && "Unable to load the function");
    builder.CreateCall(F, {destPtr, lhsPtr, rhsPtr, numElemVal});
    break;
  }

    // Alloc and Dealloc instructions are handled by the memory allocator.
  case Kinded::Kind::AllocActivationInstKind:
  case Kinded::Kind::DeallocActivationInstKind:
  case Kinded::Kind::TensorViewInstKind:
    break;

  default:
    llvm_unreachable("ERROR: Cannot select the instruction.");
  }
}
