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

#include "CPULLVMIRGen.h"

#include "glow/IR/Instrs.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"
#include "glow/Quantization/Base/Base.h"

using namespace glow;
using llvm::cast;

CPULLVMIRGen::CPULLVMIRGen(const IRFunction *F,
                           AllocationsInfo &allocationsInfo,
                           std::string mainEntryName, llvm::StringRef libjitBC)
    : LLVMIRGen(F, allocationsInfo, mainEntryName, libjitBC) {}

CPULLVMIRGen::CPULLVMIRGen(const IRFunction *F,
                           AllocationsInfo &allocationsInfo,
                           std::string mainEntryName, llvm::StringRef libjitBC,
                           llvm::ArrayRef<llvm::MemoryBufferRef> objectRegistry)
    : LLVMIRGen(F, allocationsInfo, mainEntryName, libjitBC, objectRegistry) {}

void CPULLVMIRGen::generateLLVMIRForModule(llvm::IRBuilder<> &builder) {
  // TODO: Add here any backend specific logic.
  LLVMIRGen::generateLLVMIRForModule(builder);
}

void CPULLVMIRGen::generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                          const glow::Instruction *I) {
  setCurrentDebugLocation(builder, I);
  assert(!canBePartOfDataParallelKernel(I) &&
         "data parallel instructions are not handled here");
  // Perform any backend-specific code generation here and delegate everything
  // else to LLVMIRGen.
  switch (I->getKind()) {
  case Kinded::Kind::CPUConvDKKC8InstKind: {
    auto *CI = cast<CPUConvDKKC8Inst>(I);
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

    size_t inChannels = src->dims()[3];
    size_t outChannels = dest->dims()[3];

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
           2 * depthStrips * numDepthRegs * 8 <= outChannels / CI->getGroup() &&
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
                filterDims, biasDims, kernels, strides, pads, group,
                pixelScanFirstVal, numDepthRegsVal, sizeGroupYVal,
                depthStripsVal});
    break;
  }
  default:
    LLVMIRGen::generateLLVMIRForInstr(builder, I);
  }
}

void CPULLVMIRGen::generateLLVMIRForDataParallelInstr(
    llvm::IRBuilder<> &builder, const glow::Instruction *I,
    llvm::Function *kernel, llvm::DenseMap<Value *, int> &bufferToArgNum,
    llvm::Value *loopCount) {
  setCurrentDebugLocation(builder, I);
  assert(canBePartOfDataParallelKernel(I) &&
         "Expected a data parallel instruction");
  // Perform any backend-specific code generation here and delegate everything
  // else to LLVMIRGen.
  switch (I->getKind()) {
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
      auto *stackedOpCall = createUncheckedCall(
          builder, F, {loopCount, val, lhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *val = emitConst(builder, V, lhs->getElementType());
      auto *stackedOpCall = createUncheckedCall(
          builder, F, {loopCount, val, lhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }

    break;
  }

  default:
    LLVMIRGen::generateLLVMIRForDataParallelInstr(builder, I, kernel,
                                                  bufferToArgNum, loopCount);
  }
}
