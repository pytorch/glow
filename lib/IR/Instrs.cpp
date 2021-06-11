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

#include "glow/IR/Instrs.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include "llvm/Support/Casting.h"

#include <cassert>

using namespace glow;
using llvm::cast;
using llvm::isa;

//===----------------------------------------------------------------------===//
//                      Instruction textual printers
//===----------------------------------------------------------------------===//

const char *WeightVar::getMutabilityStr(MutabilityKind kind) {
  const char *names[] = {"const", "mutable", nullptr};
  return names[static_cast<int>(kind)];
}

const char *WeightVar::getMutabilityStr() const {
  return getMutabilityStr(mut_);
}

void WeightVar::dump(llvm::raw_ostream &os) const {
  os << "%" << getName() << " = WeightVar ";
  os << *getType() << " " << getMutabilityStr();
}

//===----------------------------------------------------------------------===//
//                       Instruction verification
//===----------------------------------------------------------------------===//

void CopyInst::verify() const {
  auto *dest = getDest();
  auto *src = getSrc();
  (void)dest;
  (void)src;
  assert(dest->getType() == src->getType() && "Invalid type.");
}

void TensorViewInst::verify() const {
  assert(getSrc()->getType()->size() >= getType()->size() &&
         "TensorView view size should be no larger than Src size");
  assert(getSrc()->getElementType() == getType()->getElementType() &&
         "TensorView view element type should be the same as Src type");
  assert(getSrc()->getType()->dims().size() == getOffsets().size() &&
         "TensorView offsets should have the same number of dims as Src type "
         "shape");
}

void AllocActivationInst::verify() const {
  unsigned numDealloc = 0;
  for (const Use &U : getUsers()) {
    numDealloc += isa<DeallocActivationInst>(U.get());
  }

  // Make sure that there is exactly one user is a deallocation.
  assert(numDealloc == 1 && "Invalid number of tensor deallocation");
}

void DeallocActivationInst::verify() const {
  // The operand of this instruction needs to be an AllocActivationInst.
  assert(isa<AllocActivationInst>(getSrc()) && "Invalid operand");
}

void InsertTensorInst::verify() const {
  assert(getSrc()->getElementType() == getDest()->getElementType() &&
         "InsertTensor dest element type should be the same as Src type.");
  assert(getCount() > 0 && "Count must be non-zero.");
  assert(getAxis() >= 0 && getAxis() < getDest()->dims().size() &&
         "Axis must fit inside Dest dims.");
  assert(
      getDest()->getType()->dims().size() == getOffsets().size() &&
      "InsertTensor offsets should have the same number of dims as Dest type "
      "shape");
}

void ExtractTensorInst::verify() const {
  assert(getSrc()->getElementType() == getDest()->getElementType() &&
         "ExtractTensor dest element type should be the same as Src type.");
  assert(
      getSrc()->getType()->dims().size() == getOffsets().size() &&
      "ExtractTensor offsets should have the same number of dims as Src type "
      "shape");
}

static void verifyRelu(TypeRef srcTy, TypeRef destTy) {
  if (srcTy->isQuantizedType()) {
    assert(srcTy->isQuantizedType() == destTy->isQuantizedType() &&
           "Mismatching isQuantized");
    assert(srcTy->dims() == destTy->dims() && "Mismatching dimensions");
    assert(srcTy->getElementType() == destTy->getElementType() &&
           "Mismatching element type");
    return;
  }
  assert(destTy->isEqual(*srcTy) && "Mismatching types");
}

void ReluInst::verify() const {
  verifyRelu(getSrc()->getType(), getDest()->getType());
}

void ReluGradInst::verify() const {
  verifyRelu(getSrcGrad()->getType(), getDest()->getType());
  verifyRelu(getSrcGrad()->getType(), getDestGrad()->getType());
}

//===----------------------------------------------------------------------===//
//                       Instruction scratch requirements
//===----------------------------------------------------------------------===//
dim_t TopKInst::getScratchSize() const {
  // Allocate enough scratch space to hold N values and N indices.
  dim_t N = getInput()->dims().back();
  dim_t elemSize = getIndices()->getType()->getElementSize();
  return (2 * N * elemSize);
}

dim_t AudioSpectrogramInst::getWinOutScratchSize() const {
  dim_t spectrogramLen = getSpectrogram()->dims()[1];
  dim_t fftLen = (spectrogramLen - 1) * 2;
  return fftLen * sizeof(float);
}

dim_t AudioSpectrogramInst::getFftOutScratchSize() const {
  dim_t spectrogramLen = getSpectrogram()->dims()[1];
  dim_t fftLen = (spectrogramLen - 1) * 2;
  return (fftLen + 2) * sizeof(float);
}

dim_t MFCCInst::getScratchSize() const {
  return getFilterBankCount() * sizeof(float);
}

dim_t TFLiteDetectionPostProcessInst::getScratchSize() const {

  dim_t numBoxes = getAnchors()->dims()[0];
  dim_t numClasses = getNumClasses();
  dim_t maxDetections = getMaxDetections();
  dim_t maxDetectionsPerClass = getMaxDetectionsPerClass();

  dim_t scratchSize = 0;
  if (getRegularNMS()) {
    // Compute scratch size for regular NMS.
    scratchSize += numBoxes * sizeof(float);
    scratchSize += (numBoxes + maxDetections) * sizeof(int32_t);
    scratchSize += (numBoxes + maxDetections) * sizeof(float);
    scratchSize += (numBoxes + maxDetections) * sizeof(int32_t);
    scratchSize += std::min(numBoxes, maxDetectionsPerClass) * sizeof(float);
    scratchSize += numBoxes * sizeof(int32_t);
    scratchSize += numBoxes * sizeof(int32_t);
    scratchSize += numBoxes * sizeof(float);
    scratchSize += numBoxes * sizeof(int32_t);
  } else {
    // Compute scratch size for fast NMS.
    scratchSize += numBoxes * sizeof(float);
    scratchSize +=
        numBoxes * std::min(maxDetections, numClasses) * sizeof(int32_t);
    scratchSize += numBoxes * sizeof(int32_t);
    scratchSize += numBoxes * sizeof(int32_t);
    scratchSize += numBoxes * sizeof(float);
    scratchSize += numBoxes * sizeof(int32_t);
  }
  return scratchSize;
}
