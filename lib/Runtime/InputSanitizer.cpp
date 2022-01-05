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

#include "glow/Runtime/InputSanitizer.h"
#include "glow/Flags/Flags.h"

#include <folly/Random.h>
#include <glog/logging.h>
#include <llvm/Support/Casting.h>

namespace glow {
namespace runtime {

namespace {

template <class T>
static Error sanitizeIndices(const Tensor *indicesTensor, size_t tableHeight,
                             llvm::StringRef tensorName) {
  auto indices = indicesTensor->getHandle<T>();
  size_t indicesLen = indices.getRealNumElements();
  // indices in [0, tableHeight)
  for (auto i = 0; i < indicesLen; i++) {
    RETURN_ERR_IF_NOT(indices.raw(i) >= 0 && indices.raw(i) < tableHeight,
                      "Indices sanitization failed on tensor " +
                          tensorName.str() + ": index " +
                          std::to_string(indices.raw(i)) + " at pos " +
                          std::to_string(i) + " is out of range [0, " +
                          std::to_string(tableHeight) + ")");
  }

  return Error::success();
}

template <class T>
static Error sanitizeLengths(const Tensor *lengthsTensor,
                             const size_t indicesLen,
                             llvm::StringRef tensorName) {
  auto lengths = lengthsTensor->getHandle<T>();

  size_t totalLensSum = 0;
  for (auto i = 0; i < lengths.getRealNumElements(); ++i) {
    auto length = lengths.raw(i);
    RETURN_ERR_IF_NOT(length >= 0,
                      "SLS lengths sanitization failed on tensor " +
                          tensorName.str() + ": length " +
                          std::to_string(length) + " at pos " +
                          std::to_string(i) + " is negative");
    totalLensSum += length;
  }

  RETURN_ERR_IF_NOT(
      indicesLen == totalLensSum,
      strFormat("SLS lengths sanitization failed on tensor %s: indices "
                "length %lu is not equal to sum of lengths %lu",
                tensorName.str().c_str(), indicesLen, totalLensSum));

  return Error::success();
}

template <class T>
static Error sanitizeOffsets(const Tensor *offsetsTensor,
                             const size_t numberOfIndices,
                             llvm::StringRef tensorName) {
  auto offsets = offsetsTensor->getHandle<T>();

  RETURN_ERR_IF_NOT(offsets.raw(0) == 0,
                    "EBB offsets sanitization failed on tensor " +
                        tensorName.str() + ": the first offset is not zero " +
                        std::to_string(offsets.raw(0)));

  bool zeroTensor = true;
  size_t offsetsLen = offsets.getRealNumElements();
  for (auto i = 0; i < offsetsLen - 1; i++) {
    RETURN_ERR_IF_NOT(offsets.raw(i) <= offsets.raw(i + 1),
                      "EBB offsets sanitization failed on tensor " +
                          tensorName.str() + ": decreasing offsets " +
                          std::to_string(offsets.raw(i)) + " and " +
                          std::to_string(offsets.raw(i + 1)) + " at pos " +
                          std::to_string(i));

    if (zeroTensor && offsets.raw(i + 1) != 0) {
      zeroTensor = false;
    }
  }

  size_t lastOffset = offsets.raw(offsetsLen - 1);
  RETURN_ERR_IF_NOT(
      zeroTensor || lastOffset == numberOfIndices,
      strFormat("EBB offsets sanitization failed on tensor %s: "
                "the last offset %lu is not equal to the number of indices %lu",
                tensorName.str().c_str(), lastOffset, numberOfIndices));

  return Error::success();
}

} // namespace

//
// SparseLengthsSum input sanitization
//
SparseLengthsSumInputSanitizer::SparseLengthsSumInputSanitizer(
    const size_t tableHeight, Placeholder *indicesPH, Placeholder *weightsPH,
    Placeholder *lengthsPH)
    : tableHeight_{tableHeight}, indicesPH_{indicesPH}, weightsPH_{weightsPH},
      lengthsPH_{lengthsPH} {}

Error SparseLengthsSumInputSanitizer::sanitize(
    const PlaceholderBindings &bindings) {
  auto *indices = bindings.get(indicesPH_);

  // Either a constant or some node internal to the function, skip
  if (indices == nullptr) {
    return Error::success();
  }

  size_t indicesLen = indices->getRealNumElements();

  if (weightsPH_) {
    auto *weights = bindings.get(weightsPH_);
    // If this is a weigthed one and the placeholder is real (not a constant
    // or internal to the function, then sanitize
    if (weights != nullptr) {
      size_t weightsLen = weights->getRealNumElements();
      RETURN_ERR_IF_NOT(
          indicesLen == weightsLen,
          strFormat("SLS weights sanitization failed on %s: number of indices "
                    "%lu is not equal to number of weights %lu",
                    weightsPH_->getName().str().c_str(), indicesLen,
                    weightsLen));
    }
  }

  // Sanitize indices
  if (indices->getElementType() == ElemKind::Int64ITy) {
    RETURN_IF_ERR(
        sanitizeIndices<int64_t>(indices, tableHeight_, indicesPH_->getName()));
  } else if (indices->getElementType() == ElemKind::Int32ITy) {
    RETURN_IF_ERR(
        sanitizeIndices<int32_t>(indices, tableHeight_, indicesPH_->getName()));
  } else {
    return MAKE_ERR(strFormat(
        "SLS indices sanitization failed on tensor %s: unsupported "
        "element type %s",
        indicesPH_->getName().str().c_str(),
        Type::getElementName(indices->getElementType()).str().c_str()));
  }

  // Sanitize SLS lengths
  auto *lengths = bindings.get(lengthsPH_);

  // Either a constant or some node internal to the function, skip
  if (lengths == nullptr) {
    return Error::success();
  }

  if (lengths->getElementType() == ElemKind::Int32ITy) {
    RETURN_IF_ERR(
        sanitizeLengths<int32_t>(lengths, indicesLen, lengthsPH_->getName()));
  } else if (lengths->getElementType() == ElemKind::Int64ITy) {
    RETURN_IF_ERR(
        sanitizeLengths<int64_t>(lengths, indicesLen, lengthsPH_->getName()));
  } else {
    return MAKE_ERR(strFormat(
        "SLS lengths sanitization failed on tensor %s: unsupported "
        "element type %s",
        lengthsPH_->getName().str().c_str(),
        Type::getElementName(lengths->getElementType()).str().c_str()));
  }

  return Error::success();
}

std::string SparseLengthsSumInputSanitizer::toString() {
  std::ostringstream ss;
  ss << "SparseLengthsSumInputSanitizer[";
  ss << "tableHeight=" << tableHeight_;
  ss << ", indices=";
  if (indicesPH_) {
    ss << indicesPH_->getName().str();
  }
  ss << ", weigths=";
  if (weightsPH_) {
    ss << weightsPH_->getName().str();
  }
  ss << ", lengths=";
  if (lengthsPH_) {
    ss << lengthsPH_->getName().str();
  }
  ss << "]";
  return ss.str();
}

//
// EmbeddingBag input sanitization
//
EmbeddingBagInputSanitizer::EmbeddingBagInputSanitizer(size_t tableHeight,
                                                       Placeholder *indicesPH,
                                                       Placeholder *weightsPH,
                                                       Placeholder *offsetsPH)
    : tableHeight_{tableHeight}, indicesPH_{indicesPH}, weightsPH_{weightsPH},
      offsetsPH_{offsetsPH} {}

Error EmbeddingBagInputSanitizer::sanitize(
    const PlaceholderBindings &bindings) {
  auto *indices = bindings.get(indicesPH_);

  // Either a constant or some node internal to the function, skip
  if (indices == nullptr) {
    return Error::success();
  }

  size_t indicesLen = indices->getRealNumElements();

  if (weightsPH_) {
    auto *weights = bindings.get(weightsPH_);
    // If this is a weigthed one and the placeholder is real (not a constant
    // or internal to the function, then sanitize
    if (weights != nullptr) {
      size_t weightsLen = weights->getRealNumElements();
      RETURN_ERR_IF_NOT(
          indicesLen == weightsLen,
          strFormat("EBB weights sanitization failed on %s: number of indices "
                    "%lu is not equal to number of weights %lu",
                    weightsPH_->getName().str().c_str(), indicesLen,
                    weightsLen));
    }
  }

  // Sanitize indices
  if (indices->getElementType() == ElemKind::Int64ITy) {
    RETURN_IF_ERR(
        sanitizeIndices<int64_t>(indices, tableHeight_, indicesPH_->getName()));
  } else if (indices->getElementType() == ElemKind::Int32ITy) {
    RETURN_IF_ERR(
        sanitizeIndices<int32_t>(indices, tableHeight_, indicesPH_->getName()));
  } else {
    return MAKE_ERR(strFormat(
        "EBB indices sanitization failed on tensor %s: unsupported "
        "element type %s",
        indicesPH_->getName().str().c_str(),
        Type::getElementName(indices->getElementType()).str().c_str()));
  }

  // Sanitize offsets
  auto *offsets = bindings.get(offsetsPH_);

  // Either a constant or some node internal to the function, skip
  if (offsets == nullptr) {
    return Error::success();
  }

  if (offsets->getElementType() == ElemKind::Int32ITy) {
    RETURN_IF_ERR(
        sanitizeOffsets<int32_t>(offsets, indicesLen, offsetsPH_->getName()));
  } else if (offsets->getElementType() == ElemKind::Int64ITy) {
    RETURN_IF_ERR(
        sanitizeOffsets<int64_t>(offsets, indicesLen, offsetsPH_->getName()));
  } else {
    return MAKE_ERR(strFormat(
        "EBB offsets sanitization failed on tensor %s: unsupported "
        "element type %s",
        offsetsPH_->getName().str().c_str(),
        Type::getElementName(offsets->getElementType()).str().c_str()));
  }

  return Error::success();
}

std::string EmbeddingBagInputSanitizer::toString() {
  std::ostringstream ss;
  ss << "EmbeddingBagInputSanitizer[";
  ss << "tableHeight=" << tableHeight_;
  ss << ", indices=";
  if (indicesPH_) {
    ss << indicesPH_->getName().str();
  }
  ss << ", weigths=";
  if (weightsPH_) {
    ss << weightsPH_->getName().str();
  }
  ss << ", offsets=";
  if (offsetsPH_) {
    ss << offsetsPH_->getName().str();
  }
  ss << "]";
  return ss.str();
}

//
// Public utility functions
//
std::vector<InputSanitizerPtr> getInputSanitizers(const Function &function) {
  std::vector<InputSanitizerPtr> result;

  for (const auto &node : function.getNodes()) {
    if (auto *SLS =
            llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
                &node)) {
      VLOG(1) << SLS->getIndices() << " " << SLS->getWeights() << " "
              << SLS->getLengths() << " " << SLS->getData().dims()[0];
      result.push_back(std::make_shared<SparseLengthsSumInputSanitizer>(
          SLS->getData().dims()[0],
          llvm::dyn_cast<Placeholder>(SLS->getIndices()),
          llvm::dyn_cast<Placeholder>(SLS->getWeights()),
          llvm::dyn_cast<Placeholder>(SLS->getLengths())));
    } else if (auto *SLS =
                   llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsSumNode>(
                       &node)) {
      VLOG(1) << SLS->getIndices() << " " << SLS->getLengths() << " "
              << SLS->getData().dims()[0];
      result.push_back(std::make_shared<SparseLengthsSumInputSanitizer>(
          SLS->getData().dims()[0],
          llvm::dyn_cast<Placeholder>(SLS->getIndices()),
          /* weights */ nullptr,
          llvm::dyn_cast<Placeholder>(SLS->getLengths())));
    } else if (auto *SLS = llvm::dyn_cast<SparseLengthsSumNode>(&node)) {
      VLOG(1) << SLS->getIndices() << " " << SLS->getLengths() << " "
              << SLS->getData().dims()[0];
      result.push_back(std::make_shared<SparseLengthsSumInputSanitizer>(
          SLS->getData().dims()[0],
          llvm::dyn_cast<Placeholder>(SLS->getIndices()),
          /* weights */ nullptr,
          llvm::dyn_cast<Placeholder>(SLS->getLengths())));
    } else if (auto *SLS =
                   llvm::dyn_cast<SparseLengthsWeightedSumNode>(&node)) {
      VLOG(1) << SLS->getIndices() << " " << SLS->getWeights() << " "
              << SLS->getLengths() << " " << SLS->getData().dims()[0];
      result.push_back(std::make_shared<SparseLengthsSumInputSanitizer>(
          SLS->getData().dims()[0],
          llvm::dyn_cast<Placeholder>(SLS->getIndices()),
          llvm::dyn_cast<Placeholder>(SLS->getWeights()),
          llvm::dyn_cast<Placeholder>(SLS->getLengths())));
    } else if (auto *EBB = llvm::dyn_cast<EmbeddingBagNode>(&node)) {
      VLOG(1) << EBB->getIndices() << " " << EBB->getOffsets() << " "
              << EBB->getData().dims()[0];
      result.push_back(std::make_shared<EmbeddingBagInputSanitizer>(
          EBB->getData().dims()[0],
          llvm::dyn_cast<Placeholder>(EBB->getIndices()),
          /* weights */ nullptr,
          llvm::dyn_cast<Placeholder>(EBB->getOffsets())));
    } else if (auto *EBB =
                   llvm::dyn_cast<EmbeddingBagByteRowwiseOffsetsNode>(&node)) {
      VLOG(1) << EBB->getIndices() << " " << EBB->getWeights() << " "
              << EBB->getOffsets() << " " << EBB->getData().dims()[0];
      result.push_back(std::make_shared<EmbeddingBagInputSanitizer>(
          EBB->getData().dims()[0],
          llvm::dyn_cast<Placeholder>(EBB->getIndices()),
          llvm::dyn_cast<Placeholder>(EBB->getWeights()),
          llvm::dyn_cast<Placeholder>(EBB->getOffsets())));
    }
  }

  return result;
}

Error sanitizeInputs(const std::vector<InputSanitizerPtr> &sanitizers,
                     const PlaceholderBindings &bindings) {
  if (flags::SanitizeInputsPercent == 0 ||
      folly::Random::rand32() % 100 > flags::SanitizeInputsPercent) {
    return Error::success();
  }

  for (auto &sanitizer : sanitizers) {
    RETURN_IF_ERR(sanitizer->sanitize(bindings));
  }

  return Error::success();
}

} // namespace runtime
} // namespace glow
