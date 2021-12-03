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

#include "Loader.h"

#include "glow/Base/Image.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cfloat>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

#include "ExecutorCore.h"
#include "ExecutorCoreHelperFunctions.h"

using namespace glow;

namespace {

/// Image loader options.
llvm::cl::OptionCategory imageClassifierCat("Image Loader Options");

llvm::cl::opt<unsigned> labelOffset(
    "label-offset",
    llvm::cl::desc("Label offset for TF ONNX models with 1001 classes"),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(imageClassifierCat));

llvm::cl::opt<bool>
    computeSoftmax("compute-softmax",
                   llvm::cl::desc("Compute softmax of the network output"),
                   llvm::cl::Optional, llvm::cl::init(false),
                   llvm::cl::cat(imageClassifierCat));

llvm::cl::opt<unsigned>
    topKCount("topk",
              llvm::cl::desc("Number of highest likelihood labels to print and "
                             "match the correspondent expected-labels"),
              llvm::cl::Optional, llvm::cl::init(1),
              llvm::cl::cat(imageClassifierCat));

llvm::cl::list<unsigned> expectedMatchingLabels(
    "expected-labels",
    llvm::cl::desc("The comma delimited list of the matching labels"),
    llvm::cl::value_desc("int"), llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(imageClassifierCat));
} // unnamed namespace

/// A pair representing a float and the index where the float was found.
using FloatIndexPair = std::pair<float, size_t>;

/// Given a Handle \p H of a 1D tensor with float elements, \returns the top K
/// (topKCount) [float, index] pairs, i.e. the pairs with the highest floats.
template <typename ElemTy>
static std::vector<FloatIndexPair> getTopKPairs(Handle<ElemTy> H) {
  DCHECK_LE(topKCount, H.size()) << "Function requires k < number of labels.";
  DCHECK_EQ(H.dims().size(), 1) << "H must be a Handle of a 1d Tensor.";

  // Use a priority queue of pairs of floats (probabilities) to size_t (indices)
  // to determine the top K pairs, and then return the indices from it.
  std::priority_queue<FloatIndexPair, std::vector<FloatIndexPair>,
                      std::greater<FloatIndexPair>>
      topKQueue;

  // Loop over all the probabilites, finding the highest k probability pairs.
  for (dim_t i = 0, e = H.size(); i < e; i++) {
    float currProbability = H.at({i});
    if (topKQueue.size() < topKCount) {
      // Always push the first k elements.
      topKQueue.push(std::make_pair(currProbability, i));
    } else if (topKQueue.top().first < currProbability) {
      // If the lowest element has lower probability than the current, then pop
      // the lowest and insert the current pair.
      topKQueue.pop();
      topKQueue.push(std::make_pair(currProbability, i));
    }
  }

  // We now have the top K pairs in reverse order.
  std::vector<FloatIndexPair> res(topKCount);
  for (size_t i = 0; i < topKCount; i++) {
    res[topKCount - i - 1] = topKQueue.top();
    topKQueue.pop();
  }

  return res;
}

/// Print out the top K pairs to stdout, which were passed in via \p topKPairs.
static void printTopKPairs(const std::vector<FloatIndexPair> &topKPairs) {
  for (size_t i = 0; i < topKPairs.size(); i++) {
    // Some models are trained with more classes. E.g. Some imagenet models
    // exported from TensorFlow have 1 extra "neutral" class.
    const size_t label = topKPairs[i].second - labelOffset;
    // Tab out the label so it aligns nicely with Label-K1.
    if (i != 0) {
      llvm::outs() << "\t\t\t\t\t";
    }
    llvm::outs() << "\tLabel-K" << i + 1 << ": " << label << " (probability: "
                 << llvm::format("%0.4f", topKPairs[i].first) << ")\n";
  }
}

/// Checks if \p topKPairs have the index that matches the provided index,
/// \returns 0 on success and 1 if mismatches found.
static int checkExpectedLabel(llvm::ArrayRef<FloatIndexPair> topKPairs,
                              llvm::StringRef fileName,
                              unsigned expectedCategoryIndex) {
  // Loop through pairs and try to find a matching label.
  for (const auto &p : topKPairs) {
    if (p.second - labelOffset == expectedCategoryIndex) {
      return 0;
    }
  }

  llvm::outs() << " File: " << fileName
               << " doesn't match index: " << expectedCategoryIndex
               << " in the top " << topKPairs.size() << " pairs\n";

  return 1;
}

/// Apply the softmax function to the given handle.
template <typename ElemTy> static void applySoftmax(Handle<ElemTy> H) {
  DCHECK_EQ(H.dims().size(), 1) << "H must be a Handle of a 1d Tensor.";
  float denominator = 0.0f;

  for (auto elem : H) {
    denominator += std::exp(static_cast<float>(elem));
  }

  for (auto &elem : H) {
    elem = std::exp(static_cast<float>(elem)) / denominator;
  }
}

/// Given the output Softmax Tensor \p SMT and \p imageList, prints the
/// results of inference and returns number of incorrect predictions,
/// \returns the number of found mismatches.
template <typename ElemTy>
static int processAndPrintResultsImpl(Tensor *SMT,
                                      llvm::ArrayRef<std::string> imageList) {
  // Softmax should have at least two dimensions: batchSize (first dimension),
  // numLabels (any other dimension), and optionally - 1 in all other
  // dimensions. The value of numLabels should be greater than 1.
  DCHECK_GE(SMT->dims().size(), 2) << "Softmax should have at least 2 dims.";
  const dim_t batchSize = SMT->dims()[0];
  DCHECK_EQ(batchSize, imageList.size())
      << "Softmax batch size must equal the input number of images.";
  size_t labelsDim = 0;
  for (size_t i = 1; i < SMT->dims().size(); i++) {
    if (SMT->dims()[i] > 1) {
      DCHECK_EQ(labelsDim, 0) << "More than one dimension of size > 1?";
      labelsDim = i;
    }
  }
  DCHECK_NE(labelsDim, 0) << "Labels dimension not found!";
  const dim_t numLabels = SMT->dims()[labelsDim];
  // Get a view with canonical layout {batches, labels}.
  Tensor canonical = SMT->getUnowned({batchSize, numLabels});
  SMT = &canonical;

  std::vector<dim_t> sliceOffset(SMT->dims().size(), 0);

  int retVal = 0;
  for (unsigned i = 0; i < imageList.size(); i++) {
    const auto &fileName = imageList[i];
    if (topKCount) {
      llvm::outs() << " File: " << fileName;
    }

    // batchSize is the first dimension, so update it to get the next slice.
    sliceOffset[0] = i;
    Tensor slice = SMT->getUnowned({numLabels}, sliceOffset);
    auto SH = slice.getHandle<ElemTy>();

    if (computeSoftmax) {
      applySoftmax(SH);
    }

    if (topKCount) {
      auto topKPairs = getTopKPairs(SH);
      printTopKPairs(topKPairs);
      if (!expectedMatchingLabels.empty()) {
        retVal +=
            checkExpectedLabel(topKPairs, fileName, expectedMatchingLabels[i]);
      }
    }
  }

  return retVal;
}

class ImageClassifierProcessResult : public PostProcessOutputDataExtension {
public:
  int processOutputs(const llvm::StringMap<Placeholder *> &PHM,
                     PlaceholderBindings &bindings,
                     VecVecRef<std::string> imageList) override;
};

/// Given the output PlaceHolder StringMap \p PHM, of size 1, from SoftMax and
/// \p functionName, switch between the correct element type to print the
/// results of inference as contained in \p PHM, \returns the number of found
/// mismatches.
int ImageClassifierProcessResult::processOutputs(
    const llvm::StringMap<Placeholder *> &PHM, PlaceholderBindings &bindings,
    VecVecRef<std::string> imageList) {

  if (profilingGraph()) {
    LOG(INFO) << "Graph profiling is ON. Processing of output is disabled.";
    return 0;
  }

  Placeholder *phOut = getOutputForPostProcessing(PHM);
  if (!phOut) {
    return 0;
  }

  auto *SMT = bindings.get(PHM.begin()->second);
  switch (SMT->getElementType()) {
  case ElemKind::FloatTy:
    return processAndPrintResultsImpl<float>(SMT, imageList[0]);
  case ElemKind::Float16Ty:
    return processAndPrintResultsImpl<float16_t>(SMT, imageList[0]);
  case ElemKind::BFloat16Ty:
    return processAndPrintResultsImpl<bfloat16_t>(SMT, imageList[0]);
  default:
    llvm_unreachable("Type not supported");
  }
  return 0;
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  if (!expectedMatchingLabels.empty()) {
    // The number of category indices must match the number of files.
    if (expectedMatchingLabels.size() != inputImageFilenamesOpt[0].size()) {
      llvm::errs() << "Number of matching indices: "
                   << expectedMatchingLabels.size()
                   << " doesn't match the number of files: "
                   << inputImageFilenamesOpt[0].size() << "\n";
      return 1;
    }
  }

  glow::Executor core("ImageClassifier", argc, argv);
  auto printResultCreator =
      []() -> std::unique_ptr<PostProcessOutputDataExtension> {
    return std::make_unique<ImageClassifierProcessResult>();
  };
  core.registerPostProcessOutputExtension(printResultCreator);

  int numErrors = core.executeNetwork();
  return numErrors;
}
