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

#include "glow/Base/Image.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Optimizer/GraphOptimizer/TrainingPreparation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Timer.h"

#include <fstream>

/// This is an example demonstrating how to load resnet50 model,
/// randomize weights and perform training on a limited data set,
/// and evaluate model.
using namespace glow;

constexpr dim_t IMAGE_COLORS = 3;
constexpr dim_t IMAGE_HEIGHT = 224;
constexpr dim_t IMAGE_WIDTH = 224;

namespace {
llvm::cl::OptionCategory category("resnet-training Options");
llvm::cl::opt<unsigned> numEpochs("epochs", llvm::cl::desc("Number of epochs."),
                                  llvm::cl::init(32), llvm::cl::value_desc("N"),
                                  llvm::cl::cat(category));
llvm::cl::opt<unsigned> miniBatchSize("mini-batch",
                                      llvm::cl::desc("Mini batch size."),
                                      llvm::cl::init(32),
                                      llvm::cl::value_desc("N"),
                                      llvm::cl::cat(category));
llvm::cl::opt<float> learningRate("learning-rate",
                                  llvm::cl::desc("Learning rate parameter."),
                                  llvm::cl::init(0.01),
                                  llvm::cl::value_desc("N"),
                                  llvm::cl::cat(category));
llvm::cl::opt<float> momentum("momentum", llvm::cl::desc("Momentum parameter."),
                              llvm::cl::init(0.9), llvm::cl::value_desc("N"),
                              llvm::cl::cat(category));
llvm::cl::opt<float> l2Decay("l2-decay", llvm::cl::desc("L2Decay parameter."),
                             llvm::cl::init(0.01), llvm::cl::value_desc("N"),
                             llvm::cl::cat(category));
llvm::cl::opt<bool> verboseOpt("verbose", llvm::cl::desc("verbose outputs."),
                               llvm::cl::init(false), llvm::cl::cat(category));

llvm::cl::opt<std::string> imageListFile(
    "input-image-list-file",
    llvm::cl::desc(
        "Name of the file containing list of images (one image per line)"),
    llvm::cl::Positional, llvm::cl::cat(category));

llvm::cl::opt<std::string>
    resnet50Path("resnet50", llvm::cl::desc("Path to the ResNet50 model."),
                 llvm::cl::init("resnet50"), llvm::cl::value_desc("model path"),
                 llvm::cl::Optional, llvm::cl::cat(category));

llvm::cl::opt<std::string> executionBackend(
    "backend",
    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
    llvm::cl::Optional, llvm::cl::init("Interpreter"), llvm::cl::cat(category));

/// Finds the maximum value and its index within \p N range.
template <typename H> size_t findMaxIndex(const H &handle, size_t range) {
  size_t ret{0};
  auto maxVal{handle.raw(0)};
  for (size_t c = 1; c < range; ++c) {
    auto val = handle.raw(c);
    if (maxVal <= val) {
      maxVal = val;
      ret = c;
    }
  }
  return ret;
}

bool parseInputImageList(const std::string &inputImageListFile,
                         std::vector<std::string> &filenames,
                         std::vector<dim_t> &labels) {
  std::ifstream inFile;
  inFile.open(inputImageListFile);
  if (!inFile.good()) {
    llvm::outs() << "Could not open input-image-list-file: "
                 << inputImageListFile << ", exiting.\n";
    return false;
  }

  while (!inFile.eof()) {
    std::string line;
    getline(inFile, line);
    if (!line.empty()) {
      auto delim = line.find_last_of(',');
      filenames.push_back(line.substr(0, delim));
      labels.push_back(std::stoi(line.substr(delim + 1)));
    }
  }

  return true;
}

/// A pair representing a float and the index where the float was found.
using FloatIndexPair = std::pair<float, size_t>;

/// Given a Handle \p H of a 1D tensor with float elements, \returns the top K
/// (topKCount) [float, index] pairs, i.e. the pairs with the highest floats.
template <typename ElemTy>
static std::vector<FloatIndexPair> getTopKPairs(Handle<ElemTy> H,
                                                dim_t topKCount) {
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
    const size_t label = topKPairs[i].second;
    // Tab out the label so it aligns nicely with Label-K1.
    if (i != 0) {
      llvm::outs() << "\t\t\t\t\t";
    }
    llvm::outs() << "\tLabel-K" << i + 1 << ": " << label << " (probability: "
                 << llvm::format("%0.4f", topKPairs[i].first) << ")\n";
  }
}
} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    " ResNet50 Training Example\n\n");

  // We expect the input to be NCHW.
  std::vector<dim_t> allImagesDims = {miniBatchSize, IMAGE_COLORS, IMAGE_HEIGHT,
                                      IMAGE_WIDTH};
  std::vector<dim_t> initImagesDims = {1, IMAGE_COLORS, IMAGE_HEIGHT,
                                       IMAGE_WIDTH};
  std::vector<dim_t> allLabelsDims = {miniBatchSize, 1};

  std::vector<std::string> filenames;
  std::vector<dim_t> labels;
  if (!parseInputImageList(imageListFile, filenames, labels)) {
    return 1;
  }

  ExecutionEngine EE(executionBackend);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("resnet50");

  const char inputName[] = "gpu_0/data";
  TypeRef inputType = mod.uniqueType(ElemKind::FloatTy, initImagesDims);

  // Load ResNet model.
  llvm::outs() << "Loading resnet50 model.\n";
  Error error = Error::empty();

  // Loader has randomly initialized trainable weights.
  Caffe2ModelLoader loader(resnet50Path + "/predict_net.pbtxt",
                           resnet50Path + "/init_net.pb", {inputName},
                           {inputType}, *F, &error);

  if (ERR_TO_BOOL(std::move(error))) {
    llvm::errs() << "Loader failed to load resnet50 model from path: "
                 << resnet50Path << "\n";
    return -1;
  }

  llvm::outs() << "Preparing for training.\n";

  // Get input and output placeholders.
  auto *A = llvm::cast<glow::Placeholder>(
      EXIT_ON_ERR(loader.getNodeValueByName(inputName)));
  auto *E = EXIT_ON_ERR(loader.getSingleOutput());
  PlaceholderBindings bindings;
  bindings.allocate(mod.getPlaceholders());

  Placeholder *selected{nullptr};
  if (ERR_TO_BOOL(glow::prepareFunctionForTraining(F, bindings, selected))) {
    return -1;
  }

  llvm::outs() << "Differentiating graph.\n";
  TrainingConfig TC;
  TC.learningRate = learningRate;
  TC.momentum = momentum;
  TC.L2Decay = l2Decay;
  TC.batchSize = 1;
  Function *TF = glow::differentiate(F, TC);
  auto tfName = TF->getName();

  llvm::outs() << "Compiling backend.\n";
  EE.compile(CompilationMode::Train);
  // New PH's for the grad need to be allocated.
  bindings.allocate(mod.getPlaceholders());

  // Get input and output tensors.
  auto *input = bindings.get(A);
  auto *result = bindings.get(E);
  auto *label = bindings.get(selected);
  result->zero();

  // These tensors allocate memory for all images and labels prepared for
  // training.
  Tensor imagesT(ElemKind::FloatTy, allImagesDims);
  Tensor labelsT(ElemKind::Int64ITy, allLabelsDims);

  size_t idx = 0;
  size_t orig_size = filenames.size();
  while (filenames.size() < miniBatchSize) {
    filenames.push_back(filenames[idx % orig_size]);
    labels.push_back(labels[idx % orig_size]);
    idx++;
  }

  loadImagesAndPreprocess({filenames}, {&imagesT},
                          {ImageNormalizationMode::k0to1},
                          {ImageChannelOrder::BGR}, {ImageLayout::NCHW});

  auto LH = labelsT.getHandle<int64_t>();
  for (idx = 0; idx < miniBatchSize; ++idx) {
    LH.raw(idx) = labels[idx];
  }

  llvm::Timer timer("Training", "Training");

  for (size_t e = 0; e < numEpochs; ++e) {
    llvm::outs() << "Training - epoch #" << e << " from total " << numEpochs
                 << "\n";
    // Train and evaluating network on all examples.
    unsigned score = 0;
    unsigned total = 0;
    for (unsigned int i = 0; i < miniBatchSize; ++i) {
      input->copyConsecutiveSlices(&imagesT, i);
      label->copyConsecutiveSlices(&labelsT, i);

      llvm::outs() << ".";

      timer.startTimer();
      EE.run(bindings, tfName);
      timer.stopTimer();

      auto correct = labelsT.getHandle<sdim_t>().raw(i);
      auto guess = findMaxIndex(result->getHandle(), 1000);

      auto slice = result->getUnowned({1000});
      if (verboseOpt) {
        llvm::outs() << "correct: " << correct << " guess: " << guess << "\n";
        auto k = getTopKPairs(slice.getHandle(), 3);
        printTopKPairs(k);
      }

      score += guess == correct;
      ++total;
    }

    llvm::outs() << "Total accuracy: " << 100. * score / total << "\n";
  }

  return 0;
}
