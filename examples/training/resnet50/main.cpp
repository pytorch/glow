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

constexpr size_t IMAGE_COLORS = 3;
constexpr size_t IMAGE_HEIGHT = 224;
constexpr size_t IMAGE_WIDTH = 224;
constexpr size_t CIFAR_NUM_IMAGES = 80;

using namespace glow;

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
llvm::cl::opt<std::string>
    cifar10DbPath("cifar10", llvm::cl::desc("Path to the CIFAR-10 database."),
                  llvm::cl::init("cifar-10-batches-bin/data_batch_1.bin"),
                  llvm::cl::value_desc("db path"), llvm::cl::Optional,
                  llvm::cl::cat(category));

llvm::cl::opt<std::string>
    resnet50Path("resnet50", llvm::cl::desc("Path to the ResNet50 model."),
                 llvm::cl::init("resnet50"), llvm::cl::value_desc("model path"),
                 llvm::cl::Optional, llvm::cl::cat(category));
llvm::cl::opt<std::string> executionBackend(
    "backend",
    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
    llvm::cl::Optional, llvm::cl::init("Interpreter"), llvm::cl::cat(category));

/// Finds the maximum value and its index within \p N range.
size_t findMaxIndex(const Tensor &t, size_t range) {
  auto handle = t.getHandle();
  size_t ret{0};
  float maxVal{handle.raw(0)};
  for (size_t c = 1; c < range; ++c) {
    auto val = handle.raw(c);
    if (maxVal <= val) {
      maxVal = val;
      ret = c;
    }
  }
  return ret;
}

void loadImagesAndLabels(Tensor &images, Tensor &labels) {
  llvm::outs() << "Loading the CIFAR-10 database.\n";
  /// ResNet expects ImageChannelOrder::BGR, ImageLayout::NCHW.
  std::ifstream dbInput(cifar10DbPath, std::ios::binary);

  if (!dbInput.is_open()) {
    llvm::errs() << "Failed to open cifar10 data file, probably missing.\n";
    llvm::errs() << "Run 'python ../glow/utils/download_test_db.py'\n";
    exit(1);
  }

  /// The CIFAR file format is structured as one byte label in the range 0..9.
  /// The label is followed by an image: 32 x 32 pixels, in RGB format. Each
  /// color is 1 byte. The first 1024 red bytes are followed by 1024 of green
  /// and blue. Each 1024 byte color slice is organized in row-major format.
  /// The database contains 10000 images.
  /// Size: (1 + (32 * 32 * 3)) * 10000 = 30730000.
  llvm::outs() << "Assigning images/labels bytes.\n";
  auto labelsH = labels.getHandle<int64_t>();
  auto imagesH = images.getHandle();
  float scale = 1. / 255.0;
  float bias = 0;
  images.zero();

  /// CIFAR image dimensions are 32x32, ResNet50 expects 224x224.
  /// Simple scaling would be one CIFAR image copied to the center area with
  /// offset 96 pixels
  constexpr unsigned imageOffset = 96;
  for (unsigned n = 0; n < CIFAR_NUM_IMAGES; ++n) {
    labelsH.at({n, 0}) = dbInput.get();
    // ResNet50 model got trained in NCHW format.
    for (unsigned c = 0; c < IMAGE_COLORS; ++c) {
      auto bgrc = IMAGE_COLORS - 1 - c;
      for (unsigned h = 0; h < 32; ++h) {
        for (unsigned w = 0; w < 32; ++w) {
          // ResNet BGR color space vs CIFAR RGB.
          auto val = static_cast<float>(static_cast<uint8_t>(dbInput.get()));
          // Normalize and scale image.
          val = (val - imagenetNormMean[c]) * scale / imagenetNormStd[c] + bias;
          imagesH.at({n, bgrc, h + imageOffset, w + imageOffset}) = val;
        }
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    " ResNet50 Training Example\n\n");

  // We expect the input to be NCHW.
  llvm::ArrayRef<size_t> allImagesDims = {CIFAR_NUM_IMAGES, IMAGE_COLORS,
                                          IMAGE_HEIGHT, IMAGE_WIDTH};
  llvm::ArrayRef<size_t> initImagesDims = {1, IMAGE_COLORS, IMAGE_HEIGHT,
                                           IMAGE_WIDTH};
  llvm::ArrayRef<size_t> allLabelsDims = {CIFAR_NUM_IMAGES, 1};

  ExecutionEngine EE(executionBackend);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("resnet50");

  const char inputName[] = "gpu_0/data";
  TypeRef inputType = mod.uniqueType(ElemKind::FloatTy, initImagesDims);

  // Load ResNet model.
  llvm::outs() << "Loading resnet50 model.\n";
  llvm::Error errPtr = llvm::Error::success();

  // Loader has randomly initialized trainable weights.
  Caffe2ModelLoader loader(resnet50Path + "/predict_net.pbtxt",
                           resnet50Path + "/init_net.pb", {inputName},
                           {inputType}, *F, &errPtr);

  if (errPtr) {
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
  if ((errPtr = glow::prepareFunctionForTraining(F, bindings, selected))) {
    return -1;
  }

  llvm::outs() << "Differentiating graph.\n";
  TrainingConfig TC;
  TC.learningRate = learningRate;
  TC.momentum = momentum;
  TC.L2Decay = l2Decay;
  TC.batchSize = miniBatchSize;
  Function *TF = glow::differentiate(F, TC);

  llvm::outs() << "Compiling backend.\n";
  EE.compile(CompilationMode::Train, TF);

  // Get input and output tensors.
  auto *input = bindings.get(A);
  auto *result = bindings.get(E);
  auto *label = bindings.get(selected);

  // These tensors allocate memory for all images and labels prepared for
  // training.
  Tensor images(ElemKind::FloatTy, allImagesDims);
  Tensor labels(ElemKind::Int64ITy, allLabelsDims);

  loadImagesAndLabels(images, labels);

  llvm::Timer timer("Training", "Training");

  auto labelsH = labels.getHandle<int64_t>();
  for (size_t e = 0; e < numEpochs; ++e) {
    llvm::outs() << "Training - epoch #" << e << " from total " << numEpochs
                 << "\n";
    // Train and evaluating network on all examples.
    unsigned score = 0;
    unsigned total = 0;
    for (unsigned int i = 0; i < CIFAR_NUM_IMAGES; ++i) {
      llvm::outs() << ".";

      input->copyConsecutiveSlices(&images, i);
      label->copyConsecutiveSlices(&labels, i);

      timer.startTimer();
      EE.run(bindings);
      timer.stopTimer();

      int64_t correct = labelsH.at({i, 0});
      auto guess = findMaxIndex(*result, 10);
      score += guess == correct;
      ++total;
    }

    llvm::outs() << "Total accuracy: " << 100. * score / total << "\n";
  }

  return 0;
}
