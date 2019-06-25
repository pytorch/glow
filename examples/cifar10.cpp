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
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <glog/logging.h>

#include <fstream>
#include <iostream>

using namespace glow;

enum class ModelKind {
  MODEL_SIMPLE,
  MODEL_VGG,
};

namespace {
llvm::cl::OptionCategory cifarCat("CIFAR10 Options");
llvm::cl::opt<std::string> executionBackend(
    "backend",
    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
    llvm::cl::Optional, llvm::cl::init("Interpreter"), llvm::cl::cat(cifarCat));
llvm::cl::opt<ModelKind>
    model(llvm::cl::desc("Model to use:"), llvm::cl::Optional,
          llvm::cl::values(clEnumValN(ModelKind::MODEL_SIMPLE, "model-simple",
                                      "simple default model"),
                           clEnumValN(ModelKind::MODEL_VGG, "model-vgg",
                                      "model similar to vgg11")),
          llvm::cl::init(ModelKind::MODEL_SIMPLE), llvm::cl::cat(cifarCat));
} // namespace

/// The CIFAR file format is structured as one byte label in the range 0..9.
/// The label is followed by an image: 32 x 32 pixels, in RGB format. Each
/// color is 1 byte. The first 1024 red bytes are followed by 1024 of green
/// and blue. Each 1024 byte color slice is organized in row-major format.
/// The database contains 10000 images.
/// Size: (1 + (32 * 32 * 3)) * 10000 = 30730000.
const size_t cifarImageSize = 1 + (32 * 32 * 3);
const size_t cifarNumImages = 10000;
const unsigned numLabels = 10;

static Placeholder *createDefaultModel(PlaceholderBindings &bindings,
                                       Function *F, NodeValue input,
                                       NodeValue expected) {
  auto *CV0 = F->createConv(bindings, "conv", input, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu", CV0);
  auto *MP0 = F->createMaxPool("pool", RL0, 2, 2, 0);

  auto *CV1 = F->createConv(bindings, "conv", MP0, 20, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("relu", CV1);
  auto *MP1 = F->createMaxPool("pool", RL1, 2, 2, 0);

  auto *CV2 = F->createConv(bindings, "conv", MP1, 20, 5, 1, 2, 1);
  auto *RL2 = F->createRELU("relu", CV2);
  auto *MP2 = F->createMaxPool("pool", RL2, 2, 2, 0);

  auto *FCL1 = F->createFullyConnected(bindings, "fc", MP2, numLabels);
  auto *SM = F->createSoftMax("softmax", FCL1, expected);
  auto *save = F->createSave("ret", SM);
  return save->getPlaceholder();
}

/// Creates a VGG Model. Inspired by pytorch/torchvision vgg.py/vgg11:
/// https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
static Placeholder *createVGGModel(PlaceholderBindings &bindings, Function *F,
                                   NodeValue input, NodeValue expected) {
  NodeValue v = input;

  // Create feature detection part.
  unsigned cfg[] = {64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0};
  for (unsigned c : cfg) {
    if (c == 0) {
      v = F->createMaxPool("pool", v, 2, 2, 0);
    } else {
      auto *conv = F->createConv(bindings, "conv", v, c, 3, 1, 1, 1);
      auto *relu = F->createRELU("relu", conv);
      v = relu;
    }
  }

  // Create classifier part.
  for (unsigned i = 0; i < 2; ++i) {
    auto *fc0 = F->createFullyConnected(bindings, "fc", v, 4096);
    auto *relu0 = F->createRELU("relu", fc0);
    // TODO: There is not builtin dropout node in glow yet
    // Dropout
    v = relu0;
  }
  v = F->createFullyConnected(bindings, "fc", v, numLabels);
  auto *softmax = F->createSoftMax("softmax", v, expected);
  auto *save = F->createSave("ret", softmax);
  return save->getPlaceholder();
}

/// This test classifies digits from the CIFAR labeled dataset.
/// Details: http://www.cs.toronto.edu/~kriz/cifar.html
/// Dataset: http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
void testCIFAR10() {
  (void)cifarImageSize;
  const char *textualLabels[] = {"airplane", "automobile", "bird", "cat",
                                 "deer",     "dog",        "frog", "horse",
                                 "ship",     "truck"};

  std::ifstream dbInput("cifar-10-batches-bin/data_batch_1.bin",
                        std::ios::binary);

  CHECK(dbInput.is_open())
      << "Failed to open cifar10 data file, probably missing. Run 'python "
         "../glow/utils/download_datasets_and_models.py -d cifar10'";

  LOG(INFO) << "Loading the CIFAR-10 database.\n";

  /// Load the CIFAR database into a 4d tensor.
  Tensor images(ElemKind::FloatTy, {cifarNumImages, 32, 32, 3});
  Tensor labels(ElemKind::Int64ITy, {cifarNumImages, 1});
  size_t idx = 0;

  auto labelsH = labels.getHandle<int64_t>();
  auto imagesH = images.getHandle<>();
  for (unsigned w = 0; w < cifarNumImages; w++) {
    labelsH.at({w, 0}) = static_cast<uint8_t>(dbInput.get());
    idx++;

    for (unsigned z = 0; z < 3; z++) {
      for (unsigned y = 0; y < 32; y++) {
        for (unsigned x = 0; x < 32; x++) {
          imagesH.at({w, x, y, z}) =
              static_cast<float>(static_cast<uint8_t>(dbInput.get())) / 255.0;
          idx++;
        }
      }
    }
  }
  CHECK_EQ(idx, cifarImageSize * cifarNumImages) << "Invalid input file";

  unsigned minibatchSize = 8;

  // Construct the network:
  TrainingConfig TC;

  ExecutionEngine EE(executionBackend);
  PlaceholderBindings bindings;

  TC.learningRate = 0.001;
  TC.momentum = 0.9;
  TC.L2Decay = 0.0001;
  TC.batchSize = minibatchSize;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Create the input layer:
  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {minibatchSize, 32, 32, 3},
                                  "input", false);
  bindings.allocate(A);
  auto *E = mod.createPlaceholder(ElemKind::Int64ITy, {minibatchSize, 1},
                                  "expected", false);
  bindings.allocate(E);

  auto createModel =
      model == ModelKind::MODEL_SIMPLE ? createDefaultModel : createVGGModel;
  auto *resultPH = createModel(bindings, F, A, E);
  auto *result = bindings.allocate(resultPH);

  Function *TF = glow::differentiate(F, TC);
  EE.compile(CompilationMode::Train, TF);

  // Report progress every this number of training iterations.
  // Report less often for fast models.
  int reportRate = model == ModelKind::MODEL_SIMPLE ? 256 : 64;

  LOG(INFO) << "Training.";

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  for (int iter = 0; iter < 100000; iter++) {
    unsigned epoch = (iter * reportRate) / labels.getType().sizes_[0];
    LOG(INFO) << "Training - iteration #" << iter << " (epoch #" << epoch
              << ")";

    llvm::Timer timer("Training", "Training");
    timer.startTimer();

    // Bind the images tensor to the input array A, and the labels tensor
    // to the softmax node SM.
    runBatch(EE, bindings, reportRate, sampleCounter, {A, E},
             {&images, &labels});

    unsigned score = 0;

    for (unsigned int i = 0; i < 100 / minibatchSize; i++) {
      Tensor sample(ElemKind::FloatTy, {minibatchSize, 32, 32, 3});
      sample.copyConsecutiveSlices(&images, minibatchSize * i);
      updateInputPlaceholders(bindings, {A}, {&sample});
      EE.run(bindings);

      for (unsigned int iter = 0; iter < minibatchSize; iter++) {
        auto T = result->getHandle<>().extractSlice(iter);
        size_t guess = T.getHandle<>().minMaxArg().second;
        size_t correct = labelsH.at({minibatchSize * i + iter, 0});
        score += guess == correct;

        if ((iter < numLabels) && i == 0) {
          LOG(INFO) << iter << ") Expected: " << textualLabels[correct]
                    << " Got: " << textualLabels[guess];
        }
      }
    }

    timer.stopTimer();

    LOG(INFO) << "Iteration #" << iter << " score: " << score << "%";
  }
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, " The CIFAR10 test\n\n");
  testCIFAR10();

  return 0;
}
