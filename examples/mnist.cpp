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

#include <cassert>
#include <fstream>

using namespace glow;

const size_t mnistNumImages = 50000;

namespace {
llvm::cl::OptionCategory mnistCat("MNIST Options");
llvm::cl::opt<BackendKind> executionBackend(
    llvm::cl::desc("Backend to use:"),
    llvm::cl::values(clEnumValN(BackendKind::Interpreter, "interpreter",
                                "Use interpreter (default option)"),
                     clEnumValN(BackendKind::CPU, "cpu", "Use CPU"),
                     clEnumValN(BackendKind::OpenCL, "opencl", "Use OpenCL")),
    llvm::cl::init(BackendKind::Interpreter), llvm::cl::cat(mnistCat));
} // namespace

unsigned loadMNIST(Tensor &imageInputs, Tensor &labelInputs) {
  /// Load the MNIST database into 4D tensor of images and 2D tensor of labels.
  imageInputs.reset(ElemKind::FloatTy, {50000u, 28, 28, 1});
  labelInputs.reset(ElemKind::IndexTy, {50000u, 1});

  std::ifstream imgInput("mnist_images.bin", std::ios::binary);
  std::ifstream labInput("mnist_labels.bin", std::ios::binary);

  if (!imgInput.is_open()) {
    llvm::errs() << "Error loading mnist_images.bin\n";
    std::exit(EXIT_FAILURE);
  }
  if (!labInput.is_open()) {
    llvm::errs() << "Error loading mnist_labels.bin\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<char> images((std::istreambuf_iterator<char>(imgInput)),
                           (std::istreambuf_iterator<char>()));
  std::vector<char> labels((std::istreambuf_iterator<char>(labInput)),
                           (std::istreambuf_iterator<char>()));
  float *imagesAsFloatPtr = reinterpret_cast<float *>(&images[0]);

  GLOW_ASSERT(labels.size() * 28 * 28 * sizeof(float) == images.size() &&
              "The size of the image buffer does not match the labels vector");

  size_t idx = 0;

  auto LIH = labelInputs.getHandle<uint64_t>();
  auto IIH = imageInputs.getHandle<>();

  for (unsigned w = 0; w < mnistNumImages; w++) {
    LIH.at({w, 0}) = labels[w];
    for (unsigned x = 0; x < 28; x++) {
      for (unsigned y = 0; y < 28; y++) {
        IIH.at({w, x, y, 0}) = imagesAsFloatPtr[idx++];
      }
    }
  }
  size_t numImages = labels.size();
  GLOW_ASSERT(numImages && "No images were found.");
  return numImages;
}

/// This test classifies digits from the MNIST labeled dataset.
void testMNIST() {
  llvm::outs() << "Loading the mnist database.\n";

  Tensor imageInputs;
  Tensor labelInputs;

  unsigned numImages = loadMNIST(imageInputs, labelInputs);
  llvm::outs() << "Loaded " << numImages << " images.\n";

  unsigned minibatchSize = 8;

  ExecutionEngine EE(executionBackend);
  llvm::Timer timer("Training", "Training");

  /// The training configuration.
  TrainingConfig TC;

  // Construct the network:
  TC.learningRate = 0.001;
  TC.momentum = 0.9;
  TC.L2Decay = 0.001;
  TC.batchSize = minibatchSize;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Variable *A =
      mod.createVariable(ElemKind::FloatTy, {minibatchSize, 28, 28, 1}, "input",
                         VisibilityKind::Public, false);

  auto *CV0 = F->createConv("conv", A, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu", CV0);
  auto *MP0 = F->createMaxPool("pool", RL0, 3, 3, 0);

  auto *CV1 = F->createConv("conv", MP0, 16, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("relu", CV1);
  auto *MP1 = F->createMaxPool("pool", RL1, 3, 3, 0);

  auto *FCL1 = F->createFullyConnected("fc", MP1, 10);
  Variable *selected =
      mod.createVariable(ElemKind::IndexTy, {minibatchSize, 1}, "selected",
                         VisibilityKind::Public, false);
  auto *SM = F->createSoftMax("sm", FCL1, selected);

  auto *result = F->createSave("return", SM);

  Function *T = glow::differentiate(F, TC);

  EE.compile(CompilationMode::Train, T);

  const int numIterations = 30;

  llvm::outs() << "Training.\n";

  for (int epoch = 0; epoch < 60; epoch++) {
    llvm::outs() << "Training - epoch #" << epoch << "\n";

    timer.startTimer();

    // On each training iteration take a slice of imageInputs and labelInputs
    // and put them into variables A and B, then run forward and backward passes
    // and update weights.
    EE.runBatch(numIterations, {A, selected}, {&imageInputs, &labelInputs});

    timer.stopTimer();
  }
  llvm::outs() << "Validating.\n";
  EE.compile(CompilationMode::Infer, F);

  auto LIH = labelInputs.getHandle<uint64_t>();

  // Check how many examples out of eighty previously unseen digits we can
  // classify correctly.
  int rightAnswer = 0;

  Tensor sample(ElemKind::FloatTy, {minibatchSize, 28, 28, 1});

  for (int iter = numIterations; iter < numIterations + 10; iter++) {
    sample.copyConsecutiveSlices(&imageInputs, minibatchSize * iter);
    EE.run({A}, {&sample});

    Tensor &res = result->getVariable()->getPayload();

    for (unsigned i = 0; i < minibatchSize; i++) {
      auto T = res.getHandle<>().extractSlice(i);
      size_t guess = T.getHandle<>().minMaxArg().second;

      size_t correct = LIH.at({minibatchSize * iter + i, 0});
      rightAnswer += (guess == correct);

      if (iter == numIterations) {
        auto I = sample.getHandle<>().extractSlice(i);

        llvm::outs() << "MNIST Input";
        I.getHandle<>().dumpAscii();
        llvm::outs() << "Expected: " << correct << " Guessed: " << guess
                     << "\n";

        T.getHandle<>().dump();
        llvm::outs() << "\n-------------\n";
      }
    }
  }

  llvm::outs() << "Results: guessed/total:" << rightAnswer << "/"
               << minibatchSize * 10 << "\n";
  GLOW_ASSERT(rightAnswer >= 74 &&
              "Did not classify as many digits as expected");
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, " The MNIST test\n\n");
  testMNIST();

  return 0;
}
