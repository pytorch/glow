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

#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <fstream>
#include <iostream>

using namespace glow;

/// The CIFAR file format is structured as one byte label in the range 0..9.
/// The label is followed by an image: 32 x 32 pixels, in RGB format. Each
/// color is 1 byte. The first 1024 red bytes are followed by 1024 of green
/// and blue. Each 1024 byte color slice is organized in row-major format.
/// The database contains 10000 images.
/// Size: (1 + (32 * 32 * 3)) * 10000 = 30730000.
const size_t cifarImageSize = 1 + (32 * 32 * 3);
const size_t cifarNumImages = 10000;

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

  if (!dbInput.is_open()) {
    llvm::outs() << "Failed to open cifar10 data file, probably missing.\n";
    llvm::outs() << "Run 'python ../glow/utils/download_test_db.py'\n";
    exit(1);
  }

  llvm::outs() << "Loading the CIFAR-10 database.\n";

  /// Load the CIFAR database into a 4d tensor.
  Tensor images(ElemKind::FloatTy, {cifarNumImages, 32, 32, 3});
  Tensor labels(ElemKind::IndexTy, {cifarNumImages, 1});
  size_t idx = 0;

  auto labelsH = labels.getHandle<size_t>();
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
  GLOW_ASSERT(idx == cifarImageSize * cifarNumImages && "Invalid input file");

  unsigned minibatchSize = 8;

  // Construct the network:
  ExecutionEngine EE;
  EE.getConfig().learningRate = 0.001;
  EE.getConfig().momentum = 0.9;
  EE.getConfig().L2Decay = 0.0001;
  EE.getConfig().batchSize = minibatchSize;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Create the input layer:
  auto *A =
      mod.createVariable(ElemKind::FloatTy, {minibatchSize, 32, 32, 3}, "input",
                         VisibilityKind::Public, Variable::TrainKind::None);
  auto *E =
      mod.createVariable(ElemKind::IndexTy, {minibatchSize, 1}, "expected",
                         VisibilityKind::Public, Variable::TrainKind::None);

  // Create the rest of the network.
  auto *CV0 = F->createConv("conv", A, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu", CV0);
  auto *MP0 = F->createPoolMax("pool", RL0, 2, 2, 0);

  auto *CV1 = F->createConv("conv", MP0, 20, 5, 1, 2, 1);
  auto *RL1 = F->createRELU("relu", CV1);
  auto *MP1 = F->createPoolMax("pool", RL1, 2, 2, 0);

  auto *CV2 = F->createConv("conv", MP1, 20, 5, 1, 2, 1);
  auto *RL2 = F->createRELU("relu", CV2);
  auto *MP2 = F->createPoolMax("pool", RL2, 2, 2, 0);

  auto *FCL1 = F->createFullyConnected("fc", MP2, 10);
  auto *SM = F->createSoftMax("softmax", FCL1, E);
  auto *result = F->createSave("ret", SM);

  Function *TF = glow::differentiate(F, EE.getConfig());
  EE.compile(CompilationMode::Train, TF);

  // Report progress every this number of training iterations.
  int reportRate = 256;

  llvm::outs() << "Training.\n";

  for (int iter = 0; iter < 100000; iter++) {
    llvm::outs() << "Training - iteration #" << iter << "\n";

    llvm::Timer timer("Training", "Training");
    timer.startTimer();

    // Bind the images tensor to the input array A, and the labels tensor
    // to the softmax node SM.
    EE.runBatch(reportRate, {A, E}, {&images, &labels});

    unsigned score = 0;

    for (unsigned int i = 0; i < 100 / minibatchSize; i++) {
      Tensor sample(ElemKind::FloatTy, {minibatchSize, 32, 32, 3});
      sample.copyConsecutiveSlices(&images, minibatchSize * i);
      EE.run({A}, {&sample});
      result->getOutput();
      Tensor &res = result->getVariable()->getPayload();

      for (unsigned int iter = 0; iter < minibatchSize; iter++) {
        auto T = res.getHandle<>().extractSlice(iter);
        size_t guess = T.getHandle<>().minMaxArg().second;
        size_t correct = labelsH.at({minibatchSize * i + iter, 0});
        score += guess == correct;

        if ((iter < 10) && i == 0) {
          llvm::outs() << iter << ") Expected: " << textualLabels[correct]
                       << " Got: " << textualLabels[guess] << "\n";
        }
      }
    }

    timer.stopTimer();

    llvm::outs() << "Iteration #" << iter << " score: " << score << "%\n";
  }
}

int main() {
  testCIFAR10();

  return 0;
}
