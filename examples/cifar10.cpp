#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Support/Support.h"

#include "llvm/Support/Timer.h"

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

  std::cout << "Loading the CIFAR-10 database.\n";

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

  // Construct the network:
  ExecutionEngine EE;
  EE.getConfig().learningRate = 0.001;
  EE.getConfig().momentum = 0.9;
  EE.getConfig().L2Decay = 0.0001;

  unsigned minibatchSize = 8;

  auto &G = EE.getGraph();

  // Create the input layer:
  auto *A =
      G.createVariable(ElemKind::FloatTy, {minibatchSize, 32, 32, 3}, "input");
  auto *E = G.createVariable(ElemKind::IndexTy, {minibatchSize, 1}, "expected",
                             Variable::InitKind::Extern);

  // Create the rest of the network.
  auto *CV0 = G.createConv("conv", A, 16, 5, 1, 2);
  auto *RL0 = G.createRELU("relu", CV0);
  auto *MP0 = G.createPool("pool", RL0, PoolNode::Mode::Max, 2, 2, 0);

  auto *CV1 = G.createConv("conv", MP0, 20, 5, 1, 2);
  auto *RL1 = G.createRELU("relu", CV1);
  auto *MP1 = G.createPool("pool", RL1, PoolNode::Mode::Max, 2, 2, 0);

  auto *CV2 = G.createConv("conv", MP1, 20, 5, 1, 2);
  auto *RL2 = G.createRELU("relu", CV2);
  auto *MP2 = G.createPool("pool", RL2, PoolNode::Mode::Max, 2, 2, 0);

  auto *FCL1 = G.createFullyConnected("fc", MP2, 10);
  auto *RL3 = G.createRELU("relu", FCL1);
  auto *SM = G.createSoftMax("softmax", RL3, E);
  auto *result = G.createSave("ret", SM);

  EE.compile(OptimizationMode::Train);

  // Report progress every this number of training iterations.
  int reportRate = 256;

  std::cout << "Training.\n";

  for (int iter = 0; iter < 100000; iter++) {
    std::cout << "Training - iteration #" << iter << "\n";

    llvm::Timer timer("Training", "Training");
    timer.startTimer();

    // Bind the images tensor to the input array A, and the labels tensor
    // to the softmax node SM.
    EE.train(reportRate, {A, E}, {&images, &labels});

    unsigned score = 0;

    for (unsigned int i = 0; i < 100 / minibatchSize; i++) {
      Tensor sample(ElemKind::FloatTy, {minibatchSize, 3, 32, 32});
      sample.copyConsecutiveSlices(&images, minibatchSize * i);
      EE.infer({A}, {&sample});
      result->getOutput();
      Tensor &res = result->getOutput()->getPayload();

      for (unsigned int iter = 0; iter < minibatchSize; iter++) {
        auto T = res.getHandle<>().extractSlice(iter);
        size_t guess = T.getHandle<>().maxArg();
        size_t correct = labelsH.at({minibatchSize * i + iter, 0});
        score += guess == correct;

        if ((iter < 10) && i == 0) {
          // T.getHandle<FloatTy>().dump("softmax: "," ");
          std::cout << iter << ") Expected: " << textualLabels[correct]
                    << " Got: " << textualLabels[guess] << "\n";
        }
      }
    }

    timer.stopTimer();

    std::cout << "Batch #" << iter << " score: " << score << "%\n";
  }
}

int main() {
  testCIFAR10();

  return 0;
}
