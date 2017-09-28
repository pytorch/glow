#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Interpreter/Interpreter.h"
#include "glow/Support/Support.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

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
  auto imagesH = images.getHandle<FloatTy>();
  for (unsigned w = 0; w < cifarNumImages; w++) {
    labelsH.at({w, 0}) = static_cast<uint8_t>(dbInput.get());
    idx++;

    for (unsigned z = 0; z < 3; z++) {
      for (unsigned y = 0; y < 32; y++) {
        for (unsigned x = 0; x < 32; x++) {
          imagesH.at({w, x, y, z}) =
              FloatTy(static_cast<uint8_t>(dbInput.get())) / 255.0;
          idx++;
        }
      }
    }
  }
  GLOW_ASSERT(idx == cifarImageSize * cifarNumImages && "Invalid input file");

  // Construct the network:
  Interpreter IP;
  IP.getConfig().learningRate = 0.001;
  IP.getConfig().momentum = 0.9;
  IP.getConfig().L2Decay = 0.0001;

  Instruction *SM;
  Value *E;
  Value *A;
  unsigned minibatchSize = 8;

  {
    IRBuilder bb(IP.getModule());

    // Create the input layer:
    A = bb.createWeightVar(ElemKind::FloatTy, {minibatchSize, 32, 32, 3});
    E = bb.createWeightVar(ElemKind::IndexTy, {minibatchSize, 1});

    // Create the rest of the network.  NodeBase *SM = createSimpleNet(N, A, E);
    auto *CV0 = bb.createConvOp(A, 16, 5, 1, 2);
    auto *RL0 = bb.createRELUOp(*CV0);
    auto *MP0 = bb.createPoolOp(*RL0, PoolInst::OpKind::kMax, 2, 2, 0);

    auto *CV1 = bb.createConvOp(*MP0, 20, 5, 1, 2);
    auto *RL1 = bb.createRELUOp(*CV1);
    auto *MP1 = bb.createPoolOp(*RL1, PoolInst::OpKind::kMax, 2, 2, 0);

    auto *CV2 = bb.createConvOp(*MP1, 20, 5, 1, 2);
    auto *RL2 = bb.createRELUOp(*CV2);
    auto *MP2 = bb.createPoolOp(*RL2, PoolInst::OpKind::kMax, 2, 2, 0);

    auto *FCL1 = bb.createFullyConnectedOp(*MP2, 10);
    auto *RL3 = bb.createRELUOp(*FCL1);
    SM = bb.createSoftMaxOp(*RL3, E);
  }

  IP.getModule().dump();
  IP.initVars();

  // Report progress every this number of training iterations.
  int reportRate = 256;

  std::cout << "Training.\n";

  for (int iter = 0; iter < 100000; iter++) {
    std::cout << "Training - iteration #" << iter << "\n";
    TimerGuard reportTime(reportRate * minibatchSize);

    // Bind the images tensor to the input array A, and the labels tensor
    // to the softmax node SM.
    IP.train(reportRate, {A, E}, {&images, &labels});

    unsigned score = 0;

    for (unsigned int i = 0; i < 100 / minibatchSize; i++) {
      Tensor sample(ElemKind::FloatTy, {minibatchSize, 3, 32, 32});
      sample.copyConsecutiveSlices(&images, minibatchSize * i);
      IP.infer({A}, {&sample});
      auto *res = IP.getTensorForValue(*SM);

      for (unsigned int iter = 0; iter < minibatchSize; iter++) {
        auto T = res->getHandle<FloatTy>().extractSlice(iter);
        size_t guess = T.getHandle<FloatTy>().maxArg();
        size_t correct = labelsH.at({minibatchSize * i + iter, 0});
        score += guess == correct;

        if ((iter < 10) && i == 0) {
          // T.getHandle<FloatTy>().dump("softmax: "," ");
          std::cout << iter << ") Expected: " << textualLabels[correct]
                    << " Got: " << textualLabels[guess] << "\n";
        }
      }
    }

    std::cout << "Batch #" << iter << " score: " << score << "%\n";
  }
}

int main() {
  testCIFAR10();

  return 0;
}
