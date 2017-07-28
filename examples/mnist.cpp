#include "noether/Image.h"
#include "noether/Network.h"
#include "noether/Nodes.h"
#include "noether/Support.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace noether;

const size_t mnistNumImages = 50000;

unsigned loadMNIST(Tensor &imageInputs, Tensor &labelInputs) {

  /// Load the MNIST database into two 4d tensors for images and labels.
  imageInputs.reset(ElemKind::FloatTy, {50000, 28, 28, 1});
  labelInputs.reset(ElemKind::IndexTy, {50000u});

  std::ifstream imgInput("mnist_images.bin", std::ios::binary);
  std::ifstream labInput("mnist_labels.bin", std::ios::binary);

  std::vector<char> images((std::istreambuf_iterator<char>(imgInput)),
                           (std::istreambuf_iterator<char>()));
  std::vector<char> labels((std::istreambuf_iterator<char>(labInput)),
                           (std::istreambuf_iterator<char>()));
  float *imagesAsFloatPtr = reinterpret_cast<float *>(&images[0]);

  assert(labels.size() * 28 * 28 * sizeof(float) == images.size() &&
         "The size of the image buffer does not match the labels vector");

  size_t idx = 0;

  auto LIH = labelInputs.getHandle<size_t>();
  auto IIH = imageInputs.getHandle<FloatTy>();

  for (unsigned w = 0; w < mnistNumImages; w++) {
    LIH.at({w}) = labels[w];
    for (unsigned y = 0; y < 28; y++) {
      for (unsigned x = 0; x < 28; x++) {
        IIH.at({w, x, y, 0}) = imagesAsFloatPtr[idx++];
      }
    }
  }
  size_t numImages = labels.size();
  assert(numImages && "No images were found.");
  return numImages;
}

/// This test classifies digits from the MNIST labeled dataset.
void testMNIST() {
  std::cout << "Loading the mnist database.\n";

  Tensor imageInputs;
  Tensor labelInputs;

  unsigned numImages = loadMNIST(imageInputs, labelInputs);
  std::cout << "Loaded " << numImages << " images.\n";

  // Construct the network:
  Network N;
  N.getConfig().learningRate = 0.01;
  N.getConfig().momentum = 0.9;
  N.getConfig().batchSize = 20;
  N.getConfig().L2Decay = 0.001;

  auto *A = N.createArrayNode({28, 28, 1});
  auto *CV0 = N.createConvNode(A, 16, 5, 1, 2);
  auto *RL0 = N.createRELUNode(CV0);
  auto *MP0 = N.createMaxPoolNode(RL0, 2, 2, 0);

  auto *CV1 = N.createConvNode(MP0, 16, 5, 1, 2);
  auto *RL1 = N.createRELUNode(CV1);
  auto *MP1 = N.createMaxPoolNode(RL1, 3, 3, 0);

  auto *FCL1 = N.createFullyConnectedNode(MP1, 10);
  auto *RL2 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL2);

  // Report progress every this number of training iterations.
  constexpr int reportRate = 100;

  std::cout << "Training.\n";

  for (int iter = 0; iter < 60; iter++) {
    std::cout << "Training - iteration #" << iter << " ";
    TimerGuard reportTime(reportRate);
    // On each training iteration take an input from imageInputs and update
    // the input variable A, and add take a corresponding label and update the
    // softmax layer.
    N.train(SM, reportRate, {A, SM}, {&imageInputs, &labelInputs});
  }
  std::cout << "Validating.\n";

  auto LIH = labelInputs.getHandle<size_t>();
  auto IIH = imageInputs.getHandle<FloatTy>();
  // Check how many digits out of ten we can classify correctly.
  int rightAnswer = 0;

  for (int iter = 0; iter < 10; iter++) {
    size_t imageIndex = (iter * 19 + 124) % numImages;
    Tensor sample = IIH.extractSlice(imageIndex);

    auto *res = N.infer(SM, {A}, {&sample});

    size_t guess = res->getHandle<FloatTy>().maxArg();
    size_t correct = LIH.at(imageIndex);
    rightAnswer += (guess == correct);

    sample.getHandle<FloatTy>().dumpAscii("MNIST Input");
    std::cout << "Expected: " << correct << " Guessed: " << guess << "\n";
    res->getHandle<FloatTy>().dump("", "\n");
    std::cout << "\n-------------\n";
  }

  assert(rightAnswer >= 6 && "Did not classify as many digits as expected");
}

int main() {
  testMNIST();

  return 0;
}
