#include "glow/Network/Image.h"
#include "glow/Network/Network.h"
#include "glow/Network/Nodes.h"
#include "glow/Network/Tensor.h"
#include "glow/Support/Support.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace glow;

const size_t mnistNumImages = 50000;

unsigned loadMNIST(Tensor &imageInputs, Tensor &labelInputs) {

  /// Load the MNIST database into two 4d tensors for images and labels.
  imageInputs.reset(ElemKind::FloatTy, {50000, 28, 28, 1});
  labelInputs.reset(ElemKind::IndexTy, {50000u, 1});

  std::ifstream imgInput("mnist_images.bin", std::ios::binary);
  std::ifstream labInput("mnist_labels.bin", std::ios::binary);

  std::vector<char> images((std::istreambuf_iterator<char>(imgInput)),
                           (std::istreambuf_iterator<char>()));
  std::vector<char> labels((std::istreambuf_iterator<char>(labInput)),
                           (std::istreambuf_iterator<char>()));
  float *imagesAsFloatPtr = reinterpret_cast<float *>(&images[0]);

  GLOW_ASSERT(labels.size() * 28 * 28 * sizeof(float) == images.size() &&
              "The size of the image buffer does not match the labels vector");

  size_t idx = 0;

  auto LIH = labelInputs.getHandle<size_t>();
  auto IIH = imageInputs.getHandle<FloatTy>();

  for (unsigned w = 0; w < mnistNumImages; w++) {
    LIH.at({w, 0}) = labels[w];
    for (unsigned y = 0; y < 28; y++) {
      for (unsigned x = 0; x < 28; x++) {
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
  std::cout << "Loading the mnist database.\n";

  Tensor imageInputs;
  Tensor labelInputs;

  unsigned numImages = loadMNIST(imageInputs, labelInputs);
  std::cout << "Loaded " << numImages << " images.\n";

  unsigned minibatchSize = 8;

  // Construct the network:
  Network N;
  N.getConfig().learningRate = 0.001;
  N.getConfig().momentum = 0.9;
  N.getConfig().L2Decay = 0.001;

  auto *A = N.createVariable({minibatchSize, 28, 28, 1}, ElemKind::FloatTy);
  auto *CV0 = N.createConvNode(A, 16, 5, 1, 2);
  auto *RL0 = N.createRELUNode(CV0);
  auto *MP0 = N.createMaxPoolNode(RL0, MaxPoolNode::OpKind::kMax, 3, 3, 0);

  auto *CV1 = N.createConvNode(MP0, 16, 5, 1, 2);
  auto *RL1 = N.createRELUNode(CV1);
  auto *MP1 = N.createMaxPoolNode(RL1, MaxPoolNode::OpKind::kMax, 3, 3, 0);

  auto *FCL1 = N.createFullyConnectedNode(MP1, 10);
  auto *RL2 = N.createRELUNode(FCL1);

  auto *selected = N.createVariable({minibatchSize, 1}, ElemKind::IndexTy);

  auto *SM = N.createSoftMaxNode(RL2, selected);

  // Report progress every this number of training iterations.
  constexpr int reportRate = 30;

  std::cout << "Training.\n";

  for (int iter = 0; iter < 60; iter++) {
    std::cout << "Training - iteration #" << iter << " ";
    TimerGuard reportTime(reportRate * minibatchSize);
    // On each training iteration take an input from imageInputs and update
    // the input variable A, and add take a corresponding label and update the
    // softmax layer.
    N.train(SM, reportRate, {A, selected}, {&imageInputs, &labelInputs});
  }
  std::cout << "Validating.\n";

  auto LIH = labelInputs.getHandle<size_t>();

  // Check how many digits out of ten we can classify correctly.
  int rightAnswer = 0;

  Tensor sample(ElemKind::FloatTy, {minibatchSize, 1, 28, 28});
  sample.copyConsecutiveSlices(&imageInputs, 0);
  auto *res = N.infer(SM, {A}, {&sample});

  for (unsigned int iter = 0; iter < minibatchSize; iter++) {
    auto T = res->getHandle<FloatTy>().extractSlice(iter);
    size_t guess = T.getHandle<FloatTy>().maxArg();

    size_t correct = LIH.at(iter);
    rightAnswer += (guess == correct);

    auto I = sample.getHandle<FloatTy>().extractSlice(iter);
    auto J = I.getHandle<FloatTy>().extractSlice(0);

    J.getHandle<FloatTy>().dumpAscii("MNIST Input");
    std::cout << "Expected: " << correct << " Guessed: " << guess << "\n";

    T.getHandle<FloatTy>().dump("", "\n");
    std::cout << "\n-------------\n";
  }

  std::cout << "Results: guessed/total:" << rightAnswer << "/" << minibatchSize
            << "\n";
  GLOW_ASSERT(rightAnswer >= 6 &&
              "Did not classify as many digits as expected");
}

int main() {
  testMNIST();

  return 0;
}
