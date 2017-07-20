#include "noether/Image.h"
#include "noether/Network.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"
#include "noether/Support.h"


#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace noether;

const size_t mnistNumImages = 50000;

/// This test classifies digits from the MNIST labeled dataset.
void testMNIST() {
  std::cout << "Loading the mnist database.\n";


  std::ifstream imgInput("mnist_images.bin", std::ios::binary);
  std::ifstream labInput("mnist_labels.bin", std::ios::binary);

  std::vector<char> images((std::istreambuf_iterator<char>(imgInput)),
                           (std::istreambuf_iterator<char>()));
  std::vector<char> labels((std::istreambuf_iterator<char>(labInput)),
                           (std::istreambuf_iterator<char>()));
  float *imagesAsFloatPtr = reinterpret_cast<float *>(&images[0]);

  assert(labels.size() * 28 * 28 * sizeof(float) == images.size() &&
         "The size of the image buffer does not match the labels vector");

  /// Load the MNIST database into two 4d tensors for images and labels.
  Tensor<float> imageInputs({50000, 28, 28, 1});
  Tensor<size_t> labelInputs(ArrayRef<size_t>((size_t)50000u));

  size_t idx = 0;

  auto LIH = labelInputs.getHandle();
  auto IIH = imageInputs.getHandle();

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

  std::cout << "Loaded " << numImages << " images.\n";

  // Construct the network:
  Network N;
  N.getTrainingConfig().learningRate = 0.01;
  N.getTrainingConfig().momentum = 0.9;
  N.getTrainingConfig().batchSize = 40;
  N.getTrainingConfig().inputSize = 50000;

  auto *A = N.createArrayNode({28, 28, 1});
  auto *CV0 = N.createConvNode(A, 8, 5, 1, 2);
  auto *RL0 = N.createRELUNode(CV0);
  auto *MP0 = N.createMaxPoolNode(RL0, 2, 2, 0);

  auto *CV1 = N.createConvNode(MP0, 16, 5, 1, 2);
  auto *RL1 = N.createRELUNode(CV1);
  auto *MP1 = N.createMaxPoolNode(RL1, 2, 2, 0);

  auto *FCL1 = N.createFullyConnectedNode(MP1, 10);
  auto *RL2 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL2);


  // Report progress every this number of training iterations.
  constexpr int reportRate = 100;

  // On each training iteration the inputs are loaded from the image db.
  A->bind(&imageInputs);

  // On each  iteration the expected value is loaded from the labels vector.
  SM->bind(&labelInputs);

  std::cout << "Training.\n";

  for (int iter = 0; iter < 60; iter++) {
    std::cout << "Training - iteration #" << iter << " ";
    TimerGuard reportTime(reportRate);
    N.train(SM, reportRate);

  }
  std::cout << "Validating.\n";

  // Check how many digits out of ten we can classify correctly.
  int rightAnswer = 0;

  for (int iter = 0; iter < 10; iter++) {
    size_t imageIndex = (iter * 17512 + 9124) % numImages;
    A->getOutput().weight_ = IIH.extractSlice(imageIndex);

    N.infer(SM);

    size_t guess = SM->maxArg();
    size_t correct = labels[imageIndex];
    rightAnswer += (guess == correct);

    A->getOutput().weight_.getHandle().dumpAscii("MNIST Input");
    std::cout << "Expected: " << correct << " Guessed: " << guess << "\n";
    SM->getOutput().weight_.dump("", "\n");
    std::cout << "\n-------------\n";
  }

  assert(rightAnswer >= 6 && "Did not classify as many digits as expected");
}

int main() {
  testMNIST();

  return 0;
}
