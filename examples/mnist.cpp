#include "noether/Image.h"
#include "noether/Network.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace noether;

const size_t mnistNumImages = 50000;

/// This test classifies digits from the MNIST labeled dataset.
void testMNIST(bool verbose = false) {
  if (verbose) {
    std::cout << "Loading the mnist database.\n";
  }

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
  Array4D<float> imageInputs(50000, 28, 28, 1);
  Array4D<size_t> labelInputs(50000, 1, 1, 1);

  size_t idx = 0;

  for (size_t w = 0; w < mnistNumImages; w++) {
    labelInputs.at(w, 0, 0, 0) = labels[w];
    for (size_t y = 0; y < 28; y++) {
      for (size_t x = 0; x < 28; x++) {
        imageInputs.at(w, x, y, 0) = imagesAsFloatPtr[idx++];
      }
    }
  }
  size_t numImages = labels.size();
  assert(numImages && "No images were found.");

  if (verbose) {
    std::cout << "Loaded " << numImages << " images.\n";
  }

  // Construct the network:
  Network N;
  N.getTrainingConfig().learningRate = 0.01;
  N.getTrainingConfig().momentum = 0.9;
  N.getTrainingConfig().batchSize = 20;
  N.getTrainingConfig().inputSize = 50000;

  auto *A = N.createArrayNode(28, 28, 1);
  auto *CV0 = N.createConvNode(A, 8, 5, 1, 2);
  auto *RL0 = N.createRELUNode(CV0);
  auto *MP0 = N.createMaxPoolNode(RL0, 2, 2, 0);

  auto *CV1 = N.createConvNode(MP0, 16, 5, 1, 2);
  auto *RL1 = N.createRELUNode(CV1);
  auto *MP1 = N.createMaxPoolNode(RL1, 2, 2, 0);

  auto *FCL1 = N.createFullyConnectedNode(MP1, 10);
  auto *RL2 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL2);

  // On each training iteration the inputs are loaded from the image db.
  A->bind(&imageInputs);

  // On each  iteration the expected value is loaded from the labels vector.
  SM->bind(&labelInputs);

  if (verbose) {
    std::cout << "Training.\n";
  }

  for (int iter = 0; iter < 5000; iter++) {
    if (verbose && !(iter % 1000)) {
      std::cout << "Training - iteration #" << iter << "\n";
    }
    N.train(SM);
  }

  if (verbose) {
    std::cout << "Validating.\n";
  }

  // Check how many digits out of ten we can classify correctly.
  int rightAnswer = 0;

  for (int iter = 0; iter < 10; iter++) {
    size_t imageIndex = (iter * 17512 + 9124) % numImages;
    A->getOutput().weight_ = imageInputs.extractSlice(imageIndex);

    N.infer(SM);

    size_t guess = SM->maxArg();
    size_t correct = labels[imageIndex];
    rightAnswer += (guess == correct);

    if (verbose) {
      A->getOutput().weight_.dumpAscii("MNIST Input");
      std::cout << "Expected: " << correct << " Guessed: " << guess << "\n";
      SM->getOutput().weight_.dump("", "\n");
      std::cout << "\n-------------\n";
    }
  }

  assert(rightAnswer >= 6 && "Did not classify as many digits as expected");
}

int main() {

  testMNIST(1);

  return 0;
}
