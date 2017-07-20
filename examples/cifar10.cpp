#include "noether/Image.h"
#include "noether/Network.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include <chrono>
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
void testCIFAR10(bool verbose = false) {
  (void)cifarImageSize;
  const char *textualLabels[] = {"airplane", "automobile", "bird", "cat",
                                 "deer",     "dog",        "frog", "horse",
                                 "ship",     "truck"};

  std::ifstream dbInput("cifar-10-batches-bin/data_batch_1.bin",
                        std::ios::binary);

  if (verbose) {
    std::cout << "Loading the CIFAR-10 database.\n";
  }

  /// Load the CIFAR database into a 4d tensor.
  Tensor<float> images({cifarNumImages, 32, 32, 3});
  Tensor<size_t> labels(ArrayRef<size_t>((size_t)cifarNumImages));
  size_t idx = 0;

  auto labelsH = labels.getHandle();
  auto imagesH = images.getHandle();
  for (unsigned w = 0; w < cifarNumImages; w++) {
    labelsH.at({w}) = static_cast<uint8_t>(dbInput.get());
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
  assert(idx == cifarImageSize * cifarNumImages && "Invalid input file");

  // Construct the network:
  Network N;
  N.getTrainingConfig().learningRate = 0.001;
  N.getTrainingConfig().momentum = 0.9;
  N.getTrainingConfig().batchSize = 8;
  N.getTrainingConfig().L2Decay = 0.0001;
  N.getTrainingConfig().inputSize = cifarImageSize;

  auto *A = N.createArrayNode({32, 32, 3});
  auto *CV0 = N.createConvNode(A, 16, 5, 1, 2);
  auto *RL0 = N.createRELUNode(CV0);
  auto *MP0 = N.createMaxPoolNode(RL0, 2, 2, 0);

  auto *CV1 = N.createConvNode(MP0, 20, 5, 1, 2);
  auto *RL1 = N.createRELUNode(CV1);
  auto *MP1 = N.createMaxPoolNode(RL1, 2, 2, 0);

  auto *CV2 = N.createConvNode(MP1, 20, 5, 1, 2);
  auto *RL2 = N.createRELUNode(CV2);
  auto *MP2 = N.createMaxPoolNode(RL2, 2, 2, 0);

  auto *FCL1 = N.createFullyConnectedNode(MP2, 10);
  auto *RL3 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL3);

  // On each training iteration the inputs are loaded from the image db.
  A->bind(&images);

  // On each  iteration the expected value is loaded from the labels vector.
  SM->bind(&labels);


  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  end = std::chrono::system_clock::now();

  // Report progress every this number of training iterations.
  constexpr int reportRate = 1024;

  if (verbose) {
    std::cout << "Training.\n";
  }

  for (int iter = 0; iter < 96000; iter++) {
    if (verbose && !(iter % reportRate)) {

      end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
      std::cout << "Training - iteration #" << iter << " ";
      std::cout <<"Rate: " << (reportRate/elapsed_seconds.count()) << "/sec\n";
      start = std::chrono::system_clock::now();
      
      unsigned score = 0;
      for (size_t iter = 0; iter < 100; iter++) {
        // Pick a random image from the stack:
        const unsigned imageIndex = (iter * 17512 + 9124) % cifarNumImages;

        // Load the image.
        A->getOutput().weight_ = imagesH.extractSlice(imageIndex);

        N.infer(SM);

        // Read the expected label.
        auto expectedLabel = labelsH.at({imageIndex});

        unsigned result = SM->maxArg();
        score += textualLabels[expectedLabel] ==  textualLabels[result];
      }

      std::cout << "Score : " <<score<<" / 100.\n";
    }

    N.train(SM);
  }
}

int main() {

  testCIFAR10(true);

  return 0;
}
