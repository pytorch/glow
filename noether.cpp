#include "noether/Image.h"
#include "noether/Nodes.h"
#include "noether/Network.h"
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

void testArray() {
  Array3D<float> X(320, 200, 3);
  X.at(10u, 10u, 2u) = 2;
  assert((X.at(10u, 10u, 2u) == 2) && "Invalid load/store");
}

float delta(float a, float b) { return std::fabs(a - b); }

/// Test the fully connected layer and the softmax function.
/// Example from:
/// http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
void testFCSoftMax(bool verbose = false) {

  // Construct the network:
  Network N;
  N.getTrainingConfig().momentum = 0.0;
  auto *A = N.createArrayNode(1, 1, 2);
  auto *FCL0 = N.createFullyConnectedNode( A, 6);
  auto *RL0 = N.createRELUNode (FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 2);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL1);

  // Generate some random numbers in the range -1 .. 1.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1, 1);

  // Generate lots of samples and learn them.
  for (int iter = 0; iter < 99000; iter++) {
    float x = dis(gen);
    float y = dis(gen);

    // Check if the dot falls within some inner circle.
    float r2 = (x * x + y * y);

    bool InCircle = r2 < 0.6;

    SM->setSelected(InCircle);
    A->getOutput().weight_.at(0, 0, 0) = x;
    A->getOutput().weight_.at(0, 0, 1) = y;
    N.train();
  }

  // Print a diagram that depicts the network decision on a grid.
  if (verbose) {
    for (int x = -10; x < 10; x++) {
      for (int y = -10; y < 10; y++) {
        // Load the inputs:
        A->getOutput().weight_.at(0, 0, 0) = float(x) / 10;
        A->getOutput().weight_.at(0, 0, 1) = float(y) / 10;

        N.infer();
        auto A = SM->getOutput().weight_.at(0, 0, 0);
        auto B = SM->getOutput().weight_.at(0, 0, 1);

        char ch = '=';
        if (A > (B + 0.2)) {
          ch = '+';
        } else if (B > (A + 0.2)) {
          ch = '-';
        }

        std::cout << ch;
      }
      std::cout << "\n";
    }
  }

  // Verify the label for some 10 random points.
  for (int iter = 0; iter < 10; iter++) {
    float x = dis(gen);
    float y = dis(gen);

    float r2 = (x * x + y * y);

    // Throw away confusing samples.
    if (r2 > 0.5 && r2 < 0.7)
      continue;

    // Load the inputs:
    A->getOutput().weight_.at(0, 0, 0) = x;
    A->getOutput().weight_.at(0, 0, 1) = y;

    N.infer();

    // Inspect the outputs:
    if (r2 < 0.50) {
      assert(SM->maxArg() == 1);
    }
    if (r2 > 0.7) {
      assert(SM->maxArg() == 0);
    }
  }
}

/// A helper function to load a one-hot vector.
void setOneHot(Array3D<FloatTy> &A, float background, float foreground,
               size_t idx) {
  for (int j = 0; j < A.size(); j++) {
    A.at(0, 0, j) = (j == idx ? foreground : background);
  }
}

void testLearnSingleInput() {
  Network N;
  N.getTrainingConfig().learningRate = 0.005;
  auto *A = N.createArrayNode(1, 1, 10);
  auto *FCL0 = N.createFullyConnectedNode (A, 10);
  auto *RL0 = N.createRELUNode (FCL0);
  auto *FCL1 = N.createFullyConnectedNode (RL0, 10);
  auto *RL1= N.createRELUNode (FCL1);
  auto *RN = N.createRegressionNode(RL1);

  // Put in [15, 0, 0, 0, 0 ... ]
  setOneHot(A->getOutput().weight_, 0.0, 15, 0);
  // Expect [0, 9.0, 0 , 0 , ...]
  setOneHot(RN->getExpected(), 0.0, 9.0, 1);

  // Train the network:
  for (int iter = 0; iter < 10000; iter++) {
    N.train();
  }

  N.infer();

  // Test the output:
  assert(RN->getOutput().weight_.sum() < 10);
  assert(RN->getOutput().weight_.at(0, 0, 1) > 8.5);
}

void testRegression() {
  Network N;

  /// This test takes the first element from the input vector, adds one to it
  /// and places the result in the second element of the output vector.
  constexpr int numInputs = 4;

  auto *A = N.createArrayNode(1, 1, numInputs);
  auto *FCL0 = N.createFullyConnectedNode(A, 4);
  auto *RL0 = N.createRELUNode (FCL0);
  auto *RN = N.createRegressionNode (RL0);

  // Train the network:
  for (int iter = 0; iter < 9000; iter++) {
    float target = float(iter % 9);
    setOneHot(A->getOutput().weight_, 0.0, target, 0);
    setOneHot(RN->getExpected(), 0.0, target + 1, 1);

    N.train();
  }

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    setOneHot(A->getOutput().weight_, 0.0, target, 0);
    setOneHot(RN->getExpected(), 0.0, target + 1, 1);

    N.infer();
    assert(delta(A->getOutput().weight_.at(0, 0, 0) + 1,
                 RN->getOutput().weight_.at(0, 0, 1)) < 0.1);
  }
}

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

  /// Load the MNIST database into a 4d tensor.
  Array4D<float> data(50000, 28, 28, 1);
  size_t idx = 0;

  for (size_t w = 0; w < mnistNumImages; w++) {
      for (size_t y = 0; y < 28; y++) {
        for (size_t x = 0; x < 28; x++) {
          data.at(w,x,y,0) = imagesAsFloatPtr[idx++];
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

  auto *A= N.createArrayNode(28, 28, 1);
  auto *CV0 = N.createConvNode(A, 8, 5, 1, 2);
  auto *RL0 = N.createRELUNode (CV0);
  auto *MP0= N.createMaxPoolNode (RL0, 2, 2, 0);

  auto *CV1= N.createConvNode (MP0, 16, 5, 1, 2);
  auto *RL1= N.createRELUNode (CV1);
  auto *MP1 = N.createMaxPoolNode (RL1, 2, 2, 0);

  auto *FCL1 = N.createFullyConnectedNode (MP1, 10);
  auto *RL2 = N.createRELUNode (FCL1);
  auto *SM= N.createSoftMaxNode (RL2);

  if (verbose) {
    std::cout << "Training.\n";
  }

  for (int iter = 0; iter < 5000; iter++) {
    if (verbose && !(iter % 1000)) {
      std::cout << "Training - iteration #" << iter << "\n";
    }

    size_t imageIndex = iter % numImages;

    A->getOutput().weight_ = data.extractSlice(imageIndex);
    SM->setSelected(labels[imageIndex]);

    N.train();
  }

  if (verbose) {
    std::cout << "Validating.\n";
  }

  // Check how many digits out of ten we can classify correctly.
  int rightAnswer = 0;

  for (int iter = 0; iter < 10; iter++) {
    size_t imageIndex = (iter * 17512 + 9124) % numImages;
    A->getOutput().weight_ = data.extractSlice(imageIndex);

    N.infer();

    size_t guess = SM->maxArg();
    size_t correct = labels[imageIndex];
    rightAnswer += (guess == correct);

    if (verbose) {
      A->getOutput().weight_.dumpAscii("MNIST Input");
      std::cout << "Expected: " << correct
                << " Guessed: " << guess << "\n";
      SM->getOutput().weight_.dump("", "\n");
      std::cout << "\n-------------\n";
    }
  }

  assert(rightAnswer >= 6 && "Did not classify as many digits as expected");
}

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
  (void) cifarImageSize;
  if (verbose) {
    std::cout << "Loading the mnist database.\n";
  }

  const char* textualLabels[] = { "airplane", "automobile", "bird", "cat",
    "deer", "dog", "frog", "horse", "ship", "truck"};

  std::ifstream dbInput("cifar-10-batches-bin/data_batch_1.bin",
                        std::ios::binary);

  if (verbose) {
    std::cout << "Loaded the CIFAR-10 database.\n";
  }

  /// Load the CIFAR database into a 4d tensor.
  Array4D<float> data(cifarNumImages, 32, 32, 3);
  std::vector<uint8_t> labels;
  size_t idx = 0;

  for (size_t w = 0; w < cifarNumImages; w++) {
    labels.push_back(static_cast<uint8_t>(dbInput.get()));
    idx++;

    for (size_t z = 0; z < 3; z++) {
      for (size_t y = 0; y < 32; y++) {
        for (size_t x = 0; x < 32; x++) {
          data.at(w,x,y,z) = FloatTy(static_cast<uint8_t>(dbInput.get()))/255.0;
          idx++;
        }
      }
    }
  }
  assert(idx == cifarImageSize * cifarNumImages && "Invalid input file");


  // Construct the network:
  Network N;
  N.getTrainingConfig().learningRate = 0.01;
  N.getTrainingConfig().momentum = 0.9;
  N.getTrainingConfig().batchSize = 10;
  N.getTrainingConfig().L2Decay = 0.0001;

  auto *A= N.createArrayNode(32, 32, 3);
  auto *CV0= N.createConvNode (A, 16, 5, 1, 2);
  auto *RL0= N.createRELUNode (CV0);
  auto *MP0= N.createMaxPoolNode (RL0, 2, 2, 0);

  auto *CV1= N.createConvNode (MP0, 20, 5, 1, 2);
  auto *RL1= N.createRELUNode (CV1);
  auto *MP1= N.createMaxPoolNode (RL1, 2, 2, 0);

  auto *CV2= N.createConvNode (MP1, 20, 5, 1, 2);
  auto *RL2= N.createRELUNode (CV2);
  auto *MP2= N.createMaxPoolNode (RL2, 2, 2, 0);

  auto *FCL1 = N.createFullyConnectedNode(MP2, 10);
  auto *RL3= N.createRELUNode (FCL1);
  auto *SM= N.createSoftMaxNode(RL3);

  if (verbose) {
    std::cout << "Training.\n";
  }

  for (int iter = 0; iter < 18000; iter++) {
    if (verbose && !(iter % 100)) {
      std::cout << "Training - iteration #" << iter << "\n";
    }

    const size_t imageIndex = iter % cifarNumImages;

    // Load the image.
    A->getOutput().weight_ = data.extractSlice(imageIndex);
    // Set the expected label.
    auto label =  labels[imageIndex];
    SM->setSelected(label);

    N.train();
  }

  if (verbose) {
    std::cout << "Validating.\n";
  }

  for (int iter = 0; iter < 10; iter++) {
    // Pick a random image from the stack:
    const size_t imageIndex = (iter * 17512 + 9124) % cifarNumImages;

    // Load the image.
    A->getOutput().weight_ = data.extractSlice(imageIndex);
    // Load the expected label.
    auto expectedLabel =  labels[imageIndex];

    N.infer();

    unsigned result = SM->maxArg();
    std::cout << "Expected: " << textualLabels[expectedLabel] << " Guessed: " <<
    textualLabels[result] << "\n";
  }
}

int main() {

  testArray();

  testLearnSingleInput();

  testRegression();

  testFCSoftMax();

  testMNIST();

  testCIFAR10();

  return 0;
}
