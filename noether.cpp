#include "noether/Image.h"
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
  ArrayNode A(&N, 1, 1, 2);
  FullyConnectedNode FCL0(&N, &A, 6);
  RELUNode RL0(&N, &FCL0);
  FullyConnectedNode FCL1(&N, &RL0, 2);
  RELUNode RL1(&N, &FCL1);
  SoftMaxNode SM(&N, &RL1);

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

    SM.setSelected(InCircle);
    A.getOutput().weight_.at(0, 0, 0) = x;
    A.getOutput().weight_.at(0, 0, 1) = y;
    N.train();
  }

  // Print a diagram that depicts the network decision on a grid.
  if (verbose) {
    for (int x = -10; x < 10; x++) {
      for (int y = -10; y < 10; y++) {
        // Load the inputs:
        A.getOutput().weight_.at(0, 0, 0) = float(x) / 10;
        A.getOutput().weight_.at(0, 0, 1) = float(y) / 10;

        N.infer();
        auto A = SM.getOutput().weight_.at(0, 0, 0);
        auto B = SM.getOutput().weight_.at(0, 0, 1);

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
    A.getOutput().weight_.at(0, 0, 0) = x;
    A.getOutput().weight_.at(0, 0, 1) = y;

    N.infer();

    // Inspect the outputs:
    if (r2 < 0.50) {
      assert(SM.maxArg() == 1);
    }
    if (r2 > 0.7) {
      assert(SM.maxArg() == 0);
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
  ArrayNode A(&N, 1, 1, 10);
  FullyConnectedNode FCL0(&N, &A, 10);
  RELUNode RL0(&N, &FCL0);
  FullyConnectedNode FCL1(&N, &RL0, 10);
  RELUNode RL1(&N, &FCL1);
  RegressionNode RN(&N, &RL1);

  // Put in [15, 0, 0, 0, 0 ... ]
  setOneHot(A.getOutput().weight_, 0.0, 15, 0);
  // Expect [0, 9.0, 0 , 0 , ...]
  setOneHot(RN.getExpected(), 0.0, 9.0, 1);

  // Train the network:
  for (int iter = 0; iter < 10000; iter++) {
    N.train();
  }

  N.infer();

  // Test the output:
  assert(RN.getOutput().weight_.sum() < 10);
  assert(RN.getOutput().weight_.at(0, 0, 1) > 8.5);
}

void testRegression() {
  Network N;

  /// This test takes the first element from the input vector, adds one to it
  /// and places the result in the second element of the output vector.
  constexpr int numInputs = 4;

  ArrayNode A(&N, 1, 1, numInputs);

  FullyConnectedNode FCL0(&N, &A, 4);
  RELUNode RL0(&N, &FCL0);

  RegressionNode RN(&N, &RL0);

  // Train the network:
  for (int iter = 0; iter < 9000; iter++) {
    float target = float(iter % 9);
    setOneHot(A.getOutput().weight_, 0.0, target, 0);
    setOneHot(RN.getExpected(), 0.0, target + 1, 1);

    N.train();
  }

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    setOneHot(A.getOutput().weight_, 0.0, target, 0);
    setOneHot(RN.getExpected(), 0.0, target + 1, 1);

    N.infer();
    assert(delta(A.getOutput().weight_.at(0, 0, 0) + 1,
                 RN.getOutput().weight_.at(0, 0, 1)) < 0.1);
  }
}

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

  ArrayNode A(&N, 28, 28, 1);
  ConvNode CV0(&N, &A, 8, 5, 1, 2);
  RELUNode RL0(&N, &CV0);
  MaxPoolNode MP0(&N, &RL0, 2, 2, 0);

  ConvNode CV1(&N, &MP0, 16, 5, 1, 2);
  RELUNode RL1(&N, &CV1);
  MaxPoolNode MP1(&N, &RL1, 2, 2, 0);

  FullyConnectedNode FCL1(&N, &MP1, 10);
  RELUNode RL2(&N, &FCL1);
  SoftMaxNode SM(&N, &RL2);

  if (verbose) {
    std::cout << "Training.\n";
  }

  for (int iter = 0; iter < 5000; iter++) {
    if (verbose && !(iter % 1000)) {
      std::cout << "Training - iteration #" << iter << "\n";
    }

    size_t imageIndex = iter % numImages;

    A.loadRaw(imagesAsFloatPtr + 28 * 28 * imageIndex, 28 * 28);
    SM.setSelected(labels[imageIndex]);

    N.train();
  }

  if (verbose) {
    std::cout << "Validating.\n";
  }

  // Check how many digits out of ten we can classify correctly.
  int rightAnswer = 0;

  for (int iter = 0; iter < 10; iter++) {
    size_t imageIndex = (iter * 17512 + 9124) % numImages;

    A.loadRaw(imagesAsFloatPtr + 28 * 28 * imageIndex, 28 * 28);

    N.infer();

    size_t guess = SM.maxArg();
    size_t correct = labels[imageIndex];
    rightAnswer += (guess == correct);

    if (verbose) {
      A.getOutput().weight_.dumpAscii("MNIST Input");
      std::cout << "Expected: " << correct
                << " Guessed: " << guess << "\n";
      SM.getOutput().weight_.dump("", "\n");
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

/// Loads a CIFAR image into the input vector \p A.
/// \returns the label that's associated with the image.
uint8_t loadCIFARImage(Array3D<FloatTy> &A, uint8_t *rawArray,
                       size_t imageIndex) {
  assert(imageIndex < cifarNumImages && "Invalid image index");

  // Find the start pointer for the image.
  const size_t imageBase = cifarImageSize * imageIndex;

  // The rest of the bytes are the image:
  size_t idx = 1;
  for (size_t z = 0; z < 3; z++) {
    for (size_t y = 0; y < 32; y++) {
      for (size_t x = 0; x < 32; x++) {
        uint8_t byte = rawArray[imageBase + idx];
        A.at(x,y,z) = FloatTy(byte)/255.0;
        idx++;
      }
    }
  }

  // Return the label, which is the first field.
  return rawArray[imageBase];
}

/// This test classifies digits from the CIFAR labeled dataset.
/// Details: http://www.cs.toronto.edu/~kriz/cifar.html
/// Dataset: http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
void testCIFAR10(bool verbose = false) {
  if (verbose) {
    std::cout << "Loading the mnist database.\n";
  }

  const char* textualLabels[] = { "airplane", "automobile", "bird", "cat",
    "deer", "dog", "frog", "horse", "ship", "truck"};

  std::ifstream dbInput("cifar-10-batches-bin/data_batch_1.bin",
                        std::ios::binary);

  std::vector<char> db((std::istreambuf_iterator<char>(dbInput)),
                       (std::istreambuf_iterator<char>()));

  assert(db.size() == cifarImageSize * cifarNumImages && "Invalid input file");

  // This is a pointer to the raw data.
  uint8_t *rawArray = reinterpret_cast<uint8_t *>(&db[0]);

  if (verbose) {
    std::cout << "Loaded the CIFAR-10 database.\n";
  }

  // Construct the network:
  Network N;
  N.getTrainingConfig().learningRate = 0.01;
  N.getTrainingConfig().momentum = 0.9;
  N.getTrainingConfig().batchSize = 10;
  N.getTrainingConfig().L2Decay = 0.0001;

  ArrayNode A(&N, 32, 32, 3);
  ConvNode CV0(&N, &A, 16, 5, 1, 2);
  RELUNode RL0(&N, &CV0);
  MaxPoolNode MP0(&N, &RL0, 2, 2, 0);

  ConvNode CV1(&N, &MP0, 20, 5, 1, 2);
  RELUNode RL1(&N, &CV1);
  MaxPoolNode MP1(&N, &RL1, 2, 2, 0);

  ConvNode CV2(&N, &MP1, 20, 5, 1, 2);
  RELUNode RL2(&N, &CV2);
  MaxPoolNode MP2(&N, &RL2, 2, 2, 0);

  FullyConnectedNode FCL1(&N, &MP2, 10);
  RELUNode RL3(&N, &FCL1);
  SoftMaxNode SM(&N, &RL3);

  if (verbose) {
    std::cout << "Training.\n";
  }

  for (int iter = 0; iter < 18000; iter++) {
    if (verbose && !(iter % 100)) {
      std::cout << "Training - iteration #" << iter << "\n";
    }

    const size_t imageIndex = iter % cifarNumImages;

    // Load the image.
    auto label =  loadCIFARImage(A.getOutput().weight_, rawArray, imageIndex);
    // Set the expected label.
    SM.setSelected(label);

    N.train();
  }

  if (verbose) {
    std::cout << "Validating.\n";
  }


  for (int iter = 0; iter < 10; iter++) {
    // Pick a random image from the stack:
    const size_t imageIndex = (iter * 17512 + 9124) % cifarNumImages;

    // Load the image.
    auto expectedLabel =  loadCIFARImage(A.getOutput().weight_, rawArray,
                                         imageIndex);
    
    N.infer();
    
    unsigned result = SM.maxArg();
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
