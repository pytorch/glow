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

float delta(float a, float b) { return std::fabs(a - b); }


/// Generate data in two classes. The circle of dots that's close to the axis is
/// L0, and the rest of the dots, away from the axis are L1.
void generateCircleData(Tensor &coordinates, Tensor &labels) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> r_radius(0, 0.4);
  std::uniform_real_distribution<> r_angle(0, 3.14159 * 2);

  auto C = coordinates.getHandle<FloatTy>();
  auto L = labels.getHandle<size_t>();

  for (size_t i = 0; i < 50; i++) {
    float r = r_radius(gen);
    float a = r_angle(gen);
    float y = r * sin(a);
    float x = r * cos(a);

    C.at({i*2, 0u}) = x;
    C.at({i*2, 1u}) = y;
    L.at({i*2}) = 1;

    r = r_radius(gen) + 0.8;
    a = r_angle(gen);
    y = r * sin(a);
    x = r * cos(a);

    C.at({i * 2 + 1, 0u}) = x;
    C.at({i * 2 + 1, 1u}) = y;
    L.at({i * 2 + 1}) = 0;
  }
}


/// Test the fully connected layer and the softmax function.
/// Example from:
/// http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
void testFCSoftMax(bool verbose = false) {
  if (verbose) {
    std::cout << "Testing the softmax layer.\n";
  }

  // Construct the network:
  Network N;
  N.getTrainingConfig().momentum = 0.0;
  N.getTrainingConfig().learningRate = 0.1;
  N.getTrainingConfig().batchSize = 10;
  N.getTrainingConfig().inputSize = 100;

  auto *A = N.createArrayNode({2});
  auto *FCL0 = N.createFullyConnectedNode(A, 6);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 2);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL1);


  Tensor coordinates(ElemKind::FloatTy, {100, 2});
  Tensor labels(ElemKind::IndexTy, {100});
  generateCircleData(coordinates, labels);

  // Setup a handle to access array A and SM.
  auto AWH = A->getOutput().weight_.getHandle<FloatTy>();
  auto SMH = SM->getOutput().weight_.getHandle<FloatTy>();

  // On each training iteration the inputs are loaded from the image db.
  A->bind(&coordinates);

  // On each  iteration the expected value is loaded from the labels vector.
  SM->bind(&labels);

  std::cout << "Training.\n";

  for (int iter = 0; iter < 2000; iter++) {
    N.train(SM, 10);
  }
  // Print a diagram that depicts the network decision on a grid.
  if (verbose) {

    for (int x = -10; x < 10; x++) {
      for (int y = -10; y < 10; y++) {
        // Load the inputs:
        AWH.at({0}) = float(x) / 10;
        AWH.at({1}) = float(y) / 10;

        N.infer(SM);
        auto A = SMH.at({0});
        auto B = SMH.at({1});

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
    std::cout << "\n";
  }
}

/// A helper function to load a one-hot vector.
void setOneHot(Tensor &A, float background, float foreground,
               size_t idx) {
  auto H = A.getHandle<FloatTy>();
  for (unsigned j = 0; j < A.size(); j++) {
    H.at({j}) = (j == idx ? foreground : background);
  }
}

void testRegression(bool verbose = false) {
  if (verbose) {
    std::cout << "Testing the regression layer.\n";
  }

  /// This test takes the first element from the input vector, adds one to it
  /// and places the result in the second element of the output vector.
  constexpr int numInputs = 4;

  Network N;
  auto *A = N.createArrayNode({numInputs});
  auto *FCL0 = N.createFullyConnectedNode(A, 4);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *RN = N.createRegressionNode(RL0);

  // Train the network:
  for (int iter = 0; iter < 9000; iter++) {
    float target = float(iter % 9);
    setOneHot(A->getOutput().weight_, 0.0, target, 0);
    setOneHot(RN->getExpected(), 0.0, target + 1, 1);

    N.train(RN);
  }
  if (verbose) {
    std::cout << "Verify the result of the regression layer.\n";
  }

  auto AWH = A->getOutput().weight_.getHandle<FloatTy>(); (void) AWH;
  auto RNWH = RN->getOutput().weight_.getHandle<FloatTy>(); (void) RNWH;

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    setOneHot(A->getOutput().weight_, 0.0, target, 0);
    setOneHot(RN->getExpected(), 0.0, target + 1, 1);

    N.infer(RN);
    assert(delta(AWH.at({0}) + 1, RNWH.at({1})) < 0.1);
  }
  if (verbose) {
    std::cout << "Done.\n";
  }
}

void testLearnSingleInput(bool verbose = false) {
  if (verbose) {
    std::cout << "Learning a single input vector.\n";
  }

  Network N;
  N.getTrainingConfig().learningRate = 0.05;
  auto *A = N.createArrayNode(10);
  auto *FCL0 = N.createFullyConnectedNode(A, 10);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 10);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *RN = N.createRegressionNode(RL1);

  // Put in [15, 0, 0, 0, 0 ... ]
  setOneHot(A->getOutput().weight_, 0.0, 15, 0);
  // Expect [0, 9.0, 0 , 0 , ...]
  setOneHot(RN->getExpected(), 0.0, 9.0, 1);

  // Train the network:
  for (int iter = 0; iter < 10000; iter++) {
    N.train(RN);
  }

  if (verbose) {
    std::cout << "Testing the output vector.\n";
  }

  N.infer(RN);

  auto RNWH = RN->getOutput().weight_.getHandle<FloatTy>(); (void) RNWH;

  // Test the output:
  assert(RNWH.at({1}) > 8.5);

  if (verbose) {
    std::cout << "Done.\n";
  }
}

int main() {

  testLearnSingleInput(true);

  testRegression(true);

  testFCSoftMax(true);

  return 0;
}
