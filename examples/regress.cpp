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

unsigned numSamples = 100;

/// Generate data in two classes. The circle of dots that's close to the axis is
/// L0, and the rest of the dots, away from the axis are L1.
void generateCircleData(Tensor &coordinates, Tensor &labels) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> r_radius(0, 0.4);
  std::uniform_real_distribution<> r_angle(0, 3.14159 * 2);

  auto C = coordinates.getHandle<FloatTy>();
  auto L = labels.getHandle<size_t>();

  for (size_t i = 0; i < numSamples / 2; i++) {
    float r = r_radius(gen);
    float a = r_angle(gen);
    float y = r * sin(a);
    float x = r * cos(a);

    C.at({i * 2, 0u}) = x;
    C.at({i * 2, 1u}) = y;
    L.at({i * 2}) = 1;

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
  N.getConfig().momentum = 0.0;
  N.getConfig().learningRate = 0.1;
  N.getConfig().batchSize = 10;

  auto *A = N.createArrayNode({2});
  auto *FCL0 = N.createFullyConnectedNode(A, 6);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 2);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *SM = N.createSoftMaxNode(RL1);

  Tensor coordinates(ElemKind::FloatTy, {numSamples, 2});
  Tensor labels(ElemKind::IndexTy, {numSamples});
  generateCircleData(coordinates, labels);

  std::cout << "Training.\n";

  for (int iter = 0; iter < 2000; iter++) {
    N.train(SM, 1, {A, SM}, {&coordinates, &labels});
  }

  // Print a diagram that depicts the network decision on a grid.
  if (verbose) {

    for (int x = -10; x < 10; x++) {
      for (int y = -10; y < 10; y++) {
        // Load the inputs:
        Tensor sample(ElemKind::FloatTy, {2});
        sample.getHandle<FloatTy>() = {float(x) / 10, float(y) / 10};

        auto res = N.infer(SM, {A}, {&sample});

        auto SMH = res->getHandle<FloatTy>();

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
void setOneHot(Tensor &A, float background, float foreground, size_t idx) {
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

  Tensor inputs(ElemKind::FloatTy, {numInputs});
  Tensor expected(ElemKind::FloatTy, {numInputs});
  auto I = inputs.getHandle<FloatTy>();
  auto E = expected.getHandle<FloatTy>();

  // Train the network:
  for (int iter = 0; iter < 9000; iter++) {
    float target = float(iter % 9);
    I = {target, 0., 0., 0.};
    E = {0., target + 1, 0., 0.};
    N.train(RN, {A, RN}, {&inputs, &expected});
  }
  if (verbose) {
    std::cout << "Verify the result of the regression layer.\n";
  }

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    I = {target, 0., 0., 0.};
    auto *res = N.infer(RN, {A}, {&inputs});

    auto resH = res->getHandle<FloatTy>();
    (void)resH;

    assert(delta(I.at({0}) + 1, resH.at({1})) < 0.1);
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
  N.getConfig().learningRate = 0.05;

  auto *A = N.createArrayNode(4);
  auto *FCL0 = N.createFullyConnectedNode(A, 10);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 4);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *RN = N.createRegressionNode(RL1);

  Tensor inputs(ElemKind::FloatTy, {4});
  Tensor expected(ElemKind::FloatTy, {4});

  inputs.getHandle<FloatTy>() = {0.15, 0.15, 0.15, 0.15};
  expected.getHandle<FloatTy>() = {0.9, 0.9, 0.9, 0.9};

  // Train the network:
  for (int iter = 0; iter < 1000; iter++) {
    N.train(RN, {A, RN}, {&inputs, &expected});
  }

  if (verbose) {
    std::cout << "Testing the output vector.\n";
  }

  auto res = N.infer(RN, {A}, {&inputs});
  auto RNWH = res->getHandle<FloatTy>();
  (void)RNWH;

  // Test the output:
  assert(delta(RNWH.at({0}), 0.9) < 0.1);

  if (verbose) {
    std::cout << "Done.\n";
  }
}

void testLearnXor(bool verbose = false) {
  if (verbose) {
    std::cout << "Learning the Xor function.\n";
  }

  Network N;
  N.getConfig().learningRate = 0.1;

  auto *A = N.createArrayNode(2);
  auto *FCL0 = N.createFullyConnectedNode(A, 4);
  auto *RL0 = N.createRELUNode(FCL0);
  auto *FCL1 = N.createFullyConnectedNode(RL0, 1);
  auto *RL1 = N.createRELUNode(FCL1);
  auto *RN = N.createRegressionNode(RL1);

  Tensor inputs(ElemKind::FloatTy, {4, 2});
  Tensor expected(ElemKind::FloatTy, {4, 1});

  auto I = inputs.getHandle<FloatTy>();
  auto E = expected.getHandle<FloatTy>();

  /// The XOR lookup table:
  I.at({0, 0}) = 0;
  I.at({0, 1}) = 0;
  I.at({1, 0}) = 0;
  I.at({1, 1}) = 1;
  I.at({2, 0}) = 1;
  I.at({2, 1}) = 0;
  I.at({3, 0}) = 1;
  I.at({3, 1}) = 1;
  // Xor result:
  E.at({0, 0}) = 0;
  E.at({1, 0}) = 1;
  E.at({2, 0}) = 1;
  E.at({3, 0}) = 0;

  // Train the network:
  N.train(RN, 400, {A, RN}, {&inputs, &expected});

  if (verbose) {
    std::cout << "Testing the output vector.\n";
  }

  for (size_t i = 0; i < 4; i++) {
    Tensor in = I.extractSlice(i);
    Tensor exp = E.extractSlice(i);
    auto expH = exp.getHandle<FloatTy>();
    (void)expH;

    auto res = N.infer(RN, {A}, {&in});
    auto resH = res->getHandle<FloatTy>();
    (void)resH;

    // Test the output:
    assert(delta(expH.at({0}), resH.at({0})) < 0.1);
  }

  if (verbose) {
    std::cout << "Done.\n";
  }
}

int main() {

  testLearnSingleInput(true);

  testLearnXor(true);

  testRegression(true);

  testFCSoftMax(true);

  return 0;
}
