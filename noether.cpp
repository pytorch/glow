#include "noether/Image.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace noether;

void testArray() {
  Array3D<float> X(320, 200, 3);
  X.at(10u, 10u, 2u) = 2;
  assert((X.at(10u, 10u, 2u) == 2) && "Invalid load/store");
}

float delta(float a, float b) {
  return std::fabs(a - b);
}

void testFCSoftMax() {
  Network N;
  /// Example from:
  /// http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html

  ArrayNode<float> A(&N, 1,1,2);

  FullyConnectedNode<float> FCL0(&N, &A, 6);
  RELUNode<float> RL0(&N, &FCL0);
  FullyConnectedNode<float> FCL1(&N, &RL0, 2);
  RELUNode<float> RL1(&N, &FCL1);
  SoftMaxNode<float> SM(&N, &RL1);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1, 1);

  for (int iter = 0; iter < 90000; iter++) {
    float x = dis(gen);
    float y = dis(gen);

    // Check if the dot falls within some inner circle.
    float r2 = (x*x + y*y);

    bool InCircle = r2 <0.6;

    SM.setSelected(InCircle);
    A.getOutput().weight_.at(0,0, 0) = x;
    A.getOutput().weight_.at(0,0, 1) = y;
    N.train();
  }

  // Verify the label for the 10 points.
  for (int iter = 0; iter < 10; iter++) {
    float x = dis(gen);
    float y = dis(gen);
    float r2 = (x*x + y*y);

    // Throw away confusing samples.
    if (r2 > 0.5 && r2 < 0.7) continue;

    A.getOutput().weight_.at(0,0, 0) = x;
    A.getOutput().weight_.at(0,0, 1) = y;

    N.infer();

    if (r2 < 0.50) {
      assert(SM.getOutput().weight_.at(0,0,1) > 0.90);
    }

    if (r2 > 0.7) {
      assert(SM.getOutput().weight_.at(0,0,0) > 0.90);
    }
  }
}

void setData(Array3D<float> &A, float seed) {
  A.clear();
  for (int j = 0; j < A.size(); j++) {
    A.at(0, 0, 0) = seed;
  }
}

void testRegression() {
  Network N;

  /// This test takes the first element from the input vector, adds one to it
  /// and places the result in the second element of the output vector.
  constexpr int numInputs = 4;

  ArrayNode<float> A(&N, 1, 1, numInputs);

  FullyConnectedNode<float> FCL0(&N, &A, 4);
  RELUNode<float> RL0(&N, &FCL0);

  RegressionNode<float> RN(&N, &RL0);

  // Train the network:
  for (int iter = 0; iter < 9000; iter++) {
    float target = float(iter % 9);
    setData(A.getOutput().weight_, target);
    RN.getExpected().clear();
    RN.getExpected().at(0, 0, 1) = target + 1;
    N.train();
  }

  // Test the output:
  for (int iter = 0; iter < 5; iter++) {
    float target = iter % 9 + 1;
    setData(A.getOutput().weight_, target);
    RN.getExpected().clear();
    RN.getExpected().at(0, 0, 1) = target + 1;

    N.infer();
    assert(delta(A.getOutput().weight_.at(0,0,0) + 1, RN.getOutput().weight_.at(0,0,1)) < 0.1);
  }
}


int main() {
  testArray();

  testRegression();

  testFCSoftMax();
}
