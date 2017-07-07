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

/// Test the fully connected layer and the softmax function.
/// Example from:
/// http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
void testFCSoftMax(bool print = false) {

  // Construct the network:
  Network N;
  N.getTrainingConfig().momentum = 0.0;
  ArrayNode<float> A(&N, 1,1,2);
  FullyConnectedNode<float> FCL0(&N, &A, 6);
  RELUNode<float> RL0(&N, &FCL0);
  FullyConnectedNode<float> FCL1(&N, &RL0, 2);
  RELUNode<float> RL1(&N, &FCL1);
  SoftMaxNode<float> SM(&N, &RL1);

  // Generate some random numbers in the range -1 .. 1.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1, 1);

  // Generate lots of samples and learn them.
  for (int iter = 0; iter < 99000; iter++) {
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

  // Print a diagram that depicts the network decision on a grid.
  if (print) {
    for (int x = -10; x < 10; x++) {
      for (int y = -10; y < 10; y++) {
        // Load the inputs:
        A.getOutput().weight_.at(0,0, 0) = float(x)/10;
        A.getOutput().weight_.at(0,0, 1) = float(y)/10;

        N.infer();
        auto A = SM.getOutput().weight_.at(0,0,0);
        auto B = SM.getOutput().weight_.at(0,0,1);

        char ch = '=';
        if (A > (B + 0.2)) {
          ch = '+';
        } else if (B > (A + 0.2)) {
          ch = '-';
        }

        std::cout<<ch;
      }
      std::cout<<"\n";
    }
  }

  // Verify the label for some 10 random points.
  for (int iter = 0; iter < 10; iter++) {
    float x = dis(gen);
    float y = dis(gen);

    float r2 = (x*x + y*y);

    // Throw away confusing samples.
    if (r2 > 0.5 && r2 < 0.7) continue;

    // Load the inputs:
    A.getOutput().weight_.at(0,0, 0) = x;
    A.getOutput().weight_.at(0,0, 1) = y;

    N.infer();

    // Inspect the outputs:
    if (r2 < 0.50) {
      assert(SM.getOutput().weight_.at(0,0,1) > 0.90);
    }
    if (r2 > 0.7) {
      assert(SM.getOutput().weight_.at(0,0,0) > 0.90);
    }
  }
}

/// A helper function to load a one-hot vector.
void setOneHot(Array3D<float> &A, float background, float foreground,
               size_t idx) {
  for (int j = 0; j < A.size(); j++) {
    A.at(0, 0, j) = (j == idx ? foreground : background);
  }
}

void testLearnSingleInput() {
  Network N;
  N.getTrainingConfig().learningRate = 0.005;
  ArrayNode<float> A(&N, 1, 1, 10);
  FullyConnectedNode<float> FCL0(&N, &A, 10);
  RELUNode<float> RL0(&N, &FCL0);
  FullyConnectedNode<float> FCL1(&N, &A, 10);
  RELUNode<float> RL1(&N, &FCL1);
  RegressionNode<float> RN(&N, &RL1);

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
  assert(RN.getOutput().weight_.at(0,0, 1) > 8.5);
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
    assert(delta(A.getOutput().weight_.at(0,0,0) + 1, RN.getOutput().weight_.at(0,0,1)) < 0.1);
  }
}


int main() {
  testArray();

  testLearnSingleInput();

  testRegression();

  testFCSoftMax();
}
