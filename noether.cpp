#include "noether/Image.h"
#include "noether/Nodes.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace noether;

void testArray() {
  Array3D<float> X(320, 200, 3);
  X.at(10u, 10u, 2u) = 2;
  assert((X.at(10u, 10u, 2u) == 2) && "Invalid load/store");
}


void testFCSoftMax() {
  Network N;

  ArrayNode<float> A(&N, 1,1,10);

  FullyConnectedNode<float> FCL0(&N, &A, 100);
  RELUNode<float> RL0(&N, &FCL0);

  FullyConnectedNode<float> FCL1(&N, &RL0, 10);
  RELUNode<float> RL1(&N, &FCL1);

  SoftMaxNode<float> SM(&N, &RL1);

  for (int iter = 0; iter < 19000; iter++) {
    int target = iter % 10;

    SM.setSelected(target);
    for (int j = 0; j < 10; j++) {
      A.getOutput().weight_.at(0,0, j) = (j == target);
    }

    N.train();
  }


  for (int iter = 0; iter < 10; iter++) {

    int target = iter % 10;

    SM.setSelected(target);
    for (int j = 0; j < 10; j++) {
      A.getOutput().weight_.at(0,0, j) = (j == target);
    }

    A.getOutput().weight_.dump("Input: ","\n");
    N.infer();
    SM.getOutput().weight_.dump("Output:","\n");
  }
}


void setData(Array3D<float> &A, int seed) {
  A.clear();
  for (int j = 0; j < A.size(); j++) {
    A.at(0, 0, 0) = seed;
  }
}

void testRegression() {
  Network N;

  constexpr int numInputs = 4;

  ArrayNode<float> A(&N, 1, 1, numInputs);

  FullyConnectedNode<float> FCL0(&N, &A, 10);
  RELUNode<float> RL0(&N, &FCL0);

  RegressionNode<float> RN(&N, &RL0);

  // Train the network:
  for (int iter = 0; iter < 9000; iter++) {
    float target = iter % 9 + 1;

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

    A.getOutput().weight_.dump("A w:");

    N.infer();

    RN.getOutput().weight_.dump("Network output: ", "\n");
  }

}


int main() {
  testArray();

  testRegression();


  testFCSoftMax();
}
