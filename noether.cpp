#include "noether/Image.h"
#include "noether/Layers.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace noether;

int main() {

  Array3D<float> X(320, 200, 3);
  X.at(10u, 10u, 2u) = 2;

  Network N;
  /*PNGLayer<float> In(&N);
  In.readImage("./build/map.png");
  ConvLayer<float> CL(&N, &In, 30, 5, 2, 0);
  FullyConnectedLayer<float> FL(&N, &CL, 100);
  RELULayer<float> RL(&N, &In);
  SoftMaxLayer<float> SSML(&N, &FL, 1);
*/

  ArrayLayer<float> A(&N, 1,1,10);

  FullyConnectedLayer<float> FCL0(&N, &A, 100);
  //RELULayer<float> RL0(&N, &FCL0);

  //FullyConnectedLayer<float> FCL1(&N, &RL0, 10);
  //RELULayer<float> RL1(&N, &FCL1);

  SoftMaxLayer<float> SM(&N, &FCL0);

  for (int iter = 0; iter < 100000; iter++) {

    int target = iter % 10;

    SM.setSelected(target);
    for (int j = 0; j < 10; j++) {
      A.getOutput().weight_.at(0,0, j) = (j == target);
    }

    N.train();


    for (int j = 0; j < 10; j++) {
      A.getOutput().weight_.at(0,0, j) = (j == target);
    }

    N.infer();

    std::cout<<"Inputs: [";
    for (int i = 0; i < 10; i++) { std::cout<<  A.getOutput().weight_.at(0, 0, i) << " "; }
    std::cout<<"]\n";


    std::cout<<"Outputs: [";
    for (int i = 0; i < 10; i++) { std::cout<<  SM.getOutput().weight_.at(0, 0, i) << " "; }
    std::cout<<"]\n";
  }


  //x.writeImage("./map2.png");
}
