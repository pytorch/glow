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

  PNGLayer<float> In(&N);
  In.readImage("./build/map.png");
  ConvLayer<float> CL(&N, &In, 30, 5, 2, 0);
  FullyConnectedLayer<float> FL(&N, &CL, 100);
  RELULayer<float> RL(&N, &In);
  SoftMaxLayer<float> SM(&N, &FL, 1);
  SM.forward();

  //x.writeImage("./map2.png");
}
