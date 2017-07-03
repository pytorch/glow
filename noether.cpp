#include "noether/Image.h"
#include "noether/Layers.h"
#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
  Array3D<float> X(320, 200, 3);
  X.at(10u, 10u, 2u) = 2;
  PNGLayer<float> In;
  In.readImage("./build/map.png");
  ConvLayer<float> CL(&In, 30, 5, 2, 0);
  FullyConnectedLayer<float> FL(&CL, 100);

  FL.forward();

  //x.writeImage("./map2.png");
}
