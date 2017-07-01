#include "noether/Layers.h"
#include "noether/Tensor.h"
#include "noether/Image.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include <cassert>

int main() {
  Array3D<float> X(320,200,3);
  X.get(10u,10u,2u) = 2;
  PNGLayer<float> x;
  x.readImage("./map.png");

  x.writeImage("./map2.png");
}
