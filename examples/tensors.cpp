#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

using namespace noether;

void testInit() {
  std::cout<<"Testing initialization.\n";

  Tensor T = {1.2, 12.1, 51.0, 1515.2};

  auto H = T.getHandle<FloatTy>();

  H.dump("", "\n");

  assert(int(H.at({2})) == 51);

  H = {1.1, 1.2, 1.3, 1.4};

  assert(int(H.at({0})) == 1);
  H.dump("", "\n");
}

void testTensor() {
  std::cout<<"Testing some tensor operations.\n";
  Tensor T(ElemKind::FloatTy, {320, 200, 64});

  auto Handle = T.getHandle<FloatTy>();

  for (unsigned i = 0; i < 10; i++) {
    for (unsigned x = 0; x < 320; x++) {
      for (unsigned y = 0; y < 200; y++) {
        for (unsigned z = 0; z < 64; z++) {
          Handle.at({x,y,z}) = x + y + z;
        }
      }
    }
  }

  assert(Handle.at({10, 10, 10}) == 10 + 10 + 10);

  auto TT = Handle.extractSlice(1);
  auto H2 = TT.getHandle<FloatTy>();

  assert(H2.at({10, 10}) == 1 + 10 + 10);

  for (unsigned y = 0; y < 200; y++) {
    for (unsigned z = 0; z < 64; z++) {
      H2.at({y,z}) = 2;
    }
  }

  assert(H2.at({10, 10}) == 2);

  std::cout<<"Done.\n";
}


int main() {
  testInit();

  testTensor();

  return 0;
}
