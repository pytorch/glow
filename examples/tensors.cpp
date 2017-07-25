#include "noether/Tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

using namespace noether;

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
  testTensor();

  return 0;
}
