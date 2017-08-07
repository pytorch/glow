#include "noether/Tensor.h"

#include "gtest/gtest.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <algorithm>

using namespace noether;

TEST(Tensor, init) {
  std::cout << "Testing initialization.\n";

  Tensor T = {1.2, 12.1, 51.0, 1515.2};

  auto H = T.getHandle<FloatTy>();

  H.dump("", "\n");

  EXPECT_EQ(int(H.at({2})), 51);

  H = {1.1, 1.2, 1.3, 1.4};

  EXPECT_EQ(int(H.at({0})), 1);

  H.dump("", "\n");
}

TEST(Tensor, clone) {
  std::cout << "Testing clone.\n";

  Tensor T = {1.2, 12.1, 51.0, 1515.2};
  auto H = T.getHandle<FloatTy>();

  Tensor v;
  v.copyFrom(&T);
  auto vH = v.getHandle<FloatTy>();

  EXPECT_EQ(int(vH.at({0})), 1);

  // Update the original tensor
  H = {0.11, 0.22, 0.33, 0.44};

  // The cloned vector is unmodified.
  EXPECT_EQ(int(vH.at({1})), 12);
}

TEST(Tensor, assignment) {
  //Testing some tensor operations.
  Tensor T(ElemKind::FloatTy, {320, 200, 64});

  auto Handle = T.getHandle<FloatTy>();

  for (unsigned i = 0; i < 10; i++) {
    for (unsigned x = 0; x < 32; x++) {
      for (unsigned y = 0; y < 20; y++) {
        for (unsigned z = 0; z < 64; z++) {
          Handle.at({x, y, z}) = x + y + z;
        }
      }
    }
  }

  EXPECT_EQ(Handle.at({10, 10, 10}) , 10 + 10 + 10);

  auto TT = Handle.extractSlice(1);
  auto H2 = TT.getHandle<FloatTy>();

  EXPECT_EQ(H2.at({10, 10}) , 1 + 10 + 10);

  for (unsigned y = 0; y < 20; y++) {
    for (unsigned z = 0; z < 64; z++) {
      H2.at({y, z}) = 2;
    }
  }

  EXPECT_EQ(H2.at({10, 10}), 2);
}


TEST(Tensor, concatTensors1D) {
  Tensor X = {1.1, 2.1, 3.1, 4.1};
  Tensor Y = {5.2, 6.2, 7.2, 8.2};
  Tensor Z = {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3};
  Tensor expected = {5.2, 6.2, 7.2, 8.2, 1.1, 2.1, 3.1, 4.1};

  auto xH = X.getHandle<FloatTy>();
  auto yH = Y.getHandle<FloatTy>();
  auto zH = Z.getHandle<FloatTy>();
  auto eH = expected.getHandle<FloatTy>();

  insertTensors<FloatTy>(xH, zH, {4});
  insertTensors<FloatTy>(yH, zH, {0});

  for (size_t i = 0, e = eH.size(); i < e; i++) {
    EXPECT_EQ(eH.at({i}), zH.at({i}));
  }
}

TEST(Tensor, concatTensors2D) {
  Tensor X(ElemKind::FloatTy, {10, 10});
  Tensor Y(ElemKind::FloatTy, {10, 10});
  Tensor Z(ElemKind::FloatTy, {20, 20});

  auto xH = X.getHandle<FloatTy>();
  auto yH = Y.getHandle<FloatTy>();
  auto zH = Z.getHandle<FloatTy>();

  // Create a nice picture:
  for (size_t i = 0, e = xH.size(); i < e; i++) {
    xH.raw(i) = (float(i) - 30) / 50;
  }

  // Insert the tensors and create a picture of three cards one on to of the
  // other.
  insertTensors<FloatTy>(xH, zH, {0, 0});
  insertTensors<FloatTy>(xH, zH, {5, 5});
  insertTensors<FloatTy>(xH, zH, {10, 10});

  zH.dumpAscii();

  /// Check some pixels in the image:
  EXPECT_EQ(zH.at({0,0}), xH.at({0,0}));
  EXPECT_EQ(zH.at({19,0}), 0);
  EXPECT_EQ(zH.at({0,19}), 0);
  EXPECT_EQ(zH.at({19,19}), xH.at({9,9}));
  EXPECT_EQ(zH.at({10,10}), xH.at({0,0}));


  // Extract an image from the tensor.
  extractTensors<FloatTy>(yH, zH, {10, 10});

  // Make sure that what we've extracted is equal to what we've inserted.
  for (size_t i = 0, e = xH.size(); i < e; i++) {
    EXPECT_EQ(yH.raw(i), xH.raw(i));
  }

}
