// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Base/Tensor.h"

#include "gtest/gtest.h"

using namespace glow;

TEST(Tensor, init) {
  Tensor T = {1.2, 12.1, 51.0, 1515.2};

  auto H = T.getHandle<>();

  H.dump();

  EXPECT_EQ(int(H.at({2})), 51);

  H = {1.1, 1.2, 1.3, 1.4};

  EXPECT_EQ(int(H.at({0})), 1);

  H.dump();
}

TEST(Tensor, clone) {
  Tensor T = {1.2, 12.1, 51.0, 1515.2};
  auto H = T.getHandle<>();

  Tensor v;
  v.copyFrom(&T);
  auto vH = v.getHandle<>();

  EXPECT_EQ(int(vH.at({0})), 1);

  // Update the original tensor
  H = {0.11, 0.22, 0.33, 0.44};

  // The cloned vector is unmodified.
  EXPECT_EQ(int(vH.at({1})), 12);
}

TEST(Tensor, assignment) {
  // Testing some tensor operations.
  Tensor T(ElemKind::FloatTy, {320, 200, 64});

  auto Handle = T.getHandle<>();

  for (unsigned i = 0; i < 10; i++) {
    for (unsigned x = 0; x < 32; x++) {
      for (unsigned y = 0; y < 20; y++) {
        for (unsigned z = 0; z < 64; z++) {
          Handle.at({x, y, z}) = x + y + z;
        }
      }
    }
  }

  EXPECT_EQ(Handle.at({10, 10, 10}), 10 + 10 + 10);

  auto TT = Handle.extractSlice(1);
  auto H2 = TT.getHandle<>();

  EXPECT_EQ(H2.at({10, 10}), 1 + 10 + 10);

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

  auto xH = X.getHandle<>();
  auto yH = Y.getHandle<>();
  auto zH = Z.getHandle<>();
  auto eH = expected.getHandle<>();

  zH.insertTensors(xH, {4});
  zH.insertTensors(yH, {0});

  for (size_t i = 0, e = eH.size(); i < e; i++) {
    EXPECT_EQ(eH.at({i}), zH.at({i}));
  }
}

TEST(Tensor, concatTensors2D) {
  Tensor X(ElemKind::FloatTy, {10, 10});
  Tensor Y(ElemKind::FloatTy, {10, 10});
  Tensor Z(ElemKind::FloatTy, {20, 20});

  auto xH = X.getHandle<>();
  auto yH = Y.getHandle<>();
  auto zH = Z.getHandle<>();

  // Create a nice picture:
  for (size_t i = 0, e = xH.size(); i < e; i++) {
    xH.raw(i) = (float(i) - 30) / 50;
  }

  // Insert the tensors and create a picture of three cards one on to of the
  // other.
  zH.insertTensors(xH, {0, 0});
  zH.insertTensors(xH, {5, 5});
  zH.insertTensors(xH, {10, 10});

  zH.dumpAscii();

  /// Check some pixels in the image:
  EXPECT_EQ(zH.at({0, 0}), xH.at({0, 0}));
  EXPECT_EQ(zH.at({19, 0}), 0);
  EXPECT_EQ(zH.at({0, 19}), 0);
  EXPECT_EQ(zH.at({19, 19}), xH.at({9, 9}));
  EXPECT_EQ(zH.at({10, 10}), xH.at({0, 0}));

  // Extract an image from the tensor.
  zH.extractTensors(yH, {10, 10});

  // Make sure that what we've extracted is equal to what we've inserted.
  for (size_t i = 0, e = xH.size(); i < e; i++) {
    EXPECT_EQ(yH.raw(i), xH.raw(i));
  }
}

TEST(Tensor, MeanAndVariance) {

  Tensor T1 = {3, 4, 4, 5, 6, 8};
  Tensor T2 = {1, 2, 4, 5, 7, 11};

  auto H1 = T1.getHandle<>();
  auto H2 = T2.getHandle<>();

  auto MV1 = H1.calculateMeanVariance();
  auto MV2 = H2.calculateMeanVariance();

  EXPECT_EQ(int(MV1.first), 5);
  EXPECT_NEAR(MV1.second, 3.2, 0.01);
  EXPECT_EQ(int(MV2.first), 5);
  EXPECT_NEAR(MV2.second, 13.2, 0.01);
}

TEST(Tensor, getDimForPtr) {
  // Testing some tensor operations.
  Tensor T(ElemKind::FloatTy, {10, 5, 3});
  auto H = T.getHandle<>();

  for (unsigned x = 0; x < 10; x++) {
    for (unsigned y = 0; y < 5; y++) {
      for (unsigned z = 0; z < 3; z++) {
        size_t ptr = H.getElementPtr({x, y, z});
        EXPECT_EQ(x, H.getDimForPtr(0, ptr));
        EXPECT_EQ(y, H.getDimForPtr(1, ptr));
        EXPECT_EQ(z, H.getDimForPtr(2, ptr));
      }
    }
  }
}

TEST(Tensor, copySlice) {
  // Testing some tensor operations.
  Tensor A(ElemKind::FloatTy, {10, 5, 3});
  Tensor B(ElemKind::FloatTy, {5, 3});

  auto AH = A.getHandle<>();
  auto BH = B.getHandle<>();

  AH.randomize(1);

  B.copySlice(&A, 0);

  for (unsigned y = 0; y < 5; y++) {
    for (unsigned z = 0; z < 3; z++) {
      EXPECT_EQ(AH.at({0, y, z}), BH.at({y, z}));
    }
  }
}

TEST(Tensor, transpose) {
  Tensor X(ElemKind::FloatTy, {5, 2});
  auto H = X.getHandle<>();
  H = {
      0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 0.8, 1.0, 2.0, 3.0,
  };

  Tensor Xhat;
  X.getHandle<>().transpose(&Xhat, {1, 0});

  auto XhatH = Xhat.getHandle<>();

  for (size_t i = 0; i < 5; i++) {
    EXPECT_EQ(H.at({i, 0}), XhatH.at({0, i}));
    EXPECT_EQ(H.at({i, 1}), XhatH.at({1, i}));
  }
}

TEST(Tensor, transpose2) {
  Tensor X(ElemKind::FloatTy, {10, 6, 3});
  auto H = X.getHandle<>();
  H.randomize(10);

  Tensor Xhat;
  X.getHandle<>().transpose(&Xhat, {1, 2, 0});

  auto XhatH = Xhat.getHandle<>();

  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 6; j++) {
      for (size_t k = 0; k < 3; k++) {
        EXPECT_EQ(H.at({i, j, k}), XhatH.at({j, k, i}));
      }
    }
  }
}
