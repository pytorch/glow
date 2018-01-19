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

TEST(Tensor, meanAndVariance) {

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

TEST(Tensor, nonOwnedTensor) {
  Tensor T1 = {1.2, 12.1, 51.0, 1515.2};

  auto H1 = T1.getHandle<>();
  H1.dump();
  EXPECT_EQ(int(H1.at({0})), 1);

  {
    // Create a view on T1 which makes it look like 2x2
    Tensor T2 = T1.getUnowned({2, 2});
    EXPECT_EQ(T2.getRawDataPointer<float>(), T1.getRawDataPointer<float>());
    auto H2 = T2.getHandle<>();
    // Check that T2 has the same values as T1.
    EXPECT_EQ(int(H2.at({0, 0})), 1);
    EXPECT_EQ(int(H2.at({0, 1})), 12);
    EXPECT_EQ(int(H2.at({1, 0})), 51);
    EXPECT_EQ(int(H2.at({1, 1})), 1515);
    // Modify a value through T2.
    H2.at({1, 1}) = 30.3;
    EXPECT_EQ(int(H2.at({1, 1})), 30);
    // Modify a value through T1 and check
    // that this update is visible through
    // T2.
    H1.at({1}) = 40.4;
    EXPECT_EQ(int(H2.at({0, 1})), 40);
    H2.dump();
  }

  // Check that modifications through T2 changed
  // T1 as well, i.e. T2 was acting like a view
  // on T1.
  EXPECT_EQ(int(H1.at({3})), 30);

  // Check that T1 is still alive
  H1.dump();
}

TEST(Tensor, broadcastDir0) {
  const unsigned dimX = 3;
  const unsigned dimY = 4;
  const unsigned broadcastDim = 5;
  Tensor X(ElemKind::FloatTy, {dimX, dimY});
  auto H = X.getHandle<>();
  H = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111};

  Tensor broadcastedX;
  const unsigned direction = 0;
  const bool addingNewDim = true;
  X.getHandle<>().broadcastOneDimension(&broadcastedX, broadcastDim, direction,
                                        addingNewDim);

  auto broadcastedXHandle = broadcastedX.getHandle<>();

  for (size_t i = 0; i < dimX; i++) {
    for (size_t j = 0; j < dimY; j++) {
      const float origVal = H.at({i, j});
      for (size_t k = 0; k < broadcastDim; k++) {
        EXPECT_EQ(origVal, broadcastedXHandle.at({k, i, j}));
      }
    }
  }
}

TEST(Tensor, broadcastDir1) {
  const unsigned dimX = 3;
  const unsigned dimY = 4;
  const unsigned broadcastDim = 5;
  Tensor X(ElemKind::FloatTy, {dimX, dimY});
  auto H = X.getHandle<>();
  H = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111};

  Tensor broadcastedX;
  const unsigned direction = 1;
  const bool addingNewDim = true;
  X.getHandle<>().broadcastOneDimension(&broadcastedX, broadcastDim, direction,
                                        addingNewDim);

  auto broadcastedXHandle = broadcastedX.getHandle<>();

  for (size_t i = 0; i < dimX; i++) {
    for (size_t j = 0; j < dimY; j++) {
      const float origVal = H.at({i, j});
      for (size_t k = 0; k < broadcastDim; k++) {
        EXPECT_EQ(origVal, broadcastedXHandle.at({i, k, j}));
      }
    }
  }
}

TEST(Tensor, broadcastDir2) {
  const unsigned dimX = 3;
  const unsigned dimY = 4;
  const unsigned broadcastDim = 5;
  Tensor X(ElemKind::FloatTy, {dimX, dimY});
  auto H = X.getHandle<>();
  H = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111};

  Tensor broadcastedX;
  const unsigned direction = 2;
  const bool addingNewDim = true;
  X.getHandle<>().broadcastOneDimension(&broadcastedX, broadcastDim, direction,
                                        addingNewDim);

  auto broadcastedXHandle = broadcastedX.getHandle<>();

  for (size_t i = 0; i < dimX; i++) {
    for (size_t j = 0; j < dimY; j++) {
      const float origVal = H.at({i, j});
      for (size_t k = 0; k < broadcastDim; k++) {
        EXPECT_EQ(origVal, broadcastedXHandle.at({i, j, k}));
      }
    }
  }
}

/// Broadcast a Tensor of shape (2,1) to (3,2,2,2).
TEST(Tensor, broadcastNewShape) {
  const size_t numDims_A = 4;

  const size_t dimX_A = 2;
  const size_t dimY_A = 2;
  const size_t dimZ_A = 2;
  const size_t dimW_A = 3;

  size_t dims_A[numDims_A];
  dims_A[0] = dimX_A;
  dims_A[1] = dimY_A;
  dims_A[2] = dimZ_A;
  dims_A[3] = dimW_A;

  const size_t dimX_B = 1;
  const size_t dimY_B = 2;
  Tensor X_B(ElemKind::FloatTy, {dimX_B, dimY_B});
  auto H_B = X_B.getHandle<>();
  H_B = {200, 201};

  Tensor broadcastedB;
  X_B.getHandle<>().broadcastToNewShape(&broadcastedB, dims_A);

  auto broadcastedBHandle = broadcastedB.getHandle<>();

  // Verify broadcasted B has same shape.
  EXPECT_EQ(numDims_A, broadcastedBHandle.dims().size());
  for (int i = 0; i < numDims_A; i++) {
    EXPECT_EQ(dims_A[i], broadcastedBHandle.dims()[i]);
  }

  // Look at the two values in X_B (in dimension Y) and verify in the three
  // dimensions it was broadcasted that the values were correctly broadcasted.
  const size_t i_B = 0;
  for (size_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({i_B, j_B});
    const size_t j_A = j_B; // This dim was not broadcasted.
    for (size_t i_A = 0; i_A < dimX_A; ++i_A) {
      for (size_t k_A = 0; k_A < dimZ_A; ++k_A) {
        for (size_t l_A = 0; l_A < dimW_A; ++l_A) {
          EXPECT_EQ(origVal, broadcastedBHandle.at({i_A, j_A, k_A, l_A}));
        }
      }
    }
  }
}
