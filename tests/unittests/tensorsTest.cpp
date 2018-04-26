/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

TEST(Tensor, randomizeInt) {
  Tensor T(ElemKind::Int8QTy, {10, 10}, 1.0, 0);
  auto H = T.getHandle<int8_t>();
  H.randomize(-50, 50);

  // Check that all of the numbers fall in the range -50 to 50.
  for (size_t i = 0, e = H.size(); i < e; i++) {
    EXPECT_NEAR(H.raw(0), 0, 50);
  }
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

TEST(Tensor, minMaxArg) {
  {
    Tensor T = {1, 10, 20, -1, 30};
    auto res = T.getHandle().minMaxArg();
    EXPECT_EQ(3, res.first);
    EXPECT_EQ(4, res.second);
  }

  {
    Tensor T = {1, 1, 1, 1, 1, 1};
    auto res = T.getHandle().minMaxArg();
    EXPECT_EQ(0, res.first);
    EXPECT_EQ(0, res.second);
  }
}

TEST(Tensor, isZero) {
  {
    Tensor T = {4, 0, 0, 0, 0};
    EXPECT_FALSE(T.getHandle<>().isZero());
  }

  {
    Tensor T = {0, 0, 0, 0, 0};
    EXPECT_TRUE(T.getHandle<>().isZero());
  }

  {
    Tensor T = {0, 0, 0, 0, 0, 5};
    EXPECT_FALSE(T.getHandle<>().isZero());
  }
}

TEST(Tensor, inBounds) {
  Tensor A(ElemKind::FloatTy, {15, 5, 3});

  EXPECT_TRUE(A.isInBounds({14, 4, 2}));
  EXPECT_TRUE(A.isInBounds({0, 0, 0}));
  EXPECT_FALSE(A.isInBounds({15, 4, 2}));
  EXPECT_FALSE(A.isInBounds({5, 4, 3}));
}

TEST(Tensor, equalHandles) {
  {
    Tensor A = {1.0, 20};
    Tensor B = {1.0};
    EXPECT_FALSE(A.isEqual(B));
  }

  {
    Tensor A = {1.0, 20};
    Tensor B = {1.0, 20};
    EXPECT_TRUE(A.isEqual(B));
  }

  {
    Tensor A = {1.0, 20};
    Tensor B = {1.0, 30};
    EXPECT_FALSE(A.isEqual(B));
  }
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

  AH.randomize(-2.0, 2.0);

  B.copySlice(&A, 0);

  for (unsigned y = 0; y < 5; y++) {
    for (unsigned z = 0; z < 3; z++) {
      EXPECT_EQ(AH.at({0, y, z}), BH.at({y, z}));
    }
  }
}

TEST(Tensor, reset) {
  Tensor A(ElemKind::FloatTy, {2, 3});
  Tensor QA(ElemKind::Int8QTy, {3, 4}, 2.2, 7);
  auto H = A.getHandle();
  auto QH = QA.getHandle<int8_t>();

  H = {1.5, 17.3, -20.3, 10.0, 1.2, -2.3};
  QH = {5, 9, -2, 4, 3, -10, 21, -9, 0, -51, 73, 2};

  A.reset(ElemKind::FloatTy, {5, 2, 6});
  QA.reset(ElemKind::Int8QTy, {4, 7, 3, 8}, 1.4, -13);

  H = A.getHandle();
  QH = QA.getHandle<int8_t>();

  EXPECT_EQ(H.dims().size(), 3);
  EXPECT_EQ(QH.dims().size(), 4);
  EXPECT_TRUE(H.dims().equals({5, 2, 6}));
  EXPECT_TRUE(QH.dims().equals({4, 7, 3, 8}));
  EXPECT_EQ(H.size(), 5 * 2 * 6);
  EXPECT_EQ(QH.size(), 4 * 7 * 3 * 8);
  EXPECT_NEAR(QA.getType().getScale(), 1.4, 0.0001);
  EXPECT_EQ(QA.getType().getOffset(), -13);

  for (size_t i = 0; i < 5 * 2 * 6; i++) {
    EXPECT_EQ(H.raw(i), 0.0);
  }
  for (size_t i = 0; i < 4 * 7 * 3 * 8; i++) {
    EXPECT_EQ(QH.raw(i), 0);
  }
}

TEST(Tensor, transpose) {
  Tensor X(ElemKind::FloatTy, {5, 2});
  auto H = X.getHandle<>();
  H = {
      0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 0.8, 1.0, 2.0, 3.0,
  };

  Tensor Xhat;
  X.transpose(&Xhat, {1, 0});

  auto XhatH = Xhat.getHandle<>();

  for (size_t i = 0; i < 5; i++) {
    EXPECT_EQ(H.at({i, 0}), XhatH.at({0, i}));
    EXPECT_EQ(H.at({i, 1}), XhatH.at({1, i}));
  }
}

TEST(Tensor, transpose2) {
  Tensor X(ElemKind::FloatTy, {10, 6, 3});
  auto H = X.getHandle<>();
  H.randomize(-2.0, 2.0);

  Tensor Xhat;
  X.transpose(&Xhat, {1, 2, 0});

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

/// Verify that accessing/modifying a tensor with offsets correctly modifies the
/// underlying base Tensor's data. Transforms a 2D tensor:
/// 0.0 0.0 0.0       0.0 0.0 1.0
/// 0.0 0.0 0.0  -->  1.0 2.0 1.0
/// 0.0 0.0 0.0       1.0 1.0 0.0
TEST(Tensor, modifyOffsetIntoTensor2D) {
  // Zero out the base tensor.
  Tensor orig(ElemKind::FloatTy, {3, 3});
  orig.zero();

  // View contiguous data from the original tensor from {0, 2} to {2, 1} as a
  // single dimensional tensor of length 6.
  Tensor subview = orig.getUnowned({6}, {0, 2});
  auto H_subview = subview.getHandle<>();
  // Clear this row of 6 to 1.0.
  H_subview.clear(1.0);
  // Set the 3rd element to 2.0.
  H_subview.at({2}) = 2.0;

  // Verify the underlying data was correctly modified, according to the picture
  // above.
  auto H_orig = orig.getHandle<>();
  EXPECT_EQ(H_orig.at({0, 0}), 0.0);
  EXPECT_EQ(H_orig.at({0, 1}), 0.0);
  EXPECT_EQ(H_orig.at({0, 2}), 1.0);
  EXPECT_EQ(H_orig.at({1, 0}), 1.0);
  EXPECT_EQ(H_orig.at({1, 1}), 2.0);
  EXPECT_EQ(H_orig.at({1, 2}), 1.0);
  EXPECT_EQ(H_orig.at({2, 0}), 1.0);
  EXPECT_EQ(H_orig.at({2, 1}), 1.0);
  EXPECT_EQ(H_orig.at({2, 2}), 0.0);
}

/// Three-dimensional test of modifying a subtensor; similar in idea to the
/// two-dimensional version, modifyOffsetIntoTensor2D.
TEST(Tensor, modifyOffsetIntoTensor3D) {
  // Zero out the base tensor.
  Tensor orig(ElemKind::FloatTy, {4, 3, 2});
  orig.zero();

  // Get a 2D view of the subtensor.
  Tensor subview = orig.getUnowned({2, 6}, {1, 0, 0});
  auto H_subview = subview.getHandle<>();
  // Clear subview to 1.0.
  H_subview.clear(1.0);

  // Verify the underlying data was correctly modified.
  auto H_orig = orig.getHandle<>();
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 2; k++) {
        if (i == 1 || i == 2) {
          EXPECT_EQ(H_orig.at({i, j, k}), 1.0);
        } else {
          EXPECT_EQ(H_orig.at({i, j, k}), 0.0);
        }
      }
    }
  }
}

/// Verify that checking equality using a sub-tensor with offsets works
/// correctly.
TEST(Tensor, equalsOffsetIntoTensor) {
  // 0.0 1.0
  // 2.0 3.0
  // 4.0 5.0
  // 6.0 7.0
  Tensor orig(ElemKind::FloatTy, {4, 2});
  auto H_orig = orig.getHandle<>();
  H_orig = {0, 1, 2, 3, 4, 5, 6, 7};

  // View the data from rows 2 and 3 (each of length 2) as a single dimensional
  // tensor of size 4.
  Tensor subview = orig.getUnowned({4}, {2, 0});
  auto H_subview = subview.getHandle<>();

  // Create another tensor with same expected dimensions/data as the subview.
  Tensor recreatedSubview(ElemKind::FloatTy, {4});
  auto H_recreatedSubview = recreatedSubview.getHandle<>();
  H_recreatedSubview = {4, 5, 6, 7};

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(H_subview.at({i}), H_recreatedSubview.at({i}));
  }
}

TEST(Tensor, externallyManagedPayload) {
  // Allocate and initialize payload "externally", without using the Tensor API.
  // For example the data may come from a different library, be read from a
  // file, etc.
  std::vector<float> payload{1.2, 12.1, 51.0, 1515.2};

  {
    // Work with an existing payload buffer by means of the Tensor APIs.
    Type ty(ElemKind::FloatTy, {2, 2});
    Tensor T1(payload.data(), &ty);

    auto H1 = T1.getHandle<>();
    H1.dump();
    EXPECT_EQ(int(H1.at({0, 0})), 1);

    H1.at({1, 1}) = 30.3;
  }

  // Check that modifications through T1 and H1 changed
  // payload as well, i.e. T1/H1 were acting like a view
  // on the payload.
  EXPECT_EQ(int(payload[3]), 30);
}

/// Broadcast a Tensor of shape (2,1) to (3,2,4,2) with axis 1.
TEST(Tensor, broadcastNewShape) {
  const size_t numDims_A = 4;

  const size_t dimX_A = 3;
  const size_t dimY_A = 2;
  const size_t dimZ_A = 4;
  const size_t dimW_A = 2;

  size_t dims_A[numDims_A];
  dims_A[0] = dimX_A;
  dims_A[1] = dimY_A;
  dims_A[2] = dimZ_A;
  dims_A[3] = dimW_A;

  const size_t dimY_B = 2;
  const size_t dimZ_B = 1;
  Tensor X_B(ElemKind::FloatTy, {dimY_B, dimZ_B});
  auto H_B = X_B.getHandle<>();
  H_B = {200, 201};

  Tensor broadcastedB;
  const unsigned axis = 1;
  X_B.broadcastToNewShape(&broadcastedB, dims_A, axis);

  auto broadcastedBHandle = broadcastedB.getHandle<>();

  // Verify broadcasted B has same shape.
  EXPECT_EQ(numDims_A, broadcastedBHandle.dims().size());
  for (size_t i = 0; i < broadcastedBHandle.dims().size(); i++) {
    EXPECT_EQ(dims_A[i], broadcastedBHandle.dims()[i]);
  }

  // Look at the two values in X_B and verify in the three dimensions it was
  // broadcasted that the values were correctly broadcasted.
  const size_t k_B = 0;
  for (size_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({j_B, k_B});
    const size_t j_A = j_B; // This dim was not broadcasted (dims were equal).
    for (size_t i_A = 0; i_A < dimX_A; ++i_A) {
      for (size_t k_A = 0; k_A < dimZ_A; ++k_A) {
        for (size_t l_A = 0; l_A < dimW_A; ++l_A) {
          EXPECT_EQ(origVal, broadcastedBHandle.at({i_A, j_A, k_A, l_A}));
        }
      }
    }
  }
}

TEST(Tensor, integerTensors) {
  Tensor X;
  // Integer tensors must have scale and offset.
  Type I32Ty(ElemKind::Int32QTy, {1, 3}, 0.1, 4);
  Type I8Ty(ElemKind::Int8QTy, {3, 3}, 0.5, 2);

  Type I8Ty2(ElemKind::Int8QTy, {3, 3}, 4, 4);
  Type I8Ty3(ElemKind::Int8QTy, {3, 3}, 4, 4);

  // Float tensors must not have scale and offsets.
  Type FlTy(ElemKind::FloatTy, {1, 3});

  // Check that basic operations work.
  Tensor I(I8Ty);
  auto H = I.getHandle<int8_t>();
  H.at({0, 2}) = 3;

  EXPECT_EQ(H.at({0, 2}), 3);
  EXPECT_EQ(0.5, I.getType().getScale());
  EXPECT_EQ(2, I.getType().getOffset());

  // These types have a different scale and offset.
  EXPECT_FALSE(I8Ty.isEqual(I8Ty2));

  // These types have the same scale and offset.
  EXPECT_TRUE(I8Ty2.isEqual(I8Ty3));
}
