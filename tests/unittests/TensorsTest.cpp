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
#include "glow/Quantization/Base/Base.h"

#include "gtest/gtest.h"

using namespace glow;

TEST(Tensor, init) {
  Tensor T = {1.2f, 12.1f, 51.0f, 1515.2f};

  auto H = T.getHandle<>();

  H.dump();

  EXPECT_EQ(int(H.at({2})), 51);

  H = {1.1f, 1.2f, 1.3f, 1.4f};

  EXPECT_EQ(int(H.at({0})), 1);

  H.dump();
}

TEST(Tensor, getSliceSize) {
  // Test the Type::getSliceSize() function.

  Tensor X(ElemKind::FloatTy, {3, 2, 10, 4});
  Tensor Y(ElemKind::FloatTy, {1, 2, 3, 4});

  EXPECT_EQ(X.getType().getSliceSize(0), 3 * 2 * 10 * 4);
  EXPECT_EQ(X.getType().getSliceSize(1), 2 * 10 * 4);
  EXPECT_EQ(X.getType().getSliceSize(2), 10 * 4);
  EXPECT_EQ(X.getType().getSliceSize(3), 4);
  EXPECT_EQ(Y.getType().getSliceSize(0), 1 * 2 * 3 * 4);
  EXPECT_EQ(Y.getType().getSliceSize(3), 4);
}

TEST(Tensor, randomizeInt) {
  PseudoRNG PRNG;
  Tensor T(ElemKind::Int8QTy, {10, 10}, 1.0, 0);
  auto H = T.getHandle<int8_t>();
  H.randomize(-50, 50, PRNG);

  // Check that all of the numbers fall in the range -50 to 50.
  for (auto elem : H) {
    EXPECT_NEAR(elem, 0, 50);
  }
}

TEST(Tensor, randomizeFloat16) {
  PseudoRNG PRNG;
  Tensor T(ElemKind::Float16Ty, {10, 10});
  auto H = T.getHandle<float16_t>();
  H.randomize(-50, 50, PRNG);

  // Check that all of the numbers fall in the range -50 to 50.
  for (auto elem : H) {
    EXPECT_NEAR(elem, 0, 50);
  }
}

TEST(Tensor, clone) {
  Tensor T = {1.2f, 12.1f, 51.0f, 1515.2f};
  auto H = T.getHandle<>();

  Tensor v;
  v.assign(&T);
  auto vH = v.getHandle<>();

  EXPECT_EQ(int(vH.at({0})), 1);

  // Update the original tensor
  H = {0.11f, 0.22f, 0.33f, 0.44f};

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

TEST(Tensor, equalNAN) {
  {
    Tensor A = {0.5, 0, 0, 25};
    Tensor B = {NAN, 0, NAN, NAN};
    EXPECT_FALSE(A.isEqual(B));
  }
  {
    Tensor A = {NAN, 0, NAN, NAN};
    Tensor B = {0.5, 0, 0, 25};
    EXPECT_FALSE(A.isEqual(B));
  }
  {
    Tensor A = {NAN, 0, NAN, NAN};
    Tensor B = {NAN, 0, NAN, NAN};
    EXPECT_FALSE(A.isEqual(B));
  }
}

template <typename Ty> void testAssignment(const Type &ty) {
  // Testing some tensor operations.
  Tensor T(ty);

  auto Handle = T.getHandle<Ty>();

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

  Tensor TT = Handle.extractSlice(1);
  auto H2 = TT.getHandle<Ty>();

  EXPECT_EQ(H2.at({10, 10}), 1 + 10 + 10);

  for (unsigned y = 0; y < 20; y++) {
    for (unsigned z = 0; z < 64; z++) {
      H2.at({y, z}) = 2;
    }
  }

  EXPECT_EQ(H2.at({10, 10}), 2);
}

TEST(Tensor, assignment) {
  size_t dim[] = {320, 200, 64};
  testAssignment<float>(Type{ElemKind::FloatTy, dim});
  testAssignment<int8_t>(Type{ElemKind::Int8QTy, dim, 1., 0});
  testAssignment<int16_t>(Type{ElemKind::Int16QTy, dim, 1., 0});
  testAssignment<int32_t>(Type{ElemKind::Int32QTy, dim, 1., 0});
  testAssignment<int32_t>(Type{ElemKind::Int32ITy, dim});
  testAssignment<int64_t>(Type{ElemKind::Int64ITy, dim});
}

TEST(Tensor, concatTensors1D) {
  Tensor X = {1.1f, 2.1f, 3.1f, 4.1f};
  Tensor Y = {5.2f, 6.2f, 7.2f, 8.2f};
  Tensor Z = {0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
  Tensor expected = {5.2f, 6.2f, 7.2f, 8.2f, 1.1f, 2.1f, 3.1f, 4.1f};

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
  PseudoRNG PRNG;
  // Testing some tensor operations.
  Tensor A(ElemKind::FloatTy, {10, 5, 3});
  Tensor B(ElemKind::FloatTy, {5, 3});

  auto AH = A.getHandle<>();
  auto BH = B.getHandle<>();

  AH.randomize(-2.0, 2.0, PRNG);

  B.copySlice(&A, 0);

  for (unsigned y = 0; y < 5; y++) {
    for (unsigned z = 0; z < 3; z++) {
      EXPECT_EQ(AH.at({0, y, z}), BH.at({y, z}));
    }
  }
}

/// Check that we can copy tensors across different types.
TEST(Tensor, copyWithCast) {
  PseudoRNG PRNG;
  Tensor A(ElemKind::Float16Ty, {10, 5, 3});
  Tensor B(ElemKind::FloatTy, {10, 5, 3});

  auto AH = A.getHandle<float16_t>();
  auto BH = B.getHandle<>();

  AH.randomize(-2.0, 2.0, PRNG);

  B.copyWithCast<float, float16_t>(&A);

  EXPECT_EQ(A.size(), B.size());
  for (size_t idx = 0, end = A.size(); idx != end; ++idx) {
    EXPECT_NEAR(AH.raw(idx), BH.raw(idx), 0.0001);
  }
}

/// Check that we can convert a tensor from float to float16_t and the other way
/// around.
TEST(Tensor, convertToType) {
  PseudoRNG PRNG;
  Tensor A(ElemKind::FloatTy, {10, 5, 3});
  Tensor B(ElemKind::FloatTy, {10, 5, 3});

  auto AH = A.getHandle<>();

  AH.randomize(-2.0, 2.0, PRNG);

  B.copyRawFrom(&A);
  ASSERT_EQ(B.getElementType(), ElemKind::FloatTy);

  // Cast B from float to float16_t.
  B.convertToType(ElemKind::Float16Ty);
  ASSERT_EQ(B.getElementType(), ElemKind::Float16Ty);
  {
    auto BH = B.getHandle<float16_t>();

    EXPECT_EQ(A.size(), B.size());
    for (size_t idx = 0, end = A.size(); idx != end; ++idx) {
      EXPECT_NEAR(AH.raw(idx), BH.raw(idx), 0.001);
    }
  }

  // Cast back B from float16_t to float.
  B.convertToType(ElemKind::FloatTy);
  ASSERT_EQ(B.getElementType(), ElemKind::FloatTy);
  EXPECT_TRUE(B.isEqual(A, 0.001));
}

TEST(Tensor, reset) {
  Tensor A(ElemKind::FloatTy, {2, 3});
  Tensor QA(ElemKind::Int8QTy, {3, 4}, 2.2, 7);
  auto H = A.getHandle();
  auto QH = QA.getHandle<int8_t>();

  H = {1.5f, 17.3f, -20.3f, 10.0f, 1.2f, -2.3f};
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
    EXPECT_EQ(QH.raw(i), QA.getType().getOffset());
  }
}

TEST(Tensor, transpose) {
  Tensor X(ElemKind::FloatTy, {5, 2});
  auto H = X.getHandle<>();
  H = {
      0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 0.6f, 0.8f, 1.0f, 2.0f, 3.0f,
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
  PseudoRNG PRNG;
  Tensor X(ElemKind::FloatTy, {10, 6, 3});
  auto H = X.getHandle<>();
  H.randomize(-2.0, 2.0, PRNG);

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
  Tensor T1 = {1.2f, 12.1f, 51.0f, 1515.2f};

  auto H1 = T1.getHandle<>();
  H1.dump();
  EXPECT_EQ(int(H1.at({0})), 1);

  {
    // Create a view on T1 which makes it look like 2x2
    Tensor T2 = T1.getUnowned({2, 2});
    EXPECT_EQ(T2.getUnsafePtr(), T1.getUnsafePtr());
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

/// Check that we properly take ownership of
/// the underlying memory when we reset the tensor
/// shape. This test used to fail leak sanitizer.
TEST(Tensor, nonOwnedTensorFollowedByReset) {
  float raw_data = 0.;
  Type F32Ty(ElemKind::FloatTy, {1});

  // Create an unowned tensor.
  Tensor T1(&raw_data, &F32Ty);

  auto H1 = T1.getHandle<>();
  EXPECT_EQ(int(H1.at({0})), 0);

  Type F32x2Ty(ElemKind::FloatTy, {2});

  // Resizing the tensor will trigger some memory allocation.
  // Given the previous data was coming from outside, this
  // tensor was unowned and we used to not reset that state
  // as well and were leaking memory.
  T1.reset(F32x2Ty);
  H1 = T1.getHandle<>();
  EXPECT_EQ(int(H1.at({0})), 0);
  EXPECT_EQ(int(H1.at({1})), 0);

  // When T1 gets delete the memory allocated through reset should
  // be released.
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
  std::vector<float> payload{1.2f, 12.1f, 51.0f, 1515.2f};

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

TEST(Tensor, insertWithCountAndAxis) {
  Tensor X(ElemKind::FloatTy, {3, 2});
  Tensor Y(ElemKind::FloatTy, {3, 6});

  auto xH = X.getHandle<>();
  auto yH = Y.getHandle<>();

  for (size_t i = 0, e = xH.size(); i < e; i++) {
    xH.raw(i) = float(i);
  }

  // Insert three of these slices on axis 1
  yH.insertTensors(xH, {0, 0}, /* count */ 3, /* axis */ 1);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 6; j++) {
      EXPECT_EQ(xH.at({i, j % 2}), yH.at({i, j}));
    }
  }
}

/// Verify that tensors that are quantized begin zero'd to their type's offset
/// and are reset back to that offset.
TEST(Tensor, zeroQuantizedTensor) {
  const int32_t offsetQ8 = 0;
  Tensor Q8T(ElemKind::Int8QTy, {3, 4, 5, 6}, 127, offsetQ8);

  const int32_t offsetQ16 = 223;
  Tensor Q16T(ElemKind::Int16QTy, {3, 4, 5}, 1234.7, offsetQ16);

  const int32_t offsetQ32 = 53452;
  Tensor Q32T(ElemKind::Int32QTy, {3, 4}, 500.4, offsetQ32);

  auto Q8H = Q8T.getHandle<int8_t>();
  EXPECT_TRUE(Q8H.isZero());
  for (auto elem : Q8H) {
    EXPECT_EQ(elem, offsetQ8);
  }

  auto Q16H = Q16T.getHandle<int16_t>();
  EXPECT_TRUE(Q16H.isZero());
  for (auto elem : Q16H) {
    EXPECT_EQ(elem, offsetQ16);
  }

  auto Q32H = Q32T.getHandle<int32_t>();
  EXPECT_TRUE(Q32H.isZero());
  for (auto elem : Q32H) {
    EXPECT_EQ(elem, offsetQ32);
  }

  Q32H = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_FALSE(Q32H.isZero());

  for (auto elem : Q32H) {
    EXPECT_NE(elem, offsetQ32);
  }

  Q32T.zero();
  EXPECT_TRUE(Q32H.isZero());
  for (auto elem : Q32H) {
    EXPECT_EQ(elem, offsetQ32);
  }
}

// Verify that if the tensor is set to the offset manually then isZero() is
// true
TEST(Tensor, manuallySetToOffset) {
  const int8_t offsetQ8 = 6;
  Tensor Q8T(ElemKind::Int8QTy, {3, 2}, 10.1, offsetQ8);

  auto Q8H = Q8T.getHandle<int8_t>();
  EXPECT_TRUE(Q8H.isZero());

  Q8H = {1, 2, 3, 4, 5, 6};
  EXPECT_FALSE(Q8H.isZero());

  Q8H = {offsetQ8, offsetQ8, offsetQ8, offsetQ8, offsetQ8, offsetQ8};
  EXPECT_TRUE(Q8H.isZero());

  Q8H.raw(1) = offsetQ8 - 2;
  EXPECT_FALSE(Q8H.isZero());

  Q8H.raw(1) = offsetQ8;
  EXPECT_TRUE(Q8H.isZero());
}

TEST(ZeroDimensionalTensor, handleAt) {
  Tensor T(ElemKind::FloatTy, {});
  auto H = T.getHandle<>();
  H.at({}) = 7.1;
  EXPECT_FLOAT_EQ(H.at({}), 7.1);
  EXPECT_FLOAT_EQ(((float *)T.getUnsafePtr())[0], 7.1);
}

TEST(ZeroDimensionalTensor, handleAssign) {
  Tensor T(ElemKind::FloatTy, {});
  auto H = T.getHandle<>();
  H = {1.14f};
  EXPECT_FLOAT_EQ(H.at({}), 1.14);
  EXPECT_FLOAT_EQ(((float *)T.getUnsafePtr())[0], 1.14);
}

TEST(ZeroDimensionalTensor, compareAndDumpTwo) {
  Tensor T1(ElemKind::FloatTy, {});
  Tensor T2(ElemKind::FloatTy, {});

  EXPECT_TRUE(T1.isEqual(T2));

  auto H = T1.getHandle<>();
  H.dump();

  EXPECT_FLOAT_EQ(H.raw(0), 0.0);
  H.raw(0) = 4.2;
  EXPECT_FLOAT_EQ(H.raw(0), 4.2);

  EXPECT_FALSE(T1.isEqual(T2));
  H.dump();
}

TEST(ZeroDimensionalTensor, compareToNonZeroDimensional) {
  Tensor T1(ElemKind::FloatTy, {});
  Tensor T2(ElemKind::FloatTy, {1});
  T1.zero();
  T2.zero();

  EXPECT_FALSE(T1.isEqual(T2));
}

TEST(ZeroDimensionalTensor, transpose) {
  Tensor T(ElemKind::Int64ITy, {});
  T.getHandle<int64_t>() = {15};

  Tensor TT;
  T.transpose(&TT, {});

  EXPECT_TRUE(T.isEqual(TT));
}

TEST(Type, compare) {
  Type T1(ElemKind::FloatTy, {});
  Type T2(ElemKind::FloatTy, {});
  Type T3(ElemKind::FloatTy, {1});
  Type T4(ElemKind::Int64ITy, {});

  EXPECT_TRUE(T1.isEqual(T2));
  EXPECT_FALSE(T1.isEqual(T3));
  EXPECT_FALSE(T1.isEqual(T4));
}

TEST(Tensor, insertSlice) {
  Tensor big(ElemKind::FloatTy, {3, 4});
  Tensor small({1.0f, 2.0f, 3.0f, 4.0f});
  big.zero();
  big.getHandle<>().insertSlice(small, 1);
  Tensor expected(ElemKind::FloatTy, {3, 4});
  expected.getHandle<>() = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f,
                            3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  EXPECT_TRUE(big.isEqual(expected));
}

/// Check that after initializing a fused tensor to zero that the scale and
/// offset are not changed and that the values for each row are set to that
/// row's offset.
TEST(Tensor, initZeroFused) {
  Tensor T(ElemKind::UInt8FusedQTy, {10, 10}, 0.0, 0);
  auto TH = T.getHandle<uint8_t>();
  auto *TData = reinterpret_cast<uint8_t *>(T.getUnsafePtr());
  TH.clear(127);

  // Now set the scale/offset of each row. Set the scale to 0.1 so that we are
  // multiplying by 10 when calculating zero. Offset is dependent on each row.
  float scaleForAllRows = 0.1;
  for (size_t i = 0; i < 10; i++) {
    const float offset = -(i + 0.7);
    uint8_t *scaleOffsetPtr = &TData[(i + 1) * 10] - 2 * sizeof(float);
    memcpy(scaleOffsetPtr, &scaleForAllRows, sizeof(float));
    memcpy(scaleOffsetPtr + sizeof(float), &offset, sizeof(float));
  }

  // Now reset so that all row's actual data is set to zero based on the
  // scale/offset in the row.
  PseudoRNG PRNG;
  T.init(Tensor::InitKind::Zero, 1, PRNG);

  EXPECT_TRUE(TH.isZero(0.00001f));

  // Now check that we correctly set the data, and that the scale/offsets are
  // the same as expected (untouched by initializing to zero).
  for (size_t i = 0; i < 10; i++) {
    uint8_t *scaleOffsetPtr = &TData[(i + 1) * 10] - 2 * sizeof(float);
    float scale, offset;
    memcpy(&scale, scaleOffsetPtr, sizeof(float));
    memcpy(&offset, scaleOffsetPtr + sizeof(float), sizeof(float));

    EXPECT_NEAR(quantization::dequantizeWithFloatOffset<uint8_t>(TH.at({i, 0}),
                                                                 scale, offset),
                0, 1E-5);
    EXPECT_NEAR(quantization::dequantizeWithFloatOffset<uint8_t>(TH.at({i, 1}),
                                                                 scale, offset),
                0, 1E-5);
  }
}

/// Check that initializing a fused tensor with Xavier that the scale and offset
/// are not changed.
TEST(Tensor, initXavierFused) {
  Tensor T(ElemKind::UInt8FusedQTy, {10, 10}, 0.0, 0);
  PseudoRNG PRNG;
  auto TH = T.getHandle<uint8_t>();
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++) {
      TH.at({i, j}) = i * 10 + j;
    }
  }
  T.init(Tensor::InitKind::Xavier, 1, PRNG);
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 2; j < 10; j++) {
      // Check that the scales/offsets are unchanged.
      EXPECT_EQ(TH.at({i, j}), i * 10 + j);
    }
  }
}

/// Check that initializing a fused tensor with Broadcast that the scale and
/// offset are not changed, and broadcast value is set correctly.
TEST(Tensor, initBroadcastFused) {
  Tensor T(ElemKind::UInt8FusedQTy, {10, 10}, 0.0, 0);
  auto TH = T.getHandle<uint8_t>();
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++) {
      TH.at({i, j}) = i * 10 + j;
    }
  }
  PseudoRNG PRNG;
  T.init(Tensor::InitKind::Broadcast, 5, PRNG);
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++) {
      // Check that the scales/offsets are unchanged, and that the broadcast
      // value is everywhere else.
      if (j < 2) {
        EXPECT_EQ(TH.at({i, j}), 5);
      } else {
        EXPECT_EQ(TH.at({i, j}), i * 10 + j);
      }
    }
  }
}
