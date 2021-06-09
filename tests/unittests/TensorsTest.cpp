/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#include "glow/Base/TensorSerialization.h"
#include "glow/Graph/Graph.h"
#include "glow/Quantization/Base/Base.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace glow;

TEST(Tensor, iteration) {
  auto content = {1.2f, 12.1f, 51.0f, 1515.2f};
  Tensor T = content;

  auto H = T.getHandle<float>();

  std::vector<float> elems;
  for (auto e : H) {
    elems.push_back(e);
  }

  EXPECT_TRUE(elems == std::vector<float>(content));
}

TEST(Tensor, init) {
  Tensor T = {1.2f, 12.1f, 51.0f, 1515.2f};

  auto H = T.getHandle<>();

  H.dump();

  EXPECT_EQ(int(H.at({2})), 51);

  H = {1.1f, 1.2f, 1.3f, 1.4f};

  EXPECT_EQ(int(H.at({0})), 1);

  H.dump();
}

/// Test that Tensors with zero-dimensions work as expected.
TEST(Tensor, zeroDimTensors) {
  Tensor T0(ElemKind::FloatTy, {0});
  Tensor T1(ElemKind::FloatTy, {0, 100});
  Tensor T2(ElemKind::FloatTy, {100, 0});

  EXPECT_EQ(T0.getUnpaddedSizeInBytes(), 0);
  EXPECT_EQ(T1.getUnpaddedSizeInBytes(), 0);
  EXPECT_EQ(T2.getUnpaddedSizeInBytes(), 0);
  EXPECT_EQ(T0.getSizeInBytes(), 0);
  EXPECT_EQ(T1.getSizeInBytes(), 0);
  EXPECT_EQ(T2.getSizeInBytes(), 0);
  EXPECT_EQ(T0.size(), 0);
  EXPECT_EQ(T1.size(), 0);
  EXPECT_EQ(T2.size(), 0);

  // Nothing is allocated for these tensors.
  EXPECT_EQ(T0.getUnsafePtr(), nullptr);
  EXPECT_EQ(T1.getUnsafePtr(), nullptr);
  EXPECT_EQ(T2.getUnsafePtr(), nullptr);

  T0.getHandle<>().dump();
  T1.getHandle<>().dump();
  T2.getHandle<>().dump();

  // Now test getting unowned views of partial tensors that are zero sized.
  Tensor T4(ElemKind::FloatTy, {10, 0, 10});
  Type ty(ElemKind::FloatTy, {10, 5, 10});
  Tensor T5(T4.getUnsafePtr(), &ty, T4.getSizeInBytes());
  EXPECT_EQ(T4.getUnsafePtr(), T5.getUnsafePtr());
  EXPECT_EQ(T5.getUnpaddedSizeInBytes(), 0);
  EXPECT_EQ(T5.getSizeInBytes(), ty.getSizeInBytes());
  EXPECT_EQ(T5.size(), ty.size());
  T5.getHandle<>().dump();
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

TEST(Tensor, randomizeBFloat16) {
  PseudoRNG PRNG;
  Tensor T(ElemKind::BFloat16Ty, {10, 10});
  auto H = T.getHandle<bfloat16_t>();
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

TEST(Tensor, isTiled) {
  // Single axis testing.
  {
    Tensor T(ElemKind::FloatTy, {2, 3});
    T.getHandle() = {
        1, 2, 3, 1, 2, 3,
    };
    EXPECT_TRUE(T.isTiled(0, 1));
    EXPECT_TRUE(T.isTiled(0, 2));
    EXPECT_FALSE(T.isTiled(1, 1));
    EXPECT_FALSE(T.isTiled(1, 2));
    EXPECT_TRUE(T.isTiled(1, 3));
  }
  {
    Tensor T(ElemKind::FloatTy, {2, 4});
    T.getHandle() = {1, 2, 1, 2, 3, 4, 3, 4};
    EXPECT_FALSE(T.isTiled(0, 1));
    EXPECT_TRUE(T.isTiled(0, 2));
    EXPECT_FALSE(T.isTiled(1, 1));
    EXPECT_TRUE(T.isTiled(1, 2));
    EXPECT_FALSE(T.isTiled(1, 3));
    EXPECT_TRUE(T.isTiled(1, 4));
  }
  {
    Tensor T(ElemKind::FloatTy, {2, 4});
    T.getHandle() = {1, 2, 1, 2, 1, 2, 1, 2};
    EXPECT_TRUE(T.isTiled(0, 1));
    EXPECT_TRUE(T.isTiled(0, 2));
    EXPECT_FALSE(T.isTiled(1, 1));
    EXPECT_TRUE(T.isTiled(1, 2));
    EXPECT_FALSE(T.isTiled(1, 3));
    EXPECT_TRUE(T.isTiled(1, 4));
  }
  {
    Tensor T(ElemKind::FloatTy, {2, 4});
    T.getHandle() = {1, 2, 1, 2, 3, 4, 3, 44};
    EXPECT_FALSE(T.isTiled(0, 1));
    EXPECT_TRUE(T.isTiled(0, 2));
    EXPECT_FALSE(T.isTiled(1, 1));
    EXPECT_FALSE(T.isTiled(1, 2));
    EXPECT_FALSE(T.isTiled(1, 3));
    EXPECT_TRUE(T.isTiled(1, 4));
  }
  {
    Tensor T(ElemKind::FloatTy, {5});
    T.getHandle() = {1, 2, 3, 1, 2};
    EXPECT_FALSE(T.isTiled(0, 3));
    EXPECT_TRUE(T.isTiled(0, 3, /* fractional */ true));
  }
  // Multiple axis testing.
  {
    Tensor T(ElemKind::FloatTy, {2, 3});
    T.getHandle() = {
        1, 2, 1, 1, 2, 1,
    };
    EXPECT_FALSE(T.isTiled({0, 1}, {1, 1}));
    EXPECT_FALSE(T.isTiled({0, 1}, {1, 2}));
    EXPECT_TRUE(T.isTiled({0, 1}, {1, 2}, /* fractional */ true));
    EXPECT_TRUE(T.isTiled({0, 1}, {1, 3}));
    EXPECT_FALSE(T.isTiled({0, 1}, {2, 1}));
    EXPECT_FALSE(T.isTiled({0, 1}, {2, 2}));
    EXPECT_TRUE(T.isTiled({0, 1}, {2, 2}, /* fractional */ true));
    EXPECT_TRUE(T.isTiled({0, 1}, {2, 3}));
  }
  {
    Tensor T(ElemKind::FloatTy, {2, 4});
    T.getHandle() = {
        1, 2, 1, 2, 1, 2, 1, 2,
    };
    EXPECT_FALSE(T.isTiled({0, 1}, {1, 1}));
    EXPECT_TRUE(T.isTiled({0, 1}, {1, 2}));
    EXPECT_FALSE(T.isTiled({0, 1}, {1, 3}));
    EXPECT_TRUE(T.isTiled({0, 1}, {1, 4}));
    EXPECT_FALSE(T.isTiled({0, 1}, {2, 1}));
    EXPECT_TRUE(T.isTiled({0, 1}, {2, 2}));
    EXPECT_FALSE(T.isTiled({0, 1}, {2, 3}));
    EXPECT_TRUE(T.isTiled({0, 1}, {2, 4}));
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
  dim_t dim[] = {320, 200, 64};
  testAssignment<float>(Type{ElemKind::FloatTy, dim});
  testAssignment<double>(Type{ElemKind::Float64Ty, dim});
  testAssignment<int8_t>(Type{ElemKind::Int8QTy, dim, 1., 0});
  testAssignment<uint8_t>(Type{ElemKind::UInt8QTy, dim, 1., 0});
  testAssignment<int16_t>(Type{ElemKind::Int16QTy, dim, 1., 0});
  testAssignment<int32_t>(Type{ElemKind::Int32QTy, dim, 1., 0});
  testAssignment<uint8_t>(Type{ElemKind::UInt8ITy, dim});
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

  for (dim_t i = 0, e = eH.size(); i < e; i++) {
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

  // Zero Y and Z but not X.
  Y.zero();
  Z.zero();

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
        dim_t ptr = H.getElementPtr({x, y, z});
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

/// Check that we can copy tensors across different types.
TEST(Tensor, copyWithCastBFloat16) {
  PseudoRNG PRNG;
  Tensor A(ElemKind::BFloat16Ty, {10, 5, 3});
  Tensor B(ElemKind::FloatTy, {10, 5, 3});

  auto AH = A.getHandle<bfloat16_t>();
  auto BH = B.getHandle<>();

  AH.randomize(-2.0, 2.0, PRNG);

  B.copyWithCast<float, bfloat16_t>(&A);

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
  A.zero();
  QA.reset(ElemKind::Int8QTy, {4, 7, 3, 8}, 1.4, -13);
  QA.zero();

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

  for (dim_t i = 0; i < 5; i++) {
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

  for (dim_t i = 0; i < 10; i++) {
    for (dim_t j = 0; j < 6; j++) {
      for (dim_t k = 0; k < 3; k++) {
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
  T1.zero();
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
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 3; j++) {
      for (dim_t k = 0; k < 2; k++) {
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

  for (dim_t i = 0; i < 4; i++) {
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

  for (dim_t i = 0; i < 3; i++) {
    for (dim_t j = 0; j < 6; j++) {
      EXPECT_EQ(xH.at({i, j % 2}), yH.at({i, j}));
    }
  }
}

/// Verify that tensors that are quantized begin zero'd to their type's offset
/// and are reset back to that offset.
TEST(Tensor, zeroQuantizedTensor) {
  const int32_t offsetQ8 = 0;
  Tensor Q8T(ElemKind::Int8QTy, {3, 4, 5, 6}, 127, offsetQ8);
  Q8T.zero();

  const int32_t offsetUQ8 = 3;
  Tensor UQ8T(ElemKind::UInt8QTy, {3, 4, 5, 6}, 2, offsetUQ8);
  UQ8T.zero();

  const int32_t offsetQ16 = 223;
  Tensor Q16T(ElemKind::Int16QTy, {3, 4, 5}, 1234.7, offsetQ16);
  Q16T.zero();

  const int32_t offsetQ32 = 53452;
  Tensor Q32T(ElemKind::Int32QTy, {3, 4}, 500.4, offsetQ32);
  Q32T.zero();

  auto Q8H = Q8T.getHandle<int8_t>();
  EXPECT_TRUE(Q8H.isZero());
  for (auto elem : Q8H) {
    EXPECT_EQ(elem, offsetQ8);
  }

  auto UQ8H = UQ8T.getHandle<uint8_t>();
  EXPECT_TRUE(UQ8H.isZero());
  for (auto elem : UQ8H) {
    EXPECT_EQ(elem, offsetUQ8);
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
  Q8T.zero();

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
  T1.zero();
  Tensor T2(ElemKind::FloatTy, {});
  T2.zero();

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

TEST(ZeroDimensionalTensor, iterate) {
  Tensor T(ElemKind::Int64ITy, {});
  T.getHandle<int64_t>() = {15};

  auto TH = T.getHandle<int64_t>();
  std::vector<int64_t> elems;
  for (auto e : TH) {
    elems.push_back(e);
  }

  EXPECT_EQ(elems.size(), 1);
  EXPECT_EQ(elems[0], 15);
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

TEST(Type, isEqual) {
  {
    Type T1(ElemKind::FloatTy, {1, 2, 3});
    Type T2(ElemKind::Int64ITy, {1, 2, 3});
    EXPECT_FALSE(T1.isEqual(T2));
    EXPECT_FALSE(T2.isEqual(T1));
  }
  {
    Type T1(ElemKind::FloatTy, {1, 2, 3});
    Type T2(ElemKind::FloatTy, {1, 2});
    EXPECT_FALSE(T1.isEqual(T2));
    EXPECT_FALSE(T2.isEqual(T1));
  }
  {
    Type T1(ElemKind::FloatTy, {1, 2, 3});
    Type T2(ElemKind::FloatTy, {1, 2, 4});
    EXPECT_FALSE(T1.isEqual(T2));
    EXPECT_FALSE(T2.isEqual(T1));
  }
  {
    Type T1(ElemKind::FloatTy, {1, 2, 3});
    Type T2(ElemKind::FloatTy, {1, 2, 4});
    EXPECT_TRUE(T1.isEqual(T2, /* allowDifferentShape */ true));
    EXPECT_TRUE(T2.isEqual(T1, /* allowDifferentShape */ true));
  }
  {
    Type T1(ElemKind::FloatTy, {1, 2, 3});
    Type T2(ElemKind::FloatTy, {4, 2, 3});
    EXPECT_TRUE(T1.isEqual(T2, /* allowDifferentShape */ true));
    EXPECT_TRUE(T2.isEqual(T1, /* allowDifferentShape */ true));
  }
  {
    Type T1(ElemKind::Int8QTy, {1, 2, 3}, 0, 0);
    Type T2(ElemKind::Int8QTy, {1, 2, 3}, 1, 0);
    EXPECT_FALSE(T1.isEqual(T2));
    EXPECT_FALSE(T2.isEqual(T1));
  }
  {
    Type T1(ElemKind::Int8QTy, {1, 2, 3}, 1, 4);
    Type T2(ElemKind::Int8QTy, {1, 2, 3}, 1, 4);
    EXPECT_TRUE(T1.isEqual(T2));
    EXPECT_TRUE(T2.isEqual(T1));
  }
  {
    Type T1(ElemKind::FloatTy, {1, 2, 3});
    Type T2(ElemKind::FloatTy, {1, 2, 3});
    EXPECT_TRUE(T1.isEqual(T2));
    EXPECT_TRUE(T2.isEqual(T1));
  }
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

/// Check that after converting to UInt8FusedQTy, the data, scale and offset are
/// the same as original ones.
template <class Ty>
static void testConvertToUInt8FusedQTy(ElemKind fusedKind, dim_t row,
                                       dim_t col) {
  EXPECT_LT(row, 100);
  EXPECT_LT(col, 100);
  Tensor T(fusedKind, {row, col}, 1.0, 0);
  auto dataCol = col - 2 * sizeof(Ty);
  auto TH = T.getHandle<uint8_t>();
  for (dim_t i = 0; i < row; i++) {
    TH.setFusedScaleOffsetInRow<Ty>(i, i, i);
    for (dim_t j = 0; j < dataCol; j++) {
      TH.at({i, j}) = i + j;
    }
  }

  Tensor newT = T.getCopyConvertedToType(ElemKind::UInt8FusedQTy);
  auto newTH = newT.getHandle<uint8_t>();
  bool is4Bit = fusedKind == ElemKind::UInt4FusedFP16QTy ||
                fusedKind == ElemKind::UInt4FusedQTy;

  // Check the converted dims.
  auto expectedCol = dataCol * (is4Bit ? 2 : 1) + 2 * sizeof(float);
  EXPECT_EQ(newTH.dims().size(), 2);
  EXPECT_EQ(newTH.dims()[0], TH.dims()[0]);
  EXPECT_EQ(newTH.dims()[1], expectedCol);

  // Check the converted FP32 scale/offset are correctly cast from Fp16
  // scale/offset.
  for (dim_t i = 0; i < row; i++) {
    float scale, offset;
    std::tie(scale, offset) = newTH.getFusedScaleOffsetFromRow<float>(i);
    EXPECT_EQ(scale, (float)i);
    EXPECT_EQ(offset, (float)i);
  }

  // Check the converted data are the same as original ones.
  for (dim_t i = 0; i < row; i++) {
    for (dim_t j = 0; j < dataCol; j++) {
      if (is4Bit) {
        EXPECT_EQ(newTH.at({i, j * 2}), (i + j) & 0x0F);
        EXPECT_EQ(newTH.at({i, j * 2 + 1}), ((i + j) >> 4) & 0x0F);
      } else {
        EXPECT_EQ(newTH.at({i, j}), i + j);
      }
    }
  }
}

/// Check that after initializing a fused tensor to zero that the scale and
/// offset are not changed and that the values for each row are set to that
/// row's offset.
template <typename ScaleOffsetT>
static void testInitZeroFused(ElemKind fusedKind, float allowedError) {
  constexpr dim_t numTotalColumns = 2 + 2 * sizeof(ScaleOffsetT);
  Tensor T(fusedKind, {10, numTotalColumns}, 0.0, 0);
  auto TH = T.getHandle<uint8_t>();
  auto *TData = reinterpret_cast<uint8_t *>(T.getUnsafePtr());
  TH.clear(127);
  auto rowLength = TH.getElementPtr({1, 0});
  auto width = TH.dims()[1];

  // Now set the scale/offset of each row. Set the scale to 0.1 so that we are
  // multiplying by 10 when calculating zero. Offset is dependent on each row.
  const ScaleOffsetT scaleForAllRows = 0.1;
  for (size_t i = 0; i < 10; i++) {
    const ScaleOffsetT offset = -(i + 0.7);
    uint8_t *scaleOffsetPtr =
        &TData[i * rowLength] + width - 2 * sizeof(ScaleOffsetT);
    memcpy(scaleOffsetPtr, &scaleForAllRows, sizeof(ScaleOffsetT));
    memcpy(scaleOffsetPtr + sizeof(ScaleOffsetT), &offset,
           sizeof(ScaleOffsetT));
  }

  // Now reset so that all row's actual data is set to zero based on the
  // scale/offset in the row.
  PseudoRNG PRNG;
  T.init(Tensor::InitKind::Zero, 1, PRNG);

  EXPECT_TRUE(TH.isZero(allowedError));

  // Now check that we correctly set the data, and that the scale/offsets are
  // the same as expected (untouched by initializing to zero).
  for (dim_t i = 0; i < 10; i++) {
    uint8_t *scaleOffsetPtr =
        &TData[i * rowLength] + width - 2 * sizeof(ScaleOffsetT);
    ScaleOffsetT scale, offset;
    memcpy(&scale, scaleOffsetPtr, sizeof(ScaleOffsetT));
    memcpy(&offset, scaleOffsetPtr + sizeof(ScaleOffsetT),
           sizeof(ScaleOffsetT));

    EXPECT_NEAR(quantization::dequantizeWithFloatOffset<uint8_t>(
                    TH.at({i, 0}), static_cast<float>(scale),
                    static_cast<float>(offset)),
                0, allowedError);
    EXPECT_NEAR(quantization::dequantizeWithFloatOffset<uint8_t>(
                    TH.at({i, 1}), static_cast<float>(scale),
                    static_cast<float>(offset)),
                0, allowedError);
  }
}

/// Test zeroing a Fused tensor with Float scale/offsets.
TEST(Tensor, initZeroFused_Float) {
  testInitZeroFused<float>(ElemKind::UInt8FusedQTy, 1E-5);
}

/// Test zeroing a Fused tensor with Float16 scale/offsets.
TEST(Tensor, initZeroFused_Float16) {
  testInitZeroFused<float16_t>(ElemKind::UInt8FusedFP16QTy, 1E-2);
}

/// Check that initializing a fused tensor with Broadcast that the scale and
/// offset are not changed, and broadcast value is set correctly.
static void testBroadcastFused(ElemKind fusedKind) {
  const dim_t numTotalColumns =
      2 + 2 * ((fusedKind == ElemKind::UInt8FusedQTy) ? sizeof(float)
                                                      : sizeof(float16_t));
  Tensor T(fusedKind, {10, numTotalColumns}, 0.0, 0);
  auto TH = T.getHandle<uint8_t>();
  for (dim_t i = 0; i < 10; i++) {
    for (dim_t j = 0; j < numTotalColumns; j++) {
      TH.at({i, j}) = i * 10 + j;
    }
  }
  PseudoRNG PRNG;
  T.init(Tensor::InitKind::Broadcast, 5, PRNG);
  for (dim_t i = 0; i < 10; i++) {
    for (dim_t j = 0; j < numTotalColumns; j++) {
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

/// Test broadcasting a Fused tensor with Float scale/offsets.
TEST(Tensor, initBroadcastFused_Float) {
  testBroadcastFused(ElemKind::UInt8FusedQTy);
}

/// Test broadcasting a Fused tensor with Float16 scale/offsets.
TEST(Tensor, initBroadcastFused_Float16) {
  testBroadcastFused(ElemKind::UInt8FusedFP16QTy);
}

/// Check that when randomizing a fused quantized tensor, the scale and offset
/// are not changed.
static void testRandomizeFused(ElemKind fusedKind) {
  const dim_t numTotalColumns =
      2 + 2 * ((fusedKind == ElemKind::UInt8FusedQTy) ? sizeof(float)
                                                      : sizeof(float16_t));
  Tensor T(fusedKind, {10, numTotalColumns}, 1.0, 0);
  auto TH = T.getHandle<uint8_t>();
  for (dim_t i = 0; i < 10; i++) {
    for (dim_t j = 0; j < numTotalColumns; j++) {
      TH.at({i, j}) = i * 10 + j;
    }
  }
  PseudoRNG PRNG;
  TH.randomize(0, 255, PRNG);
  for (dim_t i = 0; i < 10; i++) {
    for (dim_t j = 2; j < numTotalColumns; j++) {
      // Check that the scales/offsets are unchanged.
      EXPECT_EQ(TH.at({i, j}), i * 10 + j);
    }
  }
}

/// Test randomizing a Fused tensor with Float scale/offsets.
TEST(Tensor, randomizeFused_Float) {
  testRandomizeFused(ElemKind::UInt8FusedQTy);
}

/// Test randomizing a Fused tensor with Float16 scale/offsets.
TEST(Tensor, randomizeFused_Float16) {
  testRandomizeFused(ElemKind::UInt8FusedFP16QTy);
}

/// Check that getting and setting fused tensors works correctly.
template <typename ScaleOffsetT>
static void testGetSetFusedScaleOffset(ElemKind fusedKind) {
  Tensor T(fusedKind, {10, 10}, 1.0, 0);
  auto TH = T.getHandle<uint8_t>();
  for (size_t i = 0; i < 10; i++) {
    TH.setFusedScaleOffsetInRow<ScaleOffsetT>(i, i, i);
  }
  for (size_t i = 0; i < 10; i++) {
    ScaleOffsetT scale, offset;
    std::tie(scale, offset) = TH.getFusedScaleOffsetFromRow<ScaleOffsetT>(i);
    EXPECT_EQ(scale, (ScaleOffsetT)i);
    EXPECT_EQ(offset, (ScaleOffsetT)i);
  }
}

/// Test getting and setting fused scales and offsets from UInt8FusedQTy.
TEST(Tensor, GetFusedScaleOffset_UInt8FusedQTy) {
  testGetSetFusedScaleOffset<float>(ElemKind::UInt8FusedQTy);
}

/// Test getting and setting fused scales and offsets from UInt8FusedFP16QTy.
TEST(Tensor, GetFusedScaleOffset_UInt8FusedFP16QTy) {
  testGetSetFusedScaleOffset<float16_t>(ElemKind::UInt8FusedFP16QTy);
}

/// Test getting and setting fused scales and offsets from UInt4FusedFP16QTy.
TEST(Tensor, GetFusedScaleOffset_UInt4FusedFP16QTy) {
  testGetSetFusedScaleOffset<float16_t>(ElemKind::UInt4FusedFP16QTy);
}

/// Test getting and setting fused scales and offsets from UInt4FusedQTy.
TEST(Tensor, GetFusedScaleOffset_UInt4FusedQTy) {
  testGetSetFusedScaleOffset<float>(ElemKind::UInt4FusedQTy);
}

/// Check if dump functions work for Tensor
TEST(Tensor, dump) {
  Tensor T = {1.2f, 12.1f, 51.0f, 1515.2f};
  std::string mes = T.toString();
  std::string storageT1;
  llvm::raw_string_ostream osT1(storageT1);
  T.dump(osT1);
  std::string expectMes = R"(shape: ( 4 )
elemkind: float
max: 1515.19995  min: 1.20000  avg: 394.87499
[1.20000, 12.10000, 51.00000, 1515.19995, ]
)";
  EXPECT_EQ(mes, expectMes);
  EXPECT_EQ(mes, osT1.str());
  std::string storageT2;
  llvm::raw_string_ostream osT2(storageT2);
  osT2 << T;
  EXPECT_EQ(mes, osT2.str());
  T.dump(2);
  std::string expectMes2 = R"(shape: ( 4 )
elemkind: float
max: 1515.19995  min: 1.20000  avg: 394.87499
[1.20000, 12.10000, ...]
)";
  std::string storageT3;
  llvm::raw_string_ostream osT3(storageT3);
  // Only dump 2 elements.
  T.dump(osT3, 2);
  std::string mes2 = T.toString(2);
  EXPECT_EQ(mes2, expectMes2);
  EXPECT_EQ(mes2, osT3.str());

  // Get an unowned padded (partial) tensor sharing storage with T.
  auto paddedType = Type::newShape(T.getType(), {256});
  Tensor partialT(T.getUnsafePtr(), &paddedType, T.getSizeInBytes());
  std::string expectPartial = R"(shape: ( 256 ) ; partial num elements: 4
elemkind: float
max: 1515.19995  min: 1.20000  avg: 394.87499
[1.20000, 12.10000, 51.00000, ...]
)";
  std::string partialString = partialT.toString(3);
  EXPECT_EQ(partialString, expectPartial);
}

/// Check Type serialization functions.
TEST(Tensor, typeSerialization) {
  auto testType = [](Type ty) {
    EXPECT_EQ(ty, Type::fromString(ty.toString()));
  };
  testType(Type(ElemKind::FloatTy, {1}));
  testType(Type(ElemKind::Float16Ty, {1, 2}));
  testType(Type(ElemKind::Float64Ty, {1}));
  testType(Type(ElemKind::Int8QTy, {1, 2, 3}, 1.1, 1));
  testType(Type(ElemKind::UInt8QTy, {1, 2, 3}, 1.2, 2));
  testType(Type(ElemKind::Int16QTy, {1, 2, 3}, 1.3, 3));
  testType(Type(ElemKind::Int32QTy, {1, 2, 3}, 1.4, 4));
  testType(Type(ElemKind::UInt8ITy, {1, 2, 3}));
  testType(Type(ElemKind::Int32ITy, {1, 2, 3}));
  testType(Type(ElemKind::Int64ITy, {1, 2, 3}));
  testType(Type(ElemKind::UInt8FusedQTy, {1, 2, 3}, 1.5, 5));
  testType(Type(ElemKind::UInt8FusedFP16QTy, {1, 2, 3}, 1.6, 6));
  testType(Type(ElemKind::UInt4FusedFP16QTy, {1, 2, 3}, 1.7, 7));
  testType(Type(ElemKind::UInt4FusedQTy, {1, 2, 3}, 1.7, 7));
  testType(Type(ElemKind::BoolTy, {1, 2, 3}));
}

/// Test unpadded size.
TEST(Tensor, unpaddedSize) {
  Tensor partial(ElemKind::FloatTy, {11});
  PseudoRNG PRNG;
  partial.init(Tensor::InitKind::Broadcast, 5, PRNG);
  auto bytes = partial.getSizeInBytes();

  auto H = partial.getHandle<float>();
  for (const auto &e : H) {
    EXPECT_EQ(e, 5);
  }

  // Get an unowned padded tensor sharing storage with partial.
  auto paddedType = Type::newShape(partial.getType(), {256});
  auto paddedBytes = paddedType.getSizeInBytes();
  Tensor T(partial.getUnsafePtr(), &paddedType, bytes);
  EXPECT_EQ(T.getUnpaddedSizeInBytes(), bytes);
  EXPECT_EQ(T.getSizeInBytes(), paddedBytes);
  EXPECT_EQ(T.getRealNumElements(), 11);
  auto partialH = partial.getHandle<float>();
  int numElemCount = 0;
  for (const auto &e : partialH) {
    EXPECT_EQ(e, 5);
    numElemCount += 1;
  }
  EXPECT_EQ(numElemCount, 11);

  // Test that moving the padded tensor preserves properties.
  auto moved = std::move(T);
  EXPECT_EQ(moved.getUnpaddedSizeInBytes(), bytes);
  EXPECT_EQ(moved.getSizeInBytes(), paddedBytes);

  // Test getting an unowned tensor from a padded tensor.
  auto copy = moved.getUnowned();
  EXPECT_EQ(copy.getUnpaddedSizeInBytes(), bytes);
  EXPECT_EQ(copy.getSizeInBytes(), paddedBytes);

  // Test that a clone of a partial is still partial.
  auto clone = moved.clone();
  EXPECT_EQ(clone.getUnpaddedSizeInBytes(), bytes);
  EXPECT_EQ(clone.getSizeInBytes(), paddedBytes);

  // Test that assigning a Tensor to a partial is still partial.
  Tensor assigned;
  assigned.assign(&moved);
  EXPECT_EQ(assigned.getUnpaddedSizeInBytes(), bytes);
  EXPECT_EQ(assigned.getSizeInBytes(), paddedBytes);

  // Check that when we reset a partial Tensor with the same Type but without
  // specifying the reset should be partial that we do not have the same ptr,
  // as it should have been reallocated.
  char *oldPtr = assigned.getUnsafePtr();
  assigned.reset(paddedType);
  EXPECT_NE(assigned.getUnsafePtr(), oldPtr);
}

TEST(CustomAlignedTensor, sizes) {
  Type T(ElemKind::FloatTy, {2, 2, 1}, {12, 8, 1});
  Tensor aligned(T);

  // EXPECT_EQ(aligned.size(), 4);
  // EXPECT_EQ(aligned.actualSize(), 12);
}

TEST(CustomAlignedTensor, iteration) {
  Type T(ElemKind::FloatTy, {2, 2, 1}, {12, 8, 1});
  Tensor aligned(T);

  auto H = aligned.getHandle<float>();

  std::vector<float> content = {13.5f, -3.3f, 4.2f, 33.0f};
  H.at({0, 0, 0}) = content[0];
  H.at({0, 1, 0}) = content[1];
  H.at({1, 0, 0}) = content[2];
  H.at({1, 1, 0}) = content[3];

  std::vector<float> elems;
  for (auto e : H) {
    elems.push_back(e);
  }

  EXPECT_TRUE(elems == content);
}

TEST(CustomAlignedTensor, raw) {
  Type T(ElemKind::FloatTy, {2, 2, 1}, {12, 8, 1});
  Tensor aligned(T);
  aligned.zero();

  auto H = aligned.getHandle<float>();

  std::vector<float> content{13.5f, -3.3f, 4.2f, 33.0f};
  H.at({0, 0, 0}) = content[0];
  H.at({0, 1, 0}) = content[1];
  H.at({1, 0, 0}) = content[2];
  H.at({1, 1, 0}) = content[3];

  std::vector<float> elems;
  for (size_t i = 0; i < 12; i++) {
    elems.push_back(H.raw(i));
  }

  std::vector<float> alignedContent = {
      13.5, 0, -3.3, 0, 0, 0, 4.2, 0, 33, 0, 0, 0,
  };

  EXPECT_TRUE(elems == alignedContent);
}

TEST(CustomAlignedTensor, getUnowned) {
  Type T(ElemKind::FloatTy, {2, 2, 1}, {12, 8, 1});
  Tensor aligned(T);

  auto H = aligned.getHandle<float>();
  // Fill everything including pads with 1.0
  for (size_t i = 0; i < 12; i++) {
    H.raw(i) = 1.0;
  }

  std::vector<float> content{13.5f, -3.3f, 4.2f, 33.0f};
  H.at({0, 0, 0}) = content[0];
  H.at({0, 1, 0}) = content[1];
  H.at({1, 0, 0}) = content[2];
  H.at({1, 1, 0}) = content[3];

  Tensor UO = aligned.getUnowned({1, 2, 2}, {1, 1, 0});
  EXPECT_EQ(UO.size(), 4);
  EXPECT_EQ(UO.actualSize(), 4);
  EXPECT_EQ(UO.getHandle<float>().at({0, 0, 0}), 33);
  EXPECT_EQ(UO.getHandle<float>().at({0, 0, 1}), 1);
  EXPECT_EQ(UO.getHandle<float>().at({0, 1, 0}), 1);
  EXPECT_EQ(UO.getHandle<float>().at({0, 1, 1}), 1);
  EXPECT_EQ(UO.getHandle<float>().raw(0), 33);
  EXPECT_EQ(UO.getHandle<float>().raw(1), 1);
  EXPECT_EQ(UO.getHandle<float>().raw(2), 1);
  EXPECT_EQ(UO.getHandle<float>().raw(3), 1);
}

TEST(CustomAlignedTensor, getDimForPtr) {
  Type T(ElemKind::FloatTy, {2, 2, 1}, {12, 8, 1});
  Tensor aligned(T);

  auto H = aligned.getHandle<float>();

  EXPECT_EQ(H.getDimForPtr(0, 0), 0);
  EXPECT_EQ(H.getDimForPtr(1, 0), 0);
  EXPECT_EQ(H.getDimForPtr(2, 0), 0);

  EXPECT_EQ(H.getDimForPtr(0, 1), 0);
  EXPECT_EQ(H.getDimForPtr(1, 1), 1);
  EXPECT_EQ(H.getDimForPtr(2, 1), 0);

  EXPECT_EQ(H.getDimForPtr(0, 2), 1);
  EXPECT_EQ(H.getDimForPtr(1, 2), 0);
  EXPECT_EQ(H.getDimForPtr(2, 2), 0);

  EXPECT_EQ(H.getDimForPtr(0, 3), 1);
  EXPECT_EQ(H.getDimForPtr(1, 3), 1);
  EXPECT_EQ(H.getDimForPtr(2, 3), 0);
}

// Check that we iterate over tensors correctly: unit test for a bug wherein
// we used size() instead of actualSize() when treating the data as a raw
// pointer.
TEST(Tensor, sameAlignment) {
  Type Ty1(ElemKind::Float16Ty, {2, 1}, {4, 1});
  Type Ty2(ElemKind::Float16Ty, {2, 1}, {4, 1});
  Tensor T1(Ty1);
  Tensor T2(Ty2);
  auto T1H = T1.getHandle<float16_t>();
  auto T2H = T2.getHandle<float16_t>();
  T1H.clear(0);
  T2H.clear(1);
  T1H.at({0, 0}) = T2H.at({0, 0}) = 1;
  T1H.at({1, 0}) = T2H.at({1, 0}) = 2;

  EXPECT_TRUE(T1.isEqual(T2));
  T2H.at({1, 0}) = 1;
  EXPECT_FALSE(T1.isEqual(T2));
}

// Check that our tensor iteration is aware of padding: unit-test that checks
// we iterate correctly when accessing elements in tensors that have different
// alignment requirements.
TEST(Tensor, differentAlignment) {
  Type Ty1(ElemKind::Float16Ty, {2, 1}, {4, 1});
  Type Ty2(ElemKind::Float16Ty, {2, 1}, {2, 1});
  Tensor T1(Ty1);
  Tensor T2(Ty2);
  auto T1H = T1.getHandle<float16_t>();
  auto T2H = T2.getHandle<float16_t>();
  T1H.at({0, 0}) = T2H.at({0, 0}) = 1;
  T1H.at({1, 0}) = T2H.at({1, 0}) = 2;

  EXPECT_TRUE(T1.isEqual(T2));
  T2H.at({1, 0}) = 1;
  EXPECT_FALSE(T1.isEqual(T2));
}

// Check that write/read of tensors data from/to raw-text files is
// working properly.
TEST(Tensor, accessToTextFile) {
  Tensor tensorRef = {0.75f,  0.23f, 0.76f,  0.99f,  1.00f,
                      -0.78f, 0.23f, -0.97f, -0.37f, 0.00f};
  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile("tensor", ".txt", path);
  if (tempFileRes.value() != 0) {
    FAIL() << "Failed to create temp file to write into.";
  }
  TensorSerializationOptions opts;
  opts.withType = true;
  dumpTensorToTextFile(tensorRef, path, opts);
  Tensor tensorTest;
  loadTensorFromTextFile(tensorTest, path, opts);
  llvm::sys::fs::remove(path);

  auto handleRef = tensorRef.getHandle<>();
  auto handleTest = tensorTest.getHandle<>();

  EXPECT_EQ(handleRef.size(), handleTest.size());
  EXPECT_EQ(handleRef.actualSize(), handleTest.actualSize());
  for (size_t rcnt = 0; rcnt < tensorTest.actualSize(); rcnt++) {
    EXPECT_FLOAT_EQ(handleTest.raw(rcnt), handleRef.raw(rcnt));
  }
}

// Check that write/read of tensors data from/to raw-binary files is
// working properly.
TEST(Tensor, accessToBinaryFile) {
  Tensor tensorRef = {0.75f,  0.23f, 0.76f,  0.99f,  1.00f,
                      -0.78f, 0.23f, -0.97f, -0.37f, 0.00f};
  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile("tensor", ".bin", path);
  if (tempFileRes.value() != 0) {
    FAIL() << "Failed to create temp file to write into.";
  }
  TensorSerializationOptions opts;
  opts.withType = true;
  dumpTensorToBinaryFile(tensorRef, path, opts);
  Tensor tensorTest;
  loadTensorFromBinaryFile(tensorTest, path, opts);
  llvm::sys::fs::remove(path);

  auto handleRef = tensorRef.getHandle<>();
  auto handleTest = tensorTest.getHandle<>();

  EXPECT_EQ(handleRef.size(), handleTest.size());
  EXPECT_EQ(handleRef.actualSize(), handleTest.actualSize());
  for (size_t rcnt = 0; rcnt < tensorTest.actualSize(); rcnt++) {
    EXPECT_FLOAT_EQ(handleTest.raw(rcnt), handleRef.raw(rcnt));
  }
}

// Check that write/read of tensors data from/to raw-text files is
// working properly.
TEST(Tensor, accessToRawTextFile) {
  Tensor tensorRef = {0.75f,  0.23f, 0.76f,  0.99f,  1.00f,
                      -0.78f, 0.23f, -0.97f, -0.37f, 0.00f};
  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile("tensor", ".txt", path);
  if (tempFileRes.value() != 0) {
    FAIL() << "Failed to create temp file to write into.";
  }
  TensorSerializationOptions opts;
  opts.withType = false;
  dumpTensorToTextFile(tensorRef, path, opts);
  Tensor tensorTest(ElemKind::FloatTy, {10});
  loadTensorFromTextFile(tensorTest, path, opts);
  llvm::sys::fs::remove(path);

  auto handleRef = tensorRef.getHandle<>();
  auto handleTest = tensorTest.getHandle<>();

  EXPECT_EQ(handleRef.size(), handleTest.size());
  EXPECT_EQ(handleRef.actualSize(), handleTest.actualSize());
  for (size_t rcnt = 0; rcnt < tensorTest.actualSize(); rcnt++) {
    EXPECT_FLOAT_EQ(handleTest.raw(rcnt), handleRef.raw(rcnt));
  }
}

#ifdef WITH_PNG

/// Testing loading of input tensors from a file.
static void tensorInputWriterLoader(ImageLayout outImageLayout,
                                    ImageLayout inImageLayout) {
  Tensor tensorRef(Type{ElemKind::FloatTy, {1, 2, 4, 3}});
  tensorRef.getHandle<>() = {0.75f,  0.23f,  0.76f,  0.99f,  1.00f, -0.78f,
                             0.23f,  -0.97f, -0.37f, 0.00f,  0.25f, 0.13f,
                             0.66f,  0.69f,  2.00f,  -0.18f, 0.43f, -0.92f,
                             -0.33f, 0.01f,  0.21f,  0.11f,  0.13f, 0.87f};
  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile("tensor", ".txt", path);
  if (tempFileRes.value() != 0) {
    FAIL() << "Failed to create temp file to write into.";
  }
  dumpInputTensorToFileWithType({path.str().str()}, tensorRef, outImageLayout);
  //
  Tensor tensorTest;
  loadInputImageFromFileWithType({path.str().str()}, &tensorTest,
                                 inImageLayout);

  if (outImageLayout == ImageLayout::NHWC) {
    Tensor transposed;
    tensorRef.transpose(&transposed, NHWC2NCHW);
    tensorRef = std::move(transposed);
  }

  if (inImageLayout == ImageLayout::NHWC) {
    Tensor transposed;
    tensorTest.transpose(&transposed, NHWC2NCHW);
    tensorTest = std::move(transposed);
  }

  auto handleRef = tensorRef.getHandle<>();
  auto handleTest = tensorTest.getHandle<>();
  EXPECT_EQ(handleRef.size(), handleTest.size());
  EXPECT_EQ(tensorRef.dims(), tensorTest.dims());
  for (size_t rcnt = 0, e = tensorTest.actualSize(); rcnt < e; rcnt++) {
    EXPECT_FLOAT_EQ(handleTest.raw(rcnt), handleRef.raw(rcnt));
  }
}

TEST(Tensor, tensorInputWriterLoaderNCHW) {
  tensorInputWriterLoader(ImageLayout::NCHW, ImageLayout::NCHW);
}

TEST(Tensor, tensorInputWriterLoaderNCHW_NHWC) {
  tensorInputWriterLoader(ImageLayout::NCHW, ImageLayout::NHWC);
}

TEST(Tensor, tensorInputWriterLoaderNHWC_NCHW) {
  tensorInputWriterLoader(ImageLayout::NHWC, ImageLayout::NCHW);
}

TEST(Tensor, tensorInputWriterLoaderNHWC) {
  tensorInputWriterLoader(ImageLayout::NHWC, ImageLayout::NHWC);
}

// Test custom input tensor loader
TEST(Tensor, tensorCustomInputLoader) {
  bool entered = false;
  auto loader = [&entered](Tensor &T, llvm::StringRef filename,
                           ImageLayout imageLayout) {
    EXPECT_EQ(imageLayout, ImageLayout::NHWC);
    EXPECT_EQ(filename, "input.tensor");
    T.reset(ElemKind::FloatTy, {1, 2, 3, 4});
    entered = true;
  };
  Tensor testT(Type{ElemKind::Int32ITy, {4, 4, 4, 4}});
  registerInputTensorFileLoader(loader);
  loadInputImageFromFileWithType({"input.tensor"}, &testT, ImageLayout::NHWC);
  EXPECT_EQ(entered, true);
  EXPECT_EQ(testT.dims(), llvm::ArrayRef<dim_t>({1, 2, 3, 4}));
}

#endif // WITH_PNG

// Check that write/read of tensors data from/to raw-binary files is
// working properly.
TEST(Tensor, accessToRawBinaryFile) {
  Tensor tensorRef = {0.75f,  0.23f, 0.76f,  0.99f,  1.00f,
                      -0.78f, 0.23f, -0.97f, -0.37f, 0.00f};
  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile("tensor", ".bin", path);
  if (tempFileRes.value() != 0) {
    FAIL() << "Failed to create temp file to write into.";
  }
  TensorSerializationOptions opts;
  opts.withType = false;
  dumpTensorToBinaryFile(tensorRef, path, opts);
  Tensor tensorTest(ElemKind::FloatTy, {10});
  loadTensorFromBinaryFile(tensorTest, path, opts);
  llvm::sys::fs::remove(path);

  auto handleRef = tensorRef.getHandle<>();
  auto handleTest = tensorTest.getHandle<>();

  EXPECT_EQ(handleRef.size(), handleTest.size());
  EXPECT_EQ(handleRef.actualSize(), handleTest.actualSize());
  for (size_t rcnt = 0; rcnt < tensorTest.actualSize(); rcnt++) {
    EXPECT_FLOAT_EQ(handleTest.raw(rcnt), handleRef.raw(rcnt));
  }
}

/// Test convert UInt4FusedFP16QTy tensor to a UInt8FusedQTy tensor.
TEST(Tensor, typeConvert_UInt4FusedFP16QTy_To_UInt8FusedQTY) {
  testConvertToUInt8FusedQTy<float16_t>(ElemKind::UInt4FusedFP16QTy, 10, 10);
}

/// Test convert UInt8FusedFP16QTy tensor to a UInt8FusedQTy tensor.
TEST(Tensor, typeConvert_UInt8FusedFP16QTy_To_UInt8FusedQTy) {
  testConvertToUInt8FusedQTy<float16_t>(ElemKind::UInt8FusedFP16QTy, 10, 10);
}

/// Test convert UInt4FusedQTy tensor to a UInt8FusedQTy tensor.
TEST(Tensor, typeConvert_UInt4FusedQTy_To_UInt8FusedQTy) {
  testConvertToUInt8FusedQTy<float>(ElemKind::UInt4FusedQTy, 10, 10);
}
