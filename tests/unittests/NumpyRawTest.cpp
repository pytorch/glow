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

#include "glow/Base/Image.h"

#include "llvm/Support/FileSystem.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <utility>

using namespace glow;

class NumpyRawTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

template <typename T>
static void numpyTestHelper(llvm::ArrayRef<std::string> filenames,
                            llvm::ArrayRef<dim_t> expDims, std::vector<T> &vals,
                            ElemKind ek, bool unsqueeze = true) {
  Tensor tensor;
  loadNumpyRaw({filenames}, tensor, unsqueeze);

  ASSERT_EQ(ek, tensor.getType().getElementType());
  ASSERT_EQ(expDims.size(), tensor.dims().size());
  EXPECT_EQ(tensor.dims(), expDims);
  auto H = tensor.getHandle<T>();
  for (dim_t i = 0; i < H.size(); i++) {
    switch (ek) {
    case ElemKind::UInt8ITy:
    case ElemKind::Int64ITy:
    case ElemKind::Int32ITy:
      EXPECT_EQ(H.raw(i), vals[i]);
      break;
    case ElemKind::FloatTy:
    case ElemKind::Float64Ty:
      EXPECT_NEAR(H.raw(i), vals[i], 0.000001) << "at index: " << i;
      break;
    default:
      FAIL() << "Unsupported ElemKind";
    }
  }
}

// Test loading numpy U8 tensor
TEST_F(NumpyRawTest, readNpyTensor1D_U8) {
  std::vector<uint8_t> vals;
  for (int i = 0; i < 48; i++) {
    vals.push_back(i);
  }
  numpyTestHelper<uint8_t>({"tests/images/npy/tensor48_u8.npy"}, {1, 48}, vals,
                           ElemKind::UInt8ITy);
}

// Test loading numpy i32 tensor
TEST_F(NumpyRawTest, readNpyTensor_1_2_4_i32) {
  std::vector<int32_t> vals;
  for (auto i = 1000; i < 1008; i++) {
    vals.push_back(static_cast<int32_t>(i));
  }

  // Test with file count inserted at dimension 0
  numpyTestHelper<int32_t>({"tests/images/npy/tensor_1x2x4_i32.npy"},
                           {1, 1, 2, 4}, vals, ElemKind::Int32ITy);

  // Test without file count inserted at dimension 0
  numpyTestHelper<int32_t>({"tests/images/npy/tensor_1x2x4_i32.npy"}, {1, 2, 4},
                           vals, ElemKind::Int32ITy, false);
}

// Test loading numpy I64 tensor
TEST_F(NumpyRawTest, readNpyTensor_1_2_4_i64) {
  std::vector<int64_t> vals;
  for (auto i = 1000; i < 1008; i++) {
    vals.push_back(static_cast<int64_t>(i));
  }

  // Test with file count inserted at dimension 0
  numpyTestHelper<int64_t>({"tests/images/npy/tensor_1x2x4_i64.npy"},
                           {1, 1, 2, 4}, vals, ElemKind::Int64ITy);

  // Test without file count inserted at dimension 0
  numpyTestHelper<int64_t>({"tests/images/npy/tensor_1x2x4_i64.npy"}, {1, 2, 4},
                           vals, ElemKind::Int64ITy, false);
}

// Test loading numpy F32 tensor
TEST_F(NumpyRawTest, readNpyTensor_1_2_4_f32) {
  std::vector<float> vals;
  for (auto i = 1000; i < 1008; i++) {
    vals.push_back(static_cast<float>(i));
  }

  // Test with file count inserted at dimension 0
  numpyTestHelper<float>({"tests/images/npy/tensor_1x2x4_f32.npy"},
                         {1, 1, 2, 4}, vals, ElemKind::FloatTy);

  // Test without file count inserted at dimension 0
  numpyTestHelper<float>({"tests/images/npy/tensor_1x2x4_f32.npy"}, {1, 2, 4},
                         vals, ElemKind::FloatTy, false);
}

// Test loading numpy F64 tensor
TEST_F(NumpyRawTest, readNpyTensor_1_2_4_f64) {
  std::vector<double> vals;
  for (auto i = 1000; i < 1008; i++) {
    vals.push_back(static_cast<double>(i));
  }

  // Test with file count inserted at dimension 0
  numpyTestHelper<double>({"tests/images/npy/tensor_1x2x4_f64.npy"},
                          {1, 1, 2, 4}, vals, ElemKind::Float64Ty);

  // Test without file count inserted at dimension 0
  numpyTestHelper<double>({"tests/images/npy/tensor_1x2x4_f64.npy"}, {1, 2, 4},
                          vals, ElemKind::Float64Ty, false);
}
