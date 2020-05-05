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

static void numpyTestHelper(llvm::ArrayRef<std::string> filenames,
                            llvm::ArrayRef<dim_t> expDims,
                            std::vector<float> &vals, ImageLayout inLayout,
                            ImageLayout imgLayout,
                            llvm::ArrayRef<float> mean = {},
                            llvm::ArrayRef<float> stddev = {}) {
  Tensor image;
  loadNumpyImagesAndPreprocess(filenames, image,
                               ImageNormalizationMode::k0to255, inLayout,
                               imgLayout, mean, stddev);

  ASSERT_EQ(ElemKind::FloatTy, image.getType().getElementType());
  ASSERT_EQ(expDims.size(), image.dims().size());
  EXPECT_EQ(image.dims(), expDims);
  auto H = image.getHandle();
  for (dim_t i = 0; i < H.size(); i++) {
    EXPECT_FLOAT_EQ(H.raw(i), vals[i]);
  }
}

TEST(Image, readNpyNCHWtoNCHW_4D_image) {
  std::vector<float> vals;
  for (int i = 0; i < 48; i++) {
    vals.push_back(i);
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2x2_u16.npy"}, {3, 4, 2, 2},
                  vals, ImageLayout::NCHW, ImageLayout::NCHW);
}

TEST(Image, readNpy_stddev_mean) {
  std::vector<float> vals;
  std::vector<float> mean = {1.1, 1.2, 1.3, 1.4};
  std::vector<float> stddev = {2.1, 2.2, 2.3, 2.4};
  for (int i = 0; i < 48; i++) {
    vals.push_back(((float)i - mean[(i / 4) % 4]) / stddev[(i / 4) % 4]);
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2x2_u16.npy"}, {3, 4, 2, 2},
                  vals, ImageLayout::NCHW, ImageLayout::NCHW,
                  {1.1, 1.2, 1.3, 1.4}, {2.1, 2.2, 2.3, 2.4});
}

TEST(Image, readNpyNHWCtoNHWC_3D_image) {
  std::vector<float> vals;
  for (int i = 0; i < 24; i++) {
    vals.push_back(i);
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2_u32.npy"}, {1, 3, 4, 2}, vals,
                  ImageLayout::NHWC, ImageLayout::NHWC);
}

TEST(Image, readNpyNCHWtoNHWC_4D_image) {
  std::vector<float> vals;
  for (int i = 0; i < 48; i++) {
    vals.push_back(i);
  }
  Tensor tensor(ElemKind::FloatTy, {3, 4, 2, 2});
  tensor.getHandle() = vals;
  Tensor transposed;
  tensor.transpose(&transposed, {0u, 2u, 3u, 1u});
  vals.clear();
  for (int i = 0; i < 48; i++) {
    vals.push_back(transposed.getHandle().raw(i));
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2x2_u16.npy"}, transposed.dims(),
                  vals, ImageLayout::NHWC, ImageLayout::NCHW);
}

TEST(Image, readNpyNHWCtoNCHW_3D_image) {
  std::vector<float> vals;
  for (int i = 0; i < 24; i++) {
    vals.push_back(i);
  }
  Tensor tensor(ElemKind::FloatTy, {1, 3, 4, 2});
  tensor.getHandle() = vals;
  Tensor transposed;
  tensor.transpose(&transposed, {0u, 3u, 1u, 2u});
  vals.clear();
  for (int i = 0; i < 24; i++) {
    vals.push_back(transposed.getHandle().raw(i));
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2_u32.npy"}, transposed.dims(),
                  vals, ImageLayout::NCHW, ImageLayout::NHWC);
}

TEST(Image, readNpyNHWCtoNHWC_multi_image) {
  std::vector<float> vals;
  for (int i = 0; i < 16; i++) {
    vals.push_back(i);
  }
  for (int i = 0; i < 48; i++) {
    vals.push_back(i);
  }
  numpyTestHelper({"tests/images/npy/tensor1x4x2x2_u8.npy",
                   "tests/images/npy/tensor3x4x2x2_u16.npy"},
                  {4, 4, 2, 2}, vals, ImageLayout::NHWC, ImageLayout::NHWC);
}

TEST(Image, readNonSquarePngImage) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor vgaTensor;
  bool loadSuccess =
      !readPngImage(&vgaTensor, "tests/images/other/vga_image.png", range);
  ASSERT_TRUE(loadSuccess);

  auto &type = vgaTensor.getType();
  auto shape = vgaTensor.dims();

  // The loaded image is a 3D HWC tensor
  ASSERT_EQ(ElemKind::FloatTy, type.getElementType());
  ASSERT_EQ(3, shape.size());
  ASSERT_EQ(480, shape[0]);
  ASSERT_EQ(640, shape[1]);
  ASSERT_EQ(3, shape[2]);
}

TEST(Image, readBadImages) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor tensor;
  bool loadSuccess =
      !readPngImage(&tensor, "tests/images/other/dog_corrupt.png", range);
  ASSERT_FALSE(loadSuccess);

  loadSuccess =
      !readPngImage(&tensor, "tests/images/other/ghost_missing.png", range);
  ASSERT_FALSE(loadSuccess);
}

TEST(Image, readPngImageAndPreprocessWithAndWithoutInputTensor) {
  auto image1 = readPngImageAndPreprocess(
      "tests/images/imagenet/cat_285.png", ImageNormalizationMode::k0to1,
      ImageChannelOrder::RGB, ImageLayout::NHWC, imagenetNormMean,
      imagenetNormStd);
  Tensor image2;
  readPngImageAndPreprocess(image2, "tests/images/imagenet/cat_285.png",
                            ImageNormalizationMode::k0to1,
                            ImageChannelOrder::BGR, ImageLayout::NCHW,
                            imagenetNormMean, imagenetNormStd);

  // Test if the preprocess actually happened.
  dim_t imgHeight = image1.dims()[0];
  dim_t imgWidth = image1.dims()[1];
  dim_t numChannels = image1.dims()[2];

  Tensor transposed;
  image2.transpose(&transposed, {1u, 2u, 0u});
  image2 = std::move(transposed);

  Tensor swizzled(image1.getType());
  auto IH = image1.getHandle();
  auto SH = swizzled.getHandle();
  for (dim_t z = 0; z < numChannels; z++) {
    for (dim_t y = 0; y < imgHeight; y++) {
      for (dim_t x = 0; x < imgWidth; x++) {
        SH.at({x, y, numChannels - 1 - z}) = IH.at({x, y, z});
      }
    }
  }
  image1 = std::move(swizzled);
  EXPECT_TRUE(image1.isEqual(image2, 0.01));
}

TEST(Image, writePngImage) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor localCopy;
  bool loadSuccess =
      !readPngImage(&localCopy, "tests/images/imagenet/cat_285.png", range);
  ASSERT_TRUE(loadSuccess);

  llvm::SmallVector<char, 10> resultPath;
  llvm::sys::fs::createTemporaryFile("prefix", "suffix", resultPath);
  std::string outfilename(resultPath.begin(), resultPath.end());

  bool storeSuccess = !writePngImage(&localCopy, outfilename.c_str(), range);
  ASSERT_TRUE(storeSuccess);

  Tensor secondLocalCopy;
  loadSuccess = !readPngImage(&secondLocalCopy, outfilename.c_str(), range);
  ASSERT_TRUE(loadSuccess);
  EXPECT_TRUE(secondLocalCopy.isEqual(localCopy, 0.01));

  // Delete the temporary file.
  std::remove(outfilename.c_str());
}

/// Test writing a png image along with using the standard Imagenet
/// normalization when reading the image.
TEST(Image, writePngImageWithImagenetNormalization) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor localCopy;
  bool loadSuccess =
      !readPngImage(&localCopy, "tests/images/imagenet/cat_285.png", range,
                    imagenetNormMean, imagenetNormStd);
  ASSERT_TRUE(loadSuccess);

  llvm::SmallVector<char, 10> resultPath;
  llvm::sys::fs::createTemporaryFile("prefix", "suffix", resultPath);
  std::string outfilename(resultPath.begin(), resultPath.end());

  bool storeSuccess = !writePngImage(&localCopy, outfilename.c_str(), range,
                                     imagenetNormMean, imagenetNormStd);
  ASSERT_TRUE(storeSuccess);

  Tensor secondLocalCopy;
  loadSuccess = !readPngImage(&secondLocalCopy, outfilename.c_str(), range,
                              imagenetNormMean, imagenetNormStd);
  ASSERT_TRUE(loadSuccess);
  EXPECT_TRUE(secondLocalCopy.isEqual(localCopy, 0.02));

  // Delete the temporary file.
  std::remove(outfilename.c_str());
}
