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

#include "glow/Base/Image.h"

#include "llvm/Support/FileSystem.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <utility>

using namespace glow;

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
  size_t imgHeight = image1.dims()[0];
  size_t imgWidth = image1.dims()[1];
  size_t numChannels = image1.dims()[2];

  Tensor transposed;
  image2.transpose(&transposed, {1u, 2u, 0u});
  image2 = std::move(transposed);

  Tensor swizzled(image1.getType());
  auto IH = image1.getHandle();
  auto SH = swizzled.getHandle();
  for (size_t z = 0; z < numChannels; z++) {
    for (size_t y = 0; y < imgHeight; y++) {
      for (size_t x = 0; x < imgWidth; x++) {
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
