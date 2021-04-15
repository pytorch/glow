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

class ImageTest : public ::testing::Test {
protected:
  void SetUp() override { initImageCmdArgVars(); }
  void TearDown() override {}
};

static void numpyTestHelper(llvm::ArrayRef<std::string> filenames,
                            llvm::ArrayRef<dim_t> expDims,
                            std::vector<float> &vals,
                            llvm::ArrayRef<ImageLayout> imgLayout,
                            llvm::ArrayRef<ImageLayout> inLayout,
                            llvm::ArrayRef<ImageNormalizationMode> normMode,
                            VecVecRef<float> mean = {{}},
                            VecVecRef<float> stddev = {{}}) {
  Tensor image;
  loadImagesAndPreprocess({filenames}, {&image}, normMode,
                          {ImageChannelOrder::RGB}, imgLayout, inLayout, mean,
                          stddev);

  ASSERT_EQ(ElemKind::FloatTy, image.getType().getElementType());
  ASSERT_EQ(expDims.size(), image.dims().size());
  EXPECT_EQ(image.dims(), expDims);
  auto H = image.getHandle();
  for (dim_t i = 0; i < H.size(); i++) {
    EXPECT_NEAR(H.raw(i), vals[i], 0.000001) << "at index: " << i;
  }
}

// Test loading numpy 1D U8 tensor with mean/stddev.
TEST_F(ImageTest, readNpyTensor1D_U8) {
  std::vector<float> vals;
  for (int i = 0; i < 48; i++) {
    vals.push_back((i - 5) / 2.);
  }
  numpyTestHelper({"tests/images/npy/tensor48_u8.npy"}, {48}, vals,
                  {ImageLayout::Unspecified}, {},
                  ImageNormalizationMode::k0to255, {{5.}}, {{2.}});
}

// Test loading numpy 1D U8 tensor with mean/stddev and normalization.
TEST_F(ImageTest, readNpyTensor1D_U8Norm) {
  std::vector<float> vals = {
      -1.000000, -0.992157, -0.984314, -0.976471, -0.968627, -0.960784,
      -0.952941, -0.945098, -0.937255, -0.929412, -0.921569, -0.913725,
      -0.905882, -0.898039, -0.890196, -0.882353, -0.874510, -0.866667,
      -0.858824, -0.850980, -0.843137, -0.835294, -0.827451, -0.819608,
      -0.811765, -0.803922, -0.796078, -0.788235, -0.780392, -0.772549,
      -0.764706, -0.756863, -0.749020, -0.741176, -0.733333, -0.725490,
      -0.717647, -0.709804, -0.701961, -0.694118, -0.686275, -0.678431,
      -0.670588, -0.662745, -0.654902, -0.647059, -0.639216, -0.631373};
  numpyTestHelper({"tests/images/npy/tensor48_u8.npy"}, {48}, vals,
                  {ImageLayout::Unspecified}, {},
                  ImageNormalizationMode::kneg1to1, {{0.}}, {{1.}});
}

// Test loading numpy 1D I8 tensors.
TEST_F(ImageTest, readNpyTensor1D_I8) {
  std::vector<float> vals;
  for (int i = 0; i < 48; i++) {
    // mean adjusted as loader converts S8 as U8.
    vals.push_back((i - 5 + 128) / 2.);
  }
  numpyTestHelper({"tests/images/npy/tensor48_i8.npy"}, {48}, vals,
                  ImageLayout::Unspecified, {}, {}, {{5.}}, {{2.}});
}

// Test loading numpy 1D I8 with normalization tensors.
TEST_F(ImageTest, readNpyTensor1D_I8Norm) {
  std::vector<float> vals = {
      0.003922, 0.011765, 0.019608, 0.027451, 0.035294, 0.043137, 0.050980,
      0.058824, 0.066667, 0.074510, 0.082353, 0.090196, 0.098039, 0.105882,
      0.113726, 0.121569, 0.129412, 0.137255, 0.145098, 0.152941, 0.160784,
      0.168628, 0.176471, 0.184314, 0.192157, 0.200000, 0.207843, 0.215686,
      0.223529, 0.231373, 0.239216, 0.247059, 0.254902, 0.262745, 0.270588,
      0.278431, 0.286275, 0.294118, 0.301961, 0.309804, 0.317647, 0.325490,
      0.333333, 0.341177, 0.349020, 0.356863, 0.364706, 0.372549};
  numpyTestHelper({"tests/images/npy/tensor48_i8.npy"}, {48}, vals,
                  {ImageLayout::Unspecified}, {},
                  ImageNormalizationMode::kneg1to1, {{0.}}, {{1.}});
}

// Test loading numpy 2D U8 tensors.
TEST_F(ImageTest, readNpyTensor2D_U8) {
  std::vector<float> vals;
  for (int i = 0; i < 48; i++) {
    vals.push_back((i - 2) / 3.);
  }
  numpyTestHelper({"tests/images/npy/tensor3x16_u8.npy"}, {3, 16}, vals,
                  {ImageLayout::Unspecified}, {}, {}, {{2.}}, {{3.}});
}

// Test loading numpy 3D U8 tensors.
TEST_F(ImageTest, readNpyTensor3D_U8) {
  std::vector<float> vals;
  for (int i = 0; i < 48; i++) {
    vals.push_back((i - 2) / 3.);
  }
  numpyTestHelper({"tests/images/npy/tensor2x3x8_u8.npy"}, {2, 3, 8}, vals,
                  {ImageLayout::Unspecified}, {}, {}, {{2.}}, {{3.}});
}

// Test loading numpy 4D U8 tensors.
TEST_F(ImageTest, readNpyTensor4D_U8) {
  std::vector<float> vals;
  for (int i = 0; i < 48; i++) {
    vals.push_back((i - 2) / 3.);
  }
  numpyTestHelper({"tests/images/npy/tensor1x2x3x8_u8.npy"}, {1, 2, 3, 8}, vals,
                  {ImageLayout::Unspecified}, {}, {}, {{2.}}, {{3.}});
}

// Test loading from numpy file w/o changing layout.
TEST_F(ImageTest, readNpyNCHWtoNCHW_4D_image) {
  std::vector<float> vals;
  for (int i = 0; i < 48; i++) {
    vals.push_back(i);
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2x2_u8.npy"}, {3, 4, 2, 2}, vals,
                  {ImageLayout::NCHW}, {ImageLayout::NCHW}, {});
}

// Test loading from numpy file with mean/stddev.
TEST_F(ImageTest, readNpy_stddev_mean) {
  std::vector<float> vals;
  std::vector<float> mean = {1.1, 1.2, 1.3, 1.4};
  std::vector<float> stddev = {2.1, 2.2, 2.3, 2.4};
  for (int i = 0; i < 48; i++) {
    vals.push_back(((float)i - mean[(i / 4) % 4]) / stddev[(i / 4) % 4]);
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2x2_u8.npy"}, {3, 4, 2, 2}, vals,
                  {ImageLayout::NCHW}, {ImageLayout::NCHW}, {}, {mean},
                  {stddev});
}

// Test loading 3D image from numpy file.
TEST_F(ImageTest, readNpyNHWCtoNHWC_3D_image) {
  std::vector<float> vals;
  for (int i = 0; i < 24; i++) {
    vals.push_back(i);
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2_u8.npy"}, {1, 3, 4, 2}, vals,
                  {ImageLayout::NHWC}, {ImageLayout::NHWC}, {});
}

// Test loading from numpy file with change of layout.
TEST_F(ImageTest, readNpyNCHWtoNHWC_4D_image) {
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
  numpyTestHelper({"tests/images/npy/tensor3x4x2x2_u8.npy"}, transposed.dims(),
                  vals, {ImageLayout::NHWC}, {ImageLayout::NCHW}, {});
}

// Test loading 3D image from numpy file with change of layout.
TEST_F(ImageTest, readNpyNHWCtoNCHW_3D_image) {
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
    vals.push_back(transposed.getHandle().raw(i) + 128);
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2_i8.npy"}, transposed.dims(),
                  vals, {ImageLayout::NCHW}, {ImageLayout::NHWC}, {});
}

// Test loading multiple images from numpy files.
TEST_F(ImageTest, readNpyNHWCtoNHWC_multi_image) {
  std::vector<float> vals;
  for (int i = 0; i < 48; i++) {
    vals.push_back(i);
  }
  for (int i = 0; i < 48; i++) {
    // S8 tensor - adjust mean.
    vals.push_back(i + 128);
  }
  numpyTestHelper({"tests/images/npy/tensor3x4x2x2_u8.npy",
                   "tests/images/npy/tensor3x4x2x2_i8.npy"},
                  {6, 4, 2, 2}, vals, {ImageLayout::NHWC}, {ImageLayout::NHWC},
                  {});
}

TEST_F(ImageTest, readNonSquarePngImage) {
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

TEST_F(ImageTest, readBadImages) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor tensor;
  bool loadSuccess =
      !readPngImage(&tensor, "tests/images/other/dog_corrupt.png", range);
  ASSERT_FALSE(loadSuccess);

  loadSuccess =
      !readPngImage(&tensor, "tests/images/other/ghost_missing.png", range);
  ASSERT_FALSE(loadSuccess);
}

TEST_F(ImageTest, readPngImageAndPreprocessWithAndWithoutInputTensor) {
  auto image1 = readPngImageAndPreprocess(
      "tests/images/imagenet/cat_285.png", ImageNormalizationMode::k0to1,
      ImageChannelOrder::RGB, ImageLayout::NHWC, imagenetNormMean,
      imagenetNormStd);

  Tensor image2;
  std::vector<float> meanBGR(llvm::makeArrayRef(imagenetNormMean));
  std::vector<float> stddevBGR(llvm::makeArrayRef(imagenetNormStd));
  std::reverse(meanBGR.begin(), meanBGR.end());
  std::reverse(stddevBGR.begin(), stddevBGR.end());
  readPngImageAndPreprocess(image2, "tests/images/imagenet/cat_285.png",
                            ImageNormalizationMode::k0to1,
                            ImageChannelOrder::BGR, ImageLayout::NCHW, meanBGR,
                            stddevBGR);

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

TEST_F(ImageTest, writePngImage) {
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

TEST_F(ImageTest, readMultipleInputsOpt) {
  imageLayoutOpt = {ImageLayout::NCHW, ImageLayout::NCHW};
  meanValuesOpt = {{127.5, 127.5, 127.5}, {0, 0, 0}};
  stddevValuesOpt = {{2, 2, 2}, {1, 1, 1}};
  imageChannelOrderOpt = {ImageChannelOrder::RGB, ImageChannelOrder::RGB};
  imageNormMode = {ImageNormalizationMode::k0to255,
                   ImageNormalizationMode::k0to255};

  std::vector<std::vector<std::string>> filenamesList = {
      {"tests/images/imagenet/cat_285.png"},
      {"tests/images/imagenet/cat_285.png"}};
  Tensor image1;
  Tensor image2;
  loadImagesAndPreprocess(filenamesList, {&image1, &image2});

  auto H1 = image1.getHandle();
  auto H2 = image2.getHandle();
  EXPECT_EQ(H1.size(), H2.size());
  for (dim_t i = 0; i < H1.size(); i++) {
    EXPECT_FLOAT_EQ((H2.raw(i) - 127.5) / 2, H1.raw(i));
  }
}

TEST_F(ImageTest, readMultipleInputsApi) {
  std::vector<ImageLayout> layout = {ImageLayout::NHWC, ImageLayout::NHWC};
  std::vector<std::vector<float>> mean = {{100, 100, 100}, {0, 0, 0}};
  std::vector<std::vector<float>> stddev = {{1.5, 1.5, 1.5}, {1, 1, 1}};
  std::vector<ImageChannelOrder> chOrder = {ImageChannelOrder::BGR,
                                            ImageChannelOrder::BGR};
  std::vector<ImageNormalizationMode> norm = {ImageNormalizationMode::k0to1,
                                              ImageNormalizationMode::k0to1};

  std::vector<std::vector<std::string>> filenamesList = {
      {"tests/images/imagenet/cat_285.png"},
      {"tests/images/imagenet/cat_285.png"}};
  Tensor image1;
  Tensor image2;
  loadImagesAndPreprocess(filenamesList, {&image1, &image2}, norm, chOrder,
                          layout, {}, mean, stddev);

  auto H1 = image1.getHandle();
  auto H2 = image2.getHandle();
  EXPECT_EQ(H1.size(), H2.size());
  for (dim_t i = 0; i < H1.size(); i++) {
    EXPECT_NEAR((H2.raw(i) - (100 / 255.)) / 1.5, H1.raw(i), 0.0000001);
  }
}

/// Test writing a png image along with using the standard Imagenet
/// normalization when reading the image.
TEST_F(ImageTest, writePngImageWithImagenetNormalization) {
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

/// Test PNG w/ order and layout transposes, and different mean/stddev per
/// channel.
TEST_F(ImageTest, readNonSquarePngBGRNCHWTest) {
  auto image = readPngImageAndPreprocess(
      "tests/images/other/tensor_2x4x3.png", ImageNormalizationMode::k0to255,
      ImageChannelOrder::BGR, ImageLayout::NCHW, {0, 1, 2}, {3, 4, 5});

  std::vector<float> expected = {1.,   2.0, 3.,   4.,  5.,   6.0, 7.,   8.,
                                 0.25, 1.,  1.75, 2.5, 3.25, 4.,  4.75, 5.5,
                                 -0.2, 0.4, 1.,   1.6, 2.2,  2.8, 3.4,  4.};

  auto H = image.getHandle();
  for (dim_t i = 0; i < H.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], H.raw(i));
  }
}
