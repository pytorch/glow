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
#ifndef GLOW_BASE_IMAGE_H
#define GLOW_BASE_IMAGE_H

#include "glow/Base/Tensor.h"

#include <tuple>

namespace glow {

/// Pixel value ranges.
enum class ImageNormalizationMode {
  kneg1to1,     // Values are in the range: -1 and 1.
  k0to1,        // Values are in the range: 0 and 1.
  k0to255,      // Values are in the range: 0 and 255.
  kneg128to127, // Values are in the range: -128 .. 127
};

/// Layout of image dimensions (batch, channels, height, width).
enum class ImageLayout {
  NCHW,
  NHWC,
};

/// Order of color channels (red, green, blue).
enum class ImageChannelOrder {
  BGR,
  RGB,
};

/// -image_mode flag.
extern ImageNormalizationMode imageNormMode;

/// -image_channel_order flag.
extern ImageChannelOrder imageChannelOrder;

/// -image_layout flag.
extern ImageLayout imageLayout;

/// -use-imagenet-normalization flag.
extern bool useImagenetNormalization;

/// \returns the floating-point range corresponding to enum value \p mode.
std::pair<float, float> normModeToRange(ImageNormalizationMode mode);

/// Reads a png image header from png file \p filename and \returns a tuple
/// containing height, width, and a bool if it is grayscale or not.
std::tuple<size_t, size_t, bool> getPngInfo(const char *filename);

/// Reads a png image. \returns True if an error occurred. The values of the
/// image are in the range \p range. If \p useImagenetNormalization then
/// specialized normalization for Imagenet images is applied to the image.
/// \pre !(useImagenetNormalization && numChannels != 3)
bool readPngImage(Tensor *T, const char *filename,
                  std::pair<float, float> range,
                  bool useImagenetNormalization = false);

/// Writes a png image. \returns True if an error occurred. The values of the
/// image are in the range \p range. If \p useImagenetNormalization then
/// specialized normalization for Imagenet images is unapplied to the image.
bool writePngImage(Tensor *T, const char *filename,
                   std::pair<float, float> range,
                   bool useImagenetNormalization = false);

/// Read a png image and preprocess it according to several parameters.
/// \param filename the png file to read.
/// \param imageNormMode normalize values to this range.
/// \param imageChannelOrder the order of color channels.
/// \param imageLayout the order of dimensions (channel, height, and width).
/// \param useImagenetNormalization use special normalization for Imagenet.
Tensor readPngImageAndPreprocess(llvm::StringRef filename,
                                 ImageNormalizationMode imageNormMode,
                                 ImageChannelOrder imageChannelOrder,
                                 ImageLayout imageLayout,
                                 bool useImagenetNormalization);
} // namespace glow

#endif // GLOW_BASE_IMAGE_H
