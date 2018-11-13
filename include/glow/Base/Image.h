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

enum class ImageNormalizationMode {
  kneg1to1,     // Values are in the range: -1 and 1.
  k0to1,        // Values are in the range: 0 and 1.
  k0to255,      // Values are in the range: 0 and 255.
  kneg128to127, // Values are in the range: -128 .. 127
};

enum class ImageLayout {
  NCHW,
  NHWC,
};

enum class ImageChannelOrder {
  BGR,
  RGB,
};

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

Tensor readPngImageAndPreprocess(const std::string &filename,
                                 ImageNormalizationMode imageNormMode,
                                 ImageChannelOrder imageChannelOrder,
                                 ImageLayout imageLayer,
                                 bool useImagenetNormalization);
} // namespace glow

#endif // GLOW_BASE_IMAGE_H
