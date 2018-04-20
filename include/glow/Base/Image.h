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
/// Reads a png image header from png file \p filename and \returns a tuple
/// containing height, width, and a bool if it is grayscale or not.
std::tuple<size_t, size_t, bool> getPngInfo(const char *filename);

/// Reads a png image. \returns True if an error occurred. The values of the
/// image are in the range \p range.
bool readPngImage(Tensor *T, const char *filename,
                  std::pair<float, float> range);

/// Writes a png image. \returns True if an error occurred. The values of the
/// image are in the range \p range.
bool writePngImage(Tensor *T, const char *filename,
                   std::pair<float, float> range);

} // namespace glow

#endif // GLOW_BASE_IMAGE_H
