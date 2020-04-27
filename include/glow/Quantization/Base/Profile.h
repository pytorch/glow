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

#ifndef GLOW_QUANTIZATION_BASE_PROFILE_H
#define GLOW_QUANTIZATION_BASE_PROFILE_H

#include "glow/Base/Tensor.h"

namespace glow {
namespace quantization {

/// Generate input tensor histogram based on the input tensor and existing
/// histogram.
/// \param inputTensor tensor to process.
/// \param existingHistogram histogram of float numbers seen so far.
///                          This histogram serves as an output parameter to the
///                          final histogram after processing inputTensor.
/// \param min min value seen so far, at the end of this method it could be
///            updated.
/// \param max max value seen so far, at the end of this method it
///            could be updated.
void generateTensorHistogram(const Handle<float> inputTensor,
                             Handle<float> existingHistogram, float &min,
                             float &max);

/// Function to rescale the input histogram \p srcHist initially computed in the
/// range \p srcHistMin and \p srcHistMax to a new range given by \p destHistMin
/// and \p destHistMax. The rescaled histogram has the same number of bins as
/// the input histogram. If the rescale range includes the initial range then
/// the rescaled histogram will preserve the total bin sum. If the rescale range
/// does not include the initial range then the input histogram will be cropped
/// and the total bin sum will not be preserved.
/// \returns the rescaled histogram.
std::vector<float> rescaleHistogram(const std::vector<float> &srcHist,
                                    const float srcHistMin,
                                    const float srcHistMax,
                                    const float destHistMin,
                                    const float destHistMax);

} // namespace quantization
} // namespace glow

#endif // GLOW_QUANTIZATION_BASE_PROFILE_H
