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

#ifndef GLOW_QUANTIZATION_PROFILE_H
#define GLOW_QUANTIZATION_PROFILE_H

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

} // namespace quantization
} // namespace glow

#endif
