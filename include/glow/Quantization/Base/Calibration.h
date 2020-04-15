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

#ifndef GLOW_QUANTIZATION_BASE_CALIBRATION_H
#define GLOW_QUANTIZATION_BASE_CALIBRATION_H

#include "glow/Quantization/Base/Base.h"

namespace glow {
namespace quantization {

/// Function to compute the optimal quantization range (thresholds) for a tensor
/// based on minimizing the Kullback-Leibler divergence. The algorithm requires
/// the tensor histogram \p hist computed within the range given by \p histMin
/// and \p histMax. The algorithm is searching for the optimal range by
/// increasingly shrinking and saturating the input histogram and computing the
/// divergence relative to a quantized histogram using \p numQuantizedBins. The
/// input histogram length must be larger than or equal to \p numQuantizedBins
/// in order for any optimization to take place. If not, the input \p histMin
/// and \p histMax are returned. In order to accommodate both symmetric and
/// asymmetric schema, you can choose how the histogram is shrunk by using
/// the flag \p symmetric which if set to TRUE then the histogram is shrunk
/// symmetrically such that the output optimal range is also symmetric. If the
/// \p symmetric flag is FALSE then the shrinking will be done asymmetrically
/// depending on which shrinkage consumes least histogram data. The histogram
/// \p hist is not required to be normalized.
/// \returns the optimized min/max thresholds.
/// To be noted that the complexity of this procedure is roughly O(N^2) where
/// N is the length of the input histogram \p hist.
FloatRange optimizeKL(const std::vector<float> &hist, const float histMin,
                      const float histMax, const size_t numQuantizedBins = 255,
                      const bool symmetric = false);

} // namespace quantization
} // namespace glow

#endif // GLOW_QUANTIZATION_BASE_CALIBRATION_H
