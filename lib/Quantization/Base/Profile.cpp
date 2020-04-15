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

#include "glow/Quantization/Base/Profile.h"

#include <cmath>

namespace glow {
namespace quantization {

/// Gen a bin number to insert \p value into the histogram which has \p nBins
/// with \p minValue and binWidth in histogram.
static size_t getBin(size_t nBins, float binWidth, float minValue,
                     float value) {
  size_t result =
      binWidth == 0
          ? 0
          : std::min(static_cast<size_t>((value - minValue) / binWidth),
                     nBins - 1);
  return result;
}

void generateTensorHistogram(const Handle<float> inputTensor,
                             Handle<float> existingHistogram, float &min,
                             float &max) {
  auto minMaxPos = inputTensor.minMaxArg();
  float minInput = inputTensor.raw(minMaxPos.first);
  float maxInput = inputTensor.raw(minMaxPos.second);

  if (existingHistogram.isZero()) {
    min = minInput;
    max = maxInput;
  }

  size_t nBins = existingHistogram.size();

  // Check if we need to rescale histogram.
  if (minInput < min || maxInput > max) {
    float newMin = std::min(minInput, min);
    float newMax = std::max(maxInput, max);

    float destBinWidth = (newMax - newMin) / nBins;
    float srcBinWidth = (max - min) / nBins;

    std::vector<float> scaledHistogram(nBins, 0);

    for (size_t i = 0; i < nBins; ++i) {
      if (existingHistogram.raw(i) == 0)
        continue;

      float srcBinBegin = min + srcBinWidth * i;
      size_t destBin = (srcBinBegin - newMin) / destBinWidth;
      float destBinEnd = newMin + destBinWidth * (destBin + 1);

      float srcBinEnd = srcBinBegin + srcBinWidth;
      size_t destBinToVerify = (srcBinEnd - newMin) / destBinWidth;
      // Make sure that destination bin is mapped at most to 2 final bins, based
      // on that redistribute percentage is calculated.
      assert(destBinToVerify <= destBin + 2);
      (void)destBinToVerify;

      // Calculate how much we need to redistribute.
      uint64_t dstBinCnt = static_cast<uint64_t>(std::min(
          static_cast<float>(round((destBinEnd - srcBinBegin) / srcBinWidth *
                                   existingHistogram.raw(i))),
          existingHistogram.raw(i)));

      size_t newBin = getBin(nBins, destBinWidth, newMin, srcBinBegin);
      scaledHistogram[newBin] += dstBinCnt;

      if (dstBinCnt < existingHistogram.raw(i)) {
        size_t newBin =
            getBin(nBins, destBinWidth, newMin, srcBinBegin + destBinWidth);
        scaledHistogram[newBin] += existingHistogram.raw(i) - dstBinCnt;
      }
    }

    // Copy scaled histogram back to the existing histogram.
    for (size_t i = 0, e = scaledHistogram.size(); i < e; ++i) {
      existingHistogram.raw(i) = scaledHistogram[i];
    }

    // Update global min and max.
    min = newMin;
    max = newMax;
  }

  float binWidth = (max - min) / nBins;
  for (auto elem : inputTensor) {
    size_t newBin = getBin(nBins, binWidth, min, elem);
    existingHistogram.raw(newBin)++;
  }
}

std::vector<float> rescaleHistogram(const std::vector<float> &srcHist,
                                    const float srcHistMin,
                                    const float srcHistMax,
                                    const float destHistMin,
                                    const float destHistMax) {

  // If histogram is empty then return.
  if (srcHist.size() == 0) {
    return srcHist;
  }

  // Check if we need to rescale the histogram.
  assert(srcHistMin < srcHistMax && "Invalid source histogram min/max range!");
  assert(destHistMin < destHistMax &&
         "Invalid destination histogram min/max range!");
  if ((srcHistMin == destHistMin) && (srcHistMax == destHistMax)) {
    return srcHist;
  }

  // Number of histogram bins and bin widths.
  const size_t numBins = srcHist.size();
  const float srcBinWidth = (srcHistMax - srcHistMin) / numBins;
  const float destBinWidth = (destHistMax - destHistMin) / numBins;

  // Iterate the source bins and distribute into the destination bins.
  std::vector<float> destHist(numBins, 0);
  for (size_t srcBinIdx = 0; srcBinIdx < numBins; srcBinIdx++) {

    // Get current source bin value.
    float srcBinVal = srcHist[srcBinIdx];
    if (srcBinVal == 0) {
      continue;
    }

    // Get source bin start/stop values for this bin.
    float srcBinStart = srcHistMin + srcBinIdx * srcBinWidth;
    float srcBinStop = srcHistMin + (srcBinIdx + 1) * srcBinWidth;

    // Get destination bin indices (inclusive) which overlap with the current
    // source bin.
    float dstBinIdxStartF =
        std::floor((srcBinStart - destHistMin) / destBinWidth);
    float dstBinIdxStopF = std::ceil((srcBinStop - destHistMin) / destBinWidth);
    size_t dstBinIdxStart = static_cast<size_t>(std::max(dstBinIdxStartF, 0.f));
    size_t dstBinIdxStop = static_cast<size_t>(std::max(dstBinIdxStopF, 0.f));

    // Upper saturate the destination bin indices.
    if (dstBinIdxStart >= numBins) {
      dstBinIdxStart = numBins - 1;
    }
    if (dstBinIdxStop >= numBins) {
      dstBinIdxStop = numBins - 1;
    }

    // Redistribute the source bin into all the destination bins.
    // Only integer values will be distributed.
    float srcBinRem = srcBinVal;
    for (size_t destBinIdx = dstBinIdxStart; destBinIdx <= dstBinIdxStop;
         destBinIdx++) {

      // Get destination bin start/stop values for this bin.
      float destBinStart = destHistMin + destBinIdx * destBinWidth;
      float destBinStop = destHistMin + (destBinIdx + 1) * destBinWidth;

      // Get source/destination overlap boundaries and ratio.
      float overlapStart = std::max(srcBinStart, destBinStart);
      float overlapStop = std::min(srcBinStop, destBinStop);
      float overlapRatio = (overlapStop - overlapStart) / srcBinWidth;
      overlapRatio = overlapRatio >= 0.0f ? overlapRatio : 0.0f;
      overlapRatio = overlapRatio <= 1.0f ? overlapRatio : 1.0f;

      // Compute distribution value.
      float distVal = std::round(overlapRatio * srcBinVal);
      distVal = distVal <= srcBinRem ? distVal : srcBinRem;

      // Distribute value.
      destHist[destBinIdx] += distVal;
      srcBinRem -= distVal;
    }
  }

  return destHist;
}

} // namespace quantization
} // namespace glow
