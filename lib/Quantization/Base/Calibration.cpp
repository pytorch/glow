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

#include "glow/Quantization/Base/Calibration.h"

#include <cmath>
#include <numeric>

namespace glow {
namespace quantization {

/// Function to prepare the histogram \p hist with length \p length before
/// computing the relative entropy. The histogram is NOT normalized and is
/// assumed to be a sequence of positive integers (0,1,...) stored as a float
/// sequence to accommodate very large values. The histogram is conditioned
/// in the following way: zero values are replaced with \p epsZero while the
/// corresponding amount is subtracted from the non-zero values.
static void conditionHistogram(float *hist, const size_t length,
                               const float epsZero = 0.0001) {

  // If histogram is empty then return.
  if (length == 0) {
    return;
  }

  // Get information about the zero values within the histogram.
  int isZero[length];
  size_t numZeros = 0;
  for (size_t idx = 0, e = length; idx < e; idx++) {
    isZero[idx] = static_cast<int>(hist[idx] == 0.f);
    numZeros += isZero[idx];
  }

  // If histogram is all zeros then return.
  if (numZeros == length) {
    return;
  }

  // Compute epsilon to subtract from non-zero histogram values.
  size_t numNonZeros = length - numZeros;
  float epsNonZero =
      epsZero * static_cast<float>(numZeros) / static_cast<float>(numNonZeros);

  // If value to subtract from non-zero values is higher than 1.0 then return.
  if (epsNonZero >= 1.0) {
    return;
  }

  // Perform histogram conditioning:
  // - zero histogram values are increased with epsZero.
  // - non-zero histogram values are decreased with epsNonZero.
  for (size_t idx = 0, e = length; idx < e; idx++) {
    hist[idx] += epsZero * isZero[idx];
    hist[idx] -= epsNonZero * (1 - isZero[idx]);
  }
}

/// Function to compute the Kullback-Leibler divergence (relative entropy)
/// of the distribution \p P with respect to \p Q denoted with D(P||Q) and
/// defined as:
///               D(P||Q) = sum(P[k] * log(P[k] / Q[k]))
/// Depending on the base of the logarithm, the unit of measurement for the
/// divergence is "bits" for log2 and "nats" for ln (natural logarithm).
/// This function does NOT require the distributions \p P and \p Q to be
/// normalized because are normalized automatically in-place. The length
/// of the distributions is \p length. \returns the relative entropy metric
/// as a float scalar.
/// The meaning of this metric is the amount of information (entropy) lost
/// when the distribution \p Q is used to approximate the ground truth
/// (reference) distribution \p P. The divergence metric is always positive
/// that is D(P||Q) >= 0.
static float computeKL(float *P, float *Q, size_t length) {

  // Compute sum of P and Q to use for normalization.
  float sumP = std::accumulate(P, P + length, 0.f);
  float sumQ = std::accumulate(Q, Q + length, 0.f);

  // Return 0 when one of the distributions is all zero.
  if ((sumP == 0.f) || (sumQ == 0.f)) {
    return 0;
  }

  // Compute relative entropy.
  float divergence = 0;
  for (size_t idx = 0, e = length; idx < e; idx++) {
    P[idx] /= sumP;
    Q[idx] /= sumQ;
    if ((P[idx] > 0) && (Q[idx] > 0)) {
      divergence += P[idx] * std::log(P[idx] / Q[idx]);
    }
  }
  return divergence;
}

FloatRange optimizeKL(const std::vector<float> &hist, const float histMin,
                      const float histMax, const size_t numQuantizedBins,
                      const bool symmetric) {

  // Number of histogram bins.
  const size_t numBins = hist.size();

  // If the input histogram is empty or the number of histogram bins is smaller
  // than numQuantizedBins then return the histogram range.
  if ((numBins == 0) || (numBins < numQuantizedBins)) {
    return {histMin, histMax};
  }

  // Histogram bin width.
  assert(histMin < histMax && "Invalid histogram min/max range!");
  const float histBinWidth = (histMax - histMin) / (float)numBins;

  // Optimal divergence value (minimum).
  float divergenceOpt = std::numeric_limits<float>::infinity();

  // Optimal threshold values for minimum divergence.
  float thresholdMinOpt = histMin;
  float thresholdMaxOpt = histMax;

  // Initialize start/stop bin indices (inclusive) with the first and last bin.
  size_t histWinIdxStart = 0;
  size_t histWinIdxStop = numBins - 1;

  // Start iterations by increasingly saturating the input histogram while
  // the histogram window is larger or equal to numQuantizedBins. The expected
  // behavior of the computed divergence is either:
  // (1) increase monotonically in which case it would make sense to include
  //     some logic to exit the loop prematurely in order to not waste time.
  // (2) either slightly decrease in the first iterations, settle in a local
  //     minimum and then increase monotonically. This is the case in which
  //     this algorithm hopes to achieve better ranges for quantizing a tensor.
  while ((histWinIdxStop - histWinIdxStart + 1) >= numQuantizedBins) {

    // Current histogram window size.
    const size_t histWinSize = histWinIdxStop - histWinIdxStart + 1;

    // Current histogram window raw pointer.
    const float *histWinPtr = hist.data() + histWinIdxStart;

    // Note: MXNet / TVM have an error in their programs since they explicitly
    // extract the histogram window in the variable 'sliced_nd_hist' which has
    // always 0 on the first position that is sliced_nd_hist.front() == 0.

    // -------------------------------------------------------------------------
    // Compute the reference distribution P as the input histogram saturated in
    // the current window given by histWinIdxStart and histWinIdxStop.
    // -------------------------------------------------------------------------
    std::vector<float> P(histWinSize);

    // Saturate the histogram left.
    float leftSum = 0;
    for (size_t histIdx = 0; histIdx <= histWinIdxStart; histIdx++) {
      leftSum += hist[histIdx];
    }
    P.front() += leftSum;

    // Extract the non-saturated part of the histogram.
    for (size_t histIdx = histWinIdxStart + 1; histIdx < histWinIdxStop;
         histIdx++) {
      P[histIdx - histWinIdxStart] = hist[histIdx];
    }

    // Saturate the histogram right.
    float rightSum = 0;
    for (size_t histIdx = histWinIdxStop; histIdx < numBins; histIdx++) {
      rightSum += hist[histIdx];
    }
    P.back() += rightSum;

    // -------------------------------------------------------------------------
    // Compute the approximation distribution Q as the input histogram sliced in
    // the current window given by histWinIdxStart and histWinIdxStop, rescaled
    // to numQuantizedBins and then expanded back to the current window length.
    // -------------------------------------------------------------------------
    // The bins from the current histogram window are distributed equally in the
    // quantized bins. The remainder is distributed in the last quantized bin.
    assert(histWinSize >= numQuantizedBins && "Invalid histogram window size!");
    const size_t numMergedBins = histWinSize / numQuantizedBins;

    // Compute Q.
    std::vector<float> Q(histWinSize, 0);
    for (size_t qIdx = 0; qIdx < numQuantizedBins; qIdx++) {

      // Histogram window bin start index (inclusive) for this quantized bin.
      const size_t idxStart = qIdx * numMergedBins;

      // Histogram window bin stop index (exclusive) for this quantized bin.
      // If last quantized bin then go to the end of the window.
      const size_t idxStop = (qIdx < (numQuantizedBins - 1))
                                 ? (idxStart + numMergedBins)
                                 : histWinSize;

      // Sum all the values for this quantized bin.
      // Count all the non-negative values for this quantized bin to use for
      // normalization.
      float sum = 0;
      size_t norm = 0;
      for (size_t idx = idxStart; idx < idxStop; idx++) {
        sum += histWinPtr[idx];
        norm += (histWinPtr[idx] != 0);
      }

      // Compute Q by expanding and normalizing the quantized bins.
      if (norm != 0) {
        for (size_t idx = idxStart; idx < idxStop; idx++) {
          if (P[idx]) {
            Q[idx] = sum / (float)norm;
          }
        }
      }
    }

    // -------------------------------------------------------------------------
    // Compute the KL divergence metric and check for optimal values.
    // -------------------------------------------------------------------------
    // Condition the histograms P and Q.
    conditionHistogram(P.data(), P.size());
    conditionHistogram(Q.data(), Q.size());

    // Compute the divergence of P with respect to Q.
    float divergence = computeKL(P.data(), Q.data(), P.size());

    // Check if current divergence is the new optimal.
    if (divergence < divergenceOpt) {

      // Update optimal divergence with current divergence.
      divergenceOpt = divergence;

      // Update optimal thresholds with current thresholds.
      thresholdMinOpt = histMin + histWinIdxStart * histBinWidth;
      thresholdMaxOpt = histMin + (histWinIdxStop + 1) * histBinWidth;
    }

    // -------------------------------------------------------------------------
    // Update histogram window for next iteration.
    // -------------------------------------------------------------------------
    if (symmetric) {
      // For symmetric schema we shrink the histogram window symmetrically.
      histWinIdxStart++;
      histWinIdxStop--;

    } else {
      // For asymmetric schema we shrink the histogram window either left-only,
      // right-only or symmetrically depending on which case has minimum
      // histogram data loss.
      float symmLoss = hist[histWinIdxStart] + hist[histWinIdxStop];
      float leftLoss = hist[histWinIdxStart] + hist[histWinIdxStart + 1];
      float rightLoss = hist[histWinIdxStop] + hist[histWinIdxStop - 1];

      std::vector<float> loss = {symmLoss, leftLoss, rightLoss};
      auto lossMinIdx = std::distance(
          loss.begin(), std::min_element(loss.begin(), loss.end()));
      if (lossMinIdx == 0) {
        // Saturate symmetrically.
        histWinIdxStart++;
        histWinIdxStop--;
      } else if (lossMinIdx == 1) {
        // Saturate left.
        histWinIdxStart += 2;
      } else {
        // Saturate right.
        histWinIdxStop -= 2;
      }
    }
  }

  // For symmetric schema we must make sure the optimized thresholds maintain
  // the same ratio as the input min/max of the histogram in order to map the
  // zero-point to quantized 0.
  if (symmetric) {
    assert(histMin < 0 && "Invalid histogram minimum!");
    assert(histMax > 0 && "Invalid histogram maximum!");
    assert(thresholdMinOpt < 0 && "Invalid threshold minimum!");
    assert(thresholdMaxOpt > 0 && "Invalid threshold maximum!");
    double ratioMin = (double)thresholdMinOpt / (double)histMin;
    double ratioMax = (double)thresholdMaxOpt / (double)histMax;
    if (ratioMin > ratioMax) {
      thresholdMaxOpt = ratioMin * histMax;
    } else {
      thresholdMinOpt = ratioMax * histMin;
    }
  }

  return {thresholdMinOpt, thresholdMaxOpt};
}

} // namespace quantization
} // namespace glow
