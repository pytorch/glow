// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Quantization/Quantization.h"

#include <cmath>
#include <vector>

namespace glow {

/// Calculate TensorQuantizationParams based on the clipped min and max float
/// values.
static TensorQuantizationParams chooseQuantizationParams(float min, float max) {
  assert(min <= max && "min must not be bigger than max");

  // Given 8 bit precision.
  const int32_t qmin = -128;
  const int32_t qmax = 127;

  double scale =
      (std::max(max, 0.f) - std::min(min, 0.f)) / ((double)qmax - qmin);

  // Dequantization uses the following formula scale * (X - offset), so
  // scale should not be equal to zero.
  // If scale is 0, we arbitrary adjust the scale to 0.1.
  if (scale == 0)
    scale = 0.1;

  assert(scale > 0 && "Scale must be non negative");

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zeroPointFromMin = qmin - min / scale;
  double zeroPointFromMax = qmax - max / scale;
  double zeroPointFromMinError = std::abs(qmin) + std::abs(min / scale);
  double zeroPointFromMaxError = std::abs(qmax) + std::abs(max / scale);
  double initialZeroPoint = zeroPointFromMinError < zeroPointFromMaxError
                                ? zeroPointFromMin
                                : zeroPointFromMax;

  // For symmetric quantization (min == -max), we force zero_point to 128
  // to model signed integer (FIXME: this is a workaround that gemmlowp
  // doesn't support signed int AFAIK. Once we have an (efficient) gemm for
  // signed as well, we can just use signed int with zero_point = 0
  if (min == -max) {
    initialZeroPoint = (qmin + qmax) / 2 + 1;
  }

  // Now we need to nudge the zero point to be an integer (our zero points are
  // integer, and this is motivated by the requirement to be able to represent
  // the real value "0" exactly as a quantized value, which is required in
  // multiple places, for example in Im2col with SAME padding).
  int32_t nudgedZeroPoint = 0;
  if (initialZeroPoint < qmin) {
    nudgedZeroPoint = qmin;
  } else if (initialZeroPoint > qmax) {
    nudgedZeroPoint = qmax;
  } else {
    nudgedZeroPoint = static_cast<int32_t>(round(initialZeroPoint));
  }

  TensorQuantizationParams result{static_cast<float>(scale),
                                  static_cast<float>(nudgedZeroPoint)};
  return result;
}

/// \returns the norm.
static double GetNorm(double begin, double end, double density) {
  double norm = (end * end * end - begin * begin * begin) / 3;
  return norm * density;
}

/// Calculate TensorQuantizationParams based on the histogram, min and max.
static TensorQuantizationParams
calculateTensorQuantizationParams(const Handle<float> &bins, float min,
                                  float max) {
  const int precision = 8;
  int nBins = bins.size();
  double binWidth = (max - min) / nBins;
  int zeroBin = round(-min / binWidth);
  int dstNbins = 1 << precision;

  std::vector<std::pair<int, double>> bestStartBins(nBins + 1);

  // Look at mapping [startBin, startBin + nbinsSelected) to
  // [0, 1 << precision) for every (startBin, nbinsSelected) combination and
  // pick the one with smallest L2 quantization error
  for (int nbinsSelected = 1; nbinsSelected <= nBins; ++nbinsSelected) {
    double normMin = std::numeric_limits<double>::max();
    int bestStartBin = 0;

    int startBinBegin = 0, startBinEnd = nBins - nbinsSelected + 1;
    if (min == 0) {
      startBinBegin = 0;
      startBinEnd = 1;
    } else {
      startBinBegin = zeroBin - nbinsSelected / 2;
      startBinEnd = startBinBegin + 1;
    }
    double dstBinWidth = binWidth * nbinsSelected / dstNbins;

    int startBin;
    for (startBin = startBinBegin; startBin < startBinEnd; ++startBin) {
      double norm = 0;

      // Go over each histogram bin and accumulate errors
      for (int srcBin = 0; srcBin < nBins; ++srcBin) {
        // Distances from the beginning of first dstBin to the beginning and
        // end of srcBin
        double srcBinBegin = (srcBin - startBin) * binWidth;
        double srcBinEnd = srcBinBegin + binWidth;

        // Which dstBins the beginning and end of srcBin belong to?
        int dstBinOfBegin =
            std::min((1 << precision) - 1.,
                     std::max(0., floor(srcBinBegin / dstBinWidth)));
        int dstBinOfEnd =
            std::min((1 << precision) - 1.,
                     std::max(0., floor(srcBinEnd / dstBinWidth)));

        double dstBinOfBeginCenter =
            dstBinOfBegin * dstBinWidth + dstBinWidth / 2;
        double density = bins.raw(srcBin) / binWidth;

        if (dstBinOfBegin == dstBinOfEnd) {
          // if srcBin is entirely within 1 dstBin
          double deltaBegin = srcBinBegin - dstBinOfBeginCenter;
          double deltaEnd = srcBinEnd - dstBinOfBeginCenter;
          norm += GetNorm(deltaBegin, deltaEnd, density);
        } else {
          double deltaBegin = srcBinBegin - dstBinOfBeginCenter;
          double deltaEnd = dstBinWidth / 2;
          norm += GetNorm(deltaBegin, deltaEnd, density);

          norm += (dstBinOfEnd - dstBinOfBegin - 1) *
                  GetNorm(-dstBinWidth / 2, dstBinWidth / 2, density);

          double dst_bin_of_end_center =
              dstBinOfEnd * dstBinWidth + dstBinWidth / 2;
          deltaBegin = -dstBinWidth / 2;
          deltaEnd = srcBinEnd - dst_bin_of_end_center;
          norm += GetNorm(deltaBegin, deltaEnd, density);
        }
      }

      if (norm < normMin) {
        normMin = norm;
        bestStartBin = startBin;
      }
    } // for each startBin

    bestStartBins[nbinsSelected] = {bestStartBin, normMin};
  } // for each nbinsSelected

  double normMin = std::numeric_limits<double>::max();
  int bestNbinsSelected = 1, bestStartBin = 0;
  for (int nbinsSelected = 1; nbinsSelected <= nBins; ++nbinsSelected) {
    double norm = bestStartBins[nbinsSelected].second;
    if (norm < normMin) {
      normMin = norm;
      bestStartBin = bestStartBins[nbinsSelected].first;
      bestNbinsSelected = nbinsSelected;
    }
  }

  double totalSum = 0;
  for (int i = 0; i < bins.size(); ++i) {
    totalSum += bins.raw(i);
  }
  double selectedSum = 0;
  int iBegin = std::max(0, bestStartBin);
  int iEnd = std::min(nBins, bestStartBin + bestNbinsSelected);
  for (int i = iBegin; i < iEnd; ++i) {
    selectedSum += bins.raw(i);
  }

  float newMin = min + binWidth * bestStartBin;
  float newMax = min + binWidth * (bestStartBin + bestNbinsSelected);

  return chooseQuantizationParams(newMin, newMax);
}

std::vector<NodeQuantizationInfo>
generateNodeQuantizationInfos(const Graph &G) {
  std::vector<NodeQuantizationInfo> quantizationInfos;

  for (auto *node : G.getNodes()) {
    auto *QPN = llvm::dyn_cast<QuantizationProfileNode>(node);

    if (QPN) {
      auto CI = QPN->getComputationInfoVar()->getHandle<float>();
      auto histogram = QPN->getHistogramVar()->getHandle<float>();
      float min = CI.raw(0);
      float max = CI.raw(1);

      NodeValue &observedNodeValue = node->getNthInput(0);
      unsigned resNum = observedNodeValue.getResNo();
      Node *observedNode = observedNodeValue.getNode();

      std::string nodeName =
          observedNode->getName().str() + ":" + std::to_string(resNum);

      TensorQuantizationParams TQP =
          calculateTensorQuantizationParams(histogram, min, max);
      quantizationInfos.emplace_back(nodeName, TQP);
    }
  }

  return quantizationInfos;
}

} // namespace glow
