// Copyright 2017 Facebook Inc.  All Rights Reserved.

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
