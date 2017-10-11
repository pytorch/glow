#ifndef GLOW_BASE_IMAGE_H
#define GLOW_BASE_IMAGE_H

#include "glow/Base/Tensor.h"

namespace glow {

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
