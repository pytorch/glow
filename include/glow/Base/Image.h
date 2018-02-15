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
