#ifndef GLOW_NETWORK_IMAGE_H
#define GLOW_NETWORK_IMAGE_H

#include "glow/Network/Tensor.h"

namespace glow {

/// Reads a png image. \returns True if an error occurred.
bool readPngImage(Tensor *T, const char *filename);

/// Writes a png image. \returns True if an error occurred.
bool writePngImage(Tensor *T, const char *filename);

} // namespace glow

#endif // GLOW_NETWORK_IMAGE_H
