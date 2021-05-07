// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "glow/lib/Backends/NNPI/CustomKernels/GetNNPIKernels.h"
#include "glow/Flags/Flags.h"

namespace glow {

std::string GetNNPIKernels::getCompiledIAKernelsFilePath() {
  std::string path = glow::nnpi::flags::InjectedIAOpKernelPath;

  // If the InjectedIAOpKernelPath flag was set then we should use that path.
  if (!path.empty()) {
    return path;
  }

#if FACEBOOK_INTERNAL
  path = GetNNPIKernels::getCompiledIAKernelsFilePathInternal();
#endif

  return path;
}

std::string GetNNPIKernels::getCompiledDSPKernelsFilePath() {
  // We don't need to set a default path value with a gflag because we already
  // have NNPI_CustomDSPLib environment variable for setting the DSP library
  // file manually.
  std::string path;

#if FACEBOOK_INTERNAL
  path = GetNNPIKernels::getCompiledDSPKernelsFilePathInternal();
#endif

  return path;
}

} // namespace glow
