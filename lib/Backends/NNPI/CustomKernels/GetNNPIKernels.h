// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <string>
#include <utility>

namespace glow {

struct GetNNPIKernels {
  /// \returns a path to the file where the custom IA kernel binary resides.
  static std::string getCompiledIAKernelsFilePath();

  /// \returns a path to the file where the custom DSP kernel binary resides.
  static std::string getCompiledDSPKernelsFilePath();

#if FACEBOOK_INTERNAL
  static std::string getCompiledIAKernelsFilePathInternal();
  static std::string getCompiledDSPKernelsFilePathInternal();
#endif
};

} // namespace glow
