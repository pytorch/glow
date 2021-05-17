// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once
#include <memory>
#include <vector>

#include "../CustomKernelInjector.h"

namespace glow {

/// \returns the list of all CustomKernelInjectors for DSP nodes.
std::vector<std::unique_ptr<CustomKernelInjector>> buildIAInjectors();

} // namespace glow
