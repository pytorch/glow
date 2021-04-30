// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once
#include <memory>
#include <vector>

#include "CustomKernelInjector.h"

namespace glow {

/// Injector for custom Relu node.
struct CustomReluNodeDSPKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override;
};

/// \returns the list of all CustomKernelInjectors for DSP nodes.
std::vector<std::unique_ptr<CustomKernelInjector>> buildDSPInjectors();

} // namespace glow
