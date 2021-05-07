// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once
#include <memory>
#include <vector>

#include "../CustomKernelInjector.h"

namespace glow {

/// Injector for custom Relu node.
struct CustomReluNodeDSPKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override;
};

/// Injector for custom CmpNEQ node.
struct CustomCmpNEQNodeDSPKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override;
};

/// Injector for custom CmpEQ node.
struct CustomCmpEQNodeDSPKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override;
};

/// Injector for custom CmpLT node.
struct CustomCmpLTNodeDSPKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override;
};

/// Injector for custom CmpLTE node.
struct CustomCmpLTENodeDSPKernelInjector : public CustomKernelInjector {
  bool tryInject(Function *F, Node *node) const override;
};

/// \returns the list of all CustomKernelInjectors for DSP nodes.
std::vector<std::unique_ptr<CustomKernelInjector>> buildDSPInjectors();

} // namespace glow
