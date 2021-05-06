// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "glow/lib/Backends/NNPI/CustomKernels/DSPInjectors/DSPInjectors.h"
#include "glow/Graph/Graph.h"

namespace glow {
std::vector<std::unique_ptr<CustomKernelInjector>> buildDSPInjectors() {
  std::vector<std::unique_ptr<CustomKernelInjector>> injectors;
  // Custom relu, used as a sample, disabled by default.
  // injectors.emplace_back(std::make_unique<CustomReluNodeDSPKernelInjector>());
  injectors.emplace_back(std::make_unique<CustomCmpNEQNodeDSPKernelInjector>());
  return injectors;
}
} // namespace glow
