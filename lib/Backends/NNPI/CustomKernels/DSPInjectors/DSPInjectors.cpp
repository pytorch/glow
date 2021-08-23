/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DSPInjectors.h"
#include "glow/Graph/Graph.h"

namespace glow {
std::vector<std::unique_ptr<CustomKernelInjector>> buildDSPInjectors() {
  std::vector<std::unique_ptr<CustomKernelInjector>> injectors;
  // Custom relu, used as a sample, disabled by default.
  // injectors.emplace_back(std::make_unique<CustomReluNodeDSPKernelInjector>());

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION < 8
  injectors.emplace_back(std::make_unique<CustomCmpNEQNodeDSPKernelInjector>());
  injectors.emplace_back(std::make_unique<CustomCmpEQNodeDSPKernelInjector>());
  injectors.emplace_back(std::make_unique<CustomCmpLTNodeDSPKernelInjector>());
  injectors.emplace_back(std::make_unique<CustomCmpLTENodeDSPKernelInjector>());
#endif // NNPI < 1.8

  return injectors;
}
} // namespace glow
