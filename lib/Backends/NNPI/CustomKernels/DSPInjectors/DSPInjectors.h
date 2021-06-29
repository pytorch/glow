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

#pragma once
#include <memory>
#include <vector>

#include "../CustomKernelInjector.h"

namespace glow {

/// Injector for custom Relu node.
struct CustomReluNodeDSPKernelInjector : public CustomKernelInjector {
  Node *tryInject(Function *F, Node *node) const override;
};

/// Injector for custom CmpNEQ node.
struct CustomCmpNEQNodeDSPKernelInjector : public CustomKernelInjector {
  Node *tryInject(Function *F, Node *node) const override;
};

/// Injector for custom CmpEQ node.
struct CustomCmpEQNodeDSPKernelInjector : public CustomKernelInjector {
  Node *tryInject(Function *F, Node *node) const override;
};

/// Injector for custom CmpLT node.
struct CustomCmpLTNodeDSPKernelInjector : public CustomKernelInjector {
  Node *tryInject(Function *F, Node *node) const override;
};

/// Injector for custom CmpLTE node.
struct CustomCmpLTENodeDSPKernelInjector : public CustomKernelInjector {
  Node *tryInject(Function *F, Node *node) const override;
};

/// \returns the list of all CustomKernelInjectors for DSP nodes.
std::vector<std::unique_ptr<CustomKernelInjector>> buildDSPInjectors();

} // namespace glow
