/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_TRAININGPREPARATION_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_TRAININGPREPARATION_H

#include "glow/Graph/Graph.h"
#include "glow/Support/Error.h"

#include <functional>

namespace glow {

/// Forward declaration.
class Tensor;

/// Function for \p tensor initialization.
/// Method gets following parameters Glow Function \p F, current \p node,
/// \p inputIdx, and fills out \p tensor.
using TensorInitializer = std::function<void(
    Function *F, Node *node, unsigned inputIdx, Tensor *tensor)>;

/// Method generates and returns the default tensor initializer.
TensorInitializer getDefaultTensorInitializer();

/// Function takes glow::Function \p F, \p bindings, \p selected placeholder,
// and \p initializer for the input weights.
llvm::Error prepareFunctionForTraining(
    Function *F, PlaceholderBindings &bindings, Placeholder *&selected,
    TensorInitializer &&initializer = getDefaultTensorInitializer());
} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_TRAININGPREPARATION_H
