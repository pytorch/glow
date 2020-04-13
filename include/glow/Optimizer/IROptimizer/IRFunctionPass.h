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
#ifndef GLOW_OPTIMIZER_IROPTIMIZER_IRFUNCTIONPASS_H
#define GLOW_OPTIMIZER_IROPTIMIZER_IRFUNCTIONPASS_H

#include "glow/PassManager/Pass.h"
#include "glow/PassManager/PassConfig.h"

namespace glow {

class IRFunction;
enum class IRFunctionPassID;

// template <typename X, typename Y> class IRFunctionPass;
using IRFunctionPassConfig = PassConfig<IRFunctionPassID>;

/// Pass over low-level IR-level functions.
using IRFunctionPass = Pass<IRFunction, IRFunctionPassConfig>;

} // namespace glow

#endif // GLOW_OPTIMIZER_IROPTIMIZER_IRFUNCTIONPASS_H
