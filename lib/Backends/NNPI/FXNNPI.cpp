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

#include "glow/lib/Backends/NNPI/NNPI.h"
#include "glow/lib/Backends/NNPI/NNPICompiledFunction.h"
#include <folly/dynamic.h>

using namespace glow;

Expected<std::unique_ptr<CompiledFunction>>
NNPIBackend::compileFX(const folly::dynamic &FXIR, const std::string &submod,
                       const llvm::StringMap<const void *> &constants,
                       const BackendOptions &opts, Module *glowModule) const {
  auto compiledFunc = std::make_unique<NNPICompiledFunction>(
      FXIR, submod, constants, glowModule);
  auto compileHasError =
      compiledFunc->compileFX(FXIR, submod, constants, opts, glowModule);

  return compileHasError ? std::move(compileHasError)
                         : Expected<std::unique_ptr<CompiledFunction>>(
                               std::move(compiledFunc));
}
