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
#ifndef GLOW_BACKENDS_COMPILEDFUNCTION_H
#define GLOW_BACKENDS_COMPILEDFUNCTION_H

#include "llvm/ADT/ArrayRef.h"

namespace glow {

class Placeholder;
class Tensor;

/// Interface for executing a compiled function.
class CompiledFunction {
public:
  /// Dtor.
  virtual ~CompiledFunction() = default;

  /// Execute the network. The backing tensors in \p tensors are mapped to the
  /// placeholders of \p placeholders for the invocation of the program.
  virtual void execute(llvm::ArrayRef<Placeholder *> placeholders,
                       llvm::ArrayRef<Tensor *> tensors) = 0;
};

} // end namespace glow

#endif // GLOW_BACKENDS_COMPILEDFUNCTION_H
