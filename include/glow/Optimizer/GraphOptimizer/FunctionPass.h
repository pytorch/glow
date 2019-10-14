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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASS_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASS_H

#include "llvm/ADT/StringRef.h"

namespace glow {

class Function;
struct CompilationContext;
enum class FunctionPassID;

/// Class used for all passes over Functions. All passes over Functions should
/// derive from this class, implementing the pass logic and additionally can add
/// logic for running before and after the pass runs.
class FunctionPass {
  friend class FunctionPassManager;

private:
  /// Run the pass on \p F. \returns whether the pass modifies \p F.
  virtual bool run(Function *F, const CompilationContext &cctx) = 0;

  /// \returns the name of the pass.
  virtual llvm::StringRef getName() const = 0;

  /// \returns the name of the pass.
  virtual FunctionPassID getID() const = 0;

public:
  FunctionPass() = default;
  virtual ~FunctionPass() = default;
};

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_FUNCTIONPASS_H
