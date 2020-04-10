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
#ifndef GLOW_PASSMANAGER_PASS_H
#define GLOW_PASSMANAGER_PASS_H

#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "llvm/ADT/StringRef.h"

namespace glow {

/// Class used for all passes over functions. All passes over functions should
/// derive from this class, implementing the pass logic and additionally can add
/// logic for running before and after the pass runs.
template <typename UNIT, typename PASSID> class Pass {
  // friend PASSMANAGER;

public:
  using Unit = UNIT;
  using PassID = PASSID;

public:
  /// Run the pass on \p F. \returns whether the pass modifies \p F.
  virtual bool run(Unit *F, const CompilationContext &cctx) = 0;

  /// \returns the name of the pass.
  virtual llvm::StringRef getName() const = 0;

  /// \returns the id of the pass.
  virtual PassID getID() const = 0;

public:
  Pass() = default;
  virtual ~Pass() = default;
};

} // namespace glow

#endif // GLOW_PASSMANAGER_PASS_H
