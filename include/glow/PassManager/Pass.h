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

class PassBase : public Named {
public:
  PassBase(llvm::StringRef name) : Named(name) {}

  virtual ~PassBase() = default;
};

/// Class used for all passes over IR containers represented by the type \p
/// IRCONTAINER, which can be e.g. Function or IRFunction. All passes over units
/// should derive from this class, implementing the pass logic and additionally
/// can add logic for running before and after the pass runs. The pass configs
/// are represented by the type \p IRPASSCONFIG.
template <typename IRCONTAINER, typename IRPASSCONFIG>
class Pass : public PassBase {
public:
  using IRContainerTy = IRCONTAINER;
  using IRPassConfigTy = IRPASSCONFIG;
  using PassIDTy = typename IRPassConfigTy::PassIDTy;

public:
  /// Constructor.
  Pass(llvm::StringRef name) : PassBase(name) {}

  virtual ~Pass() = default;

  /// Run the pass on \p C. \returns whether the pass modifies \p C.
  virtual bool run(IRContainerTy *C, const CompilationContext &cctx) = 0;

  /// \returns the id of the pass.
  virtual PassIDTy getID() const = 0;
};

} // namespace glow

#endif // GLOW_PASSMANAGER_PASS_H
