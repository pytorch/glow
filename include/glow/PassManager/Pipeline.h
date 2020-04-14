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
#ifndef GLOW_PASSMANAGER_PIPELINE_H
#define GLOW_PASSMANAGER_PIPELINE_H

#include "PassConfig.h"
#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Support/Support.h"

#include <bitset>
#include <iterator>

namespace glow {

/// Implementation of a pipeline for executing a series of passes. Each pass
/// should be of type \p PASS or a type derived from it.
template <typename PASS>
class PassPipeline
    : private llvm::SmallVector<typename PASS::IRPassConfigTy, 64> {
public:
  using IRPassConfigTy = typename PASS::IRPassConfigTy;
  using PassIDTy = typename IRPassConfigTy::PassIDTy;

private:
  using ParentImpl = llvm::SmallVectorImpl<IRPassConfigTy>;

  /// Removes the first instance of a pass with ID \p passID. \returns whether
  /// an instance of the pass was successfully found and removed.
  bool removeFirstInstanceOfPass(PassIDTy passID) {
    for (auto it = begin(); it != end(); it++) {
      if (it->getPassID() == passID) {
        this->erase(it);
        return true;
      }
    }
    return false;
  }

public:
  using Base = llvm::SmallVector<IRPassConfigTy, 64>;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  PassPipeline() = default;

  /// Constructs a FunctionPassPipeline from an initializer_list \p IL.
  PassPipeline(std::initializer_list<IRPassConfigTy> IL) { this->assign(IL); }

  /// Forward iterator creation methods.
  ///@{
  iterator begin() { return ParentImpl::begin(); }
  const_iterator begin() const { return ParentImpl::begin(); }
  iterator end() { return begin() + size(); }
  const_iterator end() const { return begin() + size(); }
  /// @}

  /// Forward to parent size() method. \returns size of pipeline.
  size_t size() const { return ParentImpl::size(); }

  /// Helper to get the IRPassConfig at index \p i in the pipeline.
  const IRPassConfigTy &at(size_t i) const { return begin()[i]; }

  /// Push a new \p IRPC to the end of the pipeline.
  void pushBack(IRPassConfigTy IRPC) { ParentImpl::push_back(IRPC); }

  /// Push a new \p IRPC to the start of the pipeline.
  void pushFront(IRPassConfigTy IRPC) { ParentImpl::insert(begin(), IRPC); }

  /// Removes all instances of a pass with ID \p passID.
  void removeAllInstancesOfPass(PassIDTy passID) {
    while (removeFirstInstanceOfPass(passID)) {
    }
  }

  /// Dump a textual representation of the pipeline to \p os.
  void dump(llvm::raw_ostream &os = llvm::outs()) const {
    os << "Pipeline contains:\n";
    for (size_t i = 0, e = this->size(); i < e; i++) {
      const auto &passConfig = (*this)[i];
      os << "FunctionPassIdx " << i << ": {\n";
      passConfig.dump(os);
      os << "}\n";
    }
  }
};

} // namespace glow

#endif // GLOW_PASSMANAGER_PIPELINE_H
