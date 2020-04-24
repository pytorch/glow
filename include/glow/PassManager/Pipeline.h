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

#include <iterator>

namespace glow {

/// Base class for all pass pipelines providing some common functionality.
class PassPipelineBase {
protected:
  /// \returns pass config at index \p i.
  virtual const PassConfigBase &elementAt(size_t i) const = 0;

public:
  /// Constructor.
  PassPipelineBase() = default;

  /// Destructor.
  virtual ~PassPipelineBase() = default;

  /// Dump a textual representation of the pipeline to \p os.
  virtual void dump(llvm::raw_ostream &os = llvm::outs()) const;

  /// \returns size of pipeline.
  virtual size_t size() const = 0;
};

/// Implementation of a pipeline for executing a series of passes. Each pass
/// should be of type \p PASS or a type derived from it.
template <typename PASS>
class PassPipeline
    : public PassPipelineBase,
      private llvm::SmallVector<typename PASS::IRPassConfigTy, 16> {
public:
  using IRPassConfigTy = typename PASS::IRPassConfigTy;
  using PassIDTy = typename IRPassConfigTy::PassIDTy;
  using Base = llvm::SmallVector<IRPassConfigTy, 16>;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;

private:
  /// Removes the first instance of a pass with ID \p passID. \returns whether
  /// an instance of the pass was successfully found and removed.
  bool removeFirstInstanceOfPass(PassIDTy passID) {
    for (auto it = begin(); it != end(); it++) {
      if (static_cast<PassIDTy>(it->getID()) == passID) {
        this->erase(it);
        return true;
      }
    }
    return false;
  }

  const PassConfigBase &elementAt(size_t i) const override { return at(i); }

public:
  /// Constructor.
  PassPipeline() = default;

  /// Constructor for a PassPipeline from an initializer_list \p configs.
  PassPipeline(std::initializer_list<IRPassConfigTy> configs) {
    pushBack(configs);
  }

  /// \returns size of pipeline.
  size_t size() const override { return Base::size(); }

  /// Forward iterator creation methods.
  ///@{
  iterator begin() { return Base::begin(); }
  const_iterator begin() const { return Base::begin(); }
  iterator end() { return begin() + size(); }
  const_iterator end() const { return begin() + size(); }
  /// @}

  /// Helper to get the IRPassConfig at index \p i in the pipeline.
  const IRPassConfigTy &at(size_t i) const {
    const PassConfigBase &config = begin()[i];
    return *static_cast<const IRPassConfigTy *>(&config);
  }

  /// Push a new \p IRPC to the end of the pipeline.
  void pushBack(const IRPassConfigTy &IRPC) { Base::push_back(IRPC); }

  /// Push \p configs to the end of the pipeline.
  void pushBack(const std::initializer_list<IRPassConfigTy> &configs) {
    for (auto &config : configs) {
      pushBack(config);
    }
  }

  /// Push \p configs to the end of the pipeline.
  void pushBack(llvm::ArrayRef<IRPassConfigTy> configs) {
    for (auto &config : configs) {
      pushBack(config);
    }
  }

  /// Push a new \p IRPC to the start of the pipeline.
  void pushFront(const IRPassConfigTy &IRPC) { Base::insert(begin(), IRPC); }

  /// Removes all instances of a pass with ID \p passID.
  void removeAllInstancesOfPass(PassIDTy passID) {
    while (removeFirstInstanceOfPass(passID)) {
    }
  }

  /// Initialize the pipeline from a file with a name \p pipelineDefFilename.
  virtual void initFromFile(llvm::StringRef pipelineDefFilename);

  /// Dump pipeline definition into a file with a name \p pipelineDefFilename.
  virtual void dumpToFile(llvm::StringRef pipelineDefFilename);

  bool equals(const PassPipeline &other) const {
    if (size() != other.size()) {
      return false;
    }
    for (unsigned idx = 0, e = size(); idx < e; ++idx) {
      if (!at(idx).equals(other.at(idx))) {
        return false;
      }
    }
    return true;
  }
};

} // namespace glow

#endif // GLOW_PASSMANAGER_PIPELINE_H
