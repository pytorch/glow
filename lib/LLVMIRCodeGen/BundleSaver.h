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
#ifndef GLOW_LLVMIRCODEGEN_BUNDLESAVER_H
#define GLOW_LLVMIRCODEGEN_BUNDLESAVER_H

#include "glow/LLVMIRCodeGen/AllocationsInfo.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"

#include "glow/IR/IR.h"

namespace glow {
class LLVMBackend;

class BundleSaver {
public:
  /// Information about a saved IR function.
  struct SavedIRFunction {
    /// Entry name for the IR function.
    std::string entryName;
    /// Saved IRFunction.
    const IRFunction *savedF{nullptr};
    /// LLVM IR function created for this IR function.
    llvm::Function *llvmF{nullptr};
  };
  /// WeightInfo represents a constant weight and a constant it is produced
  /// from.
  using WeightInfo = std::pair<const WeightVar *, const Constant *>;
  /// Comparator for WeightInfo objects, sorting them by their allocated
  /// address.
  class WeightAddrComparator {
  public:
    WeightAddrComparator(BundleSaver &bundleSaver)
        : bundleSaver_(&bundleSaver) {}
    bool operator()(const WeightInfo &LHS, const WeightInfo &RHS) const;

  private:
    const BundleSaver *bundleSaver_;
  };
  /// Ctor.
  explicit BundleSaver(const LLVMBackend &llvmBackend,
                       llvm::StringRef outputDir, llvm::StringRef bundleName);
  /// Dtor.
  virtual ~BundleSaver() = default;
  virtual void save(llvm::StringRef mainEntryName, const IRFunction *F);
  /// Produce a bundle.
  virtual void produceBundle();

protected:
  /// Perform memory allocation for a bundle.
  virtual void performBundleMemoryAllocation();
  /// Save weights for the bundle.
  virtual void saveWeights(llvm::StringRef weightsFileName);
  /// Save header file for the bundle.
  virtual void saveHeader(llvm::StringRef headerFileName);
  /// Emit config for a bundle.
  virtual void emitBundleConfig();
  /// Emit the symbol table for a bundle.
  virtual void emitSymbolTable();
  /// Emit the entry function for the saved function \p savedF.
  virtual void emitBundleEntryFunction(SavedIRFunction &savedF);
  /// Set current IRFunction.
  virtual void setIRFunction(llvm::StringRef mainEntryName,
                             const IRFunction *F);
  /// Returns a set of placeholders associated with IR functions inside this
  /// bundle.
  virtual std::set<const Placeholder *> findPlaceholders() const;
  /// Returns a set of constant weights associated with IR functions inside this
  /// bundle.
  virtual std::set<WeightInfo, WeightAddrComparator>
  findConstantWeights() const;
  /// \returns the weight that the variable \p v is lowered into in one of the
  /// IR functions inside this bundle, or null if the variable is unknown.
  virtual Value *getWeightForNode(const Storage *V) const;
  /// Information about allocations.
  AllocationsInfo allocationsInfo_;
  /// The LLVM IR code generator.
  std::unique_ptr<LLVMIRGen> irgen_;
  /// Information about IR functions inside this bundle.
  std::vector<SavedIRFunction> savedIRFunctions_;
  /// Bundle API to use.
  BundleApiType bundleAPI_;
  /// Indicates if this bundle was saved already.
  bool isSaved_{false};
};

} // namespace glow

#endif // GLOW_LLVMIRCODEGEN_BUNDLESAVER_H
