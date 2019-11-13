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

class BundleSaver final {
public:
  /// Information about a saved IR function.
  struct SavedIRFunction {
    /// Entry name for the IR function.
    std::string entryName;
    /// Saved IRFunction.
    const IRFunction *savedF;
  };
  /// WeightInfo represents a constant weight and a constant it is produced
  /// from.
  using WeightInfo = std::pair<const WeightVar *, const Constant *>;
  /// Comparator for WeightInfo objects, sorting them by their allocated
  /// address.
  class WeightAddrComparator {
  public:
    WeightAddrComparator(const BundleSaver &bundleSaver)
        : bundleSaver_(bundleSaver) {}
    bool operator()(const WeightInfo &LHS, const WeightInfo &RHS) const;

  private:
    const BundleSaver &bundleSaver_;
  };
  /// Ctor.
  explicit BundleSaver(const IRFunction *F, const LLVMBackend &llvmBackend);
  explicit BundleSaver(const LLVMBackend &llvmBackend,
                       llvm::StringRef outputDir, llvm::StringRef bundleName);
  /// Save code bundle built for \p target, \p arch, \p cpu and \p
  /// targetFeatures to \p outputDir under name \p bundleName. Make
  /// \p mainEntryName the function name for the entry point of the network and
  /// prepend all generated files with this name.
  void save(llvm::StringRef target, llvm::StringRef arch, llvm::StringRef cpu,
            const llvm::SmallVectorImpl<std::string> &targetFeatures,
            llvm::StringRef outputDir, llvm::StringRef bundleName,
            llvm::StringRef mainEntryName, llvm::CodeModel::Model codeModel,
            llvm::Reloc::Model relocModel);
  void save(llvm::StringRef mainEntryName, const IRFunction *F);
  /// Produce a bundle.
  void produceBundle();

private:
  /// Perform memory allocation for a bundle.
  void performBundleMemoryAllocation();
  /// Save weights for the bundle.
  void saveWeights(llvm::StringRef weightsFileName);
  /// Save header file for the bundle.
  void saveHeader(llvm::StringRef headerFileName);
  /// Emit config for a bundle.
  void emitBundleConfig();
  /// Emit the symbol table for a bundle.
  void emitSymbolTable();
  /// Emit the entry function for the bundle.
  void emitBundleEntryFunction();
  /// Set current IRFunction.
  void setIRFunction(llvm::StringRef mainEntryName, const IRFunction *F);
  /// Returns a set of placeholders associated with IR functions inside this
  /// bundle.
  std::set<const Placeholder *> findPlaceholders() const;
  /// Returns a set of constant weights associated with IR functions inside this
  /// bundle.
  std::set<WeightInfo, WeightAddrComparator> findConstantWeights() const;
  /// \returns the weight that the variable \p v is lowered into in one of the
  /// IR functions inside this bundle, or null if the variable is unknown.
  Value *getWeightForNode(const Storage *V) const;
  /// Information about allocations.
  AllocationsInfo allocationsInfo_;
  /// The LLVM IR code generator.
  std::unique_ptr<LLVMIRGen> irgen_;
  /// The output directory to be used.
  std::string outputDir_;
  /// The name of the bundle to be saved.
  std::string bundleName_;
  /// Information about IR functions inside this bundle.
  std::vector<SavedIRFunction> savedIRFunctions_;
  /// Indicates if this bundle was saved already.
  bool isSaved_{false};
};

} // namespace glow

#endif // GLOW_LLVMIRCODEGEN_BUNDLESAVER_H
