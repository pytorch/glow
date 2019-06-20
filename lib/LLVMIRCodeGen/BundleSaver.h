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
#ifndef GLOW_LLVMIRCODEGEN_BUNDLESAVER_H
#define GLOW_LLVMIRCODEGEN_BUNDLESAVER_H

#include "glow/LLVMIRCodeGen/AllocationsInfo.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"

#include "glow/IR/IR.h"

namespace glow {
class LLVMBackend;

class BundleSaver final {
  /// The IR to be compiled.
  const IRFunction *F_;
  /// Information about allocations.
  AllocationsInfo allocationsInfo_;
  /// The LLVM IR code generator.
  std::unique_ptr<LLVMIRGen> irgen_;

  /// Perform memory allocation for a bundle.
  void performBundleMemoryAllocation();
  /// Save weights for the bundle.
  void saveWeights(llvm::StringRef weightsFileName);
  /// Produce a bundle.
  void produceBundle(llvm::StringRef outputDir);
  /// Emit config for a bundle.
  void emitBundleConfig();
  /// Emit the symbol table for a bundle.
  void emitSymbolTable();
  /// Emit the entry function for the bundle.
  void emitBundleEntryFunction();

public:
  /// Ctor.
  explicit BundleSaver(const IRFunction *F, const LLVMBackend &llvmBackend);
  /// Save code bundle built for \p target, \p arch, \p cpu and \p
  /// targetFeatures to \p outputDir under name \p bundleName. Make
  /// \p mainEntryName the function name for the entry point of the network and
  /// prepend all generated files with this name.
  void save(llvm::StringRef target, llvm::StringRef arch, llvm::StringRef cpu,
            const llvm::SmallVectorImpl<std::string> &targetFeatures,
            llvm::StringRef outputDir, llvm::StringRef bundleName,
            llvm::StringRef mainEntryName);
};

} // namespace glow

#endif // GLOW_LLVMIRCODEGEN_BUNDLESAVER_H
