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
#ifndef GLOW_LLVMIRCODEGEN_LLVMBACKEND_H
#define GLOW_LLVMIRCODEGEN_LLVMBACKEND_H

#include "glow/Backend/Backend.h"
#include "glow/Backend/CompiledFunction.h"
#include "glow/Base/Tensor.h"
#include "glow/LLVMIRCodeGen/GlowJIT.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/IRBuilder.h"

namespace glow {

class AllocationsInfo;
class BundleSaver;
class PlaceholderBindings;
class LLVMIRGen;

/// LLVM backend options used to configure e.g. the LLVM TargetMachine, ORC JIT
/// or BundleSaver.
class LLVMBackendOptions {
  /// Target triple used by this backend.
  std::string target_;
  /// Arch used by this backend.
  std::string arch_;
  /// Cpu used by this backend.
  std::string cpu_;
  /// ABI to be used by this backend.
  std::string abi_;
  /// Float ABI to be used by this backend.
  llvm::Optional<llvm::FloatABI::ABIType> floatABI_;
  /// Code model used by this backend.
  llvm::CodeModel::Model codeModel_;
  /// Code model used by this backend for bundles.
  llvm::CodeModel::Model bundleCodeModel_;
  /// Relocation model used by this backend.
  llvm::Reloc::Model relocModel_;
  /// LLVM target features used by this backend.
  llvm::SmallVector<std::string, 0> targetFeatures_;
  /// Bundle API to use.
  BundleApiType bundleAPI_;

public:
  LLVMBackendOptions();
  /// \returns target triple used by this backend.
  const std::string &getTarget() const { return target_; }
  /// Sets target triple used by this backend.
  void setTarget(llvm::StringRef target) { target_ = target.str(); }
  /// \returns arch used by this backend.
  const std::string &getArch() const { return arch_; }
  /// Sets arch used by this backend.
  void setArch(llvm::StringRef arch) { arch_ = arch.str(); }
  /// \returns cpu used by this backend.
  const std::string &getCPU() const { return cpu_; }
  /// Sets cpu used by this backend.
  void setCPU(llvm::StringRef cpu) { cpu_ = cpu.str(); }
  /// \returns ABI used by this backend.
  const std::string &getABIName() const { return abi_; }
  /// Sets ABI used by this backend.
  void setABIName(llvm::StringRef abi) { abi_ = abi.str(); }
  /// \returns Float ABI used by this backend.
  llvm::Optional<llvm::FloatABI::ABIType> getFloatABI() const {
    return floatABI_;
  }
  /// Sets Float ABI used by this backend.
  void setFloatABI(llvm::Optional<llvm::FloatABI::ABIType> floatABI) {
    floatABI_ = floatABI;
  }
  /// \returns code model used by this backend.
  llvm::CodeModel::Model getCodeModel() const { return codeModel_; }
  /// Sets code model used by this backend.
  void setCodeModel(llvm::CodeModel::Model codeModel) {
    codeModel_ = codeModel;
  }
  /// \returns code model used by this backend for bundles.
  llvm::CodeModel::Model getBundleCodeModel() const { return bundleCodeModel_; }
  /// Sets code model used by this backend for bundles.
  void setBundleCodeModel(llvm::CodeModel::Model codeModel) {
    bundleCodeModel_ = codeModel;
  }
  /// \returns bundle API used by this backend for bundles.
  BundleApiType getBundleAPI() const { return bundleAPI_; }
  /// Sets bundle API used by this backend for bundles.
  void setBundleAPI(BundleApiType api) { bundleAPI_ = api; }
  /// \returns relocation model used by this backend.
  llvm::Reloc::Model getRelocModel() const { return relocModel_; }
  /// Sets relocation model used by this backend.
  void setRelocModel(llvm::Reloc::Model relocModel) {
    relocModel_ = relocModel;
  }
  /// \returns target features used by this backend.
  const llvm::SmallVectorImpl<std::string> &getTargetFeatures() const {
    return targetFeatures_;
  }
  /// Adds target features used by this backend.
  void addTargetFeatures(llvm::ArrayRef<std::string> targetFeatures) {
    targetFeatures_.append(targetFeatures.begin(), targetFeatures.end());
  }
  /// Sets target features used by this backend.
  void setTargetFeatures(llvm::ArrayRef<std::string> targetFeatures) {
    targetFeatures_.clear();
    addTargetFeatures(targetFeatures);
  }
};

class LLVMBackend : public BackendUsingGlowIR {
public:
  LLVMBackend();
  /// @name Backend methods.
  /// This is the implementation of the Backend interface.
  ///@{
  virtual ~LLVMBackend() override = default;

  /// \returns the LLVM target triple for the host.
  static std::string getHostTarget();

  /// \returns the LLVM CPU name for the host.
  static std::string getHostCPU();

  /// \returns the LLVM CPU feature list for the host.
  static llvm::SmallVector<std::string, 0> getHostFeatures();

  /// \returns LLVM backend options.
  const LLVMBackendOptions &getOptions() const { return options_; }

  /// \returns LLVM backend options.
  LLVMBackendOptions &getOptions() { return options_; }

  /// Sets LLVM backend options.
  void setOptions(const LLVMBackendOptions &options) { options_ = options; }

  /// \returns whether the provided \p NI is supported by the backend.
  bool isOpSupported(const NodeInfo &NI) const override;

  virtual std::unique_ptr<CompiledFunction>
  compileIR(std::unique_ptr<IRFunction> IR) const override;

  virtual std::unique_ptr<CompiledFunction>
  compileIRWithoutConstants(IRFunction *IR) const;

  virtual Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

  virtual void save(Function *F, llvm::StringRef outputDir,
                    llvm::StringRef bundleName,
                    llvm::StringRef mainEntryName) const override;

  virtual void saveFunctions(llvm::ArrayRef<BundleEntry> entries,
                             llvm::StringRef outputDir,
                             llvm::StringRef bundleName) const override;
  /// @}

  /// \returns the size of metrics collected for a single TraceEvent.
  virtual size_t getTraceEventDataSize() const override {
    return sizeof(uint64_t);
  }

  /// Method that creates the LLVM IR generator. This gives the possibility to
  /// create a backend that inherits from the CPU backend, while providing
  /// a specific version of the LLVM IR generator derived from LLVMIRGen.
  /// \param IR the IRFunction function to be converted into the LLVM IR.
  /// \param allocationsInfo information about allocation of weights and
  /// activations.
  /// \returns backend-specific LLVMIRGen instance.
  virtual std::unique_ptr<LLVMIRGen>
  createIRGen(const IRFunction *IR, AllocationsInfo &allocationsInfo) const = 0;

  /// Method that creates a BundleSaver. This gives the possibility to
  /// create a backend that inherits from the LLVMBackend backend, while
  /// providing a specific version of the BundleSaver derived from BundleSaver.
  /// \param llvmBackend backend to be used to produce in a bundle.
  /// \param outputDir output directory for the bundle.
  /// \param bundleName the name of the bundle.
  /// \returns backend-specific BundleSaver instance.
  virtual std::unique_ptr<BundleSaver>
  createBundleSaver(const LLVMBackend &llvmBackend, llvm::StringRef outputDir,
                    llvm::StringRef bundleName) const;

protected:
  /// Method that creates a CompiledFunction.
  /// \param JIT GlowJIT to be used.
  /// \param runtimeBundle bundle to be used for compiling the function.
  /// \returns created CompiledFunction.
  virtual std::unique_ptr<CompiledFunction>
  createCompiledFunction(std::unique_ptr<GlowJIT> JIT,
                         runtime::RuntimeBundle &&runtimeBundle) const = 0;

  /// \returns libjit bitcode for the current backend.
  virtual llvm::StringRef getLibjitBitcode() const = 0;

  /// Emit the jitmain function.
  virtual void emitJitMain(LLVMIRGen &irgen) const;

  /// LLVM backend options.
  LLVMBackendOptions options_;
};

} // namespace glow

#endif // GLOW_LLVMIRCODEGEN_LLVMBACKEND_H
