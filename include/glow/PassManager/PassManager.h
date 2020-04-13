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
#ifndef GLOW_PASSMANAGER_PASSMANAGER_H
#define GLOW_PASSMANAGER_PASSMANAGER_H

#include "glow/Backend/Backend.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

#include <atomic>

namespace glow {

struct CompilationContext;
class IRContainer;
class PassBase;

/// A set of options and command-line options for a pass manager.
struct PassManagerOptions {
  /// The unique pass manager id.
  const std::string passManagerID;
  /// Command-line options category to be used for this pass manager.
  llvm::cl::OptionCategory passManagerCat;

  llvm::cl::opt<bool> verifyBeforeAllPassesOpt;

  llvm::cl::list<std::string> verifyBeforePassesOpt;

  llvm::cl::opt<bool> verifyAfterAllPassesOpt;

  llvm::cl::list<std::string> verifyAfterPassesOpt;

  llvm::cl::opt<bool> dumpIRBeforeAllPassesOpt;

  llvm::cl::list<std::string> dumpIRBeforePassesOpt;

  llvm::cl::opt<bool> dumpIRAfterAllPassesOpt;

  llvm::cl::list<std::string> dumpIRAfterPassesOpt;

  llvm::cl::opt<bool> printPassesOpt;

  llvm::cl::opt<unsigned> stopAfterPassNumOpt;

  PassManagerOptions(const char *id);
  /// Helper to check if \p otherStr is in \p strList.
  static bool listContainsString(const llvm::cl::list<std::string> &strList,
                                 llvm::StringRef otherStr);
};

/// The base class for pass managers. It contains most of the logic common for
/// all pass managers, but provides a number of hooks that can be overridden by
/// concrete pass manager to customize the behavior.
class PassManagerBase : public Named {
protected:
  /// The index of the current pass being executed in the pipeline.
  size_t passIdx_ = 0;
  /// The Backend we have for backend-specific verification.
  const Backend *backend_;

  /// Logic to execute before pass \p P is run on \p C, given \p cctx. \returns
  /// if \p C was modified.
  bool runPrePass(IRContainer *C, const CompilationContext &cctx,
                  const PassBase &P);

  /// A runPrePass customization point for the derived classes.
  virtual void runPrePassHook(IRContainer *C, const CompilationContext &cctx,
                              const PassBase &P);

  /// Logic to execute after pass \p P is run on \p C, given \p cctx. \returns
  /// if \p C was modified.
  bool runPostPass(IRContainer *C, const CompilationContext &cctx,
                   const PassBase &P);

  /// A runPostPass customization point for the derived classes.
  virtual void runPostPassHook(IRContainer *C, const CompilationContext &cctx,
                               const PassBase &P);

  // /// Runs a FunctionPass described by \p passConfig over \p C given \p cctx.
  /// Run a pass configured by \p passConfig over IR \p C given \p cctx.
  virtual bool runPass(const PassConfigBase &passConfig, IRContainer *C,
                       const CompilationContext &cctx);

  /// A runPass customization point for the derived classes.
  virtual bool runPassHook(const PassConfigBase &passConfig, glow::PassBase &P,
                           IRContainer *C, const CompilationContext &cctx) = 0;

  /// Runs a pass corresponding to the provided \p passConfig on the IR
  /// container \p C given \p cctx.
  virtual bool runPassWithConfig(const PassConfigBase &passConfig,
                                 IRContainer *C,
                                 const CompilationContext &cctx) = 0;

  /// \returns the name of the pass corresponding to \p passConfig.
  virtual llvm::StringRef
  getNameOfPass(const PassConfigBase &passConfig) const = 0;

  /// Creates and \returns a Pass given a provided \p passConfig.
  virtual std::unique_ptr<PassBase>
  createFunctionPass(const PassConfigBase &passConfig) const = 0;

  /// Run the PassPipeline given the \ref pipeline_ and
  /// \p cctx. \returns whether \p C was modified.
  bool run(IRContainer *C, const CompilationContext &cctx);

  /// \returns the size of the pass pipeline.
  virtual size_t getPipelineSize() const = 0;

  /// \returns the \p idx of the pass pipeline.
  virtual const PassConfigBase &getPipelineElement(size_t idx) const = 0;

  /// Dump the IR of \p C into the output stream \p or into a file \p
  /// outputFileName, depending on the backend type.
  virtual void dumpIR(IRContainer *C, llvm::raw_ostream &os,
                      const std::string &outputFileName) const = 0;

public:
  /// Constructor.
  PassManagerBase(llvm::StringRef name) : Named(name) {}
  virtual ~PassManagerBase() = default;

  /// Dump a textual representation of the Manager to \p os.
  void dump(llvm::raw_ostream &os = llvm::outs()) const;

  /// \returns the result of verification for a provided IR container \p C.
  virtual bool verify(IRContainer &C) const = 0;

  /// \returns the result of a backend-specific verification for a provided IR
  /// container \p C.
  virtual bool verify(const Backend &B, IRContainer &C) const = 0;

  /// Get options of this pass manager.
  virtual const PassManagerOptions &getOptions() const = 0;
  /// Get the global pass counter associated with the pass manager.
  virtual std::atomic<unsigned> &globalPassCounter() = 0;
  /// Get the global pass counter associated with the pass manager.
  virtual const std::atomic<unsigned> &globalPassCounter() const = 0;
};

template <typename IRPassTy, typename PassIDTy>
std::unique_ptr<IRPassTy> createFunctionPass(PassIDTy);

/// Manager for running a series of FunctionPasses. Given some Function,
/// CompilationContext, and provided Pipeline, it will run all passes on the
/// Function. Enables easier debugging given runPrePass() and runPostPass()
/// calls, which can be modified to run some code before or before every pass.
template <typename IRPASS_PIPELINE, typename IRPASS>
class PassManager : public PassManagerBase {
public:
  using IRPassTy = IRPASS;
  using IRPassPipelineTy = IRPASS_PIPELINE;
  using IRContainerTy = typename IRPassTy::IRContainerTy;
  using PassIDTy = typename IRPassTy::PassIDTy;
  using IRPassConfigTy = typename IRPassPipelineTy::IRPassConfigTy;

private:
  /// The pipeline of passes to run.
  IRPassPipelineTy pipeline_;
  /// Options and command-line options for this pass manager.
  static PassManagerOptions &options_;

  /// Global pass counter used to identify each pass.
  std::atomic<unsigned> globalPassCounter_;

  /// Creates and \returns a Pass given a provided \p passID.
  std::unique_ptr<IRPassTy> createFunctionPass(PassIDTy passID) const;

  std::unique_ptr<PassBase>
  createFunctionPass(const PassConfigBase &passConfig) const override {
    return createFunctionPass(
        static_cast<const IRPassConfigTy *>(&passConfig)->getPassID());
  }

  bool runPassHook(const PassConfigBase &passConfig, PassBase &P,
                   IRContainer *C, const CompilationContext &cctx) override;

  bool runPassWithConfig(const PassConfigBase &passConfig, IRContainer *C,
                         const CompilationContext &cctx) override {
    return runPass(*static_cast<const IRPassConfigTy *>(&passConfig),
                   static_cast<IRContainerTy *>(C), cctx);
  }

  llvm::StringRef
  getNameOfPass(const PassConfigBase &passConfig) const override {
    return glow::getNameOfPass(
        static_cast<const IRPassConfigTy *>(&passConfig)->getPassID());
  }

  size_t getPipelineSize() const override { return getPipeline().size(); }

  const PassConfigBase &getPipelineElement(size_t idx) const override {
    return getPipeline().at(idx);
  }

  bool verify(IRContainer &C) const override {
    return static_cast<IRContainerTy *>(&C)->verify();
  }

  bool verify(const Backend &B, IRContainer &C) const override {
    return B.verify(*static_cast<IRContainerTy *>(&C));
  }

public:
  /// Constructor.
  /// Create a pass with a given \p name, provided the \p pipeline and an
  /// optional \p backend.
  PassManager(llvm::StringRef name, IRPassPipelineTy pipeline,
              const Backend *backend = nullptr)
      : PassManagerBase(name), pipeline_(pipeline) {
    backend_ = backend;
    passIdx_ = 0;
  }

  virtual ~PassManager() = default;

  /// Run the PassPipeline pipeline_ on the IR container \p C and
  /// \p cctx. \returns whether \p C was modified.
  bool run(IRContainerTy *C, const CompilationContext &cctx) {
    return PassManagerBase::run(C, cctx);
  }

  /// Getter for a reference to the Pipeline used by this PassManager.
  const IRPassPipelineTy &getPipeline() const { return pipeline_; };

  /// Dump a textual representation of the PassManager to \p os.
  void dump(llvm::raw_ostream &os = llvm::outs()) const {
    PassManagerBase::dump(os);
    getPipeline().dump();
  }

  void dumpIR(IRContainer *C, llvm::raw_ostream &os,
              const std::string &outputFileName) const override;

  /// Get options of this pass manager.
  const PassManagerOptions &getOptions() const override { return options_; }

  std::atomic<unsigned> &globalPassCounter() override {
    return globalPassCounter_;
  }

  const std::atomic<unsigned> &globalPassCounter() const override {
    return globalPassCounter_;
  }
};

} // namespace glow

#endif // GLOW_PASSMANAGER_PASSMANAGER_H
