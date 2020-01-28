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
#ifndef GLOW_BACKENDS_BACKEND_H
#define GLOW_BACKENDS_BACKEND_H

#include "glow/Backend/CompiledFunction.h"
#include "glow/Backends/BackendOptions.h"
#include "glow/Base/Traits.h"
#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Support/Register.h"

#include "llvm/ADT/StringRef.h"

namespace glow {

class IRFunction;
class Node;
class PlaceholderBindings;
class IRGenVisitor;
class FunctionPassPipeline;
class TensorLayoutCommon;

/// Information about an entry point of a saved bundle.
struct BundleEntry {
  /// Name of the bundle entry point for the function to be saved.
  std::string name;
  /// Function to be saved.
  Function *func;
};

namespace runtime {

class DeviceManager;
struct DeviceInfo;
struct DeviceConfig;

} // namespace runtime

// This is the interface that glow backends need to implement.
class Backend {
public:
  /// Dtor.
  virtual ~Backend() = default;

  // \returns backend name.
  virtual std::string getBackendName() const = 0;

  /// Generate code for a vector of functions, \p functions. All compilations
  /// use the same settings provided by \p opts. This allows the compiler to
  /// support shared constants between functions.
  virtual Expected<std::vector<std::unique_ptr<CompiledFunction>>>
  compileFunctions(llvm::ArrayRef<Function *> functions,
                   BackendOptions &opts) const {
    std::vector<std::unique_ptr<CompiledFunction>> compiledFunctions;
    for (auto &function : functions) {
      if (auto resOrErr = compile(function, opts)) {
        compiledFunctions.push_back(std::move(*resOrErr));
      } else {
        return resOrErr.takeError();
      }
    }
    return Expected<std::vector<std::unique_ptr<CompiledFunction>>>(
        std::move(compiledFunctions));
  }

  virtual Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F) const {
    BackendOptions opts;
    return compile(F, opts);
  }

  /// Generate code for input function \param F given settings in \p opts.
  virtual Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const = 0;

  /// Save the bundle for \p F for a later standalone execution in \p outputDir
  /// under name \p bundleName. Make \p mainEntryName the function name for the
  /// entry point of the network and prepend all generated files with this name.
  virtual void save(Function *F, llvm::StringRef outputDir,
                    llvm::StringRef bundleName,
                    llvm::StringRef mainEntryName) const {
    LOG(FATAL) << "Saving a bundle is not supported by the backend";
  }

  virtual void saveFunctions(llvm::ArrayRef<BundleEntry> entries,
                             llvm::StringRef outputDir,
                             llvm::StringRef bundleName) const {
    LOG(FATAL) << "Saving a bundle is not supported by the backend";
  }

  /// Used by the compiler during graph optimization and before code generation,
  /// giving the backend an opportunity to transform the graph before IRGen. The
  /// backend may insert backend and device-specific nodes. The backend is
  /// responsible for cleaning up after itself.
  /// \returns True if the graph was modified.
  virtual bool transformPostLowering(
      Function *F, CompilationContext &cctx,
      const glow::runtime::DeviceInfo *devInfo = nullptr) const {
    return false;
  }

  /// \returns whether the provided \p NI is supported by the backend.
  virtual bool isOpSupported(const NodeInfo &NI) const = 0;

  /// \returns whether all nodes inside \p F are supported. \p verbose
  /// represents whether to print Nodes that are unsupported.
  bool checkAllNodesSupported(const Function &F, bool verbose = true) const;

  /// \returns whether the provided \p F conforms to the backend-dependent graph
  /// constraints. Giving the backend an opportunity to check that everything
  /// conforms to its specific restrictions by overriding this function. It is
  /// highly recommended for backends to make their backend specific
  /// verifications a super-set of target independent Function::verify() by
  /// calling it in their overridden implementation. It is not a strict
  /// requirement, of course, in case they diverge / the backend has a good
  /// reason not to call Function::verify(). \p verbose represents whether to
  /// print out nodes that are unsupported by the backend.
  virtual bool verify(const Function &F, bool verbose = true) const;

  /// \returns whether the provided \p IR conforms to the backend-dependent
  /// graph constraints. Giving the backend an opportunity to check that
  /// everything conforms to its specific restrictions by overriding this
  /// function. It is highly recommended for backends to make their backend
  /// specific verifications a super-set of target independent
  /// IRFunction::verify() by calling it in their overridden implementation. It
  /// is not a strict requirement, of course, in case they diverge / the backend
  /// has a good reason not to call IRFunction::verify().
  virtual bool verify(const IRFunction &IR) const;

  /// \returns a reference to the backend-specific tensor layout requirements
  /// singleton. If not overridden, the default requirement is Glow's
  /// "canonical" form.
  virtual TensorLayoutCommon &getTensorLayoutRequirements() const;

  /// \returns true if the supplied Node \N should be lowered. By default, all
  /// Nodes are candidates for lowering.
  virtual bool shouldLower(const Node *N) const { return true; }

  /// \returns true if the Backend wants the buffer sharing optimization
  /// performed.
  virtual bool shouldShareBuffers() const { return true; }

  /// Modify the \p optimizationOpts however desired.
  virtual FunctionPassPipeline getOptimizationPipeline() const;

  /// \returns true if the Backend supports partial, unpadded tensors for
  /// inputs that can have variable size (e.g., embedding indices).
  virtual bool supportsPartialTensors() const { return false; }

  /// \returns true if the Backend supports static Placeholders. This means
  /// an input can be treated as a placeholder that can be reused on the device
  /// for multiple requests.
  virtual bool supportsStaticPlaceholders() const { return false; }

  /// \returns whether the backend supports fusing \p activation into \p parent.
  virtual bool supportsFusedActivation(Node *parent, Node *activation) const {
    return false;
  }

  /// \returns true if Backend generated Instruction for Node \p N,
  /// using IRGenVisitor \p irgen.
  virtual bool generateInst(Node *N, IRGenVisitor &irgen) const {
    return false;
  }

  virtual size_t getTraceEventDataSize() const { return 0; }

  /// Create device manager corresponding to the backend based on the
  /// deviceConfig.
  virtual runtime::DeviceManager *
  createDeviceManager(const runtime::DeviceConfig &deviceConfig);

  /// \returns the supported options for compiled functions (name=>description).
  virtual llvm::StringMap<std::string>
  getSupportedCompiledFunctionOptions() const {
    return llvm::StringMap<std::string>();
  };

  /// \returns the supported options for device managers (name=>description).
  virtual llvm::StringMap<std::string>
  getSupportedDeviceManagerOptions() const {
    return llvm::StringMap<std::string>();
  };

protected:
  /// Parses the graph \F and builds a TraceInfo structure from any found
  /// TraceEventNodes.
  TraceInfo buildManualTraceInfo(Function *F) const;

  /// Inserts a TraceEventInst between every instruction, the most basic form of
  /// auto instrumentation. Necessary only if the Backend doesn't provide
  /// profiling/tracing in another way.
  /// Modifies \p IR and updates \p traceInfo.
  void autoInstrument(TraceInfo &traceInfo, IRFunction *IR) const;
};

/// Create a backend based on the registered backend name \p backendName.
Backend *createBackend(llvm::StringRef backendName);

// Backends that use Glow low-level IR should inherit from this class. It allows
// for unit tests to create low-level IR to compile and run.
class BackendUsingGlowIR : public Backend {
public:
  /// Generate code for input IR function \param IR.
  /// This is used only for unit testing.
  virtual std::unique_ptr<CompiledFunction>
  compileIR(std::unique_ptr<IRFunction> IR) const = 0;
};

/// Perform Backend Factory registration.
#define REGISTER_GLOW_BACKEND_FACTORY(FactoryName, BackendClass)               \
  class FactoryName : public BaseFactory<std::string, Backend> {               \
  public:                                                                      \
    Backend *create() override { return new BackendClass(); }                  \
    std::string getRegistrationKey() const override {                          \
      return BackendClass::getName();                                          \
    }                                                                          \
    unsigned numDevices() const override {                                     \
      return BackendClass::numDevices();                                       \
    }                                                                          \
  };                                                                           \
  static RegisterFactory<std::string, FactoryName, Backend>                    \
      FactoryName##_REGISTERED;

/// \returns the set of names for all available, registered backends.
std::vector<std::string> getAvailableBackends();

/// The backend name used in Glow quantization profiling.
#ifdef GLOW_WITH_CPU
constexpr const char *profilingBackend = "CPU";
#else
constexpr const char *profilingBackend = "Interpreter";
#endif

} // namespace glow

#endif // GLOW_BACKENDS_BACKEND_H
