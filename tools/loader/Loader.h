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
#ifndef GLOW_TOOLS_LOADER_LOADER_H
#define GLOW_TOOLS_LOADER_LOADER_H

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Importer/ProtobufLoader.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Runtime/HostManager/HostManager.h"

#include "llvm/Support/CommandLine.h"

#include <glog/logging.h>

/// Options.
extern llvm::cl::OptionCategory loaderCat;

/// Number of devices to use.
extern llvm::cl::opt<unsigned> numDevices;

/// Whether to run all inputs on all numDevices. Used for testing.
extern llvm::cl::opt<bool> runAllInputsOnAllDevices;

/// Timer option used to indicate if inferences should be timed -time.
extern llvm::cl::opt<bool> timeOpt;
/// Iterations used to indicate the number of iterations to run an inferece
/// -iterations.
extern llvm::cl::opt<unsigned> iterationsOpt;

namespace glow {

class Tensor;
struct CompilationContext;

/// \return true if emit bundle mode is enabled.
bool emittingBundle();

/// \return true if profiling the graph.
bool profilingGraph();

/// Parse/verify command line parameters.
void parseCommandLine(int argc, char **argv);

/// Loader extension interface from which to derive in order to extend the
/// Loader driver object.
class Loader;
class ProtobufLoader;

class LoaderExtension {
public:
  /// Called once after ONNX or Caffe2 model loading.
  virtual void postModelLoad(Loader &, PlaceholderBindings &, ProtobufLoader &,
                             llvm::StringMap<Placeholder *> &,
                             size_t compilationBatchSize) = 0;
  /// Called once at the beginning of the mini-batch inference.
  virtual void inferInitMiniBatch(Loader &, PlaceholderBindings &,
                                  size_t minibatchIndex,
                                  size_t minibatchSize) = 0;
  /// Called once after the completion of the mini-batch inference.
  virtual void inferEndMiniBatch(Loader &, PlaceholderBindings &,
                                 size_t minibatchIndex,
                                 size_t minibatchSize) = 0;
  virtual ~LoaderExtension() {}
};

/// Driver class for loading, compiling, and running inference for ONNX and
/// Caffe2 models.
class Loader {
  /// Caffe2 network file name.
  std::string caffe2NetDescFilename_;
  /// Caffe2 weights file name.
  std::string caffe2NetWeightFilename_;
  /// ONNX model file name.
  std::string onnxModelFilename_;
  /// Name of loaded function.
  std::string functionName_;
  /// Host Manager for running the model.
  std::unique_ptr<glow::runtime::HostManager> hostManager_;
  /// Backend used for saving bundle and quantization.
  std::unique_ptr<glow::Backend> backend_;
  /// Function containing the model.
  Function *F_{nullptr};
  /// Module
  std::unique_ptr<Module> M_;
  /// A map between quantization profiling names of NodeValues that were lowered
  /// from each other. Maps to a set of names of NodeValues and their NodeKinds
  /// that were replaced by the NodeValue (whose output name is the key) that
  /// replaced them.
  LoweredInfoMap loweredMap_;
  /// List of Loader owned extension objects.
  std::vector<std::unique_ptr<LoaderExtension>> loaderExtensionList_;

public:
  /// Getter for the hostManager, this can be useful for calling into the
  /// HostManager directly.
  runtime::HostManager *getHostManager() { return hostManager_.get(); }

  /// Getter for the Function. This should not be called after compile since the
  /// compile process is destructive on the original function.
  Function *getFunction() { return F_; }

  /// Getter for function name.
  std::string getFunctionName() { return functionName_; }

  /// Getter for the Module. This should not be called after compile since the
  /// compile process is destructive on the original function and module.
  Module *getModule() { return F_->getParent(); }

  /// Getter for the Caffe2 network file name.
  llvm::StringRef getCaffe2NetDescFilename() { return caffe2NetDescFilename_; }

  /// Getter for the Caffe2 weights file name.
  llvm::StringRef getCaffe2NetWeightFilename() {
    return caffe2NetWeightFilename_;
  }

  /// Getter for the ONNX model file name.
  llvm::StringRef getOnnxModelFilename() { return onnxModelFilename_; }

  /// Getter for the model path.
  /// \pre (modelPathOpt.size() == 1)
  static std::string getModelOptPath();

  /// Getter for the model path, expected to be a directory.
  /// \pre (modelPathOpt.size() == 1)
  static llvm::StringRef getModelOptDir();

  /// Get the quantization options based on the command line parameters of the
  /// Loader.
  static quantization::QuantizationConfiguration getQuantizationConfiguration();

  /// Load a Caffe2 or ONNX model into this Loader object according to the
  /// Loader command line options.
  std::unique_ptr<ProtobufLoader> loadModel();

  /// Get the compilation options (context) for a given quantization \p mode.
  /// The options are initialized by the Loader command line arguments.
  CompilationContext getCompilationContext(QuantizationMode mode);

  /// Get the default compilation options (context) initialized by the Loader
  /// command line arguments.
  CompilationContext getCompilationContext();

  /// Compiles the Function F_. Handles quantization, emitting bundles, and
  /// dumping debug information. \p bindings bind specific
  /// placeholders to concrete tensors. The concrete tensors include
  /// quantization profile guided information.
  void compile(PlaceholderBindings &bindings);

  /// Compiles the Function F_. Handles quantization, emitting bundles, and
  /// dumping debug information. \p cctx is used for compiling F_.
  void compile(CompilationContext &cctx);

  /// Runs inference, unless emit bundle mode is enabled. \p bindings
  /// binds specific placeholders to concrete tensors. The concrete
  /// tensors include quantization profile guided information.
  void runInference(PlaceholderBindings &bindings, size_t batchSize = 1);

  /// Runs inference, \p context binds both Tensors to Placeholders and
  /// potentially holds a TraceContext. This method allows obtaining TraceEvents
  /// from the run.
  void runInference(ExecutionContext *context, size_t batchSize = 1);

  /// Register a loader extension.
  Loader &registerExtension(std::unique_ptr<LoaderExtension> ext);
  /// Called once after ONNX or Caffe2 model loading.
  void postModelLoad(PlaceholderBindings &bindings, ProtobufLoader &protoLoader,
                     llvm::StringMap<Placeholder *> &,
                     size_t compilationBatchSize);
  /// Called at the beginning of each mini-batch inference.
  void inferInitMiniBatch(PlaceholderBindings &bindings, size_t minibatchIndex,
                          size_t minibatchSize);
  /// Called after the completion of each mini-batch inference.
  void inferEndMiniBatch(PlaceholderBindings &, size_t minibatchIndex,
                         size_t minibatchSize);

  /// Generates and serializes the profiling infos after gathering a profile
  /// by running inference one or more times. \p bindings
  /// binds specific placeholders to concrete tensors. The concrete tensors
  /// include quantization profile guided information.
  void generateAndSerializeProfilingInfos(PlaceholderBindings &bindings);

  /// Create the Loader driver object. If \p configDeviceIDs is empty then \ref
  /// numDevices DeviceConfigs are created for each device, otherwise
  /// configDeviceIDs is used to create DeviceConfigs with specified IDs.
  Loader(llvm::ArrayRef<size_t> configDeviceIDs = {});
};

} // namespace glow

#endif // GLOW_TOOLS_LOADER_LOADER_H
