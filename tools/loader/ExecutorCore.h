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

#ifndef GLOW_TOOLS_LOADER_EXECUTOR_CORE_H
#define GLOW_TOOLS_LOADER_EXECUTOR_CORE_H

#include "Loader.h"
#include "glow/Graph/Nodes.h"

namespace glow {
class PostProcessOutputDataExtension {
public:
  /// Called once per mini-batch after network is executed to post process
  /// output(s).
  virtual int
  processOutputs(const llvm::StringMap<Placeholder *> &PHM,
                 PlaceholderBindings &bindings,
                 VecVecRef<std::string> inputImageBatchFilenames) = 0;
  virtual ~PostProcessOutputDataExtension(){};
};

using PostProcessExtFuncPtr =
    std::function<std::unique_ptr<PostProcessOutputDataExtension>()>;

class PreProcessInputDataExtension {
public:
  /// Called once per batch after images are loaded in to Tensor.
  virtual void processInputTensor(llvm::ArrayRef<Tensor *> inputImageData,
                                  size_t startId, size_t endId,
                                  size_t batchSz) = 0;
  virtual ~PreProcessInputDataExtension(){};
};

class Executor final {
public:
  Executor(std::string appName, int argc, char **argv);
  Executor() = delete;
  Executor(const Executor &) = delete;
  Executor &operator=(const Executor &) = delete;

  /// Registers a Loader Extension that will be invoked after model is
  /// loaded. If multiple extensions are registered they will be executed in
  /// order they were registered.
  void registerLoaderExtension(
      std::function<std::unique_ptr<LoaderExtension>()> func);
  /// Registers an extension that will be invoked on Tensor containing current
  /// batch of input data. If multiple extensions are registered they will be
  /// executed in order they were registered.
  /// A new instance of the extension will be created for each thread.
  void registerInputDataPreProcessingExtension(
      std::function<std::unique_ptr<PreProcessInputDataExtension>()> func);
  /// Registers extension that will be invoked for each execution of the
  /// network. If multiple extensions are registered they will be executed in
  /// order they were registered.
  /// A new instance of the extension will be created for each thread.
  void registerPostProcessOutputExtension(PostProcessExtFuncPtr func);
  /// This will parse command line, load, build and execute a network.
  /// Returns /p 0 if no errors occured, others none zero value.
  int executeNetwork();

private:
  /// Iterates over lambda expressions and registers them with each instance of
  /// a loader in main dispatch loop.
  void addLoaderExtensions(Loader &ld);

private:
  std::vector<std::function<std::unique_ptr<PreProcessInputDataExtension>()>>
      ppInputDataExtensions_;
  std::vector<PostProcessExtFuncPtr> ppOutputDataExtensions_;
  std::vector<std::function<std::unique_ptr<LoaderExtension>()>>
      loaderextensions_;
  std::string appName_;
};

} // namespace glow
#endif // GLOW_TOOLS_LOADER_EXECUTOR_CORE_H
