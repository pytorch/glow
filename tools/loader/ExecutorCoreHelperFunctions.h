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

#ifndef GLOW_TOOLS_LOADER_EXECUTOR_CORE_HELPER_FUNCTIONS_H
#define GLOW_TOOLS_LOADER_EXECUTOR_CORE_HELPER_FUNCTIONS_H

#include "Loader.h"
#include "glow/Graph/Nodes.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Timer.h"

extern llvm::cl::list<std::string> inputImageFilenames;
extern llvm::cl::list<std::string> inputImageDirs;
extern llvm::cl::opt<std::string> inputImageListFile;
extern llvm::cl::opt<std::string> inputTensorListFile;
extern llvm::cl::opt<unsigned> excludedFirstWarmupRuns;
extern llvm::cl::opt<unsigned> warmup;
extern llvm::cl::opt<std::string> tracePath;
extern llvm::cl::opt<bool> convertInAndOutToFp16;
extern llvm::cl::opt<unsigned> miniBatch;
extern llvm::cl::opt<unsigned> miniBatchThreads;
extern llvm::cl::opt<bool> preloadAllImages;
extern llvm::cl::opt<unsigned> repeatSingleBatchCount;

extern std::unique_ptr<glow::TraceContext> traceContext;

/// Read all images from \p inputImageDir into \p inputImageFilenames.
void parseInputDir(const std::string &inputImageDir);

/// Read all images from \p inputImageListFile in to \p inputImageFilenames.
void parseInputList(const std::string &inputImageListFile);

/// Write a prompt to stdout asking for filenames for classification. Read in
/// those filenames and add them to \p filenames. \p filenames is cleared before
/// adding the new set of filenames from stdin. \returns false if the passed in
/// line was empty.
bool getNextImageFilenames(std::vector<std::string> *filenames);

/// Generate in \p imageList the list of filenames corresponding to the next
/// mini-batch of size \p miniBatchSize extracted from \p totalImageList at
/// index \p minibatchIndex. /returns true if the index is valid, false
/// otherwise. In case the function returns true, \p minibatchIndex is
/// incremented by \p miniBatchSize. Stop upon reaching \p miniBatchLimit.
bool getNextMiniBatch(std::vector<std::string> &imageList,
                      std::vector<std::string> &totalImageList,
                      size_t &miniBatchIndex, size_t miniBatchSize,
                      size_t miniBatchLimit);

/// Given \p loader, the \p bindings, and \p inputImageType, build the graph
/// from the provided protobuf file found via \p loader. Then compiles and
/// \returns a pair of pointers to the input Placeholder and output Nodes Map.
std::pair<glow::Placeholder *, llvm::StringMap<glow::Placeholder *>>
buildAndCompileAndGetInAndOutPair(glow::Loader &loader,
                                  glow::PlaceholderBindings &bindings,
                                  const glow::Type &inputImageType);

/// Setup the pool of contexts needed for a benchmark run.
std::vector<std::unique_ptr<glow::ExecutionContext>>
setupContextPool(const std::vector<glow::Placeholder *> outputPHV,
                 glow::Placeholder *inputImagePH, glow::Tensor &inputImageData);

/// Run the requested number of benchmark requests \p requestCount prepended by
/// \p warmUp cycles
/// through the HostManager from the \p loader using the provided context pool
/// \p contexts and wait for all runs to complete.
void runBenchmark(std::string name, glow::Loader &loader,
                  std::vector<std::unique_ptr<glow::ExecutionContext>> contexts,
                  unsigned requestCount, unsigned warmUp,
                  llvm::Timer *restRunsTimer, llvm::Timer *firstRunsTimer,
                  double *bestRunTime);
#endif // GLOW_TOOLS_LOADER_EXECUTOR_CORE_HELPER_FUNCTIONS_H
