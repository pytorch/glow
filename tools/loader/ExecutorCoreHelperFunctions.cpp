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

#include "glow/Base/Image.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cfloat>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

#include "ExecutorCoreHelperFunctions.h"

using namespace glow;

/// Image loader options.
llvm::cl::OptionCategory executorCat("Executor Options");
llvm::cl::list<std::string> inputImageFilenames(
    llvm::cl::Positional,
    llvm::cl::desc("<input files> (note: specifying '-' enables streaming "
                   "mode, where the model is compiled once and then can be run "
                   "many times with new input filenames passed via stdin)"),
    llvm::cl::ZeroOrMore);

llvm::cl::list<std::string> inputImageDirs(
    "input-image-dir",
    llvm::cl::desc(
        "Name of directory containing images. Can be used multiple times."),
    llvm::cl::value_desc("dir_name"), llvm::cl::Optional, llvm::cl::ZeroOrMore,
    llvm::cl::cat(executorCat));

llvm::cl::opt<std::string> inputImageListFile(
    "input-image-list-file",
    llvm::cl::desc(
        "Name of the file containing list of images (one image per line)"),
    llvm::cl::value_desc("string_name"), llvm::cl::Optional,
    llvm::cl::cat(executorCat));

llvm::cl::opt<std::string> inputTensorListFile(
    "input-tensor-list-file",
    llvm::cl::desc(
        "Name of the file containing list of tensors (one tensor per line)"),
    llvm::cl::value_desc("string_name"), llvm::cl::Optional,
    llvm::cl::cat(executorCat));

llvm::cl::opt<unsigned> miniBatch(
    "minibatch",
    llvm::cl::desc(
        "Size of mini-batches. Split the input image list into a set of "
        "mini-batches. The input model is compiled for an input tensor batch "
        "size equal to the specified mini-batch size and mini-batches of "
        "images are inferred separately. The number of input images must be a "
        "multiple of the mini-batch size. By default, mini-batch is set to 1."),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(executorCat));

llvm::cl::opt<unsigned> miniBatchThreads(
    "minibatch-threads",
    llvm::cl::desc(
        "Max number of threads used to process mini-batches. If "
        "minibatch-threads is greater than 1, and we are working in minibatch "
        "mode, then several worker threads are created to process the "
        "minibatches. Then the minibatches are distributed between these "
        "threads, and each thread processes its set of minibatches "
        "independently."
        " By default, the number of threads is 1, and no parallelization is "
        "happening. These are things to be aware of:\n"
        "\t- The actual number of worker threads can be less than specified by "
        "this option (for example, if specified number of threads is greater "
        "than number of minibatches to process). Their number may also be "
        "forced to 1 in some cases (see below);\n"
        "\t- Currently, dumping profile and emitting bundle force "
        "single-threaded mode;\n"
        "\t- If a model has operations that make reduction across images in "
        "the batch, it is a user's responsibility to make sure that this model "
        "is  not processed in multi-threaded mode. Otherwise, the correctness "
        "of  results is not guaranteed."),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(executorCat));

llvm::cl::opt<unsigned> poolSize(
    "pool-size",
    llvm::cl::desc("Size of context pool for the benchmark; default:10"),
    llvm::cl::Optional, llvm::cl::init(10), llvm::cl::cat(executorCat));

llvm::cl::opt<bool> convertInAndOutToFp16(
    "convert-inout-to-fp16",
    llvm::cl::desc(
        "Convert the input and output tensors of the network to fp16"),
    llvm::cl::cat(executorCat));

llvm::cl::opt<std::string> tracePath("trace-path",
                                     llvm::cl::desc("Write trace logs to disk"),
                                     llvm::cl::init(""),
                                     llvm::cl::cat(executorCat));

llvm::cl::opt<bool>
    autoInstrument("auto-instrument",
                   llvm::cl::desc("Add instrumentation for operator tracing"),
                   llvm::cl::Optional, llvm::cl::init(false),
                   llvm::cl::cat(executorCat));

llvm::cl::opt<unsigned> traceLevel(
    "trace-level",
    llvm::cl::desc(
        "Set tracing level (bit-field, see TraceEvents.h for details)"),
    llvm::cl::Optional, llvm::cl::init((unsigned)TraceLevel::NONE),
    llvm::cl::cat(executorCat));

llvm::cl::opt<unsigned> warmup(
    "warmup", llvm::cl::desc("How many passes to do to warm everything up"),
    llvm::cl::init(0), llvm::cl::value_desc("W"), llvm::cl::cat(executorCat));

llvm::cl::opt<unsigned> excludedFirstWarmupRuns(
    "excluded-first-warmup-runs",
    llvm::cl::desc("Exclude the time of the given number of first warmup runs "
                   "from the total time"),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(executorCat));

llvm::cl::opt<bool>
    preloadAllImages("preload-all-images",
                     llvm::cl::desc("Pre-load all images before inference"),
                     llvm::cl::init(false), llvm::cl::cat(executorCat));

llvm::cl::opt<unsigned> repeatSingleBatchCount(
    "repeat-single-batch-count",
    llvm::cl::desc(
        "Repeat a single batch input n times. Used for testing purposes. If "
        "used without minibatch then the whole input set is used as the batch "
        "size and repeated n times. Otherwise the first minibatch is repeated "
        "and all other inputs are ignored."),
    llvm::cl::init(0), llvm::cl::cat(executorCat));

/// Read all images from \p inputImageDir into \p inputImageFilenames.
void parseInputDir(const std::string &inputImageDir) {
  CHECK(llvm::sys::fs::is_directory(inputImageDir))
      << strFormat("Path '%s' is not a directory!", inputImageDir.data());
  std::error_code code;
  llvm::sys::fs::directory_iterator dirIt(inputImageDir, code);
  std::vector<std::string> imageFiles;
  while (!code && dirIt != llvm::sys::fs::directory_iterator()) {
    auto path = dirIt->path();
    if (llvm::sys::fs::is_regular_file(path)) {
      imageFiles.emplace_back(path);
    }
    dirIt.increment(code);
  }
  // The paths retrieved by the directory iterator are not sorted.
  // Sort the paths alphabetically in increasing order and add them
  // to the overall list of image filenames.
  std::sort(imageFiles.begin(), imageFiles.end());
  for (auto &imageFile : imageFiles) {
    inputImageFilenames.push_back(imageFile);
  }
}

/// Read all images from \p inputImageListFile in to \p inputImageFilenames.
void parseInputList(const std::string &inputListFile) {
  std::ifstream inFile;
  inFile.open(inputListFile);
  if (!inFile.good()) {
    llvm::outs() << "Could not open input-image-list-file: " << inputListFile
                 << ", exiting.\n";
    std::exit(1);
  }

  while (!inFile.eof()) {
    std::string img;
    getline(inFile, img);
    if (!img.empty()) {
      inputImageFilenames.push_back(img);
    }
  }
  inFile.close();
}

/// Write a prompt to stdout asking for filenames for classification. Read in
/// those filenames and add them to \p filenames. \p filenames is cleared before
/// adding the new set of filenames from stdin. \returns false if the passed in
/// line was empty.
bool getNextImageFilenames(std::vector<std::string> *filenames) {
  // Clear out old filenames before adding new ones.
  filenames->clear();

  llvm::outs() << "Enter image filenames to classify: ";

  // Add in each filename to the vector.
  std::string filenamesRaw;
  getline(std::cin, filenamesRaw);
  std::istringstream iss(filenamesRaw);
  std::string filename;
  while (iss >> filename) {
    filenames->push_back(filename);
  }

  return !filenames->empty();
}

/// Generate in \p imageList the list of filenames corresponding to the next
/// mini-batch of size \p miniBatchSize extracted from \p totalImageList at
/// index \p minibatchIndex. /returns true if the index is valid, false
/// otherwise. In case the function returns true, \p minibatchIndex is
/// incremented by \p miniBatchSize. Stop upon reaching \p miniBatchLimit.
bool getNextMiniBatch(std::vector<std::string> &imageList,
                      std::vector<std::string> &totalImageList,
                      size_t &miniBatchIndex, size_t miniBatchSize,
                      size_t miniBatchLimit) {
  if (miniBatchIndex >= miniBatchLimit) {
    return false;
  }
  imageList.clear();
  size_t endIndex = miniBatchIndex + miniBatchSize;
  for (size_t index = miniBatchIndex; index < endIndex; index++) {
    imageList.push_back(totalImageList[index]);
  }
  miniBatchIndex += miniBatchSize;
  return true;
}

/// Given \p loader, the \p bindings, and \p inputImageType, build the graph
/// from the provided protobuf file found via \p loader. Then compiles and
/// \returns a pair of pointers to the input Placeholder and output Nodes Map.
std::pair<Placeholder *, llvm::StringMap<Placeholder *>>
buildAndCompileAndGetInAndOutPair(Loader &loader, PlaceholderBindings &bindings,
                                  const glow::Type &inputImageType) {
  // Load model.
  loader.loadModel(&inputImageType);

  // Post model loader transformation.
  loader.postModelLoad(bindings, &inputImageType);

  // Allocate tensors to back all inputs and outputs.
  bindings.allocate(loader.getModule()->getPlaceholders());

  // Convert the placeholders for now. The backing Tensor's data will be
  // converted later.
  if (convertInAndOutToFp16) {
    PrecisionConfiguration precConfig;
    TypeAToTypeBFunctionConverter converter(*loader.getFunction(),
                                            ElemKind::FloatTy,
                                            ElemKind::Float16Ty, precConfig);
    for (auto *placeholder : loader.getModule()->getPlaceholders()) {
      converter.convertPlaceholder(*placeholder, &bindings);
    }
  }

  // Compile the model, and perform quantization/emit a bundle/dump debug info
  // if requested from command line.
  CompilationContext cctx = loader.getCompilationContext();
  cctx.bindings = &bindings;
  cctx.backendOpts.autoInstrument = autoInstrument;
  loader.compile(cctx);

  // Get input/output placeholder maps.
  llvm::StringMap<Placeholder *> inpMap = loader.getInputPlaceholderMap();
  llvm::StringMap<Placeholder *> outMap = loader.getOutputPlaceholderMap();

  // Get input placeholder (assumed unique).
  CHECK(inpMap.size() == 1) << "Model is expected to have only 1 input!";
  Placeholder *inputImagePH = inpMap.begin()->second;
  return std::make_pair(inputImagePH, outMap);
}

/// Setup the pool of contexts needed for a benchmark run.
std::vector<std::unique_ptr<ExecutionContext>>
setupContextPool(const std::vector<Placeholder *> outputPHV,
                 Placeholder *inputImagePH, glow::Tensor &inputImageData) {
  std::vector<std::unique_ptr<ExecutionContext>> contexts;
  // Size of the pool, the smaller of poolSize or the actual number of
  // requests.
  unsigned iterations =
      miniBatch ? std::min(int(poolSize), int(iterationsOpt / miniBatch)) : 1;
  // Setup pool of inference requests to be run.
  for (unsigned i = 0; i < iterations; i++) {
    auto newContext = glow::make_unique<ExecutionContext>();
    newContext->setTraceContext(glow::make_unique<TraceContext>(traceLevel));
    auto ph = newContext->getPlaceholderBindings();
    ph->insert(inputImagePH, Tensor(inputImageData.getType()));
    for (auto *outputPH : outputPHV) {
      ph->allocate(outputPH);
    }
    contexts.push_back(std::move(newContext));
  }
  return contexts;
}

std::mutex eventLock;
std::unique_ptr<TraceContext> traceContext;

/// Run inference request on HostManager. This method builds a runNetwork
/// request for the \p hostManager, this is a recursive call, in the callback
/// provided to the HostManager this function can call itself if the desired
/// number of warmups and requests has not yet been dispatched.
static void runInference(runtime::HostManager *hostManager, std::string name,
                         std::unique_ptr<ExecutionContext> batch,
                         std::promise<void> &runPromise,
                         std::atomic<unsigned> &inflight,
                         std::atomic<int> &dispatched, unsigned warmUp,
                         llvm::Timer *restRunsTimer = nullptr,
                         llvm::Timer *firstRunsTimer = nullptr,
                         double *bestRunTime = nullptr) {
  static std::atomic<unsigned> firstRunsDone(0);
  auto start = TraceEvent::now();
  if (firstRunsTimer != nullptr && !firstRunsTimer->isRunning() &&
      firstRunsDone < excludedFirstWarmupRuns) {
    firstRunsTimer->startTimer();
  } else if (restRunsTimer != nullptr &&
             firstRunsDone >= excludedFirstWarmupRuns &&
             !restRunsTimer->hasTriggered()) {
    restRunsTimer->startTimer();
  }

  llvm::Timer *bestRunTimer = nullptr;
  if (bestRunTime != nullptr) {
    bestRunTimer = new llvm::Timer("Best Run", "Best Inference Run");
    bestRunTimer->startTimer();
  }

  hostManager->runNetwork(
      name, std::move(batch),
      [&runPromise, &inflight, &dispatched, hostManager, name, warmUp,
       restRunsTimer, firstRunsTimer, bestRunTime, bestRunTimer,
       start](runtime::RunIdentifierTy, Error err,
              std::unique_ptr<ExecutionContext> contextPtr) {
        EXIT_ON_ERR(std::move(err));
        if (!tracePath.empty()) {
          if (!warmUp) {
            std::lock_guard<std::mutex> l(eventLock);
            // Temporary (AIBench relies on inference_e2e metric)
            // Later we switch AIBench to the metric from
            // HostManager::dispatchNextRun()
            traceContext->logCompleteTraceEvent("inference_e2e",
                                                TraceLevel::RUNTIME, start);
            // Merge this run's TraceEvents into the global
            // TraceContext.
            traceContext->merge(contextPtr->getTraceContext());
          } else {
            contextPtr->getTraceContext()->getTraceEvents().clear();
          }
        }
        firstRunsDone++;
        if (firstRunsTimer != nullptr && firstRunsTimer->isRunning() &&
            firstRunsDone == excludedFirstWarmupRuns) {
          firstRunsTimer->stopTimer();
        }
        if (bestRunTime != nullptr) {
          bestRunTimer->stopTimer();
          double wallTime = bestRunTimer->getTotalTime().getWallTime();
          if (wallTime < *bestRunTime)
            *bestRunTime = wallTime;
          bestRunTimer->clear();
          delete bestRunTimer;
        }

        // Kick off another run.
        if (dispatched.fetch_sub(1) > 0) {
          inflight++;
          runInference(hostManager, name, std::move(contextPtr), runPromise,
                       inflight, dispatched, warmUp > 0 ? warmUp - 1 : 0,
                       restRunsTimer, firstRunsTimer, bestRunTime);
        } else if (restRunsTimer != nullptr) {
          restRunsTimer->stopTimer();
        }

        if (--inflight == 0) {
          runPromise.set_value();
        }
      });
}

/// Run the requested number of benchmark requests \p requestCount prepended by
/// \p warmUp cycles
/// through the HostManager from the \p loader using the provided context pool
/// \p contexts and wait for all runs to complete.
void runBenchmark(std::string name, Loader &loader,
                  std::vector<std::unique_ptr<ExecutionContext>> contexts,
                  unsigned requestCount, unsigned warmUp,
                  llvm::Timer *restRunsTimer = nullptr,
                  llvm::Timer *firstRunsTimer = nullptr,
                  double *bestRunTime = nullptr) {
  runtime::HostManager *hostManager = loader.getHostManager();
  std::atomic<unsigned> inflight(0);
  std::atomic<int> dispatched(requestCount + warmUp * contexts.size());
  std::promise<void> runPromise;
  auto fut = runPromise.get_future();

  // Kick off initial pool of requests.
  for (size_t i = 0, e = contexts.size(); i < e; i++) {
    auto batch = std::move(contexts[i]);
    inflight++;
    dispatched--;
    runInference(hostManager, name, std::move(batch), runPromise, inflight,
                 dispatched, warmUp, restRunsTimer, firstRunsTimer,
                 bestRunTime);
  }

  // Wait for all to finish.
  fut.wait();
}
