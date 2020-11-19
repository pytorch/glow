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

#include "ExecutorCore.h"

#include "ExecutorCoreHelperFunctions.h"
#include "Loader.h"

#include "glow/Base/Image.h"
#include "glow/Base/TensorSerialization.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Optimizer/IROptimizer/CommandLine.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
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

extern llvm::cl::opt<unsigned> traceLevel;

using namespace glow;

namespace {

class PostProcessExecutor : public PostProcessOutputDataExtension {
public:
  /// Iterates over registered extensions for processing and printing results
  /// and executes them.
  /// \return accumulated errors. Value greater then 0 indicates one or more
  /// errros have occured.
  int processOutputs(
      const llvm::StringMap<Placeholder *> &PHM, PlaceholderBindings &bindings,
      llvm::ArrayRef<std::string> inputImageBatchFilenames) override;

  /// Registers Post Processing Output extensions.
  void registerPostProcessOutputExtensions(
      const std::vector<
          std::function<std::unique_ptr<PostProcessOutputDataExtension>()>>
          &extVector);

private:
  std::vector<std::unique_ptr<PostProcessOutputDataExtension>> extensions_;
};

class PreProcessInputExecutor : public PreProcessInputDataExtension {
public:
  /// Iterates over PreProcessInputDataExtension extensions and executes them
  /// one by ene.
  void processInputTensor(Tensor &inputImageData, size_t startId, size_t endId,
                          size_t batchSz) override;

  /// Registers Input Data Preprocessing Extensions.
  void registerInputDataPreProcessingExtension(
      const std::vector<
          std::function<std::unique_ptr<PreProcessInputDataExtension>()>>
          &extVector);

private:
  std::vector<std::unique_ptr<PreProcessInputDataExtension>> extensions_;
};

void PostProcessExecutor::registerPostProcessOutputExtensions(
    const std::vector<
        std::function<std::unique_ptr<PostProcessOutputDataExtension>()>>
        &extVector) {
  for (auto &f : extVector) {
    extensions_.push_back(f());
  }
}

} // namespace

/// Iterates over registered extensions for processing and Printing results
/// and executes them.
int PostProcessExecutor::processOutputs(
    const llvm::StringMap<Placeholder *> &PHM, PlaceholderBindings &bindings,
    llvm::ArrayRef<std::string> inputImageBatchFilenames) {
  int numErrors = 0;
  for (auto &f : extensions_) {
    numErrors += f->processOutputs(PHM, bindings, inputImageBatchFilenames);
  }
  return numErrors;
}

/// Iterates over PreProcessInputDataExtension extensions and executes them one
/// by ene.
void PreProcessInputExecutor::processInputTensor(Tensor &inputImageData,
                                                 size_t startId, size_t endId,
                                                 size_t batchSz) {
  for (auto &f : extensions_) {
    f->processInputTensor(inputImageData, startId, endId, batchSz);
  }
}

void PreProcessInputExecutor::registerInputDataPreProcessingExtension(
    const std::vector<
        std::function<std::unique_ptr<PreProcessInputDataExtension>()>>
        &extVector) {
  for (auto &f : extVector) {
    extensions_.push_back(f());
  }
}

Executor::Executor(std::string appName, int argc, char **argv) {
  appName_ = appName;
  // Verify/initialize command line parameters, and then loader initializes
  // the ExecutionEngine and Function.
  parseCommandLine(argc, argv);

  for (const auto &inputImageDir : inputImageDirs) {
    parseInputDir(inputImageDir);
  }

  if (!inputImageListFile.empty()) {
    CHECK_EQ(inputImageFilenames.size(), 0)
        << "When using -input-image-list-file all Input images must be "
           "specified "
           "using -input-image-list-file option.";
    parseInputList(inputImageListFile);
  }
}

/// Registers a Loader Extension that will be invoked after model is loaded.
/// If multiple extensions are registered they will be executed in order they
/// were registered.
void Executor::registerLoaderExtension(
    std::function<std::unique_ptr<LoaderExtension>()> func) {
  loaderextensions_.push_back(func);
}

/// Registers an extension that will be invoked on Tensor containing current
/// batch of input data. If multiple extensions are registered they will be
/// executed in order they were registered.
void Executor::registerInputDataPreProcessingExtension(
    std::function<std::unique_ptr<PreProcessInputDataExtension>()> func) {
  ppInputDataExtensions_.push_back(func);
}

/// Registers extension that will be invoked for each execution of the
/// network. If multiple extensions are registered they will be executed in
/// order they were registered.
void Executor::registerPostProcessOutputExtension(
    std::function<std::unique_ptr<PostProcessOutputDataExtension>()> func) {
  ppOutputDataExtensions_.push_back(func);
}

/// Iterates over lambda expressions and registers them with each instance of a
/// loader in main dispatch loop.
void Executor::addLoaderExtensions(Loader &ld) {
  for (auto &f : loaderextensions_) {
    ld.registerExtension(f());
  }
}

/// This will parse command line, load, build and execute a network.
int Executor::executeNetwork() {

  if (inputImageListFile.empty() && inputTensorListFile.empty() &&
      inputImageFilenames.size() == 0) {
    llvm::errs() << "Args: Either positional inputImageFilenames or "
                    "-inputImageListFile or "
                    "-inputTensorListFile "
                    "must be used to specify input images.\n";
    return 1;
  }

  if (!inputTensorListFile.empty()) {
    CHECK_EQ(inputImageFilenames.size(), 0)
        << "When using -input-tensor-list-file all Input images must be "
           "specified "
           "using -input-tensor-list-file option.";
    parseInputList(inputTensorListFile);
  }

  if (excludedFirstWarmupRuns && excludedFirstWarmupRuns >= warmup) {
    llvm::errs() << "Excluding all warmup runs does not make sense\n";
    return 1;
  }
  // Stream input mode.
  const bool streamInputFilenamesMode =
      inputImageFilenames.size() == 1 && inputImageFilenames.front() == "-";

  CHECK(!(streamInputFilenamesMode && emittingBundle()))
      << "Cannot emit a bundle and also stream inputs.";

  // If tracing is enabled, create a TraceContext to merge each runs events
  // into.
  if (!tracePath.empty()) {
    traceContext = glow::make_unique<TraceContext>(TraceLevel::STANDARD);
  }

  // Mini-batch mode.
  const bool miniBatchMode = miniBatch > 0;
  CHECK(((!miniBatchMode) || (!streamInputFilenamesMode)))
      << "The minibatch option is not compatible with the stream input "
         "image mode.";
  CHECK(((!miniBatchMode) || (inputImageFilenames.size() % miniBatch == 0)))
      << "The number of input images must be a multiple of the mini-batch.";

  CHECK(((!iterationsOpt) || (!miniBatchMode) ||
         (iterationsOpt % miniBatch == 0)))
      << "Benchmark count must be a multiple of the mini-batch.";
  CHECK(!preloadAllImages || miniBatchMode)
      << "preload-all-images can only be used with minibatch";

  const bool singleBatchRepeatedMode = repeatSingleBatchCount > 0;
  CHECK(!(streamInputFilenamesMode && singleBatchRepeatedMode))
      << "singleBatchRepeatedMode is not compatible with "
         "streamInputFilenamesMode";

  // When the mini-batch mode is enabled do not allow debug instrumentation.
  if (miniBatchMode) {
    CHECK(!instrumentDebug)
        << "The minibatch option is not compatible with debug instrumentation.";
  }

  // Print out the inferred image classification.
  llvm::outs() << "Model: " << Loader::getModelOptPath() << "\n";
  std::mutex ioMu;
  int numErrors = 0;

  if (runAllInputsOnAllDevices) {
    if (numDevices != miniBatchThreads) {
      llvm::outs() << "Setting " << miniBatchThreads.ArgStr << " to match "
                   << numDevices.ArgStr << " (" << numDevices
                   << ") as required by " << runAllInputsOnAllDevices.ArgStr
                   << "\n";
      miniBatchThreads.getValue() = numDevices;
    }
  }

  // If preloading then load+process all images here in preloadedInputImageData.
  Tensor preloadedInputImageData;
  if (preloadAllImages) {
    Loader loader;
    PreProcessInputExecutor ppImageExecutor;
    addLoaderExtensions(loader);
    ppImageExecutor.registerInputDataPreProcessingExtension(
        ppInputDataExtensions_);

    if (!inputTensorListFile.empty()) {
      loadInputImageFromFileWithType(inputImageFilenames,
                                     &preloadedInputImageData, imageLayout);
    } else {
      loadImagesAndPreprocess(inputImageFilenames, &preloadedInputImageData,
                              imageNormMode, imageChannelOrder, imageLayout);

      ppImageExecutor.processInputTensor(preloadedInputImageData, 0,
                                         inputImageFilenames.size(),
                                         preloadedInputImageData.dims()[0]);
    }
  }

  // Process a set of minibatches with indices [startIndex, endIndex).
  auto processImageRange = [&](size_t startIndex, size_t endIndex, size_t TID) {
    std::unique_ptr<ExecutionContext> exContext =
        glow::make_unique<ExecutionContext>();
    PlaceholderBindings &bindings = *exContext->getPlaceholderBindings();
    if (traceContext) {
      exContext->setTraceContext(
          glow::make_unique<TraceContext>(TraceLevel::STANDARD));
    }
    // If runAllInputsOnAllDevices, then assign this thread with TID to device
    // TID. E.g. if this is TID 2 then this will be assigned to device 2.
    Loader loader = runAllInputsOnAllDevices ? Loader(TID) : Loader();
    PostProcessExecutor ppResultExecutor;
    PreProcessInputExecutor ppImageExecutor;

    // Registering all the extensions per thread.
    addLoaderExtensions(loader);
    ppResultExecutor.registerPostProcessOutputExtensions(
        ppOutputDataExtensions_);
    ppImageExecutor.registerInputDataPreProcessingExtension(
        ppInputDataExtensions_);

    // Used to make sure we only compile once, and run only once if not
    // streaming.
    bool isFirstRun = true;

    // These will be set during the first run.
    Placeholder *inputImagePH = nullptr;
    std::vector<Placeholder *> outputPHV;
    llvm::StringMap<Placeholder *> PHM;

    size_t miniBatchIndex = startIndex;
    Tensor inputImageData;
    if (preloadAllImages) {
      inputImageData = preloadedInputImageData.getUnowned();
    }
    std::vector<std::string> inputImageBatchFilenames;
    if (!miniBatchMode && !streamInputFilenamesMode) {
      inputImageBatchFilenames = inputImageFilenames;
    } else if (singleBatchRepeatedMode) {
      inputImageBatchFilenames =
          miniBatchMode ? std::vector<std::string>(inputImageFilenames.begin(),
                                                   inputImageFilenames.begin() +
                                                       miniBatch)
                        : inputImageFilenames;
    }
    if (!tracePath.empty()) {
      loader.getHostManager()->setTraceContext(
          glow::make_unique<TraceContext>(traceLevel));
      Error err = loader.getHostManager()->startDeviceTrace();
      if (err) {
        LOG(INFO) << "Failed to start device trace.";
        numErrors = 1;
        return;
      } else {
        llvm::outs() << "Device trace started.";
      }
    }

    unsigned repeatedLoopCountRemaining = repeatSingleBatchCount;

    auto loopCond = [&]() {
      // If in stream mode then get the next image filenames if they exist,
      // otherwise exit.
      if (streamInputFilenamesMode) {
        return getNextImageFilenames(&inputImageBatchFilenames);
      }

      // If a single batch is going to be loaded once and repeated then keep
      // running repeatedLoopCountRemaining mores times.
      if (singleBatchRepeatedMode) {
        return repeatedLoopCountRemaining-- != 0;
      }

      // If in miniBatchMode then continue if we have already preloaded all
      // images (will break inside loop once done), or otherwise get the next
      // miniBatch image filenames if they exist, otherwise exit.
      if (miniBatchMode) {
        return getNextMiniBatch(inputImageBatchFilenames, inputImageFilenames,
                                miniBatchIndex, miniBatch, endIndex);
      }

      // At least enter once, e.g. to just dump a bundle.
      return isFirstRun;
    };

    while (loopCond()) {
      if (!preloadAllImages && (!singleBatchRepeatedMode || isFirstRun)) {
        // Load and process the image data into the inputImageData Tensor.
        if (!inputTensorListFile.empty()) {
          loadInputImageFromFileWithType(inputImageBatchFilenames,
                                         &inputImageData, imageLayout);
        } else {
          loadImagesAndPreprocess(inputImageBatchFilenames, &inputImageData,
                                  imageNormMode, imageChannelOrder,
                                  imageLayout);

          ppImageExecutor.processInputTensor(
              inputImageData, startIndex, endIndex, inputImageData.dims()[0]);
        }
      }

      // Note: At this point miniBatchIndex is the end index, so subtract
      // miniBatch to get the start index.
      const dim_t startMiniBatchIndex = miniBatchIndex - miniBatch;

      ShapeVector imageShape(inputImageData.getType().dims().begin(),
                             inputImageData.getType().dims().end());
      if (miniBatch) {
        imageShape[0] = miniBatch;
      } else if (iterationsOpt) {
        imageShape[0] = iterationsOpt;
      }

      // If we are benchmarking reset the image data to the batch size we need.
      if (iterationsOpt) {
        // Resize the Tensor to the appropriate size.
        inputImageData.reset(ElemKind::FloatTy, imageShape);
      }

      // If this is the first run, then we need to build and compile the model.
      if (isFirstRun) {
        isFirstRun = false;

        // Build and compile the graph, and then get back the input Placeholder
        // and output Placeholder.
        std::pair<Placeholder *, llvm::StringMap<Placeholder *>>
            inputOutputPair = buildAndCompileAndGetInAndOutPair(
                loader, bindings,
                preloadAllImages
                    ? Type::newShape(inputImageData.getType(), imageShape)
                    : inputImageData.getType());

        // If in bundle mode, the bundle has been saved by the above call, so we
        // can safely return.
        if (emittingBundle()) {
          LOG(INFO) << "Emit bndle mode is on. Network is compiled only.";
          return;
        }

        inputImagePH = inputOutputPair.first;
        PHM = inputOutputPair.second;
        for (auto phI = PHM.begin(), e = PHM.end(); phI != e; ++phI) {
          CHECK(phI->second) << "Placeholder in output map is NULL.";
          outputPHV.push_back(phI->second);
        }
      }

      Tensor inputImageDataBatch = inputImageData.getUnowned(
          imageShape, {preloadAllImages ? startMiniBatchIndex : 0, 0, 0, 0});

      CHECK(inputImagePH) << "Input must be valid.";
      CHECK(!PHM.empty()) << "Output must be valid.";
      CHECK(inputImagePH->dims() == inputImageDataBatch.dims())
          << "New input shape does not match the compiled function: "
          << inputImagePH->dims() << " vs " << inputImageDataBatch.dims();

      // Convert the raw input to fp16. This must be done every time we get new
      // image data.
      if (convertInAndOutToFp16) {
        inputImageDataBatch.convertToType(ElemKind::Float16Ty);
      }

      // If we are benchmarking we are done with the while loop.
      if (iterationsOpt) {
        break;
      }

      // Minibatch inference initialization of loader extensions
      loader.inferInitMiniBatch(bindings, startMiniBatchIndex, miniBatch);

      // About to run inference, so update the input image Placeholder's backing
      // Tensor with inputImageDataBatch.
      updateInputPlaceholders(bindings, {inputImagePH}, {&inputImageDataBatch});

      // Perform the inference execution, updating output tensors.
      auto batchSize = inputImageDataBatch.dims()[0];
      loader.runInference(exContext.get(), batchSize);
      if (traceContext) {
        traceContext->merge(exContext->getTraceContext());
      }

      // Process output of the network. Each app cand do its own post-processing
      // depending on type of the network.
      {
        std::lock_guard<std::mutex> lock(ioMu);
        numErrors += ppResultExecutor.processOutputs(PHM, bindings,
                                                     inputImageBatchFilenames);
      }

      // Minibatch inference initialization of loader extensions.
      loader.inferEndMiniBatch(bindings, startMiniBatchIndex, miniBatch);
    }

    if (iterationsOpt) {
      // Image tensors loaded up to be run at once for benchmark mode.
      std::vector<std::unique_ptr<ExecutionContext>> contexts =
          setupContextPool(outputPHV, inputImagePH, inputImageData);

      std::string name = loader.getFunctionName();
      std::unique_ptr<llvm::Timer> restRunsTimer = nullptr;
      std::unique_ptr<llvm::Timer> firstRunsTimer = nullptr;
      std::unique_ptr<double> bestRunTime = nullptr;
      if (timeOpt) {
        if (excludedFirstWarmupRuns) {
          firstRunsTimer.reset(
              new llvm::Timer("First Runs", "First inference runs"));
          restRunsTimer.reset(
              new llvm::Timer("Rest Inferences", "Rest of the inference runs"));
        } else {
          restRunsTimer.reset(
              new llvm::Timer("Inferences", "All inference runs"));
        }
        bestRunTime.reset(new double);
        *bestRunTime = DBL_MAX;
      }
      unsigned requestCount = miniBatch ? iterationsOpt / miniBatch : 1;

      runBenchmark(name, loader, std::move(contexts), requestCount, warmup,
                   restRunsTimer.get(), firstRunsTimer.get(),
                   bestRunTime.get());
      if (timeOpt) {
        double wallTime = restRunsTimer->getTotalTime().getWallTime();
        llvm::outs() << llvm::formatv(
            "Average wall time per item (s): {0:f4}\n",
            wallTime / (iterationsOpt + warmup - excludedFirstWarmupRuns));
        llvm::outs() << llvm::formatv(
            "            Best wall time (s): {0:f4}\n", *bestRunTime);
      }
    }

    // If profiling, generate and serialize the profiling infos now that we
    // have run inference one or more times to gather the profile.
    if (profilingGraph()) {
      loader.generateAndSerializeProfilingInfos(bindings);
    }
    if (!tracePath.empty()) {
      Error err = loader.getHostManager()->stopDeviceTrace();
      if (err) {
        LOG(INFO) << "Failed to stop device trace:";
        numErrors = 1;
        return;
      } else {
        traceContext->merge(loader.getHostManager()->getTraceContext());
      }
    }
  };

  // We will force single-threaded execution if:
  // - Minibatch mode and runAllInputsOnAllDevices are disabled;
  // - We are going to emit bundle and do not do inference;
  // - We are collecting inference profile.
  // Otherwise, there can be several minibatches of equal size.
  const bool multiThreadingAllowed =
      (runAllInputsOnAllDevices || miniBatchMode) && !emittingBundle() &&
      !profilingGraph();
  const size_t numBatches =
      miniBatchMode ? inputImageFilenames.size() / miniBatch : 1u;
  const size_t numThreads =
      runAllInputsOnAllDevices
          ? miniBatchThreads
          : (multiThreadingAllowed
                 ? std::min(size_t(miniBatchThreads), numBatches)
                 : 1u);
  if (miniBatchThreads > 1 && !multiThreadingAllowed) {
    llvm::outs() << "WARNING: multi-threaded execution is not possible. Make "
                    "sure that minibatch size is specified and you are not "
                    "trying to dump profile or emit bundle.\n";
  }

  llvm::outs() << "Running " << numThreads << " thread(s).\n";
  std::vector<std::thread> threads(numThreads);
  const size_t miniBatchesPerThread =
      (numBatches + numThreads - 1) / numThreads;
  for (size_t i = 0; i < numThreads; i++) {
    size_t startIndex, endIndex;
    if (!runAllInputsOnAllDevices && numThreads > 1) {
      startIndex = i * miniBatchesPerThread * miniBatch;
      endIndex = std::min((i + 1) * miniBatchesPerThread * miniBatch,
                          inputImageFilenames.size());
    } else {
      startIndex = 0;
      endIndex = inputImageFilenames.size();
    }
    auto worker = [&processImageRange, startIndex, endIndex, i]() {
      processImageRange(startIndex, endIndex, i);
    };
    threads.push_back(std::thread(worker));
  }

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  if (!tracePath.empty()) {
    traceContext->dump(tracePath, appName_);
  }

  return numErrors;
}
