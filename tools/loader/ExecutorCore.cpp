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
#include "glow/Support/Support.h"

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
  int processOutputs(const llvm::StringMap<Placeholder *> &PHM,
                     PlaceholderBindings &bindings,
                     VecVecRef<std::string> inputImageBatchFilenames) override;

  /// Registers Post Processing Output extensions.
  void registerPostProcessOutputExtensions(
      const std::vector<PostProcessExtFuncPtr> &extVector);

private:
  UniquePtrVec<PostProcessOutputDataExtension> extensions_;
};

class PreProcessInputExecutor : public PreProcessInputDataExtension {
public:
  /// Iterates over PreProcessInputDataExtension extensions and executes them
  /// one by one.
  void processInputTensor(llvm::ArrayRef<Tensor *> inputImageData,
                          size_t startId, size_t endId,
                          size_t batchSz) override;

  /// Registers Input Data Preprocessing Extensions.
  void registerInputDataPreProcessingExtension(
      const std::vector<
          std::function<std::unique_ptr<PreProcessInputDataExtension>()>>
          &extVector);

private:
  UniquePtrVec<PreProcessInputDataExtension> extensions_;
};

void PostProcessExecutor::registerPostProcessOutputExtensions(
    const std::vector<PostProcessExtFuncPtr> &extVector) {
  for (auto &f : extVector) {
    extensions_.push_back(f());
  }
}

} // namespace

/// Iterates over registered extensions for processing and Printing results and
/// executes them.
int PostProcessExecutor::processOutputs(
    const llvm::StringMap<Placeholder *> &PHM, PlaceholderBindings &bindings,
    VecVecRef<std::string> inputImageBatchFilenames) {
  int numErrors = 0;
  for (auto &f : extensions_) {
    numErrors += f->processOutputs(PHM, bindings, inputImageBatchFilenames);
  }
  return numErrors;
}

/// Iterates over PreProcessInputDataExtension extensions and execute them one
/// by one.
void PreProcessInputExecutor::processInputTensor(
    llvm::ArrayRef<Tensor *> inputImageData, size_t startId, size_t endId,
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
  // Clear all external storage for command args set variables. This is
  // necessary in order to support multiple calls to parse the command
  // line; it seems that clearing the command line options is not possible,
  // thus, we clear their external storage only. With each successive
  // call to parse the arguments, arguments are pilling up in the ::cl
  // argument, however, external storage will be set by the arguments from the
  // current call only.
  // NOTE: llvm::cl::ResetAllOptionOccurrences() or opt.reset() should do the
  // job but they don't work.
  // TODO: Loader should provide function to register callbacks.
  initExecutorCoreCmdArgVars();
  initImageCmdArgVars();
  // Verify/initialize command line parameters, and then loader initializes
  // the ExecutionEngine and Function.

  parseCommandLine(argc, argv);
  processImageCmdArgVars(modelInputsOpt.size());
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
void Executor::registerPostProcessOutputExtension(PostProcessExtFuncPtr func) {
  ppOutputDataExtensions_.push_back(func);
}

/// Iterates over lambda expressions and registers them with each instance of a
/// loader in main dispatch loop.
void Executor::addLoaderExtensions(Loader &ld) {
  for (auto &f : loaderextensions_) {
    ld.registerExtension(f());
  }
}

void parseInputFiles(VecVec<std::string> &inputImageFiles) {
  if (inputImageListFileOpt.empty() && inputImageDirs.empty() &&
      inputTensorListFile.empty() && inputImageFilenamesOpt.size() == 0) {
    llvm::errs() << "Args: Either positional image list or "
                    "-input-image-dir or "
                    "-input-image-list-file or "
                    "-input-tensor-list-file  "
                    "must be used to specify input images.\n";
    return;
  }

  if (!inputImageDirs.empty() &&
      (!inputImageListFileOpt.empty() || inputImageFilenamesOpt.size() != 0)) {
    LOG(FATAL) << "Args: Specifying image using input-image-dir cannot be "
                  "combined with "
                  "-input-image-list-file or the positional image list.\n";
  }

  if (!inputImageListFileOpt.empty() && inputImageFilenamesOpt.size() != 0) {
    LOG(FATAL) << "Args: positional image list cannot be combined with "
                  "-input-image-list-file to specify input images.\n";
  }

  int32_t numInputNames = modelInputsOpt.size();

  // if positional list of images, we support one input only. Assign 1st input
  // vector list.
  if (inputImageFilenamesOpt.size() != 0) {
    CHECK_EQ(numInputNames, 1) << "When using positional image list, single "
                                  "input networks are supported only.";
    inputImageFiles.push_back(inputImageFilenamesOpt);
    return;
  }

  if (!inputTensorListFile.empty()) {
    CHECK_EQ(inputImageFilenamesOpt.size(), 0)
        << "When using -input-tensor-list-file all Input images must be "
           "specified "
           "using -input-tensor-list-file option.";
    CHECK_EQ(inputImageListFileOpt.size(), 0)
        << "When using -input-tensor-list-file all Input images must be "
           "specified "
           "using -input-tensor-list-file option.";
    CHECK_EQ(numInputNames, 1) << "When using -input-tensor-list-file single "
                                  "input networks are supported only.";
    std::vector<std::string> imageFiles;
    parseInputList(inputTensorListFile, imageFiles);
    inputImageFiles.push_back(imageFiles);
    return;
  }

  if (!inputImageDirs.empty()) {
    CHECK_EQ(numInputNames, 1)
        << "When using image dir. single input networks are supported only.";
    for (const auto &inputImageDir : inputImageDirs) {
      std::vector<std::string> imageFiles;
      parseInputDir(inputImageDir, imageFiles);
      inputImageFiles.push_back(imageFiles);
    }
    return;
  }

  // If images are given using vector of lists of images
  CHECK_EQ(numInputNames, inputImageListFileOpt.size())
      << "Args: number of inputs and number of inputs image lists must match.";

  size_t numInputImages = 0;
  for (int i = 0; i < numInputNames; i++) {
    std::vector<std::string> imageFiles;
    parseInputList(inputImageListFileOpt[i], imageFiles);
    inputImageFiles.push_back(imageFiles);
    if (i > 0) {
      CHECK_EQ(numInputImages, inputImageFiles[i].size())
          << "Each image list file should have the same number of images.";
    } else {
      numInputImages = inputImageFiles[i].size();
    }
  }
}

/// This will parse command line, load, build and execute a network.
int Executor::executeNetwork() {

  parseInputFiles(inputImageFilenames_);

  if (excludedFirstWarmupRuns && excludedFirstWarmupRuns >= warmup) {
    llvm::errs() << "Excluding all warmup runs does not make sense\n";
    return 1;
  }
  // Stream input mode.
  const bool streamInputFilenamesMode = inputImageFilenamesOpt.size() == 1 &&
                                        inputImageFilenamesOpt.front() == "-";

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
  CHECK(((!miniBatchMode) || (inputImageFilenames_[0].size() % miniBatch == 0)))
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

  CHECK(!preloadAllImages || (modelInputsOpt.size() == 1))
      << "Preloading all images doesn't support networks with multiple inputs.";

  CHECK(!iterationsOpt || (modelInputsOpt.size() == 1))
      << "Benchmark mode doesn't support networks with multiple inputs.";

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
      loadInputImageFromFileWithType(
          inputImageFilenames_[0], &preloadedInputImageData, imageLayoutOpt[0]);
    } else {
      // Load and process the image data into the inputImageData Tensor.
      loadImagesAndPreprocess(inputImageFilenames_, {&preloadedInputImageData});
      ppImageExecutor.processInputTensor({&preloadedInputImageData}, 0,
                                         inputImageFilenames_[0].size(),
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

    // Perform graph profiling initialization if needed.
    // if (profilingGraph()) {
    //  loader.initGraphProfiling(
    //      bindings, miniBatch > 0 ? miniBatch :
    //      inputImageFilenames_[0].size(), inputImageFilenames_[0].size());
    //}

    // These will be set during the first run.
    llvm::StringMap<Placeholder *> iPHM;
    llvm::StringMap<Placeholder *> oPHM;
    std::vector<Placeholder *> inPHs;
    std::vector<Placeholder *> outPHs;

    size_t miniBatchIndex = startIndex;
    std::vector<Tensor> inputData(modelInputsOpt.size());
    if (preloadAllImages) {
      inputData[0] = preloadedInputImageData.getUnowned();
    }

    VecVec<std::string> inputImageBatchFilenames;
    if ((!miniBatchMode) &&
        (!streamInputFilenamesMode || singleBatchRepeatedMode)) {
      inputImageBatchFilenames = inputImageFilenames_;
    } else if (singleBatchRepeatedMode) {
      for (size_t i = 0, e = modelInputsOpt.size(); i < e; i++) {
        std::vector<std::string> names(inputImageFilenames_[0].begin(),
                                       inputImageFilenames_[0].begin() +
                                           miniBatch);
        inputImageBatchFilenames.push_back(names);
      }
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

    // Pass input tensors around as array of pointers.
    std::vector<Tensor *> inputImageData;
    for (auto &data : inputData) {
      inputImageData.push_back(&data);
    }

    unsigned repeatedLoopCountRemaining = repeatSingleBatchCount;

    auto loopCond = [&]() {
      // If in stream mode then get the next image filenames if they exist,
      // otherwise exit.
      if (streamInputFilenamesMode) {
        return getNextStdinImageFilenames(inputImageBatchFilenames);
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
        return getNextMiniBatch(inputImageBatchFilenames, inputImageFilenames_,
                                miniBatchIndex, miniBatch, endIndex);
      }

      // At least enter once, e.g. to just dump a bundle.
      return isFirstRun;
    };

    while (loopCond()) {
      if (!preloadAllImages && (!singleBatchRepeatedMode || isFirstRun)) {
        // Load and process the image data into the inputImageData Tensor.
        if (!inputTensorListFile.empty()) {
          loadInputImageFromFileWithType(inputImageBatchFilenames[0],
                                         inputImageData[0], imageLayoutOpt[0]);
        } else {
          loadImagesAndPreprocess(inputImageBatchFilenames, inputImageData);
          ppImageExecutor.processInputTensor(inputImageData, startIndex,
                                             endIndex,
                                             inputImageData[0]->dims()[0]);
        }
      }

      // Note: At this point miniBatchIndex is the end index, so subtract
      // miniBatch to get the start index.
      const dim_t startMiniBatchIndex = miniBatchIndex - miniBatch;

      ShapeVector imageShape(inputImageData[0]->getType().dims().begin(),
                             inputImageData[0]->getType().dims().end());
      if (miniBatch) {
        imageShape[0] = miniBatch;
      } else if (iterationsOpt) {
        imageShape[0] = iterationsOpt;
      }

      // If we are benchmarking reset the image data to the batch size we need.
      if (iterationsOpt) {
        auto resetTensor = [](Tensor *tensor) {
          ShapeVector imageSize(tensor->getType().dims().begin(),
                                tensor->getType().dims().end());
          imageSize[0] = miniBatch ? miniBatch : iterationsOpt;
          tensor->reset(ElemKind::FloatTy, imageSize);
        };
        std::for_each(inputImageData.begin(), inputImageData.end(),
                      resetTensor);
      }

      // If this is the first run, then we need to build and compile the model.
      if (isFirstRun) {
        isFirstRun = false;

        std::vector<TypeRef> types;
        auto preloadTy =
            Type::newShape(inputImageData[0]->getType(), imageShape);

        if (preloadAllImages) {
          types.push_back(&preloadTy);
        } else {
          // get types of all input tensors.
          for_each(inputImageData.begin(), inputImageData.end(),
                   [&](auto *t) { types.push_back(&t->getType()); });
        }

        // Build and compile the graph, then get input and output Placeholders.
        std::tie(iPHM, oPHM) =
            buildAndCompileAndGetInAndOutPair(loader, bindings, types);

        // If in bundle mode, the bundle has been saved by the above call, so we
        // can safely return.
        if (emittingBundle()) {
          LOG(INFO) << "Emit bundle mode is on. Network is compiled only.";
          return;
        }

        // Obtain input/output placeholders from input/output map.
        // For inputs, we got map but need to convert to array - need to
        // take from map in order specified by modelInputsOpt.
        for (size_t i = 0, e = modelInputsOpt.size(); i < e; i++) {
          auto it = iPHM.find(modelInputsOpt[i]);
          CHECK(it != iPHM.end())
              << "Couldn't find placeholder: " << modelInputsOpt[i];
          CHECK((*it).second) << "Placeholder in input map is NULL.";
          inPHs.push_back((*it).second);
        };
        for_each(oPHM.begin(), oPHM.end(), [&](auto &p) {
          CHECK(p.second) << "Placeholder in output map is NULL.";
          outPHs.push_back(p.second);
        });
      }

      // preloadAllImages - set a new Tensor that takes a slice from the 1st
      // (and only) input tensor. Assign this new Tensor the tensor array of
      // pointers, inputImageData, used further.
      Tensor inputImageDataBatch;
      if (preloadAllImages) {
        std::vector<dim_t> imgSliceStart(imageShape.size(), 0);
        imgSliceStart[0] = startMiniBatchIndex;
        inputImageDataBatch =
            inputImageData[0]->getUnowned(imageShape, imgSliceStart);
        inputImageData[0] = &inputImageDataBatch;
      }

      // Compile done.
      CHECK(!inPHs.empty()) << "Input must be valid.";
      CHECK(!outPHs.empty()) << "Output must be valid.";
      CHECK_EQ(inPHs.size(), inputImageData.size())
          << "Number of input placeholders and tensors must match";
      for (size_t i = 0, e = inputImageData.size(); i < e; i++) {
        CHECK(inPHs[i]->dims() == inputImageData[i]->dims())
            << "New input shape does not match the compiled function: "
            << inPHs[i]->dims() << " vs " << inputImageData[i]->dims();
      }

      // Convert the raw input to fp16. This must be done every time we get new
      // image data.
      // Convert the raw input to fp16.
      if (convertInAndOutToFp16) {
        for (auto &t : inputImageData) {
          t->convertToType(ElemKind::Float16Ty);
        }
      }

      // If we are benchmarking we are done with the while loop.
      if (iterationsOpt) {
        break;
      }

      // Minibatch inference initialization of loader extensions
      loader.inferInitMiniBatch(bindings, startMiniBatchIndex, miniBatch);

      // About to run inference, so update the input image Placeholder's backing
      // Tensor with inputImageDataBatch.
      updateInputPlaceholders(bindings, inPHs, inputImageData);

      // Perform the inference execution, updating output tensors.
      auto batchSize = inputImageData[0]->dims()[0];
      loader.runInference(exContext.get(), batchSize);
      if (traceContext) {
        traceContext->merge(exContext->getTraceContext());
      }

      // Process output of the network. Each app cand do its own post-processing
      // depending on type of the network.
      {
        std::lock_guard<std::mutex> lock(ioMu);
        numErrors += ppResultExecutor.processOutputs(oPHM, bindings,
                                                     inputImageBatchFilenames);
      }

      // Minibatch inference initialization of loader extensions.
      loader.inferEndMiniBatch(bindings, startMiniBatchIndex, miniBatch);
    }

    if (iterationsOpt) {
      // Image tensors loaded up to be run at once for benchmark mode.
      UniquePtrVec<ExecutionContext> contexts =
          setupContextPool(outPHs, inPHs[0], *inputImageData[0]);

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
      miniBatchMode ? inputImageFilenames_[0].size() / miniBatch : 1u;
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
                          inputImageFilenames_[0].size());
    } else {
      startIndex = 0;
      endIndex = inputImageFilenames_[0].size();
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
