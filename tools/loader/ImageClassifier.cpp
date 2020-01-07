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

#include "Loader.h"

#include "glow/Base/Image.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

using namespace glow;

namespace {

/// Image loader options.
llvm::cl::OptionCategory imageLoaderCat("Image Loader Options");
llvm::cl::list<std::string> inputImageFilenames(
    llvm::cl::Positional,
    llvm::cl::desc("<input files> (note: specifying '-' enables streaming "
                   "mode, where the model is compiled once and then can be run "
                   "many times with new input filenames passed via stdin)"),
    llvm::cl::ZeroOrMore);

llvm::cl::opt<std::string> inputImageListFile(
    "input-image-list-file",
    llvm::cl::desc(
        "Name of the file containing list of images (one image per line)"),
    llvm::cl::value_desc("string_name"), llvm::cl::Optional,
    llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned> miniBatch(
    "minibatch",
    llvm::cl::desc(
        "Size of mini-batches. Split the input image list into a set of "
        "mini-batches. The input model is compiled for an input tensor batch "
        "size equal to the specified mini-batch size and mini-batches of "
        "images are inferred separately. The number of input images must be a "
        "multiple of the mini-batch size. By default, splitting the input "
        "image list into mini-batches is deactivated."),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(imageLoaderCat));

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
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned> labelOffset(
    "label-offset",
    llvm::cl::desc("Label offset for TF ONNX models with 1001 classes"),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned> poolSize(
    "pool-size",
    llvm::cl::desc("Size of context pool for the benchmark; default:10"),
    llvm::cl::Optional, llvm::cl::init(10), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<bool> computeSoftmax(
    "compute-softmax", llvm::cl::desc("Compute softmax of the network output"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned>
    topKCount("topk",
              llvm::cl::desc("Number of highest likelihood labels to print and "
                             "match the correspondent expected-lables"),
              llvm::cl::Optional, llvm::cl::init(1),
              llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<std::string> modelInputName(
    "model-input-name",
    llvm::cl::desc("The name of the variable for the model's input image."),
    llvm::cl::value_desc("string_name"), llvm::cl::Required,
    llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<bool> convertInAndOutToFp16(
    "convert-inout-to-fp16",
    llvm::cl::desc(
        "Convert the input and output tensors of the network to fp16"),
    llvm::cl::cat(imageLoaderCat));

llvm::cl::list<unsigned> expectedMatchingLabels(
    "expected-labels",
    llvm::cl::desc("The comma delimited list of the matching lables"),
    llvm::cl::value_desc("int"), llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<std::string> tracePath("trace-path",
                                     llvm::cl::desc("Write trace logs to disk"),
                                     llvm::cl::init(""),
                                     llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<bool>
    autoInstrument("auto-instrument",
                   llvm::cl::desc("Add instrumentation for operator tracing"),
                   llvm::cl::Optional, llvm::cl::init(false),
                   llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned> traceLevel(
    "trace-level",
    llvm::cl::desc(
        "Set tracing level (bit-field, see TraceEvents.h for details)"),
    llvm::cl::Optional, llvm::cl::init((unsigned)TraceLevel::NONE),
    llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned>
    warmup("warmup",
           llvm::cl::desc("How many passes to do to warm everything up"),
           llvm::cl::init(0), llvm::cl::value_desc("W"),
           llvm::cl::cat(imageLoaderCat));

std::mutex eventLock;
std::unique_ptr<TraceContext> traceContext;

} // unnamed namespace

/// Write a prompt to stdout asking for filenames for classification. Read in
/// those filenames and add them to \p filenames. \p filenames is cleared before
/// adding the new set of filenames from stdin. \returns false if the passed in
/// line was empty.
static bool getNextImageFilenames(std::vector<std::string> *filenames) {
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
static bool getNextMiniBatch(std::vector<std::string> &imageList,
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

/// Creates and \returns the ProtobufLoader given \p loader and the
/// \p inputImageType. Note that this must come after loading images for
/// inference so that \p inputImageType is known.
static std::unique_ptr<ProtobufLoader>
createProtobufLoader(Loader &loader, TypeRef inputImageType) {
  // The image name that the model expects must be passed on the command line.
  const char *inputName = modelInputName.c_str();

  // Create the model based on the input model format.
  std::unique_ptr<ProtobufLoader> LD;
  bool c2Model = !loader.getCaffe2NetDescFilename().empty();
  if (c2Model) {
    LD.reset(new Caffe2ModelLoader(
        loader.getCaffe2NetDescFilename(), loader.getCaffe2NetWeightFilename(),
        {inputName}, {inputImageType}, *loader.getFunction()));
  } else {
    LD.reset(new ONNXModelLoader(loader.getOnnxModelFilename(), {inputName},
                                 {inputImageType}, *loader.getFunction()));
  }

  return LD;
}

/// Given \p loader, the \p bindings, and \p inputImageType, build the graph
/// from the provided protobuf file found via \p loader. Then compiles and
/// \returns a pair of pointers to the input Placeholder and output Placeholder
/// for the Softmax.
static std::pair<Placeholder *, Placeholder *>
buildAndCompileAndGetInAndOutPair(Loader &loader, PlaceholderBindings &bindings,
                                  TypeRef inputImageType) {
  auto LD = createProtobufLoader(loader, inputImageType);

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

  // The image name that the model expects must be passed on the command line.
  const char *inputName = modelInputName.c_str();
  Placeholder *inputImagePH =
      llvm::cast<Placeholder>(EXIT_ON_ERR(LD->getNodeValueByName(inputName)));

  // When profiling the graph do not return the output placeholder. This allows
  // profiling SSD models which have two output placeholders (scores and boxes).
  if (profilingGraph()) {
    return std::make_pair(inputImagePH, nullptr);
  }

  // Get the Tensor from the Placeholder that the final expected Softmax writes
  // into at the end of image inference.
  Placeholder *SMPH = EXIT_ON_ERR(LD->getSingleOutput());

  return std::make_pair(inputImagePH, SMPH);
}

/// A pair representing a float and the index where the float was found.
using FloatIndexPair = std::pair<float, size_t>;

/// Given a Handle \p H of a 1D tensor with float elements, \returns the top K
/// (topKCount) [float, index] pairs, i.e. the pairs with the highest floats.
template <typename ElemTy>
static std::vector<FloatIndexPair> getTopKPairs(Handle<ElemTy> H) {
  DCHECK_LE(topKCount, H.size()) << "Function requires k < number of labels.";
  DCHECK_EQ(H.dims().size(), 1) << "H must be a Handle of a 1d Tensor.";

  // Use a priority queue of pairs of floats (probabilities) to size_t (indices)
  // to determine the top K pairs, and then return the indices from it.
  std::priority_queue<FloatIndexPair, std::vector<FloatIndexPair>,
                      std::greater<FloatIndexPair>>
      topKQueue;

  // Loop over all the probabilites, finding the highest k probability pairs.
  for (dim_t i = 0, e = H.size(); i < e; i++) {
    float currProbability = H.at({i});
    if (topKQueue.size() < topKCount) {
      // Always push the first k elements.
      topKQueue.push(std::make_pair(currProbability, i));
    } else if (topKQueue.top().first < currProbability) {
      // If the lowest element has lower probability than the current, then pop
      // the lowest and insert the current pair.
      topKQueue.pop();
      topKQueue.push(std::make_pair(currProbability, i));
    }
  }

  // We now have the top K pairs in reverse order.
  std::vector<FloatIndexPair> res(topKCount);
  for (size_t i = 0; i < topKCount; i++) {
    res[topKCount - i - 1] = topKQueue.top();
    topKQueue.pop();
  }

  return res;
}

/// Print out the top K pairs to stdout, which were passed in via \p topKPairs.
static void printTopKPairs(const std::vector<FloatIndexPair> &topKPairs) {
  for (size_t i = 0; i < topKPairs.size(); i++) {
    // Some models are trained with more classes. E.g. Some imagenet models
    // exported from TensorFlow have 1 extra "neutral" class.
    const size_t label = topKPairs[i].second - labelOffset;
    // Tab out the label so it aligns nicely with Label-K1.
    if (i != 0) {
      llvm::outs() << "\t\t\t\t\t";
    }
    llvm::outs() << "\tLabel-K" << i + 1 << ": " << label << " (probability: "
                 << llvm::format("%0.4f", topKPairs[i].first) << ")\n";
  }
}

/// Checks if \p topKPairs have the index that matches the provided index,
/// \returns 0 on success and 1 if mismatches found.
static int checkExpectedLabel(llvm::ArrayRef<FloatIndexPair> topKPairs,
                              llvm::StringRef fileName,
                              unsigned expectedCategoryIndex) {
  // Loop through pairs and try to find a matching label.
  for (const auto &p : topKPairs) {
    if (p.second - labelOffset == expectedCategoryIndex) {
      return 0;
    }
  }

  llvm::outs() << " File: " << fileName
               << " doesn't match index: " << expectedCategoryIndex
               << " in the top " << topKPairs.size() << " pairs\n";

  return 1;
}

/// Apply the softmax function to the given handle.
template <typename ElemTy> static void applySoftmax(Handle<ElemTy> H) {
  DCHECK_EQ(H.dims().size(), 1) << "H must be a Handle of a 1d Tensor.";
  float denominator = 0.0f;

  for (auto elem : H) {
    denominator += std::exp(static_cast<float>(elem));
  }

  for (auto &elem : H) {
    elem = std::exp(static_cast<float>(elem)) / denominator;
  }
}

/// Given the output Softmax Tensor \p SMT and \p imageList, prints the
/// results of inference and returns number of incorrect predictions,
/// \returns the number of found mismatches.
template <typename ElemTy>
static int processAndPrintResultsImpl(Tensor *SMT,
                                      llvm::ArrayRef<std::string> imageList) {
  // Softmax should have at least two dimensions: batchSize (first dimension),
  // numLabels (any other dimension), and optionally - 1 in all other
  // dimensions. The value of numLabels should be greater than 1.
  DCHECK_GE(SMT->dims().size(), 2) << "Softmax should have at least 2 dims.";
  const dim_t batchSize = SMT->dims()[0];
  DCHECK_EQ(batchSize, imageList.size())
      << "Softmax batch size must equal the input number of images.";
  size_t labelsDim = 0;
  for (size_t i = 1; i < SMT->dims().size(); i++) {
    if (SMT->dims()[i] > 1) {
      DCHECK_EQ(labelsDim, 0) << "More than one dimension of size > 1?";
      labelsDim = i;
    }
  }
  DCHECK_NE(labelsDim, 0) << "Labels dimension not found!";
  const dim_t numLabels = SMT->dims()[labelsDim];
  // Get a view with canonical layout {batches, labels}.
  Tensor canonical = SMT->getUnowned({batchSize, numLabels});
  SMT = &canonical;

  std::vector<dim_t> sliceOffset(SMT->dims().size(), 0);

  int retVal = 0;
  for (unsigned i = 0; i < imageList.size(); i++) {
    const auto &fileName = imageList[i];
    llvm::outs() << " File: " << fileName;

    // batchSize is the first dimension, so update it to get the next slice.
    sliceOffset[0] = i;
    Tensor slice = SMT->getUnowned({numLabels}, sliceOffset);
    auto SH = slice.getHandle<ElemTy>();

    if (computeSoftmax) {
      applySoftmax(SH);
    }

    auto topKPairs = getTopKPairs(SH);
    printTopKPairs(topKPairs);
    if (!expectedMatchingLabels.empty()) {
      retVal +=
          checkExpectedLabel(topKPairs, fileName, expectedMatchingLabels[i]);
    }
  }

  return retVal;
}

/// Given the output Softmax Tensor \p SMT and \p functionName, switch between
/// the correct element type to print the results of inference as contained in
/// \p SMT, \returns the number of found mismatches.
static int processAndPrintResults(Tensor *SMT,
                                  llvm::ArrayRef<std::string> imageList) {
  switch (SMT->getElementType()) {
  case ElemKind::FloatTy:
    return processAndPrintResultsImpl<float>(SMT, imageList);
  case ElemKind::Float16Ty:
    return processAndPrintResultsImpl<float16_t>(SMT, imageList);
  default:
    llvm_unreachable("Type not supported");
  }
}

/// Read all images from \p inputImageListFile in to \p inputImageFilenames.
static void parseInputImageList(const std::string &inputImageListFile) {
  std::ifstream inFile;
  inFile.open(inputImageListFile);
  if (!inFile.good()) {
    llvm::outs() << "Could not open input-image-list-file: "
                 << inputImageListFile << ", exiting.\n";
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

/// Run inference request on HostManager. This method builds a runNetwork
/// request for the \p hostManager, this is a recursive call, in the callback
/// provided to the HostManager this function can call itself if the desired
/// number of warmups and requests has not yet been dispatched.
static void runInference(runtime::HostManager *hostManager, std::string name,
                         std::unique_ptr<ExecutionContext> batch,
                         std::promise<void> &runPromise,
                         std::atomic<unsigned> &inflight,
                         std::atomic<int> &dispatched, unsigned warmUp) {
  auto start = TraceEvent::now();
  hostManager->runNetwork(
      name, std::move(batch),
      [&runPromise, &inflight, &dispatched, hostManager, name, warmUp,
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
        // Kick off another run.
        if (dispatched.fetch_sub(1) > 0) {
          inflight++;
          runInference(hostManager, name, std::move(contextPtr), runPromise,
                       inflight, dispatched, warmUp > 0 ? warmUp - 1 : 0);
        }
        if (--inflight == 0) {
          runPromise.set_value();
        }
      });
}

/// Run the requested number of benchmark requests \p requestCount prepended by
/// \p warmUp cycles
// through the HostManager from the \p loader using the provided context pool \p
// contexts
/// and wait for all runs to complete.
static void
runBenchmark(std::string name, Loader &loader,
             std::vector<std::unique_ptr<ExecutionContext>> contexts,
             unsigned requestCount, unsigned warmUp) {
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
                 dispatched, warmUp);
  }
  // Wait for all to finish.
  fut.wait();
}

/// Setup the pool of contexts needed for a benchmark run.
static std::vector<std::unique_ptr<ExecutionContext>>
setupContextPool(Placeholder *outputPH, Placeholder *inputImagePH,
                 glow::Tensor &inputImageData) {
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
    ph->allocate(outputPH);
    contexts.push_back(std::move(newContext));
  }
  return contexts;
}

int main(int argc, char **argv) {
  // Verify/initialize command line parameters, and then loader initializes
  // the ExecutionEngine and Function.
  parseCommandLine(argc, argv);

  if (inputImageListFile.empty() && inputImageFilenames.size() == 0) {
    llvm::errs() << "Args: Either positional inputImageFilenames or "
                    "-inputImageListFile "
                    "must be used to specify input images.\n";
    std::exit(1);
  }

  if (!inputImageListFile.empty()) {
    CHECK_EQ(inputImageFilenames.size(), 0)
        << "When using -input-image-list-file all Input images must be "
           "specified "
           "using -input-image-list-file option.";
    parseInputImageList(inputImageListFile);
  }

  if (!expectedMatchingLabels.empty()) {
    // The number of category indices must match the number of files.
    if (expectedMatchingLabels.size() != inputImageFilenames.size()) {
      llvm::errs() << "Number of matching indices: "
                   << expectedMatchingLabels.size()
                   << " doesn't match the number of files: "
                   << inputImageFilenames.size() << "\n";
      return 1;
    }
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

  // Print out the inferred image classification.
  llvm::outs() << "Model: " << Loader::getModelOptPath() << "\n";
  std::mutex ioMu;
  int numErrors = 0;

  // Process a set of minibatches with indices [startIndex, endIndex).
  auto processImageRange = [&](size_t startIndex, size_t endIndex) {
    std::unique_ptr<ExecutionContext> exContext =
        glow::make_unique<ExecutionContext>();
    PlaceholderBindings &bindings = *exContext->getPlaceholderBindings();
    if (traceContext) {
      exContext->setTraceContext(
          glow::make_unique<TraceContext>(TraceLevel::STANDARD));
    }
    Loader loader;
    // Used to make sure we only compile once, and run only once if not
    // streaming.
    bool isFirstRun = true;

    // These will be set during the first run.
    Placeholder *inputImagePH = nullptr;
    Placeholder *outputPH = nullptr;

    size_t miniBatchIndex = startIndex;
    Tensor inputImageData;
    std::vector<std::string> inputImageBatchFilenames;
    if ((!miniBatchMode) && (!streamInputFilenamesMode)) {
      inputImageBatchFilenames = inputImageFilenames;
    }

    while ((streamInputFilenamesMode &&
            getNextImageFilenames(&inputImageBatchFilenames)) ||
           (miniBatchMode &&
            getNextMiniBatch(inputImageBatchFilenames, inputImageFilenames,
                             miniBatchIndex, miniBatch, endIndex)) ||
           isFirstRun) {
      // Load and process the image data into the inputImageData Tensor.
      loadImagesAndPreprocess(inputImageBatchFilenames, &inputImageData,
                              imageNormMode, imageChannelOrder, imageLayout);

      // It we are benchmarking reset the image data to the batch size we need.
      if (iterationsOpt) {
        ShapeVector imageSize(inputImageData.getType().dims().begin(),
                              inputImageData.getType().dims().end());
        if (miniBatch) {
          imageSize[0] = miniBatch;
        } else {
          imageSize[0] = iterationsOpt;
        }
        // Resize the Tensor to the appropriate size.
        inputImageData.reset(ElemKind::FloatTy, imageSize);
      }
      // If this is the first run, then we need to build and compile the model.
      if (isFirstRun) {
        isFirstRun = false;

        // Build and compile the graph, and then get back the input Placeholder
        // and output Placeholder.
        std::pair<Placeholder *, Placeholder *> inputOutputPair =
            buildAndCompileAndGetInAndOutPair(loader, bindings,
                                              &inputImageData.getType());

        // If in bundle mode, the bundle has been saved by the above call, so we
        // can safely return.
        if (emittingBundle()) {
          return;
        }

        inputImagePH = inputOutputPair.first;
        outputPH = inputOutputPair.second;
      }
      CHECK(inputImagePH) << "Input must be valid.";
      CHECK(inputImagePH->dims() == inputImageData.dims())
          << "New input shape does not match the compiled function: "
          << inputImagePH->dims() << " vs " << inputImageData.dims();

      // Validate the out placeholder when doing inference (and not profiling).
      if (!profilingGraph()) {
        CHECK(outputPH) << "Output must be valid.";
      }

      // Convert the raw input to fp16. This must be done every time we get new
      // image data.
      if (convertInAndOutToFp16) {
        inputImageData.convertToType(ElemKind::Float16Ty);
      }

      // If we are benchmarking we are done with the while loop.
      if (iterationsOpt) {
        break;
      }
      // About to run inference, so update the input image Placeholder's backing
      // Tensor with inputImageData.
      updateInputPlaceholders(bindings, {inputImagePH}, {&inputImageData});

      // Perform the inference execution, updating SMT.
      auto batchSize = inputImageData.dims()[0];
      loader.runInference(exContext.get(), batchSize);
      if (traceContext) {
        traceContext->merge(exContext->getTraceContext());
      }

      // Print the top-k results from the output Softmax tensor. Do this only
      // when doing inference (and not profiling).
      if (!profilingGraph()) {
        std::lock_guard<std::mutex> lock(ioMu);
        numErrors += processAndPrintResults(bindings.get(outputPH),
                                            inputImageBatchFilenames);
      }
    }

    if (iterationsOpt) {
      // Image tensors loaded up to be run at once for benchmark mode.
      std::vector<std::unique_ptr<ExecutionContext>> contexts =
          setupContextPool(outputPH, inputImagePH, inputImageData);

      std::string name = loader.getFunctionName();
      llvm::Timer timer("Infer", "Infer");
      if (timeOpt) {
        timer.startTimer();
      }
      unsigned requestCount = miniBatch ? iterationsOpt / miniBatch : 1;
      runBenchmark(name, loader, std::move(contexts), requestCount, warmup);
      if (timeOpt) {
        timer.stopTimer();
        llvm::outs() << llvm::formatv("Wall time per item (s): {0:f4}\n",
                                      timer.getTotalTime().getWallTime() /
                                          (iterationsOpt + warmup));
      }
    }

    // If profiling, generate and serialize the quantization infos now that we
    // have run inference one or more times to gather the profile.
    if (profilingGraph()) {
      loader.generateAndSerializeQuantizationInfos(bindings);
    }
  };

  // We will force single-threaded execution if:
  // - Minibatch mode is disabled;
  // - We are going to emit bundle and do not do inference;
  // - We are collecting inference profile.
  // Otherwise, there can be several minibatches of equal size.
  const bool multiThreadingAllowed =
      miniBatchMode && !emittingBundle() && !profilingGraph();
  const size_t numBatches =
      miniBatchMode ? inputImageFilenames.size() / miniBatch : 1u;
  const size_t numThreads = multiThreadingAllowed
                                ? std::min(size_t(miniBatchThreads), numBatches)
                                : 1u;
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
    if (numThreads > 1) {
      startIndex = i * miniBatchesPerThread * miniBatch;
      endIndex = std::min((i + 1) * miniBatchesPerThread * miniBatch,
                          inputImageFilenames.size());
    } else {
      startIndex = 0;
      endIndex = inputImageFilenames.size();
    }
    auto worker = [&processImageRange, startIndex, endIndex]() {
      processImageRange(startIndex, endIndex);
    };
    threads.push_back(std::thread(worker));
  }

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  if (!tracePath.empty()) {
    traceContext->dump(tracePath, "ImageClassifier");
  }

  return numErrors;
}
