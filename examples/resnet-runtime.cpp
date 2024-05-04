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
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/Error.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

#include <glog/logging.h>

#include <chrono>
#include <future>

using namespace glow;
using namespace glow::runtime;

namespace {
llvm::cl::OptionCategory category("resnet-runtime Options");
llvm::cl::opt<std::string>
    inputDirectory(llvm::cl::desc("input directory for images, which must be "
                                  "png's with standard imagenet normalization"),
                   llvm::cl::init("../tests/images/imagenet/"),
                   llvm::cl::Positional, llvm::cl::cat(category));
llvm::cl::opt<unsigned> numDevices("num-devices",
                                   llvm::cl::desc("Number of Devices to use"),
                                   llvm::cl::init(5), llvm::cl::value_desc("N"),
                                   llvm::cl::cat(category));
llvm::cl::opt<unsigned>
    maxImages("max-images",
              llvm::cl::desc("Maximum number of images to load and classify"),
              llvm::cl::init(100), llvm::cl::value_desc("N"),
              llvm::cl::cat(category));
llvm::cl::opt<std::string> tracePath("trace-path",
                                     llvm::cl::desc("Write trace logs to disk"),
                                     llvm::cl::init(""),
                                     llvm::cl::cat(category));
llvm::cl::opt<std::string>
    backend("backend",
            llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
            llvm::cl::Optional, llvm::cl::init("CPU"), llvm::cl::cat(category));

llvm::cl::opt<bool>
    autoInstrument("auto-instrument",
                   llvm::cl::desc("Add instrumentation for operator tracing"),
                   llvm::cl::Optional, llvm::cl::init(false),
                   llvm::cl::cat(category));

std::mutex eventLock;
std::unique_ptr<TraceContext> traceContext;

} // namespace

/// Loads the model into /p module and returns the input and output
/// Placeholders. Appending count to the function name.
Placeholder *loadResnet50Model(TypeRef inputType, Module *module,
                               unsigned int count) {
  Function *F = module->createFunction("resnet50" + std::to_string(count));

  LOG(INFO) << "Loading resnet50 model.";

  const char inputName[] = "gpu_0/data";
  Caffe2ModelLoader loader("resnet50/predict_net.pb", "resnet50/init_net.pb",
                           {inputName}, {inputType}, *F);
  Placeholder *input = llvm::cast<Placeholder>(
      EXIT_ON_ERR(loader.getNodeValueByName(inputName)));
  return input;
}

/// Starts a run of resnet50 on the given image. The image must be already
/// loaded into the input placeholder in /p context.
/// If, at the end of the run the number of \p returned results is equal to
/// maxImages, the \p finished promise is set.
void dispatchClassify(unsigned int id, HostManager *hostManager,
                      std::string path,
                      std::unique_ptr<ExecutionContext> context,
                      std::atomic<size_t> &returned,
                      std::promise<void> &finished) {
  auto runid = hostManager->runNetwork(
      "resnet50" + std::to_string(id), std::move(context),
      [path, &returned, &finished](RunIdentifierTy runid, Error err,
                                   std::unique_ptr<ExecutionContext> context) {
        EXIT_ON_ERR(std::move(err));
        auto *bindings = context->getPlaceholderBindings();
        size_t maxIdx =
            bindings->get(bindings->getPlaceholderByNameSlow("gpu_0_softmax"))
                ->getHandle()
                .minMaxArg()
                .second;
        // This output is verified by OutputCheck in tests so must be written to
        // stdout.
        llvm::outs() << "(" << runid << ") " << path << ": " << maxIdx << "\n";

        if (!tracePath.empty()) {
          std::lock_guard<std::mutex> l(eventLock);
          // Merge this run's TraceEvents into the global TraceContext.
          traceContext->merge(context->getTraceContext());
        }

        if (++returned == maxImages) {
          finished.set_value();
        }
      });
  LOG(INFO) << "Started run ID: " << runid;
}

/// Run ResNet concurrently on the number of devices provided by the user.
int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Run ResNet concurrently on a fixed number of devices");

  LOG(INFO) << "Initializing " << numDevices << " " << backend
            << " devices on HostManager.";

  std::vector<std::unique_ptr<DeviceConfig>> configs;
  for (unsigned int i = 0; i < numDevices; ++i) {
    auto config = glow::make_unique<DeviceConfig>(backend);
    configs.push_back(std::move(config));
  }

  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs));

  // If tracing is enabled, create a TraceContext to merge each runs events
  // into.
  if (!tracePath.empty()) {
    traceContext = glow::make_unique<TraceContext>(TraceLevel::STANDARD);
  }

  // Load model, create a context, and add to HostManager.

  std::vector<dim_t> inputShape{1, 3, 224, 224};

  Placeholder *input;
  PlaceholderList phList;

  std::unique_ptr<Module> module = glow::make_unique<Module>();
  TypeRef inputType = module->uniqueType(ElemKind::FloatTy, inputShape);
  input = loadResnet50Model(inputType, module.get(), 0);
  phList = module->getPlaceholders();
  CompilationContext cctx;
  cctx.backendOpts.autoInstrument = autoInstrument;
  cctx.saturateHost = true;
  EXIT_ON_ERR(hostManager->addNetwork(std::move(module), cctx));

  LOG(INFO) << "Loading files from " << inputDirectory;
  std::error_code code;
  llvm::sys::fs::directory_iterator dirIt(inputDirectory, code);
  if (code.value()) {
    LOG(ERROR) << "Couldn't read from directory: " << inputDirectory
               << " - code" << code.value() << "\n";
    exit(code.value());
  }

  std::promise<void> finished;

  size_t started = 0;
  std::atomic<size_t> returned{0};

  // Run up to maxImages classifications.
  while (started++ < maxImages) {
    if (code.value() != 0 || dirIt == llvm::sys::fs::directory_iterator()) {
      started--;
      returned += maxImages - started;

      if (returned == maxImages) {
        finished.set_value();
      }
      break;
    }

    std::string path = dirIt->path();

    auto image = readPngPpmImageAndPreprocess(
        path, ImageNormalizationMode::k0to1, ImageChannelOrder::BGR,
        ImageLayout::NCHW, imagenetNormMean, imagenetNormStd);
    std::unique_ptr<ExecutionContext> context =
        glow::make_unique<ExecutionContext>();
    context->setTraceContext(
        glow::make_unique<TraceContext>(TraceLevel::STANDARD));

    context->getPlaceholderBindings()->allocate(phList);
    Tensor batch = image.getUnowned(inputShape);
    updateInputPlaceholders(*(context->getPlaceholderBindings()), {input},
                            {&batch});

    dispatchClassify(0, hostManager.get(), std::move(path), std::move(context),
                     returned, finished);

    dirIt.increment(code);
  }

  finished.get_future().wait();

  LOG(INFO) << "Finished classifying " << started << " images.";

  if (!tracePath.empty()) {
    traceContext->dump(tracePath, "resnet-runtime");
  }

  return 0;
}
