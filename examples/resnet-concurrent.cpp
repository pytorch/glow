/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#include "glow/Runtime/RuntimeTypes.h"

// We have not written the piece in the HostManager that initializes and creates
// DeviceManagers for individual devices, so for now just reach into lib/
#include "Backends/CPU/CPUDeviceManager.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

#include <chrono>
#include <future>

using namespace glow;
using namespace glow::runtime;

namespace {
llvm::cl::OptionCategory category("resnet-concurrent Options");
llvm::cl::opt<std::string>
    inputDirectory(llvm::cl::desc("input directory for images, which must be "
                                  "png's with standard imagenet normalization"),
                   llvm::cl::init("../tests/images/imagenet/"),
                   llvm::cl::Positional, llvm::cl::cat(category));
llvm::cl::opt<unsigned> numDevices("numDevices",
                                   llvm::cl::desc("Number of Devices to use"),
                                   llvm::cl::init(7), llvm::cl::value_desc("N"),
                                   llvm::cl::cat(category));
llvm::cl::opt<unsigned>
    maxImages("maxImages",
              llvm::cl::desc("Maximum number of images to load and classify"),
              llvm::cl::init(1000), llvm::cl::value_desc("N"),
              llvm::cl::cat(category));
} // namespace

/// Loads the model into /p module and returns the input and output
/// Placeholders.
std::pair<Placeholder *, Placeholder *> loadResnet50Model(TypeRef inputType,
                                                          Module &module) {
  Function *F = module.createFunction("resnet50");

  llvm::outs() << "Loading resnet50 model.\n";

  const char inputName[] = "gpu_0/data";
  Caffe2ModelLoader loader("resnet50/predict_net.pb", "resnet50/init_net.pb",
                           {inputName}, {inputType}, *F);
  Placeholder *input = llvm::cast<Placeholder>(
      EXIT_ON_ERR(loader.getNodeValueByName(inputName)));
  Placeholder *output = EXIT_ON_ERR(loader.getSingleOutput());

  return std::make_pair(input, output);
}

/// Compiles the resnet50 function.
std::unique_ptr<CompiledFunction> compileModel(Module &module) {
  auto *backend = createBackend(BackendKind::CPU);
  Function *F = module.getFunction("resnet50");

  llvm::outs() << "Starting compile.\n";

  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  backend->optimizeFunction(F, opts);
  return backend->compile(F, opts);
}

/// Loads the CompliedFunction into device \p device.
/// Returns a future which is completed when the device is initialized.
std::future<void> addToDevice(unsigned int id, DeviceManager *device,
                              Module &module, FunctionMapTy functions) {
  auto compilePromise = std::make_shared<std::promise<void>>();
  auto future = compilePromise->get_future();

  device->addNetwork(&module, functions,
                     [compilePromise, id](const Module *, llvm::Error err) {
                       if (err) {
                         llvm::errs() << "Failed to compile model for device "
                                      << id << ".\n";
                         EXIT_ON_ERR(std::move(err));
                       } else {
                         llvm::outs()
                             << "Successfully added to Device " << id << ".\n";
                       }
                       compilePromise->set_value();
                     });

  return future;
}

/// Starts a run of resnet50 on the given image. The image must be already
/// loaded into the input placeholder in /p context.
/// If, at the end of the run the number of \p returned results is equal to
/// maxImages, the \p finished promise is set.
void dispatchClassify(unsigned int id, DeviceManager *device, std::string path,
                      Placeholder *output,
                      std::unique_ptr<ExecutionContext> context,
                      std::atomic<size_t> &returned,
                      std::promise<void> &finished) {
  device->runFunction("resnet50", std::move(context),
                      [id, path, output, &returned,
                       &finished](RunIdentifierTy, llvm::Error err,
                                  std::unique_ptr<ExecutionContext> context) {
                        EXIT_ON_ERR(std::move(err));
                        size_t maxIdx = context->getPlaceholderBindings()
                                            ->get(output)
                                            ->getHandle<>()
                                            .minMaxArg()
                                            .second;

                        llvm::outs() << "(" << id << ") " << path << ": "
                                     << maxIdx << "\n";
                        if (++returned == maxImages) {
                          finished.set_value();
                        }
                      });
}

/// Run ResNet concurrently on a fixed number of CPU Devices
int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Run ResNet concurrently on a fixed number of CPU devices");

  llvm::outs() << "Initializing " << numDevices << " CPU Devices.\n";
  std::vector<std::unique_ptr<CPUDeviceManager>> devices;
  for (unsigned int i = 0; i < numDevices; ++i) {
    devices.emplace_back(llvm::make_unique<CPUDeviceManager>());
    EXIT_ON_ERR(devices[i]->init());
  }

  // Load and compile model.

  Module module;
  TypeRef inputType(module.uniqueType(ElemKind::FloatTy, {1, 3, 224, 224}));
  Placeholder *input, *output;

  std::tie(input, output) = loadResnet50Model(inputType, module);
  auto compiledFunction = compileModel(module);

  FunctionMapTy functions;
  functions.emplace("resnet50", compiledFunction.get());

  std::vector<std::future<void>> compiles;
  compiles.reserve(numDevices);

  for (unsigned int i = 0; i < numDevices; ++i) {
    compiles.push_back(addToDevice(i, devices[i].get(), module, functions));
  }

  for (auto &f : compiles) {
    f.wait_for(/* timeout_duration */ std::chrono::seconds(30));
  }

  llvm::outs() << "Loading files from " << inputDirectory << "\n";
  std::error_code code;
  llvm::sys::fs::directory_iterator dirIt(inputDirectory, code);
  if (code.value()) {
    llvm::errs() << "Couldn't read from directory: " << inputDirectory
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
    auto image = readPngImageAndPreprocess(
        path, ImageNormalizationMode::k0to1, ImageChannelOrder::BGR,
        ImageLayout::NCHW, imagenetNormMean, imagenetNormStd);

    Tensor batch = image.getUnowned(inputType->dims());

    auto context = llvm::make_unique<ExecutionContext>();
    context->getPlaceholderBindings()->allocate(module.getPlaceholders());
    updateInputPlaceholders(*context->getPlaceholderBindings(), {input},
                            {&batch});

    dispatchClassify(started, devices[started % numDevices].get(),
                     std::move(path), output, std::move(context), returned,
                     finished);

    dirIt.increment(code);
  }

  finished.get_future().wait();

  llvm::outs() << "Finished classifying " << started << " images.\n";

  return 0;
}
