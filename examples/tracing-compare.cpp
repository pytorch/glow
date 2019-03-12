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

#include "glow/Backends/DeviceManager.h"
#include "glow/Base/Image.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Runtime/RuntimeTypes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

#include <chrono>
#include <future>

using namespace glow;
using namespace glow::runtime;

std::array<BackendKind, 2> supportedBackends{BackendKind::Interpreter,
                                             BackendKind::CPU};

namespace {
llvm::cl::OptionCategory category("tracing-compare Options");
llvm::cl::opt<std::string>
    inputImage(llvm::cl::desc("path to input image to classify, which must be "
                              "a png with standard imagenet normalization"),
               llvm::cl::init("../tests/images/imagenet/dog_207.png"),
               llvm::cl::Positional, llvm::cl::cat(category));
llvm::cl::opt<std::string>
    outputJson(llvm::cl::desc("path to write output json trace events"),
               llvm::cl::init("./glow-trace.json"), llvm::cl::Positional,
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
std::unique_ptr<CompiledFunction> compileModel(Module &module,
                                               BackendKind backendKind) {
  auto *backend = createBackend(backendKind);
  Function *F = module.getFunction("resnet50");

  llvm::outs() << "Starting compile.\n";
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  opts.autoInstrument = true;
  backend->optimizeFunction(F, opts);
  return backend->compile(F, opts);
}

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

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Run resnet and export a json file containing trace events");

  std::array<DeviceManager *, supportedBackends.size()> devices;
  for (unsigned i = 0, e = supportedBackends.size(); i < e; ++i) {
    devices[i] = DeviceManager::createDeviceManager(supportedBackends[i]);
    EXIT_ON_ERR(devices[i]->init());
  }

  // Load and compile model.

  Module module;
  TypeRef inputType(module.uniqueType(ElemKind::FloatTy, {1, 3, 224, 224}));
  Placeholder *input, *output;

  std::tie(input, output) = loadResnet50Model(inputType, module);

  std::array<std::unique_ptr<CompiledFunction>, supportedBackends.size()>
      compiledFunctions;

  for (unsigned i = 0, e = supportedBackends.size(); i < e; ++i) {
    compiledFunctions[i] = compileModel(module, supportedBackends[i]);

    FunctionMapTy functions;
    functions.emplace("resnet50", compiledFunctions[i].get());

    auto f = addToDevice(i, devices[i], module, functions);
    f.wait_for(/* timeout_duration */ std::chrono::seconds(30));
  }

  auto image =
      readPngImageAndPreprocess(inputImage, ImageNormalizationMode::k0to1,
                                ImageChannelOrder::BGR, ImageLayout::NCHW,
                                /* useImagenetNormalization */ true);

  Tensor batch = image.getUnowned(inputType->dims());

  llvm::outs() << "Starting Run.\n";
  std::array<std::promise<std::unique_ptr<ExecutionContext>>,
             supportedBackends.size()>
      promises;

  for (unsigned i = 0, e = supportedBackends.size(); i < e; ++i) {
    auto context = llvm::make_unique<ExecutionContext>();
    context->getPlaceholderBindings()->allocate(module.getPlaceholders());
    updateInputPlaceholders(*(context->getPlaceholderBindings()), {input},
                            {&batch});

    devices[i]->runFunction(
        "resnet50", std::move(context),
        [&promises, i](RunIdentifierTy, llvm::Error err,
                       std::unique_ptr<ExecutionContext> context) {
          EXIT_ON_ERR(std::move(err));
          promises[i].set_value(std::move(context));
        });
  }

  auto context = llvm::make_unique<ExecutionContext>();
  auto &allEvents = context->getTraceEvents();

  allEvents.push_back({"thread_name", 0, "M", 0, {{"name", "Interpreter"}}});
  allEvents.push_back({"thread_name", 0, "M", 1, {{"name", "CPU"}}});

  for (unsigned i = 0, e = supportedBackends.size(); i < e; ++i) {
    auto f = promises[i].get_future();
    f.wait_for(/* timeout_duration */ std::chrono::seconds(30));
    auto runbindings = f.get();
    for (auto &event : runbindings->getTraceEvents()) {
      event.tid = i;
      allEvents.push_back(event);
    }
  }

  llvm::outs() << "Dumping json to " << outputJson << ".\n";
  TraceEvent::dumpTraceEvents(allEvents, outputJson);

  return 0;
}
