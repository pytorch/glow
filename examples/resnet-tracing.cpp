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
#include "glow/Runtime/TraceLogger.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

#include <chrono>
#include <future>

using namespace glow;
using namespace glow::runtime;

namespace {
llvm::cl::OptionCategory category("resnet-tracing Options");
llvm::cl::opt<std::string>
    inputImage(llvm::cl::desc("path to input image to classify, which must be "
                              "a png with standard imagenet normalization"),
               llvm::cl::init("../tests/images/imagenet/dog_207.png"),
               llvm::cl::Positional, llvm::cl::cat(category));
llvm::cl::opt<std::string>
    outputJson(llvm::cl::desc("path to write output json trace events"),
               llvm::cl::init("./resnet-tracing.json"), llvm::cl::Positional,
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
  Placeholder *input =
      llvm::cast<Placeholder>(cantFail(loader.getNodeValueByName(inputName)));
  Placeholder *output = cantFail(loader.getSingleOutput());

  return std::make_pair(input, output);
}

/// Compiles the resnet50 function.
std::unique_ptr<CompiledFunction> compileModel(Module &module) {
  auto *backend = createBackend(BackendKind::Interpreter);
  Function *F = module.getFunction("resnet50");

  llvm::outs() << "Starting compile.\n";
  backend->optimizeFunction(CompilationMode::Infer, F);
  return backend->compile(F);
}

std::future<ResultCode> addToDevice(unsigned int id, DeviceManager *device,
                                    Module &module, FunctionMapTy functions) {
  std::shared_ptr<std::promise<ResultCode>> compilePromise(
      new std::promise<ResultCode>);
  auto future = compilePromise->get_future();

  device->addNetwork(&module, functions,
                     [compilePromise, id](const Module *, ResultCode code) {
                       if (code != ResultCode::Ready) {
                         llvm::errs() << "Failed to compile model for device "
                                      << id << ".\n";
                       } else {
                         llvm::outs()
                             << "Successfully added to Device " << id << ".\n";
                       }
                       compilePromise->set_value(code);
                     });

  return future;
}

/// Run ResNet concurrently on a fixed number of CPU Devices
int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Run ResNet and export a json file containing trace events");

  DeviceManager *device = DeviceManager::createDeviceManager(
      BackendKind::Interpreter, "resnet-tracing");
  device->init();

  // Load and compile model.

  Module module;
  TypeRef inputType(module.uniqueType(ElemKind::FloatTy, {1, 3, 224, 224}));
  Placeholder *input, *output;

  std::tie(input, output) = loadResnet50Model(inputType, module);
  auto compiledFunction = compileModel(module);

  FunctionMapTy functions;
  functions.emplace("resnet50", compiledFunction.get());

  auto f = addToDevice(0, device, module, functions);
  f.wait_for(/* timeout_duration */ std::chrono::seconds(30));
  if (f.get() != ResultCode::Ready) {
    return 1;
  }

  TraceLogger traceLogger(0);

  auto image =
      readPngImageAndPreprocess(inputImage, ImageNormalizationMode::k0to1,
                                ImageChannelOrder::BGR, ImageLayout::NCHW,
                                /* useImagenetNormalization */ true);

  Tensor batch(inputType);
  batch.getHandle<float>().insertSlice(image, 0);

  auto ctx = llvm::make_unique<Context>();
  ctx->allocate(module.getPlaceholders());
  updateInputPlaceholders(*ctx, {input}, {&batch});

  TraceThread traceThread = traceLogger.getTraceThread();
  ctx->setTraceLogger(&traceThread);

  llvm::outs() << "Starting Run.\n";
  std::promise<void> finished;
  device->runFunction("resnet50", std::move(ctx),
                      [&finished, &ctx](RunIdentifierTy, ResultCode r,
                                        std::unique_ptr<Context> ctx2) {
                        ctx = std::move(ctx2);
                        finished.set_value();
                      });

  finished.get_future().wait();

  traceLogger.returnTraceThread(std::move(traceThread));

  llvm::outs() << "Dumping json to " << outputJson << ".\n";
  traceLogger.dumpTraceEvents(outputJson);

  return 0;
}
