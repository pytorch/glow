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
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Support/Error.h"

using namespace glow;

/// A stripped-down example of how to load a Caffe2 protobuf and perform
/// inference.
int main() {
  glow::PlaceholderBindings bindings;
  glow::ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("lenet_mnist");
  auto *inputType = mod.uniqueType(glow::ElemKind::FloatTy, {1, 1, 28, 28});
  const char *inputName = "data";

  // Load and compile LeNet MNIST model.
  glow::Caffe2ModelLoader loader("lenet_mnist/predict_net.pb",
                                 "lenet_mnist/init_net.pb", {inputName},
                                 {inputType}, *F);
  EE.compile(glow::CompilationMode::Infer);

  // Get input and output placeholders.
  auto *input = llvm::cast<glow::Placeholder>(
      EXIT_ON_ERR(loader.getNodeValueByName(inputName)));
  auto *output = EXIT_ON_ERR(loader.getSingleOutput());

  // Read an example PNG and add it to an input batch.
  auto image = glow::readPngImageAndPreprocess(
      "tests/images/mnist/5_1087.png", glow::ImageNormalizationMode::k0to1,
      glow::ImageChannelOrder::BGR, glow::ImageLayout::NCHW);
  glow::Tensor batch(inputType);
  batch.getHandle<>().insertSlice(image, 0);

  // Allocate memory for input and bind it to the placeholders.
  bindings.allocate(mod.getPlaceholders());
  glow::updateInputPlaceholders(bindings, {input}, {&batch});

  // Perform inference.
  EE.run(bindings);

  // Read output and find argmax.
  auto out = bindings.get(output)->getHandle<float>();
  printf("digit: %zu\n", (size_t)out.minMaxArg().second);
  return 0;
}
