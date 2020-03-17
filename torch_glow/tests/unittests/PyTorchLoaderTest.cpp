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

#include "PyTorchFileLoader.h"
#include "PyTorchModelLoader.h"
#include "glow/Support/Error.h"

#include <gtest/gtest.h>

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

TEST(ModelLoaderTest, Loader) {
  const std::string fileName{GLOW_DATA_PATH
                             "tests/models/pytorchModels/resnet18.pt"};
  std::shared_ptr<torch::jit::Module> module;
  glow::Error err = glow::PyTorchFileLoader::loadPyTorchModel(fileName, module);
  EXPECT_FALSE(err);
}

TEST(ModelLoaderTest, Fusion) {
  const std::string fileName{GLOW_DATA_PATH
                             "tests/models/pytorchModels/resnet18.pt"};

  glow::Module mod;
  auto *F = mod.createFunction("GlowFunction");
  std::vector<torch::jit::IValue> vec;
  auto emptyTensor = at::empty({1, 3, 224, 224});
  vec.push_back(torch::autograd::make_variable(emptyTensor));

  std::vector<glow::Placeholder *> inputPlaceholders;
  std::vector<glow::Placeholder *> outputPlaceholders;

  glow::Error err = glow::PyTorchFileLoader::loadPyTorchGraph(
      fileName, vec, *F, inputPlaceholders, outputPlaceholders);

  EXPECT_FALSE(ERR_TO_BOOL(std::move(err)));
}

TEST(ModelLoaderTest, Direct) {
  const std::string fileName{GLOW_DATA_PATH
                             "tests/models/pytorchModels/resnet18.pt"};

  glow::Module mod;
  auto *F = mod.createFunction("GlowFunction");
  std::vector<torch::jit::IValue> vec;
  auto emptyTensor = at::empty({1, 3, 224, 224});
  vec.push_back(torch::autograd::make_variable(emptyTensor));

  std::vector<glow::Placeholder *> inputPlaceholders;
  std::vector<glow::Placeholder *> outputPlaceholders;

  glow::Error err = glow::PyTorchFileLoader::parsePyTorchGraphForOnnxTraining(
      fileName, vec, *F, inputPlaceholders, outputPlaceholders);

  EXPECT_FALSE(ERR_TO_BOOL(std::move(err)));
}
