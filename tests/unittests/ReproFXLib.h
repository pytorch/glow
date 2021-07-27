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

#include <folly/dynamic.h>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#ifndef GLOW_TESTS_REPROFXLIB_H
#define GLOW_TESTS_REPROFXLIB_H

class ReproFXLib {
public:
  std::string serializedGraphJson;
  std::string nodesJson;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputs;
  std::vector<std::string> keys;
  std::vector<torch::Tensor> inputs;

  using TorchDict = torch::Dict<std::string, torch::Tensor>;
  TorchDict weights;

  void load(const folly::dynamic &data,
            const torch::jit::script::Module &container,
            const std::vector<char> &input);
  void parseCommandLine(int argc, char **argv);
  std::vector<torch::Tensor> run();
};

#endif // GLOW_TESTS_REPROFXLIB_H
