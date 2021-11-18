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

#include "glow/glow/tests/unittests/ReproFXLib.h"
#include "glow/fb/fx/fx_glow/fx_glow.h"
#include "glow/glow/torch_glow/src/GlowCompileSpec.h"
#include <folly/dynamic.h>
#include <folly/json.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

llvm::cl::OptionCategory reproTestCat("Repro Category");
llvm::cl::opt<std::string>
    jsonPathOpt("json", llvm::cl::desc("Path to dumped JSON file"),
                llvm::cl::value_desc("json path"), llvm::cl::Required,
                llvm::cl::cat(reproTestCat));
llvm::cl::opt<std::string>
    weightsPathOpt("weights", llvm::cl::desc("Path to dumped weights file"),
                   llvm::cl::value_desc("weights path"), llvm::cl::Required,
                   llvm::cl::cat(reproTestCat));
llvm::cl::opt<std::string>
    inputPathOpt("inputs", llvm::cl::desc("Path to dumped input file"),
                 llvm::cl::value_desc("input path"), llvm::cl::Required,
                 llvm::cl::cat(reproTestCat));
llvm::cl::opt<std::string> outputPathOpt(
    "output",
    llvm::cl::desc("Path to output file to be compared with reproFX output"),
    llvm::cl::value_desc("output path"), llvm::cl::cat(reproTestCat));

// Parses JSON input, weights, and user inputs
void ReproFXLib::load(const folly::dynamic &data,
                      const torch::jit::script::Module &container,
                      const std::vector<char> &input) {
  // Get runNetwork paramaters from data
  // serialized_graph_json
  serializedGraphJson = data[0].asString();

  // nodes_json
  nodesJson = data[1].asString();

  // input_names
  for (int i = 0; i < data[2].size(); i++) {
    inputNames.push_back(data[2][i].asString());
  }

  // outputs
  for (int i = 0; i < data[3].size(); i++) {
    outputs.push_back(data[3][i].asString());
  }

  // tensor keys
  for (int i = 0; i < data[4].size(); i++) {
    keys.push_back(data[4][i].asString());
  }

  // Insert all pairs into TorchDict for passing into loadNetwork.
  for (auto &key : keys) {
    torch::Tensor tensor = container.attr(key).toTensor();
    weights.insert(key, tensor);
  }

  // Load all inputs tensors
  std::vector<at::IValue> tensors =
      torch::pickle_load(input).toTupleRef().elements();

  for (auto &i : tensors) {
    torch::Tensor tensor = i.toTensor();
    inputs.push_back(tensor);
  }
}

void ReproFXLib::parseCommandLine(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Load model JSON file
  std::ifstream jsonFile(jsonPathOpt.c_str());
  std::stringstream buffer;
  buffer << jsonFile.rdbuf();
  folly::dynamic data = folly::parseJson(buffer.str());

  // Load weights file.
  torch::jit::script::Module container =
      torch::jit::load(weightsPathOpt.c_str());

  // Load inputs file.
  std::ifstream inputStream(inputPathOpt.c_str());
  inputStream >> std::noskipws;

  std::vector<char> input;
  input.insert(input.begin(), std::istream_iterator<char>(inputStream),
               std::istream_iterator<char>());

  load(data, container, input);
}

std::vector<torch::Tensor> ReproFXLib::run() {
  // Create FXGlowCompileSpec with NNPI backend.
  glow::FXGlowCompileSpec compileSpec;
  compileSpec.set_glow_backend("NNPI");

  glow::FXGlow binding(
      c10::make_intrusive<glow::FXGlowCompileSpec>(compileSpec));

  // Setup host and load network.
  binding.setupHost();
  binding.loadNetwork(serializedGraphJson, weights, nodesJson, inputNames,
                      outputs);

  // Run network with input and save output to fx_output.pt.
  std::vector<torch::Tensor> out = binding.runNetwork(inputs);
  torch::save(out, "fx_output.pt");

  // Check to see if user provided output file for comparison.
  if (!outputPathOpt.empty()) {
    std::vector<torch::Tensor> outputFileTensors;
    torch::load(outputFileTensors, outputPathOpt.c_str());

    // Calculate cosine simlarity between user's output tensors and ReproFX's
    // out.
    std::vector<torch::Tensor> cosSimilarity;
    torch::nn::CosineSimilarity cos(
        torch::nn::CosineSimilarityOptions().dim(0));

    for (int i = 0; i < out.size() && i < outputFileTensors.size(); i++) {
      torch::Tensor a = out[i].flatten();
      torch::Tensor b = outputFileTensors[i].flatten();
      cosSimilarity.push_back(cos(a, b));
    }
    std::cout << "Cosine Similarity: " << cosSimilarity << std::endl;
  }

  return out;
}
