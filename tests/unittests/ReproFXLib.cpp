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
#include <filesystem>
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
    reproPathOpt("repro_path", llvm::cl::desc("Path to Repro Base Directory."),
                 llvm::cl::Required, llvm::cl::cat(reproTestCat));
llvm::cl::opt<std::string>
    backend("backend", llvm::cl::desc("Backend target for lowering."),
            llvm::cl::value_desc("name of backend"), llvm::cl::init("NNPI"),
            llvm::cl::cat(reproTestCat));
llvm::cl::opt<bool> useCosSimEval(
    "use_cos_eval",
    llvm::cl::value_desc("Use Cosine evaluation when comparing outputs"),
    llvm::cl::init(false), llvm::cl::cat(reproTestCat));

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
    auto keyRef = llvm::StringRef(key);
    auto modPathAndWeightName = keyRef.rsplit(".");

    // Attribute is in the base module, i.e. "." isn't found.
    if (modPathAndWeightName.second == "") {
      torch::Tensor tensor = container.attr(key).toTensor();
      weights.insert(key, tensor);
      continue;
    }

    // Attribute is inside some recursive module, e.g. "a.b.c".
    llvm::SmallVector<llvm::StringRef, 4> modNames;
    modPathAndWeightName.first.split(modNames, ".");
    auto currMod = container;
    for (const auto &modName : modNames) {
      currMod = currMod.attr(modName.str()).toModule();
    }
    torch::Tensor tensor =
        currMod.attr(modPathAndWeightName.second.str()).toTensor();
    weights.insert(key, tensor);
  }

  if (input.size() != 0) {
    // Load all inputs tensors
    std::vector<at::IValue> tensors =
        torch::pickle_load(input).toTupleRef().elements();

    for (auto &i : tensors) {
      torch::Tensor tensor = i.toTensor();
      inputs.push_back(tensor);
    }
  }
}

static const std::string &checkPath(const std::string &path,
                                    bool missingPathFatal = false) {
  if (std::filesystem::exists(path)) {
    return path;
  }
  if (!missingPathFatal) {
    LOG(INFO) << "Didn't find " << path;
    static const std::string empty = "";
    return empty;
  } else {
    LOG(FATAL) << "Didn't find " << path;
  }
}

void ReproFXLib::parseCommandLine(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Load model data files.
  std::cout << "repro_path flag set to '" << reproPathOpt
            << "'; overriding other path flag values.";
  jsonPathOpt = checkPath(reproPathOpt + "/fx_data.json", true);
  weightsPathOpt = checkPath(reproPathOpt + "/fx_weights.pt", true);
  inputPathOpt = checkPath(reproPathOpt + "/fx_inputs.pt");
  compileSpecPathOpt = checkPath(reproPathOpt + "/fx_compile_spec.txt", true);
  outputPathOpt = checkPath(reproPathOpt + "/fx_outputs.pt");
  std::ifstream jsonFile(jsonPathOpt.c_str());
  std::stringstream buffer;
  buffer << jsonFile.rdbuf();
  folly::dynamic data = folly::parseJson(buffer.str());

  // Load weights file.
  torch::jit::script::Module container =
      torch::jit::load(weightsPathOpt.c_str());

  // Load inputs file.
  std::vector<char> input;
  if (!inputPathOpt.empty()) {
    std::ifstream inputStream(inputPathOpt.c_str());
    inputStream >> std::noskipws;

    input.insert(input.begin(), std::istream_iterator<char>(inputStream),
                 std::istream_iterator<char>());
  }

  load(data, container, input);
}

std::vector<torch::Tensor> ReproFXLib::run(bool fatalOnNotClose) {
  // Create FXGlowCompileSpec.
  glow::FXGlowCompileSpec compileSpec;
  if (!compileSpecPathOpt.empty()) {
    compileSpec.read_from_file(compileSpecPathOpt.c_str());
  } else {
    compileSpec.set_glow_backend(backend.c_str());
  }

  glow::FXGlow binding(
      c10::make_intrusive<glow::FXGlowCompileSpec>(compileSpec));

  // Setup host and load network.
  binding.setupHost();
  binding.loadNetwork(serializedGraphJson, weights, nodesJson, inputNames,
                      outputs);

  if (inputs.empty()) {
    std::cerr << "Did not receive a serialized input values file. Not "
                 "executing the Neural Net.\n";
    return {};
  }

  // Run network with input and save output to fx_output.pt.
  std::vector<torch::Tensor> out = binding.runNetwork(inputs);

  // Check to see if user provided output file for comparison.
  if (!outputPathOpt.empty()) {
    std::ifstream outputStream(outputPathOpt.c_str());
    outputStream >> std::noskipws;

    std::vector<char> output;
    output.insert(output.begin(), std::istream_iterator<char>(outputStream),
                  std::istream_iterator<char>());
    std::vector<at::IValue> outputFileTensors =
        torch::pickle_load(output).toTupleRef().elements();
    if (useCosSimEval) {
      // Calculate cosine similarity between user's output tensors and ReproFX's
      // out.
      std::vector<torch::Tensor> cosSimilarity;
      torch::nn::CosineSimilarity cos(
          torch::nn::CosineSimilarityOptions().dim(0));

      for (int i = 0; i < out.size() && i < outputFileTensors.size(); i++) {
        torch::Tensor a = out[i].flatten();
        torch::Tensor b = outputFileTensors[i].toTensor().flatten();
        auto c = cos(a, b);
        if (fatalOnNotClose && *c.data_ptr<float>() != 1) {
          LOG(FATAL) << "Cosine Similarity failure.";
        }
        cosSimilarity.push_back(c);
      }
      LOG(INFO) << "Cosine Similarity: " << cosSimilarity;
    } else {
      for (int i = 0; i < out.size() && i < outputFileTensors.size(); i++) {
        torch::Tensor a = out[i].flatten();
        torch::Tensor b = outputFileTensors[i].toTensor().flatten();
        LOG(INFO) << "Checking " << b.dtype() << " vs " << a.dtype();
        if (a.dtype() == torch::kFloat16 or a.dtype() == torch::kFloat32) {
          if (!at::allclose(b.toType(torch::kFloat32),
                            a.toType(torch::kFloat32), 0.0, 1e-05)) {
            LOG(FATAL) << "Run Failed for output " << i;
          }
        } else {
          if (!at::allclose(at::dequantize(b), at::dequantize(a), 0.0, 1e-05)) {
            LOG(FATAL) << "Run Failed for output " << i;
          }
        }
      }
    }
  }
  return out;
}
