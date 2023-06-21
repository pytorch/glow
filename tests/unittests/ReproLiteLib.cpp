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

#include "glow/glow/tests/unittests/ReproLiteLib.h"
#include "glow/glow/lib/Backends/FBA/FBABackend.h"
#include "mtia/host_runtime/data_bundle/raw_data_bundle.h"
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
    outBundlePath("out_bundle_path",
                  llvm::cl::desc("Path to Repro Bundle Directory."),
                  llvm::cl::Required, llvm::cl::cat(reproTestCat));

// Parses JSON input, weights, and user inputs

/// Check that \p path is a valid filepath. Treat missing paths as
/// fatal errors if \p missingPathFatal is true.
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

/// Populate data structures serializedGraphJson_, config_, and strweights_
/// using data stored in artifact files.
void ReproLiteLib::loadFromAFG() {
  std::ifstream jsonFile(jsonPathOpt_.c_str());
  std::stringstream buffer;
  buffer << jsonFile.rdbuf();
  serializedGraphJson_ = buffer.str();
  jsonFile.close();
  std::ifstream configFile(configPathOpt_.c_str());
  std::string line;
  while (std::getline(configFile, line)) {
    std::vector<std::string> tokens;
    std::stringstream tokenizer(line);
    std::string token;
    while (std::getline(tokenizer, token, ':')) {
      tokens.push_back(token);
    }
    int count = 0;
    std::string rhs;
    // Handle cases where we have multiple nested dictionaries.
    // ex: foo : {bar : 1, baz : 2}
    // This can happen with registered operators.
    for (const auto &ntoken : tokens) {
      if (count == 0) {
        count++;
        continue;
      } else if (count == 1) {
        count++;
      } else if (count == 2) {
        rhs += ":";
      }
      rhs += ntoken;
    }
    config_[tokens[0]] = rhs;
  }
  configFile.close();

  std::ifstream fin(weightsPathOpt_, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(fin)),
                          (std::istreambuf_iterator<char>()));
  weights_ = torch::pickle_load(bytes);
  fin.close();
  for (auto &item : weights_.toGenericDict()) {
    const at::IValue &key = item.key();
    const at::IValue &val = item.value();
    CHECK(val.isTensor()) << "Expected tensor but got " << val.type()->str();
    auto &tensor = val.toTensor();
    // Get the actual pointer to the tensor data.
    auto dataPtr = static_cast<const uint8_t *>(tensor.data_ptr());
    // Get the size of the tensor in bytes.
    auto size = tensor.numel() * tensor.element_size();
    // Make sure that data is accessible by looking at the first and last
    // element.
    CHECK_EQ(dataPtr[0] + dataPtr[size - 1], dataPtr[size - 1] + dataPtr[0])
        << "Tensor data is not accessible";
    strweights_[key.toStringRef()] = (void *)dataPtr;
  }
}

uint64_t align(uint64_t size, uint64_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

void ReproLiteLib::generateBundle(std::unique_ptr<glow::CompiledResult> cr) {
  if (outBundlePath.empty()) {
    return;
  }

  if (!std::filesystem::exists(outBundlePath.getValue())) {
    std::filesystem::create_directories(outBundlePath.getValue());
  }

  /// Raw data bundle builder
  facebook::fbia_streaming::raw_data_bundle::RawDataBundleBuilder
      raw_data_bundle_builder;

  raw_data_bundle_builder.addExecutable(cr->executables[0].filename);
  raw_data_bundle_builder.addActivationBufferSize(cr->activationsMemSize);

  {
    std::vector<facebook::fbia_streaming::HostPtr> data(1, nullptr);
    data[0] = cr->constants;
    raw_data_bundle_builder.addWeights(
        data, {cr->constantWeightVarsMemSize},
        {align(cr->constantWeightVarsMemSize, 512)});
  }

  std::string inputFile = checkPath(reproPathOpt + "/inputs/input_0.pt", true);
  std::ifstream fin(inputFile, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(fin)),
                          (std::istreambuf_iterator<char>()));
  auto res = torch::pickle_load(bytes).toList();
  fin.close();
  std::vector<facebook::fbia_streaming::HostPtr> inputData(res.size(), nullptr);
  std::vector<size_t> inputSizes(res.size(), 0);
  for (int i = 0; i < res.size(); i++) {
    auto tensor = res[i].get().toTensor();
    inputData[i] = tensor.data().data_ptr();
    inputSizes[i] = tensor.numel() * tensor.element_size();
  }
  raw_data_bundle_builder.addInputs(inputData, {inputSizes});

  std::string outputFile =
      checkPath(reproPathOpt + "/outputs/output_0.pt", true);
  fin = std::ifstream(outputFile, std::ios::binary);
  bytes = std::vector<char>((std::istreambuf_iterator<char>(fin)),
                            (std::istreambuf_iterator<char>()));
  res = torch::pickle_load(bytes).toList();
  fin.close();
  std::vector<facebook::fbia_streaming::HostPtr> outputData(res.size(),
                                                            nullptr);
  std::vector<size_t> outputSize(res.size(), 0);
  for (int i = 0; i < res.size(); i++) {
    auto tensor = res[i].get().toTensor();
    outputData[i] = tensor.data().data_ptr();
    outputSize[i] = tensor.numel() * tensor.element_size();
  }
  raw_data_bundle_builder.addOutputs(outputData, {outputSize});

  raw_data_bundle_builder.save(outBundlePath.getValue());
}

/// Parse the command line; checking for the existence of the
/// necessary config, graph_ir, and weights files.
void ReproLiteLib::parseCommandLine(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::cout << "repro_path flag set to '" << reproPathOpt;
  jsonPathOpt_ = checkPath(reproPathOpt + "/graph_ir.json", true);
  configPathOpt_ = checkPath(reproPathOpt + "/config.spec", true);
  weightsPathOpt_ = checkPath(reproPathOpt + "/weights.pt", true);
}

/// Lower the serialized FX IR using the C++ backend.
void ReproLiteLib::run() {
  // Create FXGlowCompileSpec.
  loadFromAFG();
  auto backend = glow::FBABackend();

  auto compiledOrErr = backend.compileFXToCompiledResults(
      folly::parseJson(serializedGraphJson_), strweights_, config_);

  if (!compiledOrErr) {
    // If an error occured return the error.
    EXIT_ON_ERR(compiledOrErr.takeError());
  }
  std::unique_ptr<glow::CompiledResult> cr = std::move(*compiledOrErr);
  generateBundle(std::move(cr));
}
