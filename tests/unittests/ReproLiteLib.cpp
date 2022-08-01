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

// Parses JSON input, weights, and user inputs

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
    assert(tokens.size() == 2);
    config_[tokens[0]] = tokens[1];
  }
  configFile.close();
  std::ifstream fin(weightsPathOpt_, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(fin)),
                          (std::istreambuf_iterator<char>()));
  auto res = torch::pickle_load(bytes);
  fin.close();
  for (auto &item : res.toGenericDict()) {
    at::IValue key = item.key();
    at::IValue val = item.value();
    strweights_[key.toString()->string()] = (void *)&(val.toTensor());
  }
}

void ReproLiteLib::parseCommandLine(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::cout << "repro_path flag set to '" << reproPathOpt;
  jsonPathOpt_ = checkPath(reproPathOpt + "/graph_ir.json", true);
  configPathOpt_ = checkPath(reproPathOpt + "/config.spec", true);
  weightsPathOpt_ = checkPath(reproPathOpt + "/weights.pt", true);
}

void ReproLiteLib::run() {
  // Create FXGlowCompileSpec.
  loadFromAFG();
  auto backend = glow::FBABackend();
  auto compiledOrErr = backend.compileFXToCompiledResults(
      folly::parseJson(serializedGraphJson_), strweights_, config_);
}
