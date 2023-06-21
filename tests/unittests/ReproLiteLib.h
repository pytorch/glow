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

#include <llvm/ADT/StringMap.h>
#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>

#ifndef GLOW_TESTS_REPROFXLIB_H
#define GLOW_TESTS_REPROFXLIB_H

namespace glow {
struct CompiledResult;
}

class ReproLiteLib {
public:
  std::string jsonPathOpt_;
  std::string configPathOpt_;
  std::string weightsPathOpt_;
  std::string serializedGraphJson_;
  std::map<std::string, std::string> config_;
  llvm::StringMap<const void *> strweights_;
  torch::IValue weights_;

  void loadFromAFG();
  void parseCommandLine(int argc, char **argv);
  void run();
  void generateBundle(std::unique_ptr<glow::CompiledResult> cr);
};

#endif // GLOW_TESTS_REPROFXLIB_H
