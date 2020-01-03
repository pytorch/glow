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

#include "CustomPyTorchOpLoader.h"

#include <glog/logging.h>

namespace glow {
namespace {
struct CustomLoaders {
  std::vector<std::unique_ptr<CustomPyTorchOpLoader>> loaders_;
  std::unordered_map<torch::jit::Symbol, CustomPyTorchOpLoader *> loaderMap_;

  void insert(std::unique_ptr<CustomPyTorchOpLoader> loader) {
    for (const char *symbolStr : loader->getSymbols()) {
      loaderMap_[torch::jit::Symbol::fromQualString(symbolStr)] = loader.get();
    }
    loaders_.push_back(std::move(loader));
  }

  CustomPyTorchOpLoader *get(const torch::jit::Symbol &symbol) {
    auto it = loaderMap_.find(symbol);
    return it == loaderMap_.end() ? nullptr : it->second;
  }

  const std::unordered_map<torch::jit::Symbol, CustomPyTorchOpLoader *> &
  getMap() {
    return loaderMap_;
  }
};

CustomLoaders &getCustomPyTorchOpLoadersInternal() {
  static CustomLoaders customLoaders_;
  return customLoaders_;
}
} // namespace

const std::unordered_map<torch::jit::Symbol, CustomPyTorchOpLoader *> &
getCustomPyTorchOpLoaders() {
  return getCustomPyTorchOpLoadersInternal().getMap();
}

CustomPyTorchOpLoader *
getCustomPyTorchOpLoaderForSymbol(torch::jit::Symbol symbol) {
  return getCustomPyTorchOpLoadersInternal().get(symbol);
}

RegisterCustomPyTorchOpLoader::RegisterCustomPyTorchOpLoader(
    std::unique_ptr<CustomPyTorchOpLoader> loader) {
  getCustomPyTorchOpLoadersInternal().insert(std::move(loader));
}

} // namespace glow
