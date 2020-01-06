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

#include <mutex>

namespace glow {
namespace {
struct CustomLoaders {
  /// Vector owning all registered loaders.
  std::vector<std::unique_ptr<CustomPyTorchOpLoader>> loaders_;

  /// Map from a jit symbol to the CustomPyTorchOpLoader that loads nodes of
  /// that symbol.
  std::unordered_map<torch::jit::Symbol, CustomPyTorchOpLoader *> loaderMap_;

  /// Mutex for protecting loaders_ and loaderMap_.
  // TODO: use a rw mutex here for efficiency.
  std::mutex m_;

  void insert(std::unique_ptr<CustomPyTorchOpLoader> loader) {
    std::lock_guard<std::mutex> g(m_);
    for (const char *symbolStr : loader->getSymbols()) {
      loaderMap_[torch::jit::Symbol::fromQualString(symbolStr)] = loader.get();
    }
    loaders_.push_back(std::move(loader));
  }

  CustomPyTorchOpLoader *get(const torch::jit::Symbol &symbol) {
    std::lock_guard<std::mutex> g(m_);
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
