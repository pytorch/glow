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

#include "glow/Backend/BackendOpRepository.h"

namespace glow {

/// Get a Backend Op Repository based on its registered name \p name.
/// Returns a nullptr if the BackendOpRepository with \p name is not registered.
BackendOpRepository *getBackendOpRepository(llvm::StringRef name) {
  BackendOpRepository *backendOpRepo =
      FactoryRegistry<std::string, BackendOpRepository>::get(name);

  if (backendOpRepo == nullptr) {
    LOG(INFO) << "List of all registered backend op repositories:";
    for (const auto &factory :
         FactoryRegistry<std::string, BackendOpRepository>::factories()) {
      LOG(INFO) << factory.first;
    }
  }
  return backendOpRepo;
}

// Get names of available backend op repositories.
std::vector<std::string> getAvailableBackendOpRepositories() {
  std::vector<std::string> backendOpRepoNames;
  const auto &factories =
      FactoryRegistry<std::string, BackendOpRepository>::factories();
  for (auto &factory : factories) {
    backendOpRepoNames.emplace_back(factory.first);
  }
  return backendOpRepoNames;
}

} // namespace glow
