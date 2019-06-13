/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "glow/Backend/Backend.h"

namespace glow {

Backend *createBackend(llvm::StringRef backendName) {
  auto *backend = FactoryRegistry<std::string, Backend>::get(backendName);

  if (backend == nullptr) {
    LOG(INFO) << "List of all registered backends:";
    for (const auto &factory :
         FactoryRegistry<std::string, Backend>::factories()) {
      LOG(INFO) << factory.first;
    }
  }
  CHECK(backend) << strFormat("Cannot find registered backend: %s",
                              backendName.data());
  return backend;
}

} // namespace glow
