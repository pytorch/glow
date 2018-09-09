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

#include "glow/Base/Context.h"
#include "glow/Base/Tensor.h"

using namespace glow;

Tensor *Context::get(Placeholder *P) const {
  auto it = map_.find(P);
  if (it == map_.end()) {
    return nullptr;
  }

  return it->second;
}

void Context::insert(Placeholder *P, Tensor &&T) {
  assert(!map_.count(P) && "Placeholder already registered");
  // Take ownership over the tensor.
  map_[P] = new Tensor(std::move(T));
}

size_t Context::count(Placeholder *P) const { return map_.count(P); }

void Context::clear() {
  // Delete all of the tensors that are owned by the context.
  for (auto PH : map_) {
    delete PH.second;
  }

  map_.clear();
}

Context::Context(llvm::ArrayRef<Placeholder *> placeholders,
                 llvm::ArrayRef<Tensor *> inputs) {
  assert(placeholders.size() == inputs.size() &&
         "Invalid number of placeholders");

  for (size_t i = 0, e = placeholders.size(); i < e; i++) {
    auto *orig = inputs[i];
    /// Create a reference to the original tensor and hand it to the Context.
    Tensor ptrT = orig->getUnowned(orig->dims());
    insert(placeholders[i], std::move(ptrT));
  }
}
