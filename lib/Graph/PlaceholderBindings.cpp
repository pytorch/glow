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

#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Nodes.h"

using namespace glow;

Tensor *PlaceholderBindings::get(const Placeholder *P) const {
  auto it = map_.find(P);
  if (it == map_.end()) {
    return nullptr;
  }

  return it->second;
}

void PlaceholderBindings::insert(const Placeholder *P, Tensor &&T) {
  assert(!map_.count(P) && "Placeholder already registered");
  // Take ownership over the tensor.
  map_[P] = new Tensor(std::move(T));
}

size_t PlaceholderBindings::count(const Placeholder *P) const {
  return map_.count(P);
}

void PlaceholderBindings::clear() {
  // Delete all of the tensors that are owned by the bindings.
  for (auto PH : map_) {
    delete PH.second;
  }

  map_.clear();
}

PlaceholderBindings PlaceholderBindings::clone() const {
  PlaceholderBindings cloned;
  for (auto PH : map_) {
    const Placeholder *P = PH.first;
    Tensor *T = PH.second;
    cloned.insert(P, T->clone());
  }

  return cloned;
}

Tensor *PlaceholderBindings::allocate(const Placeholder *P) {
  assert(!map_.count(P) && "Placeholder already registered");
  Tensor *T = new Tensor(P->getType());
  map_[P] = T;
  return T;
}

unsigned PlaceholderBindings::allocate(std::list<Placeholder *> &lst) {
  unsigned allocated = 0;
  // For each placeholder in the list:
  for (Placeholder *P : lst) {
    // Don't allocate tensors for placeholders that are already allocated.
    if (this->count(P)) {
      continue;
    }

    // Allocate a tensor to back P.
    allocate(P);
    allocated++;
  }
  return allocated;
}

Placeholder *
PlaceholderBindings::getFirstUnallocated(std::list<Placeholder *> &lst) const {
  // For each placeholder in the list:
  for (Placeholder *P : lst) {
    // If we found an unallocated placeholder then return it.
    if (!count(P))
      return P;
  }

  return nullptr;
}

uint64_t PlaceholderBindings::getDataSize() const {
  uint64_t size = 0;
  for (const auto &PH : map_) {
    Tensor *T = PH.second;
    size += T->getSizeInBytes();
  }
  return size;
}

PlaceholderBindings::PlaceholderBindings(
    llvm::ArrayRef<Placeholder *> placeholders,
    llvm::ArrayRef<Tensor *> inputs) {
  assert(placeholders.size() == inputs.size() &&
         "Invalid number of placeholders");

  for (size_t i = 0, e = placeholders.size(); i < e; i++) {
    auto *orig = inputs[i];
    /// Create a reference to the original tensor and hand it to the
    /// PlaceholderBindings.
    Tensor ptrT = orig->getUnowned(orig->dims());
    insert(placeholders[i], std::move(ptrT));
  }
}
