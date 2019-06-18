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
#include "glow/Support/TensorPool.h"

#include <glog/logging.h>

using namespace glow;

bool PlaceholderBindings::compare(const PlaceholderBindings *A,
                                  const PlaceholderBindings *B) {
  // Trivial cases.
  if (!A && !B) {
    return true;
  } else if ((!A && B) || (A && !B)) {
    return false;
  }

  // Get the map of Placeholder -> Tensor mappings within the two
  // PlaceholderBindingss.
  const PlaceholderBindings::PlaceholderMap &phMapA = A->pairs();
  const PlaceholderBindings::PlaceholderMap &phMapB = B->pairs();

  // If the maps have different sizes, the PlaceholderBindingss cannot match.
  if (phMapA.size() != phMapB.size()) {
    return false;
  }

  // Iterate through all Placeholders in A, look up the corresponding tensors
  // in A and B, and check if they match. If not, return false.
  for (const auto &phTensorPair : phMapA) {
    auto *placeholder = phTensorPair.first;
    const auto *tensorA = phTensorPair.second;
    const auto *tensorB =
        B->get(B->getPlaceholderByName(placeholder->getName()));

    if (!tensorA || !tensorB ||
        !tensorA->isEqual(*tensorB, /* allowedError */ 0.0001,
                          /* verbose */ false)) {
      return false;
    }
  }

  return true;
}

Tensor *PlaceholderBindings::get(Placeholder *P) const {
  auto it = map_.find(P);
  if (it == map_.end()) {
    return nullptr;
  }

  return it->second;
}

Placeholder *
PlaceholderBindings::getPlaceholderByName(llvm::StringRef name) const {
  auto nameIt = nameMap_.find(name);
  if (nameIt == nameMap_.end()) {
    return nullptr;
  }

  return nameIt->second;
}

void PlaceholderBindings::insert(Placeholder *P, Tensor &&T) {
  DCHECK(T.getType().isEqual(*P->getType()))
      << "Placeholder " << P->getName().str() << " has type "
      << P->getType()->toString() << " but Tensor has type "
      << T.getType().toString() << "\n";
  DCHECK(!map_.count(P)) << "Placeholder with name \"" << P->getName().str()
                         << "\" already registered";
  // Take ownership over the tensor.
  Tensor *t = new Tensor(std::move(T));
  map_[P] = t;
  nameMap_[P->getName()] = P;
}

void PlaceholderBindings::insert(Placeholder *P, Tensor *T) {
  DCHECK(!map_.count(P)) << "Placeholder with name \"" << P->getName().str()
                         << "\" already registered";
  map_[P] = T;
  nameMap_[P->getName()] = P;
}

size_t PlaceholderBindings::count(Placeholder *P) const {
  DCHECK_EQ(map_.size(), nameMap_.size())
      << "Placeholder map and name map out of sync";
  return map_.count(P);
}

void PlaceholderBindings::clear() {
  // Delete all of the tensors that are owned by the bindings.
  for (auto PH : map_) {
    if (auto *tensorPool = PH.second->getOwningPool()) {
      tensorPool->reclaim(PH.second);
    } else {
      delete PH.second;
    }
  }

  map_.clear();
  nameMap_.clear();
}

void PlaceholderBindings::erase(Placeholder *P) {
  DCHECK(nameMap_.count(P->getName()))
      << "Placeholder with name \"" << P->getName().str()
      << "\" already registered";
  nameMap_.erase(P->getName());

  auto *T = map_[P];
  if (auto *tensorPool = T->getOwningPool()) {
    tensorPool->reclaim(T);
  } else {
    delete T;
  }

  map_.erase(P);
}

PlaceholderBindings PlaceholderBindings::clone() const {
  PlaceholderBindings cloned;
  for (auto PH : map_) {
    Placeholder *P = PH.first;
    Tensor *T = PH.second;
    cloned.insert(P, T->clone());
  }

  return cloned;
}

Tensor *PlaceholderBindings::allocate(Placeholder *P) {
  DCHECK(!map_.count(P)) << "Placeholder with name \"" << P->getName().str()
                         << "\" already registered";
  Tensor *T = new Tensor(P->getType());

  // If this Tensor needs to start zeroed, then zero it.
  if (P->allocZero()) {
    T->zero();
  }

  map_[P] = T;
  nameMap_[P->getName()] = P;
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
  DCHECK_EQ(placeholders.size(), inputs.size())
      << "Invalid number of placeholders";

  for (size_t i = 0, e = placeholders.size(); i < e; i++) {
    auto *orig = inputs[i];
    /// Create a reference to the original tensor and hand it to the
    /// PlaceholderBindings.
    Tensor ptrT = orig->getUnowned(orig->dims());
    insert(placeholders[i], std::move(ptrT));
  }
}
