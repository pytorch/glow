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

#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Nodes.h"
#include "glow/Support/TensorPool.h"

#include <glog/logging.h>

using namespace glow;

bool PlaceholderBindings::compare(const PlaceholderBindings *A,
                                  const PlaceholderBindings *B,
                                  float allowedError) {
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
    const auto &tensorA = phTensorPair.second;
    const auto *tensorB =
        B->get(B->getPlaceholderByNameSlow(placeholder->getName()));

    if (!tensorB || !tensorA.isEqual(*tensorB, allowedError,
                                     /* verbose */ true)) {
      return false;
    }
  }

  return true;
}

const Tensor *PlaceholderBindings::get(Placeholder *P) const {
  auto it = map_.find(P);
  if (it == map_.end()) {
    return nullptr;
  }

  return &it->second;
}

Tensor *PlaceholderBindings::get(Placeholder *P) {
  auto it = map_.find(P);
  if (it == map_.end()) {
    return nullptr;
  }

  return &it->second;
}

Placeholder *
PlaceholderBindings::getPlaceholderByNameSlow(llvm::StringRef name) const {
  for (auto &kv : map_) {
    if (kv.first->getName() == name) {
      return kv.first;
    }
  }
  return nullptr;
}

PlaceholderBindings::PlaceholderMapIterator
PlaceholderBindings::insert(Placeholder *P, Tensor &&T) {
  DCHECK(T.getType().isEqual(*P->getType()))
      << "Placeholder " << P->getName().str() << " has type "
      << P->getType()->toString() << " but Tensor has type "
      << T.getType().toString() << "\n";
  auto ret = map_.emplace(P, std::move(T));
  DCHECK(ret.second) << "Placeholder with name \"" << P->getName().str()
                     << "\" already registered";
  return ret.first;
}

void PlaceholderBindings::copyToTarget(llvm::StringRef name,
                                       PlaceholderBindings &dst) {
  auto *srcPH = this->getPlaceholderByNameSlow(name);
  DCHECK(srcPH) << name.str() << " does not exist in source";
  auto *dstPH = dst.getPlaceholderByNameSlow(name);
  DCHECK(dstPH) << name.str() << " does not exist in destination";
  dst.erase(dstPH);
  dst.insert(dstPH, this->get(srcPH)->clone());
}

void PlaceholderBindings::copyTrainableWeightsTo(PlaceholderBindings &dst) {
  for (auto &PH : pairs()) {
    if (PH.first->isTraining()) {
      copyToTarget(PH.first->getName(), dst);
    }
  }
}

size_t PlaceholderBindings::count(Placeholder *P) const {
  return map_.count(P);
}

void PlaceholderBindings::clear() {
  // Delete all of the tensors that are owned by the bindings.
  for (auto &PH : map_) {
    if (auto *tensorPool = PH.second.getOwningPool()) {
      tensorPool->reclaim(std::move(PH.second));
    }
  }

  map_.clear();
}

void PlaceholderBindings::erase(Placeholder *P) {
  auto &T = map_[P];
  if (auto *tensorPool = T.getOwningPool()) {
    tensorPool->reclaim(std::move(T));
  }
  map_.erase(P);
}

PlaceholderBindings PlaceholderBindings::clone() const {
  PlaceholderBindings cloned;
  for (auto &PH : map_) {
    Placeholder *P = PH.first;
    cloned.insert(P, PH.second.clone());
  }

  return cloned;
}

PlaceholderBindings
PlaceholderBindings::clone(const PlaceholderList &newPHs) const {
  PlaceholderBindings cloned;
  for (const auto &PH : map_) {
    Placeholder *P = PH.first;
    const Tensor &T = PH.second;
    auto newPHIt = std::find_if(newPHs.begin(), newPHs.end(), [=](auto *newPH) {
      return newPH->getName() == P->getName();
    });
    DCHECK(newPHIt != newPHs.end())
        << "Expected to find corresponding PH by name " << P->getName().data();
    cloned.insert(*newPHIt, T.clone());
  }

  return cloned;
}

Tensor *PlaceholderBindings::allocate(Placeholder *P) {
  DCHECK(!map_.count(P)) << "Placeholder with name \"" << P->getName().str()
                         << "\" already registered";
  Tensor T(P->getType());

  // If this Tensor needs to start zeroed, then zero it.
  if (P->allocZero()) {
    T.zero();
  }

  auto ret = map_.emplace(P, std::move(T));
  return &ret.first->second;
}

unsigned PlaceholderBindings::allocate(const std::list<Placeholder *> &lst) {
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

Placeholder *PlaceholderBindings::getFirstUnallocated(
    const std::list<Placeholder *> &lst) const {
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
    const auto &T = PH.second;
    size += T.getSizeInBytes();
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
    Tensor ptrT = orig->getUnowned();
    insert(placeholders[i], std::move(ptrT));
  }
}
