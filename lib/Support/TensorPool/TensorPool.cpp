/*
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

#include "glow/Support/TensorPool.h"

namespace glow {

Tensor *TensorPool::get(TypeRef ty) {
  stats_.totalGets++;

  std::unique_lock<std::mutex> l(lock_);

  auto it = pools_.find(*ty);

  if (it == pools_.end()) {
    if (preventInlineAllocs_) {
      return nullptr;
    }

    stats_.totalTypes++;
    it = pools_.emplace(*ty, std::vector<Tensor *>()).first;
  }

  if (it->second.empty()) {
    if (preventInlineAllocs_) {
      return nullptr;
    }

    // Don't need to alloc under the lock.
    l.unlock();
    stats_.totalAllocs++;
    stats_.inlineAllocs++;
    // Don't add it to the queue because it's being claimed now.
    return new Tensor(ty, this);
  }

  auto &queue = it->second;
  Tensor *t = std::move(queue.back());
  queue.pop_back();
  stats_.currentBuffers--;
  return t;
}

void TensorPool::reclaim(Tensor *t) {
  std::lock_guard<std::mutex> l(lock_);
  auto it = pools_.find(t->getType());
  assert(it != pools_.end() && "Type has not been initialized");
  stats_.totalReclaims++;
  stats_.currentBuffers++;
  it->second.push_back(t);
}

void TensorPool::reserve(TypeRef ty, size_t count) {
  std::vector<Tensor *> temp;
  temp.reserve(count);
  for (unsigned i = 0; i < count; ++i) {
    stats_.totalAllocs++;
    temp.push_back(new Tensor(ty, this));
  }

  {
    std::lock_guard<std::mutex> l(lock_);
    auto it = pools_.find(*ty);
    if (it == pools_.end()) {
      stats_.totalTypes++;
    }

    std::vector<Tensor *> &queue = pools_[*ty];
    std::move(temp.begin(), temp.end(), std::back_inserter(queue));
    stats_.currentBuffers += count;
  }
}

void TensorPool::clear() {
  std::lock_guard<std::mutex> l(lock_);
  for (auto &p : pools_) {
    stats_.currentBuffers -= p.second.size();
    for (auto *t : p.second) {
      delete t;
      stats_.totalFrees++;
    }
    p.second.clear();
  }
  assert(stats_.currentBuffers == 0);
}

} // namespace glow
