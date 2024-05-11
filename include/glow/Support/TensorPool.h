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
#ifndef GLOW_TENSORPOOL_H
#define GLOW_TENSORPOOL_H

#include "glow/Base/Tensor.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace glow {

class TensorPool final {
private:
  struct TypeHash {
    size_t operator()(const Type &t) const { return t.equals_hash(); }
  };

  struct TypeEquals {
    bool operator()(const Type &a, const Type &b) const { return a.isEqual(b); }
  };
  /// A stack of available Tensors per Type.
  std::unordered_map<Type, std::vector<Tensor>, TypeHash, TypeEquals> pools_;

  /// Mutex around pools_;
  std::mutex lock_;

  /// Whether or not to allow allocation of new buffers if the pool is empty.
  const bool preventInlineAllocs_{false};

public:
  /// Statistics relating to the usage of the pool.
  struct Stats {
    /// The total number of Types that has ever been available in this pool.
    std::atomic<uint64_t> totalTypes{0};
    /// The number of Tensors currently allocated and available.
    std::atomic<uint64_t> currentBuffers{0};
    /// The number of Tensor allocations ever done by the pool.
    std::atomic<uint64_t> totalAllocs{0};
    /// The number of Tensor allocations that were done inline to get (as
    /// opposed to reserve).
    std::atomic<uint64_t> inlineAllocs{0};
    /// The total number of times a Tensor was retrieved from the pool.
    std::atomic<uint64_t> totalGets{0};
    /// The total number of times a Tensor was returned to the pool.
    std::atomic<uint64_t> totalReclaims{0};
    /// The total number of times a Tensor was freed (e.g. via clear()).
    std::atomic<uint64_t> totalFrees{0};
  } stats_;

  TensorPool(bool preventAllocs = false)
      : preventInlineAllocs_{preventAllocs} {}

  ~TensorPool() { clear(); }

  /// Retrieve a Tensor with type \p ty from the pool - this type must have
  /// previously been added by initialize. If the pool is empty this will
  /// allocate a new Tensor unless preventAllocs was set true at construction
  /// time.
  std::optional<Tensor> get(TypeRef ty);

  /// Return a Tensor \p t to the pool. This Tensor must have been previously
  /// allocated by this TensorPool.
  void reclaim(Tensor &&t);

  /// Add \p count elements of the provided type \p ty to the pool.
  void reserve(TypeRef ty, size_t count);

  /// Clear the pool and all allocated Tensors.
  /// Note: this does not delete tensors that were allocated by the pool but
  /// were not reclaimed.
  void clear();

  /// Return statistics about the TensorPool.
  const Stats &getStats() { return stats_; }
};
} // namespace glow

#endif
