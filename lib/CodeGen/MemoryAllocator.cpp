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

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Support/Memory.h"

#include "llvm/Support/Casting.h"

using namespace glow;

const size_t MemoryAllocator::npos = -1;

size_t MemoryAllocator::allocate(size_t size) {
  // Always allocate buffers properly aligned to hold values of any type.
  size = alignedSize(size, TensorAlignment);
  size_t prev = 0;
  for (auto it = allocations_.begin(), e = allocations_.end(); it != e; it++) {
    if (it->begin_ - prev >= size) {
      allocations_.emplace(it, prev, prev + size);
      maxMemoryAllocated_ = std::max(maxMemoryAllocated_, prev + size);
      return prev;
    }
    prev = it->end_;
  }
  // Could not find a place for the new buffer in the middle of the list. Push
  // the new allocation to the end of the stack.

  // Check that we are not allocating memory beyond the pool size.
  if (poolSize_ && (prev + size) > poolSize_) {
    return npos;
  }

  allocations_.emplace_back(prev, prev + size);
  maxMemoryAllocated_ = std::max(maxMemoryAllocated_, prev + size);
  return prev;
}

void MemoryAllocator::deallocate(size_t ptr) {
  for (auto it = allocations_.begin(), e = allocations_.end(); it != e; it++) {
    if (it->begin_ == ptr) {
      allocations_.erase(it);
      return;
    }
  }
  assert(false && "Unknown buffer to allocate");
}
