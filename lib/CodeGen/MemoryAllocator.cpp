// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/CodeGen/MemoryAllocator.h"

#include "llvm/Support/Casting.h"

using namespace glow;

const size_t MemoryAllocator::npos = -1;

size_t MemoryAllocator::allocate(size_t size) {
  // Always allocate buffers properly aligned to hold values of any type.
  size = (size + alignof(float) - 1) & ~(alignof(float) - 1);
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
