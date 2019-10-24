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

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "memory-allocator"

using namespace glow;

namespace glow {
class Value;
}

/// The type of the address returned by MemoryAllocator::allocate should be at
/// least 64-bit wide.
static_assert(sizeof(decltype(MemoryAllocator::npos)) >= 8,
              "Allocated addresses should be at least 64-bit wide");

/// The type of the address returned by MemoryAllocator::allocate should be
/// unsigned
static_assert(std::is_unsigned<decltype(MemoryAllocator::npos)>{},
              "Allocated addresses should be unsigned integers");

const uint64_t MemoryAllocator::npos = -1;

uint64_t MemoryAllocator::allocate(uint64_t size, Handle handle) {
  // Always allocate buffers properly aligned to hold values of any type.
  uint64_t segmentSize = alignedSize(size, alignment_);
  uint64_t prev = 0;
  for (auto it = allocations_.begin(), e = allocations_.end(); it != e; it++) {
    if (it->begin_ - prev >= segmentSize) {
      allocations_.emplace(it, prev, prev + segmentSize);
      maxMemoryAllocated_ = std::max(maxMemoryAllocated_, prev + segmentSize);
      setHandle(prev, size, handle);
      return prev;
    }
    prev = it->end_;
  }
  // Could not find a place for the new buffer in the middle of the list. Push
  // the new allocation to the end of the stack.

  // Check that we are not allocating memory beyond the pool size.
  if (poolSize_ && (prev + segmentSize) > poolSize_) {
    return npos;
  }

  allocations_.emplace_back(prev, prev + segmentSize);
  maxMemoryAllocated_ = std::max(maxMemoryAllocated_, prev + segmentSize);
  setHandle(prev, size, handle);
  return prev;
}

void MemoryAllocator::evictFirstFit(uint64_t size,
                                    const std::set<Handle> &mustNotEvict,
                                    std::vector<Handle> &evicted) {
  // Use the first fit strategy to evict allocated blocks.
  size = alignedSize(size, alignment_);
  bool hasSeenNonEvicted{false};
  uint64_t startAddress = 0;
  uint64_t begin = 0;
  llvm::SmallVector<std::pair<Segment, Handle>, 16> evictionCandidates;
  for (auto it = allocations_.begin(), e = allocations_.end(); it != e; it++) {
    // Skip any allocations below the start address.
    if (it->begin_ < startAddress) {
      continue;
    }
    auto curHandle = getHandle(it->begin_);
    if (mustNotEvict.count(curHandle)) {
      DEBUG_GLOW(llvm::dbgs()
                 << "Cannot evict a buffer from '" << name_ << "' : "
                 << "address: " << it->begin_ << " size: " << size << "\n");
      // The block cannot be evicted. Start looking after it.
      begin = it->end_;
      evictionCandidates.clear();
      hasSeenNonEvicted = true;
      continue;
    }
    // Remember current block as a candidate.
    evictionCandidates.emplace_back(std::make_pair(*it, curHandle));
    // If the total to be evicted size is enough, no need to look any further.
    if (it->end_ - begin >= size) {
      break;
    }
  }

  if ((!evictionCandidates.empty() &&
       evictionCandidates.back().first.end_ - begin >= size) ||
      (!hasSeenNonEvicted && poolSize_ >= size)) {
    // Now evict all eviction candidates.
    for (auto &candidate : evictionCandidates) {
      auto &curHandle = candidate.second;
      auto &segment = candidate.first;
      (void)segment;
      DEBUG_GLOW(llvm::dbgs() << "Evict a buffer from the '" << name_ << "': "
                              << "address: " << segment.begin_
                              << " size: " << segment.size() << "\n");
      deallocate(curHandle);
      evicted.emplace_back(curHandle);
    }
  }
}

uint64_t MemoryAllocator::allocate(uint64_t size, Handle handle,
                                   const std::set<Handle> &mustNotEvict,
                                   std::vector<Handle> &evicted) {
  // Try the usual allocation first.
  auto ptr = allocate(size, handle);
  // If it was possible to allocate the requested block, just return it.
  if (ptr != npos) {
    return ptr;
  }
  // Allocation was not possible, try to evict something.
  // Use the first fit strategy to evict allocated blocks.
  evictFirstFit(size, mustNotEvict, evicted);
  // Try again to allocate the space. This time it should succeed.
  ptr = allocate(size, handle);
  return ptr;
}

void MemoryAllocator::deallocate(Handle handle) {
  auto ptr = getAddress(handle);
  for (auto it = allocations_.begin(), e = allocations_.end(); it != e; it++) {
    if (it->begin_ == ptr) {
      allocations_.erase(it);
      addrToHandle_.erase(ptr);
      handleToAllocInfo_.erase(handle);
      return;
    }
  }
  llvm_unreachable("Unknown buffer to deallocate");
}

bool MemoryAllocator::hasHandle(uint64_t address) const {
  auto it = addrToHandle_.find(address);
  return it != addrToHandle_.end();
}

MemoryAllocator::Handle MemoryAllocator::getHandle(uint64_t address) const {
  auto it = addrToHandle_.find(address);
  assert(it != addrToHandle_.end() && "Unknown address");
  return it->second;
}

bool MemoryAllocator::hasAddress(Handle handle) const {
  auto it = handleToAllocInfo_.find(handle);
  return it != handleToAllocInfo_.end();
}

uint64_t MemoryAllocator::getAddress(Handle handle) const {
  auto it = handleToAllocInfo_.find(handle);
  assert(it != handleToAllocInfo_.end() && "Unknown handle");
  return it->second.begin_;
}

uint64_t MemoryAllocator::getSize(Handle handle) const {
  auto it = handleToAllocInfo_.find(handle);
  assert(it != handleToAllocInfo_.end() && "Unknown handle");
  return it->second.size();
}

void MemoryAllocator::setHandle(uint64_t ptr, uint64_t size, Handle handle) {
  // TODO: Check that ptr is an allocated address.
  assert(contains(ptr) && "The address is not allocated");
  assert(!hasHandle(ptr) && "The address has an associated handle already");
  addrToHandle_[ptr] = handle;
  handleToAllocInfo_.insert(std::make_pair(handle, Segment(ptr, ptr + size)));
}
