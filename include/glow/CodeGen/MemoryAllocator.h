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
#ifndef GLOW_CODEGEN_MEMORYALLOCATOR_H
#define GLOW_CODEGEN_MEMORYALLOCATOR_H
#include <cstddef>
#include <cstdint>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace glow {

/// A POD struct that represents a single half-open allocation [start .. end).
class Segment {
public:
  /// The allocation starts at this address.
  uint64_t begin_;
  /// The allocation ends before this address (half-open interval).
  uint64_t end_;

  Segment(uint64_t begin, uint64_t end) : begin_(begin), end_(end) {}

  /// \returns the size of the interval.
  uint64_t size() const { return end_ - begin_; }

  /// \returns True if the value \p idx falls within this segment.
  bool contains(uint64_t idx) const { return idx >= begin_ && idx < end_; }
};

/// Allocates segments of memory.
/// Each allocation is associated with a user-defined handle, typically
/// representing a client-specific object, e.g. a handle can be a `Value *` and
/// represent a value whose payload is going to be stored in the allocated
/// memory block. This simplifies the clients of MemoryAllocator and allows them
/// to use higher-level client-side objects instead of raw allocated addresses
/// to refer to the allocated memory blocks.
class MemoryAllocator {
public:
  /// Type that should be used as a handle.
  using Handle = const void *;

  /// A reserved value to mark invalid allocation.
  static const uint64_t npos;

  explicit MemoryAllocator(const std::string &name, uint64_t poolSize)
      : name_(name), poolSize_(poolSize) {}

  void reset() {
    maxMemoryAllocated_ = 0;
    allocations_.clear();
    handleToAddr_.clear();
    addrToHandle_.clear();
  }

  /// \returns True if the value \p idx is within the currently allocated range.
  bool contains(uint64_t idx) const {
    for (auto &s : allocations_) {
      if (s.contains(idx)) {
        return true;
      }
    }
    return false;
  }

  /// Allocate a region of size \p size and associate a \p handle with it.
  /// \returns the allocated pointer, or MemoryAllocator::npos, if the
  /// allocation failed.
  uint64_t allocate(uint64_t size, Handle handle);

  /// Allocate a region of size \p size and associate a handle \p Handle with
  /// it. If the allocation is not possible, the allocator should try to evict
  /// some entries that are not needed at the moment, but it is not allowed to
  /// evict any entries from \p mustNotEvict set. All evicted entries are stored
  /// in the \p evicted set.
  ///
  /// \returns the allocated pointer, or MemoryAllocator::npos, if the
  /// allocation failed.
  uint64_t allocate(uint64_t size, Handle handle,
                    const std::set<Handle> &mustNotEvict,
                    std::vector<Handle> &evicted);

  /// \returns the handle currently associated with the allocation at \p
  /// address.
  Handle getHandle(uint64_t ptr) const;

  /// \returns true if there is a handle currently associated with the
  /// allocation at \p address.
  bool hasHandle(uint64_t ptr) const;

  /// \returns the address currently associated with the \p handle.
  uint64_t getAddress(Handle handle) const;

  /// \returns true if there is an address currently associated with the \p
  /// handle.
  bool hasAddress(Handle handle) const;

  /// Frees the allocation associated with \p handle.
  void deallocate(Handle handle);

  /// \returns the high water mark for the allocated memory.
  uint64_t getMaxMemoryUsage() const { return maxMemoryAllocated_; }

  /// \returns the name of the memory region.
  const std::string &getName() const { return name_; }

private:
  /// The name of the memory region.
  std::string name_;
  /// A list of live buffers.
  std::list<Segment> allocations_;
  /// The size of the memory region that we can allocate segments into.
  uint64_t poolSize_;
  /// This is the high water mark for the allocated memory.
  uint64_t maxMemoryAllocated_{0};
  /// Maps allocated addresses to the currently associated handles.
  std::unordered_map<uint64_t, Handle> addrToHandle_;
  /// Maps handles to allocated addresses currently associated with them.
  std::unordered_map<Handle, uint64_t> handleToAddr_;

  /// Tries to evict some entries that are not needed at the moment to free
  /// enough memory for the allocation of \p size bytes, but it is not allowed
  /// to evict any entries from \p mustNotEvict set. All evicted entries are
  /// stored in the \p evicted set. Uses first-fit approach for finding eviction
  /// candidates.
  void evictFirstFit(uint64_t size, const std::set<Handle> &mustNotEvict,
                     std::vector<Handle> &evicted);
  /// Associates a \p handle with an allocated address \p ptr.
  void setHandle(uint64_t ptr, Handle handle);
};

} // namespace glow

#endif // GLOW_CODEGEN_MEMORYALLOCATOR_H
