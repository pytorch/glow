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
class MemoryAllocator {
  /// A list of live buffers.
  std::list<Segment> allocations_;
  /// The size of the memory region that we can allocate segments into.
  uint64_t poolSize_;
  /// This is the high water mark for the allocated memory.
  uint64_t maxMemoryAllocated_{0};

public:
  /// A reserved value to mark invalid allocation.
  static const uint64_t npos;

  explicit MemoryAllocator(uint64_t poolSize) : poolSize_(poolSize) {}

  void reset() {
    maxMemoryAllocated_ = 0;
    allocations_.clear();
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

  /// Allocate a region of size \p size.
  /// \returns the allocated pointer, or MemoryAllocator::npos, if the
  /// allocation failed.
  uint64_t allocate(uint64_t size);

  /// Frees the allocation at \p ptr.
  void deallocate(uint64_t ptr);

  /// \returns the high water mark for the allocated memory.
  uint64_t getMaxMemoryUsage() const { return maxMemoryAllocated_; }
};

} // namespace glow

#endif // GLOW_CODEGEN_MEMORYALLOCATOR_H
