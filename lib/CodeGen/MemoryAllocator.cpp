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

float MemoryAllocator::getAllocationEfficiency() const {
  if (maxUsedSize_ != 0) {
    return static_cast<float>(maxLiveSize_) / static_cast<float>(maxUsedSize_);
  } else {
    return 0;
  }
};

uint64_t MemoryAllocator::getEffectiveSize(uint64_t size) const {
  return alignedSize(size, alignment_);
}

uint64_t MemoryAllocator::allocate(uint64_t size, Handle handle) {
  // Always allocate buffers properly aligned to hold values of any type.
  uint64_t segmentSize = getEffectiveSize(size);
  uint64_t prev = 0;
  for (auto it = segments_.begin(), e = segments_.end(); it != e; it++) {
    if (it->begin_ - prev >= segmentSize) {
      segments_.emplace(it, prev, prev + segmentSize);
      maxUsedSize_ = std::max(maxUsedSize_, prev + segmentSize);
      liveSize_ += segmentSize;
      maxLiveSize_ = std::max(maxLiveSize_, liveSize_);
      setHandle(prev, size, handle);
      return prev;
    }
    prev = it->end_;
  }
  // Could not find a place for the new buffer in the middle of the list. Push
  // the new allocation to the end of the stack.

  // Check that we are not allocating memory beyond the pool size.
  if (memorySize_ && (prev + segmentSize) > memorySize_) {
    return npos;
  }

  segments_.emplace_back(prev, prev + segmentSize);
  maxUsedSize_ = std::max(maxUsedSize_, prev + segmentSize);
  liveSize_ += segmentSize;
  maxLiveSize_ = std::max(maxLiveSize_, liveSize_);
  setHandle(prev, size, handle);
  return prev;
}

void MemoryAllocator::evictFirstFit(uint64_t size,
                                    const std::set<Handle> &mustNotEvict,
                                    std::vector<Handle> &evicted) {
  // Use the first fit strategy to evict allocated blocks.
  size = getEffectiveSize(size);
  bool hasSeenNonEvicted{false};
  uint64_t startAddress = 0;
  uint64_t begin = 0;
  llvm::SmallVector<std::pair<Segment, Handle>, 16> evictionCandidates;
  for (auto it = segments_.begin(), e = segments_.end(); it != e; it++) {
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
      (!hasSeenNonEvicted && memorySize_ >= size)) {
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
  for (auto it = segments_.begin(), e = segments_.end(); it != e; it++) {
    if (it->begin_ == ptr) {
      segments_.erase(it);
      liveSize_ -= it->size();
      maxLiveSize_ = std::max(maxLiveSize_, liveSize_);
      addrToHandleMap_.erase(ptr);
      handleToSegmentMap_.erase(handle);
      return;
    }
  }
  llvm_unreachable("Unknown buffer to deallocate");
}

bool MemoryAllocator::hasHandle(uint64_t address) const {
  auto it = addrToHandleMap_.find(address);
  return it != addrToHandleMap_.end();
}

MemoryAllocator::Handle MemoryAllocator::getHandle(uint64_t address) const {
  auto it = addrToHandleMap_.find(address);
  assert(it != addrToHandleMap_.end() && "Unknown address");
  return it->second;
}

bool MemoryAllocator::hasAddress(Handle handle) const {
  auto it = handleToSegmentMap_.find(handle);
  return it != handleToSegmentMap_.end();
}

Segment MemoryAllocator::getSegment(Handle handle) const {
  auto it = handleToSegmentMap_.find(handle);
  assert(it != handleToSegmentMap_.end() && "Unknown handle");
  return it->second;
}

uint64_t MemoryAllocator::getAddress(Handle handle) const {
  return getSegment(handle).begin_;
}

uint64_t MemoryAllocator::getSize(Handle handle) const {
  return getSegment(handle).size();
}

void MemoryAllocator::setHandle(uint64_t ptr, uint64_t size, Handle handle) {
  // TODO: Check that ptr is an allocated address.
  assert(contains(ptr) && "The address is not allocated");
  assert(!hasHandle(ptr) && "The address has an associated handle already");
  addrToHandleMap_[ptr] = handle;
  handleToSegmentMap_.insert(std::make_pair(handle, Segment(ptr, ptr + size)));
}

bool MemoryAllocator::verifyAllocations(
    const std::list<Allocation> &allocList) const {

  // Allocations length must be even.
  if (allocList.size() % 2) {
    return false;
  }

  // Number of ALLOC must be equal to number of FREE.
  size_t numAlloc = 0;
  for (const auto &alloc : allocList) {
    if (alloc.alloc_) {
      numAlloc++;
    }
  }
  if (numAlloc != (allocList.size() / 2)) {
    return false;
  }

  // Verify each ALLOC has an associated FREE following it.
  // Verify each ALLOC has a unique handle.
  std::list<Handle> allocHandleList;
  for (auto allocIt = allocList.begin(); allocIt != allocList.end();
       ++allocIt) {
    if (!allocIt->alloc_) {
      continue;
    }
    // Find a FREE instruction associated to this ALLOC.
    auto allocHandle = allocIt->handle_;
    bool freeFound = false;
    for (auto it = allocIt; it != allocList.end(); ++it) {
      if ((!it->alloc_) && (it->handle_ == allocHandle)) {
        freeFound = true;
        break;
      }
    }
    if (!freeFound) {
      return false;
    }
    // Verify ALLOC handle is unique.
    auto handleIt =
        std::find(allocHandleList.begin(), allocHandleList.end(), allocHandle);
    if (handleIt != allocHandleList.end()) {
      return false;
    }
    allocHandleList.push_back(allocHandle);
  }

  return true;
}

bool MemoryAllocator::verifySegments(
    const std::list<Allocation> &allocList) const {

  // Verify number of segments.
  if (handleToSegmentMap_.size() != (allocList.size() / 2)) {
    return false;
  }

  // Segments handles should match allocation handles.
  // Segments sizes should match allocation sizes (with alignment).
  for (const auto &alloc : allocList) {
    if (!alloc.alloc_) {
      continue;
    }
    auto it = handleToSegmentMap_.find(alloc.handle_);
    if (it == handleToSegmentMap_.end()) {
      return false;
    }
    Segment seg = it->second;
    if (seg.size() != getEffectiveSize(alloc.size_)) {
      return false;
    }
  }

  // Allocations which are simultaneously alive must be assigned non-overlapping
  // segments.
  std::list<Handle> liveHandleList;
  for (const auto &alloc : allocList) {
    auto allocHandle = alloc.handle_;
    if (alloc.alloc_) {
      // Verify current segment is not overlapping with the other live segments.
      auto it = handleToSegmentMap_.find(alloc.handle_);
      Segment seg = it->second;
      for (const auto &liveHandle : liveHandleList) {
        auto liveIt = handleToSegmentMap_.find(liveHandle);
        Segment liveSeg = liveIt->second;
        bool segOverlap = intervalsOverlap(seg.begin_, seg.end_, liveSeg.begin_,
                                           liveSeg.end_);
        if (segOverlap) {
          return false;
        }
      }
      // Add handle to live handles.
      liveHandleList.push_back(allocHandle);
    } else {
      // Remove handle from live handles.
      auto it =
          std::find(liveHandleList.begin(), liveHandleList.end(), allocHandle);
      assert(it != liveHandleList.end() && "Handle not found for removal!");
      liveHandleList.erase(it);
    }
  }
  assert(liveHandleList.empty() && "Inconsistent allocations!");

  return true;
}

uint64_t MemoryAllocator::allocateAll(const std::list<Allocation> &allocList) {

  // Reset memory allocator object.
  reset();

  // Verify allocations.
  assert(verifyAllocations(allocList) && "Allocations are invalid!");

  // If allocation list is empty then return early.
  size_t allocNum = allocList.size();
  if (allocNum == 0) {
    return 0;
  }

  // Number of buffers/segments to allocate.
  assert((allocNum % 2 == 0) &&
         "The allocation list must have an even number of entries!");
  size_t buffNum = allocNum / 2;

  // Map Handles to consecutive unique IDs between 0 and numBuff - 1 since this
  // makes the algorithm implementation easier/faster by using IDs as vector
  // indices.
  std::unordered_map<Handle, size_t> handleToIdMap;
  std::vector<Handle> idToHandleMap(buffNum);
  {
    size_t id = 0;
    for (const auto &alloc : allocList) {
      // We only map the Handles of ALLOCs.
      if (alloc.alloc_) {
        idToHandleMap[id] = alloc.handle_;
        handleToIdMap[alloc.handle_] = id;
        id++;
      }
    }
    assert(id == buffNum && "Inconsistent Handle to ID mapping!");
  }

  // -----------------------------------------------------------------------
  // Get overall information about all the buffers.
  // -----------------------------------------------------------------------
  // Array with the size for each buffer.
  std::vector<uint64_t> buffSizeArray(buffNum);

  // Array with the start/stop time (both inclusive) for each buffer.
  std::vector<uint64_t> buffTimeStart(buffNum);
  std::vector<uint64_t> buffTimeStop(buffNum);

  // The maximum total required memory of all the live buffers reached during
  // all allocation time steps. Note that this is the best size any allocation
  // algorithm can hope for and is used to compute the allocation efficiency.
  uint64_t liveSizeMax = 0;

  // Array with the total required memory of all the live buffers for each
  // allocation time step.
  std::vector<uint64_t> liveBuffSizeArray(allocNum);

  // Array with lists of IDs of all the live buffers for each allocation time
  // step.
  std::vector<std::list<uint64_t>> liveBuffIdListArray(allocNum);

  // Gather information.
  {
    uint64_t allocIdx = 0;
    uint64_t liveBuffSize = 0;
    std::list<uint64_t> liveBuffIdList;
    for (const auto &alloc : allocList) {

      // Current buffer handle and mapped ID.
      auto buffHandle = alloc.handle_;
      auto buffId = handleToIdMap[buffHandle];

      // Current buffer size. We only use the buffer size of an ALLOC request.
      // For a FREE request we use the buffer size of the associated ALLOC.
      // We round the requested buffer size using the alignment.
      uint64_t buffSize;
      if (alloc.alloc_) {
        buffSize = getEffectiveSize(alloc.size_);
      } else {
        buffSize = buffSizeArray[buffId];
      }

      // Update buffer information.
      if (alloc.alloc_) {
        buffSizeArray[buffId] = buffSize;
        buffTimeStart[buffId] = allocIdx;
      } else {
        buffTimeStop[buffId] = allocIdx;
      }

      // Update liveness information.
      if (alloc.alloc_) {
        liveBuffSize = liveBuffSize + buffSize;
        liveBuffIdList.push_back(buffId);
      } else {
        liveBuffSize = liveBuffSize - buffSize;
        auto it =
            std::find(liveBuffIdList.begin(), liveBuffIdList.end(), buffId);
        assert(it != liveBuffIdList.end() &&
               "Buffer ID not found for removal!");
        liveBuffIdList.erase(it);
      }
      liveSizeMax = std::max(liveSizeMax, liveBuffSize);
      liveBuffSizeArray[allocIdx] = liveBuffSize;
      liveBuffIdListArray[allocIdx] = liveBuffIdList;
      allocIdx++;
    }
    assert(liveBuffSize == 0 &&
           "Mismatch between total allocated and deallocated size!");
    assert(liveBuffIdList.empty() &&
           "Mismatch between total allocated and deallocated buffers!");
  }

  // If the theoretical required memory is larger than the available memory size
  // then we return early.
  if (memorySize_ && (liveSizeMax > memorySize_)) {
    return npos;
  }

  // ---------------------------------------------------------------------------
  // Allocate all the buffers.
  // ---------------------------------------------------------------------------
  // Local map between allocated IDs and segments.
  std::unordered_map<size_t, Segment> idSegMap;

  // The maximum total memory used for segment allocation.
  uint64_t usedSizeMax = 0;

  // Allocate all buffers.
  for (size_t buffIdx = 0; buffIdx < buffNum; buffIdx++) {

    // -----------------------------------------------------------------------
    // Select buffer to allocate:
    // 1. Find maximum live allocation size.
    // 2. Take largest buffer from the maximum live allocation.
    // 3. If multiple buffers with same size, take the buffer with highest
    //    live interval.
    // -----------------------------------------------------------------------

    // Find maximum total live allocation.
    auto liveBuffSizeMaxIt =
        std::max_element(liveBuffSizeArray.begin(), liveBuffSizeArray.end());
    auto liveBuffSizeMaxIdx =
        std::distance(liveBuffSizeArray.begin(), liveBuffSizeMaxIt);
    auto &liveBuffIdList = liveBuffIdListArray[liveBuffSizeMaxIdx];

    // Find buffer with maximum size within the maximum allocation.
    uint64_t buffIdMax = 0;
    uint64_t buffSizeMax = 0;
    for (auto buffIdIter : liveBuffIdList) {
      // If size is the same choose buffer with maximum live interval.
      if (buffSizeArray[buffIdIter] == buffSizeMax) {
        auto currTime = buffTimeStop[buffIdMax] - buffTimeStart[buffIdMax];
        auto iterTime = buffTimeStop[buffIdIter] - buffTimeStart[buffIdIter];
        if (iterTime > currTime) {
          buffIdMax = buffIdIter;
        }
      }
      // Choose largest buffer.
      if (buffSizeArray[buffIdIter] > buffSizeMax) {
        buffSizeMax = buffSizeArray[buffIdIter];
        buffIdMax = buffIdIter;
      }
    }

    // Current segment ID chosen for allocation.
    auto currSegId = buffIdMax;

    // Check that this segment was not previously allocated.
    assert(idSegMap.find(currSegId) == idSegMap.end() &&
           "Segment previously allocated!");

    // -----------------------------------------------------------------------
    // Find previously allocated segments which overlap with the current segment
    // in time, that is segments which are alive at the same time with the
    // current segment. We keep only those segments and store them in buffers.
    // We also sort the found segments in increasing order of the stop address.
    // Note: The number of previous segments is usually small.
    // -----------------------------------------------------------------------
    typedef std::pair<uint64_t, uint64_t> AddressPair;

    // We initialize the "previous segments" buffers with a virtual segment of
    // size 0 since this will simplify the logic used in the following section.
    std::vector<AddressPair> prevSegAddr = {AddressPair(0, 0)};

    for (const auto &idSeg : idSegMap) {

      // Previously allocated segment.
      auto prevSegId = idSeg.first;
      auto prevSeg = idSeg.second;

      // Verify if the previous segment overlaps with current segment in time.
      bool overlap =
          intervalsOverlap(buffTimeStart[currSegId], buffTimeStop[currSegId],
                           buffTimeStart[prevSegId], buffTimeStop[prevSegId]);

      // If segment overlaps with previous then store the previous segment.
      if (overlap) {
        prevSegAddr.push_back(AddressPair(prevSeg.begin_, prevSeg.end_));
      }
    }

    // Order segments in the increasing order of the stop address.
    std::sort(prevSegAddr.begin(), prevSegAddr.end(),
              [](const AddressPair &a, const AddressPair &b) {
                return a.second < b.second;
              });

    // -----------------------------------------------------------------------
    // Find a position for the current segment by trying to allocate at the
    // end of all the previously allocated segments which were previously
    // found. Since the previous segments are ordered by their stop address
    // in ascending order this procedure is guaranteed to find a place at
    // least at the end of the last segment.
    // -----------------------------------------------------------------------

    uint64_t currSegAddrStart = 0;
    uint64_t currSegAddrStop = 0;

    for (size_t prevSegIdx = 0; prevSegIdx < prevSegAddr.size(); prevSegIdx++) {

      // Try to place current segment after this previously allocated segment.
      currSegAddrStart = prevSegAddr[prevSegIdx].second;
      currSegAddrStop = currSegAddrStart + buffSizeArray[currSegId];

      // Verify if this placement overlaps with all the other segments.
      // Note that this verification with all the previous segments is required
      // because the previous segments can overlap between themselves.
      bool overlap = false;
      for (size_t ovrSegIdx = 0; ovrSegIdx < prevSegAddr.size(); ovrSegIdx++) {
        // Check overlap.
        overlap = overlap || intervalsOverlap(currSegAddrStart, currSegAddrStop,
                                              prevSegAddr[ovrSegIdx].first,
                                              prevSegAddr[ovrSegIdx].second);
        // Early break if overlaps.
        if (overlap) {
          break;
        }
      }

      // If no overlap than we found the solution for the placement.
      if (!overlap) {
        break;
      }
    }

    // Update maximum used size.
    usedSizeMax = std::max(usedSizeMax, currSegAddrStop);

    // If max available memory is surpassed with the new segment then we stop
    // the allocation and return.
    if (memorySize_ && (usedSizeMax > memorySize_)) {
      return npos;
    }

    // Allocate current segment.
    Segment currSeg(currSegAddrStart, currSegAddrStop);
    idSegMap.insert(std::make_pair(currSegId, currSeg));

    // Update buffer liveness information.
    for (size_t allocIdx = buffTimeStart[currSegId];
         allocIdx < buffTimeStop[currSegId]; allocIdx++) {
      // Update total live sizes.
      liveBuffSizeArray[allocIdx] =
          liveBuffSizeArray[allocIdx] - buffSizeArray[currSegId];
      // Update total live IDs.
      auto &allocIds = liveBuffIdListArray[allocIdx];
      auto it = std::find(allocIds.begin(), allocIds.end(), currSegId);
      assert(it != allocIds.end() && "Buffer ID not found for removal!");
      allocIds.erase(it);
    }
  }

  // Verify again that all the buffers were allocated.
  for (size_t allocIdx = 0; allocIdx < allocNum; allocIdx++) {
    assert(liveBuffSizeArray[allocIdx] == 0 &&
           "Not all buffers were allocated!");
    assert(liveBuffIdListArray[allocIdx].empty() &&
           "Not all buffers were allocated!");
  }

  // Update the segments, handles and the max used/live memory.
  for (const auto &idSeg : idSegMap) {
    size_t id = idSeg.first;
    Segment segment = idSeg.second;
    Handle handle = idToHandleMap[id];
    segments_.push_back(segment);
    handleToSegmentMap_.insert(std::make_pair(handle, segment));
    addrToHandleMap_[segment.begin_] = handle;
  }
  maxUsedSize_ = usedSizeMax;
  maxLiveSize_ = liveSizeMax;
  liveSize_ = 0;

  // Verify segments.
  assert(verifySegments(allocList) && "Segments are invalid!");

  return usedSizeMax;
}
