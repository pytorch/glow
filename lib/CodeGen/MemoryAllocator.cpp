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
  // Zero is a valid size and shows up in some shape computations and
  // tests. To associate a Handle with the returned ptr, we
  // need to allocate a segment.
  uint64_t segmentSize = getEffectiveSize(std::max<uint64_t>(1, size));
  uint64_t prev = 0;
  for (auto it = segments_.begin(), e = segments_.end(); it != e; it++) {
    if (it->begin - prev >= segmentSize) {
      segments_.emplace(it, prev, prev + segmentSize);
      maxUsedSize_ = std::max(maxUsedSize_, prev + segmentSize);
      liveSize_ += segmentSize;
      maxLiveSize_ = std::max(maxLiveSize_, liveSize_);
      setHandle(prev, size, handle);
      return prev;
    }
    prev = it->end;
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
    if (it->begin < startAddress) {
      continue;
    }
    auto curHandle = getHandle(it->begin);
    if (mustNotEvict.count(curHandle)) {
      DEBUG_GLOW(llvm::dbgs()
                 << "Cannot evict a buffer from '" << name_ << "' : "
                 << "address: " << it->begin << " size: " << size << "\n");
      // The block cannot be evicted. Start looking after it.
      begin = it->end;
      evictionCandidates.clear();
      hasSeenNonEvicted = true;
      continue;
    }
    // Remember current block as a candidate.
    evictionCandidates.emplace_back(std::make_pair(*it, curHandle));
    // If the total to be evicted size is enough, no need to look any further.
    if (it->end - begin >= size) {
      break;
    }
  }

  if ((!evictionCandidates.empty() &&
       evictionCandidates.back().first.end - begin >= size) ||
      (!hasSeenNonEvicted && memorySize_ >= size)) {
    // Now evict all eviction candidates.
    for (auto &candidate : evictionCandidates) {
      auto &curHandle = candidate.second;
      auto &segment = candidate.first;
      (void)segment;
      DEBUG_GLOW(llvm::dbgs() << "Evict a buffer from the '" << name_ << "': "
                              << "address: " << segment.begin
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
    if (it->begin == ptr) {
      liveSize_ -= it->size();
      maxLiveSize_ = std::max(maxLiveSize_, liveSize_);
      segments_.erase(it);
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

const Segment &MemoryAllocator::getSegment(Handle handle) const {
  auto it = handleToSegmentMap_.find(handle);
  assert(it != handleToSegmentMap_.end() && "Unknown handle");
  return it->second;
}

uint64_t MemoryAllocator::getAddress(Handle handle) const {
  return getSegment(handle).begin;
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
    if (alloc.alloc) {
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
    if (!allocIt->alloc) {
      continue;
    }
    // Find a FREE instruction associated to this ALLOC.
    auto allocHandle = allocIt->handle;
    bool freeFound = false;
    for (auto it = allocIt; it != allocList.end(); ++it) {
      if ((!it->alloc) && (it->handle == allocHandle)) {
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
    allocHandleList.emplace_back(allocHandle);
  }
  return true;
}

bool MemoryAllocator::verifySegments(
    const std::list<Allocation> &allocList) const {
  // Segments handles should match allocation handles.
  // Segments sizes should match allocation sizes.
  // Segments begin addresses must be aligned.
  // Segments end addresses must not surpass the memory size.
  for (const auto &alloc : allocList) {
    if (!alloc.alloc) {
      continue;
    }
    auto it = handleToSegmentMap_.find(alloc.handle);
    if (it == handleToSegmentMap_.end()) {
      return false;
    }
    Segment seg = it->second;
    if (seg.size() != alloc.size) {
      return false;
    }
    if (seg.begin % alignment_) {
      return false;
    }
    if (memorySize_ && seg.end > memorySize_) {
      return false;
    }
  }
  // Allocations which are simultaneously alive must be assigned non-overlapping
  // segments.
  std::list<Handle> liveHandleList;
  for (const auto &alloc : allocList) {
    auto allocHandle = alloc.handle;
    if (alloc.alloc) {
      // Verify current segment is not overlapping with the other live segments.
      auto it = handleToSegmentMap_.find(alloc.handle);
      Segment seg = it->second;
      for (const auto &liveHandle : liveHandleList) {
        auto liveIt = handleToSegmentMap_.find(liveHandle);
        Segment liveSeg = liveIt->second;
        bool segOverlap =
            intervalsOverlap(seg.begin, seg.end, liveSeg.begin, liveSeg.end);
        if (segOverlap) {
          return false;
        }
      }
      // Add handle to live handles.
      liveHandleList.emplace_back(allocHandle);
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

/// Buffer information structure.
struct BuffInfo {
  // Allocation size in bytes (requested size without alignment).
  uint64_t allocSize;
  // Buffer size in bytes (effective size with alignment).
  uint64_t size;
  // Allocation start time (inclusive) for this buffer.
  uint64_t timeStart;
  // Allocation stop time (inclusive) for this buffer.
  uint64_t timeStop;
};

/// Liveness information structure for a given point in time.
struct LiveInfo {
  // Total size for all the live buffers at this point in time.
  uint64_t size;
  // List of IDs of all the live buffers at this point in time.
  std::list<uint64_t> idList;
};

/// Type definition for the function which defines the order used to allocate
/// the segments. Such a strategy is provided with the current buffer index
/// \p buffIdx, the buffer information \p buffInfoArray and liveness information
/// \p liveInfoArray. \returns the ID of the segment chosen for allocation.
using MemAllocStrategy = std::function<uint64_t(
    size_t buffIdx, const std::vector<BuffInfo> &buffInfoArray,
    const std::vector<LiveInfo> &liveInfoArray)>;

/// Utility function to retrieve the list of IDs of all the buffers within the
/// live allocation with maximum size. We must take into account the corner case
/// when all the live allocations contain buffers with size 0.
const std::list<uint64_t> &
getMaxLiveSizeBuffIdList(const std::vector<LiveInfo> &liveInfoArray) {
  size_t liveSizeMaxIdx = 0;
  uint64_t liveSizeMax = 0;
  for (size_t idx = 0, end = liveInfoArray.size(); idx < end; ++idx) {
    const auto &liveInfo = liveInfoArray[idx];
    // Skip empty live allocations.
    if (liveInfo.idList.empty()) {
      continue;
    }
    // Find maximum live allocation size. If the maximum was not updated
    // we update it regardless in case we have allocations of size 0.
    if (liveInfo.size > liveSizeMax || liveSizeMax == 0) {
      liveSizeMax = liveInfo.size;
      liveSizeMaxIdx = idx;
    }
  }
  return liveInfoArray[liveSizeMaxIdx].idList;
}

/// Memory allocation strategy based on the following logic:
/// 1. Find maximum live size.
/// 2. Find buffer with maximum size.
/// 3. If multiple buffers with same size, find maximum live interval.
static uint64_t
MaxLiveSizeMaxBuffSize(size_t buffIdx,
                       const std::vector<BuffInfo> &buffInfoArray,
                       const std::vector<LiveInfo> &liveInfoArray) {
  // Find maximum total live allocation.
  const auto &liveBuffIdList = getMaxLiveSizeBuffIdList(liveInfoArray);
  // Find buffer with maximum size within the maximum allocation.
  assert(liveBuffIdList.size() && "Live buffer ID list is empty!");
  uint64_t buffIdMax = liveBuffIdList.front();
  uint64_t buffSizeMax = buffInfoArray[buffIdMax].size;
  for (auto buffIdIter : liveBuffIdList) {
    auto buffSizeIter = buffInfoArray[buffIdIter].size;
    // Choose buffer with maximum size.
    if (buffSizeIter > buffSizeMax) {
      buffSizeMax = buffSizeIter;
      buffIdMax = buffIdIter;
    }
    // If size is the same choose buffer with maximum live interval.
    if (buffSizeIter == buffSizeMax) {
      auto currTime = buffInfoArray[buffIdMax].timeStop -
                      buffInfoArray[buffIdMax].timeStart;
      auto iterTime = buffInfoArray[buffIdIter].timeStop -
                      buffInfoArray[buffIdIter].timeStart;
      if (iterTime > currTime) {
        buffIdMax = buffIdIter;
      }
    }
  }
  return buffIdMax;
}

/// Memory allocation strategy based on the following logic:
/// 1. Find maximum live size.
/// 2. Find buffer with maximum live interval.
/// 3. If multiple buffers with same live interval, find maximum size.
static uint64_t
MaxLiveSizeMaxBuffTime(size_t buffIdx,
                       const std::vector<BuffInfo> &buffInfoArray,
                       const std::vector<LiveInfo> &liveInfoArray) {
  // Find maximum total live allocation.
  const auto &liveBuffIdList = getMaxLiveSizeBuffIdList(liveInfoArray);
  // Find buffer with maximum live interval within the maximum allocation.
  assert(liveBuffIdList.size() && "Live buffer ID list is empty!");
  uint64_t buffIdMax = 0;
  uint64_t buffTimeMax = 0;
  for (auto buffIdIter : liveBuffIdList) {
    auto buffTimeIter = buffInfoArray[buffIdIter].timeStop -
                        buffInfoArray[buffIdIter].timeStart;
    // Choose buffer with maximum live interval.
    if (buffTimeIter > buffTimeMax) {
      buffTimeMax = buffTimeIter;
      buffIdMax = buffIdIter;
    }
    // If live interval is the same choose buffer with maximum size.
    if (buffTimeIter == buffTimeMax) {
      auto currSize = buffInfoArray[buffIdMax].size;
      auto iterSize = buffInfoArray[buffIdIter].size;
      if (iterSize > currSize) {
        buffIdMax = buffIdIter;
      }
    }
  }
  return buffIdMax;
}

/// Memory allocation strategy which allocates the segments in the same order as
/// requested.
static uint64_t SameOrder(size_t buffIdx,
                          const std::vector<BuffInfo> &buffInfoArray,
                          const std::vector<LiveInfo> &liveInfoArray) {
  return buffIdx;
}

/// Array of available memory allocation strategies listed in the decreasing
/// order of the likelihood of getting the best allocation efficiency. All these
/// strategies are used one after the other in the exact order as listed here.
/// In case one such strategy provides the maximum efficiency (best theoretical
/// result) then the following ones (if any) are not used anymore.
static std::vector<MemAllocStrategy> memAllocStrategies = {
    MaxLiveSizeMaxBuffSize, MaxLiveSizeMaxBuffTime, SameOrder};

/// Utility function to allocate all the segments at once using the given
/// \p strategy and the memory size \p memorySize. The buffer information
/// is incapsulated in the array \p buffInfoArray and the buffer liveness
/// information before the allocation is incapsulated in the input array
/// \p liveInfoArrayInit. At the end of the allocation the segments are
/// written in \p idSegMap which is a map between the allocation handles
/// and the associated segments.
static uint64_t allocateAllWithStrategy(
    uint64_t memorySize, const std::vector<BuffInfo> &buffInfoArray,
    const std::vector<LiveInfo> &liveInfoArrayInit,
    std::unordered_map<size_t, Segment> &idSegMap, MemAllocStrategy strategy) {

  // Make local copy of the liveness information which is modified during
  // the allocation algorithm.
  auto liveInfoArray = liveInfoArrayInit;

  // The maximum total memory used for segment allocation.
  uint64_t usedSizeMax = 0;

  // Allocate all buffers.
  for (size_t buffIdx = 0; buffIdx < buffInfoArray.size(); buffIdx++) {

    // Choose buffer/segment to allocate based on strategy.
    uint64_t currSegId = strategy(buffIdx, buffInfoArray, liveInfoArray);

    // Check that this segment was not previously allocated.
    assert(idSegMap.find(currSegId) == idSegMap.end() &&
           "Segment previously allocated!");

    // -------------------------------------------------------------------------
    // Find previously allocated segments which overlap with the current segment
    // in time, that is segments which are alive at the same time with the
    // current segment. We keep only those segments and store them in buffers.
    // We also sort the found segments in increasing order of the stop address.
    // Note: The number of previous segments is usually small.
    // -------------------------------------------------------------------------
    typedef std::pair<uint64_t, uint64_t> AddressPair;

    // We initialize the "previous segments" buffers with a virtual segment of
    // size 0 since this will simplify the logic used in the following section.
    std::vector<AddressPair> prevSegAddr = {AddressPair(0, 0)};
    for (const auto &idSeg : idSegMap) {

      // Previously allocated segment.
      auto prevSegId = idSeg.first;
      auto prevSeg = idSeg.second;

      // Verify if the previous segment overlaps with current segment in time.
      bool overlap = intervalsOverlap(buffInfoArray[currSegId].timeStart,
                                      buffInfoArray[currSegId].timeStop,
                                      buffInfoArray[prevSegId].timeStart,
                                      buffInfoArray[prevSegId].timeStop);

      // If segment overlaps with previous then store the previous segment.
      if (overlap) {
        prevSegAddr.emplace_back(prevSeg.begin, prevSeg.end);
      }
    }

    // Order segments in the increasing order of the stop address.
    std::sort(prevSegAddr.begin(), prevSegAddr.end(),
              [](const AddressPair &a, const AddressPair &b) {
                return a.second < b.second;
              });

    // -------------------------------------------------------------------------
    // Find a position for the current segment by trying to allocate at the
    // end of all the previously allocated segments which were previously
    // found. Since the previous segments are ordered by their stop address
    // in ascending order this procedure is guaranteed to find a place at
    // least at the end of the last segment.
    // -------------------------------------------------------------------------
    uint64_t currSegAddrStart = 0;
    uint64_t currSegAddrStop = 0;
    for (size_t prevSegIdx = 0; prevSegIdx < prevSegAddr.size(); prevSegIdx++) {

      // Try to place current segment after this previously allocated segment.
      currSegAddrStart = prevSegAddr[prevSegIdx].second;
      currSegAddrStop = currSegAddrStart + buffInfoArray[currSegId].size;

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
    // the allocation and return early.
    if (usedSizeMax > memorySize) {
      return MemoryAllocator::npos;
    }

    // Allocate current segment.
    Segment currSeg(currSegAddrStart, currSegAddrStop);
    idSegMap.insert(std::make_pair(currSegId, currSeg));

    // Update buffer liveness information.
    for (size_t allocIdx = buffInfoArray[currSegId].timeStart;
         allocIdx < buffInfoArray[currSegId].timeStop; allocIdx++) {
      // Update total live size.
      liveInfoArray[allocIdx].size -= buffInfoArray[currSegId].size;
      // Update total live IDs.
      auto &allocIds = liveInfoArray[allocIdx].idList;
      auto it = std::find(allocIds.begin(), allocIds.end(), currSegId);
      assert(it != allocIds.end() && "Buffer ID not found for removal!");
      allocIds.erase(it);
    }
  }

  // Verify again that all the buffers were allocated.
  for (size_t allocIdx = 0; allocIdx < liveInfoArray.size(); allocIdx++) {
    assert(liveInfoArray[allocIdx].size == 0 &&
           "Not all buffers were allocated!");
    assert(liveInfoArray[allocIdx].idList.empty() &&
           "Not all buffers were allocated!");
  }

  return usedSizeMax;
}

void MemoryAllocator::mapHandlesToIds(
    const std::list<Allocation> &allocList,
    std::unordered_map<Handle, size_t> &handleToIdMap,
    std::vector<Handle> &idToHandleMap) {
  size_t buffNum = allocList.size() / 2;
  handleToIdMap.clear();
  idToHandleMap = std::vector<Handle>(buffNum);
  size_t id = 0;
  for (const auto &alloc : allocList) {
    // We only map the Handles of ALLOCs.
    if (alloc.alloc) {
      handleToIdMap[alloc.handle] = id;
      idToHandleMap[id] = alloc.handle;
      id++;
    }
  }
  assert(id == buffNum && "Inconsistent Handle to ID mapping!");
}

uint64_t MemoryAllocator::allocateAll(const std::list<Allocation> &allocList,
                                      bool reuseMemory) {

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

  // Effective memory size remaining for allocation taking into account the
  // previously allocated segments. For this parameter a value of 0 means no
  // remaining space while an infinite value is encoded as max(uint64).
  uint64_t effectiveMemorySize =
      memorySize_ ? memorySize_ : std::numeric_limits<uint64_t>::max();
  if (!reuseMemory) {
    assert(effectiveMemorySize >= maxUsedSize_ && "Memory size invalid!");
    effectiveMemorySize -= maxUsedSize_;
  }

  // Map Handles to consecutive unique IDs between 0 and numBuff - 1 since this
  // makes the algorithm implementation easier/faster by using IDs as vector
  // indices.
  std::unordered_map<Handle, size_t> handleToIdMap;
  std::vector<Handle> idToHandleMap;
  mapHandlesToIds(allocList, handleToIdMap, idToHandleMap);

  // -----------------------------------------------------------------------
  // Get overall information about all the buffers.
  // -----------------------------------------------------------------------
  // Buffer information.
  std::vector<BuffInfo> buffInfoArray(buffNum);

  // The maximum total required memory of all the live buffers reached during
  // all allocation time steps. Note that this is the best size any allocation
  // algorithm can hope for and is used to compute the allocation efficiency.
  uint64_t liveSizeMax = 0;

  // Liveness information for each allocation time step.
  std::vector<LiveInfo> liveInfoArray(allocNum);

  // Gather information.
  {
    uint64_t allocIdx = 0;
    uint64_t liveBuffSize = 0;
    std::list<uint64_t> liveBuffIdList;
    for (const auto &alloc : allocList) {

      // Current buffer handle and mapped ID.
      auto buffHandle = alloc.handle;
      auto buffId = handleToIdMap[buffHandle];

      // Current buffer size. We only use the buffer size of an ALLOC request.
      // For a FREE request we use the buffer size of the associated ALLOC.
      // We round the requested buffer size using the alignment.
      uint64_t buffSize;
      if (alloc.alloc) {
        buffSize = getEffectiveSize(alloc.size);
      } else {
        buffSize = buffInfoArray[buffId].size;
      }

      // Update buffer information.
      if (alloc.alloc) {
        buffInfoArray[buffId].allocSize = alloc.size;
        buffInfoArray[buffId].size = buffSize;
        buffInfoArray[buffId].timeStart = allocIdx;
      } else {
        buffInfoArray[buffId].timeStop = allocIdx;
      }

      // Update liveness information.
      if (alloc.alloc) {
        liveBuffSize = liveBuffSize + buffSize;
        liveBuffIdList.emplace_back(buffId);
      } else {
        liveBuffSize = liveBuffSize - buffSize;
        auto it =
            std::find(liveBuffIdList.begin(), liveBuffIdList.end(), buffId);
        assert(it != liveBuffIdList.end() &&
               "Buffer ID not found for removal!");
        liveBuffIdList.erase(it);
      }
      liveSizeMax = std::max(liveSizeMax, liveBuffSize);
      liveInfoArray[allocIdx].size = liveBuffSize;
      liveInfoArray[allocIdx].idList = liveBuffIdList;
      allocIdx++;
    }
    assert(liveBuffSize == 0 &&
           "Mismatch between total allocated and deallocated size!");
    assert(liveBuffIdList.empty() &&
           "Mismatch between total allocated and deallocated buffers!");
  }

  // If the theoretical required memory is larger than the available memory size
  // then we return early.
  if (liveSizeMax > effectiveMemorySize) {
    return MemoryAllocator::npos;
  }

  // ---------------------------------------------------------------------------
  // Allocate all the buffers using all the available strategies.
  // ---------------------------------------------------------------------------
  // Map between allocated IDs and segments for optimal strategy.
  std::unordered_map<size_t, Segment> idSegMap;

  // The maximum total memory used for segment allocation for optimal strategy.
  uint64_t usedSizeMax = std::numeric_limits<uint64_t>::max();

  // Iterate all the available strategies and pick the optimal one.
  size_t strategyNum = memAllocStrategies.size();
  for (size_t strategyIdx = 0; strategyIdx < strategyNum; strategyIdx++) {

    // Allocate segments using current strategy.
    std::unordered_map<size_t, Segment> idSegMapTemp;
    auto strategy = memAllocStrategies[strategyIdx];
    uint64_t usedSize =
        allocateAllWithStrategy(effectiveMemorySize, buffInfoArray,
                                liveInfoArray, idSegMapTemp, strategy);

    // If available memory is exceeded then we return early.
    if (usedSize == MemoryAllocator::npos) {
      return MemoryAllocator::npos;
    }

    // If maximum efficiency is reached we update and break early.
    if (usedSize == liveSizeMax) {
      usedSizeMax = usedSize;
      idSegMap = idSegMapTemp;
      break;
    }

    // If new optimal is obtained we update.
    if (usedSize < usedSizeMax) {
      usedSizeMax = usedSize;
      idSegMap = idSegMapTemp;
    }
  }

  // Update the segments, handles and the max used/live memory.
  assert(idSegMap.size() == (allocList.size() / 2) && "Segments are invalid!");
  for (const auto &idSeg : idSegMap) {
    size_t id = idSeg.first;
    Segment segment = idSeg.second;
    // Since the segment allocation was done using the aligned size, we restore
    // the segment size to the value of the original allocation size.
    segment.end = segment.begin + buffInfoArray[id].allocSize;
    // When memory is not reused we adjust the segment position based on the
    // previous memory usage.
    if (!reuseMemory) {
      segment.begin += maxUsedSize_;
      segment.end += maxUsedSize_;
    }
    Handle handle = idToHandleMap[id];
    segments_.emplace_back(segment);
    handleToSegmentMap_.insert(std::make_pair(handle, segment));
    addrToHandleMap_[segment.begin] = handle;
  }
  if (reuseMemory) {
    maxUsedSize_ = std::max(maxUsedSize_, usedSizeMax);
    maxLiveSize_ = std::max(maxLiveSize_, liveSizeMax);
  } else {
    maxUsedSize_ = maxUsedSize_ + usedSizeMax;
    maxLiveSize_ = maxLiveSize_ + liveSizeMax;
  }
  liveSize_ = 0;

  // Verify segments.
  assert(verifySegments(allocList) && "Segments are invalid!");

  return maxUsedSize_;
}
