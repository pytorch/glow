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

#include "gtest/gtest.h"

#include <fstream>

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

using namespace glow;

TEST(MemAlloc, simple) {
  MemoryAllocator MA("test", 1000);
  void *handle = reinterpret_cast<void *>(1);

  // Can't allocate huge chunks.
  EXPECT_EQ(MA.allocate(100000, handle), MemoryAllocator::npos);

  // First chunk needs to start at zero.
  EXPECT_EQ(MA.allocate(500, handle), 0);

  // Second chunk must not be zero.
  EXPECT_NE(MA.allocate(500, handle), 0);
}

TEST(MemAlloc, holes) {
  MemoryAllocator MA("test", 1000);
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  void *handle2 = reinterpret_cast<void *>(2);
  void *handle4 = reinterpret_cast<void *>(4);

  MA.allocate(10, handle0);
  auto p1 = MA.allocate(10, handle1);
  MA.allocate(10, handle2);

  MA.deallocate(handle1);
  auto maxMemoryUsageBefore = MA.getMaxMemoryUsage();
  auto p4 = MA.allocate(10, handle4);
  auto maxMemoryUsageAfter = MA.getMaxMemoryUsage();

  // Check that p4 was allocated on top of the freed p1.
  EXPECT_EQ(p4, p1);
  // Max memory usage should not be affected, as a hole was found and used.
  EXPECT_EQ(maxMemoryUsageBefore, maxMemoryUsageAfter);

  MA.deallocate(handle0);
  MA.deallocate(handle2);
}

/// Check some properties of the first-fit allocation strategy.
TEST(MemAlloc, firstFitAllocation) {
  MemoryAllocator MA("test", 1000);
  void *handle = reinterpret_cast<void *>(10000);
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  void *handle2 = reinterpret_cast<void *>(2);
  void *handle3 = reinterpret_cast<void *>(3);
  void *handle4 = reinterpret_cast<void *>(4);

  // Allocate three blocks of sizes 30, 20 and 10 and allocate blocks of size 5
  // between them.
  auto p0 = MA.allocate(30, handle0);
  MA.allocate(5, handle);
  MA.allocate(20, handle1);
  MA.allocate(5, handle);
  auto p2 = MA.allocate(10, handle2);
  MA.allocate(5, handle);

  // Free blocks p0, p1 and p2.
  MA.deallocate(handle0);
  MA.deallocate(handle1);
  MA.deallocate(handle2);

  // Try to allocate a block of size 10.
  auto maxMemoryUsageBefore = MA.getMaxMemoryUsage();
  auto p3 = MA.allocate(10, handle3);
  auto maxMemoryUsageAfter = MA.getMaxMemoryUsage();

  // Check that p4 was allocated on top of the freed p0, because the allocator
  // uses the first-fit algorithm. Best-fit would have taken the block of p2.
  EXPECT_EQ(p3, p0);
  // Max memory usage should not be affected, as a hole was found and used.
  EXPECT_EQ(maxMemoryUsageBefore, maxMemoryUsageAfter);

  // Allocate 100 bytes. Since the first-fit cannot find any big enough hole
  // between allocations, the allocator would allocate this block in the free
  // space after all existing allocations.
  maxMemoryUsageBefore = MA.getMaxMemoryUsage();
  auto p4 = MA.allocate(100, handle4);
  maxMemoryUsageAfter = MA.getMaxMemoryUsage();
  EXPECT_GT(p4, p2);
  // Max memory usage should be increased.
  EXPECT_LT(maxMemoryUsageBefore, maxMemoryUsageAfter);
}

TEST(MemAlloc, dealloc) {
  MemoryAllocator MA("test", 1000);
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  void *handle2 = reinterpret_cast<void *>(2);
  void *handle3 = reinterpret_cast<void *>(3);
  void *handle4 = reinterpret_cast<void *>(4);
  void *handle5 = reinterpret_cast<void *>(5);
  void *handle6 = reinterpret_cast<void *>(6);

  auto p0 = MA.allocate(10, handle0);
  auto p1 = MA.allocate(10, handle1);
  auto p2 = MA.allocate(10, handle2);
  auto p3 = MA.allocate(10, handle3);

  auto p4 = MA.allocate(10, handle4);

  auto p5 = MA.allocate(0, handle5);
  auto p6 = MA.allocate(0, handle6);

  EXPECT_EQ(p0, 0);
  EXPECT_NE(p1, MemoryAllocator::npos);
  EXPECT_NE(p2, MemoryAllocator::npos);
  EXPECT_NE(p3, MemoryAllocator::npos);
  EXPECT_NE(p4, MemoryAllocator::npos);
  EXPECT_NE(p5, MemoryAllocator::npos);
  EXPECT_NE(p6, MemoryAllocator::npos);

  // Deallocate in some arbitrary order.
  MA.deallocate(handle0);
  MA.deallocate(handle5);
  MA.deallocate(handle2);
  MA.deallocate(handle1);
  MA.deallocate(handle6);
  MA.deallocate(handle3);
  // Check that it is possible to deallocate using the associated handle.
  MA.deallocate(handle4);
#ifndef NDEBUG
  // Check that deallocating a non-allocated or already deallocated buffer
  // should result in an assertion failure.
  ASSERT_DEATH_IF_SUPPORTED(MA.deallocate(handle3), "Unknown handle");
#endif
  // Check that after deallocating everything we start allocating from zero.
  EXPECT_EQ(MA.allocate(10, handle0), 0);
}

TEST(MemAlloc, dealloc2) {
  MemoryAllocator MA("test", 10000);
  std::vector<uint64_t> allocations;

  for (int i = 0; i < 100; i++) {
    // Create odd-sized allocations.
    const void *handle = reinterpret_cast<void *>(i);
    auto p0 = MA.allocate(10 + i % 4, handle);
    EXPECT_TRUE(MA.hasAddress(handle));
    EXPECT_TRUE(MA.hasHandle(p0));

    EXPECT_NE(p0, MemoryAllocator::npos);
    allocations.emplace_back(p0);

    if (allocations.size() > 20) {
      MA.deallocate(MA.getHandle(allocations[0]));
      allocations.erase(allocations.begin());
    }
  }
  // Drain the allocator.
  while (!allocations.empty()) {
    MA.deallocate(MA.getHandle(allocations[0]));
    allocations.erase(allocations.begin());
  }

  // Check that after deallocating everything we start allocating from zero.
  const void *handle = reinterpret_cast<void *>(0);
  EXPECT_EQ(MA.allocate(10, handle), 0);
}

TEST(MemAlloc, allocateToTheMax) {
  MemoryAllocator MA("test", 128);
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  auto p0 = MA.allocate(64, handle0);
  auto p1 = MA.allocate(64, handle1);

  EXPECT_EQ(p0, 0);
  EXPECT_NE(p1, MemoryAllocator::npos);

  MA.deallocate(handle0);
  MA.deallocate(handle1);

  EXPECT_EQ(MA.getMaxMemoryUsage(), 128);
}

TEST(MemAlloc, testContains) {
  MemoryAllocator MA("test", 1000);
  void *handle = reinterpret_cast<void *>(0);

  EXPECT_EQ(MA.allocate(200, handle), 0);

  // Offset 100 should be inside an allocated block.
  EXPECT_TRUE(MA.contains(100));
  // Offset 300 should not be inside an allocated block.
  EXPECT_FALSE(MA.contains(300));
}

TEST(MemAlloc, testHandles) {
  MemoryAllocator MA("test", 1000);
  // Define a set of handles to be used.
  void *handle1 = reinterpret_cast<void *>(1);
  void *handle2 = reinterpret_cast<void *>(2);
  (void)handle2;
  void *handle3 = reinterpret_cast<void *>(3);
  // Allocate a block of memory p1, but do not associate any handle with it.
  auto p1 = MA.allocate(10, handle1);
  // Check that handle1 is associated with the allocated address p1.
  EXPECT_EQ(MA.getHandle(p1), handle1);
  // Check that the address p1 is associated with the handle handle1.
  EXPECT_EQ(MA.getAddress(handle1), p1);

  // Allocate a block of memory p3 and associate a handle handle3 with it.
  auto p3 = MA.allocate(10, handle3);
  // The associated handle of p3 should be handle3.
  EXPECT_EQ(MA.getHandle(p3), handle3);
  // The address associated with handle3 should be p3.
  EXPECT_EQ(MA.getAddress(handle3), p3);
  // Deallocate the memory.
  MA.deallocate(handle3);
  // Check that after deallocation there is no handle is associated with the
  // allocated address.
  EXPECT_FALSE(MA.hasHandle(p3));
  // Check that after deallocation there is no address is associated with
  // handle3.
  EXPECT_FALSE(MA.hasAddress(handle3));

  MA.reset();
  p1 = MA.allocate(10, handle1);
#ifndef NDEBUG
  // Deallocating handle2 should result in an assertion failure.
  ASSERT_DEATH_IF_SUPPORTED(MA.deallocate(handle2), "Unknown handle");
#endif
}

TEST(MemAlloc, testEviction) {
  MemoryAllocator MA("test", 1024);
  // Define a set of handles to be used.
  void *handle1 = reinterpret_cast<void *>(1);
  void *handle2 = reinterpret_cast<void *>(2);
  void *handle3 = reinterpret_cast<void *>(3);
  void *handle4 = reinterpret_cast<void *>(4);
  void *handle5 = reinterpret_cast<void *>(5);
  std::vector<const void *> evicted;

  // Allocation of 500 from 1000 bytes should not trigger any eviction.
  auto p1 = MA.allocate(500, handle1, {}, evicted);
  EXPECT_NE(p1, MA.npos);
  EXPECT_TRUE(evicted.empty());

  // Allocation of 400 from remaining 500 bytes should not trigger any eviction.
  auto p2 = MA.allocate(400, handle2, {}, evicted);
  EXPECT_NE(p2, MA.npos);
  EXPECT_TRUE(evicted.empty());

  // Allocation of 400 from remaining 100 bytes should trigger the eviction.
  auto p3 = MA.allocate(400, handle3, {}, evicted);
  // The allocation should be successful.
  EXPECT_NE(p3, MA.npos);
  EXPECT_EQ(evicted.size(), 1);

  // Allocation of 2000 bytes is impossible. It should not should trigger any
  // eviction.
  evicted.clear();
  auto p4 = MA.allocate(2000, handle4, {}, evicted);
  EXPECT_EQ(p4, MA.npos);
  EXPECT_EQ(evicted.size(), 0);

  // Allocation of 1024 bytes only possible if all other allocated blocks are
  // evicted.
  evicted.clear();
  auto p5 = MA.allocate(1024, handle5, {}, evicted);
  EXPECT_NE(p5, MA.npos);
  EXPECT_EQ(evicted.size(), 2);

  // Check how eviction works with a non-empty doNotEvict set.
  MA.reset();
  evicted.clear();
  // Allocate 3 blocks, 256 bytes each.
  p1 = MA.allocate(256, handle1, {}, evicted);
  EXPECT_EQ(p1, 0);
  p2 = MA.allocate(256, handle2, {}, evicted);
  EXPECT_EQ(p2, 256);
  p3 = MA.allocate(256, handle3, {}, evicted);
  EXPECT_EQ(p3, 512);
  // No blocks should be evicted until now.
  EXPECT_EQ(evicted.size(), 0);
  // Try to allocate a block of size 512. Without a doNotEvict set and using a
  // first-fit eviction strategy, the allocator would have to evict blocks p1
  // and p2 to satisfy this request. But due to providing a doNotEvict set which
  // forbids the eviction of p1, the allocator should evict p2 and p3 and
  // allocate the 512 bytes at the same address as p2.
  std::set<const void *> doNotEvict{handle1};
  p4 = MA.allocate(512, handle4, doNotEvict, evicted);
  EXPECT_EQ(p4, p2);
  EXPECT_EQ(evicted.size(), 2);
  EXPECT_EQ(evicted[0], handle2);
  EXPECT_EQ(evicted[1], handle3);
}

TEST(MemAlloc, testGetSize) {
  MemoryAllocator MA("test", 1024);
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  // Allocate two memory blocks and checks that the size of the allocated blocks
  // is reported correctly.
  MA.allocate(10, handle0);
  EXPECT_EQ(MA.getSize(handle0), 10);
  MA.allocate(200, handle1);
  EXPECT_EQ(MA.getSize(handle1), 200);
}

TEST(MemAlloc, testGetMemorySize) {
  MemoryAllocator MA1("test1", 1024);
  EXPECT_EQ(MA1.getMemorySize(), 1024);
  MemoryAllocator MA2("test1", 102);
  EXPECT_EQ(MA2.getMemorySize(), 102);
}

TEST(MemAlloc, testAlignment) {
  MemoryAllocator MA1("test1", 1024, 128);
  MemoryAllocator MA2("test2", 1024, 256);

  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  void *handle2 = reinterpret_cast<void *>(2);
  void *handle3 = reinterpret_cast<void *>(3);

  // Both allocators start at zero.
  auto p0 = MA1.allocate(10, handle0);
  auto p1 = MA2.allocate(10, handle1);
  EXPECT_EQ(p0, 0);
  EXPECT_EQ(p1, 0);

  // Second allocation starts at the alignment boundary.
  auto p2 = MA1.allocate(10, handle2);
  auto p3 = MA2.allocate(10, handle3);
  EXPECT_EQ(p2, 128);
  EXPECT_EQ(p3, 256);
}

/// ----------------------------------------------------------------------------
///                        Allocate all segments at once
/// ----------------------------------------------------------------------------
/// Test efficiency for allocateAll.
TEST(MemAlloc, testAllocateAllEfficiency) {

  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  void *handle2 = reinterpret_cast<void *>(2);

  // Simple allocate.
  MemoryAllocator MA1("test", 0, 10);
  MA1.allocate(10, handle0);
  MA1.allocate(10, handle1);
  MA1.deallocate(handle0);
  MA1.allocate(20, handle2);
  MA1.deallocate(handle1);
  MA1.deallocate(handle2);
  EXPECT_EQ(MA1.getMaxMemoryUsage(), 40);
  EXPECT_FLOAT_EQ(MA1.getAllocationEfficiency(), 0.75);

  // Allocate all.
  MemoryAllocator MA2("test", 0, 10);
  std::list<Allocation> allocList;
  allocList.emplace_back(handle0, true, 10);
  allocList.emplace_back(handle1, true, 10);
  allocList.emplace_back(handle0, false, 0);
  allocList.emplace_back(handle2, true, 20);
  allocList.emplace_back(handle1, false, 0);
  allocList.emplace_back(handle2, false, 0);
  MA2.allocateAll(allocList);
  EXPECT_EQ(MA2.getMaxMemoryUsage(), 30);
  EXPECT_FLOAT_EQ(MA2.getAllocationEfficiency(), 1.00);
}

/// Test alignment for allocateAll.
TEST(MemAlloc, testAllocateAllAlignment) {
  MemoryAllocator MA("test", 192, 64);

  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);

  std::list<Allocation> allocList;
  allocList.emplace_back(handle0, true, 65);
  allocList.emplace_back(handle1, true, 64);
  allocList.emplace_back(handle0, false, 0);
  allocList.emplace_back(handle1, false, 0);
  uint64_t usedSize = MA.allocateAll(allocList);

  EXPECT_EQ(usedSize, 192);
  EXPECT_EQ(MA.getSize(handle0), 65);
  EXPECT_EQ(MA.getSize(handle1), 64);
  EXPECT_EQ(MA.getAddress(handle0), 0);
  EXPECT_EQ(MA.getAddress(handle1), 128);
  EXPECT_EQ(MA.getSegment(handle0).size(), 65);
  EXPECT_EQ(MA.getSegment(handle1).size(), 64);
  EXPECT_EQ(MA.getSegment(handle0).begin, 0);
  EXPECT_EQ(MA.getSegment(handle1).begin, 128);
  EXPECT_EQ(MA.getMaxMemoryUsage(), 192);
  EXPECT_FLOAT_EQ(MA.getAllocationEfficiency(), 1.0);
}

/// Test invalid number of allocations for allocateAll.
TEST(MemAlloc, testAllocateAllInvalidNumAllocs) {
  MemoryAllocator MA("test", 0, 64);
  void *handle = reinterpret_cast<void *>(0);
  std::list<Allocation> allocList;
  allocList.emplace_back(handle, true, 10);
#ifndef NDEBUG
  ASSERT_DEATH_IF_SUPPORTED(MA.allocateAll(allocList),
                            "Allocations are invalid!");
#endif
}

/// Test invalid FREE before ALLOC for allocateAll.
TEST(MemAlloc, testAllocateAllInvalidFreeBeforeAlloc) {
  MemoryAllocator MA("test", 0, 64);
  void *handle = reinterpret_cast<void *>(0);
  std::list<Allocation> allocList;
  allocList.emplace_back(handle, false, 0);
  allocList.emplace_back(handle, true, 10);
#ifndef NDEBUG
  ASSERT_DEATH_IF_SUPPORTED(MA.allocateAll(allocList),
                            "Allocations are invalid!");
#endif
}

/// Test invalid handles for allocateAll.
TEST(MemAlloc, testAllocateAllInvalidHandles1) {
  MemoryAllocator MA("test", 0, 64);
  void *handle = reinterpret_cast<void *>(0);
  std::list<Allocation> allocList;
  allocList.emplace_back(handle, true, 10);
  allocList.emplace_back(handle, true, 10);
  allocList.emplace_back(handle, false, 0);
  allocList.emplace_back(handle, false, 0);
#ifndef NDEBUG
  ASSERT_DEATH_IF_SUPPORTED(MA.allocateAll(allocList),
                            "Allocations are invalid!");
#endif
}

/// Test invalid handles for allocateAll.
TEST(MemAlloc, testAllocateAllInvalidHandles2) {
  MemoryAllocator MA("test", 0, 64);
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(0);
  std::list<Allocation> allocList;
  allocList.emplace_back(handle0, true, 10);
  allocList.emplace_back(handle0, true, 10);
  allocList.emplace_back(handle1, false, 0);
  allocList.emplace_back(handle1, false, 0);
#ifndef NDEBUG
  ASSERT_DEATH_IF_SUPPORTED(MA.allocateAll(allocList),
                            "Allocations are invalid!");
#endif
}

/// Test allocating segment of size 0 with allocateAll.
TEST(MemAlloc, testAllocateAllSize0) {
  MemoryAllocator MA("test", 0, 64);
  void *handle = reinterpret_cast<void *>(0);
  std::list<Allocation> allocList;
  allocList.emplace_back(handle, true, 0);
  allocList.emplace_back(handle, false, 0);
  uint64_t usedSize = MA.allocateAll(allocList);
  EXPECT_EQ(usedSize, 0);
  EXPECT_EQ(MA.getSize(handle), 0);
  EXPECT_EQ(MA.getAddress(handle), 0);
  EXPECT_EQ(MA.getSegment(handle).size(), 0);
  EXPECT_EQ(MA.getSegment(handle).begin, 0);
}

/// Test allocating multiple segments of size 0 with allocateAll.
TEST(MemAlloc, testAllocateAllMultipleSize0) {
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  void *handle2 = reinterpret_cast<void *>(2);
  void *handle3 = reinterpret_cast<void *>(3);
  void *handle4 = reinterpret_cast<void *>(4);
  void *handle5 = reinterpret_cast<void *>(5);

  MemoryAllocator MA("test", 0, 10);
  std::list<Allocation> allocList;
  allocList.emplace_back(handle5, true, 0);
  allocList.emplace_back(handle0, true, 10);
  allocList.emplace_back(handle3, true, 0);
  allocList.emplace_back(handle1, true, 10);
  allocList.emplace_back(handle4, true, 0);
  allocList.emplace_back(handle0, false, 0);
  allocList.emplace_back(handle2, true, 20);
  allocList.emplace_back(handle3, false, 0);
  allocList.emplace_back(handle1, false, 0);
  allocList.emplace_back(handle4, false, 0);
  allocList.emplace_back(handle2, false, 0);
  allocList.emplace_back(handle5, false, 0);

  uint64_t usedSize = MA.allocateAll(allocList);
  EXPECT_EQ(usedSize, 30);
  EXPECT_EQ(MA.getMaxMemoryUsage(), 30);
  EXPECT_FLOAT_EQ(MA.getAllocationEfficiency(), 1.00);
  EXPECT_EQ(MA.getSize(handle3), 0);
  EXPECT_EQ(MA.getAddress(handle3), 0);
  EXPECT_EQ(MA.getSegment(handle3).size(), 0);
  EXPECT_EQ(MA.getSegment(handle3).begin, 0);
  EXPECT_EQ(MA.getSize(handle4), 0);
  EXPECT_EQ(MA.getAddress(handle4), 0);
  EXPECT_EQ(MA.getSegment(handle4).size(), 0);
  EXPECT_EQ(MA.getSegment(handle4).begin, 0);
  EXPECT_EQ(MA.getSize(handle5), 0);
  EXPECT_EQ(MA.getAddress(handle5), 0);
  EXPECT_EQ(MA.getSegment(handle5).size(), 0);
  EXPECT_EQ(MA.getSegment(handle5).begin, 0);
}

/// Test empty allocs for allocateAll.
TEST(MemAlloc, testAllocateAllEmptyAlloc) {
  MemoryAllocator MA("test", 0, 64);
  std::list<Allocation> allocList;
  uint64_t usedSize = MA.allocateAll(allocList);
  EXPECT_EQ(usedSize, 0);
}

/// Test memory overflow for allocateAll.
TEST(MemAlloc, testAllocateAllMemOverflow) {
  MemoryAllocator MA("test", 10, 64);
  void *handle = reinterpret_cast<void *>(0);
  std::list<Allocation> allocList;
  allocList.emplace_back(handle, true, 100);
  allocList.emplace_back(handle, false, 0);
  uint64_t usedSize = MA.allocateAll(allocList);
  EXPECT_EQ(usedSize, MemoryAllocator::npos);
#ifndef NDEBUG
  ASSERT_DEATH_IF_SUPPORTED(MA.getSize(handle), "Unknown handle");
#endif
}

/// Utility function to test allocation for a given model using \p alignment
/// and the allocations \p allocs. The expected memory usage and efficiency
/// are \p expectedUsedSize and \p expectedEfficiency.
static void testAllocateAllForModel(size_t alignment,
                                    const std::list<Allocation> &allocs,
                                    uint64_t expectedUsedSize,
                                    float expectedEfficiency) {
  MemoryAllocator MA("mem", 0, alignment);
  uint64_t usedSize = MA.allocateAll(allocs);
  EXPECT_EQ(usedSize, expectedUsedSize);
  for (const auto &alloc : allocs) {
    if (alloc.alloc) {
      EXPECT_EQ(MA.getSize(alloc.handle), alloc.size);
    }
  };
  EXPECT_FLOAT_EQ(MA.getAllocationEfficiency(), expectedEfficiency);
}

/// Test memory allocation for multiple models using a text file with the
/// following format:
/// MODEL <model_name>
/// ALIGN <alignment>
/// ALLOC <id> <size>
/// FREE <id>
/// MEM <expected_memory_usage>
/// EFF <expected_efficiency>
TEST(MemAlloc, testAllocateAllForModels) {
  std::string modelName;
  unsigned modelIdx = 1;
  size_t alignment;
  uint64_t expectedUsedSize;
  float expectedEfficiency;
  std::list<Allocation> allocs;
  std::ifstream fs;
  std::string filename(GLOW_DATA_PATH
                       "tests/unittests/MemoryAllocatorTestModels.txt");
  fs.open(filename);
  assert(fs.is_open() && "Error opening file!");
  std::string line;
  while (std::getline(fs, line)) {
    // Skip comments.
    if (line.front() == '#') {
      continue;
    }
    // Read model parameters.
    std::stringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "MODEL") {
      ss >> modelName;
    } else if (key == "ALIGN") {
      ss >> alignment;
    } else if (key == "ALLOC") {
      size_t id;
      uint64_t size;
      ss >> id >> size;
      allocs.emplace_back(id, /* alloc */ true, size);
    } else if (key == "FREE") {
      size_t id;
      ss >> id;
      allocs.emplace_back(id, /* alloc */ false, 0);
    } else if (key == "MEM") {
      ss >> expectedUsedSize;
    } else if (key == "EFF") {
      ss >> expectedEfficiency;
    }
    // Test allocation for model.
    if (key == "EFF") {
      std::cout << "[" << modelIdx++
                << "] Testing memory allocation for model: " << modelName
                << "\n";
      testAllocateAllForModel(alignment, allocs, expectedUsedSize,
                              expectedEfficiency);
      allocs.clear();
    }
  }
  fs.close();
}

/// Test allocating multiple functions with memory reusage for allocateAll.
TEST(MemAlloc, testAllocateAllMultipleFunctionsWithReuse) {

  // Allocation sequence for 1st function.
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  std::list<Allocation> allocList1;
  allocList1.emplace_back(handle0, true, 9);
  allocList1.emplace_back(handle1, true, 9);
  allocList1.emplace_back(handle0, false, 0);
  allocList1.emplace_back(handle1, false, 0);

  // Allocation sequence for 2nd function.
  void *handle2 = reinterpret_cast<void *>(2);
  void *handle3 = reinterpret_cast<void *>(3);
  std::list<Allocation> allocList2;
  allocList2.emplace_back(handle2, true, 19);
  allocList2.emplace_back(handle3, true, 19);
  allocList2.emplace_back(handle2, false, 0);
  allocList2.emplace_back(handle3, false, 0);

  // Allocate all for both functions.
  MemoryAllocator MA("test", 0, 10);
  uint64_t usedSize1 = MA.allocateAll(allocList1);
  uint64_t usedSize2 = MA.allocateAll(allocList2, /* reuseMemory */ true);

  EXPECT_EQ(usedSize1, 20);
  EXPECT_EQ(usedSize2, 40);
  EXPECT_EQ(MA.getSize(handle0), 9);
  EXPECT_EQ(MA.getSize(handle1), 9);
  EXPECT_EQ(MA.getSize(handle2), 19);
  EXPECT_EQ(MA.getSize(handle3), 19);
  EXPECT_EQ(MA.getSegment(handle0).size(), 9);
  EXPECT_EQ(MA.getSegment(handle1).size(), 9);
  EXPECT_EQ(MA.getSegment(handle2).size(), 19);
  EXPECT_EQ(MA.getSegment(handle3).size(), 19);
  EXPECT_EQ(MA.getAddress(handle0), 0);
  EXPECT_EQ(MA.getAddress(handle1), 10);
  EXPECT_EQ(MA.getAddress(handle2), 0);
  EXPECT_EQ(MA.getAddress(handle3), 20);
  EXPECT_EQ(MA.getSegment(handle0).begin, 0);
  EXPECT_EQ(MA.getSegment(handle1).begin, 10);
  EXPECT_EQ(MA.getSegment(handle2).begin, 0);
  EXPECT_EQ(MA.getSegment(handle3).begin, 20);
  EXPECT_EQ(MA.getMaxMemoryUsage(), 40);
  EXPECT_FLOAT_EQ(MA.getAllocationEfficiency(), 1.0);
}

/// Test allocating multiple functions without memory reusage for allocateAll.
TEST(MemAlloc, testAllocateAllMultipleFunctionsWithoutReuse) {

  // Allocation sequence for 1st function.
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  std::list<Allocation> allocList1;
  allocList1.emplace_back(handle0, true, 9);
  allocList1.emplace_back(handle1, true, 9);
  allocList1.emplace_back(handle0, false, 0);
  allocList1.emplace_back(handle1, false, 0);

  // Allocation sequence for 2nd function.
  void *handle2 = reinterpret_cast<void *>(2);
  void *handle3 = reinterpret_cast<void *>(3);
  std::list<Allocation> allocList2;
  allocList2.emplace_back(handle2, true, 19);
  allocList2.emplace_back(handle3, true, 19);
  allocList2.emplace_back(handle2, false, 0);
  allocList2.emplace_back(handle3, false, 0);

  // Allocate all for both functions.
  MemoryAllocator MA("test", 0, 10);
  uint64_t usedSize1 = MA.allocateAll(allocList1);
  uint64_t usedSize2 = MA.allocateAll(allocList2, /* reuseMemory */ false);

  EXPECT_EQ(usedSize1, 20);
  EXPECT_EQ(usedSize2, 60);
  EXPECT_EQ(MA.getSize(handle0), 9);
  EXPECT_EQ(MA.getSize(handle1), 9);
  EXPECT_EQ(MA.getSize(handle2), 19);
  EXPECT_EQ(MA.getSize(handle3), 19);
  EXPECT_EQ(MA.getSegment(handle0).size(), 9);
  EXPECT_EQ(MA.getSegment(handle1).size(), 9);
  EXPECT_EQ(MA.getSegment(handle2).size(), 19);
  EXPECT_EQ(MA.getSegment(handle3).size(), 19);
  EXPECT_EQ(MA.getAddress(handle0), 0);
  EXPECT_EQ(MA.getAddress(handle1), 10);
  EXPECT_EQ(MA.getAddress(handle2), 20);
  EXPECT_EQ(MA.getAddress(handle3), 40);
  EXPECT_EQ(MA.getSegment(handle0).begin, 0);
  EXPECT_EQ(MA.getSegment(handle1).begin, 10);
  EXPECT_EQ(MA.getSegment(handle2).begin, 20);
  EXPECT_EQ(MA.getSegment(handle3).begin, 40);
  EXPECT_EQ(MA.getMaxMemoryUsage(), 60);
  EXPECT_FLOAT_EQ(MA.getAllocationEfficiency(), 1.0);
}

/// Test allocating multiple functions without memory reusage and in which the
/// allocation for 2nd function fails due to unsufficient memory.
TEST(MemAlloc, testAllocateAllMultipleFunctionsWithoutReuseFail) {

  // Allocation sequence for 1st function.
  void *handle0 = reinterpret_cast<void *>(0);
  void *handle1 = reinterpret_cast<void *>(1);
  std::list<Allocation> allocList1;
  allocList1.emplace_back(handle0, true, 9);
  allocList1.emplace_back(handle1, true, 9);
  allocList1.emplace_back(handle0, false, 0);
  allocList1.emplace_back(handle1, false, 0);

  // Allocation sequence for 2nd function.
  void *handle2 = reinterpret_cast<void *>(2);
  void *handle3 = reinterpret_cast<void *>(3);
  std::list<Allocation> allocList2;
  allocList2.emplace_back(handle2, true, 19);
  allocList2.emplace_back(handle3, true, 19);
  allocList2.emplace_back(handle2, false, 0);
  allocList2.emplace_back(handle3, false, 0);

  // Allocate all for both functions.
  MemoryAllocator MA("test", 20, 10);
  uint64_t usedSize1 = MA.allocateAll(allocList1);
  uint64_t usedSize2 = MA.allocateAll(allocList2, /* reuseMemory */ false);

  EXPECT_EQ(usedSize1, 20);
  EXPECT_EQ(usedSize2, MemoryAllocator::npos);
  EXPECT_EQ(MA.getSize(handle0), 9);
  EXPECT_EQ(MA.getSize(handle1), 9);
  EXPECT_EQ(MA.getSegment(handle0).size(), 9);
  EXPECT_EQ(MA.getSegment(handle1).size(), 9);
  EXPECT_EQ(MA.getAddress(handle0), 0);
  EXPECT_EQ(MA.getAddress(handle1), 10);
  EXPECT_EQ(MA.getSegment(handle0).begin, 0);
  EXPECT_EQ(MA.getSegment(handle1).begin, 10);
  EXPECT_EQ(MA.getMaxMemoryUsage(), 20);
  EXPECT_FLOAT_EQ(MA.getAllocationEfficiency(), 1.0);
}
