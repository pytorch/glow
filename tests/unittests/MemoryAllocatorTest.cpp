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

  auto p0 = MA.allocate(10, handle0);
  auto p1 = MA.allocate(10, handle1);
  auto p2 = MA.allocate(10, handle2);
  auto p3 = MA.allocate(10, handle3);

  auto p4 = MA.allocate(10, handle4);

  EXPECT_EQ(p0, 0);
  EXPECT_NE(p1, MemoryAllocator::npos);
  EXPECT_NE(p2, MemoryAllocator::npos);
  EXPECT_NE(p3, MemoryAllocator::npos);
  EXPECT_NE(p4, MemoryAllocator::npos);

  // Deallocate in some arbitrary order.
  MA.deallocate(handle0);
  MA.deallocate(handle2);
  MA.deallocate(handle1);
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
    allocations.push_back(p0);

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
// TODO1: Test allocation of multiple models
// TOOD2: Test alignment of allocated segments
// TODO3: Test getter methods using handle for size, address, segment.
// TODO4: Test returned information.

// Test for memory allocation for model cifar10_quant.tflite.
TEST(MemAlloc, testMemAllocForModel1) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(16);
  allocArray.push_back(Allocation(1, 1, 32768));
  allocArray.push_back(Allocation(2, 1, 8192));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 8192));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 2048));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 4096));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 1024));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 64));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(8, 1, 64));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(8, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 40960);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 32768);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 8192);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 8192);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 2048);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 4096);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 1024);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 64);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 64);
}

// Test for memory allocation for model lenet_quant.tflite.
TEST(MemAlloc, testMemAllocForModel2) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(14);
  allocArray.push_back(Allocation(1, 1, 11520));
  allocArray.push_back(Allocation(2, 1, 2880));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 3200));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 832));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 512));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 64));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 64));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(7, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 14400);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 11520);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 2880);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 3200);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 832);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 512);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 64);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 64);
}

// Test for memory allocation for model mobilenet_v1_0.25_224.tflite.
TEST(MemAlloc, testMemAllocForModel3) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(58);
  allocArray.push_back(Allocation(1, 1, 401408));
  allocArray.push_back(Allocation(2, 1, 401408));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 802816));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 200704));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 401408));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 401408));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 401408));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(8, 1, 100352));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 200704));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(10, 1, 200704));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 200704));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 50176));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 100352));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(14, 1, 100352));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 100352));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(16, 1, 100352));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(17, 1, 100352));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 100352));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(19, 1, 100352));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 100352));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 100352));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 100352));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(23, 1, 100352));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 25088));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(25, 1, 50176));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(26, 1, 50176));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 50176));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(28, 1, 1024));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(29, 1, 4032));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(29, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 25088);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 1024);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 4032);
}

// Test for memory allocation for model mobilenet_v1_0.50_224.tflite.
TEST(MemAlloc, testMemAllocForModel4) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(58);
  allocArray.push_back(Allocation(1, 1, 802816));
  allocArray.push_back(Allocation(2, 1, 802816));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 1605632));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 401408));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 802816));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 802816));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 802816));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(8, 1, 200704));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 401408));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(10, 1, 401408));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 401408));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 100352));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 200704));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(14, 1, 200704));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 200704));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(16, 1, 200704));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(17, 1, 200704));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 200704));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(19, 1, 200704));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 200704));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 200704));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 200704));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(23, 1, 200704));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 50176));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(25, 1, 100352));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(26, 1, 100352));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 100352));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(28, 1, 2048));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(29, 1, 4032));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(29, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 2048);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 4032);
}

// Test for memory allocation for model mobilenet_v1_0.75_224.tflite.
TEST(MemAlloc, testMemAllocForModel5) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(58);
  allocArray.push_back(Allocation(1, 1, 1204224));
  allocArray.push_back(Allocation(2, 1, 1204224));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 2408448));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 602112));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 1204224));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 1204224));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 1204224));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(8, 1, 301056));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 602112));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(10, 1, 602112));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 602112));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 150528));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 301056));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(14, 1, 301056));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 301056));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(16, 1, 301056));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(17, 1, 301056));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 301056));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(19, 1, 301056));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 301056));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 301056));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 301056));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(23, 1, 301056));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 75264));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(25, 1, 150528));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(26, 1, 150528));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 150528));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(28, 1, 3072));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(29, 1, 4032));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(29, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 3612672);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 75264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 3072);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 4032);
}

// Test for memory allocation for model mobilenet_v1_1.00_224.tflite.
TEST(MemAlloc, testMemAllocForModel6) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(58);
  allocArray.push_back(Allocation(1, 1, 1605632));
  allocArray.push_back(Allocation(2, 1, 1605632));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 3211264));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 802816));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 1605632));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 1605632));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 1605632));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(8, 1, 401408));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 802816));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(10, 1, 802816));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 802816));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 200704));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 401408));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(14, 1, 401408));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 401408));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(16, 1, 401408));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(17, 1, 401408));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 401408));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(19, 1, 401408));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 401408));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 401408));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 401408));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(23, 1, 401408));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 100352));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(25, 1, 200704));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(26, 1, 200704));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 200704));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(28, 1, 4096));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(29, 1, 4032));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(29, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 4816896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 3211264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 4096);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 4032);
}

// Test for memory allocation for model mobilenet_v2_0.35_224.tflite.
TEST(MemAlloc, testMemAllocForModel7) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(108);
  allocArray.push_back(Allocation(1, 1, 802816));
  allocArray.push_back(Allocation(2, 1, 802816));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 401408));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 2408448));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 602112));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 100352));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 602112));
  allocArray.push_back(Allocation(8, 1, 602112));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 100352));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(10, 1, 602112));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 150528));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 50176));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 301056));
  allocArray.push_back(Allocation(14, 1, 301056));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 50176));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(16, 1, 301056));
  allocArray.push_back(Allocation(17, 1, 301056));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 50176));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(19, 1, 301056));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 75264));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 18816));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 112896));
  allocArray.push_back(Allocation(23, 1, 112896));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 18816));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(25, 1, 112896));
  allocArray.push_back(Allocation(26, 1, 112896));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 18816));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(28, 1, 112896));
  allocArray.push_back(Allocation(29, 1, 112896));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(30, 1, 18816));
  allocArray.push_back(Allocation(29, 0, 0));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(31, 1, 112896));
  allocArray.push_back(Allocation(30, 0, 0));
  allocArray.push_back(Allocation(32, 1, 112896));
  allocArray.push_back(Allocation(31, 0, 0));
  allocArray.push_back(Allocation(33, 1, 25088));
  allocArray.push_back(Allocation(32, 0, 0));
  allocArray.push_back(Allocation(34, 1, 150528));
  allocArray.push_back(Allocation(35, 1, 150528));
  allocArray.push_back(Allocation(34, 0, 0));
  allocArray.push_back(Allocation(36, 1, 25088));
  allocArray.push_back(Allocation(35, 0, 0));
  allocArray.push_back(Allocation(33, 0, 0));
  allocArray.push_back(Allocation(37, 1, 150528));
  allocArray.push_back(Allocation(38, 1, 150528));
  allocArray.push_back(Allocation(37, 0, 0));
  allocArray.push_back(Allocation(39, 1, 25088));
  allocArray.push_back(Allocation(38, 0, 0));
  allocArray.push_back(Allocation(36, 0, 0));
  allocArray.push_back(Allocation(40, 1, 150528));
  allocArray.push_back(Allocation(39, 0, 0));
  allocArray.push_back(Allocation(41, 1, 37632));
  allocArray.push_back(Allocation(40, 0, 0));
  allocArray.push_back(Allocation(42, 1, 11008));
  allocArray.push_back(Allocation(41, 0, 0));
  allocArray.push_back(Allocation(43, 1, 65856));
  allocArray.push_back(Allocation(44, 1, 65856));
  allocArray.push_back(Allocation(43, 0, 0));
  allocArray.push_back(Allocation(45, 1, 11008));
  allocArray.push_back(Allocation(44, 0, 0));
  allocArray.push_back(Allocation(42, 0, 0));
  allocArray.push_back(Allocation(46, 1, 65856));
  allocArray.push_back(Allocation(47, 1, 65856));
  allocArray.push_back(Allocation(46, 0, 0));
  allocArray.push_back(Allocation(48, 1, 11008));
  allocArray.push_back(Allocation(47, 0, 0));
  allocArray.push_back(Allocation(45, 0, 0));
  allocArray.push_back(Allocation(49, 1, 65856));
  allocArray.push_back(Allocation(48, 0, 0));
  allocArray.push_back(Allocation(50, 1, 65856));
  allocArray.push_back(Allocation(49, 0, 0));
  allocArray.push_back(Allocation(51, 1, 21952));
  allocArray.push_back(Allocation(50, 0, 0));
  allocArray.push_back(Allocation(52, 1, 250880));
  allocArray.push_back(Allocation(51, 0, 0));
  allocArray.push_back(Allocation(53, 1, 5120));
  allocArray.push_back(Allocation(52, 0, 0));
  allocArray.push_back(Allocation(54, 1, 4032));
  allocArray.push_back(Allocation(53, 0, 0));
  allocArray.push_back(Allocation(54, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 3010560);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 75264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 18816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 18816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 18816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(30)), 18816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(31)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(32)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(33)), 25088);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(34)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(35)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(36)), 25088);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(37)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(38)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(39)), 25088);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(40)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(41)), 37632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(42)), 11008);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(43)), 65856);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(44)), 65856);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(45)), 11008);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(46)), 65856);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(47)), 65856);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(48)), 11008);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(49)), 65856);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(50)), 65856);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(51)), 21952);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(52)), 250880);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(53)), 5120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(54)), 4032);
}

// Test for memory allocation for model mobilenet_v2_0.50_224.tflite.
TEST(MemAlloc, testMemAllocForModel8) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(108);
  allocArray.push_back(Allocation(1, 1, 802816));
  allocArray.push_back(Allocation(2, 1, 802816));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 401408));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 2408448));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 602112));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 200704));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 1204224));
  allocArray.push_back(Allocation(8, 1, 1204224));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 200704));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(10, 1, 1204224));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 301056));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 50176));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 301056));
  allocArray.push_back(Allocation(14, 1, 301056));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 50176));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(16, 1, 301056));
  allocArray.push_back(Allocation(17, 1, 301056));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 50176));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(19, 1, 301056));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 75264));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 25088));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 150528));
  allocArray.push_back(Allocation(23, 1, 150528));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 25088));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(25, 1, 150528));
  allocArray.push_back(Allocation(26, 1, 150528));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 25088));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(28, 1, 150528));
  allocArray.push_back(Allocation(29, 1, 150528));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(30, 1, 25088));
  allocArray.push_back(Allocation(29, 0, 0));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(31, 1, 150528));
  allocArray.push_back(Allocation(30, 0, 0));
  allocArray.push_back(Allocation(32, 1, 150528));
  allocArray.push_back(Allocation(31, 0, 0));
  allocArray.push_back(Allocation(33, 1, 37632));
  allocArray.push_back(Allocation(32, 0, 0));
  allocArray.push_back(Allocation(34, 1, 225792));
  allocArray.push_back(Allocation(35, 1, 225792));
  allocArray.push_back(Allocation(34, 0, 0));
  allocArray.push_back(Allocation(36, 1, 37632));
  allocArray.push_back(Allocation(35, 0, 0));
  allocArray.push_back(Allocation(33, 0, 0));
  allocArray.push_back(Allocation(37, 1, 225792));
  allocArray.push_back(Allocation(38, 1, 225792));
  allocArray.push_back(Allocation(37, 0, 0));
  allocArray.push_back(Allocation(39, 1, 37632));
  allocArray.push_back(Allocation(38, 0, 0));
  allocArray.push_back(Allocation(36, 0, 0));
  allocArray.push_back(Allocation(40, 1, 225792));
  allocArray.push_back(Allocation(39, 0, 0));
  allocArray.push_back(Allocation(41, 1, 56448));
  allocArray.push_back(Allocation(40, 0, 0));
  allocArray.push_back(Allocation(42, 1, 15680));
  allocArray.push_back(Allocation(41, 0, 0));
  allocArray.push_back(Allocation(43, 1, 94080));
  allocArray.push_back(Allocation(44, 1, 94080));
  allocArray.push_back(Allocation(43, 0, 0));
  allocArray.push_back(Allocation(45, 1, 15680));
  allocArray.push_back(Allocation(44, 0, 0));
  allocArray.push_back(Allocation(42, 0, 0));
  allocArray.push_back(Allocation(46, 1, 94080));
  allocArray.push_back(Allocation(47, 1, 94080));
  allocArray.push_back(Allocation(46, 0, 0));
  allocArray.push_back(Allocation(48, 1, 15680));
  allocArray.push_back(Allocation(47, 0, 0));
  allocArray.push_back(Allocation(45, 0, 0));
  allocArray.push_back(Allocation(49, 1, 94080));
  allocArray.push_back(Allocation(48, 0, 0));
  allocArray.push_back(Allocation(50, 1, 94080));
  allocArray.push_back(Allocation(49, 0, 0));
  allocArray.push_back(Allocation(51, 1, 31360));
  allocArray.push_back(Allocation(50, 0, 0));
  allocArray.push_back(Allocation(52, 1, 250880));
  allocArray.push_back(Allocation(51, 0, 0));
  allocArray.push_back(Allocation(53, 1, 5120));
  allocArray.push_back(Allocation(52, 0, 0));
  allocArray.push_back(Allocation(54, 1, 4032));
  allocArray.push_back(Allocation(53, 0, 0));
  allocArray.push_back(Allocation(54, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 3211264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 75264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 25088);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 25088);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 25088);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(30)), 25088);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(31)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(32)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(33)), 37632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(34)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(35)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(36)), 37632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(37)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(38)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(39)), 37632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(40)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(41)), 56448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(42)), 15680);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(43)), 94080);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(44)), 94080);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(45)), 15680);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(46)), 94080);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(47)), 94080);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(48)), 15680);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(49)), 94080);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(50)), 94080);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(51)), 31360);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(52)), 250880);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(53)), 5120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(54)), 4032);
}

// Test for memory allocation for model mobilenet_v2_0.75_224.tflite.
TEST(MemAlloc, testMemAllocForModel9) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(108);
  allocArray.push_back(Allocation(1, 1, 1204224));
  allocArray.push_back(Allocation(2, 1, 1204224));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 802816));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 4816896));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 1204224));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 301056));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 1806336));
  allocArray.push_back(Allocation(8, 1, 1806336));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 301056));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(10, 1, 1806336));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 451584));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 75264));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 451584));
  allocArray.push_back(Allocation(14, 1, 451584));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 75264));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(16, 1, 451584));
  allocArray.push_back(Allocation(17, 1, 451584));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 75264));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(19, 1, 451584));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 112896));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 37632));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 225792));
  allocArray.push_back(Allocation(23, 1, 225792));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 37632));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(25, 1, 225792));
  allocArray.push_back(Allocation(26, 1, 225792));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 37632));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(28, 1, 225792));
  allocArray.push_back(Allocation(29, 1, 225792));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(30, 1, 37632));
  allocArray.push_back(Allocation(29, 0, 0));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(31, 1, 225792));
  allocArray.push_back(Allocation(30, 0, 0));
  allocArray.push_back(Allocation(32, 1, 225792));
  allocArray.push_back(Allocation(31, 0, 0));
  allocArray.push_back(Allocation(33, 1, 56448));
  allocArray.push_back(Allocation(32, 0, 0));
  allocArray.push_back(Allocation(34, 1, 338688));
  allocArray.push_back(Allocation(35, 1, 338688));
  allocArray.push_back(Allocation(34, 0, 0));
  allocArray.push_back(Allocation(36, 1, 56448));
  allocArray.push_back(Allocation(35, 0, 0));
  allocArray.push_back(Allocation(33, 0, 0));
  allocArray.push_back(Allocation(37, 1, 338688));
  allocArray.push_back(Allocation(38, 1, 338688));
  allocArray.push_back(Allocation(37, 0, 0));
  allocArray.push_back(Allocation(39, 1, 56448));
  allocArray.push_back(Allocation(38, 0, 0));
  allocArray.push_back(Allocation(36, 0, 0));
  allocArray.push_back(Allocation(40, 1, 338688));
  allocArray.push_back(Allocation(39, 0, 0));
  allocArray.push_back(Allocation(41, 1, 84672));
  allocArray.push_back(Allocation(40, 0, 0));
  allocArray.push_back(Allocation(42, 1, 23552));
  allocArray.push_back(Allocation(41, 0, 0));
  allocArray.push_back(Allocation(43, 1, 141120));
  allocArray.push_back(Allocation(44, 1, 141120));
  allocArray.push_back(Allocation(43, 0, 0));
  allocArray.push_back(Allocation(45, 1, 23552));
  allocArray.push_back(Allocation(44, 0, 0));
  allocArray.push_back(Allocation(42, 0, 0));
  allocArray.push_back(Allocation(46, 1, 141120));
  allocArray.push_back(Allocation(47, 1, 141120));
  allocArray.push_back(Allocation(46, 0, 0));
  allocArray.push_back(Allocation(48, 1, 23552));
  allocArray.push_back(Allocation(47, 0, 0));
  allocArray.push_back(Allocation(45, 0, 0));
  allocArray.push_back(Allocation(49, 1, 141120));
  allocArray.push_back(Allocation(48, 0, 0));
  allocArray.push_back(Allocation(50, 1, 141120));
  allocArray.push_back(Allocation(49, 0, 0));
  allocArray.push_back(Allocation(51, 1, 47040));
  allocArray.push_back(Allocation(50, 0, 0));
  allocArray.push_back(Allocation(52, 1, 250880));
  allocArray.push_back(Allocation(51, 0, 0));
  allocArray.push_back(Allocation(53, 1, 5120));
  allocArray.push_back(Allocation(52, 0, 0));
  allocArray.push_back(Allocation(54, 1, 4032));
  allocArray.push_back(Allocation(53, 0, 0));
  allocArray.push_back(Allocation(54, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 6021120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 4816896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 1806336);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 1806336);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 1806336);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 75264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 75264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 75264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 37632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 37632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 37632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(30)), 37632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(31)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(32)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(33)), 56448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(34)), 338688);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(35)), 338688);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(36)), 56448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(37)), 338688);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(38)), 338688);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(39)), 56448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(40)), 338688);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(41)), 84672);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(42)), 23552);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(43)), 141120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(44)), 141120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(45)), 23552);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(46)), 141120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(47)), 141120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(48)), 23552);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(49)), 141120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(50)), 141120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(51)), 47040);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(52)), 250880);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(53)), 5120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(54)), 4032);
}

// Test for memory allocation for model mobilenet_v2_1.00_224.tflite.
TEST(MemAlloc, testMemAllocForModel10) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(108);
  allocArray.push_back(Allocation(1, 1, 1605632));
  allocArray.push_back(Allocation(2, 1, 1605632));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 802816));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 4816896));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 1204224));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 301056));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 1806336));
  allocArray.push_back(Allocation(8, 1, 1806336));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 301056));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(10, 1, 1806336));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 451584));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 100352));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 602112));
  allocArray.push_back(Allocation(14, 1, 602112));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 100352));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(16, 1, 602112));
  allocArray.push_back(Allocation(17, 1, 602112));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 100352));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(19, 1, 602112));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 150528));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 50176));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 301056));
  allocArray.push_back(Allocation(23, 1, 301056));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 50176));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(25, 1, 301056));
  allocArray.push_back(Allocation(26, 1, 301056));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 50176));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(28, 1, 301056));
  allocArray.push_back(Allocation(29, 1, 301056));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(30, 1, 50176));
  allocArray.push_back(Allocation(29, 0, 0));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(31, 1, 301056));
  allocArray.push_back(Allocation(30, 0, 0));
  allocArray.push_back(Allocation(32, 1, 301056));
  allocArray.push_back(Allocation(31, 0, 0));
  allocArray.push_back(Allocation(33, 1, 75264));
  allocArray.push_back(Allocation(32, 0, 0));
  allocArray.push_back(Allocation(34, 1, 451584));
  allocArray.push_back(Allocation(35, 1, 451584));
  allocArray.push_back(Allocation(34, 0, 0));
  allocArray.push_back(Allocation(36, 1, 75264));
  allocArray.push_back(Allocation(35, 0, 0));
  allocArray.push_back(Allocation(33, 0, 0));
  allocArray.push_back(Allocation(37, 1, 451584));
  allocArray.push_back(Allocation(38, 1, 451584));
  allocArray.push_back(Allocation(37, 0, 0));
  allocArray.push_back(Allocation(39, 1, 75264));
  allocArray.push_back(Allocation(38, 0, 0));
  allocArray.push_back(Allocation(36, 0, 0));
  allocArray.push_back(Allocation(40, 1, 451584));
  allocArray.push_back(Allocation(39, 0, 0));
  allocArray.push_back(Allocation(41, 1, 112896));
  allocArray.push_back(Allocation(40, 0, 0));
  allocArray.push_back(Allocation(42, 1, 31360));
  allocArray.push_back(Allocation(41, 0, 0));
  allocArray.push_back(Allocation(43, 1, 188160));
  allocArray.push_back(Allocation(44, 1, 188160));
  allocArray.push_back(Allocation(43, 0, 0));
  allocArray.push_back(Allocation(45, 1, 31360));
  allocArray.push_back(Allocation(44, 0, 0));
  allocArray.push_back(Allocation(42, 0, 0));
  allocArray.push_back(Allocation(46, 1, 188160));
  allocArray.push_back(Allocation(47, 1, 188160));
  allocArray.push_back(Allocation(46, 0, 0));
  allocArray.push_back(Allocation(48, 1, 31360));
  allocArray.push_back(Allocation(47, 0, 0));
  allocArray.push_back(Allocation(45, 0, 0));
  allocArray.push_back(Allocation(49, 1, 188160));
  allocArray.push_back(Allocation(48, 0, 0));
  allocArray.push_back(Allocation(50, 1, 188160));
  allocArray.push_back(Allocation(49, 0, 0));
  allocArray.push_back(Allocation(51, 1, 62720));
  allocArray.push_back(Allocation(50, 0, 0));
  allocArray.push_back(Allocation(52, 1, 250880));
  allocArray.push_back(Allocation(51, 0, 0));
  allocArray.push_back(Allocation(53, 1, 5120));
  allocArray.push_back(Allocation(52, 0, 0));
  allocArray.push_back(Allocation(54, 1, 4032));
  allocArray.push_back(Allocation(53, 0, 0));
  allocArray.push_back(Allocation(54, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 6021120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 4816896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 1806336);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 1806336);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 1806336);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(30)), 50176);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(31)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(32)), 301056);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(33)), 75264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(34)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(35)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(36)), 75264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(37)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(38)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(39)), 75264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(40)), 451584);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(41)), 112896);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(42)), 31360);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(43)), 188160);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(44)), 188160);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(45)), 31360);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(46)), 188160);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(47)), 188160);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(48)), 31360);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(49)), 188160);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(50)), 188160);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(51)), 62720);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(52)), 250880);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(53)), 5120);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(54)), 4032);
}

// Test for memory allocation for model mobilenet_v2_1.30_224.tflite.
TEST(MemAlloc, testMemAllocForModel11) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(108);
  allocArray.push_back(Allocation(1, 1, 2007040));
  allocArray.push_back(Allocation(2, 1, 2007040));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 1204224));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 7225344));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 1806336));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 401408));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 2408448));
  allocArray.push_back(Allocation(8, 1, 2408448));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 401408));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(10, 1, 2408448));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 602112));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 125440));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 752640));
  allocArray.push_back(Allocation(14, 1, 752640));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 125440));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(16, 1, 752640));
  allocArray.push_back(Allocation(17, 1, 752640));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 125440));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(19, 1, 752640));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 188160));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 62720));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 376320));
  allocArray.push_back(Allocation(23, 1, 376320));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 62720));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(25, 1, 376320));
  allocArray.push_back(Allocation(26, 1, 376320));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 62720));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(28, 1, 376320));
  allocArray.push_back(Allocation(29, 1, 376320));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(30, 1, 62720));
  allocArray.push_back(Allocation(29, 0, 0));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(31, 1, 376320));
  allocArray.push_back(Allocation(30, 0, 0));
  allocArray.push_back(Allocation(32, 1, 376320));
  allocArray.push_back(Allocation(31, 0, 0));
  allocArray.push_back(Allocation(33, 1, 100352));
  allocArray.push_back(Allocation(32, 0, 0));
  allocArray.push_back(Allocation(34, 1, 602112));
  allocArray.push_back(Allocation(35, 1, 602112));
  allocArray.push_back(Allocation(34, 0, 0));
  allocArray.push_back(Allocation(36, 1, 100352));
  allocArray.push_back(Allocation(35, 0, 0));
  allocArray.push_back(Allocation(33, 0, 0));
  allocArray.push_back(Allocation(37, 1, 602112));
  allocArray.push_back(Allocation(38, 1, 602112));
  allocArray.push_back(Allocation(37, 0, 0));
  allocArray.push_back(Allocation(39, 1, 100352));
  allocArray.push_back(Allocation(38, 0, 0));
  allocArray.push_back(Allocation(36, 0, 0));
  allocArray.push_back(Allocation(40, 1, 602112));
  allocArray.push_back(Allocation(39, 0, 0));
  allocArray.push_back(Allocation(41, 1, 150528));
  allocArray.push_back(Allocation(40, 0, 0));
  allocArray.push_back(Allocation(42, 1, 40768));
  allocArray.push_back(Allocation(41, 0, 0));
  allocArray.push_back(Allocation(43, 1, 244608));
  allocArray.push_back(Allocation(44, 1, 244608));
  allocArray.push_back(Allocation(43, 0, 0));
  allocArray.push_back(Allocation(45, 1, 40768));
  allocArray.push_back(Allocation(44, 0, 0));
  allocArray.push_back(Allocation(42, 0, 0));
  allocArray.push_back(Allocation(46, 1, 244608));
  allocArray.push_back(Allocation(47, 1, 244608));
  allocArray.push_back(Allocation(46, 0, 0));
  allocArray.push_back(Allocation(48, 1, 40768));
  allocArray.push_back(Allocation(47, 0, 0));
  allocArray.push_back(Allocation(45, 0, 0));
  allocArray.push_back(Allocation(49, 1, 244608));
  allocArray.push_back(Allocation(48, 0, 0));
  allocArray.push_back(Allocation(50, 1, 244608));
  allocArray.push_back(Allocation(49, 0, 0));
  allocArray.push_back(Allocation(51, 1, 81536));
  allocArray.push_back(Allocation(50, 0, 0));
  allocArray.push_back(Allocation(52, 1, 326144));
  allocArray.push_back(Allocation(51, 0, 0));
  allocArray.push_back(Allocation(53, 1, 6656));
  allocArray.push_back(Allocation(52, 0, 0));
  allocArray.push_back(Allocation(54, 1, 4032));
  allocArray.push_back(Allocation(53, 0, 0));
  allocArray.push_back(Allocation(54, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 9031680);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 2007040);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 2007040);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 7225344);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 1806336);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 125440);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 752640);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 752640);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 125440);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 752640);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 752640);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 125440);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 752640);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 188160);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 62720);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 376320);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 376320);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 62720);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 376320);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 376320);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 62720);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 376320);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 376320);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(30)), 62720);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(31)), 376320);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(32)), 376320);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(33)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(34)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(35)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(36)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(37)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(38)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(39)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(40)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(41)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(42)), 40768);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(43)), 244608);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(44)), 244608);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(45)), 40768);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(46)), 244608);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(47)), 244608);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(48)), 40768);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(49)), 244608);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(50)), 244608);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(51)), 81536);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(52)), 326144);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(53)), 6656);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(54)), 4032);
}

// Test for memory allocation for model mobilenet_v2_1.40_224.tflite.
TEST(MemAlloc, testMemAllocForModel12) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(108);
  allocArray.push_back(Allocation(1, 1, 2408448));
  allocArray.push_back(Allocation(2, 1, 2408448));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 1204224));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 7225344));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(5, 1, 1806336));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(6, 1, 401408));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 2408448));
  allocArray.push_back(Allocation(8, 1, 2408448));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(9, 1, 401408));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(10, 1, 2408448));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(11, 1, 602112));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(12, 1, 150528));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 903168));
  allocArray.push_back(Allocation(14, 1, 903168));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(15, 1, 150528));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(16, 1, 903168));
  allocArray.push_back(Allocation(17, 1, 903168));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(18, 1, 150528));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(19, 1, 903168));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 225792));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(21, 1, 68992));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(22, 1, 413952));
  allocArray.push_back(Allocation(23, 1, 413952));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(24, 1, 68992));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(25, 1, 413952));
  allocArray.push_back(Allocation(26, 1, 413952));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(27, 1, 68992));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(28, 1, 413952));
  allocArray.push_back(Allocation(29, 1, 413952));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(30, 1, 68992));
  allocArray.push_back(Allocation(29, 0, 0));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(31, 1, 413952));
  allocArray.push_back(Allocation(30, 0, 0));
  allocArray.push_back(Allocation(32, 1, 413952));
  allocArray.push_back(Allocation(31, 0, 0));
  allocArray.push_back(Allocation(33, 1, 106624));
  allocArray.push_back(Allocation(32, 0, 0));
  allocArray.push_back(Allocation(34, 1, 639744));
  allocArray.push_back(Allocation(35, 1, 639744));
  allocArray.push_back(Allocation(34, 0, 0));
  allocArray.push_back(Allocation(36, 1, 106624));
  allocArray.push_back(Allocation(35, 0, 0));
  allocArray.push_back(Allocation(33, 0, 0));
  allocArray.push_back(Allocation(37, 1, 639744));
  allocArray.push_back(Allocation(38, 1, 639744));
  allocArray.push_back(Allocation(37, 0, 0));
  allocArray.push_back(Allocation(39, 1, 106624));
  allocArray.push_back(Allocation(38, 0, 0));
  allocArray.push_back(Allocation(36, 0, 0));
  allocArray.push_back(Allocation(40, 1, 639744));
  allocArray.push_back(Allocation(39, 0, 0));
  allocArray.push_back(Allocation(41, 1, 159936));
  allocArray.push_back(Allocation(40, 0, 0));
  allocArray.push_back(Allocation(42, 1, 43904));
  allocArray.push_back(Allocation(41, 0, 0));
  allocArray.push_back(Allocation(43, 1, 263424));
  allocArray.push_back(Allocation(44, 1, 263424));
  allocArray.push_back(Allocation(43, 0, 0));
  allocArray.push_back(Allocation(45, 1, 43904));
  allocArray.push_back(Allocation(44, 0, 0));
  allocArray.push_back(Allocation(42, 0, 0));
  allocArray.push_back(Allocation(46, 1, 263424));
  allocArray.push_back(Allocation(47, 1, 263424));
  allocArray.push_back(Allocation(46, 0, 0));
  allocArray.push_back(Allocation(48, 1, 43904));
  allocArray.push_back(Allocation(47, 0, 0));
  allocArray.push_back(Allocation(45, 0, 0));
  allocArray.push_back(Allocation(49, 1, 263424));
  allocArray.push_back(Allocation(48, 0, 0));
  allocArray.push_back(Allocation(50, 1, 263424));
  allocArray.push_back(Allocation(49, 0, 0));
  allocArray.push_back(Allocation(51, 1, 87808));
  allocArray.push_back(Allocation(50, 0, 0));
  allocArray.push_back(Allocation(52, 1, 351232));
  allocArray.push_back(Allocation(51, 0, 0));
  allocArray.push_back(Allocation(53, 1, 7168));
  allocArray.push_back(Allocation(52, 0, 0));
  allocArray.push_back(Allocation(54, 1, 4032));
  allocArray.push_back(Allocation(53, 0, 0));
  allocArray.push_back(Allocation(54, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 9031680);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 1204224);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 7225344);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 1806336);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 2408448);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 602112);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 903168);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 903168);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 903168);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 903168);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 150528);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 903168);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 225792);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 68992);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 413952);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 413952);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 68992);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 413952);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 413952);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 68992);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 413952);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 413952);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(30)), 68992);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(31)), 413952);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(32)), 413952);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(33)), 106624);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(34)), 639744);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(35)), 639744);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(36)), 106624);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(37)), 639744);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(38)), 639744);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(39)), 106624);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(40)), 639744);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(41)), 159936);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(42)), 43904);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(43)), 263424);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(44)), 263424);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(45)), 43904);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(46)), 263424);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(47)), 263424);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(48)), 43904);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(49)), 263424);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(50)), 263424);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(51)), 87808);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(52)), 351232);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(53)), 7168);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(54)), 4032);
}

// Test for memory allocation for model resnet50_v1.tflite.
TEST(MemAlloc, testMemAllocForModel13) {
  MemoryAllocator MA("mem", 0, 64);
  // Define allocation array.
  std::vector<Allocation> allocArray;
  allocArray.reserve(120);
  allocArray.push_back(Allocation(1, 1, 3211264));
  allocArray.push_back(Allocation(2, 1, 3326976));
  allocArray.push_back(Allocation(1, 0, 0));
  allocArray.push_back(Allocation(3, 1, 802816));
  allocArray.push_back(Allocation(2, 0, 0));
  allocArray.push_back(Allocation(4, 1, 3211264));
  allocArray.push_back(Allocation(5, 1, 802816));
  allocArray.push_back(Allocation(3, 0, 0));
  allocArray.push_back(Allocation(6, 1, 802816));
  allocArray.push_back(Allocation(5, 0, 0));
  allocArray.push_back(Allocation(7, 1, 3211264));
  allocArray.push_back(Allocation(6, 0, 0));
  allocArray.push_back(Allocation(7, 0, 0));
  allocArray.push_back(Allocation(8, 1, 802816));
  allocArray.push_back(Allocation(9, 1, 802816));
  allocArray.push_back(Allocation(8, 0, 0));
  allocArray.push_back(Allocation(10, 1, 3211264));
  allocArray.push_back(Allocation(9, 0, 0));
  allocArray.push_back(Allocation(10, 0, 0));
  allocArray.push_back(Allocation(11, 1, 802816));
  allocArray.push_back(Allocation(12, 1, 802816));
  allocArray.push_back(Allocation(11, 0, 0));
  allocArray.push_back(Allocation(13, 1, 3211264));
  allocArray.push_back(Allocation(12, 0, 0));
  allocArray.push_back(Allocation(13, 0, 0));
  allocArray.push_back(Allocation(14, 1, 1605632));
  allocArray.push_back(Allocation(15, 1, 401408));
  allocArray.push_back(Allocation(4, 0, 0));
  allocArray.push_back(Allocation(16, 1, 401408));
  allocArray.push_back(Allocation(15, 0, 0));
  allocArray.push_back(Allocation(17, 1, 1605632));
  allocArray.push_back(Allocation(16, 0, 0));
  allocArray.push_back(Allocation(17, 0, 0));
  allocArray.push_back(Allocation(18, 1, 401408));
  allocArray.push_back(Allocation(19, 1, 401408));
  allocArray.push_back(Allocation(18, 0, 0));
  allocArray.push_back(Allocation(20, 1, 1605632));
  allocArray.push_back(Allocation(19, 0, 0));
  allocArray.push_back(Allocation(20, 0, 0));
  allocArray.push_back(Allocation(21, 1, 401408));
  allocArray.push_back(Allocation(22, 1, 401408));
  allocArray.push_back(Allocation(21, 0, 0));
  allocArray.push_back(Allocation(23, 1, 1605632));
  allocArray.push_back(Allocation(22, 0, 0));
  allocArray.push_back(Allocation(23, 0, 0));
  allocArray.push_back(Allocation(24, 1, 401408));
  allocArray.push_back(Allocation(25, 1, 401408));
  allocArray.push_back(Allocation(24, 0, 0));
  allocArray.push_back(Allocation(26, 1, 1605632));
  allocArray.push_back(Allocation(25, 0, 0));
  allocArray.push_back(Allocation(26, 0, 0));
  allocArray.push_back(Allocation(27, 1, 802816));
  allocArray.push_back(Allocation(28, 1, 200704));
  allocArray.push_back(Allocation(14, 0, 0));
  allocArray.push_back(Allocation(29, 1, 200704));
  allocArray.push_back(Allocation(28, 0, 0));
  allocArray.push_back(Allocation(30, 1, 802816));
  allocArray.push_back(Allocation(29, 0, 0));
  allocArray.push_back(Allocation(30, 0, 0));
  allocArray.push_back(Allocation(31, 1, 200704));
  allocArray.push_back(Allocation(32, 1, 200704));
  allocArray.push_back(Allocation(31, 0, 0));
  allocArray.push_back(Allocation(33, 1, 802816));
  allocArray.push_back(Allocation(32, 0, 0));
  allocArray.push_back(Allocation(33, 0, 0));
  allocArray.push_back(Allocation(34, 1, 200704));
  allocArray.push_back(Allocation(35, 1, 200704));
  allocArray.push_back(Allocation(34, 0, 0));
  allocArray.push_back(Allocation(36, 1, 802816));
  allocArray.push_back(Allocation(35, 0, 0));
  allocArray.push_back(Allocation(36, 0, 0));
  allocArray.push_back(Allocation(37, 1, 200704));
  allocArray.push_back(Allocation(38, 1, 200704));
  allocArray.push_back(Allocation(37, 0, 0));
  allocArray.push_back(Allocation(39, 1, 802816));
  allocArray.push_back(Allocation(38, 0, 0));
  allocArray.push_back(Allocation(39, 0, 0));
  allocArray.push_back(Allocation(40, 1, 200704));
  allocArray.push_back(Allocation(41, 1, 200704));
  allocArray.push_back(Allocation(40, 0, 0));
  allocArray.push_back(Allocation(42, 1, 802816));
  allocArray.push_back(Allocation(41, 0, 0));
  allocArray.push_back(Allocation(42, 0, 0));
  allocArray.push_back(Allocation(43, 1, 200704));
  allocArray.push_back(Allocation(44, 1, 200704));
  allocArray.push_back(Allocation(43, 0, 0));
  allocArray.push_back(Allocation(45, 1, 802816));
  allocArray.push_back(Allocation(44, 0, 0));
  allocArray.push_back(Allocation(45, 0, 0));
  allocArray.push_back(Allocation(46, 1, 401408));
  allocArray.push_back(Allocation(47, 1, 100352));
  allocArray.push_back(Allocation(27, 0, 0));
  allocArray.push_back(Allocation(48, 1, 100352));
  allocArray.push_back(Allocation(47, 0, 0));
  allocArray.push_back(Allocation(49, 1, 401408));
  allocArray.push_back(Allocation(48, 0, 0));
  allocArray.push_back(Allocation(49, 0, 0));
  allocArray.push_back(Allocation(50, 1, 100352));
  allocArray.push_back(Allocation(51, 1, 100352));
  allocArray.push_back(Allocation(50, 0, 0));
  allocArray.push_back(Allocation(52, 1, 401408));
  allocArray.push_back(Allocation(51, 0, 0));
  allocArray.push_back(Allocation(52, 0, 0));
  allocArray.push_back(Allocation(53, 1, 100352));
  allocArray.push_back(Allocation(54, 1, 100352));
  allocArray.push_back(Allocation(53, 0, 0));
  allocArray.push_back(Allocation(55, 1, 401408));
  allocArray.push_back(Allocation(54, 0, 0));
  allocArray.push_back(Allocation(55, 0, 0));
  allocArray.push_back(Allocation(56, 1, 57344));
  allocArray.push_back(Allocation(46, 0, 0));
  allocArray.push_back(Allocation(57, 1, 57344));
  allocArray.push_back(Allocation(57, 0, 0));
  allocArray.push_back(Allocation(58, 1, 8192));
  allocArray.push_back(Allocation(56, 0, 0));
  allocArray.push_back(Allocation(59, 1, 8192));
  allocArray.push_back(Allocation(59, 0, 0));
  allocArray.push_back(Allocation(60, 1, 4032));
  allocArray.push_back(Allocation(58, 0, 0));
  allocArray.push_back(Allocation(60, 0, 0));
  // Perform allocation.
  uint64_t usedSize = MA.allocate(allocArray);
  // Verifications.
  EXPECT_EQ(usedSize, 7225344);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(1)), 3211264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(2)), 3326976);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(3)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(4)), 3211264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(5)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(6)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(7)), 3211264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(8)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(9)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(10)), 3211264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(11)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(12)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(13)), 3211264);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(14)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(15)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(16)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(17)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(18)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(19)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(20)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(21)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(22)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(23)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(24)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(25)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(26)), 1605632);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(27)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(28)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(29)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(30)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(31)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(32)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(33)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(34)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(35)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(36)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(37)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(38)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(39)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(40)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(41)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(42)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(43)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(44)), 200704);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(45)), 802816);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(46)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(47)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(48)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(49)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(50)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(51)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(52)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(53)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(54)), 100352);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(55)), 401408);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(56)), 57344);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(57)), 57344);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(58)), 8192);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(59)), 8192);
  EXPECT_EQ(MA.getSize(reinterpret_cast<void *>(60)), 4032);
}
