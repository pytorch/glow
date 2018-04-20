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

#include "gtest/gtest.h"

using namespace glow;

TEST(MemAlloc, simple) {
  MemoryAllocator MA(1000);

  // Can't allocate huge chunks.
  EXPECT_EQ(MA.allocate(100000), MemoryAllocator::npos);

  // First chunk needs to start at zero.
  EXPECT_EQ(MA.allocate(500), 0);

  // Second chunk must not be zero.
  EXPECT_NE(MA.allocate(500), 0);
}

TEST(MemAlloc, holes) {
  MemoryAllocator MA(1000);

  auto p0 = MA.allocate(10);
  auto p1 = MA.allocate(10);
  auto p2 = MA.allocate(10);

  MA.deallocate(p1);
  auto p4 = MA.allocate(10);

  // Check that p4 was allocated on top of the freed p1.
  EXPECT_EQ(p4, p1);

  MA.deallocate(p0);
  MA.deallocate(p2);
}

TEST(MemAlloc, dealloc) {
  MemoryAllocator MA(1000);
  auto p0 = MA.allocate(10);
  auto p1 = MA.allocate(10);
  auto p2 = MA.allocate(10);
  auto p3 = MA.allocate(10);

  EXPECT_EQ(p0, 0);
  EXPECT_NE(p1, MemoryAllocator::npos);
  EXPECT_NE(p2, MemoryAllocator::npos);
  EXPECT_NE(p3, MemoryAllocator::npos);

  // Deallocate in some arbitrary order.
  MA.deallocate(p0);
  MA.deallocate(p2);
  MA.deallocate(p1);
  MA.deallocate(p3);

  // Check that after deallocating everything we start allocating from zero.
  EXPECT_EQ(MA.allocate(10), 0);
}

TEST(MemAlloc, dealloc2) {
  MemoryAllocator MA(10000);

  std::vector<size_t> allocations;

  for (int i = 0; i < 100; i++) {
    // Create odd-sized allocations.
    auto p0 = MA.allocate(10 + i % 4);
    EXPECT_NE(p0, MemoryAllocator::npos);
    allocations.push_back(p0);

    if (allocations.size() > 20) {
      MA.deallocate(allocations[0]);
      allocations.erase(allocations.begin());
    }
  }
  // Drain the allocator.
  while (!allocations.empty()) {
    MA.deallocate(allocations[0]);
    allocations.erase(allocations.begin());
  }

  // Check that after deallocating everything we start allocating from zero.
  EXPECT_EQ(MA.allocate(10), 0);
}

TEST(MemAlloc, allocateToTheMax) {
  MemoryAllocator MA(128);
  auto p0 = MA.allocate(64);
  auto p1 = MA.allocate(64);

  EXPECT_EQ(p0, 0);
  EXPECT_NE(p1, MemoryAllocator::npos);

  MA.deallocate(p0);
  MA.deallocate(p1);

  EXPECT_EQ(MA.getMaxMemoryUsage(), 128);
}
