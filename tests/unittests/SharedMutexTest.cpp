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

// This file tests the basic functionality of SharedMutex class.

#include "glow/Base/SharedMutex.h"
#include "gtest/gtest.h"
#include <atomic>
#include <thread>

using SM = glow::shared_mutex;
using SL = glow::shared_lock<SM>;

const int kMaxReaders = 10;
namespace {
std::atomic<bool> stopThread{false};
std::atomic<size_t> writeLocks{0};
std::atomic<size_t> readLocks{0};

template <typename M> void run(M *lock) {
  while (!stopThread) {
    // @lint-ignore HOWTOEVEN
    if (std::rand() % 10 == 0) { // 10% of writers.
      SL guard(*lock, true);
      ++writeLocks;
    } else { // 90% of readers.
      SL guard(*lock, false);
      ++readLocks;
    }
  }
}
} // namespace

TEST(SharedMutex, Performance) {
  SM l;
  std::srand(time(nullptr));
  std::vector<std::thread> threads;

  for (int i = 0; i < kMaxReaders; ++i) {
    threads.emplace_back([&l]() { run(&l); });
  }

  /* sleep override */
  sleep(1);

  stopThread = true;

  for (auto &thread : threads) {
    thread.join();
  }

  std::cerr << "Number write locks: " << (size_t)writeLocks
            << ", number read locks: " << (size_t)readLocks << "\n";
}

TEST(SharedMutex, No_Max_Readers) {
  SM l;

  for (int i = 0; i < kMaxReaders; ++i) {
    EXPECT_TRUE(l.try_lock_shared());
  }

  EXPECT_TRUE(l.try_lock_shared());
}

TEST(SharedMutex, Writer_Wait_Readers) {
  SM l;

  for (int i = 0; i < kMaxReaders; ++i) {
    EXPECT_TRUE(l.try_lock_shared());
  }

  for (int i = 0; i < kMaxReaders; ++i) {
    EXPECT_FALSE(l.try_lock());
    l.unlock_shared();
  }

  EXPECT_TRUE(l.try_lock());
}

TEST(SharedMutex, Readers_Wait_Writer) {
  SM l;

  l.lock();

  for (int i = 0; i < kMaxReaders; ++i) {
    EXPECT_FALSE(l.try_lock_shared());
  }

  l.unlock();

  for (int i = 0; i < kMaxReaders; ++i) {
    EXPECT_TRUE(l.try_lock_shared());
  }
}

TEST(SharedMutex, Writer_Wait_Writer) {
  SM l;

  l.lock();
  EXPECT_FALSE(l.try_lock());
  l.unlock();
  l.lock();
  EXPECT_FALSE(l.try_lock());
}

TEST(SharedMutex, Read_Holders) {
  SM l;

  SL guard(l, false);
  EXPECT_FALSE(l.try_lock());
  EXPECT_TRUE(l.try_lock_shared());
  l.unlock_shared();
  EXPECT_FALSE(l.try_lock());
  guard.unlock_shared();
  EXPECT_TRUE(l.try_lock());
  EXPECT_FALSE(guard.try_lock_shared());
  l.unlock();
  EXPECT_TRUE(guard.try_lock_shared());
}

TEST(SharedMutex, Counting_Read_Holders) {
  SM l;

  {
    SL g1(l, false);
    EXPECT_FALSE(g1.try_lock());
  }

  EXPECT_TRUE(l.try_lock());
  l.unlock();

  {
    SL g1(l, false);
    EXPECT_TRUE(g1.try_lock_shared());
  }

  EXPECT_TRUE(l.try_lock());
  l.unlock();

  {
    SL g1(l, false);
    SL g2(l, false);
    EXPECT_TRUE(g1.try_lock_shared());
    EXPECT_TRUE(g2.try_lock_shared());
  }

  EXPECT_TRUE(l.try_lock());
  l.unlock();
}

TEST(SharedMutex, Write_Holders) {
  SM l;

  SL guard(l, true);
  EXPECT_FALSE(l.try_lock());
  EXPECT_FALSE(l.try_lock_shared());
  guard.unlock();
  EXPECT_TRUE(l.try_lock_shared());
  EXPECT_FALSE(guard.try_lock());
  l.unlock_shared();
  EXPECT_TRUE(guard.try_lock());
}
