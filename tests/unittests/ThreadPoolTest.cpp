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
#include "glow/Support/ThreadPool.h"
#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"

#include <future>
#include <vector>

using namespace glow;

TEST(ThreadPool, BasicTest) {
  const unsigned numWorkers = 100;
  const unsigned numWorkItems = 1000;
  ThreadPool tp(numWorkers);

  // Create vectors to store futures and promises for
  // communicating results of work done on thread pool.
  std::vector<std::future<int>> futures;
  futures.reserve(numWorkItems);
  std::vector<std::promise<int>> promises(numWorkItems);

  // Submit 'numWorkItems' work items to the thread pool;
  // each task takes its index and computes and returns
  // 2x its index.
  for (unsigned i = 0; i < numWorkItems; ++i) {
    auto &p = promises[i];
    futures.emplace_back(p.get_future());
    tp.submit([&p, i]() { p.set_value(2 * i); });
  }

  // Check that every future holds the expected result
  // (2x its index).
  for (unsigned i = 0; i < numWorkItems; ++i) {
    futures[i].wait();
    auto result = futures[i].get();
    EXPECT_EQ(result, 2 * i);
  }
}

TEST(ThreadPool, moveCaptureTest) {
  ThreadPool tp(1);

  std::unique_ptr<int> input = llvm::make_unique<int>(42);
  int output = 0;
  auto func = [input = std::move(input), &output]() { output = (*input) * 2; };

  auto done = tp.submit(std::move(func));

  done.wait();
  EXPECT_EQ(output, 84);
}

TEST(ThreadPool, completionFutureTest) {
  ThreadPool tp(1);

  int input = 42, output = 0;
  std::packaged_task<void(void)> task(
      [&input, &output]() { output = input * 3; });

  auto done = tp.submit(std::move(task));

  done.wait();
  EXPECT_EQ(output, 126);
}

/// Verify that we can get an Executor that runs tasks consistently on the same
/// thread.
TEST(ThreadPool, getExecutor) {
  ThreadPool tp(3);

  std::thread::id t1;
  std::thread::id t2;

  /// Check that runs on the same executor run on the same thread.
  auto *ex = tp.getExecutor();
  auto fut1 = ex->submit([&t1]() { t1 = std::this_thread::get_id(); });
  auto fut2 = ex->submit([&t2]() { t2 = std::this_thread::get_id(); });

  fut1.get();
  fut2.get();

  ASSERT_EQ(t1, t2);
  ASSERT_NE(t1, std::thread::id());

  /// Now verify this isn't always true.
  t1 = t2 = std::thread::id();
  auto *ex2 = tp.getExecutor();

  fut1 = ex->submit([&t1] { t1 = std::this_thread::get_id(); });
  fut2 = ex2->submit([&t2] { t2 = std::this_thread::get_id(); });

  fut1.get();
  fut2.get();

  ASSERT_NE(t1, t2);
  ASSERT_NE(t1, std::thread::id());
}

/// Verify that you can get more executors than there are threads in the pool.
TEST(ThreadPool, getManyExecutors) {
  ThreadPool tp(3);

  std::atomic<size_t> left{20};
  std::promise<void> finished;

  auto F = [&left, &finished]() {
    if (--left == 0) {
      finished.set_value();
    }
  };

  for (int i = 0; i < 10; ++i) {
    auto *ex = tp.getExecutor();
    // Submit two tasks
    ex->submit(F);
    ex->submit(F);
  }

  finished.get_future().get();
  ASSERT_EQ(left, 0);
}

/// Verify we can run on all threads and that they are different.
TEST(ThreadPool, runOnAllThreads) {
  ThreadPool tp(3);
  std::vector<std::thread::id> threadIds;

  std::mutex vecLock;

  auto fut = tp.runOnAllThreads([&threadIds, &vecLock]() {
    std::lock_guard<std::mutex> l(vecLock);
    threadIds.push_back(std::this_thread::get_id());
  });

  fut.get();

  ASSERT_EQ(threadIds.size(), 3);
  ASSERT_NE(threadIds[0], threadIds[1]);
  ASSERT_NE(threadIds[1], threadIds[2]);
  ASSERT_NE(threadIds[2], threadIds[0]);
}
