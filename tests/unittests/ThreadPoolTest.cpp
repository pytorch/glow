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
