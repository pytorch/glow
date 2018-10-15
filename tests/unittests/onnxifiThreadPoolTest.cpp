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
#include "ThreadPool.h"
#include "gtest/gtest.h"

#include <future>
#include <vector>

using namespace glow::onnxifi;

TEST(onnxifiThreadPool, BasicTest) {
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
