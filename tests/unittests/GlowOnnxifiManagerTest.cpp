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

#include "../../lib/Onnxifi/GlowOnnxifiManager.h"

#include "gtest/gtest.h"

#include <thread>

using namespace glow::onnxifi;

TEST(GlowOnnxifiManagerTest, BackendIdTest) {
  auto &manager = GlowOnnxifiManager::get();
  auto *backend = manager.createBackend("Interpreter",
                                        /*use_onnx*/ true);
  // Backend is valid after it has been added to the manager.
  EXPECT_TRUE(manager.isValid(backend));
  manager.release(backend);
  // Backend isn't valid after it has been released by the manager.
  EXPECT_FALSE(manager.isValid(backend));

  // Nullptr is not a valid Backend.
  backend = nullptr;
  EXPECT_FALSE(manager.isValid(backend));
}

TEST(GlowOnnxifiManagerTest, EventTest) {
  auto &manager = GlowOnnxifiManager::get();
  auto *event = manager.createEvent();
  // Event is valid after it has been created by the manager.
  EXPECT_TRUE(manager.isValid(event));
  manager.release(event);
  // Event isn't valid after it has been released by the manager.
  EXPECT_FALSE(manager.isValid(event));

  // Nullptr is not a valid Event.
  event = nullptr;
  EXPECT_FALSE(manager.isValid(event));
}

TEST(GlowOnnxifiManagerTest, GraphTest) {
  auto &manager = GlowOnnxifiManager::get();
  auto *backend = manager.createBackend("Interpreter",
                                        /*use_onnx*/ true);

  auto *graph = manager.createGraph(backend);
  // Graph is valid after it has been created by the manager.
  EXPECT_TRUE(manager.isValid(graph));

  manager.release(graph);
  // Graph isn't valid after it has been released by the manager.
  EXPECT_FALSE(manager.isValid(graph));

  manager.release(backend);

  // Nullptr is not a valid Graph.
  graph = nullptr;
  EXPECT_FALSE(manager.isValid(graph));
}

void createAndDestroyManagerObjects() {
  auto &manager = GlowOnnxifiManager::get();
  auto *backend = manager.createBackend("Interpreter",
                                        /*use_onnx*/ true);

  auto *event = manager.createEvent();
  auto *graph = manager.createGraph(backend);

  EXPECT_TRUE(manager.isValid(backend));
  EXPECT_TRUE(manager.isValid(event));
  EXPECT_TRUE(manager.isValid(graph));

  manager.release(graph);
  manager.release(event);
  manager.release(backend);
}

TEST(GlowOnnxifiManagerTest, Concurrency) {
  constexpr auto numThreads = 12;
  constexpr auto numItersPerThread = 100;

  std::vector<std::thread> threads;
  for (auto i = 0; i < numThreads; ++i) {
    threads.emplace_back([&]() {
      for (auto j = 0; j < numItersPerThread; ++j) {
        createAndDestroyManagerObjects();
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
}
