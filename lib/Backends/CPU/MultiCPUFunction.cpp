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

#include "MultiCPUFunction.h"

#include <condition_variable>
#include <mutex>
#include <thread>

using namespace glow;

namespace {
struct SyncVariable {
  bool done{false};
  std::condition_variable cv;
  std::mutex mutex;

  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return done; });
  }

  void notify() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      done = true;
    }
    cv.notify_all();
  }
};
} // end namespace

MultiCPUFunction::MultiCPUFunction(
    FunctionGraph &&G, std::vector<std::unique_ptr<CompiledFunction>> FS)
    : G_(std::move(G)), functions_(std::move(FS)) {}

void MultiCPUFunction::execute() {
  std::unordered_map<Variable *, size_t> index;
  size_t i = 0;
  for (auto *v : G_.getChannels()) {
    index.emplace(v, i++);
  }
  std::vector<SyncVariable> sync(G_.getChannels().size());

  // For each "backend" (representing a partitioned function):
  //   Wait for its input dependencies to be satisfied.
  //   Execute it.
  //   Signal that its output dependencies are satisfied.
  std::mutex m;
  auto cfit = functions_.begin();
  auto end = functions_.end();
  auto ffit = G_.getFunctions().begin();
  assert(functions_.size() == G_.getFunctions().size());
  auto worker = [&]() {
    while (true) {
      CompiledFunction *CF;
      Function *F;
      {
        std::lock_guard<std::mutex> g(m);
        if (cfit == end)
          return;
        CF = cfit->get();
        cfit++;
        F = *ffit++;
      }
      for (auto const &input : G_.getInputs(F)) {
        sync[index[input]].wait();
      }
      CF->execute();
      for (auto const &output : G_.getOutputs(F)) {
        sync[index[output]].notify();
      }
    }
  };

  std::vector<std::thread> threads;
  for (unsigned i = 0; i < std::thread::hardware_concurrency(); i++) {
    threads.emplace_back(worker);
  }
  for (auto &t : threads) {
    t.join();
  }
}
