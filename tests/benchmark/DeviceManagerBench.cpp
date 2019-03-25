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

#include "benchmark/benchmark.h"
#include "glow/Backends/DeviceManager.h"

using namespace glow;
using namespace glow::runtime;

/// Fixture class for reusing DeviceManager setup across benchmarks.
class CPUDeviceManagerBench : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) {
    // Create device manager.
    device_.reset(DeviceManager::createDeviceManager(BackendKind::CPU));
  }

  // Pointer to the DeviceManager used for benchmarking.
  std::unique_ptr<DeviceManager> device_;
};

// Benchmark for DeviceManager::init().
BENCHMARK_DEFINE_F(CPUDeviceManagerBench, init)(benchmark::State &state) {
  // Init the DeviceManager until state requests to stop.
  for (auto _ : state) {
    (void)device_->init();

    // Don't including stopping the DeviceManager in the timing for this
    // benchmark.
    state.PauseTiming();
    (void)device_->stop();
    state.ResumeTiming();
  }
}

// Benchmark registration.
BENCHMARK_REGISTER_F(CPUDeviceManagerBench, init)->Iterations(100);

// Main.
BENCHMARK_MAIN();
