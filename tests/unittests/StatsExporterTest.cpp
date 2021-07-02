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

#include "glow/Runtime/StatsExporter.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/HostManager/HostManager.h"

#include <gtest/gtest.h>

#include <memory>

using namespace glow;

class MockStatsExporter : public StatsExporter {
  std::shared_ptr<StatsExporterRegistry> statsExporterRegistry_;

public:
  MockStatsExporter() : statsExporterRegistry_(StatsExporterRegistry::Stats()) {
    statsExporterRegistry_->registerStatsExporter(this);
  }

  ~MockStatsExporter() override {
    statsExporterRegistry_->revokeStatsExporter(this);
  }

  void addTimeSeriesValue(llvm::StringRef key, double value) override {
    timeSeries[key.str()].push_back(value);
  }

  void incrementCounter(llvm::StringRef key, int64_t value) override {
    counters[key.str()] += value;
  }

  void setCounter(llvm::StringRef key, int64_t value) override {
    counters[key.str()] = value;
  }

  void clear() {
    counters.clear();
    timeSeries.clear();
  }

  std::map<std::string, int64_t> counters;
  std::map<std::string, std::vector<double>> timeSeries;
} MockStats;

class StatsExporterTest : public ::testing::Test {
  ~StatsExporterTest() { MockStats.clear(); }
};

TEST(StatsExporter, Counter) {
  MockStats.setCounter("foo", 1);
  ASSERT_EQ(MockStats.counters["foo"], 1);
  MockStats.setCounter("foo", 3);
  ASSERT_EQ(MockStats.counters["foo"], 3);
  MockStats.incrementCounter("foo", 1);
  ASSERT_EQ(MockStats.counters["foo"], 4);
  MockStats.incrementCounter("foo", -1);
  ASSERT_EQ(MockStats.counters["foo"], 3);
}

TEST(StatsExporter, TimeSeries) {
  MockStats.addTimeSeriesValue("bar", 1);
  MockStats.addTimeSeriesValue("bar", 3.14);
  MockStats.addTimeSeriesValue("bar", 2.71);
  auto it = MockStats.timeSeries.find("bar");
  ASSERT_NE(it, MockStats.timeSeries.end());
  auto const &ts = it->second;
  ASSERT_EQ(ts.size(), 3);
  EXPECT_EQ(ts[0], 1);
  EXPECT_EQ(ts[1], 3.14);
  EXPECT_EQ(ts[2], 2.71);
}

TEST(StatsExporter, Device) {
  using namespace glow::runtime;
  EXPECT_EQ(MockStats.counters.count("glow.devices_used.interpreter"), 0);
  EXPECT_EQ(MockStats.counters.count("glow.device.used_memory.device0"), 0);
  EXPECT_EQ(MockStats.counters.count("glow.device.available_memory.device0"),
            0);
  {
    std::unique_ptr<DeviceManager> DM(
        DeviceManager::createDeviceManager(DeviceConfig("Interpreter")));
    EXPECT_EQ(MockStats.counters["glow.devices_used.interpreter"], 1);
    EXPECT_EQ(MockStats.counters["glow.device.used_memory.device0"], 0);
    EXPECT_EQ(MockStats.counters["glow.device.available_memory.device0"],
              2000000000);
  }
  EXPECT_EQ(MockStats.counters["glow.devices_used.interpreter"], 0);
}

TEST(StatsExporter, HostManager) {
  using namespace glow::runtime;
  EXPECT_EQ(MockStats.counters.count("glow.devices.used_memory.total"), 0);
  EXPECT_EQ(MockStats.counters.count("glow.devices.available_memory.total"), 0);
  EXPECT_EQ(MockStats.counters.count("glow.devices.maximum_memory.total"), 0);
  {
    auto deviceConfig = glow::make_unique<DeviceConfig>("Interpreter");
    std::vector<std::unique_ptr<DeviceConfig>> configs;
    configs.push_back(std::move(deviceConfig));
    std::unique_ptr<HostManager> HM =
        glow::make_unique<HostManager>(std::move(configs), HostConfig());
    EXPECT_EQ(MockStats.counters["glow.devices.used_memory.total"], 0);
    EXPECT_EQ(MockStats.counters["glow.devices.available_memory.total"],
              2000000000);
    EXPECT_EQ(MockStats.counters["glow.devices.maximum_memory.total"],
              2000000000);

    // Add a network so the memory used value is non-zero.
    std::unique_ptr<Module> module = glow::make_unique<Module>();
    Function *F = module->createFunction("main");
    auto *X = module->createConstant(ElemKind::FloatTy, {1024, 1024}, "X");
    auto *pow = F->createPow("Pow", X, 2.0);
    F->createSave("save", pow);

    const int64_t functionCost = X->getType()->getSizeInBytes();

    CompilationContext cctx;
    EXIT_ON_ERR(HM->addNetwork(std::move(module), cctx));
    // Currently the interpreter DM treats all added networks as size 1 byte so
    // expect to see 1 byte used.
    EXPECT_EQ(MockStats.counters["glow.devices.used_memory.total"],
              functionCost);
    EXPECT_EQ(MockStats.counters["glow.devices.available_memory.total"],
              2000000000 - functionCost);
  }
  EXPECT_EQ(MockStats.counters["glow.devices_used.interpreter"], 0);
}
