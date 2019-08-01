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

#include "glow/Runtime/StatsExporter.h"
#include "glow/Backends/DeviceManager.h"

#include <gtest/gtest.h>

#include <memory>

using namespace glow;

class MockStatsExporter : public StatsExporter {
public:
  MockStatsExporter() { Stats()->registerStatsExporter(this); }

  ~MockStatsExporter() override {}

  void addTimeSeriesValue(llvm::StringRef key, double value) override {
    timeSeries[key].push_back(value);
  }

  void incrementCounter(llvm::StringRef key, int64_t value) override {
    counters[key] += value;
  }

  void setCounter(llvm::StringRef key, int64_t value) override {
    counters[key] = value;
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
  {
    std::unique_ptr<DeviceManager> DM(
        DeviceManager::createDeviceManager(DeviceConfig("Interpreter")));
    EXPECT_EQ(MockStats.counters["glow.devices_used.interpreter"], 1);
  }
  EXPECT_EQ(MockStats.counters["glow.devices_used.interpreter"], 0);
}
