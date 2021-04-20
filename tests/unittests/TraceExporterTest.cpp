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

#include "glow/Runtime/TraceExporter.h"

#include <gtest/gtest.h>

#include <memory>

using namespace glow;

class MockTraceExporter : public TraceExporter {
  std::shared_ptr<TraceExporterRegistry> traceExporterRegistry_;
  bool enable_{false};

public:
  MockTraceExporter()
      : traceExporterRegistry_(TraceExporterRegistry::getInstance()) {
    traceExporterRegistry_->registerTraceExporter(this);

    mergedTraceContext_ = std::make_unique<TraceContext>(TraceLevel::STANDARD);
  }

  ~MockTraceExporter() override {
    traceExporterRegistry_->revokeTraceExporter(this);
  }

  bool shouldTrace() override { return enable_; }

  void enableTrace() { enable_ = true; }

  void disableTrace() { enable_ = false; }

  void exportTrace(TraceContext *tcontext) override {
    // create a copy of trace events
    mergedTraceContext_->copy(tcontext);
  }

  std::unique_ptr<TraceContext> mergedTraceContext_;
};

TEST(TraceExporter, shouldTrace) {

  // if no trace exporter is registered should not have any side effects
  auto traceExporter = TraceExporterRegistry::getInstance();
  EXPECT_FALSE(traceExporter->shouldTrace());

  MockTraceExporter mockExporter;

  mockExporter.disableTrace();
  EXPECT_FALSE(traceExporter->shouldTrace());

  mockExporter.enableTrace();
  EXPECT_TRUE(traceExporter->shouldTrace());

  mockExporter.disableTrace();
  EXPECT_FALSE(traceExporter->shouldTrace());
}

TEST(TraceExporter, traceEvents) {

  auto traceExporter = TraceExporterRegistry::getInstance();
  MockTraceExporter mockExporter;

  TraceContext glowTrace{TraceLevel::STANDARD};

  glowTrace.logTraceEvent("foo_function", TraceLevel::RUNTIME, 'B');
  glowTrace.logTraceEvent("bar_function", TraceLevel::RUNTIME, 'B');
  glowTrace.logTraceEvent("bar_function", TraceLevel::RUNTIME, 'E');
  glowTrace.logTraceEvent("foo_function", TraceLevel::RUNTIME, 'E');
  glowTrace.logCompleteTraceEvent("alice", TraceLevel::RUNTIME,
                                  TraceEvent::now() - 100);

  if (traceExporter->shouldTrace()) {
    traceExporter->exportTrace(&glowTrace);
  }

  mockExporter.enableTrace();
  if (traceExporter->shouldTrace()) {
    traceExporter->exportTrace(&glowTrace);
  }

  // add two traces
  if (traceExporter->shouldTrace()) {
    traceExporter->exportTrace(&glowTrace);
  }

  auto traceEvents = mockExporter.mergedTraceContext_->getTraceEvents();
  EXPECT_EQ(traceEvents.size(), 10);
}
