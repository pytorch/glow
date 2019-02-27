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

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IRBuilder.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

class TraceEventsTest : public ::testing::TestWithParam<BackendKind> {
public:
  ExecutionEngine EE_{GetParam()};
  Tensor inputs{ElemKind::FloatTy, {1, 32, 32, 3}};
  Placeholder *inputPH;

  Function *F;

  void SetUp() override {
    auto &mod = EE_.getModule();
    inputPH = mod.createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 3}, "input",
                                    false);
    F = mod.createFunction("main");
    F->setName("interpret");
  }

  // Split a sample network into four parts to make it easy to insert
  // TraceEventNodes.
  Node *part_one(Function *F, Context &ctx) {
    auto *CV0 = F->createConv(ctx, "conv1", inputPH, 16, 5, 1, 2, 1);
    auto *RL0 = F->createRELU("relu1", CV0);
    auto *MP0 = F->createMaxPool("pool1", RL0, 2, 2, 0);
    return MP0;
  }

  Node *part_two(Function *F, Context &ctx, Node *last) {
    auto *CV1 = F->createConv(ctx, "conv2", last, 20, 5, 1, 2, 1);
    auto *RL1 = F->createRELU("relu2", CV1);
    auto *MP1 = F->createMaxPool("pool2", RL1, 2, 2, 0);
    return MP1;
  }

  Node *part_three(Function *F, Context &ctx, Node *last) {
    auto *CV2 = F->createConv(ctx, "conv3", last, 20, 5, 1, 2, 1);
    auto *RL2 = F->createRELU("relu3", CV2);
    auto *MP2 = F->createMaxPool("pool3", RL2, 2, 2, 0);
    return MP2;
  }

  Node *part_four(Function *F, Context &ctx, Node *last) {
    auto *ex = F->getParent()->createPlaceholder(ElemKind::Int64ITy, {1, 1},
                                                 "exp", false);
    auto *FCL1 = F->createFullyConnected(ctx, "fc", last, 10);
    auto *RL3 = F->createRELU("relu4", FCL1);
    auto *SM = F->createSoftMax("sm", RL3, ex);
    auto *S = F->createSave("ret", SM);
    return S;
  }

  Placeholder *createEventPlaceholder(size_t numEvents) {
    return EE_.getModule().createPlaceholder(
        ElemKind::Int64ITy,
        {numEvents, EE_.getBackend()->getTraceEventDataSize() /
                        Type::getElementSize(ElemKind::Int64ITy)},
        "", false);
  }

  // Compares generated TraceEvents with their expected names and types.
  void checkEventMetadata(
      const std::vector<TraceEvent> &traceEvents,
      std::vector<std::pair<std::string, std::string>> expected) {

    ASSERT_EQ(traceEvents.size(), expected.size());
    unsigned index = 0;
    for (auto &pair : expected) {
      ASSERT_EQ(pair.first, traceEvents[index].name);
      ASSERT_EQ(pair.second, traceEvents[index].type);
      index++;
    }
  }

  // Check timestamps are non-zero and monotonically increasing.
  void checkEventTimestamps(const std::vector<TraceEvent> &traceEvents) {
    uint64_t last = 0;
    for (const auto &event : traceEvents) {
      ASSERT_NE(event.timestamp, 0);
      ASSERT_GE(event.timestamp, last);
      last = event.timestamp;
    }
  }
};

TEST_P(TraceEventsTest, manualEvents) {
  Context ctx;

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto *n = part_one(F, ctx);
  n = part_two(F, ctx, n);

  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, ctx, n);
  n = part_four(F, ctx, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  ctx.allocate(EE_.getModule().getPlaceholders());
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  EE_.compile(F, opts);

  updateInputPlaceholders(ctx, {inputPH}, {&inputs});
  EE_.run(ctx);

  auto &traceEvents = ctx.getTraceEvents();

  ASSERT_EQ(traceEvents.size(), numEvents);
  checkEventMetadata(traceEvents, {{"first half", "B"},
                                   {"first half", "E"},
                                   {"second half", "B"},
                                   {"second half", "E"}});

  checkEventTimestamps(traceEvents);

  // Check that each timestamp matches the tensor
  auto eventHandle = ctx.get(eventData)->getHandle<int64_t>();
  eventId = 0;
  for (const auto &event : traceEvents) {
    ASSERT_EQ(event.timestamp, eventHandle.at({eventId++, 0}));
  }
}

TEST_P(TraceEventsTest, incompleteCoverage) {
  Context ctx;

  size_t numEvents = 2;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  auto *n = part_one(F, ctx);
  n = part_two(F, ctx, n);

  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, ctx, n);
  n = part_four(F, ctx, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  ctx.allocate(EE_.getModule().getPlaceholders());
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  EE_.compile(F, opts);

  updateInputPlaceholders(ctx, {inputPH}, {&inputs});
  EE_.run(ctx);

  auto &traceEvents = ctx.getTraceEvents();

  ASSERT_EQ(traceEvents.size(), numEvents);
  checkEventMetadata(traceEvents, {{"second half", "B"}, {"second half", "E"}});

  checkEventTimestamps(traceEvents);

  // Check that each timestamp matches the tensor
  auto eventHandle = ctx.get(eventData)->getHandle<int64_t>();
  eventId = 0;
  for (const auto &event : traceEvents) {
    ASSERT_EQ(event.timestamp, eventHandle.at({eventId++, 0}));
  }
}

TEST_P(TraceEventsTest, internalGap) {
  Context ctx;

  size_t numEvents = 2;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  auto *n = part_one(F, ctx);

  F->createTraceEvent("middle section", "B", eventData, eventId++);
  n = part_two(F, ctx, n);
  n = part_three(F, ctx, n);
  F->createTraceEvent("middle section", "E", eventData, eventId++);

  n = part_four(F, ctx, n);

  ctx.allocate(EE_.getModule().getPlaceholders());
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  EE_.compile(F, opts);

  updateInputPlaceholders(ctx, {inputPH}, {&inputs});
  EE_.run(ctx);

  auto &traceEvents = ctx.getTraceEvents();

  ASSERT_EQ(traceEvents.size(), numEvents);
  checkEventMetadata(traceEvents,
                     {{"middle section", "B"}, {"middle section", "E"}});

  checkEventTimestamps(traceEvents);

  // Check that each timestamp matches the tensor
  auto eventHandle = ctx.get(eventData)->getHandle<int64_t>();
  eventId = 0;
  for (const auto &event : traceEvents) {
    ASSERT_EQ(event.timestamp, eventHandle.at({eventId++, 0}));
  }
}

TEST_P(TraceEventsTest, automaticInstrumentation) {
  Context ctx;

  auto *n = part_one(F, ctx);
  n = part_two(F, ctx, n);
  n = part_three(F, ctx, n);
  n = part_four(F, ctx, n);

  ctx.allocate(EE_.getModule().getPlaceholders());
  auto *backend = EE_.getBackend();
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  opts.autoInstrument = true;
  backend->optimizeFunction(F, opts);
  EE_.insertCompiledFunction(F->getName(), backend->compile(F, opts));

  updateInputPlaceholders(ctx, {inputPH}, {&inputs});
  EE_.run(ctx);

  auto &traceEvents = ctx.getTraceEvents();

  ASSERT_GT(traceEvents.size(), 0);
  checkEventTimestamps(traceEvents);
}

TEST_P(TraceEventsTest, manualAndAutomatic) {
  Context ctx;

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto *n = part_one(F, ctx);
  n = part_two(F, ctx, n);

  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, ctx, n);
  n = part_four(F, ctx, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  ctx.allocate(EE_.getModule().getPlaceholders());
  auto *backend = EE_.getBackend();
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  opts.autoInstrument = true;
  backend->optimizeFunction(F, opts);
  EE_.insertCompiledFunction(F->getName(), backend->compile(F, opts));

  updateInputPlaceholders(ctx, {inputPH}, {&inputs});
  EE_.run(ctx);

  auto &traceEvents = ctx.getTraceEvents();

  ASSERT_GT(traceEvents.size(), numEvents);
  size_t manualEvents = 0;
  for (const auto &event : traceEvents) {
    if (event.name == "first half" || event.name == "second half") {
      manualEvents++;
    }
  }
  ASSERT_EQ(manualEvents, numEvents);
}

TEST_P(TraceEventsTest, onlyTraceEvents) {
  Context ctx;

  size_t numEvents = 16;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  std::vector<std::pair<std::string, std::string>> expected;
  for (unsigned eventId = 0; eventId < numEvents; ++eventId) {
    std::string name = "event_" + std::to_string(eventId);
    F->createTraceEvent(name, "X", eventData, eventId);
    expected.push_back({name, "X"});
  }

  ctx.allocate(EE_.getModule().getPlaceholders());
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  EE_.compile(F, opts);

  updateInputPlaceholders(ctx, {inputPH}, {&inputs});
  EE_.run(ctx);

  auto &traceEvents = ctx.getTraceEvents();

  ASSERT_EQ(traceEvents.size(), numEvents);
  checkEventMetadata(traceEvents, expected);
  checkEventTimestamps(traceEvents);

  // Check that each timestamp matches the tensor
  auto eventHandle = ctx.get(eventData)->getHandle<int64_t>();
  eventId = 0;
  for (const auto &event : traceEvents) {
    ASSERT_EQ(event.timestamp, eventHandle.at({eventId++, 0}));
  }
}

TEST_P(TraceEventsTest, multipleBackingTensors) {
  Context ctx;

  size_t numEvents = 6;
  auto *eventData1 = createEventPlaceholder(3);
  auto *eventData2 = createEventPlaceholder(3);

  F->createTraceEvent("event1", "B", eventData1, 0);
  auto *n = part_one(F, ctx);
  F->createTraceEvent("event1", "E", eventData1, 1);

  F->createTraceEvent("event2", "B", eventData2, 0);
  n = part_two(F, ctx, n);
  F->createTraceEvent("event2", "E", eventData2, 1);

  // now lets split between two tensors

  auto *eventData3 = createEventPlaceholder(1);
  auto *eventData4 = createEventPlaceholder(1);

  F->createTraceEvent("event3", "B", eventData3, 0);
  n = part_three(F, ctx, n);
  n = part_four(F, ctx, n);
  F->createTraceEvent("event3", "E", eventData4, 0);

  ctx.allocate(EE_.getModule().getPlaceholders());
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  EE_.compile(F, opts);

  updateInputPlaceholders(ctx, {inputPH}, {&inputs});
  EE_.run(ctx);

  auto &traceEvents = ctx.getTraceEvents();

  ASSERT_EQ(traceEvents.size(), numEvents);

  // Can't use checkEventMetadata since events aren't ordered
  size_t event1{0}, event2{0}, event3{0};
  for (const auto &event : traceEvents) {
    if (event.name == "event1") {
      event1++;
    } else if (event.name == "event2") {
      event2++;
    } else if (event.name == "event3") {
      event3++;
    } else {
      ASSERT_FALSE("unknown event name");
    }
  }

  ASSERT_EQ(event1, 2);
  ASSERT_EQ(event2, 2);
  ASSERT_EQ(event3, 2);
}

TEST_P(TraceEventsTest, multipleRunsAreDistinct) {
  Context ctx;

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto *n = part_one(F, ctx);
  n = part_two(F, ctx, n);

  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, ctx, n);
  n = part_four(F, ctx, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  ctx.allocate(EE_.getModule().getPlaceholders());
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  EE_.compile(F, opts);

  updateInputPlaceholders(ctx, {inputPH}, {&inputs});

  Context ctx2{ctx.clone()};

  // run twice
  EE_.run(ctx);
  EE_.run(ctx2);

  auto &traceEvents = ctx.getTraceEvents();
  auto &traceEvents2 = ctx2.getTraceEvents();

  ASSERT_EQ(traceEvents.size(), numEvents);
  ASSERT_EQ(traceEvents2.size(), numEvents);

  for (unsigned i = 0; i < numEvents; ++i) {
    ASSERT_EQ(traceEvents[i].name, traceEvents[i].name);
    ASSERT_EQ(traceEvents[i].type, traceEvents[i].type);
    // timestamps are not equal
    ASSERT_EQ(traceEvents[i].timestamp, traceEvents[i].timestamp);
  }
}

INSTANTIATE_TEST_CASE_P(Interpreter, TraceEventsTest,
                        ::testing::Values(BackendKind::Interpreter));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, TraceEventsTest,
                        ::testing::Values(BackendKind::CPU));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
// INSTANTIATE_TEST_CASE_P(OpenCL, TraceEventsTest,
//                        ::testing::Values(BackendKind::OpenCL));
#endif // GLOW_WITH_OPENCL
