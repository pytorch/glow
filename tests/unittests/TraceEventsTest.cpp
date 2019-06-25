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

#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <thread>

using namespace glow;

class TraceEventsTest : public ::testing::TestWithParam<std::string> {
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
  }

  // Split a sample network into four parts to make it easy to insert
  // TraceEventNodes.
  Node *part_one(Function *F, ExecutionContext &context) {
    auto *CV0 = F->createConv(*context.getPlaceholderBindings(), "conv1",
                              inputPH, 16, 5, 1, 2, 1);
    auto *RL0 = F->createRELU("relu1", CV0);
    auto *MP0 = F->createMaxPool("pool1", RL0, 2, 2, 0);
    return MP0;
  }

  Node *part_two(Function *F, ExecutionContext &context, Node *last) {
    auto *CV1 = F->createConv(*context.getPlaceholderBindings(), "conv2", last,
                              20, 5, 1, 2, 1);
    auto *RL1 = F->createRELU("relu2", CV1);
    auto *MP1 = F->createMaxPool("pool2", RL1, 2, 2, 0);
    return MP1;
  }

  Node *part_three(Function *F, ExecutionContext &context, Node *last) {
    auto *CV2 = F->createConv(*context.getPlaceholderBindings(), "conv3", last,
                              20, 5, 1, 2, 1);
    auto *RL2 = F->createRELU("relu3", CV2);
    auto *MP2 = F->createMaxPool("pool3", RL2, 2, 2, 0);
    return MP2;
  }

  Node *part_four(Function *F, ExecutionContext &context, Node *last) {
    auto *ex = F->getParent()->createPlaceholder(ElemKind::Int64ITy, {1, 1},
                                                 "exp", false);
    auto *FCL1 = F->createFullyConnected(*context.getPlaceholderBindings(),
                                         "fc", last, 10);
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
  void checkEventMetadata(const std::vector<TraceEvent> &traceEvents,
                          std::vector<std::pair<std::string, char>> expected) {

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
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto *n = part_one(F, context);
  n = part_two(F, context, n);

  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, context, n);
  n = part_four(F, context, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(F, cctx);

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_EQ(traceEvents.size(), numEvents);
  checkEventMetadata(traceEvents, {{"first half", 'B'},
                                   {"first half", 'E'},
                                   {"second half", 'B'},
                                   {"second half", 'E'}});

  checkEventTimestamps(traceEvents);

  // Check that each timestamp matches the tensor
  auto eventHandle =
      context.getPlaceholderBindings()->get(eventData)->getHandle<int64_t>();
  eventId = 0;
  for (const auto &event : traceEvents) {
    ASSERT_EQ(event.timestamp, eventHandle.at({eventId++, 0}));
  }
}

TEST_P(TraceEventsTest, incompleteCoverage) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 2;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  auto *n = part_one(F, context);
  n = part_two(F, context, n);

  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, context, n);
  n = part_four(F, context, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(F, cctx);

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_GE(traceEvents.size(), numEvents);
  checkEventMetadata(traceEvents, {{"second half", 'B'}, {"second half", 'E'}});

  checkEventTimestamps(traceEvents);

  // Check that each timestamp matches the tensor
  auto eventHandle =
      context.getPlaceholderBindings()->get(eventData)->getHandle<int64_t>();
  eventId = 0;
  for (const auto &event : traceEvents) {
    ASSERT_EQ(event.timestamp, eventHandle.at({eventId++, 0}));
  }
}

TEST_P(TraceEventsTest, internalGap) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 2;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  auto *n = part_one(F, context);

  F->createTraceEvent("middle section", "B", eventData, eventId++);
  n = part_two(F, context, n);
  n = part_three(F, context, n);
  F->createTraceEvent("middle section", "E", eventData, eventId++);

  n = part_four(F, context, n);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(F, cctx);

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_GE(traceEvents.size(), numEvents);
  checkEventMetadata(traceEvents,
                     {{"middle section", 'B'}, {"middle section", 'E'}});

  checkEventTimestamps(traceEvents);

  // Check that each timestamp matches the tensor
  auto eventHandle =
      context.getPlaceholderBindings()->get(eventData)->getHandle<int64_t>();
  eventId = 0;
  for (const auto &event : traceEvents) {
    ASSERT_EQ(event.timestamp, eventHandle.at({eventId++, 0}));
  }
}

TEST_P(TraceEventsTest, automaticInstrumentation) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  auto *n = part_one(F, context);
  n = part_two(F, context, n);
  n = part_three(F, context, n);
  n = part_four(F, context, n);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  auto *backend = EE_.getBackend();
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  cctx.backendOpts.autoInstrument = true;
  EXIT_ON_ERR(::glow::optimizeFunction(F, *backend, cctx));
  EE_.insertCompiledFunction(
      F->getName(), EXIT_ON_ERR(backend->compile(F, cctx.backendOpts)));

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_GT(traceEvents.size(), 0);
  checkEventTimestamps(traceEvents);
}

TEST_P(TraceEventsTest, manualAndAutomatic) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto *n = part_one(F, context);
  n = part_two(F, context, n);

  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, context, n);
  n = part_four(F, context, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  auto *backend = EE_.getBackend();
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  cctx.backendOpts.autoInstrument = true;
  EXIT_ON_ERR(::glow::optimizeFunction(F, *backend, cctx));
  EE_.insertCompiledFunction(
      F->getName(), EXIT_ON_ERR(backend->compile(F, cctx.backendOpts)));

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();
  // Can't use CheckTimestamps here because the manual & auto inserted
  // timestamps are not sorted by time.

  ASSERT_GT(traceEvents.size(), numEvents);
  size_t manualEvents = 0;
  for (const auto &event : traceEvents) {
    if (event.name == "first half" || event.name == "second half") {
      manualEvents++;
    }
  }
  ASSERT_EQ(manualEvents, numEvents);
}

/// Compile the same function twice with auto instrumentation on - ensure that
/// instrumentation doesn't break future compiles.
TEST_P(TraceEventsTest, twoCompiles) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto *n = part_one(F, context);
  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);
  n = part_four(F, context, n);
  F->createTraceEvent("second half", "E", eventData, eventId++);

  ExecutionContext context2{context.clone()};
  context2.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());

  auto *backend = EE_.getBackend();
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  cctx.backendOpts.autoInstrument = true;
  EXIT_ON_ERR(::glow::optimizeFunction(F, *backend, cctx));

  std::string name = F->getName();
  EE_.insertCompiledFunction(
      name, EXIT_ON_ERR(backend->compile(F, cctx.backendOpts)));

  std::string name2 = name + "2";
  EE_.insertCompiledFunction(
      name2, EXIT_ON_ERR(backend->compile(F, cctx.backendOpts)));

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context, name);
  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_GT(traceEvents.size(), 0);

  context2.getPlaceholderBindings()->allocate(
      EE_.getModule().getPlaceholders());

  // Add a little delay to ensure the timestamps don't happen to occur in the
  // same millisecond.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  EE_.run(context2, name2);
  auto &traceEvents2 = context2.getTraceContext()->getTraceEvents();

  ASSERT_GT(traceEvents2.size(), 0);

  ASSERT_EQ(traceEvents.size(), traceEvents2.size());

  for (unsigned i = 0; i < traceEvents.size(); ++i) {
    ASSERT_EQ(traceEvents2[i].name, traceEvents[i].name);
    ASSERT_EQ(traceEvents2[i].type, traceEvents[i].type);
    ASSERT_GT(traceEvents2[i].timestamp, traceEvents[i].timestamp);
  }
}

TEST_P(TraceEventsTest, onlyTraceEvents) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 16;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  std::vector<std::pair<std::string, char>> expected;
  for (unsigned eventId = 0; eventId < numEvents; ++eventId) {
    std::string name = "event_" + std::to_string(eventId);
    F->createTraceEvent(name, "X", eventData, eventId);
    expected.push_back({name, 'X'});
  }

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(F, cctx);

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_GE(traceEvents.size(), numEvents);
  checkEventMetadata(traceEvents, expected);
  checkEventTimestamps(traceEvents);

  // Check that each timestamp matches the tensor
  auto eventHandle =
      context.getPlaceholderBindings()->get(eventData)->getHandle<int64_t>();
  eventId = 0;
  for (const auto &event : traceEvents) {
    ASSERT_EQ(event.timestamp, eventHandle.at({eventId++, 0}));
  }
}

TEST_P(TraceEventsTest, multipleBackingTensors) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 6;
  auto *eventData1 = createEventPlaceholder(3);
  auto *eventData2 = createEventPlaceholder(3);

  F->createTraceEvent("event1", "B", eventData1, 0);
  auto *n = part_one(F, context);
  F->createTraceEvent("event1", "E", eventData1, 1);

  F->createTraceEvent("event2", "B", eventData2, 0);
  n = part_two(F, context, n);
  F->createTraceEvent("event2", "E", eventData2, 1);

  // now lets split between two tensors

  auto *eventData3 = createEventPlaceholder(1);
  auto *eventData4 = createEventPlaceholder(1);

  F->createTraceEvent("event3", "B", eventData3, 0);
  n = part_three(F, context, n);
  n = part_four(F, context, n);
  F->createTraceEvent("event3", "E", eventData4, 0);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(F, cctx);

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_GE(traceEvents.size(), numEvents);

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
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto *n = part_one(F, context);
  n = part_two(F, context, n);

  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, context, n);
  n = part_four(F, context, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(F, cctx);

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});

  ExecutionContext context2{context.clone()};
  context2.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::OPERATOR));

  // run twice
  EE_.run(context);
  EE_.run(context2);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();
  auto &traceEvents2 = context2.getTraceContext()->getTraceEvents();

  ASSERT_GE(traceEvents.size(), numEvents);
  ASSERT_GE(traceEvents2.size(), numEvents);

  for (unsigned i = 0; i < numEvents; ++i) {
    ASSERT_EQ(traceEvents[i].name, traceEvents[i].name);
    ASSERT_EQ(traceEvents[i].type, traceEvents[i].type);
    // timestamps are not equal
    ASSERT_EQ(traceEvents[i].timestamp, traceEvents[i].timestamp);
  }
}

TEST_P(TraceEventsTest, deviceManagerEvents) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::STANDARD));

  auto *n = part_one(F, context);
  n = part_two(F, context, n);
  n = part_three(F, context, n);
  n = part_four(F, context, n);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(F, cctx);

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_GT(traceEvents.size(), 0);
  // CompleteEvents are not necessarily monotonically increasing since they are
  // added to the log when they end, not when they start.
}

/// Test that ScopedTraceBlocks can be nested.
TEST(TraceEventsTest, nestedScopedEvents) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::STANDARD));

  TraceContext *tc = context.getTraceContext();

  ScopedTraceBlock block_one(tc, TraceLevel::RUNTIME, "one");
  {
    ScopedTraceBlock block_two(tc, TraceLevel::RUNTIME, "two");
    /* sleep_override */ std::this_thread::sleep_for(
        std::chrono::milliseconds(1));
  }

  {
    ScopedTraceBlock block_three(tc, TraceLevel::RUNTIME, "three");
    /* sleep_override */ std::this_thread::sleep_for(
        std::chrono::milliseconds(1));
    {
      ScopedTraceBlock block_four(tc, TraceLevel::RUNTIME, "four");
      /* sleep_override */ std::this_thread::sleep_for(
          std::chrono::milliseconds(1));
    }
  }

  block_one.end();

  auto &traceEvents = tc->getTraceEvents();
  ASSERT_EQ(traceEvents.size(), 4);
  llvm::StringMap<uint64_t> durations;
  for (auto &tc : traceEvents) {
    durations[tc.name] = tc.duration;
  }

  ASSERT_GE(durations["one"], durations["two"] + durations["three"]);
  ASSERT_GE(durations["three"], durations["four"]);
}

/// Test that nesting scoped events work with the macro versions.
TEST(TraceEventsTest, nestedScopedEventsMacro) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::STANDARD));

  TraceContext *tc = context.getTraceContext();

  TRACE_EVENT_SCOPE(tc, TraceLevel::RUNTIME, "one");
  {
    TRACE_EVENT_SCOPE(tc, TraceLevel::RUNTIME, "two");
    /* sleep_override */ std::this_thread::sleep_for(
        std::chrono::milliseconds(1));
  }

  {
    TRACE_EVENT_SCOPE(tc, TraceLevel::RUNTIME, "three");
    /* sleep_override */ std::this_thread::sleep_for(
        std::chrono::milliseconds(1));
    {
      TRACE_EVENT_SCOPE(tc, TraceLevel::RUNTIME, "four");
      /* sleep_override */ std::this_thread::sleep_for(
          std::chrono::milliseconds(1));
    }
  }

  TRACE_EVENT_SCOPE_END();

  auto &traceEvents = tc->getTraceEvents();
  ASSERT_EQ(traceEvents.size(), 4);
  llvm::StringMap<uint64_t> durations;
  for (auto &tc : traceEvents) {
    durations[tc.name] = tc.duration;
  }

  ASSERT_GE(durations["one"], durations["two"] + durations["three"]);
  ASSERT_GT(durations["three"], durations["four"]);
}

/// Test that terminating a scoped event logs final timestamp at the end, not at
/// scope exit.
TEST(TraceEventsTest, nestedScopedEventsTerm) {
  ExecutionContext context;
  context.setTraceContext(
      llvm::make_unique<TraceContext>(TraceLevel::STANDARD));

  TraceContext *tc = context.getTraceContext();

  {
    TRACE_EVENT_SCOPE_NAMED(tc, TraceLevel::RUNTIME, "one", one);
    TRACE_EVENT_SCOPE_NAMED(tc, TraceLevel::RUNTIME, "two", two);
    /* sleep_override */ std::this_thread::sleep_for(
        std::chrono::milliseconds(1));
    TRACE_EVENT_SCOPE_END_NAMED(one);
    /* sleep_override */ std::this_thread::sleep_for(
        std::chrono::milliseconds(1));
  }

  {
    TRACE_EVENT_SCOPE_NAMED(tc, TraceLevel::RUNTIME, "three", three);
    TRACE_EVENT_SCOPE_NAMED(tc, TraceLevel::RUNTIME, "four", four);
    {
      TRACE_EVENT_SCOPE_NAMED(tc, TraceLevel::RUNTIME, "five", five);
      /* sleep_override */ std::this_thread::sleep_for(
          std::chrono::milliseconds(1));
      TRACE_EVENT_SCOPE_END_NAMED(four);
      /* sleep_override */ std::this_thread::sleep_for(
          std::chrono::milliseconds(1));
    }
    /* sleep_override */ std::this_thread::sleep_for(
        std::chrono::milliseconds(1));
  }

  auto &traceEvents = tc->getTraceEvents();
  ASSERT_EQ(traceEvents.size(), 5);
  llvm::StringMap<uint64_t> durations;
  for (auto &tc : traceEvents) {
    durations[tc.name] = tc.duration;
  }

  // Two should have two sleeps to one's one.
  ASSERT_GT(durations["two"], durations["one"]);

  // Three includes both four and five but theres some overlap so can't just
  // add.
  ASSERT_GT(durations["three"], durations["four"]);
  ASSERT_GT(durations["three"], durations["five"]);
  ASSERT_GT(durations["five"], durations["four"]);
}

TEST(TraceEventsTest, TraceLevels) {
  std::array<TraceLevel, 4> levels = {TraceLevel::NONE, TraceLevel::REQUEST,
                                      TraceLevel::RUNTIME,
                                      TraceLevel::OPERATOR};
  for (auto L : levels) {
    TraceContext context(L);
    for (auto evl : levels) {
      context.logTraceEvent("event", evl);
    }

    if (L == TraceLevel::NONE) {
      EXPECT_EQ(context.getTraceEvents().size(), 0);
    } else {
      ASSERT_EQ(context.getTraceEvents().size(), 1);
      ASSERT_EQ(context.getTraceEvents()[0].name, "event");
    }
  }

  TraceContext context(TraceLevel::STANDARD);
  for (auto evl : levels) {
    context.logTraceEvent("event", evl);
  }
  ASSERT_EQ(context.getTraceEvents().size(), 2);
}

INSTANTIATE_TEST_CASE_P(Interpreter, TraceEventsTest,
                        ::testing::Values("Interpreter"));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, TraceEventsTest, ::testing::Values("CPU"));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, TraceEventsTest, ::testing::Values("OpenCL"));
#endif // GLOW_WITH_OPENCL
