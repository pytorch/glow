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

#include "BackendTestUtils.h"

#include "glow/Backends/DeviceManager.h"
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
  NodeValue part_one(Function *F, ExecutionContext &context) {
    auto *CV0 = F->createConv(*context.getPlaceholderBindings(), "conv1",
                              inputPH, 16, 5, 1, 2, 1);
    auto *RL0 = F->createRELU("relu1", CV0);
    auto *MP0 = F->createMaxPool("pool1", RL0, 2, 2, 0);
    return MP0->getResult();
  }

  NodeValue part_two(Function *F, ExecutionContext &context, NodeValue last) {
    auto *CV1 = F->createConv(*context.getPlaceholderBindings(), "conv2", last,
                              20, 5, 1, 2, 1);
    auto *RL1 = F->createRELU("relu2", CV1);
    auto *MP1 = F->createMaxPool("pool2", RL1, 2, 2, 0);
    return MP1->getResult();
  }

  NodeValue part_three(Function *F, ExecutionContext &context, NodeValue last) {
    auto *CV2 = F->createConv(*context.getPlaceholderBindings(), "conv3", last,
                              20, 5, 1, 2, 1);
    auto *RL2 = F->createRELU("relu3", CV2);
    auto *MP2 = F->createMaxPool("pool3", RL2, 2, 2, 0);
    return MP2->getResult();
  }

  Node *part_four(Function *F, ExecutionContext &context, NodeValue last) {
    auto *ex = F->getParent()->createPlaceholder(ElemKind::Int64ITy, {1, 1},
                                                 "exp", false);
    auto *FCL1 = F->createFullyConnected(*context.getPlaceholderBindings(),
                                         "fc", last, 10);
    auto *RL3 = F->createRELU("relu4", FCL1);
    auto *SM = F->createSoftMax("sm", RL3, ex);
    auto *S = F->createSave("ret", SM);
    return S;
  }

  Placeholder *createEventPlaceholder(dim_t numEvents) {
    std::unique_ptr<Backend> backend(createBackend(EE_.getBackendName()));
    return EE_.getModule().createPlaceholder(
        ElemKind::Int64ITy,
        {numEvents, (dim_t)backend->getTraceEventDataSize() /
                        Type::getElementSize(ElemKind::Int64ITy)},
        "", false);
  }

  // Compares generated TraceEvents with their expected names and types.
  void checkEventMetadata(const std::list<TraceEvent> &traceEvents,
                          std::vector<std::pair<std::string, char>> expected) {

    ASSERT_EQ(traceEvents.size(), expected.size());
    unsigned index = 0;
    for (const auto &event : traceEvents) {
      const auto &pair = expected[index++];
      ASSERT_EQ(pair.first, event.name);
      ASSERT_EQ(pair.second, event.type);
    }
  }

  // In the function below we're trying to compare auto instrumented trace
  // output with optimized model graph. Since the code generation may make
  // further changes to the graph representation we limit checking to a set of
  // convolution / maxpool / cpumaxsplat kinds which are never changed.
  std::set<std::string> checkedKinds = {"convolution", "maxpool",
                                        "cpumaxsplat"};

  template <class ExecutionEngineTy>
  std::vector<std::pair<std::string, std::string>>
  prepareKindsForComparison(ExecutionEngineTy &ExecutionEngine) {
    std::vector<std::pair<std::string, std::string>> expectedKinds;
    for (auto &i :
         ExecutionEngine.getModule().getFunctions().front()->getNodes()) {
      std::string kind(Kinded::getKindName(i.getKind()));
      std::transform(kind.begin(), kind.end(), kind.begin(),
                     [](unsigned char c) { return ::tolower(c); });
      if (checkedKinds.find(kind) != checkedKinds.end()) {
        std::string name(Module::getPrefix(i.getName()));
        // Let's remove all digits at the end of name since the code generation
        // may create new nodes with names consisting of existing node names
        // and additional numeric indexes
        expectedKinds.emplace_back(name, kind);
      }
    }
    std::sort(expectedKinds.begin(), expectedKinds.end());
    return expectedKinds;
  }

  void checkEventMetadata(
      const std::list<TraceEvent> &traceEvents,
      const std::vector<std::pair<std::string, std::string>> &expected) {

    const auto &backend = GetParam();
    if (backend != "Interpreter" && backend != "CPU") {
      return;
    }

    auto map_element = [](const std::map<std::string, std::string> &m,
                          const std::string &k) {
      return m.find(k) == m.end() ? std::string() : m.find(k)->second;
    };
    std::vector<std::pair<std::string, std::string>> events_for_checking;
    for (auto &i : traceEvents) {
      std::string kind = map_element(i.args, "kind");
      if (checkedKinds.find(kind) != checkedKinds.end()) {
        std::string name(Module::getPrefix(i.name));
        // Let's remove all digits at the end of the name since the code
        // generation may create new nodes with names consisting of existing
        // node names and additional numeric indexes
        events_for_checking.emplace_back(name, kind);
      }
    }
    std::sort(events_for_checking.begin(), events_for_checking.end());
    ASSERT_EQ(expected, events_for_checking);
  }

  // Check timestamps are non-zero and monotonically increasing.
  void checkEventTimestamps(const std::list<TraceEvent> &traceEvents) {
    uint64_t last = 0;
    for (const auto &event : traceEvents) {
      ASSERT_NE(event.timestamp, 0);
      ASSERT_GE(event.timestamp, last);
      last = event.timestamp;
    }
  }
};

TEST_P(TraceEventsTest, manualEvents) {
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto n = part_one(F, context);
  n = part_two(F, context, n);

  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, context, n);
  part_four(F, context, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(cctx);

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
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 2;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  auto n = part_one(F, context);
  n = part_two(F, context, n);

  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, context, n);
  part_four(F, context, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(cctx);

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
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 2;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  auto n = part_one(F, context);

  F->createTraceEvent("middle section", "B", eventData, eventId++);
  n = part_two(F, context, n);
  n = part_three(F, context, n);
  F->createTraceEvent("middle section", "E", eventData, eventId++);

  part_four(F, context, n);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(cctx);

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
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  auto n = part_one(F, context);
  n = part_two(F, context, n);
  n = part_three(F, context, n);
  part_four(F, context, n);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  cctx.backendOpts.autoInstrument = true;
  EE_.compile(cctx);

  auto expectedKinds = prepareKindsForComparison(EE_);

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  EE_.run(context);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_GT(traceEvents.size(), 0);
  checkEventMetadata(traceEvents, expectedKinds);
  checkEventTimestamps(traceEvents);
}

TEST_P(TraceEventsTest, manualAndAutomatic) {
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto n = part_one(F, context);
  n = part_two(F, context, n);

  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, context, n);
  part_four(F, context, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  cctx.backendOpts.autoInstrument = true;
  EE_.compile(cctx);

  auto expectedKinds = prepareKindsForComparison(EE_);

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
  checkEventMetadata(traceEvents, expectedKinds);
}

/// Compile the same function twice with auto instrumentation on - ensure that
/// instrumentation doesn't break future compiles.
TEST_P(TraceEventsTest, twoCompiles) {
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto n = part_one(F, context);
  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);
  part_four(F, context, n);
  F->createTraceEvent("second half", "E", eventData, eventId++);

  ExecutionContext context2{context.clone()};
  context2.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());

  std::unique_ptr<Backend> backend(createBackend(EE_.getBackendName()));
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  cctx.backendOpts.autoInstrument = true;
  cctx.backendOpts.collectConstants = true;
  EXIT_ON_ERR(::glow::optimizeFunction(F, *backend, cctx));

  std::string name = F->getName().str();
  auto config =
      glow::make_unique<runtime::DeviceConfig>(backend->getBackendName());
  std::unique_ptr<runtime::DeviceManager> device(
      runtime::DeviceManager::createDeviceManager(*config));
  EXIT_ON_ERR(device->init());
  auto func1 = EXIT_ON_ERR(backend->compile(F, cctx.backendOpts));

  insertCompiledFunction(name, func1.get(), device.get(), &EE_.getModule());

  std::string name2 = name + "2";
  auto func2 = EXIT_ON_ERR(backend->compile(F, cctx.backendOpts));
  insertCompiledFunction(name2, func2.get(), device.get(), &EE_.getModule());
  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});

  runOnDevice(context, name, device.get());
  auto &traceEvents = context.getTraceContext()->getTraceEvents();

  ASSERT_GT(traceEvents.size(), 0);

  context2.getPlaceholderBindings()->allocate(
      EE_.getModule().getPlaceholders());

  // Add a little delay to ensure the timestamps don't happen to occur in the
  // same millisecond.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  updateInputPlaceholders(*context2.getPlaceholderBindings(), {inputPH},
                          {&inputs});
  runOnDevice(context2, name2, device.get());
  auto &traceEvents2 = context2.getTraceContext()->getTraceEvents();

  ASSERT_GT(traceEvents2.size(), 0);

  ASSERT_EQ(traceEvents.size(), traceEvents2.size());

  auto iter = traceEvents.begin();
  auto iter2 = traceEvents2.begin();
  while (iter != traceEvents.end()) {
    ASSERT_EQ(iter2->name, iter->name);
    ASSERT_EQ(iter2->type, iter->type);
    ASSERT_NE(iter2->timestamp, iter->timestamp);
    ++iter;
    ++iter2;
  }
}

TEST_P(TraceEventsTest, onlyTraceEvents) {
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

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
  EE_.compile(cctx);

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
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 6;
  auto *eventData1 = createEventPlaceholder(3);
  auto *eventData2 = createEventPlaceholder(3);

  F->createTraceEvent("event1", "B", eventData1, 0);
  auto n = part_one(F, context);
  F->createTraceEvent("event1", "E", eventData1, 1);

  F->createTraceEvent("event2", "B", eventData2, 0);
  n = part_two(F, context, n);
  F->createTraceEvent("event2", "E", eventData2, 1);

  // now lets split between two tensors

  auto *eventData3 = createEventPlaceholder(1);
  auto *eventData4 = createEventPlaceholder(1);

  F->createTraceEvent("event3", "B", eventData3, 0);
  n = part_three(F, context, n);
  part_four(F, context, n);
  F->createTraceEvent("event3", "E", eventData4, 0);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(cctx);

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
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  size_t numEvents = 4;
  auto *eventData = createEventPlaceholder(numEvents);
  unsigned eventId = 0;

  F->createTraceEvent("first half", "B", eventData, eventId++);
  auto n = part_one(F, context);
  n = part_two(F, context, n);

  F->createTraceEvent("first half", "E", eventData, eventId++);
  F->createTraceEvent("second half", "B", eventData, eventId++);

  n = part_three(F, context, n);
  part_four(F, context, n);

  F->createTraceEvent("second half", "E", eventData, eventId++);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(cctx);

  updateInputPlaceholders(*context.getPlaceholderBindings(), {inputPH},
                          {&inputs});

  ExecutionContext context2{context.clone()};
  context2.setTraceContext(
      glow::make_unique<TraceContext>(TraceLevel::OPERATOR));

  // run twice
  EE_.run(context);
  EE_.run(context2);

  auto &traceEvents = context.getTraceContext()->getTraceEvents();
  auto &traceEvents2 = context2.getTraceContext()->getTraceEvents();

  ASSERT_GE(traceEvents.size(), numEvents);
  ASSERT_GE(traceEvents2.size(), numEvents);

  auto iter = traceEvents.begin();
  auto iter2 = traceEvents2.begin();
  while (iter != traceEvents.end()) {
    ASSERT_EQ(iter2->name, iter->name);
    ASSERT_EQ(iter2->type, iter->type);
    // timestamps are not equal
    ASSERT_NE(iter2->timestamp, iter->timestamp);
    ++iter;
    ++iter2;
  }
}

TEST_P(TraceEventsTest, deviceManagerEvents) {
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(glow::make_unique<TraceContext>(
      TraceLevel::RUNTIME | TraceLevel::OPERATOR));

  auto n = part_one(F, context);
  n = part_two(F, context, n);
  n = part_three(F, context, n);
  part_four(F, context, n);

  context.getPlaceholderBindings()->allocate(EE_.getModule().getPlaceholders());

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE_.compile(cctx);

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
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(glow::make_unique<TraceContext>(
      TraceLevel::RUNTIME | TraceLevel::OPERATOR));

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
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(glow::make_unique<TraceContext>(
      TraceLevel::RUNTIME | TraceLevel::OPERATOR));

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
  CHECK_IF_ENABLED();
  ExecutionContext context;
  context.setTraceContext(glow::make_unique<TraceContext>(
      TraceLevel::RUNTIME | TraceLevel::OPERATOR));

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
  CHECK_IF_ENABLED();
  std::array<TraceLevel, 5> levels = {{TraceLevel::NONE, TraceLevel::REQUEST,
                                       TraceLevel::RUNTIME, TraceLevel::COPY,
                                       TraceLevel::OPERATOR}};
  for (auto L : levels) {
    TraceContext context(L);
    for (auto evl : levels) {
      context.logTraceEvent("event", evl);
    }

    if (L == TraceLevel::NONE) {
      EXPECT_EQ(context.getTraceEvents().size(), 0);
    } else {
      ASSERT_EQ(context.getTraceEvents().size(), 1);
      ASSERT_EQ(context.getTraceEvents().front().name, "event");
    }
  }

  TraceContext context(TraceLevel::STANDARD);
  for (auto evl : levels) {
    context.logTraceEvent("event", evl);
  }
  ASSERT_EQ(context.getTraceEvents().size(), 4);
}

TEST(TraceEventsTest, MergeEvents) {
  auto tc1 = glow::make_unique<TraceContext>(TraceLevel::RUNTIME |
                                             TraceLevel::OPERATOR);
  auto tc2 = glow::make_unique<TraceContext>(TraceLevel::RUNTIME |
                                             TraceLevel::OPERATOR);

  TRACE_EVENT_BEGIN(tc1, TraceLevel::RUNTIME, "ev1");
  TRACE_EVENT_END(tc1, TraceLevel::RUNTIME, "ev1");

  ASSERT_EQ(tc1->getTraceEvents().size(), 2);
  ASSERT_EQ(tc2->getTraceEvents().size(), 0);

  tc2->merge(tc1.get());

  ASSERT_EQ(tc1->getTraceEvents().size(), 0);
  ASSERT_EQ(tc2->getTraceEvents().size(), 2);

  TRACE_EVENT_BEGIN(tc1, TraceLevel::RUNTIME, "ev2");
  TRACE_EVENT_END(tc1, TraceLevel::RUNTIME, "ev2");

  ASSERT_EQ(tc1->getTraceEvents().size(), 2);
  ASSERT_EQ(tc2->getTraceEvents().size(), 2);

  tc2->merge(tc1.get());

  ASSERT_EQ(tc1->getTraceEvents().size(), 0);
  ASSERT_EQ(tc2->getTraceEvents().size(), 4);
}

INSTANTIATE_BACKEND_TEST(TraceEventsTest);
