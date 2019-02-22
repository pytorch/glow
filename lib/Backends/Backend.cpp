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

#include "glow/Backends/Backend.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"

using namespace glow;

void Backend::optimizeFunction(CompilationMode mode, Function *F) const {
  // Verify the function pre-optimization/lowering.
  assert(F->verify() && "Function must be valid");

  // Optimize the graph.
  ::glow::optimize(F, mode);

  // Lower the graph into a sequence of low-level linear algebra operations.
  ::glow::lower(F, /* loweredMap */ nullptr, this);

  // Optimize the graph again.
  ::glow::optimize(F, mode);

  // Allow the backend to transform the graph after lowering.
  if (transformPostLowering(F, mode)) {
    // Optimize the graph again after the backend transformation.
    // In particular, DCE is very likely to be useful.
    ::glow::optimize(F, mode);
  }
}

TraceInfo Backend::buildManualTraceInfo(Function *F) const {
  TraceInfo info(false, getTraceEventDataSize());
  const auto &nodes = F->getNodes();
  for (const auto &node : nodes) {
    if (const TraceEventNode *TEN = llvm::dyn_cast<TraceEventNode>(&node)) {
      Placeholder *backing =
          llvm::dyn_cast<Placeholder>(TEN->getData().getNode());
      assert(backing);
      info.add(backing, TEN->getIndex(), TEN->getEventName(),
               TEN->getEventType());
      info.enabled = true;
    }
  }

  return info;
}

void Backend::autoInstrument(TraceInfo &traceInfo, IRFunction *IR) const {
  if (getTraceEventDataSize() == 0) {
    GLOW_UNREACHABLE("Auto instrumentation not supported on this backend");
    return;
  }

  Function *F = IR->getGraph();
  // Get all instructions in the IRFunction.
  IRFunction::InstListTy &instructions = IR->getInstrs();

  // Build a placeholder to a backing tensor with space to fit all timestamps.
  auto *backingPH = F->getParent()->createPlaceholder(
      ElemKind::Int64ITy,
      {instructions.size() + 1,
       getTraceEventDataSize() / Type::getElementSize(ElemKind::Int64ITy)},
      F->getName().str() + "_instrumentation", false);

  // Create an associated weight and add it to the IR.
  auto *backingWeight =
      new WeightVar(IR->uniqueName(backingPH->getName()), backingPH->getType(),
                    WeightVar::MutabilityKind::Mutable);
  IR->getWeights().push_back(backingWeight);
  IR->getVariableMap()[backingPH] = backingWeight;

  traceInfo.enabled = true;
  std::string lastName = "";
  size_t index = 0;

  // For each instruction, insert a TraceEventInst to record the timestamp, and
  // two TraceInfo Events for the end of the previous Instruction and the start
  // of the next.
  auto it = instructions.begin();
  while (it != instructions.end()) {
    auto &I = *it;
    if (llvm::isa<TraceEventInst>(&I)) {
      // Don't instrument instrumentation.
      it++;
      continue;
    }

    auto name = I.getName();
    // End the previous event.
    if (lastName != "") {
      traceInfo.add(backingPH, index, lastName, "E");
    }

    // Start a new event.
    traceInfo.add(backingPH, index, name, "B");
    lastName = name;

    it = instructions.insert(
        it, new TraceEventInst(name.str() + "_trace", backingWeight, index++));

    // Skip over both I and the new TraceEvent.
    it++;
    it++;
  }

  // Add one more for the end of the function.
  traceInfo.add(backingPH, index, lastName, "E");
  IR->pushInstr(new TraceEventInst("end_trace", backingWeight, index));
}
