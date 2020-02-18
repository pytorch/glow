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

#include "glow/Backend/Backend.h"
#include "glow/Backends/DummyDeviceManager.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Graph/TensorLayout.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Optimizer/GraphOptimizerPipeline/Pipeline.h"

using namespace glow;

runtime::DeviceManager *
Backend::createDeviceManager(const runtime::DeviceConfig &deviceConfig) {
  LOG(ERROR) << "Warning: Creating a DummyDeviceManager.\n";
  return new runtime::DummyDeviceManager(deviceConfig);
}

TraceInfo Backend::buildManualTraceInfo(Function *F) const {
  TraceInfo info(false, getTraceEventDataSize());
  const auto &nodes = F->getNodes();
  for (const auto &node : nodes) {
    if (const TraceEventNode *TEN = llvm::dyn_cast<TraceEventNode>(&node)) {
      Placeholder *backing =
          llvm::dyn_cast<Placeholder>(TEN->getData().getNode());
      assert(backing);
      char type = TraceEvent::InstantType;
      if (!TEN->getEventType().empty()) {
        type = TEN->getEventType()[0];
      }
      info.add(backing, TEN->getIndex(), TEN->getEventName(), type);
      info.enabled = true;
    }
  }

  return info;
}

void Backend::autoInstrument(TraceInfo &traceInfo, IRFunction *IR) const {
  if (getTraceEventDataSize() == 0) {
    LOG(ERROR) << "Auto instrumentation not supported on this backend";
    return;
  }

  Function *F = IR->getGraph();
  // Get all instructions in the IRFunction.
  IRFunction::InstListTy &instructions = IR->getInstrs();

  // First pass, find out how many TraceEvents we should add. Existing
  // TraceEvents have their own backing Tensors, so don't count them.
  dim_t numEvents = 1; // Starts at 1 since there is always a start event.
  for (auto it = instructions.begin(); it != instructions.end(); it++) {
    auto &I = *it;
    bool isInstrumentation = llvm::isa<TraceEventInst>(&I);
    if (!isInstrumentation) {
      numEvents++;
    }
  }

  // Default name for the instrumentation placeholder, will be made unique by
  // createPlaceholder.
  std::string name = F->getName().str() + "_instrumentation";
  Placeholder *backingPH = F->getParent()->getPlaceholderByName(name);

  auto &varmap = IR->getVariableMap();
  auto type = F->getParent()->uniqueType(
      ElemKind::Int64ITy,
      {numEvents, (dim_t)getTraceEventDataSize() /
                      Type::getElementSize(ElemKind::Int64ITy)});

  WeightVar *backingWeight = nullptr;

  if (backingPH) {
    // If the standard instrumentation placeholder exists, we might be able to
    // reuse it.
    auto it = varmap.find(backingPH);
    if (it != varmap.end() && backingPH->getType()->isEqual(type)) {
      // We have a weight for it already and the types match, can reuse it.
      backingWeight = llvm::cast<WeightVar>(it->second);
    } else {
      // This isn't ideal, the placeholder exists but we have no weight.
      // Probably indicates a bug in the graph, best we can do is create a new
      // placeholder and weight for the instrumentation.
      // assert(!"could not find weight for existing instrumentation
      // placeholder");
      backingPH = nullptr;
    }
  }

  // If we don't have a Placeholder, then we need to create one.
  if (!backingPH) {
    // Build a Placeholder to a backing tensor with space to fit all
    // timestamps.
    backingPH = F->getParent()->createPlaceholder(type, name,
                                                  /* isTrainable */ false);
    assert(backingPH);
  }

  // Add Placeholder to the graph so we can add it to the runtimeBundle later.
  F->addMetadataPlaceholder(backingPH);

  // If we don't have a weight we need to create one too, whether or not we
  // just created a Placeholder.
  if (!backingWeight) {
    // Create an associated weight and add it to the IR.
    backingWeight =
        new WeightVar(IR->uniqueName(backingPH->getName()),
                      backingPH->getType(), WeightVar::MutabilityKind::Mutable);
    IR->getWeights().push_back(backingWeight);
    IR->getVariableMap()[backingPH] = backingWeight;
  }

  traceInfo.enabled = true;
  traceInfo.autoInstrumented = true;
  size_t index = 0;

  // For each instruction, insert a TraceEventInst to record the timestamp,
  // and two TraceInfo Events for the end of the previous Instruction and the
  // start of the next.
  auto it = instructions.begin();
  while (it != instructions.end()) {
    auto &I = *it;
    if (llvm::isa<TraceEventInst>(&I)) {
      // Don't instrument instrumentation.
      it++;
      continue;
    }

    auto instName = I.getName();

    // Start a new event
    traceInfo.add(backingPH, index, index + 1, instName, std::string(),
                  Kinded::getKindName(I.getKind()));

    it = instructions.insert(it, new TraceEventInst(instName.str() + "_trace",
                                                    backingWeight, index++));

    // Skip over both I and the new TraceEvent.
    it++;
    it++;
  }

  IR->pushInstr(new TraceEventInst("end_trace", backingWeight, index));
}

bool Backend::checkAllNodesSupported(const Function &F, bool verbose) const {
  bool allSupported = true;
  for (const Node &N : F.getNodes()) {
    if (!isOpSupported(N)) {
      allSupported = false;
      if (verbose) {
        report("Unsupported node found while compiling Function " +
               F.getName().str() + " for backend " + getBackendName() + ": " +
               N.getDebugDesc());
      }
    }
  }
  return allSupported;
}

bool Backend::verify(const Function &F, bool verbose) const {
  return F.verify(this) && checkAllNodesSupported(F, verbose);
}

bool Backend::verify(const IRFunction &IR) const {
  (void)IR;
  return true;
}

TensorLayoutCommon &Backend::getTensorLayoutRequirements() const {
  return CanonicalTensorLayout::getInstance();
}

FunctionPassPipeline Backend::getOptimizationPipeline() const {
  auto p = createDefaultGraphOptimizationPassPipeline();
  // Fold Tile followed by Add into BatchedAdd. Currently this is not part of
  // the default pipeline to avoid issues with some backends. If backends do not
  // want this opt then they should override getOptimizationPipeline().
  p.pushFront({FunctionPassID::FoldTileAddIntoBatchedAdd});
  return p;
};
