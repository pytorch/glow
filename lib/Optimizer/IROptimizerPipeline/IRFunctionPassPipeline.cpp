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
#include "glow/PassManager/Pipeline.h"

#include "glow/Optimizer/IROptimizer/IRFunctionPassManager.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"

#include "llvm/Support/CommandLine.h"

using namespace glow;

namespace {
static llvm::cl::opt<bool>
    instrumentDebug("instrument-debug",
                    llvm::cl::desc("Instrument the IR for debugging"),
                    llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<bool> optimizeIR("optimize-ir",
                                      llvm::cl::desc("Enable IR optimizations"),
                                      llvm::cl::init(true), llvm::cl::Hidden);

static llvm::cl::opt<bool> dumpIR("dump-ir",
                                  llvm::cl::desc("Prints IR to stdout"));
} // namespace

IRFunctionPassPipeline glow::createDefaultIRFunctionOptimizationPipeline() {
  if (!optimizeIR) {
    return {};
  }
  IRFunctionPassPipeline pipeline = {
      {IRFunctionPassID::PeepholeOptimizations},
      // Dead store elimination.
      {IRFunctionPassID::DSE},
      // Replace applicable InsertTensors and ExtractTensors with TensorViews.
      {IRFunctionPassID::OptimizeInserts},
      {IRFunctionPassID::OptimizeExtracts},
      // TODO: Only if enabled.
      {IRFunctionPassID::ShareBuffers},
      {IRFunctionPassID::PeepholeOptimizations},
      {IRFunctionPassID::HoistDealloc},
      {IRFunctionPassID::SinkAllocas},
      // Dead store elimination.
      {IRFunctionPassID::DSE},
      {IRFunctionPassID::DeleteDeadAllocs},
      {IRFunctionPassID::MakeWeightsConst},
  };
  if (instrumentDebug) {
    // Run debug instrumentation only if necessary.
    pipeline.pushBack({IRFunctionPassID::DebugInstrument});
  }
  // Always run a verifier at the end.
  pipeline.pushBack({IRFunctionPassID::IRVerify});
  // If requested, dump IR to stdout for debugging.
  if (dumpIR) {
    pipeline.pushBack({IRFunctionPassID::IRDumper});
  }
  return pipeline;
}

llvm::StringRef glow::getNameOfPass(IRFunctionPassID passID) {
  switch (passID) {
#define IR_FUN_PASS(PASS_NAME)                                                 \
  case IRFunctionPassID::PASS_NAME:                                            \
    return #PASS_NAME;
#include "glow/Optimizer/IROptimizer/IRPasses.def"
  }
  llvm_unreachable("Unexpected pass.");
}

template <> void IRFunctionPassConfig::dump(llvm::raw_ostream &os) const {
  PassConfigBase::dump(os, getNameOfPass(getPassID()));
}
