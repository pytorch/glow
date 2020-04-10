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
    // Run debug instrumentation only if neccesary.
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

static constexpr char const *tab = "  ";

template <> void IRFunctionPassConfig::dump(llvm::raw_ostream &os) const {
  os << tab << "PassName: " << getNameOfPass(getPassID()) << ",\n";

  os << tab << "ConvergenceMode: ";
  switch (getConvergenceMode()) {
  case ConvergenceMode::OnePass:
    os << "OnePass,";
    break;
  case ConvergenceMode::UntilFixedPoint:
    os << "UntilFixedPoint,";
    break;
  }
  os << "\n";

  os << tab << "CompilationModes: {";
  if (isEnabledForCompilationMode(CompilationMode::Infer)) {
    os << "[Infer]";
  }
  if (isEnabledForCompilationMode(CompilationMode::Train)) {
    os << "[Train]";
  }
  os << "},\n";

  os << "\n";
}

template <> void IRFunctionPassPipeline::dump(llvm::raw_ostream &os) const {
  os << "Pipeline contains:\n";
  for (size_t i = 0, e = this->size(); i < e; i++) {
    const IRFunctionPassConfig &passConfig = (*this)[i];
    os << "FunctionPassIdx " << i << ": {\n";
    passConfig.dump(os);
    os << "}\n";
  }
}

template <>
bool IRFunctionPassPipeline::removeFirstInstanceOfPass(IRFunctionPassID FPID) {
  for (auto it = begin(); it != end(); it++) {
    if (it->getPassID() == FPID) {
      erase(it);
      return true;
    }
  }
  return false;
}

template <>
void IRFunctionPassPipeline::removeAllInstancesOfPass(IRFunctionPassID FPID) {
  while (removeFirstInstanceOfPass(FPID)) {
  }
}
