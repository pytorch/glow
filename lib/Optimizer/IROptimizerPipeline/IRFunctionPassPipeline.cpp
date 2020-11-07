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

#include "glow/Optimizer/IROptimizer/CommandLine.h"
#include "glow/Optimizer/IROptimizer/IRFunctionPassManager.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"
#include "glow/PassManager/PassConfigUtils.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

#include <fstream>

namespace glow {
llvm::StringRef getNameOfPass(IRFunctionPassID passID) {
  switch (passID) {
#define IR_FUN_PASS(PASS_NAME)                                                 \
  case IRFunctionPassID::PASS_NAME:                                            \
    return #PASS_NAME;
#include "glow/Optimizer/IROptimizer/IRPasses.def"
  }
  llvm_unreachable("Unexpected pass.");
}

llvm::StringRef IRFunctionPassConfig::getNameOfPass() const {
  return glow::getNameOfPass(getPassID());
}
} // namespace glow

namespace {
/// A helper class to represent a IRFunctionPassConfig in a way which can be
/// easily handled by YAML functions.
struct IRPassConfigHelper {
  std::string passName;
  glow::ConvergenceMode convergenceMode;
  CompilationModes enabledCompilationModes;
  IRPassConfigHelper(const glow::IRFunctionPassConfig &config)
      : passName(config.getNameOfPass()),
        convergenceMode(config.getConvergenceMode()),
        enabledCompilationModes(config.getEnabledCompilationModes()) {}
  IRPassConfigHelper() = default;
};
} // namespace

namespace llvm {
namespace yaml {
/// Define the YAML mapping for IRPassConfigHelper.
template <> struct MappingTraits<IRPassConfigHelper> {
  static void mapping(IO &io, IRPassConfigHelper &config) {
    io.mapRequired("passName", config.passName);
    io.mapRequired("convergenceMode", config.convergenceMode);
    io.mapRequired("enabledCompilaitonModes", config.enabledCompilationModes);
  }
};
} // namespace yaml
} // namespace llvm

namespace glow {

std::unique_ptr<IRFunctionPassPipeline>
createDefaultIRFunctionOptimizationPipeline() {
  if (!optimizeIR) {
    return glow::make_unique<IRFunctionPassPipeline>();
  }
  std::initializer_list<IRFunctionPassConfig> configs{
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

  auto pipeline = glow::make_unique<IRFunctionPassPipeline>(configs);

  if (instrumentDebug) {
    // Run debug instrumentation only if necessary.
    pipeline->pushBack(IRFunctionPassID::DebugInstrument);
  }
  if (instrumentIR) {
    // Run IR instrumentation only if necessary.
    pipeline->pushBack(IRFunctionPassID::IRInstrument);
  }
  // Always run a verifier at the end.
  pipeline->pushBack(IRFunctionPassID::IRVerify);
  // If requested, dump IR to stdout for debugging.
  if (dumpIR) {
    pipeline->pushBack(IRFunctionPassID::IRDumper);
  }
  return pipeline;
}

#define IR_FUN_PASS(PASS_NAME) {#PASS_NAME, IRFunctionPassID::PASS_NAME},

static llvm::StringMap<IRFunctionPassID> passNameToID{
#include "glow/Optimizer/IROptimizer/IRPasses.def"
};

static IRFunctionPassID getPassID(llvm::StringRef name) {
  CHECK_GT(passNameToID.count(name), 0) << "Unknown pass name: " << name.str();
  return passNameToID.lookup(name);
}

template <>
void IRFunctionPassPipeline::initFromFile(llvm::StringRef pipelineDefFilename) {
  clear();
  auto configs =
      deserializeFromYaml<std::vector<IRPassConfigHelper>>(pipelineDefFilename);
  for (auto &config : configs) {
    IRFunctionPassConfig irFunctionPassConfig(getPassID(config.passName),
                                              config.convergenceMode,
                                              config.enabledCompilationModes);
    pushBack(irFunctionPassConfig);
  }
}

template <>
void IRFunctionPassPipeline::dumpToFile(llvm::StringRef pipelineDefFilename) {
  std::vector<IRPassConfigHelper> configs;
  for (unsigned idx = 0, e = size(); idx < e; ++idx) {
    const auto &config = at(idx);
    configs.emplace_back(IRPassConfigHelper(config));
  }
  serializeToYaml(pipelineDefFilename, configs);
}

void IRFunctionPassConfig::dump(llvm::raw_ostream &os) const {
  PassConfigBase::dump(os, glow::getNameOfPass(getPassID()));
}

} // namespace glow
