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
#include "glow/Optimizer/GraphOptimizerPipeline/Pipeline.h"

#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Optimizer/GraphOptimizer/PassManager.h"

using namespace glow;

FunctionPassPipeline glow::createDefaultGraphOptimizationPassPipeline() {
  return {
      // Sink transpose operations in an attempt to cancel them out.
      // Perform code sinking until a fixed-point is reached.
      // On big functions, the number of iterations until the fixpoint
      // is usually at most 2 or 3 iterations.
      {FunctionPassID::SinkCode, ConvergenceMode::UntilFixedPoint},

      // Transposes that don't move data are optimized into Reshapes, which
      // enables further optimizations.
      {FunctionPassID::OptimizeTransposeIntoReshape},

      // Reshapes and transposes can prevent other optimizations from
      // triggering,
      // so try to optimize them out first.
      {FunctionPassID::OptimizeReshape},

      // Eliminate no-op tiles, possibly unlocking more optimization
      // opportunities.
      {FunctionPassID::EliminateNoopTile},

      // Eliminate no-op slices, possibly unlocking more optimization
      // opportunities.
      {FunctionPassID::EliminateNoopSlice},

      {FunctionPassID::TransposeConstants,
       ConvergenceMode::OnePass,
       {CompilationMode::Infer}},

      // Perform Common Subexpression Elimination.
      {FunctionPassID::CSE},

      // Optimize Pad nodes
      {FunctionPassID::MergePadIntoConvolution},

      // Merge multiple matmul nodes into a single large matmul.
      {FunctionPassID::MergeMatMul},

      // Merge multiple batched adds into a larger batched add.
      {FunctionPassID::MergeBatchedAdd},

      // Merge ReduceMean into AveragePool if possible.
      {FunctionPassID::OptimizeReduceMean},

      // Convert BatchMatMuls with a broadcasted RHS to a single MatMul.
      {FunctionPassID::ConvertBroadcastedBatchMatMul},

      // Merge batch normalization operations.
      // Do after transpose constant folding, as weight transposes can prevent
      // the optimization from triggering.
      {FunctionPassID::OptimizeBatchNorm,
       ConvergenceMode::OnePass,
       {CompilationMode::Infer}},

      // Perform Common Subexpression Elimination.
      {FunctionPassID::CSE},

      // Optimize Concat nodes.
      {FunctionPassID::OptimizeConcatNodes},

      // Optimize arithmetic nodes based on algebraic identities.
      {FunctionPassID::OptimizeArithmeticNodes},

      // Optimize Splat nodes..
      {FunctionPassID::OptimizeSplat},

      // Merge Transpose into MatMul/FC.
      {FunctionPassID::MergeTransposeIntoMatMulOrFC},

      // Optimize away intermediate type conversions.
      {FunctionPassID::OptimizeConversions},

      // Optimize away intermediate consecutive Clips.
      {FunctionPassID::OptimizeClips},

      // Optimize quantization related operators.
      {FunctionPassID::OptimizeQuantization, ConvergenceMode::UntilFixedPoint},

      // Optimize reshapes introduced during above optimizations.
      {FunctionPassID::OptimizeReshape},

      // Run a round of constant folding.
      {FunctionPassID::ConstantFold},

      // Perform a round of Dead Code Elimination to cleanup the final pass.
      getDCEPassConfig(),
  };
}

FunctionPassPipeline glow::createFP16GraphOptimizationPassPipeline() {
  return {
      // Optimize away intermediate type conversions.
      {FunctionPassID::OptimizeConversions},

      // Optimize away intermediate consecutive Clips.
      {FunctionPassID::OptimizeClips},
  };
}

FunctionPassPipeline glow::createDefaultFoldPassPipeline() {
  return {
      // Get Reshape nodes merged into constants to simplify folding.
      {FunctionPassID::OptimizeReshape},

      // Fold sub-graphs corresponding to leakyRelu.
      {FunctionPassID::FoldLeakyRelu},

      // Fold Reshape->Transpose->Reshape into ChannelShuffle when applicable.
      {FunctionPassID::FoldChannelShuffle},

      // Perform Dead Code Elimination.
      getDCEPassConfig(),
  };
}

FunctionPassConfig glow::getDCEPassConfig() {
  return {FunctionPassID::DCE,
          ConvergenceMode::OnePass,
          {CompilationMode::Infer, CompilationMode::Train},
          DCERequiredMode::None};
}

llvm::StringRef glow::getNameOfPass(FunctionPassID passID) {
  switch (passID) {
#define FUN_PASS(PASS_NAME)                                                    \
  case FunctionPassID::PASS_NAME:                                              \
    return #PASS_NAME;
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.def"
  }
  LOG(DFATAL) << "Cannot reach here.";
}

static constexpr char const *tab = "  ";

void FunctionPassConfig::dump(llvm::raw_ostream &os) const {
  os << tab << "PassName: " << getNameOfPass(getFunctionPassID()) << ",\n";

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

  os << tab << "DCERequiredMode: ";
  switch (getDCERequiredMode()) {
  case DCERequiredMode::BeforePass:
    os << "BeforePass,";
    break;
  case DCERequiredMode::None:
    os << "None,";
    break;
  }
  os << "\n";
}

void FunctionPassPipeline::dump(llvm::raw_ostream &os) const {
  os << "Pipeline contains:\n";
  for (size_t i = 0, e = this->size(); i < e; i++) {
    const FunctionPassConfig &passConfig = (*this)[i];
    os << "FunctionPassIdx " << i << ": {\n";
    passConfig.dump(os);
    os << "}\n";
  }
}

bool FunctionPassPipeline::removeFirstInstanceOfPass(FunctionPassID FPID) {
  for (auto it = begin(); it != end(); it++) {
    if (it->getFunctionPassID() == FPID) {
      erase(it);
      return true;
    }
  }
  return false;
}

void FunctionPassPipeline::removeAllInstancesOfPass(FunctionPassID FPID) {
  while (removeFirstInstanceOfPass(FPID)) {
  }
}
