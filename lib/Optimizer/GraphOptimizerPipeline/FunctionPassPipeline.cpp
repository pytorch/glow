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

#include "glow/Optimizer/GraphOptimizer/FunctionPassManager.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/PassManager/PassConfigUtils.h"
#include "glow/Support/Memory.h"

#include <fstream>

namespace {
/// A helper class to represent a FunctionPassConfig in a way which can be
/// easily handled by YAML functions.
struct FunctionPassConfigHelper {
  std::string passName;
  glow::ConvergenceMode convergenceMode;
  CompilationModes enabledCompilationModes;
  glow::DCERequiredMode dceMode;
  FunctionPassConfigHelper(const glow::FunctionPassConfig &config)
      : passName(config.getNameOfPass()),
        convergenceMode(config.getConvergenceMode()),
        enabledCompilationModes(config.getEnabledCompilationModes()),
        dceMode(config.getDCERequiredMode()) {}
  FunctionPassConfigHelper() = default;
};
} // namespace

namespace llvm {
namespace yaml {
/// Define the YAML mapping for PassConfigHelper.
template <> struct MappingTraits<FunctionPassConfigHelper> {
  static void mapping(IO &io, FunctionPassConfigHelper &config) {
    io.mapRequired("passName", config.passName);
    io.mapRequired("convergenceMode", config.convergenceMode);
    io.mapRequired("enabledCompilationModes", config.enabledCompilationModes);
    io.mapRequired("dceMode", config.dceMode);
  }
};
} // namespace yaml
} // namespace llvm

namespace glow {

std::unique_ptr<FunctionPassPipeline>
createDefaultGraphOptimizationPassPipeline() {
  std::initializer_list<FunctionPassConfig> configs{
      // Eliminate nodes which do not do anything. Do it as early as
      // possible to prevent such nodes from affecting other optimizations.
      {FunctionPassID::EliminateNoop},

      // Sink transpose operations in an attempt to cancel them out.
      // Perform code sinking until a fixed-point is reached.
      // On big functions, the number of iterations until the fixpoint
      // is usually at most 2 or 3 iterations.
      {FunctionPassID::SinkCode, ConvergenceMode::UntilFixedPoint},

      // ConvTranspose + BiasAdd
      {FunctionPassID::ConvTransposeBiasAddFold},

      // Transposes that don't move data are optimized into Reshapes, which
      // enables further optimizations.
      {FunctionPassID::OptimizeTransposeIntoReshape},

      // Optimize arithmetic nodes based on algebraic identities.
      {FunctionPassID::OptimizeArithmeticNodes},

      // Fold some Arithmetic ops following a LayerNorm into LayerNorm.
      {FunctionPassID::FoldLayerNormArithmetic},

      // Reshapes and transposes can prevent other optimizations from
      // triggering,
      // so try to optimize them out first.
      {FunctionPassID::OptimizeReshape},

      {FunctionPassID::TransposeConstants,
       ConvergenceMode::OnePass,
       {CompilationMode::Infer}},

      // Perform Common Subexpression Elimination.
      {FunctionPassID::CSE},

      // Optimize Pad nodes
      {FunctionPassID::MergePadIntoConvolution},

      // Optimize Convolution nodes with small input tensors.
      {FunctionPassID::OptimizeSmallConv},

      // Merge multiple matmul nodes into a single large matmul.
      {FunctionPassID::MergeMatMulOnLHS},
      {FunctionPassID::MergeMatMulOnRHS},
      // Merge multiple batched adds into a larger batched add.
      {FunctionPassID::MergeBatchedAdd},

      // Merge ReduceMean into AveragePool if possible.
      {FunctionPassID::OptimizeReduceMean},

      // Optimize Resize nodes.
      {FunctionPassID::OptimizeResize},

      // Optimize Insert nodes.
      {FunctionPassID::OptimizeInsert},

      // Convert BatchMatMuls with a broadcasted RHS to a single MatMul.
      {FunctionPassID::ConvertBroadcastedBatchMatMul},

      // Eliminate nodes which do not do anything.
      {FunctionPassID::EliminateNoop},

      // Perform Common Subexpression Elimination.
      {FunctionPassID::CSE},

      // Optimize arithmetic nodes based on algebraic identities.
      {FunctionPassID::OptimizeArithmeticNodes},

      // Optimize Splat nodes.
      {FunctionPassID::OptimizeSplat},

      // Optimize Concat nodes.
      {FunctionPassID::OptimizeConcatNodes},

      // Eliminate Concat-Slice patterns which are unnecessary.
      {FunctionPassID::EliminateConcatSlice},

      // Merge Transpose into MatMul/FC.
      {FunctionPassID::MergeTransposeIntoMatMulOrFC},

      // Optimize away intermediate type conversions.
      {FunctionPassID::OptimizeConversions},

      // Eliminate clips outside the FP16 range. This is a specialized pass that
      // is disabled by default.
      {FunctionPassID::EliminateClipsOutsideFP16Range},

      // Look for float Relus that we can fuse up into quantized FCs.
      {FunctionPassID::OptimizeQuantFCFloatRelu},

      // Eliminate clips outside the FP16 range. This is a specialized pass that
      // is disabled by default.
      {FunctionPassID::EliminateClipsOutsideFP16Range},

      // Optimize away intermediate consecutive Clips.
      {FunctionPassID::OptimizeClips},

      // Optimize quantization related operators.
      {FunctionPassID::OptimizeQuantization, ConvergenceMode::UntilFixedPoint},

      // Optimize patterns of concats with quantization/dequantization.
      {FunctionPassID::OptimizeConcatQuantization},

      // Optimize reshapes introduced during above optimizations.
      {FunctionPassID::OptimizeReshape},

      // Run a round of constant folding.
      {FunctionPassID::ConstantFold},

      // Optimize Gather with const scalar index to Slice.
      {FunctionPassID::GatherToSlice},

      // Optimize combinations of Quantized Nodes and Clips.
      {FunctionPassID::OptimizeQuantizeClip},

      // Remove identity Relu and Clip nodes.
      {FunctionPassID::RemoveIdentityRelu},
      {FunctionPassID::RemoveIdentityClip},

      // Fold a Convolution dilated manually using Transpose, SpaceToDepth and
      // DepthToSpace nodes into a single Convolution node.
      // Run Reshape/Transpose optimizations afterwards to clean up the graph.
      {FunctionPassID::FoldDilatedConv},
      {FunctionPassID::OptimizeReshape},
      {FunctionPassID::SinkCode, ConvergenceMode::UntilFixedPoint},

      // Fold Arithmetic chain w/ constants into Batch Norm, when Conv preceeds.
      {FunctionPassID::FoldArithmeticChainUnderConvIntoBN,
       ConvergenceMode::OnePass,
       {CompilationMode::Infer}},

      // Fold Arithmetic chain w/ constants into the preceding Batch Norm.
      {FunctionPassID::FoldBatchNormalizationWithArithmeticChain,
       ConvergenceMode::OnePass,
       {CompilationMode::Infer}},

      // Merge batch normalization operations.
      // Do after transpose constant folding, as weight transposes can prevent
      // the optimization from triggering.
      {FunctionPassID::OptimizeBatchNorm,
       ConvergenceMode::UntilFixedPoint,
       {CompilationMode::Infer}},

      // Try to remove unnecessary Split-Concat operations
      {FunctionPassID::EliminateSliceConcat},

      // Perform Common Subexpression Elimination.
      {FunctionPassID::CSE},

      // Some sinking transformations are harmful for performance if a sunken
      // node does not get optimized out (e.g. sinking of Transpose below Tile).
      // Run code hoisting pass to undo such unsuccessful sinking.
      {FunctionPassID::HoistCode, ConvergenceMode::UntilFixedPoint},

      // Try to eliminate Reshape nodes by sinking them through the graph.
      // Such sinking can create new optimization opportunities as well as
      // prevent some optimizations from happening, so do it at the very end of
      // the pipeline to keep the current iteration unaffected and bear all
      // benefits/consequences on the next pipeline iteration.
      {FunctionPassID::SinkReshapes, ConvergenceMode::UntilFixedPoint},
      {FunctionPassID::OptimizeReshape},

      // Perform a round of Dead Code Elimination to cleanup the final pass.
      getDCEPassConfig(),
  };
  return glow::make_unique<FunctionPassPipeline>(configs);
}

std::unique_ptr<FunctionPassPipeline>
createFP16GraphOptimizationPassPipeline() {
  std::initializer_list<FunctionPassConfig> configs{
      // Optimize away intermediate type conversions.
      {FunctionPassID::OptimizeConversions},

      // Eliminate clips outside the FP16 range. This is a specialized pass that
      // is disabled by default.
      {FunctionPassID::EliminateClipsOutsideFP16Range},

      // Look for float Relus that we can fuse up into quantized FCs.
      {FunctionPassID::OptimizeQuantFCFloatRelu},

      // Eliminate clips outside the FP16 range. This is a specialized pass that
      // is disabled by default.
      {FunctionPassID::EliminateClipsOutsideFP16Range},

      // Optimize away intermediate consecutive Clips.
      {FunctionPassID::OptimizeClips},
  };
  return glow::make_unique<FunctionPassPipeline>(configs);
}

std::unique_ptr<FunctionPassPipeline> createDefaultFoldPassPipeline() {
  std::initializer_list<FunctionPassConfig> configs{
      // Optimize arithmetic nodes based on algebraic identities.
      // In this function, constant operators in communative nodes are moved to
      // the RHS. Some folding functions depend on this. (e.g. FoldMinMaxToClip)
      {FunctionPassID::OptimizeArithmeticNodes},

      // Get Reshape nodes merged into constants to simplify folding.
      {FunctionPassID::OptimizeReshape},

      // Fold sub-graphs corresponding to leakyRelu.
      {FunctionPassID::FoldLeakyRelu},

      // Fold Reshape->Transpose->Reshape into ChannelShuffle when applicable.
      {FunctionPassID::FoldChannelShuffle},

      // Fold MatMul->Add into FullyConnected.
      {FunctionPassID::FoldMatMulAddIntoFullyConnected},

      // Fold Min + Max to Clip
      {FunctionPassID::FoldMinMaxToClip},

      // Fold exp + reduce sum + div into softmax
      {FunctionPassID::FoldExpSumDivIntoSoftmax},

      // Perform Dead Code Elimination.
      getDCEPassConfig(),
  };
  return glow::make_unique<FunctionPassPipeline>(configs);
}

FunctionPassConfig getDCEPassConfig() {
  return FunctionPassConfig(
      FunctionPassID::DCE, ConvergenceMode::OnePass,
      std::set<CompilationMode>{CompilationMode::Infer, CompilationMode::Train},
      DCERequiredMode::None);
}

llvm::StringRef getNameOfPass(FunctionPassID passID) {
  switch (passID) {
#define FUN_PASS(PASS_NAME)                                                    \
  case FunctionPassID::PASS_NAME:                                              \
    return #PASS_NAME;
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.def"
  }
  llvm_unreachable("Unexpected pass.");
}

llvm::StringRef FunctionPassConfig::getNameOfPass() const {
  return glow::getNameOfPass(getPassID());
}

#define FUN_PASS(PASS_NAME) {#PASS_NAME, FunctionPassID::PASS_NAME},

static llvm::StringMap<FunctionPassID> passNameToID{
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.def"
};

static FunctionPassID getPassID(llvm::StringRef name) {
  CHECK_GT(passNameToID.count(name), 0) << "Unknown pass name: " << name.str();
  return passNameToID.lookup(name);
}

template <>
void FunctionPassPipeline::initFromFile(llvm::StringRef pipelineDefFilename) {
  clear();
  auto configs = deserializeFromYaml<std::vector<FunctionPassConfigHelper>>(
      pipelineDefFilename);
  for (auto &config : configs) {
    FunctionPassConfig functionPassConfig(
        getPassID(config.passName), config.convergenceMode,
        config.enabledCompilationModes, config.dceMode);
    pushBack(functionPassConfig);
  }
}

template <>
void FunctionPassPipeline::dumpToFile(llvm::StringRef pipelineDefFilename) {
  std::vector<FunctionPassConfigHelper> configs;
  for (unsigned idx = 0, e = size(); idx < e; ++idx) {
    const auto &config = at(idx);
    configs.emplace_back(FunctionPassConfigHelper(config));
  }
  serializeToYaml(pipelineDefFilename, configs);
}

static constexpr char const *tab = "  ";

void FunctionPassConfig::dump(llvm::raw_ostream &os,
                              llvm::StringRef passName) const {
  PassConfigBase::dump(os, passName);

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

bool FunctionPassConfig::equals(const PassConfigBase &other) const {
  return dceMode_ == static_cast<const FunctionPassConfig &>(other).dceMode_ &&
         PassConfigBase::equals(other);
}

} // namespace glow
