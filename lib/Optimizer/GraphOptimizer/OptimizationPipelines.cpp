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
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Optimizer/GraphOptimizer/PassManager.h"

using namespace glow;

void glow::addDefaultGraphOptimizationPasses(FunctionPassManager &PM) {
  // Optimize may be called after backend specific transformations and
  // some nodes may have become unused. It is a good idea to remove
  // them, before proceeding with any further optimizations.
  PM.addPass({FunctionPassID::DCE});

  // Sink transpose operations in an attempt to cancel them out.
  // Perform code sinking until a fixed-point is reached.
  // On big functions, the number of iterations until the fixpoint
  // is usually at most 2 or 3 iterations.
  PM.addPass({FunctionPassID::SinkCode,
              FunctionPassConfig::ConvergenceMode::UntilFixedPoint});

  // Transposes that don't move data are optimized into Reshapes, which
  // enables further optimizations.
  PM.addPass({FunctionPassID::OptimizeTransposeIntoReshape});

  // Need to remove old uses that would prohibit Reshape(Constant)
  // optimization.
  PM.addPass({FunctionPassID::DCE});

  // Reshapes and transposes can prevent other optimizations from
  // triggering,
  // so try to optimize them out first.
  PM.addPass({FunctionPassID::OptimizeReshape});

  PM.addPass({FunctionPassID::TransposeConstants,
              FunctionPassConfig::ConvergenceMode::OnePass,
              {CompilationMode::Infer}});

  // Perform Common Subexpression Elimination.
  PM.addPass({FunctionPassID::CSE});

  // Optimize Pad nodes
  PM.addPass({FunctionPassID::MergePadIntoConvolution});

  // Perform Dead Code Elimination.
  PM.addPass({FunctionPassID::DCE});

  // Merge multiple matmul nodes into a single large matmul.
  PM.addPass({FunctionPassID::MergeMatMul});

  // Merge multiple batched adds into a larger batched add.
  PM.addPass({FunctionPassID::MergeBatchedAdd});

  // Merge ReduceMean into AveragePool if possible.
  PM.addPass({FunctionPassID::OptimizeReduceMean});

  // Convert BatchMatMuls with a broadcasted RHS to a single MatMul.
  PM.addPass({FunctionPassID::ConvertBroadcastedBatchMatMul});

  // Perform Dead Code Elimination.
  PM.addPass({FunctionPassID::DCE});

  // Merge batch normalization operations.
  // Do after transpose constant folding, as weight transposes can prevent
  // the optimization from triggering.
  PM.addPass({FunctionPassID::OptimizeBatchNorm,
              FunctionPassConfig::ConvergenceMode::OnePass,
              {CompilationMode::Infer}});

  // Perform Common Subexpression Elimination.
  PM.addPass({FunctionPassID::CSE});

  // Optimize Concat nodes.
  PM.addPass({FunctionPassID::OptimizeConcatNodes});

  // Optimize arithmetic nodes based on algebraic identities.
  PM.addPass({FunctionPassID::OptimizeArithmeticNodes});

  // Optimize Tensor shape transformations.
  PM.addPass({FunctionPassID::OptimizeSliceOfSplat});

  // Merge Transpose into MatMul/FC.
  // Run DCE to ensure correct number of node users.
  PM.addPass({FunctionPassID::DCE});
  PM.addPass({FunctionPassID::MergeTransposeIntoMatMulOrFC});

  // Optimize away intermediate type conversions.
  PM.addPass({FunctionPassID::OptimizeConversions});

  // Optimize quantization related operators.
  PM.addPass({FunctionPassID::OptimizeQuantization,
              FunctionPassConfig::ConvergenceMode::UntilFixedPoint});

  // Optimize reshapes introduced during above optimizations.
  PM.addPass({FunctionPassID::OptimizeReshape});

  // Run a round of constant folding.
  PM.addPass({FunctionPassID::ConstantFold});

  // Perform Dead Code Elimination.
  PM.addPass({FunctionPassID::DCE});
}

void glow::addDefaultFoldPasses(FunctionPassManager &PM) {
  // Get Reshape nodes merged into constants to simplify folding.
  PM.addPass({FunctionPassID::OptimizeReshape});

  // Fold sub-graphs corresponding to leakyRelu.
  PM.addPass({FunctionPassID::FoldLeakyRelu});

  // Fold Reshape->Transpose->Reshape into ChannelShuffle when applicable.
  PM.addPass({FunctionPassID::FoldChannelShuffle});

  // Perform Dead Code Elimination.
  PM.addPass({FunctionPassID::DCE});
}

FunctionPassSet glow::getTargetDependentPassSet() {
  return FunctionPassSet({
      FunctionPassID::TransposeConstants,
      FunctionPassID::MergePadIntoConvolution,
      FunctionPassID::MergeMatMul,
      FunctionPassID::MergeBatchedAdd,
      FunctionPassID::MergeTransposeIntoMatMulOrFC,
      FunctionPassID::OptimizeConversions,
      FunctionPassID::OptimizeQuantization,
  });
}
