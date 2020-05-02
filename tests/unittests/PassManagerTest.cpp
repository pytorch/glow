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
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPassManager.h"
#include "glow/Optimizer/IROptimizer/IRFunctionPassManager.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

/// Test the ability to construct IRFunctionPassPipelines.
TEST(Optimizer, IRFunctionPassPipeline) {
  Module mod;
  Function *F = mod.createFunction("IRPassManagerTest");
  IRFunction M(F);
  IRBuilder bb(&M);

  // Create a WeightVar for TensorViews to use as their source operand.
  auto *A = bb.createWeightVar(glow::ElemKind::FloatTy, {4, 2}, "A",
                               WeightVar::MutabilityKind::Mutable);

  // Create a view into A.
  bb.createTensorViewInst(
      "view1", A, mod.uniqueType(Type(glow::ElemKind::FloatTy, {1, 2, 1})),
      {0, 0});

  IRFunctionPassManager IRFPM("opt",
                              glow::make_unique<IRFunctionPassPipeline>(
                                  std::initializer_list<IRFunctionPassConfig>(
                                      {{IRFunctionPassID::DSE}})));
  IRFPM.run(&M, CompilationContext());
  ASSERT_TRUE(M.verify());
}

/// Test the ability to construct FunctionPassPipelines.
TEST(Optimizer, FunctionPassPipeline) {
  Module mod;
  Function *F = mod.createFunction("PassManagerTest");

  FunctionPassManager FPM("opt", glow::make_unique<FunctionPassPipeline>(
                                     std::initializer_list<FunctionPassConfig>(
                                         {{FunctionPassID::CSE}})));
  FPM.run(F, CompilationContext());
  ASSERT_TRUE(F->verify());
}

/// Test that IRFunctionPassPipeline can be stored into files and can be read
/// from files.
TEST(Optimizer, IRFunctionPassPipelineFromFile) {
  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile(
      "ir_function_pass_pipeline", "def", path);
  EXPECT_EQ(tempFileRes.value(), 0);

  IRFunctionPassPipeline origPipeline{
      {IRFunctionPassID::DSE, ConvergenceMode::OnePass,
       std::set{CompilationMode::Train}},

      {IRFunctionPassID::OptimizeInserts, ConvergenceMode::UntilFixedPoint,
       std::set{CompilationMode::Infer}},

      {IRFunctionPassID::OptimizeExtracts, ConvergenceMode::OnePass,
       std::set{CompilationMode::Infer, CompilationMode::Train}}};

  /// Store the pipeline into a file.
  origPipeline.dumpToFile(path);

  /// Constuct a pipeline by reading its definition from a file.
  IRFunctionPassManager IRFPM("irfpm", path);

  /// Pipeline read from the file should be equivalent to the original pipeline.
  EXPECT_TRUE(origPipeline.equals(IRFPM.getPipeline()));
}

/// Test that FunctionPassPipeline can be stored into files and can be read
/// from files.
TEST(Optimizer, FunctionPassPipelineFromFile) {
  llvm::SmallString<64> path;
  auto tempFileRes =
      llvm::sys::fs::createTemporaryFile("function_pass_pipeline", "def", path);
  EXPECT_EQ(tempFileRes.value(), 0);

  FunctionPassPipeline origPipeline{
      {FunctionPassID::CSE, ConvergenceMode::OnePass,
       std::set{CompilationMode::Infer}, DCERequiredMode::BeforePass},

      {FunctionPassID::SinkCode, ConvergenceMode::UntilFixedPoint,
       std::set{CompilationMode::Infer, CompilationMode::Train},
       DCERequiredMode::BeforePass},

      {FunctionPassID::OptimizeReshape, ConvergenceMode::OnePass,
       std::set{CompilationMode::Infer}, DCERequiredMode::None}};

  /// Store the pipeline into a file.
  origPipeline.dumpToFile(path);

  /// Constuct a pipeline by reading its definition from a file.
  FunctionPassManager FPM("fpm", path);

  /// Pipeline read from the file should be equivalent to the original pipeline.
  EXPECT_TRUE(origPipeline.equals(FPM.getPipeline()));
}
