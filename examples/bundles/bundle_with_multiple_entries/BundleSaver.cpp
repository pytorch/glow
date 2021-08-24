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
#include "glow/Graph/Graph.h"
#include "glow/Graph/Hook.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "llvm/Support/CommandLine.h"

using namespace glow;

/// Emit a bundle into the specified output directory.
llvm::cl::opt<std::string>
    emitBundle("emit-bundle",
               llvm::cl::desc("Output directory for the bundle serialization"),
               llvm::cl::init("."));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Bundle Saver\n");

  // Create a bundle with multiple entry points and save it. We use multiple
  // MatMul layers to force usage of activations memory pool for each function.
  Module M;
  constexpr dim_t tensorSize = 16;

  // Common constants.
  Constant *matMulConst1 = M.createConstant(
      ElemKind::FloatTy, {tensorSize, tensorSize}, "matmulconst1");
  matMulConst1->getHandle().clear(0.0625f * 1.0f);

  Constant *matMulConst2 = M.createConstant(
      ElemKind::FloatTy, {tensorSize, tensorSize}, "matmulconst2");
  matMulConst2->getHandle().clear(0.0625f * 2.0f);

  Constant *matMulConst3 = M.createConstant(
      ElemKind::FloatTy, {tensorSize, tensorSize}, "matmulconst3");
  matMulConst3->getHandle().clear(0.0625f * 3.0f);

  // Create a simple graph with 2 MatMul layers.
  Function *F1 = M.createFunction("F1");
  auto *inputPH1 =
      M.createPlaceholder(ElemKind::FloatTy, {tensorSize}, "input1", false);
  auto *reshapeNode11 =
      F1->createReshape("reshape11", inputPH1, {1, tensorSize});
  auto *matMulNode11 =
      F1->createMatMul("matmul11", reshapeNode11, matMulConst1);
  auto *matMulNode12 = F1->createMatMul("matmul12", matMulNode11, matMulConst2);
  auto *reshapeNode12 =
      F1->createReshape("reshape12", matMulNode12, {tensorSize});
  auto *save1 = F1->createSave("output1", reshapeNode12);

  // Create a simple graph with 3 MatMul layers.
  Function *F2 = M.createFunction("F2");
  auto *inputPH2 =
      M.createPlaceholder(ElemKind::FloatTy, {tensorSize}, "input2", false);
  auto *reshapeNode21 =
      F2->createReshape("reshape21", inputPH2, {1, tensorSize});
  auto *matMulNode21 =
      F2->createMatMul("matmul21", reshapeNode21, matMulConst1);
  auto *matMulNode22 = F2->createMatMul("matmul22", matMulNode21, matMulConst2);
  auto *matMulNode23 = F2->createMatMul("matmul23", matMulNode22, matMulConst3);
  auto *reshapeNode22 =
      F2->createReshape("reshape22", matMulNode23, {tensorSize});
  auto *save2 = F2->createSave("output2", reshapeNode22);

  // Save a bundle with multiple functions.
  llvm::StringRef outputDir = emitBundle;
  llvm::StringRef bundleName = "testBundle";
  std::unique_ptr<Backend> cpuBackend(
      reinterpret_cast<Backend *>(createBackend("CPU")));
  std::vector<BundleEntry> bundleEntries;
  bundleEntries.emplace_back(BundleEntry{"testMainEntry1", F1});
  bundleEntries.emplace_back(BundleEntry{"testMainEntry2", F2});
  cpuBackend->saveFunctions(bundleEntries, outputDir, bundleName);

  // Check that each entry point produces correct results.
  std::unique_ptr<Backend> backend(
      reinterpret_cast<Backend *>(createBackend("Interpreter")));
  ExecutionContext ctx1;
  ctx1.getPlaceholderBindings()->allocate(inputPH1);
  ctx1.getPlaceholderBindings()->allocate(save1->getPlaceholder());
  ctx1.getPlaceholderBindings()->get(inputPH1)->getHandle().clear(1.0);
  EXIT_ON_ERR(EXIT_ON_ERR(backend->compile(F1))->execute(&ctx1));
  Tensor expected1(ElemKind::FloatTy, {tensorSize});
  expected1.getHandle().clear(2.0f);
  DCHECK(ctx1.getPlaceholderBindings()
             ->get(save1->getPlaceholder())
             ->isEqual(expected1))
      << "Wrong output1";

  ExecutionContext ctx2;
  ctx2.getPlaceholderBindings()->allocate(inputPH2);
  ctx2.getPlaceholderBindings()->allocate(save2->getPlaceholder());
  ctx2.getPlaceholderBindings()->get(inputPH2)->getHandle().clear(1.0);
  EXIT_ON_ERR(EXIT_ON_ERR(backend->compile(F2))->execute(&ctx2));
  Tensor expected2(ElemKind::FloatTy, {tensorSize});
  expected2.getHandle().clear(6.0f);
  DCHECK(ctx2.getPlaceholderBindings()
             ->get(save2->getPlaceholder())
             ->isEqual(expected2))
      << "Wrong output2";
  return 0;
}
