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

  // Create a bundle with multiple entry points and save it.
  Module M;
  constexpr dim_t tensorSize = 64;

  Constant *addConst =
      M.createConstant(ElemKind::FloatTy, {tensorSize}, "const");
  addConst->getHandle().clear(10.0f);
  // Create a simple graph.
  Function *F1 = M.createFunction("F1");
  auto *inputPH1 =
      M.createPlaceholder(ElemKind::FloatTy, {tensorSize}, "input1", false);
  auto *addConst1 = M.createConstant(ElemKind::FloatTy, {tensorSize}, "const1");
  addConst1->getHandle().clear(20.0f);
  Node *addNode11 = F1->createAdd("add", inputPH1, addConst);
  Node *addNode12 = F1->createAdd("add", addNode11, addConst1);
  auto *save1 = F1->createSave("output1", addNode12);

  // Create a simple graph.
  Function *F2 = M.createFunction("F2");
  auto *inputPH2 =
      M.createPlaceholder(ElemKind::FloatTy, {tensorSize}, "input2", false);
  auto *addConst2 = M.createConstant(ElemKind::FloatTy, {tensorSize}, "const2");
  addConst2->getHandle().clear(30.0f);
  Node *addNode21 = F2->createAdd("add", inputPH2, addConst);
  Node *addNode22 = F2->createSub("add", addNode21, addConst2);
  auto *save2 = F2->createSave("output2", addNode22);

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
  expected1.getHandle().clear(31.0f);
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
  expected2.getHandle().clear(-19.0f);
  DCHECK(ctx2.getPlaceholderBindings()
             ->get(save2->getPlaceholder())
             ->isEqual(expected2))
      << "Wrong output2";
  return 0;
}
