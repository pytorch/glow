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

#include "gtest/gtest.h"

using namespace glow;

/// Test BundleSaver. This unit test is for code coverage.
TEST(BundleSaver, testAPI) {
  Module M;
  Function *F = M.createFunction("F");

  // Create a simple graph.
  Node *inputPH = M.createPlaceholder(ElemKind::FloatTy, {1}, "input", false);
  Node *addConst = M.createConstant(ElemKind::FloatTy, {1}, "const");
  Node *addNode = F->createAdd("add", inputPH, addConst);
  F->createSave("output", addNode);

  // Save bundle.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "testBundle";
  llvm::StringRef mainEntryName = "testMainEntry";
  std::unique_ptr<Backend> backend(createBackend("CPU"));
  backend->save(F, outputDir, bundleName, mainEntryName);
}

TEST(BundleSaver, testSaveMultipleFunctions) {
  Module M;

  Node *addConst = M.createConstant(ElemKind::FloatTy, {1}, "const");
  // Create a simple graph.
  Function *F1 = M.createFunction("F1");
  Node *inputPH1 = M.createPlaceholder(ElemKind::FloatTy, {1}, "input1", false);
  Node *addConst1 = M.createConstant(ElemKind::FloatTy, {1}, "const1");
  Node *addNode11 = F1->createAdd("add", inputPH1, addConst);
  Node *addNode12 = F1->createAdd("add", addNode11, addConst1);
  F1->createSave("output", addNode12);

  // Create a simple graph.
  Function *F2 = M.createFunction("F2");
  Node *inputPH2 = M.createPlaceholder(ElemKind::FloatTy, {1}, "input2", false);
  Node *subConst2 = M.createConstant(ElemKind::FloatTy, {1}, "const2");
  Node *subNode21 = F2->createSub("sub", inputPH2, addConst);
  Node *subNode22 = F2->createSub("sub", subNode21, subConst2);
  F2->createSave("output", subNode22);

  // Save a bundle with multiple functions.
  llvm::StringRef outputDir = ".";
  llvm::StringRef bundleName = "testBundle";
  std::unique_ptr<Backend> backend(
      reinterpret_cast<Backend *>(createBackend("CPU")));
  std::vector<BundleEntry> bundleEntries;
  bundleEntries.emplace_back(BundleEntry{"testMainEntry1", F1});
  bundleEntries.emplace_back(BundleEntry{"testMainEntry2", F2});
  backend->saveFunctions(bundleEntries, outputDir, bundleName);
}
