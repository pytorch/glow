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

#include "glow/Graph/Graph.h"
#include "BackendTestUtils.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Flags/Flags.h"
#include "glow/Graph/Hook.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FileSystem.h"

#include "gtest/gtest.h"

using namespace glow;

// Helper to find a node in the Function by name
static const Node *nodeByName(const Function *F, const std::string &name) {
  for (auto &n : F->getNodes()) {
    if (n.getName().str() == name) {
      return &n;
    }
  }
  return nullptr;
}

/// Mock backend that does lower FC nodes.
class MockBackendNoLowerConv3D : public MockBackend {
  bool shouldLower(const Node *N) const override {
    if (N->getKind() == Kinded::Kind::Convolution3DNodeKind) {
      return false;
    } else {
      return true;
    }
  }
};

TEST(Graph, testVariableErasure) {
  Module MD;
  auto &vars = MD.getConstants();
  EXPECT_EQ(vars.size(), 0);
  EXPECT_EQ(std::distance(vars.begin(), vars.end()), vars.size());

  Constant *V = MD.createConstant(ElemKind::FloatTy, {1, 1}, "dummy");
  EXPECT_EQ(vars.size(), 1);
  EXPECT_EQ(std::distance(vars.begin(), vars.end()), vars.size());

  MD.eraseConstant(V);
  EXPECT_EQ(vars.size(), 0);
  EXPECT_EQ(std::distance(vars.begin(), vars.end()), vars.size());
}

/// Check that the clear method completely reset a module.
TEST(Graph, clear) {
  Module M;

  // Check that the module is initially empty.
  EXPECT_EQ(M.getConstants().size(), 0);
  EXPECT_EQ(M.getPlaceholders().size(), 0);
  EXPECT_EQ(M.getFunctions().size(), 0);

  // Create a few things.
  M.createFunction("main");
  M.createPlaceholder(ElemKind::FloatTy, {1}, "placeholder", true);
  M.createConstant(ElemKind::FloatTy, {1}, "var");

  EXPECT_EQ(M.getConstants().size(), 1);
  EXPECT_EQ(M.getPlaceholders().size(), 1);
  EXPECT_EQ(M.getFunctions().size(), 1);

  // Check that clearing the module makes it completely free of any kind of
  // objects.
  M.clear();
  EXPECT_EQ(M.getConstants().size(), 0);
  EXPECT_EQ(M.getPlaceholders().size(), 0);
  EXPECT_EQ(M.getFunctions().size(), 0);
}

/// Check that the clear method works as expected.
TEST(Graph, clearFunctions) {
  Module M;

  // Check that the module is initially empty.
  EXPECT_EQ(M.getConstants().size(), 0);
  EXPECT_EQ(M.getPlaceholders().size(), 0);
  EXPECT_EQ(M.getFunctions().size(), 0);

  // Create a few things.
  Function *F = M.createFunction("main");
  auto *PH = M.createPlaceholder(ElemKind::FloatTy, {1}, "placeholder", true);
  auto *C = M.createConstant(ElemKind::FloatTy, {1}, "var");
  auto *AN = F->createAdd("add", PH, C);
  F->createSave("save", AN);

  EXPECT_EQ(M.getConstants().size(), 1);
  EXPECT_EQ(M.getPlaceholders().size(), 2); // Input PH and PH for Save
  EXPECT_EQ(M.getFunctions().size(), 1);
  EXPECT_EQ(F->getNodes().size(), 2); // Add, Save

  M.clearFunctions();
  EXPECT_EQ(M.getConstants().size(), 1);
  EXPECT_EQ(M.getPlaceholders().size(), 2);
  ASSERT_EQ(M.getFunctions().size(), 1);
  // Same Function ptr should exist, just nothing left in them.
  EXPECT_EQ(*M.getFunctions().begin(), F);
  EXPECT_EQ(F->getNodes().size(), 0);
}

/// Test the graph nodes names and utilities.
TEST(Graph, testGraphNames) {
  Module MD;
  Function *F = MD.createFunction("F");

  Node *op1 = MD.createPlaceholder(ElemKind::FloatTy, {1, 10}, "op1",
                                   false /*isTrainable*/);
  Node *op2 = MD.createConstant(ElemKind::FloatTy, {1, 10}, "op2");
  Node *add = F->createAdd("add", op1, op2);
  auto *top = F->createTopK("top", add, 5);
  Node *save = F->createSave("out", top->getValues());

  EXPECT_TRUE(MD.getPlaceholderByNameSlow("op1"));
  EXPECT_TRUE(MD.getConstantByName("op2"));
  EXPECT_TRUE(F->getNodeByName("add"));
  EXPECT_TRUE(F->getNodeByName("top"));
  EXPECT_TRUE(F->getNodeByName("out_save"));

  NodeValue op1Res = op1->getNthResult(0);
  NodeValue op2Res = op2->getNthResult(0);
  NodeValue addRes = add->getNthResult(0);
  EXPECT_TRUE(top->getNumResults() == 2);
  NodeValue topValRes = top->getNthResult(0);
  NodeValue topIndRes = top->getNthResult(1);

  auto op1ResName =
      op1Res.generateNodeOutputName(false /*stripResNoFor0thInput*/);
  auto op2ResName =
      op2Res.generateNodeOutputName(false /*stripResNoFor0thInput*/);
  auto addResName =
      addRes.generateNodeOutputName(true /*stripResNoFor0thInput*/);
  auto topValResName =
      topValRes.generateNodeOutputName(false /*stripResNoFor0thInput*/);
  auto topIndResName =
      topIndRes.generateNodeOutputName(false /*stripResNoFor0thInput*/);

  EXPECT_EQ(op1ResName, "op1:0");
  EXPECT_EQ(op2ResName, "op2:0");
  EXPECT_EQ(addResName, "add");
  EXPECT_EQ(topValResName, "top:0");
  EXPECT_EQ(topIndResName, "top:1");

  EXPECT_EQ(F->getNodeValueByName(op1ResName), op1Res);
  EXPECT_EQ(F->getNodeValueByName(op2ResName), op2Res);
  EXPECT_EQ(F->getNodeValueByName(addResName), addRes);
  EXPECT_EQ(F->getNodeValueByName(topValResName), topValRes);
  EXPECT_EQ(F->getNodeValueByName(topIndResName), topIndRes);

  EXPECT_EQ(F->getNodeValueByName("op1"), op1Res);
  EXPECT_EQ(F->getNodeValueByName("op2"), op2Res);
  EXPECT_EQ(F->getNodeValueByName("add:0"), addRes);

  // Verify the node value is invalid for the SaveNode which has no outputs.
  EXPECT_EQ(F->getNodeValueByName(save->getName()).getNode(), nullptr);
}

/// Check node names.
TEST(Graph, testNodeNames) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  PlaceholderBindings bindings;
  Node *K =
      MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);
  Node *S = MD.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", true);

  K = F->createConv(bindings, "Conv1", K, 16, 3, 2, 3, 1);
  K = F->createRELU("Relu", K);
  K = F->createSoftMax("SoftMax", K, S);
  F->createSave("Save", K);
  F->dump();
  auto filePath = F->dumpDAG();
  auto backend = MockBackend();
  CompilationContext cctx;
  lower(F, cctx, &backend);
  ::optimize(F, CompilationMode::Train);
  M.generateIR(backend);
  M.dump();
  EXPECT_GT(M.getInstrs().size(), 0);
  llvm::sys::fs::remove(filePath);
}

/// Check that a createConv3D can be run.
TEST(Graph, simpleTestConv3D) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  PlaceholderBindings bindings;
  Node *K = MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 100, 3},
                                 "input", true);
  K = F->createConv3D(bindings, /* name */ "Conv3D", /* input */ K,
                      /* outChannels */ 16, /* kernel */ 3, /* stride */ 2,
                      /* pad */ 3, /* group */ 1);
  K = F->createRELU("Relu", K);
  F->createSave("Save", K);
  F->dump();
  auto filePath = F->dumpDAG();
  auto backend = MockBackend();
  CompilationContext cctx;
  lower(F, cctx, &backend);
  ::optimize(F, CompilationMode::Train);
  M.generateIR(backend);
  M.dump();
  EXPECT_GT(M.getInstrs().size(), 0);
  llvm::sys::fs::remove(filePath);
}

/// Tests custom lowering from Node to Instruction IR
TEST(Graph, simpleTestConvCustomLower) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  PlaceholderBindings bindings;
  Node *K =
      MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);
  Node *S = MD.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", true);

  K = F->createConv(bindings, "Conv1", K, 16, 3, 2, 3, 1);
  K = F->createRELU("Relu", K);
  K = F->createSoftMax("SoftMax", K, S);
  F->createSave("Save", K);
  F->dump();
  auto filePath = F->dumpDAG();
  auto backend = MockBackendCustomIRGen();
  CompilationContext cctx;
  lower(F, cctx, &backend);
  ::optimize(F, CompilationMode::Train);
  M.generateIR(MockBackendCustomIRGen());
  M.dump();
  auto &instrList = M.getInstrs();
  bool customHappened = false;
  for (auto begin = instrList.begin(); begin != instrList.end(); ++begin) {
    if (begin->getName().equals("CustomConvolutionInstruction")) {
      customHappened = true;
      break;
    }
  }

  EXPECT_EQ(customHappened, true);
  llvm::sys::fs::remove(filePath);
}

/// Check that we can create convolution with float16.
TEST(Graph, float16Conv) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  Node *K = MD.createConstant(ElemKind::Float16Ty, {4, 320, 200, 3}, "input");

  auto *conv = F->createConv(bindings, "Conv", K, 16, 3, 2, 3, 1);
  F->createSave("Save", conv);
  EXPECT_TRUE(conv->verify());
  EXPECT_EQ(conv->getResult().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(conv->getFilter().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(conv->getBias().getElementType(), ElemKind::Float16Ty);

  auto backend = MockBackend();
  CompilationContext cctx;
  lower(F, cctx, &backend);

  IRFunction M(F);

  M.generateIR(backend);
  EXPECT_GT(M.getInstrs().size(), 0);
  auto convIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                             [](const Instruction &inst) -> bool {
                               return llvm::isa<ConvolutionInst>(inst);
                             });
  ASSERT_TRUE(convIt != M.getInstrs().end());
  const auto *convInst = llvm::cast<ConvolutionInst>(&*convIt);
  EXPECT_EQ(convInst->getSrc()->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(convInst->getFilter()->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(convInst->getBias()->getElementType(), ElemKind::Float16Ty);
}

/// Check that we can create layernormalization with float16.
TEST(Graph, float16LayerNorm) {
  const auto origFlagVal = interpreter::flags::LowerLayerNormalization;
  interpreter::flags::LowerLayerNormalization = false;

  Module MD;
  Function *F = MD.createFunction("F");

  PlaceholderBindings bindings;
  auto *input =
      MD.createPlaceholder(ElemKind::Float16Ty, {1, 4, 5, 5}, "in", false);

  Tensor scaleT(ElemKind::Float16Ty, {5, 5});
  scaleT.getHandle<float16_t>().randomize(0.0f, 1.0f, MD.getPRNG());
  Constant *scaleC = MD.createConstant("scale", std::move(scaleT));
  Tensor biasT(ElemKind::Float16Ty, {5, 5});
  biasT.getHandle<float16_t>().randomize(0.0f, 1.0f, MD.getPRNG());
  Constant *biasC = MD.createConstant("bias", std::move(biasT));

  LayerNormalizationNode *LNN = F->createLayerNormalization(
      "LN", input->getType(), input, scaleC, biasC, 1e-5);
  F->createSave("Save", LNN);

  std::unique_ptr<const Backend> backend(createBackend("Interpreter"));

  CompilationContext cctx;
  lower(F, cctx, backend.get());

  IRFunction M(F);

  M.generateIR(*backend);
  EXPECT_GT(M.getInstrs().size(), 0);
  auto lnIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                           [](const Instruction &inst) -> bool {
                             return llvm::isa<LayerNormalizationInst>(inst);
                           });
  ASSERT_TRUE(lnIt != M.getInstrs().end());
  interpreter::flags::LowerLayerNormalization = origFlagVal;
}

/// Check that we can create batch_matmul with float16.
TEST(Graph, float16BatchMatMul) {
  const auto origFlagVal = interpreter::flags::LowerBatchMatMul;
  interpreter::flags::LowerBatchMatMul = false;

  Module MD;
  Function *F = MD.createFunction("F");

  PlaceholderBindings bindings;
  auto *LHS = MD.createPlaceholder(ElemKind::Float16Ty, {2, 3, 4}, "A", false);
  auto *RHS = MD.createPlaceholder(ElemKind::Float16Ty, {2, 4, 5}, "B", false);

  BatchMatMulNode *BMM = F->createBatchMatMul("BMM", LHS, RHS);
  F->createSave("Save", BMM);

  std::unique_ptr<const Backend> backend(createBackend("Interpreter"));

  CompilationContext cctx;
  lower(F, cctx, backend.get());

  IRFunction M(F);

  M.generateIR(*backend);
  EXPECT_GT(M.getInstrs().size(), 0);
  auto bmmIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                            [](const Instruction &inst) -> bool {
                              return llvm::isa<BatchMatMulInst>(inst);
                            });
  ASSERT_TRUE(bmmIt != M.getInstrs().end());
  interpreter::flags::LowerBatchMatMul = origFlagVal;
}

/// Check that we can create batch_matmul with float.
TEST(Graph, floatBatchMatMul) {
  const auto origFlagVal = interpreter::flags::LowerBatchMatMul;
  interpreter::flags::LowerBatchMatMul = false;

  Module MD;
  Function *F = MD.createFunction("F");

  PlaceholderBindings bindings;
  auto *LHS = MD.createPlaceholder(ElemKind::FloatTy, {2, 3, 4}, "A", false);
  auto *RHS = MD.createPlaceholder(ElemKind::FloatTy, {2, 4, 5}, "B", false);

  BatchMatMulNode *BMM = F->createBatchMatMul("BMM", LHS, RHS);
  F->createSave("Save", BMM);

  std::unique_ptr<const Backend> backend(createBackend("Interpreter"));

  CompilationContext cctx;
  lower(F, cctx, backend.get());

  IRFunction M(F);

  M.generateIR(*backend);
  EXPECT_GT(M.getInstrs().size(), 0);
  auto bmmIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                            [](const Instruction &inst) -> bool {
                              return llvm::isa<BatchMatMulInst>(inst);
                            });
  ASSERT_TRUE(bmmIt != M.getInstrs().end());
  interpreter::flags::LowerBatchMatMul = origFlagVal;
}

/// Check that we can create conv3D with float16.
TEST(Graph, float16Conv3DLower) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  Node *K =
      MD.createConstant(ElemKind::Float16Ty, {4, 320, 200, 200, 3}, "input");

  auto *conv = F->createConv3D(bindings, "Conv3D", K, 16, 3, 2, 3, 1);
  F->createSave("Save", conv);
  EXPECT_TRUE(conv->verify());
  EXPECT_EQ(conv->getResult().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(conv->getFilter().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(conv->getBias().getElementType(), ElemKind::Float16Ty);

  auto backend = MockBackend();
  CompilationContext cctx;
  lower(F, cctx, &backend);

  IRFunction M(F);

  M.generateIR(backend);
  EXPECT_GT(M.getInstrs().size(), 0);
  auto convIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                             [](const Instruction &inst) -> bool {
                               return llvm::isa<Convolution3DInst>(inst);
                             });
  ASSERT_TRUE(convIt == M.getInstrs().end());
}

/// Check that we can create conv3D with float16.
TEST(Graph, float16Conv3DNoLower) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  Node *K =
      MD.createConstant(ElemKind::Float16Ty, {4, 320, 200, 200, 3}, "input");

  auto *conv = F->createConv3D(bindings, "Conv3D", K, 16, 3, 2, 3, 1);
  F->createSave("Save", conv);
  EXPECT_TRUE(conv->verify());
  EXPECT_EQ(conv->getResult().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(conv->getFilter().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(conv->getBias().getElementType(), ElemKind::Float16Ty);

  auto backend = MockBackendNoLowerConv3D();
  CompilationContext cctx;
  lower(F, cctx, &backend);

  IRFunction M(F);

  M.generateIR(backend);
  EXPECT_GT(M.getInstrs().size(), 0);
  auto convIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                             [](const Instruction &inst) -> bool {
                               return llvm::isa<Convolution3DInst>(inst);
                             });
  ASSERT_TRUE(convIt != M.getInstrs().end());
  const auto *convInst = llvm::cast<Convolution3DInst>(&*convIt);
  EXPECT_EQ(convInst->getSrc()->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(convInst->getFilter()->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(convInst->getBias()->getElementType(), ElemKind::Float16Ty);
}

/// Check that we can create batchNorm with float16.
TEST(Graph, float16BatchNorm) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  auto *input =
      MD.createPlaceholder(ElemKind::Float16Ty, {1, 10, 20, 3}, "input", false);
  BatchNormalizationNode *BN =
      F->createBatchNormalization(bindings, "batch", input, 3, 0.0001, 0.9);

  EXPECT_TRUE(BN->verify());
  EXPECT_EQ(BN->getResult().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(BN->getScale().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(BN->getBias().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(BN->getMean().getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(BN->getVar().getElementType(), ElemKind::Float16Ty);

  auto backend = MockBackend();
  CompilationContext cctx;
  lower(F, cctx, &backend);

  EXPECT_TRUE(std::all_of(
      F->getNodes().begin(), F->getNodes().end(), [](const Node &node) -> bool {
        for (unsigned idx = 0, end = node.getNumResults(); idx != end; ++idx) {
          if (node.getType(idx)->getElementType() != ElemKind::Float16Ty) {
            return false;
          }
        }
        return true;
      }));
}

/// Check that we can create convolution with bfloat16.
TEST(Graph, bfloat16Conv) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  Node *K = MD.createConstant(ElemKind::BFloat16Ty, {4, 320, 200, 3}, "input");

  auto *conv = F->createConv(bindings, "Conv", K, 16, 3, 2, 3, 1);
  F->createSave("Save", conv);
  EXPECT_TRUE(conv->verify());
  EXPECT_EQ(conv->getResult().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(conv->getFilter().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(conv->getBias().getElementType(), ElemKind::BFloat16Ty);

  auto backend = MockBackend();
  CompilationContext cctx;
  lower(F, cctx, &backend);

  IRFunction M(F);

  M.generateIR(backend);
  EXPECT_GT(M.getInstrs().size(), 0);
  auto convIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                             [](const Instruction &inst) -> bool {
                               return llvm::isa<ConvolutionInst>(inst);
                             });
  ASSERT_TRUE(convIt != M.getInstrs().end());
  const auto *convInst = llvm::cast<ConvolutionInst>(&*convIt);
  EXPECT_EQ(convInst->getSrc()->getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(convInst->getFilter()->getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(convInst->getBias()->getElementType(), ElemKind::BFloat16Ty);
}

/// Check that we can create conv3D with bfloat16.
TEST(Graph, bfloat16Conv3DLower) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  Node *K =
      MD.createConstant(ElemKind::BFloat16Ty, {4, 320, 200, 200, 3}, "input");

  auto *conv = F->createConv3D(bindings, "Conv3D", K, 16, 3, 2, 3, 1);
  F->createSave("Save", conv);
  EXPECT_TRUE(conv->verify());
  EXPECT_EQ(conv->getResult().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(conv->getFilter().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(conv->getBias().getElementType(), ElemKind::BFloat16Ty);

  auto backend = MockBackend();
  CompilationContext cctx;
  lower(F, cctx, &backend);

  IRFunction M(F);

  M.generateIR(backend);
  EXPECT_GT(M.getInstrs().size(), 0);
  auto convIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                             [](const Instruction &inst) -> bool {
                               return llvm::isa<Convolution3DInst>(inst);
                             });
  ASSERT_TRUE(convIt == M.getInstrs().end());
}

/// Check that we can create conv3D with bfloat16.
TEST(Graph, bfloat16Conv3DNoLower) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  Node *K =
      MD.createConstant(ElemKind::BFloat16Ty, {4, 320, 200, 200, 3}, "input");

  auto *conv = F->createConv3D(bindings, "Conv3D", K, 16, 3, 2, 3, 1);
  F->createSave("Save", conv);
  EXPECT_TRUE(conv->verify());
  EXPECT_EQ(conv->getResult().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(conv->getFilter().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(conv->getBias().getElementType(), ElemKind::BFloat16Ty);

  auto backend = MockBackendNoLowerConv3D();
  CompilationContext cctx;
  lower(F, cctx, &backend);

  IRFunction M(F);

  M.generateIR(backend);
  EXPECT_GT(M.getInstrs().size(), 0);
  auto convIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                             [](const Instruction &inst) -> bool {
                               return llvm::isa<Convolution3DInst>(inst);
                             });
  ASSERT_TRUE(convIt != M.getInstrs().end());
  const auto *convInst = llvm::cast<Convolution3DInst>(&*convIt);
  EXPECT_EQ(convInst->getSrc()->getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(convInst->getFilter()->getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(convInst->getBias()->getElementType(), ElemKind::BFloat16Ty);
}

/// Check that we can create batchNorm with float16.
TEST(Graph, bfloat16BatchNorm) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  auto *input = MD.createPlaceholder(ElemKind::BFloat16Ty, {1, 10, 20, 3},
                                     "input", false);
  BatchNormalizationNode *BN =
      F->createBatchNormalization(bindings, "batch", input, 3, 0.0001, 0.9);

  EXPECT_TRUE(BN->verify());
  EXPECT_EQ(BN->getResult().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(BN->getScale().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(BN->getBias().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(BN->getMean().getElementType(), ElemKind::BFloat16Ty);
  EXPECT_EQ(BN->getVar().getElementType(), ElemKind::BFloat16Ty);

  auto backend = MockBackend();
  CompilationContext cctx;
  lower(F, cctx, &backend);

  EXPECT_TRUE(std::all_of(
      F->getNodes().begin(), F->getNodes().end(), [](const Node &node) -> bool {
        for (unsigned idx = 0, end = node.getNumResults(); idx != end; ++idx) {
          if (node.getType(idx)->getElementType() != ElemKind::BFloat16Ty) {
            return false;
          }
        }
        return true;
      }));
}

/// Test that our use lists are correctly reflecting the state of the IR
/// and in particular that it is not polluted by temporary variable.
TEST(Graph, useList) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  PlaceholderBindings bindings;
  auto *K =
      MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);

  EXPECT_EQ(K->getNumUsers(), 0);

  ConvolutionNode *conv = F->createConv(bindings, "Conv1", K, 16, 3, 2, 3, 1);

  EXPECT_TRUE(K->hasOneUse());
  EXPECT_EQ(K->getNumUsers(), 1);
  EXPECT_EQ(conv->getNumUsers(), 0);

  // Although the filter of the convolution is only used by the convolution
  // node, calling getFilter creates a temporary NodeValue that messes up
  // with the actual use list.
  // Therefore those checks are currently inverted but should be
  // fixed eventually.
  // Test with implicit temporary NodeValue.
  EXPECT_TRUE(conv->getFilter().getNode()->hasOneUse());
  EXPECT_EQ(conv->getFilter().getNode()->getNumUsers(), 1);

  // Test with explicit temporary NodeValue.
  Node *nodeFilter;
  {
    NodeValue tmp = conv->getFilter();
    EXPECT_TRUE(tmp.getNode()->hasOneUse());
    EXPECT_EQ(tmp.getNode()->getNumUsers(), 1);
    nodeFilter = tmp.getNode();
    // Test with NodeValue still around.
    EXPECT_TRUE(nodeFilter->hasOneUse());
    EXPECT_EQ(nodeFilter->getNumUsers(), 1);
  }

  // Test with NodeValue took out.
  EXPECT_TRUE(nodeFilter->hasOneUse());
  EXPECT_EQ(nodeFilter->getNumUsers(), 1);

  // Same kind of test but with the convolution node itself.
  {
    NodeValue tmpConvRes(conv, 0);
    EXPECT_EQ(conv->getNumUsers(), 0);
    EXPECT_EQ(tmpConvRes.getNode()->getNumUsers(), 0);
  }

  // Add a couple of uses to conv and make sure it reflects on its use list.
  F->createSave("Save", conv, K);

  EXPECT_FALSE(K->hasOneUse());
  EXPECT_EQ(K->getNumUsers(), 2);
  EXPECT_EQ(conv->getNumUsers(), 1);
  EXPECT_TRUE(conv->hasOneUse());

  {
    NodeValue tmpConvRes(conv, 0);
    EXPECT_TRUE(tmpConvRes.getNode()->hasOneUse());
    EXPECT_TRUE(conv->hasOneUse());
    EXPECT_EQ(conv->getNumUsers(), 1);
    EXPECT_EQ(tmpConvRes.getNode()->getNumUsers(), 1);
  }

  F->createSave("Save", conv, K);

  EXPECT_FALSE(K->hasOneUse());
  EXPECT_EQ(K->getNumUsers(), 3);
  EXPECT_EQ(conv->getNumUsers(), 2);
  EXPECT_FALSE(conv->hasOneUse());

  {
    NodeValue tmpConvRes(conv, 0);
    EXPECT_FALSE(tmpConvRes.getNode()->hasOneUse());
    EXPECT_FALSE(conv->hasOneUse());
    EXPECT_EQ(conv->getNumUsers(), 2);
    EXPECT_EQ(tmpConvRes.getNode()->getNumUsers(), 2);
  }
}

TEST(Graph, useListIteration) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Node *K =
      MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);

  EXPECT_EQ(K->getNumUsers(), 0);

  PlaceholderBindings bindings;
  ConvolutionNode *conv1 = F->createConv(bindings, "Conv1", K, 16, 3, 2, 3, 1);
  ConvolutionNode *conv2 = F->createConv(bindings, "Conv2", K, 16, 3, 2, 3, 1);
  // Check the number of users for different nodes.
  EXPECT_EQ(K->getNumUsers(), 2);
  EXPECT_EQ(conv1->getNumUsers(), 0);
  EXPECT_TRUE(conv2->getFilter().getNode()->hasOneUse());
  EXPECT_EQ(conv1->getFilter().getNode()->getNumUsers(), 1);
  // Check that the first user of K is conv1.
  EXPECT_EQ(K->getUsers().begin()->getUser(), conv1);
  // Check that the second user of K is conv2.
  EXPECT_EQ((++K->getUsers().begin())->getUser(), conv2);
}

TEST(Graph, simpleTestFC) {
  unsigned numInputs = 10;
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);

  auto *A = MD.createPlaceholder(ElemKind::FloatTy, {numInputs, 2}, "A", true);
  auto *Ex =
      MD.createPlaceholder(ElemKind::FloatTy, {numInputs, 1}, "Ex", true);

  PlaceholderBindings bindings;
  Node *O = F->createFullyConnected(bindings, "FC1", A, 6);
  O = F->createRELU("RELU1", O);
  O = F->createFullyConnected(bindings, "FC2", O, 1);
  O = F->createRELU("RELU2", O);
  O = F->createRegression("Regression", O, Ex);
  F->createSave("Save", O);
  F->dump();
  auto filePath = F->dumpDAG();
  auto backend = MockBackend();
  CompilationContext cctx;
  lower(F, cctx, &backend);
  ::optimize(F, CompilationMode::Train);
  M.generateIR(backend);
  M.dump();
  EXPECT_GT(M.getInstrs().size(), 0);
  llvm::sys::fs::remove(filePath);
}

TEST(Graph, QuantizationProfileNodes) {
  unsigned numInputs = 10;
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);

  auto *A = MD.createPlaceholder(ElemKind::FloatTy, {numInputs, 2}, "A", true);

  // Add non float operation, which should not be profiled.
  auto *outQTy = F->getParent()->uniqueType(glow::ElemKind::Int8QTy,
                                            {numInputs, 2}, 1.5, 6);
  auto *quantize = F->createQuantize("quantize", A, outQTy);
  // Make sure that quantize is not optimized away.
  PlaceholderBindings bindings;
  F->createSave("save", quantize);

  // Multiple nodes read from the same variable.
  // Only one Quantization Profile node should be created for the output
  // from the variable.
  Node *O = F->createFullyConnected(bindings, "FC1", A, 6);
  Node *C = F->createFullyConnected(bindings, "FC2", A, 6);
  O = F->createRELU("RELU1", O);
  F->createSave("save", O);
  F->createSave("save", C);

  LoweredInfoMap loweredMapForProf;
  CompilationContext cctx{&bindings, &loweredMapForProf};
  cctx.precisionConfig.quantMode = QuantizationMode::Profile;
  std::unique_ptr<Backend> backend(createBackend("Interpreter"));
  EXIT_ON_ERR(::optimizeFunction(F, *backend, cctx));

  size_t numberOfProfileNodes =
      std::count_if(F->getNodes().begin(), F->getNodes().end(), [](Node &node) {
        return llvm::isa<QuantizationProfileNode>(&node);
      });

  // 1 from A
  // 8 from two lowered FCs: MM, BA, weight PH, bias PH
  // 2 from RELU (lowered to Max+Splat)
  EXPECT_EQ(11, numberOfProfileNodes);
}

TEST(Graph, simpleQuant) {
  ExecutionEngine EE;
  auto &MD = EE.getModule();
  auto *F = MD.createFunction("main");

  unsigned depth = 16;
  llvm::SmallVector<unsigned_t, 2> kernels = {5, 5};
  llvm::SmallVector<unsigned_t, 4> pads = {0, 0, 0, 0};
  llvm::SmallVector<unsigned_t, 2> steps = {1, 1};
  unsigned width = 224;

  auto *input = MD.createPlaceholder(ElemKind::Int8QTy, {1, width, width, 3},
                                     0.4, 2, "Input", true);

  // Calculate the size and allocate the output buffer.
  std::array<dim_t, 4> filterDim = {{depth, kernels[0], kernels[1], 3}};
  auto *filter =
      MD.createPlaceholder(ElemKind::Int8QTy, filterDim, 3.3, 4, "F", true);
  auto *bias =
      MD.createPlaceholder(ElemKind::Int32QTy, {depth}, 1.3, 5, "B", true);

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvPoolOutputDims(width, width, kernels, steps, pads);
  std::array<dim_t, 4> outDims = {{1, outSz.first, outSz.second, 16}};
  auto t = F->getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims, 1.5, 6);

  auto *conv =
      F->createConv("conv", input, filter, bias, t, kernels, steps, pads, 1);

  auto s = conv->getResult().getType()->size();
  auto *fcFilter =
      MD.createPlaceholder(ElemKind::Int8QTy, {s, 6}, 0.4, 2, "F", true);
  auto *fcBias =
      MD.createPlaceholder(ElemKind::Int32QTy, {6}, 0.4, 2, "B", true);
  Node *O = F->createFullyConnected("fc1", conv, fcFilter, fcBias);
  PlaceholderBindings bindings;
  F->createSave("ret", O);
  EE.compile(CompilationMode::Infer);
}

TEST(Graph, quantizeDequantizeNodes) {
  ExecutionEngine EE;
  auto &MD = EE.getModule();
  auto F = MD.createFunction("main");

  auto *input = MD.createPlaceholder(ElemKind::FloatTy, {1, 3}, "Input", true);
  auto qType = F->getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 0.3, 5);

  auto *Q = F->createQuantize("quantize", input, qType);

  auto transform =
      F->getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 1.4, 3);
  auto *A = F->createRescaleQuantized("rescale", Q, transform);

  auto *D = F->createDequantize("dequantize", A, ElemKind::FloatTy);
  PlaceholderBindings bindings;
  F->createSave("ret", D);
  EE.compile(CompilationMode::Infer);
}

TEST(Graph, quantizeGather) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::Int8QTy, {2, 2}, 0.4, 2, "input", true);
  auto *indices = mod.createPlaceholder(ElemKind::Int64ITy, {1}, "index", true);
  auto *gather = F->createGather("gather", input, indices);
  PlaceholderBindings bindings;
  F->createSave("ret", gather);
  EE.compile(CompilationMode::Infer);
}

TEST(Graph, cloneTest) {
  Module M;
  PlaceholderBindings bindings;

  Function *F = M.createFunction("main");
  Node *K =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);
  Node *S = M.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", true);
  Node *conv = F->createConv(bindings, "Conv1", K, 16, 3, 2, 3, 1);
  Node *relu = F->createRELU("Relu", conv);
  Node *SM = F->createSoftMax("SoftMax", relu, S);
  F->createSave("Save", SM);

  auto *newConv = F->addNode(conv->clone());
  auto *newRelu = F->addNode(relu->clone());
  auto *newSM = F->addNode(SM->clone());

  EXPECT_TRUE(newConv != conv && conv->isEqual(*newConv));
  EXPECT_TRUE(newRelu != relu && relu->isEqual(*newRelu));
  EXPECT_TRUE(newSM != SM && SM->isEqual(*newSM));
}

TEST(Graph, moduleTest) {
  Module M;
  M.createFunction("one");
  M.createFunction("two");
  M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V1", true);
  M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V2", true);
  EXPECT_TRUE(M.hasFunction("one"));
  EXPECT_TRUE(M.hasFunction("two"));
  EXPECT_FALSE(M.hasFunction("four"));
  M.dumpDAG();
}

TEST(Graph, functionDependenciesTest) {
  Module M;
  auto *F1 = M.createFunction("one");
  auto *F2 = M.createFunction("two");
  auto *V1 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V1", true);
  auto *V2 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V2", true);
  auto *V3 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V3", true);
  M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V4", true);

  PlaceholderBindings bindings;
  auto sum = F1->createSub("1_sub_2", V1, V2);
  F1->createSave("sv", sum, V1);
  F2->createSave("sv", V3, V2);

  EXPECT_TRUE(M.hasFunction("one"));
  EXPECT_TRUE(M.hasFunction("two"));
  EXPECT_FALSE(M.hasFunction("four"));
  M.dumpDAG();
}

TEST(Graph, functionCloneTest) {
  Module M;
  PlaceholderBindings bindings;

  auto *F = M.createFunction("main");
  Node *K =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);
  Node *S = M.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", true);
  Node *conv = F->createConv(bindings, "Conv", K, 16, 3, 2, 3, 1);
  Node *relu = F->createRELU("Relu", conv);
  Node *concat = F->createConcat("concat", {relu, relu, relu}, 0);
  Node *SM = F->createSoftMax("SoftMax", concat, S);
  F->createSave("Save", SM);

  auto *newF = F->clone("new_main");

  EXPECT_TRUE(newF->verify());

  EXPECT_EQ(newF->getNodes().size(), F->getNodes().size());
  EXPECT_EQ(newF->getParent(), F->getParent());
}

/// Compile the module \p M inside the execution engine \p EE and then run it
/// using the provided \p bindings. Use the provided \p inputName and \p
/// outputName.
static void compileAndRun(ExecutionEngine &EE, PlaceholderBindings &bindings,
                          Module &M, llvm::StringRef inputName,
                          llvm::StringRef outputName) {
  EE.compile(glow::CompilationMode::Infer);
  // Allocate stprage for placeholders and initialize inputs.
  bindings.allocate(M.getPlaceholderByNameSlow(inputName))
      ->getHandle()
      .clear(2.0);
  bindings.allocate(M.getPlaceholderByNameSlow(outputName));
  EE.run(bindings);
}

/// Check the module cloning functionality.
TEST(Graph, moduleCloneTest) {
  // State related to the cloned module and its execution.
  ExecutionEngine clonedEE("Interpreter");
  Module &clonedM = clonedEE.getModule();
  PlaceholderBindings clonedBindings;
  Tensor clonedResult;
  // State related to the original module and its execution.
  PlaceholderBindings originalBindings;
  Tensor originalResult;
  // Name of the placeholder holding the results of executions.
  std::string resultName;
  {
    // Define the original execution engine and module.
    ExecutionEngine originalEE("Interpreter");
    Module &originalM = originalEE.getModule();

    // Create a function.
    auto *F = originalM.createFunction("main");
    auto *input1 = originalM.createPlaceholder(ElemKind::FloatTy,
                                               {4, 10, 10, 3}, "input", true);

    auto *add = F->createAdd("add", input1, input1);
    auto *relu = F->createRELU("Relu", add);
    auto *concat = F->createConcat("concat", {relu, relu, relu}, 0);
    auto *C = originalM.createConstant(concat->getResult().getType(), "C");
    C->getPayloadMutable().getHandle().clear(1.0f);
    auto *SM = F->createAdd("add", concat, C);
    auto *SN = F->createSave("Save", SM);
    resultName = SN->getPlaceholder()->getName().str();

    // Clone the original module into the cloned module.
    originalM.clone(&clonedM);
    // The cloned module should have the same numer of types, functions,
    // constants and placeholders.
    EXPECT_EQ(originalM.getFunctions().size(), clonedM.getFunctions().size());
    EXPECT_EQ(originalM.getPlaceholders().size(),
              clonedM.getPlaceholders().size());
    EXPECT_EQ(originalM.getConstants().size(), clonedM.getConstants().size());
    EXPECT_EQ(originalM.getTypes().size(), clonedM.getTypes().size());
    // String representations of the original and cloned modules should be the
    // same.
    EXPECT_EQ(originalM.toString(), clonedM.toString());
    for (auto *originalF : originalM.getFunctions()) {
      EXPECT_EQ(originalF->toString(),
                clonedM.getFunction(originalF->getName())->toString());
    }

    // Compile and run the original module.
    compileAndRun(originalEE, originalBindings, originalM, "input", resultName);
    // Store the result of running the original module.
    originalResult.assign(originalBindings.get(
        originalBindings.getPlaceholderByNameSlow(resultName)));
    // The old module should be removed when this scope ends. Thus, if the
    // cloned module newM refers to any deleted nodes from the original module,
    // it would result in a dangling reference and most likely in a crash.
  }
  // Check that the cloned module is still alive and valid after the original
  // module was deleted.
  EXPECT_TRUE(clonedM.verify());
  // Compile and run the cloned model.
  compileAndRun(clonedEE, clonedBindings, clonedM, "input", resultName);
  // Store the result of running the cloned module.
  clonedResult.assign(
      clonedBindings.get(clonedBindings.getPlaceholderByNameSlow(resultName)));
  // The results of execution should be exactly the same in both cases.
  EXPECT_TRUE(originalResult.isEqual(clonedResult, 0));
}

TEST(Graph, cloneWithPredicates) {
  Module M;
  PlaceholderBindings bindings;

  auto *F = M.createFunction("main");
  auto *input =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", false);
  auto *counters =
      M.createPlaceholder(ElemKind::FloatTy, {10}, "counters", false);
  auto *reluExt = F->createRELU("reluExt", input);
  auto *reluInt = F->createRELU("reluInt", input);
  auto *externalPredicate =
      M.createPlaceholder(ElemKind::Int64ITy, {1}, "predicate", false);
  auto *C10 = F->createSplat("C10", counters->getType(), 10.0);
  auto *internalPredicate = F->createCmpLTE("lte", C10, counters);

  reluExt->setPredicate(externalPredicate);
  reluInt->setPredicate(internalPredicate);

  auto *newF = F->clone("new_main");

  EXPECT_TRUE(newF->verify());
  EXPECT_EQ(newF->getNodes().size(), F->getNodes().size());
  EXPECT_EQ(newF->getParent(), F->getParent());

  // Original predicates are not changed
  EXPECT_EQ(reluExt->getPredicate().getNode(), externalPredicate);
  EXPECT_EQ(reluInt->getPredicate().getNode(), internalPredicate);
  // Clone of predicate that points to a node outside the graph
  // points to the same node (predicate is shared)
  EXPECT_EQ(nodeByName(newF, "reluExt")->getPredicate().getNode(),
            externalPredicate);
  // Clone of predicate that points to a node that belongs to the graph
  // points to the predicate clone
  EXPECT_EQ(nodeByName(newF, "reluInt")->getPredicate().getNode(),
            nodeByName(newF, "lte"));
}

TEST(Graph, NodeValue) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  auto *inputX = mod.createPlaceholder(ElemKind::FloatTy, {1}, "input", true);
  bindings.allocate(inputX)->init(Tensor::InitKind::Broadcast, 3.0,
                                  mod.getPRNG());

  NodeValue a = F->createAdd("x2", inputX, inputX);
  a = F->createAdd("x4", a, a);
  a = F->createAdd("x8", a, a);
  auto *S = F->createSave("Save", a);
  auto *res = bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  EXPECT_EQ(res->getHandle().raw(0), 24);
}

/// Check that by deleting one function, the variables that refernced
/// by this function, will reduce its number of uses by one.
TEST(Graph, deleteFunction) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F1 = mod.createFunction("f1");
  auto *inputX = mod.createPlaceholder(ElemKind::FloatTy, {1}, "input", true);
  F1->createLog("log1", inputX);
  Function *F2 = mod.createFunction("f2");
  F2->createLog("log2", inputX);
  // We check the number of user of inputX to be 2 as only F1 and F2 are
  // using it.
  EXPECT_EQ(inputX->getNumUsers(), 2);
  // Erase this function here to see if we can see the number of user of inputX
  // reduce to 1.
  mod.eraseFunction(F1);
  EXPECT_EQ(inputX->getNumUsers(), 1);
}

TEST(Graph, nodesWithPredicates) {
  ExecutionEngine EE;

  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  F->setName("interpret");
  PlaceholderBindings bindings;
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 3}, "input", true);
  auto *ex = mod.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "exp", true);
  auto *pred =
      mod.createPlaceholder(ElemKind::Int64ITy, {1}, "predicate", false);
  bindings.allocate(input);
  bindings.allocate(ex);
  bindings.allocate(pred);

  auto *CV0 = F->createConv(bindings, "conv1", input, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createMaxPool("pool1", RL0, 2, 2, 0);

  CV0->setPredicate(pred);
  RL0->setPredicate(pred);
  MP0->setPredicate(pred);

  auto *FCL1 = F->createFullyConnected(bindings, "fc", MP0->getResult(), 10);
  auto *RL3 = F->createRELU("relu4", FCL1);
  auto *SM = F->createSoftMax("sm", RL3, ex);
  auto *save = F->createSave("ret", SM);
  bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {input}, {&inputs});
  EE.run(bindings);
}

// Return the number of ConvolutionNode after lower.
unsigned getConvNodeSize(llvm::StringRef kind) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  PlaceholderBindings bindings;
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 32}, "input", true);
  ConvolutionNode *CN = F->createConv(bindings, "conv", input, 6, 1, 1, 0, 2);
  F->createSave("save", CN);

  std::unique_ptr<Backend> backend(createBackend(kind));
  CompilationContext cctx;
  lower(F, cctx, backend.get());

  unsigned count = 0;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == Kinded::Kind::ConvolutionNodeKind) {
      count++;
    }
  }

  if (kind == "Interpreter") {
    EXPECT_EQ(count, 1);
  }

  return count;
}

// Check the unrolling grouped convolution opt status:
// -- disabled for Interpreter, CPU and OpenCL backend,
TEST(Graph, disableUnrollingGroupConv) {
  unsigned numberOfNodesInterpreter = getConvNodeSize("Interpreter");
  (void)numberOfNodesInterpreter;

#ifdef GLOW_WITH_CPU
  unsigned numberOfNodesCPU = getConvNodeSize("CPU");
  EXPECT_EQ(numberOfNodesCPU, numberOfNodesInterpreter);
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
  unsigned numberOfNodesOpenCL = getConvNodeSize("OpenCL");
  EXPECT_EQ(numberOfNodesOpenCL, numberOfNodesInterpreter);
#endif // GLOW_WITH_OPENCL
}

/// Check that save nodes are properly scheduled.
/// That is, they happen after the last use of the related variable.
/// In that test, the order of the creation of the nodes give a valid schedule.
TEST(Graph, schedulingOfSavesOrderProvided) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {3, 32}, "A", true);
  auto *B = mod.createPlaceholder(A->getType(), "B", true);
  auto *zero = mod.createPlaceholder(A->getType(), "zero", true);

  PlaceholderBindings bindings;
  bindings.allocate(A)->init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  bindings.allocate(B)->init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  bindings.allocate(zero)->init(Tensor::InitKind::Broadcast, 0.0,
                                mod.getPRNG());

  auto *addAB = F->createAdd("addAB", A, B);

  auto *saveNode = F->createSave("ret", addAB);
  auto *savePH = saveNode->getPlaceholder();
  bindings.allocate(savePH);
  F->createSave("resetA", zero, A);

  // Copy the value of A.
  Tensor AOrig = bindings.get(A)->clone();

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);
  auto *ret = bindings.get(savePH);
  auto handleAOrig = AOrig.getHandle<>();
  auto handleB = bindings.get(B)->getHandle<>();
  auto handleRet = ret->getHandle<>();
  bool allEqual = true;
  for (unsigned row = 0; row != 3; ++row) {
    for (unsigned column = 0; column != 32; ++column) {
      allEqual &= handleAOrig.at({row, column}) + handleB.at({row, column}) ==
                  handleRet.at({row, column});
    }
  }
  EXPECT_TRUE(bindings.get(A)->isEqual(*bindings.get(zero), 0.0));
  EXPECT_TRUE(allEqual);
}

/// Same as schedulingOfSavesOrderProvided except the order in which the nodes
/// are added to the function don't form a valid schedule.
/// In other words, the scheduler won't get away with scheduling
/// using only the order of the nodes in the list of nodes.
TEST(Graph, schedulingOfSaves) {
  ExecutionEngine EE;
  PlaceholderBindings bindings;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {3, 32}, "A", true);
  auto *B = mod.createPlaceholder(A->getType(), "B", true);
  auto *zero = mod.createPlaceholder(A->getType(), "zero", true);
  F->createSave("resetA", zero, A);

  bindings.allocate(A)->init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  bindings.allocate(B)->init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  bindings.allocate(zero)->init(Tensor::InitKind::Broadcast, 0.0,
                                mod.getPRNG());

  auto *addAB = F->createAdd("addAB", A, B);

  auto *saveNode = F->createSave("ret", addAB);
  bindings.allocate(saveNode->getPlaceholder());

  // Copy the value of A.
  Tensor AOrig = bindings.get(A)->clone();
  auto *ret = saveNode->getPlaceholder();
  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  auto handleAOrig = AOrig.getHandle<>();
  auto handleB = bindings.get(B)->getHandle<>();
  auto handleRet = bindings.get(ret)->getHandle<>();
  bool allEqual = true;
  for (unsigned row = 0; row != 3; ++row) {
    for (unsigned column = 0; column != 32; ++column) {
      allEqual &= handleAOrig.at({row, column}) + handleB.at({row, column}) ==
                  handleRet.at({row, column});
    }
  }
  EXPECT_TRUE(bindings.get(A)->isEqual(*bindings.get(zero), 0.0));
  EXPECT_TRUE(allEqual);
}

/// Check that the parent link is properly updated while tweaking
/// nodes and their function.
TEST(Graph, parentLink) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Constant *V =
      new Constant("V", mod.uniqueType(ElemKind::FloatTy, {3, 32}), ANY_LAYOUT);

  // Variables don't belong to any function...
  EXPECT_EQ(V->getParent(), nullptr);
  // Even when we create them from a module...
  Constant *V2 = mod.createConstant(V->getType(), "V2");
  EXPECT_EQ(V2->getParent(), nullptr);

  Function *F = mod.createFunction("main");

  // Nodes created with function helper belong to the related function.
  auto *addNode = F->createAdd("addnode", V, V2);
  EXPECT_EQ(addNode->getParent(), F);

  // Nodes created directly don't belong to any function.
  auto *addNode2 = new AddNode("addnode2", V->getType(), addNode, addNode);
  EXPECT_EQ(addNode2->getParent(), nullptr);

  // Nodes added to a function belong to that function.
  F->addNode(addNode2);
  EXPECT_EQ(addNode2->getParent(), F);

  // Cloned nodes don't belong to anything.
  auto *clonedAddNode = addNode->clone();
  EXPECT_EQ(clonedAddNode->getParent(), nullptr);

  // Check that the setter properly sets things.
  clonedAddNode->setParent(F);
  EXPECT_EQ(clonedAddNode->getParent(), F);
  clonedAddNode->setParent(nullptr);
  EXPECT_EQ(clonedAddNode->getParent(), nullptr);

  // Add the cloned node to F so that the memory is properly
  // cleaned at the end of the test.
  F->addNode(clonedAddNode);
  EXPECT_EQ(clonedAddNode->getParent(), F);

  delete V;
}

/// Check that verification can detect that Storage nodes are being used by
/// Functions in a Module that doesn't own the Storage nodes.
TEST(Graph, moduleLink) {
  ExecutionEngine EEA, EEB;

  auto &modA = EEA.getModule();
  auto &modB = EEB.getModule();

  auto *FA = modA.createFunction("FA");
  auto *FB = modB.createFunction("FB");

  auto *C = modA.createConstant(ElemKind::FloatTy, {1}, "C");
  auto *P = modA.createPlaceholder(ElemKind::FloatTy, {1}, "P", false);

  auto *AA = FA->createAdd("AA", C, P);
  FA->createSave("SA", AA);

  // These nodes use Storage nodes that reside in modA
  auto *AB = FB->createAdd("AB", C, P);
  FB->createSave("SB", AB);

  EXPECT_TRUE(modA.verify());
  EXPECT_FALSE(
      modB.verify()); // Module::verify calls Function::verify on all functions
                      // within the module, so this should fail
}

/// Check that Cmp nodes are created with proper output types.
TEST(Graph, cmpOutputTypes) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  // Define two different quntized types.
  auto qType1 = F->getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 0.3, 5);
  auto qType2 = F->getParent()->uniqueType(ElemKind::Int8QTy, {1, 3}, 0.4, 5);
  // Define two variables of quantized types.
  auto *qv1 = mod.createPlaceholder(qType1, "V1", true);
  auto *qv2 = mod.createPlaceholder(qType2, "V2", true);
  // Create cmp nodes using quantized inputs.
  auto *cmpNode1 = F->createCmpEQ("cmpeq", qv1, qv2);
  auto *cmpNode2 = F->createCmpLTE("cmplte", qv1, qv2);
  // Check that the output type of cmp nodes is BoolKind.
  EXPECT_TRUE(cmpNode1->getResult().getElementType() == ElemKind::BoolTy);
  EXPECT_TRUE(cmpNode2->getResult().getElementType() == ElemKind::BoolTy);

  // Define a non-quantized type.
  auto nqType3 = F->getParent()->uniqueType(ElemKind::FloatTy, {1, 3});
  // Define two variables of non-quantized types.
  auto *nqv3 = mod.createPlaceholder(nqType3, "V3", true);
  auto *nqv4 = mod.createPlaceholder(nqType3, "V4", true);
  // Create cmp nodes using non-quantized inputs.
  auto *cmpNode3 = F->createCmpEQ("cmpeq", nqv3, nqv4);
  auto *cmpNode4 = F->createCmpLTE("cmplte", nqv3, nqv4);
  // Check that the output type of cmp nodes is BoolKind.
  EXPECT_TRUE(cmpNode3->getResult().getElementType() == ElemKind::BoolTy);
  EXPECT_TRUE(cmpNode4->getResult().getElementType() == ElemKind::BoolTy);
}

/// Check that the users of value are equal to expectedUsers.
static bool
hasAllTheseUses(const llvm::SmallPtrSetImpl<const Node *> &expectedUsers,
                const NodeValue &value) {
  llvm::SmallPtrSet<const Node *, 4> uses;
  for (const NodeUse &use : value.getUsers()) {
    const Node *user = use.getUser();
    if (!expectedUsers.count(user)) {
      // We found a user that wasn't on the list.
      return false;
    }
    uses.insert(user);
  }
  return expectedUsers.size() == uses.size();
}

/// Check that our uses lists are correct for nodes with multiple results.
TEST(Graph, usesListsWithSeveralResult) {
  ExecutionEngine EE;
  PlaceholderBindings bindings;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {3, 32}, "input", true);
  auto *topK = F->createTopK("topK", input, 12);
  EXPECT_EQ(topK->getNumUsers(), 0);

  NodeValue values = topK->getValues();
  NodeValue indices = topK->getIndices();
  llvm::SmallPtrSet<const Node *, 4> savesOfValues;
  llvm::SmallPtrSet<const Node *, 4> savesOfIndices;

  EXPECT_EQ(indices.getNumUsers(), 0);
  EXPECT_EQ(values.getNumUsers(), 0);

  EXPECT_FALSE(indices.hasOneUse());
  EXPECT_FALSE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfIndices, indices));
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Now add a user to only one result of the topK node.
  savesOfValues.insert(F->createSave("saveValues1", values));

  // The whole node should inherit the uses of each of its results.
  EXPECT_EQ(topK->getNumUsers(), 1);

  // Each result should have its own use list.
  EXPECT_EQ(indices.getNumUsers(), 0);
  EXPECT_EQ(values.getNumUsers(), 1);

  EXPECT_FALSE(indices.hasOneUse());
  EXPECT_TRUE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfIndices, indices));
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Add a user to the other result of the topK node.
  savesOfIndices.insert(F->createSave("saveIndices1", indices));

  // The whole node should inherit the uses of each of its results.
  EXPECT_EQ(topK->getNumUsers(), 2);

  // Each result should have its own use list.
  EXPECT_EQ(indices.getNumUsers(), 1);
  EXPECT_EQ(values.getNumUsers(), 1);

  EXPECT_TRUE(indices.hasOneUse());
  EXPECT_TRUE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfIndices, indices));
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Add a couple more users of values and indices.
  // Interleaves the insertions in the uses list for both values and indices.
  savesOfValues.insert(F->createSave("saveValues2", values));
  savesOfValues.insert(F->createSave("saveValues3", values));
  savesOfIndices.insert(F->createSave("saveIndices2", indices));

  EXPECT_EQ(topK->getNumUsers(), 5);

  EXPECT_EQ(indices.getNumUsers(), 2);
  EXPECT_EQ(values.getNumUsers(), 3);

  EXPECT_FALSE(indices.hasOneUse());
  EXPECT_FALSE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfIndices, indices));
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));
}

/// Check that our uses lists are correct when accessed through
/// NodeValue.
TEST(Graph, usesListsThroughNodeValues) {
  ExecutionEngine EE;
  PlaceholderBindings bindings;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {3, 32}, "input", true);
  auto *reLU = F->createRELU("reLU", input);
  EXPECT_EQ(reLU->getNumUsers(), 0);

  NodeValue values = reLU->getResult();
  llvm::SmallPtrSet<const Node *, 4> savesOfValues;

  EXPECT_EQ(values.getNumUsers(), 0);

  EXPECT_FALSE(values.hasOneUse());

  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Now add a user to only one result of the reLU node.
  savesOfValues.insert(F->createSave("saveValues1", values));

  // The whole node should inherit the uses of each of its results.
  EXPECT_EQ(reLU->getNumUsers(), 1);

  // The NodeValue should match.
  EXPECT_EQ(values.getNumUsers(), 1);
  EXPECT_TRUE(values.hasOneUse());
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Add one more use.
  savesOfValues.insert(F->createSave("saveValues2", values));

  // The whole node should inherit the uses of each of its results.
  EXPECT_EQ(reLU->getNumUsers(), 2);

  EXPECT_EQ(values.getNumUsers(), 2);
  EXPECT_FALSE(values.hasOneUse());
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));

  // Add a couple more users.
  savesOfValues.insert(F->createSave("saveValues3", values));
  savesOfValues.insert(F->createSave("saveValues4", values));

  EXPECT_EQ(reLU->getNumUsers(), 4);

  EXPECT_EQ(values.getNumUsers(), 4);
  EXPECT_FALSE(values.hasOneUse());
  EXPECT_TRUE(hasAllTheseUses(savesOfValues, values));
}

/// Verify that the pre-order visitor works correctly.
TEST(Graph, PreOrderTest) {
  Module M;
  PlaceholderBindings bindings;
  auto *F = M.createFunction("main");

  auto *input1 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 10}, "input1", true);
  auto *input2 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 10}, "input2", true);
  SplatNode *zero = F->createSplat("zero", input1->getType(), 0.);
  MulNode *mul1 = F->createMul("mul1", zero, input1);
  MulNode *mul2 = F->createMul("mul2", zero, input2);
  MulNode *mul3 = F->createMul("mul3", mul1, mul2);
  SaveNode *ret1 = F->createSave("ret1", mul3);

  SplatNode *one = F->createSplat("one", input2->getType(), 1.0);
  AddNode *add1 = F->createAdd("add1", input2, one);
  AddNode *add2 = F->createAdd("add2", add1, one);
  AddNode *add3 = F->createAdd("add3", add2, one);
  SaveNode *ret2 = F->createSave("ret2", add2);

  GraphPreOrderVisitor visitor(*F);
  auto order = visitor.getPreOrder();

  ASSERT_EQ(order.size(), 14);
  EXPECT_EQ(order[0], ret1);
  EXPECT_EQ(order[1], mul3);
  EXPECT_EQ(order[2], mul1);
  EXPECT_EQ(order[3], zero);
  EXPECT_EQ(order[4], input1);
  EXPECT_EQ(order[5], mul2);
  EXPECT_EQ(order[6], input2);
  EXPECT_EQ(order[7], ret1->getOutput());
  EXPECT_EQ(order[8], add3);
  EXPECT_EQ(order[9], add2);
  EXPECT_EQ(order[10], add1);
  EXPECT_EQ(order[11], one);
  EXPECT_EQ(order[12], ret2);
  EXPECT_EQ(order[13], ret2->getOutput());
}

/// Verify that the post-order visitor works correctly.
TEST(Graph, PostOrderTest) {
  Module M;
  PlaceholderBindings bindings;
  auto *F = M.createFunction("main");

  auto *input1 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 10}, "input1", true);
  auto *input2 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 10}, "input2", true);
  SplatNode *zero = F->createSplat("zero", input1->getType(), 0.);
  MulNode *mul1 = F->createMul("mul1", zero, input1);
  MulNode *mul2 = F->createMul("mul2", zero, input2);
  MulNode *mul3 = F->createMul("mul3", mul1, mul2);
  SaveNode *ret1 = F->createSave("ret1", mul3);

  SplatNode *one = F->createSplat("one", input2->getType(), 1.0);
  AddNode *add1 = F->createAdd("add1", input2, one);
  AddNode *add2 = F->createAdd("add2", add1, one);
  AddNode *add3 = F->createAdd("add3", add2, one);
  SaveNode *ret2 = F->createSave("ret2", add2);

  GraphPostOrderVisitor visitor(*F);
  auto order = visitor.getPostOrder();

  ASSERT_EQ(order.size(), 14);
  EXPECT_EQ(order[0], zero);
  EXPECT_EQ(order[1], input1);
  EXPECT_EQ(order[2], mul1);
  EXPECT_EQ(order[3], input2);
  EXPECT_EQ(order[4], mul2);
  EXPECT_EQ(order[5], mul3);
  EXPECT_EQ(order[6], ret1->getOutput());
  EXPECT_EQ(order[7], ret1);
  EXPECT_EQ(order[8], one);
  EXPECT_EQ(order[9], add1);
  EXPECT_EQ(order[10], add2);
  EXPECT_EQ(order[11], add3);
  EXPECT_EQ(order[12], ret2->getOutput());
  EXPECT_EQ(order[13], ret2);
}

TEST(Graph, placeholder) {
  Module MD;
  PlaceholderBindings bindings;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Node *K =
      MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", false);
  Node *S = MD.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", false);

  K = F->createFullyConnected(bindings, "FC", K, 10);
  K = F->createRELU("Relu", K);
  K = F->createSoftMax("SoftMax", K, S);
  F->createSave("Save", K);
}

/// Check that the setType API allows to change the type of the
/// related result and only the related result.
TEST(Graph, setType) {
  Module M;
  auto *F = M.createFunction("main");

  const dim_t inputDims[] = {4, 10};
  const dim_t top5Dims[] = {4, 5};
  auto *input =
      M.createPlaceholder(ElemKind::FloatTy, inputDims, "input", true);
  TopKNode *topK = F->createTopK("add", input, 5);
  TypeRef origTopKRes0 = M.uniqueType(ElemKind::FloatTy, top5Dims);
  TypeRef origTopKRes1 = M.uniqueType(ElemKind::Int64ITy, top5Dims);

  EXPECT_EQ(topK->getType(TopKNode::ValuesIdx), origTopKRes0);
  EXPECT_EQ(topK->getType(TopKNode::IndicesIdx), origTopKRes1);

  // Modify the type of result 0 and make sure type 1 is not
  // affected. Similarly the input shouldn't be affected.
  TypeRef inputTy = M.uniqueType(ElemKind::FloatTy, inputDims);
  TypeRef topKRes0 = M.uniqueType(ElemKind::Float16Ty, top5Dims);
  topK->setType(TopKNode::ValuesIdx, topKRes0);
  EXPECT_EQ(input->getType(), inputTy);
  EXPECT_EQ(topK->getType(TopKNode::ValuesIdx), topKRes0);
  EXPECT_EQ(topK->getType(TopKNode::IndicesIdx), origTopKRes1);

  // Make sure the NodeValue API works the same way
  // as the Node::setType API.
  NodeValue valRes1 = topK->getNthResult(TopKNode::IndicesIdx);
  valRes1.setType(topKRes0);
  EXPECT_EQ(input->getType(), inputTy);
  EXPECT_EQ(topK->getType(TopKNode::ValuesIdx), topKRes0);
  EXPECT_EQ(topK->getType(TopKNode::IndicesIdx), topKRes0);
  EXPECT_EQ(valRes1.getType(), topKRes0);

  // Now restore sane types.
  NodeValue valRes0 = topK->getNthResult(TopKNode::ValuesIdx);
  valRes0.setType(origTopKRes0);
  topK->setType(TopKNode::IndicesIdx, origTopKRes1);
  EXPECT_EQ(input->getType(), inputTy);
  EXPECT_EQ(topK->getType(TopKNode::ValuesIdx), origTopKRes0);
  EXPECT_EQ(valRes0.getType(), origTopKRes0);
  EXPECT_EQ(topK->getType(TopKNode::IndicesIdx), origTopKRes1);
  EXPECT_EQ(valRes1.getType(), origTopKRes1);
}

/// Check that we fixed the bug with Function::eraseNode. This method used to
/// erase a node that was equal to the node we wanted to delete, which may be
/// two different entities.
/// To see this bug in action, we create a bunch of nodes with the same value.
/// Then we erase them in reserve order. This reserve ordering was actually
/// freeing the node in the original order, thus at some point we try to delete
/// a node that has already deleted and an assert (debug mode) or segmentation
/// fault (release would occur).
/// Note: Which node is actually freed depend on the implementation of
/// std::find, thus we cannot really predict when the bug occurs.
TEST(Graph, eraseNodeBug) {
  Module M;
  auto *F = M.createFunction("main");

  auto *input = M.createPlaceholder(ElemKind::FloatTy, {3, 2}, "input", true);
  std::vector<Node *> ReLUs;
  // Create a bunch of ReLUs.
  for (unsigned idx = 0; idx != 5; ++idx) {
    ReLUs.push_back(F->createRELU("relu", input));
  }
  // Check that we can erase all the nodes.
  for (int idx = 4; idx != -1; --idx) {
    F->eraseNode(ReLUs[idx]);
  }
  EXPECT_EQ(F->getNodes().size(), 0);
}

/// Verify that two Nodes with different predicates but the same inputs are not
/// considered equal.
TEST(Graph, nodeEqualityWithDifferentPredicates) {
  Module M;
  auto *F = M.createFunction("main");

  Node *in = M.createPlaceholder(ElemKind::FloatTy, {5}, "in", false);
  Node *pred1 = M.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);
  Node *pred2 = M.createPlaceholder(ElemKind::FloatTy, {1}, "pred", false);

  Node *RN1 = F->createRELU("relu1", in);
  RN1->setPredicate(pred1);

  Node *RN2 = F->createRELU("relu2", in);
  RN2->setPredicate(pred2);

  EXPECT_FALSE(RN1->isEqual(*RN2));
}

/// Check that verify doesn't allow for multiple writers to the same node.
TEST(Graph, verifyOneWriter) {
  Module M;
  auto *F = M.createFunction("main");

  auto *input = M.createPlaceholder(ElemKind::FloatTy, {5}, "input", false);
  auto *output = M.createPlaceholder(ElemKind::FloatTy, {5}, "output", false);
  F->createSave("Save1", input, output);
  F->createSave("Save2", input, output);

  EXPECT_FALSE(M.verify());
}

/// Check that verify doesn't allow for Constants to be written to. Note that
/// createSave() cannot do this as the API only accepts Placeholders to write
/// to, however it could happen during graph transformations, e.g. via
/// replaceAllUsesOfWith() as shown here.
TEST(Graph, verifyConstantNoWriters) {
  Module M;
  auto *F = M.createFunction("main");

  auto *input = M.createPlaceholder(ElemKind::FloatTy, {5}, "input", false);
  auto *outputPH = M.createPlaceholder(ElemKind::FloatTy, {5}, "outPH", false);
  F->createSave("save", input, outputPH);

  // Replace the output Placeholder with a Constant. This should fail
  // verification.
  auto *outputC = M.createConstant(ElemKind::FloatTy, {5}, "outC");
  NodeValue(outputPH).replaceAllUsesOfWith(outputC);

  EXPECT_FALSE(M.verify());
}

TEST(Graph, typeUnsafeReplaceAllUsesOfWith) {
  Module M;
  auto *F = M.createFunction("main");

  auto *LHS = M.createPlaceholder(ElemKind::FloatTy, {3, 4}, "A", false);
  auto *RHS = M.createPlaceholder(ElemKind::FloatTy, {4, 5}, "B", false);
  auto *FC = F->createMatMul("fc", LHS, RHS);
  F->createSave("save", FC);

  auto newLHS = M.createPlaceholder(ElemKind::FloatTy, {10, 10}, "A", false);
  LHS->getOutput().typeUnsafeReplaceAllUsesOfWith(newLHS);
}

/// Check that the verifier will complain if a constant and its
/// underlying tensor have mismatching types.
/// Here the constant is updated but not the tensor.
TEST(Graph, verifyConstantTensorTypeMatchesConstantTypeChanged) {
  Module M;

  auto *input = M.createConstant(ElemKind::FloatTy, {5}, "input");
  // Fresh constant should verify just fine.
  EXPECT_TRUE(input->verify());

  input->setType(Storage::OutputIdx, M.uniqueType(ElemKind::Float16Ty, {5}));

  EXPECT_FALSE(input->verify());
}

/// Check that the verifier will complain if a constant and its
/// underlying tensor have mismatching types.
/// Here the tensor is updated but not the constant.
TEST(Graph, verifyConstantTensorTypeMatchesTensorTypeChanged) {
  Module M;

  auto *input = M.createConstant(ElemKind::FloatTy, {5}, "input");
  // Fresh constant should verify just fine.
  EXPECT_TRUE(input->verify());
  input->getPayloadMutable().convertToType(ElemKind::Float16Ty);

  EXPECT_FALSE(input->verify());
}

/// Check that Constants backed by unowned Tensors are in fact unowned until
/// a mutable reference to their payload is obtained at which point the backing
/// Tensor is copied and becomes owned.
TEST(Graph, verifyConstantWithUnownedTensorCopiesOnWrite) {
  Module M;

  Tensor originalT(ElemKind::FloatTy, {3});
  Tensor unownedT = originalT.getUnowned({3});

  auto originalH = originalT.getHandle();

  for (size_t i = 0; i < originalT.size(); i++) {
    originalH.raw(i) = i;
  }

  // Both Tensors should have the same underlying memory because unownedT shares
  // originalT's memory.
  EXPECT_EQ(originalT.getUnsafePtr(), unownedT.getUnsafePtr());

  Constant *originalC = M.createConstant("original", std::move(originalT));
  Constant *unownedC = M.createConstant("unowned", std::move(unownedT));

  const Tensor &originalCT = originalC->getPayload();
  const Tensor &unownedCT = unownedC->getPayload();

  const auto originalCTH = originalCT.getHandle();
  const auto unownedCTH = unownedCT.getHandle();

  ASSERT_EQ(originalCTH.size(), unownedCTH.size());

  // Both Constants should have the same values because their Tensors have the
  // same underlying memory.
  for (size_t i = 0; i < originalCTH.size(); i++) {
    EXPECT_EQ(i, originalCTH.raw(i));
    EXPECT_EQ(i, unownedCTH.raw(i));
  }

  Tensor &originalCTM = originalC->getPayloadMutable();
  auto originalCTMH = originalCTM.getHandle();

  // Bump up the value in the original Constant, this change should be
  // reflected in the unowned Constant as well.
  for (size_t i = 0; i < originalCTMH.size(); i++) {
    originalCTMH.raw(i) += 1;
  }

  // After changing the values in the original Constant, we should see an update
  // in the values of the unowned Constant because they share the same
  // underlying memory.
  for (size_t i = 0; i < unownedCTH.size(); i++) {
    EXPECT_EQ(unownedCTH.raw(i), i + 1);
  }

  Tensor &unownedCTM = unownedC->getPayloadMutable();
  auto unownedCTMH = unownedCTM.getHandle();

  ASSERT_EQ(originalCTH.size(), unownedCTMH.size());

  // After getting a mutable reference to the unowned Constant's payload, the
  // underlying memory should have been copied but should still contain the same
  // values as it did previously at this point.
  EXPECT_NE(unownedCTM.getUnsafePtr(), originalCT.getUnsafePtr());
  for (size_t i = 0; i < unownedCTMH.size(); i++) {
    EXPECT_EQ(unownedCTMH.raw(i), i + 1);
  }

  // Bump up the value in the original Constant again, this change should not be
  // reflected in the unowned Constant now because at this point, after a
  // mutable reference to its payload has been obtained, it should have it's own
  // memory.
  for (size_t i = 0; i < originalCTMH.size(); i++) {
    originalCTMH.raw(i) += 1;
  }

  // Now that the unowned Constant's payload has been obtained as mutable, it
  // should have been copied and thus have its own memory and changes to the
  // original constant should not be reflected in the unowned Constant.
  for (size_t i = 0; i < unownedCTMH.size(); i++) {
    EXPECT_EQ(unownedCTMH.raw(i), i + 1);
  }
}

/// Check that hooking an intermediate node works.
TEST(Graph, hookTest) {
  Module mod;
  auto *F = mod.createFunction("main");
  auto *in = mod.createPlaceholder(ElemKind::FloatTy, {1}, "in", false);
  auto *relu1 = F->createRELU("relu1", in);
  auto *relu2 = F->createRELU("relu2", relu1);
  F->createSave("save", relu2);
  EXPECT_EQ(F->getNodes().size(), 3);
  EXPECT_EQ(mod.getPlaceholders().size(), 2);

  // Hook the first relu and verify that the hooked graph looks right.
  auto hooked = glow::hookNode(F, relu1);
  auto const &nodes = hooked.function->getNodes();
  ASSERT_EQ(mod.getPlaceholders().size(), 3);
  ASSERT_EQ(nodes.size(), 2);
  auto const *hookSave = *hooked.outputSaves.begin();
  ASSERT_TRUE(hookSave);
  auto *inp = llvm::dyn_cast<ReluNode>(hookSave->getInput());
  ASSERT_TRUE(inp);
  auto *ph = llvm::dyn_cast<Placeholder>(inp->getInput());
  ASSERT_TRUE(ph);
  ASSERT_EQ(ph, in);
}

/// Check that getConstantsSize returns the correct size of constants.
TEST(Graph, moduleSize) {
  Module mod;

  EXPECT_EQ(mod.getConstantsSize(), 0);

  auto *cons1 = mod.createConstant(ElemKind::FloatTy, {1}, "var");
  EXPECT_EQ(mod.getConstantsSize(), sizeof(float) * cons1->getPayload().size());

  auto *cons2 = mod.createConstant(ElemKind::FloatTy, {1, 32, 32, 16}, "var2");
  EXPECT_EQ(mod.getConstantsSize(),
            sizeof(float) + sizeof(float) * cons2->getPayload().size());
}

/// Check that getDataSize() returns the correct size of backing tensors.
TEST(Graph, contextSize) {
  Module mod;
  PlaceholderBindings bindings;

  Placeholder *PH =
      mod.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);

  EXPECT_EQ(bindings.getDataSize(), 0);
  bindings.allocate(PH);
  EXPECT_EQ(bindings.get(PH)->size(), 4 * 320 * 200 * 3);
  EXPECT_EQ(bindings.getDataSize(), sizeof(float) * bindings.get(PH)->size());
}

/// Check that clones of the context are distinct and share no references back
/// to the original object.
TEST(Graph, clonePlaceholderBindings) {
  Module mod;

  Placeholder *PH1 =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 4}, "PH1", false);

  PlaceholderBindings bindings1;
  bindings1.allocate(PH1);

  PlaceholderBindings bindings2 = bindings1.clone();

  Tensor *t1 = bindings1.get(PH1);
  Tensor *t2 = bindings2.get(PH1);

  EXPECT_NE(t1, nullptr);
  EXPECT_NE(t2, nullptr);
  EXPECT_NE(t1, t2);

  // The new PlaceholderBindings has no references back, and changing it does
  // not affect bindings1
  Placeholder *PH2 =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 4}, "PH2", false);

  bindings2.allocate(PH2);
  // now exists in bindings1 but not bindings2
  EXPECT_EQ(bindings1.get(PH2), nullptr);
  EXPECT_NE(bindings2.get(PH2), nullptr);

  // Likewise changing bindings1 does not affect bindings2
  bindings1.clear();
  EXPECT_EQ(bindings1.count(PH1), 0);
  EXPECT_EQ(bindings2.count(PH1), 1);

  // Adds are distinct
  Placeholder *PH3 =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 4}, "PH3", false);
  bindings1.allocate(PH3);
  bindings2.allocate(PH3);
  EXPECT_NE(bindings1.get(PH3), nullptr);
  EXPECT_NE(bindings2.get(PH3), nullptr);
  EXPECT_NE(bindings1.get(PH3), bindings2.get(PH3));
}

/// Check that running a function multiple times on cloned PlaceholderBindingss
/// have distinct outputs.
TEST(Graph, clonePlaceholderBindingsRuns) {
  ExecutionEngine EE;
  PseudoRNG PRNG;

  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 3}, "input", true);

  bindings.allocate(input);

  auto *FCL1 = F->createFullyConnected(bindings, "fc", input, 10);
  auto *RL3 = F->createRELU("relu4", FCL1);
  auto *save = F->createSave("ret", RL3);
  auto *savePH = save->getPlaceholder();

  bindings.allocate(save->getPlaceholder());

  // Compile once.
  EE.compile(CompilationMode::Infer);

  // Run with random inputs.
  inputs.getHandle<>().randomize(-3.0, 3.0, PRNG);
  updateInputPlaceholders(bindings, {input}, {&inputs});
  EE.run(bindings);

  // Clone the context.
  PlaceholderBindings bindings2 = bindings.clone();

  // PlaceholderBindingss are identical.
  Tensor *saveBacking1, *saveBacking2;
  saveBacking1 = bindings.get(savePH);
  saveBacking2 = bindings2.get(savePH);
  EXPECT_NE(saveBacking1, saveBacking2);
  EXPECT_EQ(saveBacking1->size(), saveBacking2->size());
  EXPECT_TRUE(saveBacking1->isEqual(*saveBacking2));

  // Run again with different random inputs using the cloned context.
  Tensor inputs2(ElemKind::FloatTy, {1, 32, 32, 3});
  inputs2.getHandle<>().randomize(-3.0, 3.0, PRNG);
  updateInputPlaceholders(bindings2, {input}, {&inputs2});
  EE.run(bindings2);

  // PlaceholderBindingss are no longer identical.
  EXPECT_EQ(saveBacking1->size(), saveBacking2->size());
  EXPECT_FALSE(saveBacking1->isEqual(*saveBacking2));
}

/// Check that using the indices enums in nodes works correctly, with
/// multi-input, multi-output, and single-input/output nodes.
TEST(Graph, TestNodeEnums) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  Placeholder *I =
      MD.createPlaceholder(ElemKind::FloatTy, {10, 10}, "input", true);
  Placeholder *O = MD.createPlaceholder(ElemKind::FloatTy, {3}, "output", true);

  TopKNode *TKN = F->createTopK("topk", I, 3);
  GatherNode *GN =
      F->createGather("gather", TKN->getValues(), TKN->getIndices());
  TanhNode *TN = F->createTanh("tanh", GN);
  SaveNode *SN = F->createSave("save", TN, O);

  // Check structure of Placeholders.
  EXPECT_EQ(I->getNthResult(Storage::OutputIdx), I->getOutput());
  EXPECT_EQ(O->getNthResult(Storage::OutputIdx), O->getOutput());

  // Check structure of TopK.
  EXPECT_EQ(TKN->getInput(), TKN->getNthInput(TopKNode::InputIdx));
  EXPECT_EQ(TKN->getNthResult(TopKNode::ValuesIdx), TKN->getValues());
  EXPECT_EQ(TKN->getNthResult(TopKNode::IndicesIdx), TKN->getIndices());

  // Check structure of Gather.
  EXPECT_EQ(GN->getNthInput(GatherNode::DataIdx), GN->getData());
  EXPECT_EQ(GN->getNthInput(GatherNode::IndicesIdx), GN->getIndices());
  EXPECT_EQ(GN->getNthResult(GatherNode::ResultIdx), GN->getResult());

  // Check structure of Tanh.
  EXPECT_EQ(TN->getNthInput(TanhNode::InputIdx), TN->getInput());
  EXPECT_EQ(TN->getNthResult(TanhNode::ResultIdx), TN->getResult());

  // Check structure of Save.
  EXPECT_EQ(SN->getNthInput(SaveNode::InputIdx), SN->getInput());
  EXPECT_EQ(SN->getNthInput(SaveNode::OutputIdx), SN->getOutput());

  // Check connection between Placeholder and TopK.
  EXPECT_EQ(TKN->getNthInput(TopKNode::InputIdx), I->getOutput());

  // Check connections between TopK and Gather.
  EXPECT_EQ(TKN->getNthResult(TopKNode::ValuesIdx),
            GN->getNthInput(GatherNode::DataIdx));
  EXPECT_EQ(TKN->getNthResult(TopKNode::IndicesIdx),
            GN->getNthInput(GatherNode::IndicesIdx));

  // Check connection between Gather and Tanh.
  EXPECT_EQ(GN->getNthResult(GatherNode::ResultIdx),
            TN->getNthInput(TanhNode::InputIdx));

  // Check connection between Gather and Tanh.
  EXPECT_EQ(TN->getNthResult(TanhNode::ResultIdx),
            SN->getNthInput(SaveNode::InputIdx));

  // Check connection between Gather and Tanh.
  EXPECT_EQ(SN->getNthInput(SaveNode::OutputIdx), O->getOutput());
}

/// Searched \p F for a single instance of a node of Kind T. If more than one is
/// found, \returns nullptr, otherwise returns the single instance.
template <class T> static T *findSingleInstanceOfNode(Function *F) {
  T *found = nullptr;
  for (auto &n : F->getNodes()) {
    if (auto *currNode = llvm::dyn_cast<T>(&n)) {
      if (found != nullptr) {
        return nullptr;
      }
      found = currNode;
    }
  }
  return found;
}

/// Check that group Conv is not lowered when specified to lower by backend if
/// doNotLowerKinds contains Conv.
TEST(Graph, GroupTestConvNoLower) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  PlaceholderBindings bindings;
  Node *K =
      MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 8}, "input", true);
  Node *S = MD.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", true);

  K = F->createConv(bindings, "Conv1", K, 16, 3, 2, 3, /* group */ 8);
  K = F->createRELU("Relu", K);
  K = F->createSoftMax("SoftMax", K, S);
  F->createSave("Save", K);
  F->dump();
  auto filePath = F->dumpDAG();
  auto backend = MockBackend();

  {
    // Before we lower, we should have a single Conv node with group = 8.
    ConvolutionNode *CN = findSingleInstanceOfNode<ConvolutionNode>(F);
    if (!CN) {
      llvm::sys::fs::remove(filePath);
    }
    ASSERT_TRUE(CN);
    EXPECT_EQ(CN->getGroup(), 8);
  }

  // Now lower, but prevent ConvolutionNodeKinds from being lowered.
  KindSet doNotLower;
  doNotLower.insert(Kinded::Kind::ConvolutionNodeKind);
  CompilationContext cctx;
  lower(F, cctx, &backend, doNotLower);

  {
    // Now have lowered but should still have a single Conv node with group = 8.
    ConvolutionNode *CN = findSingleInstanceOfNode<ConvolutionNode>(F);
    if (!CN) {
      llvm::sys::fs::remove(filePath);
    }
    ASSERT_TRUE(CN);
    EXPECT_EQ(CN->getGroup(), 8);
  }
}

/// Check that getOutputSave returns SaveNode object for the correct Placeholder
/// and nullptr in other cases.
TEST(Graph, GetOutputSaveTest) {
  Module MD;
  Function *F = MD.createFunction("F");
  PlaceholderBindings bindings;
  Placeholder *I =
      MD.createPlaceholder(ElemKind::FloatTy, {10, 10}, "input", true);
  Placeholder *O = MD.createPlaceholder(ElemKind::FloatTy, {3}, "output", true);
  TopKNode *TKN = F->createTopK("topk", I, 3);
  GatherNode *GN =
      F->createGather("gather", TKN->getValues(), TKN->getIndices());
  TanhNode *TN = F->createTanh("tanh", GN);
  SaveNode *SN = F->createSave("save", TN, O);

  // Check the return value of getOutputSave method.
  // Placeholder parent is null.
  auto *FoundNode = glow::getOutputSave(F, O);
  EXPECT_NE(nullptr, FoundNode);
  EXPECT_EQ(SN, FoundNode);

  // Placeholder parent is set to the correct value.
  O->setParent(F);
  EXPECT_EQ(F, O->getParent());
  FoundNode = glow::getOutputSave(F, O);
  EXPECT_NE(nullptr, FoundNode);
  EXPECT_EQ(SN, FoundNode);

  // Invalid placeholder type is provided.
  EXPECT_EQ(nullptr, glow::getOutputSave(F, I));

  // Save belongs to a different function
  Function *F2 = MD.createFunction("F2");
  TopKNode *TKN2 = F2->createTopK("topk", I, 3);
  GatherNode *GN2 =
      F2->createGather("gather", TKN2->getValues(), TKN2->getIndices());
  TanhNode *TN2 = F2->createTanh("tanh", GN2);
  SaveNode *SN2 = F2->createSave("save", TN2, O);

  FoundNode = glow::getOutputSave(F, O);
  EXPECT_NE(nullptr, FoundNode);
  EXPECT_EQ(SN, FoundNode);

  O->setParent(F2);
  FoundNode = glow::getOutputSave(F2, O);
  EXPECT_NE(nullptr, FoundNode);
  EXPECT_EQ(SN2, FoundNode);
}

/// Check if dump functions work for Node, Function and Module.
TEST(Graph, testDumpStructure) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  PlaceholderBindings bindings;
  Node *K = MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 100, 3},
                                 "input", true);
  // Test Node
  std::string storageN1;
  llvm::raw_string_ostream osN1(storageN1);
  K->dump(osN1);
  std::string mesN = K->toString();
  std::string expectMes = R"(Placeholder
Name : input
Layout : *
Output : float<4 x 320 x 200 x 100 x 3>
Trainable : 1
Static : 0
Users : 0
)";
  EXPECT_EQ(mesN, expectMes);
  EXPECT_EQ(mesN, osN1.str());
  std::string storageN2;
  llvm::raw_string_ostream osN2(storageN2);
  osN2 << K;
  EXPECT_EQ(mesN, osN2.str());
  // Test Function
  Placeholder *I =
      MD.createPlaceholder(ElemKind::FloatTy, {10, 10}, "input", true);
  I->setStatic(true);
  Function *F2 = MD.createFunction("F2");
  F2->createTopK("topk", I, 3);
  std::string storageF1;
  llvm::raw_string_ostream osF1(storageF1);
  F2->dump(osF1);
  std::string mesF = F2->toString();
  std::string expectMesF = R"(Graph structure F2:
TopK
Name : topk
Input : float<10 x 10>
K : 3
Users : 0
Values : float<10 x 3>
Indices : index64<10 x 3>
Placeholder
Name : input__1
Layout : *
Output : float<10 x 10>
Trainable : 1
Static : 1
Users : 1
)";
  EXPECT_EQ(mesF, expectMesF);
  EXPECT_EQ(mesF, osF1.str());
  std::string storageF2;
  llvm::raw_string_ostream osF2(storageF2);
  osF2 << F2;
  EXPECT_EQ(mesF, osF2.str());
  storageF1.clear();
  F2->dump(osF1, /* skipUsersForStorage */ true);
  mesF = F2->toString(/* skipUsersForStorage */ true);
  expectMesF = R"(Graph structure F2:
TopK
Name : topk
Input : float<10 x 10>
K : 3
Users : 0
Values : float<10 x 3>
Indices : index64<10 x 3>
Placeholder
Name : input__1
Layout : *
Output : float<10 x 10>
Trainable : 1
Static : 1
)";
  EXPECT_EQ(mesF, expectMesF);
  EXPECT_EQ(mesF, osF1.str());
  // Test Module
  MD.createConstant(ElemKind::FloatTy, {1, 1}, "dummy");
  std::string storageM1;
  llvm::raw_string_ostream osM1(storageM1);
  MD.dump(osM1);
  std::string mesM = MD.toString();
  std::string expectMesM = R"(Module structure:
Constant
Name : dummy
Layout : *
Output : float<1 x 1>
Users : 0

Placeholder
Name : input__1
Layout : *
Output : float<10 x 10>
Trainable : 1
Static : 1
Users : 1

Placeholder
Name : input
Layout : *
Output : float<4 x 320 x 200 x 100 x 3>
Trainable : 1
Static : 0
Users : 0

Function : F2
Function : F
)";
  EXPECT_EQ(mesM, expectMesM);
  EXPECT_EQ(mesM, osM1.str());
  std::string storageM2;
  llvm::raw_string_ostream osM2(storageM2);
  osM2 << MD;
  EXPECT_EQ(mesM, osM2.str());
}

/// Initialize tensor payload for testing purposes. The value at index i is set
/// to i.
template <typename ElemTy> static void initTensor(Tensor &T) {
  Handle<ElemTy> handle = T.getHandle<ElemTy>();
  float val = 0;
  for (auto &elem : handle) {
    elem = val;
    val += 1.0;
  }
}

// Test that randomizing Constants in a Function works.
TEST(Graph, testRandomizeConstants) {
  Module MD;
  Function *F = MD.createFunction("F");

  // Create tensors to be used in Constants
  Tensor floatT(ElemKind::FloatTy, {10});
  initTensor<float>(floatT);

  Tensor halfT(ElemKind::Float16Ty, {10});
  initTensor<float16_t>(halfT);

  Tensor bfloat16T(ElemKind::BFloat16Ty, {10});
  initTensor<bfloat16_t>(bfloat16T);

  Tensor int8QT(ElemKind::Int8QTy, {10}, 1.0, 0);
  initTensor<int8_t>(int8QT);

  Tensor uint8QT(ElemKind::UInt8QTy, {10}, 1.0, 0);
  initTensor<uint8_t>(uint8QT);

  Tensor int16QT(ElemKind::Int16QTy, {10}, 1.0, 0);
  initTensor<int16_t>(int16QT);

  Tensor int32QT(ElemKind::Int32QTy, {10}, 1.0, 0);
  initTensor<int32_t>(int32QT);

  Tensor int32IT(ElemKind::Int32ITy, {10});
  initTensor<int32_t>(int32IT);

  Tensor int64IT(ElemKind::Int64ITy, {10});
  initTensor<int64_t>(int64IT);

  Tensor uint8FusedQT(ElemKind::UInt8FusedQTy, {16, 16}, 1.0, 0);
  initTensor<uint8_t>(uint8FusedQT);

  Tensor uint8FusedFP16QT(ElemKind::UInt8FusedFP16QTy, {16, 16}, 1.0, 0);
  initTensor<uint8_t>(uint8FusedFP16QT);

  Tensor uint4FusedFP16QT(ElemKind::UInt4FusedFP16QTy, {16, 16}, 1.0, 0);
  initTensor<uint8_t>(uint4FusedFP16QT);

  Tensor boolT(ElemKind::BoolTy, {10});
  initTensor<bool>(boolT);

  // Create Constants and use them in F
  auto *floatC = MD.createConstant("floatC", floatT);
  F->createAdd("add", floatC, floatC);

  auto *halfC = MD.createConstant("halfC", halfT);
  F->createAdd("add", halfC, halfC);

  auto *bfloat16C = MD.createConstant("bloat16C", bfloat16T);
  F->createAdd("add", bfloat16C, bfloat16C);

  auto *int8QC = MD.createConstant("int8QC", int8QT);
  F->createAdd("add", int8QC, int8QC);

  auto *uint8QC = MD.createConstant("uint8QC", uint8QT);
  F->createAdd("add", uint8QC, uint8QC);

  auto *int16QC = MD.createConstant("int16QC", int16QT);
  F->createAdd("add", int16QC, int16QC);

  auto *int32QC = MD.createConstant("int32QC", int32QT);
  F->createAdd("add", int32QC, int32QC);

  auto *int32IC = MD.createConstant("int32IC", int32IT);
  F->createAdd("add", int32IC, int32IC);

  auto *int64IC = MD.createConstant("int64IC", int64IT);
  F->createAdd("add", int64IC, int64IC);

  auto *uint8FusedQC = MD.createConstant("uint8FusedQC", uint8FusedQT);
  F->createAdd("add", uint8FusedQC, uint8FusedQC);

  auto *uint8FusedFP16QC =
      MD.createConstant("uint8FusedFP16QC", uint8FusedFP16QT);
  F->createAdd("add", uint8FusedFP16QC, uint8FusedFP16QC);

  auto *uint4FusedFP16QC =
      MD.createConstant("uint4FusedFP16QC", uint4FusedFP16QT);
  F->createAdd("add", uint4FusedFP16QC, uint4FusedFP16QC);

  auto *boolC = MD.createConstant("boolC", boolT);
  F->createAdd("add", boolC, boolC);

  // Randomize Constants in F
  F->randomizeConstants();

  // Check that no Constant is the same as what it started as
  EXPECT_FALSE(floatT.isEqual(floatC->getPayload()));
  EXPECT_FALSE(halfT.isEqual(halfC->getPayload()));
  EXPECT_FALSE(bfloat16T.isEqual(bfloat16C->getPayload()));
  EXPECT_FALSE(int8QT.isEqual(int8QC->getPayload()));
  EXPECT_FALSE(uint8QT.isEqual(uint8QC->getPayload()));
  EXPECT_FALSE(int16QT.isEqual(int16QC->getPayload()));
  EXPECT_FALSE(int32QT.isEqual(int32QC->getPayload()));
  EXPECT_FALSE(int32IT.isEqual(int32IC->getPayload()));
  EXPECT_FALSE(int64IT.isEqual(int64IC->getPayload()));
  EXPECT_FALSE(uint8FusedQT.isEqual(uint8FusedQC->getPayload()));
  EXPECT_FALSE(uint8FusedFP16QT.isEqual(uint8FusedFP16QC->getPayload()));
  EXPECT_FALSE(uint4FusedFP16QT.isEqual(uint4FusedFP16QC->getPayload()));
  EXPECT_FALSE(boolT.isEqual(boolC->getPayload()));
}

TEST(Graph, testSoftmaxMultiplier) {
  glow::ExecutionEngine EE;
  Module &M = EE.getModule();
  Function *F = M.createFunction("F");

  float beta = 2.0;

  // Create a graph with single softmax.
  auto *inputPH =
      M.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input", false);
  auto *select =
      M.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "select", true);
  Node *softmaxNode =
      F->createSoftMax("softmax", inputPH, select, inputPH->getType(), beta);
  auto *saveNode = F->createSave("output", softmaxNode);

  PlaceholderBindings bindings;
  auto *inputT = bindings.allocate(inputPH);
  inputT->getHandle() = {1.0, 2.0};
  Tensor expectedT(inputT->getType());
  expectedT.getHandle() = {0.11920292, 0.88079703};
  auto *outputT = bindings.allocate(saveNode->getPlaceholder());
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  EXPECT_TRUE(outputT->isEqual(expectedT));
}

TEST(Graph, testLogSoftmaxMultiplier) {
  glow::ExecutionEngine EE;
  Module &M = EE.getModule();
  Function *F = M.createFunction("F");

  float beta = 2.0;

  // Create a graph with single softmax.
  auto *inputPH =
      M.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input", false);
  auto *select =
      M.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "select", true);
  Node *softmaxNode =
      F->createLogSoftMax("softmax", inputPH, select, inputPH->getType(), beta);
  auto *saveNode = F->createSave("output", softmaxNode);

  PlaceholderBindings bindings;
  auto *inputT = bindings.allocate(inputPH);
  inputT->getHandle() = {1.0, 2.0};
  Tensor expectedT(inputT->getType());
  expectedT.getHandle() = {-2.1269, -0.1269};
  auto *outputT = bindings.allocate(saveNode->getPlaceholder());
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  EXPECT_TRUE(outputT->isEqual(expectedT));
}
