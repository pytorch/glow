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

#include "glow/Graph/Graph.h"
#include "BackendTestUtils.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"

#include "llvm/ADT/SmallPtrSet.h"

#include "gtest/gtest.h"

using namespace glow;

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

TEST(Graph, simpleTestConv) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Context ctx;
  Node *K =
      MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);
  Node *S = MD.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", true);

  K = F->createConv(ctx, "Conv1", K, 16, 3, 2, 3, 1);
  K = F->createRELU("Relu", K);
  K = F->createSoftMax("SoftMax", K, S);
  F->createSave("Save", K);
  F->dump();
  F->dumpDAG();
  lower(F, MockBackend());
  ::optimize(F, CompilationMode::Train);
  M.generateIR();
  M.dump();
  EXPECT_GT(M.getInstrs().size(), 0);
}

/// Check that we can create convolution with float16.
TEST(Graph, float16Conv) {
  Module MD;
  Function *F = MD.createFunction("F");
  Context ctx;
  Node *K = MD.createConstant(ElemKind::Float16Ty, {4, 320, 200, 3}, "input");

  auto *conv = F->createConv(ctx, "Conv", K, 16, 3, 2, 3, 1);
  conv->verify();
  EXPECT_EQ(conv->getType(0)->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(conv->getFilter().getType()->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(conv->getBias().getType()->getElementType(), ElemKind::Float16Ty);

  lower(F, MockBackend());

  IRFunction M(F);

  M.generateIR();
  EXPECT_GT(M.getInstrs().size(), 0);
  auto convIt = std::find_if(M.getInstrs().begin(), M.getInstrs().end(),
                             [](const Instruction &inst) -> bool {
                               return llvm::isa<ConvolutionInst>(inst);
                             });
  ASSERT_TRUE(convIt != M.getInstrs().end());
  const auto *convInst = llvm::cast<ConvolutionInst>(&*convIt);
  EXPECT_EQ(convInst->getSrc()->getType()->getElementType(),
            ElemKind::Float16Ty);
  EXPECT_EQ(convInst->getFilter()->getType()->getElementType(),
            ElemKind::Float16Ty);
  EXPECT_EQ(convInst->getBias()->getType()->getElementType(),
            ElemKind::Float16Ty);
}

/// Check that we can create batchNorm with float16.
TEST(Graph, float16BatchNorm) {
  Module MD;
  Function *F = MD.createFunction("F");
  Context ctx;
  auto *input =
      MD.createPlaceholder(ElemKind::Float16Ty, {1, 10, 20, 3}, "input", false);
  BatchNormalizationNode *BN =
      F->createBatchNormalization(ctx, "batch", input, 3, 0.0001, 0.9);

  BN->verify();
  EXPECT_EQ(BN->getType(0)->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(BN->getScale().getType()->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(BN->getBias().getType()->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(BN->getMean().getType()->getElementType(), ElemKind::Float16Ty);
  EXPECT_EQ(BN->getVar().getType()->getElementType(), ElemKind::Float16Ty);

  lower(F, MockBackend());

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

/// Test that our use lists are correctly reflecting the state of the IR
/// and in particular that it is not polluted by temporary variable.
TEST(Graph, useList) {
  Module MD;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Context ctx;
  auto *K =
      MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);

  EXPECT_EQ(K->getNumUsers(), 0);

  ConvolutionNode *conv = F->createConv(ctx, "Conv1", K, 16, 3, 2, 3, 1);

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

  Context ctx;
  ConvolutionNode *conv1 = F->createConv(ctx, "Conv1", K, 16, 3, 2, 3, 1);
  ConvolutionNode *conv2 = F->createConv(ctx, "Conv2", K, 16, 3, 2, 3, 1);
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

  Context ctx;
  Node *O = F->createFullyConnected(ctx, "FC1", A, 6);
  O = F->createRELU("RELU1", O);
  O = F->createFullyConnected(ctx, "FC2", O, 1);
  O = F->createRELU("RELU2", O);
  O = F->createRegression("Regression", O, Ex);
  F->createSave("Save", O);
  F->dump();
  F->dumpDAG();
  lower(F, MockBackend());
  ::optimize(F, CompilationMode::Train);
  M.generateIR();
  M.dump();
  EXPECT_GT(M.getInstrs().size(), 0);
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
  Context ctx;
  F->createSave("save", quantize);

  // Multiple nodes read from the same variable.
  // Only one Quantization Profile node should be created for the output
  // from the variable.
  Node *O = F->createFullyConnected(ctx, "FC1", A, 6);
  Node *C = F->createFullyConnected(ctx, "FC2", A, 6);
  O = F->createRELU("RELU1", O);
  F->createSave("save", O);
  F->createSave("save", C);

  // Simulate actual usage.
  ::optimize(F, CompilationMode::Infer);
  F = ::glow::profileQuantization(ctx, F);
  lower(F, MockBackend());
  ::optimize(F, CompilationMode::Infer);

  size_t numberOfProfileNodes =
      std::count_if(F->getNodes().begin(), F->getNodes().end(), [](Node &node) {
        return llvm::isa<QuantizationProfileNode>(&node);
      });

  EXPECT_EQ(10, numberOfProfileNodes);
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
  std::array<size_t, 4> filterDim = {{depth, kernels[0], kernels[1], 3}};
  auto *filter =
      MD.createPlaceholder(ElemKind::Int8QTy, filterDim, 3.3, 4, "F", true);
  auto *bias =
      MD.createPlaceholder(ElemKind::Int8QTy, {depth}, 1.3, 5, "B", true);

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvPoolOutputDims(width, width, kernels, steps, pads);
  std::array<size_t, 4> outDims = {{1, outSz.first, outSz.second, 16}};
  auto t = F->getParent()->uniqueType(glow::ElemKind::Int8QTy, outDims, 1.5, 6);

  auto *conv =
      F->createConv("conv", input, filter, bias, t, kernels, steps, pads, 1);

  auto s = conv->getResult().getType()->size();
  auto *fcFilter =
      MD.createPlaceholder(ElemKind::Int8QTy, {s, 6}, 0.4, 2, "F", true);
  auto *fcBias =
      MD.createPlaceholder(ElemKind::Int8QTy, {6}, 0.4, 2, "B", true);
  Node *O = F->createFullyConnected("fc1", conv, fcFilter, fcBias);
  Context ctx;
  F->createSave("ret", O);
  EE.compile(CompilationMode::Infer, F, ctx);
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

  auto *D = F->createDequantize("dequantize", A);
  Context ctx;
  F->createSave("ret", D);
  EE.compile(CompilationMode::Infer, F, ctx);
}

TEST(Graph, quantizeGather) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::Int8QTy, {2, 2}, 0.4, 2, "input", true);
  auto *indices = mod.createPlaceholder(ElemKind::Int64ITy, {1}, "index", true);
  auto *gather = F->createGather("gather", input, indices);
  Context ctx;
  F->createSave("ret", gather);
  EE.compile(CompilationMode::Infer, F, ctx);
}

TEST(Graph, cloneTest) {
  Module M;
  Context ctx;

  Function *F = M.createFunction("main");
  Node *K =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);
  Node *S = M.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", true);
  Node *conv = F->createConv(ctx, "Conv1", K, 16, 3, 2, 3, 1);
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
  auto F1 = M.createFunction("one");
  auto F2 = M.createFunction("two");
  auto V1 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V1", true);
  auto V2 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V2", true);
  auto V3 =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V3", true);
  M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "V4", true);

  Context ctx;
  auto sum = F1->createSub("1_sub_2", V1, V2);
  F1->createSave("sv", sum, V1);
  F2->createSave("sv", V3, V2);

  EXPECT_TRUE(M.hasFunction("one"));
  EXPECT_TRUE(M.hasFunction("two"));
  EXPECT_FALSE(M.hasFunction("four"));
  M.dumpDAG();
}

TEST(Graph, cloneTest2) {
  Module M;
  Context ctx;

  auto *F = M.createFunction("main");
  Node *K =
      M.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", true);
  Node *S = M.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", true);
  Node *conv = F->createConv(ctx, "Conv1", K, 16, 3, 2, 3, 1);
  Node *relu = F->createRELU("Relu", conv);
  Node *concat = F->createConcat("concat", {relu, relu, relu}, 0);

  Node *SM = F->createSoftMax("SoftMax", concat, S);
  F->createSave("Save", SM);

  auto *newF = F->clone("new_main");
  newF->verify();
  F->dump();
  newF->dump();

  EXPECT_EQ(newF->getNodes().size(), F->getNodes().size());
  EXPECT_EQ(newF->getParent(), F->getParent());
}

TEST(Graph, NodeValue) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Context ctx;
  auto *inputX = mod.createPlaceholder(ElemKind::FloatTy, {1}, "input", true);
  ctx.allocate(inputX)->init(Tensor::InitKind::Broadcast, 3.0, mod.getPRNG());

  NodeValue a = F->createAdd("x2", inputX, inputX);
  a = F->createAdd("x4", a, a);
  a = F->createAdd("x8", a, a);
  auto *S = F->createSave("Save", a);
  auto *res = ctx.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run();

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
  Context ctx;
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 3}, "input", true);
  auto *ex = mod.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "exp", true);
  auto *pred =
      mod.createPlaceholder(ElemKind::Int64ITy, {1}, "predicate", false);
  ctx.allocate(input);
  ctx.allocate(ex);
  ctx.allocate(pred);

  auto *CV0 = F->createConv(ctx, "conv1", input, 16, 5, 1, 2, 1);
  auto *RL0 = F->createRELU("relu1", CV0);
  auto *MP0 = F->createMaxPool("pool1", RL0, 2, 2, 0);

  CV0->setPredicate(pred);
  RL0->setPredicate(pred);
  MP0->setPredicate(pred);

  auto *FCL1 = F->createFullyConnected(ctx, "fc", MP0, 10);
  auto *RL3 = F->createRELU("relu4", FCL1);
  auto *SM = F->createSoftMax("sm", RL3, ex);
  auto *save = F->createSave("ret", SM);
  ctx.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer, F, ctx);

  updateVariables(ctx, {input}, {&inputs});
  EE.run();
}

// Return the number of ConvolutionNode after lower.
unsigned getConvNodeSize(BackendKind kind) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  Context ctx;
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 32}, "input", true);
  ConvolutionNode *CN = F->createConv(ctx, "conv", input, 6, 1, 1, 0, 2);
  F->createSave("save", CN);

  std::unique_ptr<Backend> backend(createBackend(kind));
  lower(F, *backend);

  unsigned count = 0;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == Kinded::Kind::ConvolutionNodeKind) {
      count++;
    }
  }

  if (kind == BackendKind::Interpreter) {
    EXPECT_EQ(count, 1);
  }

  return count;
}

// Check the unrolling grouped convolution opt status:
// -- disabled for Interpreter, CPU and OpenCL backend,
TEST(Graph, disableUnrollingGroupConv) {
  unsigned numberOfNodesInterpreter = getConvNodeSize(BackendKind::Interpreter);
  (void)numberOfNodesInterpreter;

#ifdef GLOW_WITH_CPU
  unsigned numberOfNodesCPU = getConvNodeSize(BackendKind::CPU);
  EXPECT_EQ(numberOfNodesCPU, numberOfNodesInterpreter);
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
  unsigned numberOfNodesOpenCL = getConvNodeSize(BackendKind::OpenCL);
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

  Context ctx;
  ctx.allocate(A)->init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  ctx.allocate(B)->init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  ctx.allocate(zero)->init(Tensor::InitKind::Broadcast, 0.0, mod.getPRNG());

  auto *addAB = F->createAdd("addAB", A, B);

  auto *saveNode = F->createSave("ret", addAB);
  ctx.allocate(saveNode->getPlaceholder());
  F->createSave("resetA", zero, A);

  // Copy the value of A.
  Tensor AOrig = ctx.get(A)->clone();

  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run();
  auto *ret = ctx.get(saveNode->getPlaceholder());
  auto handleAOrig = AOrig.getHandle<>();
  auto handleB = ctx.get(B)->getHandle<>();
  auto handleRet = ret->getHandle<>();
  bool allEqual = true;
  for (unsigned row = 0; row != 3; ++row) {
    for (unsigned column = 0; column != 32; ++column) {
      allEqual &= handleAOrig.at({row, column}) + handleB.at({row, column}) ==
                  handleRet.at({row, column});
    }
  }
  EXPECT_TRUE(ctx.get(A)->isEqual(*ctx.get(zero), 0.0));
  EXPECT_TRUE(allEqual);
}

/// Same as schedulingOfSavesOrderProvided except the order in which the nodes
/// are added to the function don't form a valid schedule.
/// In other words, the scheduler won't get away with scheduling
/// using only the order of the nodes in the list of nodes.
TEST(Graph, schedulingOfSaves) {
  ExecutionEngine EE;
  Context ctx;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {3, 32}, "A", true);
  auto *B = mod.createPlaceholder(A->getType(), "B", true);
  auto *zero = mod.createPlaceholder(A->getType(), "zero", true);
  F->createSave("resetA", zero, A);

  ctx.allocate(A)->init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  ctx.allocate(B)->init(Tensor::InitKind::Xavier, 1.0, mod.getPRNG());
  ctx.allocate(zero)->init(Tensor::InitKind::Broadcast, 0.0, mod.getPRNG());

  auto *addAB = F->createAdd("addAB", A, B);

  auto *saveNode = F->createSave("ret", addAB);
  ctx.allocate(saveNode->getPlaceholder());

  // Copy the value of A.
  Tensor AOrig = ctx.get(A)->clone();

  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run();
  auto *ret = saveNode->getPlaceholder();
  auto handleAOrig = AOrig.getHandle<>();
  auto handleB = ctx.get(B)->getHandle<>();
  auto handleRet = ctx.get(ret)->getHandle<>();
  bool allEqual = true;
  for (unsigned row = 0; row != 3; ++row) {
    for (unsigned column = 0; column != 32; ++column) {
      allEqual &= handleAOrig.at({row, column}) + handleB.at({row, column}) ==
                  handleRet.at({row, column});
    }
  }
  EXPECT_TRUE(ctx.get(A)->isEqual(*ctx.get(zero), 0.0));
  EXPECT_TRUE(allEqual);
}

/// Check that the parent link is properly updated while tweaking
/// nodes and their function.
TEST(Graph, parentLink) {
  ExecutionEngine EE;

  auto &mod = EE.getModule();
  Constant *V = new Constant("V", mod.uniqueType(ElemKind::FloatTy, {3, 32}));

  // Variables don't belong to any function...
  EXPECT_EQ(V->getParent(), nullptr);
  // Even when we create them from a module...
  Constant *V2 = mod.createConstant(V->getType(), "V2");
  EXPECT_EQ(V2->getParent(), nullptr);
  // Or add them to a module.
  mod.addConstant(V);
  EXPECT_EQ(V->getParent(), nullptr);

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
  // Check that the output type of cmp nodes is quantized, has scale 1.0 and
  // offset 0.
  EXPECT_TRUE(cmpNode1->getResult().getType()->isQuantizedType());
  EXPECT_EQ(cmpNode1->getResult().getType()->getScale(), 1.0);
  EXPECT_EQ(cmpNode1->getResult().getType()->getOffset(), 0);
  EXPECT_TRUE(cmpNode2->getResult().getType()->isQuantizedType());
  EXPECT_EQ(cmpNode2->getResult().getType()->getScale(), 1.0);
  EXPECT_EQ(cmpNode2->getResult().getType()->getOffset(), 0);

  // Define a non-quantized type.
  auto nqType3 = F->getParent()->uniqueType(ElemKind::FloatTy, {1, 3});
  // Define two variables of non-quantized types.
  auto *nqv3 = mod.createPlaceholder(nqType3, "V3", true);
  auto *nqv4 = mod.createPlaceholder(nqType3, "V4", true);
  // Create cmp nodes using non-quantized inputs.
  auto *cmpNode3 = F->createCmpEQ("cmpeq", nqv3, nqv4);
  auto *cmpNode4 = F->createCmpLTE("cmplte", nqv3, nqv4);
  // Check that output of cmp nodes is a non-quantized type matching the type of
  // inputs.
  EXPECT_FALSE(cmpNode3->getResult().getType()->isQuantizedType());
  EXPECT_EQ(cmpNode3->getResult().getType(), nqv3->getType());
  EXPECT_FALSE(cmpNode4->getResult().getType()->isQuantizedType());
  EXPECT_EQ(cmpNode4->getResult().getType(), nqv3->getType());
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
  Context ctx;

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
  Context ctx;

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
  Context ctx;
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
  Context ctx;
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
  Context ctx;
  Function *F = MD.createFunction("F");
  IRFunction M(F);
  Node *K =
      MD.createPlaceholder(ElemKind::FloatTy, {4, 320, 200, 3}, "input", false);
  Node *S = MD.createPlaceholder(ElemKind::Int64ITy, {4, 1}, "select", false);

  K = F->createFullyConnected(ctx, "FC", K, 10);
  K = F->createRELU("Relu", K);
  K = F->createSoftMax("SoftMax", K, S);
  F->createSave("Save", K);
}

/// Check that the setType API allows to change the type of the
/// related result and only the related result.
TEST(Graph, setType) {
  Module M;
  auto *F = M.createFunction("main");

  const size_t inputDims[] = {4, 10};
  const size_t top5Dims[] = {4, 5};
  auto *input =
      M.createPlaceholder(ElemKind::FloatTy, inputDims, "input", true);
  TopKNode *topK = F->createTopK("add", input, 5);
  TypeRef origTopKRes0 = M.uniqueType(ElemKind::FloatTy, top5Dims);
  TypeRef origTopKRes1 = M.uniqueType(ElemKind::Int64ITy, top5Dims);

  EXPECT_EQ(topK->getType(0), origTopKRes0);
  EXPECT_EQ(topK->getType(1), origTopKRes1);

  // Modify the type of result 0 and make sure type 1 is not
  // affected. Similarly the input shouldn't be affected.
  TypeRef inputTy = M.uniqueType(ElemKind::FloatTy, inputDims);
  TypeRef topKRes0 = M.uniqueType(ElemKind::Float16Ty, top5Dims);
  topK->setType(0, topKRes0);
  EXPECT_EQ(input->getType(), inputTy);
  EXPECT_EQ(topK->getType(0), topKRes0);
  EXPECT_EQ(topK->getType(1), origTopKRes1);

  // Make sure the NodeValue API works the same way
  // as the Node::setType API.
  NodeValue valRes1 = topK->getNthResult(1);
  valRes1.setType(topKRes0);
  EXPECT_EQ(input->getType(), inputTy);
  EXPECT_EQ(topK->getType(0), topKRes0);
  EXPECT_EQ(topK->getType(1), topKRes0);
  EXPECT_EQ(valRes1.getType(), topKRes0);

  // Now restore sane types.
  NodeValue valRes0 = topK->getNthResult(0);
  valRes0.setType(origTopKRes0);
  topK->setType(1, origTopKRes1);
  EXPECT_EQ(input->getType(), inputTy);
  EXPECT_EQ(topK->getType(0), origTopKRes0);
  EXPECT_EQ(valRes0.getType(), origTopKRes0);
  EXPECT_EQ(topK->getType(1), origTopKRes1);
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

/// Tests that expect death from the verifier cannot currently run in Release
/// mode as they would not die, since the verifier uses assertions for
/// verification. Once the verifier moves to returning false instead of aborting
/// (GH issue #1517), this can be removed and EXPECT_DEATH can be replaced by
/// EXPECT_FALSE.
#ifndef NDEBUG

/// Check that verify doesn't allow for multiple writers to the same node.
TEST(Graph, verifyOneWriter) {
  Module M;
  auto *F = M.createFunction("main");

  auto *input = M.createPlaceholder(ElemKind::FloatTy, {5}, "input", false);
  auto *output = M.createPlaceholder(ElemKind::FloatTy, {5}, "output", false);
  F->createSave("Save1", input, output);
  F->createSave("Save2", input, output);

  EXPECT_DEATH(M.verify(), "");
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

  EXPECT_DEATH(M.verify(), "");
}

#endif /* NDEBUG */
