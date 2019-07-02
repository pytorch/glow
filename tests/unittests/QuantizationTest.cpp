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

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Quantization/Serialization.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"

namespace glow {

using llvm::cast;

class Quantization : public ::testing::TestWithParam<std::string> {};

class Operator
    : public ::testing::TestWithParam<::std::tuple<std::string, std::string>> {
protected:
  ExecutionEngine profileEE{};
  ExecutionEngine backendSpecificEE{};

  void SetUp() override {
    std::string backend1;
    std::string backend2;
    std::tie(backend1, backend2) = GetParam();
    profileEE.setBackend(backend1);
    backendSpecificEE.setBackend(backend2);
  }
};

class InterpAndCPU : public Operator {};

bool operator==(const NodeQuantizationInfo &lhs,
                const NodeQuantizationInfo &rhs) {
  return lhs.Scale() == rhs.Scale() && lhs.Offset() == rhs.Offset() &&
         lhs.nodeOutputName_ == rhs.nodeOutputName_;
}

/// This is a mock backend which extended support of quantized operators.
class MockQuantBackend : public Backend {
  // The actual backend being wrapped.
  std::unique_ptr<Backend> backend_;

public:
  MockQuantBackend() { backend_.reset(createBackend("Interpreter")); }

  std::string getBackendName() const override { return "Interpreter"; }

  llvm::Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override {
    return backend_->compile(F, opts);
  }

  bool isOpSupported(const NodeInfo &NI) const override {
    if (NI.getKind() == Kinded::Kind::SoftMaxNodeKind ||
        NI.getKind() == Kinded::Kind::LocalResponseNormalizationNodeKind ||
        NI.getKind() == Kinded::Kind::SaveNodeKind ||
        NI.getKind() == Kinded::Kind::ReluNodeKind ||
        NI.getKind() == Kinded::Kind::SelectNodeKind ||
        NI.getKind() == Kinded::Kind::LogNodeKind ||
        NI.getKind() == Kinded::Kind::SigmoidNodeKind ||
        NI.getKind() == Kinded::Kind::TanhNodeKind) {
      return true;
    }
    return backend_->isOpSupported(NI);
  }
};

void testSerialization(const std::vector<NodeQuantizationInfo> &expected) {
  llvm::SmallVector<char, 10> resultPath;
  llvm::sys::fs::createTemporaryFile("prefix", "suffix", resultPath);
  std::string filePath(resultPath.begin(), resultPath.end());

  serializeToYaml(filePath, expected);
  std::vector<NodeQuantizationInfo> deserialized =
      deserializeFromYaml(filePath);
  llvm::sys::fs::remove(filePath);
  EXPECT_EQ(expected, deserialized);
}

TEST(Quantization, Serialize) {
  std::vector<NodeQuantizationInfo> expected{
      {"first", {1, 10}}, {"second", {-1, 3}}, {"third", {-10, 30}}};

  testSerialization(expected);
}

#if LLVM_VERSION_MAJOR < 8
TEST(Quantization, SerializeEmpty) {
  std::vector<NodeQuantizationInfo> expected;

  testSerialization(expected);
}
#endif

template <typename From, typename To> static To clip(From in) {
  static_assert(sizeof(From) >= sizeof(To),
                "Clip should reduce the variable size");
  auto mx = std::numeric_limits<To>::max();
  auto mn = std::numeric_limits<To>::min();
  return std::max<From>(mn, std::min<From>(mx, in));
}

TEST(Quantization, quantScaleOffset) {
  // Test different scale values from 1<<-23 to 1>>1.
  float scales[] = {
      0.0000001596f, 0.00000025f, 0.000000995f, 0.0000035f, 0.00000952f,
      0.00000113f,   0.000721f,   0.0000721f,   0.0000172f, 0.0000951f,
      0.0000721f,    0.0000341f,  0.0000222f,   0.0000172f, 0.000752f,
      0.000371f,     0.000321f,   0.000223f,    0.000112f,  0.00852f,
      0.00671f,      0.00592f,    0.00200f,     0.00107f,   0.0931f,
      0.0721f,       0.031f,      0.014f,       0.0132f,    0.712f,
      0.613f,        0.412f,      0.223f,       0.134f,     1.0f,
      1.13f,         1.612f,      1.523f,       2.0f};

  // Try all scale factors:
  for (float scale : scales) {
    // Try all legal integers within the range:
    for (int8_t input = -128; input < 127; input++) {
      int32_t sum32num = round(input / scale);

      auto TR = quantization::quantizeScaleOffset32To8(scale, 0);
      int32_t computed = TR.transform(sum32num);

      EXPECT_NEAR(input, computed, 1);
    }
  }
}

template <class qtype>
void quantizeTensorTest(ElemKind qTy, quantization::Schema schema) {
  // Map float [0.0; 6.0] to a quantized type using its entire value range.
  TensorQuantizationParams quantParams =
      chooseQuantizationParams(0.0, 6.0, schema, qTy);

  // Create an FP32 tensor with 6 elements and initialize it with numbers from 0
  // to 5.
  Tensor inputFP32(ElemKind::FloatTy, {6});
  Handle<float> THFP32 = inputFP32.getHandle<float>();
  for (unsigned i = 0; i < 6; ++i) {
    THFP32.at({i}) = i * 1.0f;
  }

  // Quantize the tensor.
  auto quantizedFP32 =
      quantization::quantizeTensor(inputFP32, quantParams, qTy);
  // Check that the dequantized result is close to the original values before
  // the quantization.
  Handle<qtype> THquantizedFP32 = quantizedFP32.getHandle<qtype>();
  for (unsigned i = 0; i < 6; ++i) {
    EXPECT_NEAR(THFP32.at({i}),
                quantization::dequantize(THquantizedFP32.at({i}), quantParams),
                0.05f);
  }

  // Create an FP16 tensor with 6 elements and initialize it with numbers from 0
  // to 5.
  Tensor inputFP16(ElemKind::Float16Ty, {6});
  Handle<float16> THFP16 = inputFP16.getHandle<float16>();
  for (unsigned i = 0; i < 6; ++i) {
    THFP16.at({i}) = i * 1.0f;
  }

  // Quantize the tensor.
  auto quantizedFP16 =
      quantization::quantizeTensor(inputFP16, quantParams, qTy);
  // Check that the dequantized result is close to the original values before
  // the quantization.
  Handle<qtype> THquantizedFP16 = quantizedFP16.getHandle<qtype>();
  for (unsigned i = 0; i < 6; ++i) {
    EXPECT_NEAR(THFP16.at({i}),
                quantization::dequantize(THquantizedFP16.at({i}), quantParams),
                0.05f);
  }
}

TEST(Quantization, quantizeTensorAsymmetricInt8) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::Asymmetric);
}
TEST(Quantization, quantizeTensorAsymmetricInt16) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::Asymmetric);
}
TEST(Quantization, quantizeTensorAsymmetricInt32) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::Asymmetric);
}
TEST(Quantization, quantizeTensorSymmetricInt8) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::Symmetric);
}
TEST(Quantization, quantizeTensorSymmetricInt16) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::Symmetric);
}
TEST(Quantization, quantizeTensorSymmetricInt32) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::Symmetric);
}
TEST(Quantization, quantizeTensorSymmetricUInt8) {
  quantizeTensorTest<int8_t>(ElemKind::Int8QTy,
                             quantization::Schema::SymmetricWithUnsigned);
}
TEST(Quantization, quantizeTensorSymmetricUInt16) {
  quantizeTensorTest<int16_t>(ElemKind::Int16QTy,
                              quantization::Schema::SymmetricWithUnsigned);
}
TEST(Quantization, quantizeTensorSymmetricUInt32) {
  quantizeTensorTest<int32_t>(ElemKind::Int32QTy,
                              quantization::Schema::SymmetricWithUnsigned);
}

/// Helper for quantizing a simple Conv with precision \p quantizationPrecision.
static void quantizeSimpleConvGraph(ElemKind quantizationPrecision) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto *filter = mod.createConstant(ElemKind::FloatTy, {2, 2, 2, 1}, "filter");
  auto *bias = mod.createConstant(ElemKind::FloatTy, {2}, "bias");
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 4, 8, 2});
  PlaceholderBindings bindings;
  bindings.allocate(input);

  auto *CN = F->createConv("Conv", input, filter, bias, outTy, {2, 2}, {1, 1},
                           {0, 2, 1, 3}, 1);
  auto *S = F->createSave("ret", CN);
  bindings.allocate(S->getPlaceholder());

  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.2f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(filter->getName()),
       {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(bias->getName()),
       {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(CN->getName()), {0.6f, 0}},
  }};

  quantConfig.precision = quantizationPrecision;
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer, F);
  EE.run(bindings);
}

/// Test that a simple Conv graph can be quantized in Int8QTy.
TEST(Quantization, int8QuantizeGraph) {
  quantizeSimpleConvGraph(ElemKind::Int8QTy);
}

/// Test that a simple Conv graph can be quantized in Int16QTy.
TEST(Quantization, int16QuantizeGraph) {
  quantizeSimpleConvGraph(ElemKind::Int16QTy);
}

/// Test that when a node is quantized before its users are quantized then the
/// users correctly find the quantization parameters. This tests that updating
/// the nodeToTQP_ map in FunctionQuantizer::postProcessing() works correctly.
TEST(Quantization, TestQuantizedInputBeforeQuantizedNode) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {3}, "input", true);
  PlaceholderBindings bindings;
  bindings.allocate(input);

  // Note: Intentionally add successive reshapes so the GraphOptimizer merges
  // them and creates a new one. This way the newly created Reshape will be
  // placed at the end of the list of nodes in F, and then it will be quantized
  // before SN. I think this is the most straightforward way to cover the logic
  // path inside FunctionQuantizer::postProcessing() that updates nodeToTQP_.
  auto *reshape1 = F->createReshape("reshape1", input, {3, 1});
  auto *reshape2 = F->createReshape("reshape2", reshape1, {1, 3});
  auto *SN = F->createSlice("slice", reshape2, {0, 1}, {1, 2});
  auto *S = F->createSave("ret", SN);
  bindings.allocate(S->getPlaceholder());

  // We need to optimize here first so that the two reshapes are merged.
  optimize(F, CompilationMode::Infer);

  Node *newReshape = SN->getInput().getNode();
  ASSERT_TRUE(newReshape);
  ASSERT_TRUE(llvm::isa<ReshapeNode>(newReshape));

  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.2f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(newReshape->getName()),
       {0.2f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(SN->getName()), {0.2f, 0}},
  }};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  // Remove unnecessary conversions.
  optimize(F, CompilationMode::Infer);

  // Now we verify that the SliceNode was in fact quantized.
  {
    auto *saveNode = llvm::dyn_cast<SaveNode>(F->getNodeByName("ret"));
    ASSERT_TRUE(saveNode);
    auto *deqNode =
        llvm::dyn_cast<DequantizeNode>(saveNode->getInput().getNode());
    ASSERT_TRUE(deqNode);
    auto *sliceNode = llvm::dyn_cast<SliceNode>(deqNode->getInput().getNode());
    ASSERT_TRUE(sliceNode);
    EXPECT_TRUE(sliceNode->getResult().getType()->isQuantizedType());
  }
}

/// Test enabling RowwiseQuantizedFullyConnected in Glow quantization
/// procuedure. A FC can be quantized and converted to a
/// RowwiseQuantizedFullyConnected if:
/// 1. The weights of FC is constant;
/// 2. Use -enable-rowwise option or set enableRowwise param in
/// quantization::quantizeFunction to true. In unittest, the later one is used.
TEST(Quantization, enableRowwiseQuantizedFullyConnected) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *W = mod.createPlaceholder(ElemKind::FloatTy, {3, 2}, "weights", true);
  auto *B = mod.createPlaceholder(ElemKind::FloatTy, {2}, "bias", true);
  PlaceholderBindings bindings;
  bindings.allocate(input);
  bindings.allocate(W)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(B)->init(Tensor::InitKind::Broadcast, 0.1, mod.getPRNG());

  auto *WC = mod.createConstant(ElemKind::FloatTy, W->dims(), "wc");
  auto *FC = F->createFullyConnected("FC", input, WC, B);
  auto *S = F->createSave("ret", FC);
  bindings.allocate(S->getPlaceholder());

  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctx(/* bindings */ nullptr, &loweredMapForQuant);
  ::glow::lower(F, cctx, EE.getBackend());

  // Get the MatMul node and the Batched_Add node.
  Node *matMul, *batchedAdd;
  for (Node &N : F->getNodes()) {
    if (N.getKind() == Kinded::Kind::MatMulNodeKind) {
      matMul = &N;
    }
    if (N.getKind() == Kinded::Kind::BatchedAddNodeKind) {
      batchedAdd = &N;
    }
  }
  ASSERT_TRUE(matMul);
  ASSERT_TRUE(batchedAdd);

  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.2f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(WC->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(B->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(matMul->getName()),
       {0.6f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(batchedAdd->getName()),
       {0.6f, 0}},
  }};

  quantConfig.enableRowwise = true;
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend(),
                                 loweredMapForQuant);

  // Check the graph structure after quantization.
  auto *saveNode = llvm::dyn_cast<SaveNode>(F->getNodeByName("ret"));
  ASSERT_TRUE(saveNode);
  auto *deqNode =
      llvm::dyn_cast<DequantizeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(deqNode);
  auto *rwNode = llvm::dyn_cast<RowwiseQuantizedFullyConnectedNode>(
      deqNode->getInput().getNode());
  ASSERT_TRUE(rwNode);
  auto *inNode = llvm::dyn_cast<QuantizeNode>(rwNode->getInput().getNode());
  ASSERT_TRUE(inNode);
  auto *biasNode = llvm::dyn_cast<QuantizeNode>(rwNode->getBias().getNode());
  ASSERT_TRUE(biasNode);
  auto *weightsNode = llvm::dyn_cast<Constant>(rwNode->getWeights().getNode());
  ASSERT_TRUE(weightsNode);
  auto *scalesNode = llvm::dyn_cast<Constant>(rwNode->getScales().getNode());
  ASSERT_TRUE(scalesNode);
  auto *offsetsNode = llvm::dyn_cast<Constant>(rwNode->getOffsets().getNode());
  ASSERT_TRUE(offsetsNode);

  // Make sure that graph can be compiled and run. We check the correctness of
  // RowwiseQuantizedFullyConnected in operatorTests.cpp.
  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
}

/// Test enabling RowwiseQuantizedFullyConnected with Symmetric quantization.
TEST(Quantization, enableRowwiseQuantizedFullyConnectedSymmetric) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  PlaceholderBindings bindings;
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {10, 80}, "in", false);
  auto *FC = F->createFullyConnected(bindings, "FC", input, 100);
  auto *res = F->createSave("save", FC);
  bindings.allocate(res->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});

  // Note that we generate values for the Weights because they will be used
  // during rowwise-quantization to select each row's scale/offset.
  auto *WC = llvm::cast<Constant>(FC->getWeights());
  WC->getPayloadMutable().getHandle().randomize(-0.7, 1.1, mod.getPRNG());
  auto *BC = llvm::cast<Constant>(FC->getBias());

  TensorQuantizationParams inputTQP = chooseQuantizationParams(
      -1.0, 6.0, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  TensorQuantizationParams matmulTQP = chooseQuantizationParams(
      0.0, 10.0, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  TensorQuantizationParams batchedaddTQP = chooseQuantizationParams(
      0.0, 10.0, quantization::Schema::Symmetric, ElemKind::Int8QTy);
  TensorQuantizationParams biasTQP = chooseQuantizationParams(
      0, 20, quantization::Schema::Symmetric, ElemKind::Int8QTy);

  EXPECT_EQ(inputTQP.offset, 0);
  EXPECT_EQ(matmulTQP.offset, 0);
  EXPECT_EQ(batchedaddTQP.offset, 0);
  EXPECT_EQ(biasTQP.offset, 0);

  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctx(/* bindings */ nullptr, &loweredMapForQuant);
  ::glow::lower(F, cctx, EE.getBackend());

  // Get the MatMul node and the Batched_Add node.
  Node *matMul, *batchedAdd;
  for (Node &N : F->getNodes()) {
    if (N.getKind() == Kinded::Kind::MatMulNodeKind) {
      matMul = &N;
    }
    if (N.getKind() == Kinded::Kind::BatchedAddNodeKind) {
      batchedAdd = &N;
    }
  }
  ASSERT_TRUE(matMul);
  ASSERT_TRUE(batchedAdd);

  // Note: Using dummy offset for the weights, as it should be
  // rowwise-quantized.
  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       inputTQP},
      {NodeQuantizationInfo::generateNodeOutputName(WC->getName()), {1.0f, 1}},
      {NodeQuantizationInfo::generateNodeOutputName(BC->getName()), biasTQP},
      {NodeQuantizationInfo::generateNodeOutputName(matMul->getName()),
       matmulTQP},
      {NodeQuantizationInfo::generateNodeOutputName(batchedAdd->getName()),
       batchedaddTQP},
  }};

  quantConfig.schema = quantization::Schema::Symmetric;
  quantConfig.enableRowwise = true;
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend(),
                                 loweredMapForQuant);

  // Check the graph structure after quantization.
  auto *saveNode = llvm::dyn_cast<SaveNode>(F->getNodeByName("save"));
  ASSERT_TRUE(saveNode);
  auto *deqNode =
      llvm::dyn_cast<DequantizeNode>(saveNode->getInput().getNode());
  ASSERT_TRUE(deqNode);
  auto *rwNode = llvm::dyn_cast<RowwiseQuantizedFullyConnectedNode>(
      deqNode->getInput().getNode());
  ASSERT_TRUE(rwNode);
  auto *inNode = llvm::dyn_cast<QuantizeNode>(rwNode->getInput().getNode());
  ASSERT_TRUE(inNode);
  auto *biasNode = llvm::dyn_cast<QuantizeNode>(rwNode->getBias().getNode());
  ASSERT_TRUE(biasNode);
  auto *weightsNode = llvm::dyn_cast<Constant>(rwNode->getWeights().getNode());
  ASSERT_TRUE(weightsNode);
  auto *scalesNode = llvm::dyn_cast<Constant>(rwNode->getScales().getNode());
  ASSERT_TRUE(scalesNode);
  auto *offsetsNode = llvm::dyn_cast<Constant>(rwNode->getOffsets().getNode());
  ASSERT_TRUE(offsetsNode);

  // Because we're using symmetric quantization, the offsets should all be zero.
  const auto offsetsH = offsetsNode->getPayload().getHandle<int32_t>();
  EXPECT_TRUE(offsetsH.isZero());

  // Make sure that graph can be compiled and run. We check the correctness of
  // RowwiseQuantizedFullyConnected in operatorTests.cpp.
  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
}

/// Check that SLWS is correctly fused rowwise-quantized by the quantizer.
TEST(Quantization, enableRowwiseQuantizedSLWS) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  PlaceholderBindings bindings;

  auto *data = mod.createPlaceholder(ElemKind::FloatTy, {3, 1}, "data", false);
  auto *weights =
      mod.createPlaceholder(ElemKind::FloatTy, {8}, "weights", false);
  auto *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  // Don't worry about allocating them as we are not going to run anyway.
  bindings.allocate(data);
  bindings.allocate(weights);
  bindings.allocate(indices);
  bindings.allocate(lengths);

  auto *SLWS = F->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                                 lengths);
  auto *res = F->createSave("save", SLWS);
  ::glow::convertPlaceholdersToConstants(
      F, bindings, {indices, lengths, res->getPlaceholder()});
  bindings.allocate(res->getPlaceholder());

  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(
           SLWS->getData().getNode()->getName()),
       {0.2f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(
           SLWS->getWeights().getNode()->getName()),
       {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(SLWS->getName()),
       {0.4f, 0}},
  }};

  quantConfig.enableRowwise = true;
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  EE.compile(CompilationMode::Infer, F);

  // Check the graph structure after quantization.
  auto *saveNode = llvm::dyn_cast<SaveNode>(F->getNodeByName("save"));
  ASSERT_TRUE(saveNode);
  auto *FRWQSLWS =
      llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
          saveNode->getInput().getNode());
  ASSERT_TRUE(FRWQSLWS);
}

/// Quantize ReLU node and make sure that quantized version
/// has quantization parameters mapping to non-negative floating
/// point range.
TEST(Quantization, quantizeReLU) {
  ExecutionEngine EE{};
  EE.setBackend(new MockQuantBackend());
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *relu = F->createRELU("ReLU", input);
  PlaceholderBindings bindings;
  F->createSave("ret", relu);
  // Make sure that offset quantization parameter of ReLU is set
  // such that it produces non-negative floating point range.
  quantization::QuantizationConfiguration quantConfig{
      {{NodeQuantizationInfo::generateNodeOutputName(input->getName()),
        {0.2f, 0}},
       {NodeQuantizationInfo::generateNodeOutputName(relu->getName()),
        {0.2f, -128}}}};
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());
  EE.compile(CompilationMode::Infer, F);

  auto *save = llvm::cast<SaveNode>(F->getNodeByName("ret"));
  ASSERT_TRUE(llvm::isa<DequantizeNode>(save->getInput().getNode()));
  auto *dequantize = llvm::cast<DequantizeNode>(save->getInput().getNode());
  ASSERT_TRUE(llvm::isa<MaxNode>(dequantize->getInput().getNode()));

  MaxNode *max = llvm::cast<MaxNode>(dequantize->getInput().getNode());
  ASSERT_TRUE(max->getResult().getType()->isQuantizedType());
  EXPECT_EQ(max->getResult().getType()->getOffset(), -128);
  EXPECT_EQ(max->getResult().getType()->getScale(), 0.2f);
}

/// Quantize Log, Sigmoid, and Tanh nodes and make sure that quantized versions
/// are implemented as IntLookupTables, because the Interpreter only supports
/// them as such.
TEST(Quantization, quantizeLookupTables) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *LN = F->createLog("log", input);
  auto *SN = F->createSigmoid("sigmoid", LN);
  auto *TN = F->createTanh("tanh", SN);
  F->createSave("ret", TN);

  quantization::QuantizationConfiguration quantConfig{
      {{NodeQuantizationInfo::generateNodeOutputName(input->getName()),
        {0.02f, -128}},
       {NodeQuantizationInfo::generateNodeOutputName(LN->getName()),
        {0.008f, 0}},
       {NodeQuantizationInfo::generateNodeOutputName(SN->getName()),
        {0.03f, 2}},
       {NodeQuantizationInfo::generateNodeOutputName(TN->getName()),
        {0.04f, 3}}}};
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());
  optimize(F, CompilationMode::Infer);

  // Note: The scales/offsets used below are those expected based on
  // Sigmoid/Tanh requirements, or on the input values for the Log.

  auto *save = llvm::cast<SaveNode>(F->getNodeByName("ret"));
  auto *dequantizeTanh =
      llvm::dyn_cast<DequantizeNode>(save->getInput().getNode());
  ASSERT_TRUE(dequantizeTanh);
  auto *tanhILT =
      llvm::dyn_cast<IntLookupTableNode>(dequantizeTanh->getInput().getNode());
  ASSERT_TRUE(tanhILT);
  EXPECT_FLOAT_EQ(tanhILT->getResult().getType()->getScale(), 0.00784314);
  EXPECT_EQ(tanhILT->getResult().getType()->getOffset(), 0);
  EXPECT_FLOAT_EQ(tanhILT->getInput().getType()->getScale(), 0.02352941);
  EXPECT_EQ(tanhILT->getInput().getType()->getOffset(), 0);

  auto *rescaleSigmoid =
      llvm::dyn_cast<RescaleQuantizedNode>(tanhILT->getInput().getNode());
  ASSERT_TRUE(rescaleSigmoid);
  auto *sigmoidILT =
      llvm::dyn_cast<IntLookupTableNode>(rescaleSigmoid->getInput().getNode());
  ASSERT_TRUE(sigmoidILT);
  EXPECT_FLOAT_EQ(sigmoidILT->getResult().getType()->getScale(), 0.00392157);
  EXPECT_EQ(sigmoidILT->getResult().getType()->getOffset(), -128);
  EXPECT_FLOAT_EQ(sigmoidILT->getInput().getType()->getScale(), 0.047058824);
  EXPECT_EQ(sigmoidILT->getInput().getType()->getOffset(), 0);

  auto *rescaleLog =
      llvm::dyn_cast<RescaleQuantizedNode>(sigmoidILT->getInput().getNode());
  ASSERT_TRUE(rescaleLog);
  auto *logILT =
      llvm::dyn_cast<IntLookupTableNode>(rescaleLog->getInput().getNode());
  ASSERT_TRUE(logILT);
  EXPECT_FLOAT_EQ(logILT->getResult().getType()->getScale(), 0.008);
  EXPECT_EQ(logILT->getResult().getType()->getOffset(), 0);
  EXPECT_FLOAT_EQ(logILT->getInput().getType()->getScale(), 0.02);
  EXPECT_EQ(logILT->getInput().getType()->getOffset(), -128);
}

/// Quantize Log, Sigmoid, and Tanh nodes and make sure that they are not
/// replaced by LookupTables because the backend supports them directly.
TEST(Quantization, quantizeWithoutLookupTables) {
  ExecutionEngine EE{};
  EE.setBackend(new MockQuantBackend());
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *LN = F->createLog("log", input);
  auto *SN = F->createSigmoid("sigmoid", LN);
  auto *TN = F->createTanh("tanh", SN);
  F->createSave("ret", TN);

  quantization::QuantizationConfiguration quantConfig{
      {{NodeQuantizationInfo::generateNodeOutputName(input->getName()),
        {0.02f, -128}},
       {NodeQuantizationInfo::generateNodeOutputName(LN->getName()),
        {0.008f, 0}},
       {NodeQuantizationInfo::generateNodeOutputName(SN->getName()),
        {0.03f, 2}},
       {NodeQuantizationInfo::generateNodeOutputName(TN->getName()),
        {0.04f, 3}}}};
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());
  optimize(F, CompilationMode::Infer);

  auto *save = llvm::cast<SaveNode>(F->getNodeByName("ret"));
  auto *dequantize = llvm::dyn_cast<DequantizeNode>(save->getInput().getNode());
  ASSERT_TRUE(dequantize);
  auto *tanh = llvm::dyn_cast<TanhNode>(dequantize->getInput());
  ASSERT_TRUE(tanh);
  EXPECT_FLOAT_EQ(tanh->getResult().getType()->getScale(), 0.04);
  EXPECT_EQ(tanh->getResult().getType()->getOffset(), 3);
  EXPECT_FLOAT_EQ(tanh->getInput().getType()->getScale(), 0.03);
  EXPECT_EQ(tanh->getInput().getType()->getOffset(), 2);

  auto *sigmoid = llvm::dyn_cast<SigmoidNode>(tanh->getInput());
  ASSERT_TRUE(sigmoid);
  EXPECT_FLOAT_EQ(sigmoid->getResult().getType()->getScale(), 0.03);
  EXPECT_EQ(sigmoid->getResult().getType()->getOffset(), 2);
  EXPECT_FLOAT_EQ(sigmoid->getInput().getType()->getScale(), 0.008);
  EXPECT_EQ(sigmoid->getInput().getType()->getOffset(), 0);

  auto *log = llvm::dyn_cast<LogNode>(sigmoid->getInput());
  ASSERT_TRUE(log);
  EXPECT_FLOAT_EQ(log->getResult().getType()->getScale(), 0.008);
  EXPECT_EQ(log->getResult().getType()->getOffset(), 0);
  EXPECT_FLOAT_EQ(log->getInput().getType()->getScale(), 0.02);
  EXPECT_EQ(log->getInput().getType()->getOffset(), -128);
}

/// Fills the tensor \p H with some stable random data with the seed \p seed
/// and the range [-scale .. scale].
static void fillStableRandomData(Handle<float> H, size_t seed,
                                 float scale = 1) {
  for (size_t i = 0, e = H.size(); i < e; i++) {
    H.raw(i) = scale * (float((int(i * 1921 + seed) % 100) - 50) / 50);
  }
}

/// Builds a simple graph, returns back input var and save node through refs.
static Function *createSimpleGraphForQuantization(Module *M,
                                                  PlaceholderBindings &bindings,
                                                  Placeholder *A,
                                                  Placeholder *B,
                                                  llvm::StringRef funcName) {
  Function *F = M->createFunction(funcName);

  fillStableRandomData(bindings.allocate(A)->getHandle(), 1100, 1);

  fillStableRandomData(bindings.allocate(B)->getHandle(), 2001, 1);

  ConvolutionNode *CV = F->createConv(bindings, "conv", A, 16, 5, 1, 2, 2);
  auto *bias = cast<Placeholder>(CV->getBias());
  auto *filter = cast<Placeholder>(CV->getFilter());
  fillStableRandomData(bindings.get(bias)->getHandle(), 2001, 1);
  fillStableRandomData(bindings.get(filter)->getHandle(), 1000, 1);

  auto *RL = F->createRELU("relu", CV);
  auto *MP = F->createMaxPool("maxPool", RL, 2, 2, 1);
  // Just add noop transpose.
  auto *T = F->createTranspose("transpose", MP, {0, 1, 2, 3});
  // Noop reshape, make sure conversion quantization procedure works well.
  auto *R = F->createReshape("reshape", T, T->getResult().dims());
  auto *AP = F->createAvgPool("avgPool", R, 2, 2, 1);

  FullyConnectedNode *FC = F->createFullyConnected(bindings, "fc", AP, 10);

  // Noop slice, make sure conversion quantization procedure works well.
  auto *S =
      F->createSlice("slice", FC, {0, 1},
                     {FC->getResult().dims()[0], FC->getResult().dims()[1]});
  auto *bias2 = cast<Placeholder>(FC->getBias());
  auto *filter2 = cast<Placeholder>(FC->getWeights());

  fillStableRandomData(bindings.get(bias2)->getHandle(), 3001, 1);
  fillStableRandomData(bindings.get(filter2)->getHandle(), 4000, 1);

  auto *CN = F->createConcat("concat", {S, B}, 0);
  auto *SP = F->createSplat("splat", B->getType(), 10.0);
  auto *O = F->createConcat("concat", {CN, SP}, 0);
  auto *TN = F->createTranspose("transpose", O, {1, 0});
  auto *MMN = F->createMatMul("batchedreduceadd", O, TN);
  auto *BRAN = F->createBatchedReduceAdd("batchedreduceadd", MMN, 0);
  auto *TLN = F->createTile("tile", BRAN, 2, 0);
  auto *SN = F->createSplat("splat", TLN->getResult().getType(), 100.0);
  auto *MN = F->createMax("max", SN, TLN);
  auto *CLTE = F->createCmpLTE("cmplte", MN, SN);
  auto *SLN = F->createSelect("select", CLTE, SN, MN);
  auto *save = F->createSave("save", SLN);
  bindings.allocate(save->getPlaceholder());
  return F;
}

/// Helper for an end to end test profiling a model on \p profileEE, then
/// quantizing and running it on \p backendSpecificEE, quantizing with precision
/// \p quantizationPrecision and disabling quantization for all Kinds in
/// \p keepOriginalPrecisionForNodes. Results are compared from the profiling
/// run and quantization run.
static void
testQuantizationEnd2End(ExecutionEngine &profileEE,
                        ExecutionEngine &backendSpecificEE,
                        ElemKind quantizationPrecision,
                        const KindSet &keepOriginalPrecisionForNodes = {}) {
  auto *mod = &profileEE.getModule();
  PlaceholderBindings bindings;

  auto *A =
      mod->createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 2}, "A", false);
  auto *B = mod->createPlaceholder(ElemKind::FloatTy, {10, 9}, "B", false);

  // STEP1 - Generate the first network to record the quantization parameters.
  Function *F1 = createSimpleGraphForQuantization(mod, bindings, A, B, "main");
  Function *F2 = F1->clone("main2");
  SaveNode *result1 = cast<SaveNode>(F1->getNodeByName("save"));

  LoweredInfoMap loweredMapForProf;
  CompilationContext cctxProf{&bindings, &loweredMapForProf};
  cctxProf.precisionConfig.quantMode = QuantizationMode::Profile;
  profileEE.compile(F1, cctxProf);

  // Run graph to capture profile.
  profileEE.run(bindings);

  // STEP2 - Use the profile to quantize a network.
  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctxQuant{&bindings, &loweredMapForQuant};

  // Get quantization infos and build new quantized graph.
  PrecisionConfiguration &precConfig = cctxQuant.precisionConfig;
  precConfig.quantMode = QuantizationMode::Quantize;
  precConfig.quantConfig.infos = quantization::generateNodeQuantizationInfos(
      bindings, F1, loweredMapForProf, quantization::Schema::Asymmetric,
      quantizationPrecision);
  precConfig.quantConfig.precision = quantizationPrecision;
  precConfig.quantConfig.assertAllNodesQuantized = true;
  precConfig.precisionModeKindSet = keepOriginalPrecisionForNodes;

  SaveNode *result2 = cast<SaveNode>(F2->getNodeByName("save"));

  backendSpecificEE.compile(F2, cctxQuant);
  backendSpecificEE.run(bindings);

  // STEP3 - Compare the results of the original and quantized functions.
  auto result1Handle = bindings.get(result1->getPlaceholder())->getHandle();
  auto result2Handle = bindings.get(result2->getPlaceholder())->getHandle();

  EXPECT_EQ(result1Handle.size(), result2Handle.size());

  for (int i = 0, e = result1Handle.size(); i < e; ++i) {
    float mx = result2Handle.raw(result2Handle.minMaxArg().second);
    double diff = std::fabs(result2Handle.raw(i) - result1Handle.raw(i)) / mx;

    // Allow 3% difference.
    EXPECT_NEAR(diff, 0, 0.03);
  }
}

/// End to end quantization test for Int8 quantization.
TEST_P(Operator, end2endInt8) {
  // The OpenCL backend does not support some of the nodes in the test;
  // explicitly whitelist them here as staying in float, so that the quantizer
  // does not complain.
  KindSet keepOriginalPrecisionForNodes;
  if (backendSpecificEE.getBackend()->getBackendName() == "OpenCL") {
    keepOriginalPrecisionForNodes.insert(Kinded::Kind::SelectNodeKind);
    keepOriginalPrecisionForNodes.insert(Kinded::Kind::CmpLTENodeKind);
    keepOriginalPrecisionForNodes.insert(
        Kinded::Kind::BatchedReduceAddNodeKind);
  }

  testQuantizationEnd2End(profileEE, backendSpecificEE, ElemKind::Int8QTy,
                          keepOriginalPrecisionForNodes);
}

/// Fills the tensor \p H with some stable random integers with the seed \p seed
/// and the range [0, scale).
static void fillStableRandomIndex(Handle<int64_t> H, size_t seed,
                                  size_t scale = 10) {
  for (size_t i = 0, e = H.size(); i < e; i++) {
    H.raw(i) = int(i * 1921 + seed) % scale;
  }
}

/// Builds a graph with two GRUs and saves output from last hidden node.
static Function *createGRUForQuantization(Module *M,
                                          PlaceholderBindings &bindings,
                                          llvm::StringRef funcName) {
  Function *F = M->createFunction(funcName);

  constexpr unsigned sequenceSize = 2;
  constexpr unsigned embeddingSize = 10;
  constexpr unsigned languageSize = 10;
  constexpr unsigned batchSize = 5;
  constexpr unsigned hiddenSize = 3 * embeddingSize;

  // STEP1 - Initialize inputs into GRU
  auto *emb = F->getParent()->createPlaceholder(
      ElemKind::FloatTy, {languageSize, embeddingSize}, "embedding", false);
  fillStableRandomData(bindings.allocate(emb)->getHandle(), 4565, 1);

  auto *input = F->getParent()->createPlaceholder(
      ElemKind::Int64ITy, {batchSize, sequenceSize}, "input", false);
  fillStableRandomIndex(bindings.allocate(input)->getHandle<int64_t>(), 7227,
                        10);

  auto *hiddenInit = F->getParent()->createPlaceholder(
      ElemKind::FloatTy, {batchSize, embeddingSize}, "hiddenInit", false);
  bindings.allocate(hiddenInit)->zero();
  Node *hidden = hiddenInit;

  for (unsigned step = 0; step < sequenceSize; step++) {
    // STEP2 - Gather a single set of embeddings for the GRU
    Node *inputEmbedded = F->createGather("gru.embedding", emb, input);
    Node *inputSlice =
        F->createSlice("gru.inputSlice", inputEmbedded, {0, step, 0},
                       {batchSize, step + 1, embeddingSize});
    Node *reshape =
        F->createReshape("gru.reshape", inputSlice, {batchSize, embeddingSize});

    // STEP3 - Generate a GRU
    // reference implementation:
    // https://github.com/pytorch/pytorch/blob/dd5c195646b941d3e20a72847ac48c41e272b8b2/torch/nn/_functions/rnn.py#L46
    // similar to /examples/fr2en.cpp

    auto *FCi =
        F->createFullyConnected(bindings, "gru.fci", reshape, hiddenSize);
    auto *biasI = cast<Placeholder>(FCi->getBias());
    auto *filterI = cast<Placeholder>(FCi->getWeights());
    fillStableRandomData(bindings.get(biasI)->getHandle(), 8877, 1);
    fillStableRandomData(bindings.get(filterI)->getHandle(), 1441, 1);

    auto *FCh =
        F->createFullyConnected(bindings, "gru.fch", hidden, hiddenSize);
    auto *biasH = cast<Placeholder>(FCh->getBias());
    auto *filterH = cast<Placeholder>(FCh->getWeights());
    fillStableRandomData(bindings.get(biasH)->getHandle(), 9009, 1);
    fillStableRandomData(bindings.get(filterH)->getHandle(), 1001, 1);

    Node *i_r =
        F->createSlice("gru.i_r", FCi, {0, 0}, {batchSize, embeddingSize});
    Node *i_i = F->createSlice("gru.i_i", FCi, {0, embeddingSize},
                               {batchSize, 2 * embeddingSize});
    Node *i_n = F->createSlice("gru.i_n", FCi, {0, 2 * embeddingSize},
                               {batchSize, 3 * embeddingSize});

    Node *h_r =
        F->createSlice("gru.h_r", FCh, {0, 0}, {batchSize, embeddingSize});
    Node *h_i = F->createSlice("gru.h_i", FCh, {0, embeddingSize},
                               {batchSize, 2 * embeddingSize});
    Node *h_n = F->createSlice("gru.h_n", FCh, {0, 2 * embeddingSize},
                               {batchSize, 3 * embeddingSize});

    Node *resetgate = F->createSigmoid("gru.resetgate",
                                       F->createAdd("i_r_plus_h_r", i_r, h_r));
    Node *inputgate = F->createSigmoid("gru.inputgate",
                                       F->createAdd("i_i_plus_h_i", i_i, h_i));
    Node *newgate = F->createTanh(
        "gru.newgate",
        F->createAdd("i_n_plus_rg_mult_h_n", i_n,
                     F->createMul("rg_mult_h_n", resetgate, h_n)));
    hidden = F->createAdd(
        "gru.newhidden", newgate,
        F->createMul("ig_mult_hmng", inputgate,
                     F->createSub("hidden_minus_newgate", hidden, newgate)));
  }
  // No-op TopK selection to test quantization
  Node *downsample = F->createTopK("gru.downsample", hidden, embeddingSize / 2);

  auto *save = F->createSave("save", {downsample, 0});
  bindings.allocate(save->getPlaceholder());
  return F;
}

TEST_P(Operator, end2endGRU) {
  // STEP1 - Generate the first network to record the quantization parameters.
  auto *mod = &profileEE.getModule();
  PlaceholderBindings bindings;
  Function *F1 = createGRUForQuantization(mod, bindings, "main");
  Function *F2 = F1->clone("main2");
  SaveNode *result1 = cast<SaveNode>(F1->getNodeByName("save"));

  LoweredInfoMap loweredMapForProf;
  CompilationContext cctxProf{&bindings, &loweredMapForProf};
  cctxProf.precisionConfig.quantMode = QuantizationMode::Profile;
  profileEE.compile(F1, cctxProf);

  // Run graph to capture profile.
  profileEE.run(bindings);

  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctxQuant{&bindings, &loweredMapForQuant};
  cctxQuant.precisionConfig.quantMode = QuantizationMode::Quantize;
  PrecisionConfiguration &precConfig = cctxQuant.precisionConfig;
  precConfig.quantConfig.infos = quantization::generateNodeQuantizationInfos(
      bindings, F1, loweredMapForProf);

  // The OpenCL backend does not support some of the nodes in the test;
  // explicitly whitelist them here as staying in float, so that the quantizer
  // does not complain.
  KindSet doNotQuantizeKinds;
  if (backendSpecificEE.getBackend()->getBackendName() == "OpenCL") {
    precConfig.precisionModeKindSet.insert(Kinded::Kind::TanhNodeKind);
    precConfig.precisionModeKindSet.insert(Kinded::Kind::SigmoidNodeKind);
    precConfig.precisionModeKindSet.insert(Kinded::Kind::GatherNodeKind);
  }

  // STEP2 - Use the profile to quantize a network.
  SaveNode *result2 = cast<SaveNode>(F2->getNodeByName("save"));

  backendSpecificEE.compile(F2, cctxQuant);
  backendSpecificEE.run(bindings);

  // STEP3 - Compare the results of the original and quantized functions.
  auto result1Handle = bindings.get(result1->getPlaceholder())->getHandle();
  auto result2Handle = bindings.get(result2->getPlaceholder())->getHandle();

  EXPECT_EQ(result1Handle.size(), result2Handle.size());

  for (int i = 0, e = result1Handle.size(); i < e; ++i) {
    float mx = result2Handle.raw(result2Handle.minMaxArg().second);
    double diff = std::fabs(result2Handle.raw(i) - result1Handle.raw(i)) / mx;

    // Allow 3% difference.
    EXPECT_NEAR(diff, 0, 0.03);
  }
}

TEST(Quantization, rescaleSameType) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("foo");
  auto *input =
      mod.createPlaceholder(ElemKind::Int8QTy, {1, 1}, 0.5, 11, "input", true);
  bindings.allocate(input)->init(Tensor::InitKind::Broadcast, 21,
                                 mod.getPRNG());

  auto *Q = F->createRescaleQuantized(
      "rescale", input, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.5, 11));
  auto *D = F->createDequantize("dequantize", Q);
  auto *save = F->createSave("ret", D);
  auto *result = bindings.allocate(save->getPlaceholder());

  EXPECT_EQ(F->getNodes().size(), 3);
  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
  EXPECT_EQ(F->getNodes().size(), 2);

  auto RH = result->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 5.0, 0.001);
}

TEST(Quantization, optimizeRescaleQuantize) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("foo");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 1}, "input", true);
  bindings.allocate(input)->init(Tensor::InitKind::Broadcast, 21,
                                 mod.getPRNG());

  auto *Q = F->createQuantize(
      "quant", input, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.25, 4));
  auto *RS = F->createRescaleQuantized(
      "rescale", Q, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.5, 11));
  auto *D = F->createDequantize("dequantize", RS);
  auto *save = F->createSave("ret", D);
  auto *result = bindings.allocate(save->getPlaceholder());

  EXPECT_EQ(F->getNodes().size(), 4);
  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
  EXPECT_EQ(F->getNodes().size(), 1);

  auto RH = result->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 21.0, 0.001);
}

/// Check that our asymmetric quantization schema produces
/// the expected scales and offsets for various ranges.
TEST(Quantization, chooseQuantizationAsymmetric) {
  // Map float [0.0; 6.0] to int [-128; 127].
  TensorQuantizationParams asymmetricParams =
      chooseQuantizationParams(0.0, 6.0, quantization::Schema::Asymmetric);
  // Dequantization formula is scale(X - offset).
  // So
  // 1. scale(-128 - offset) == 0.0
  // 2. scale(127 - offset) == 6.0
  // Given scale != 0, #1 gives -128 == offset
  // Then #2, gives scale == 6.0 / (127 - (-128)).
  EXPECT_EQ(asymmetricParams.offset, -128);
  EXPECT_NEAR(asymmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-3.0; 3.0] to int [-128; 127].
  asymmetricParams =
      chooseQuantizationParams(-3.0, 3.0, quantization::Schema::Asymmetric);
  // Dequantization formula is scale(X - offset).
  // So in theory, we should get
  // 1. scale(-128 - offset) == -3.0
  // 2. scale(127 - offset) == 3.0
  // Given scale != 0, #1 + #2 gives scale(-128 + 127 - 2*offset) == 0.0
  // offset == -1 / -2 == 0.5
  // Then #2 or #1, gives scale == 3.0 / 127.5.
  // However, when we get symmetric ranges (i.e., [-X; X]),
  // we actually force the zero point to map to 0.
  // In other words, scale(0 - offset) == 0.0, so our offset is 0.
  // Then our scale is simply: (inputMax - inputMin) / (outputMax - outputMin).
  // (3.0 - (-3.0)) / (127 - (-128)) == 6.0 / 255.
  EXPECT_EQ(asymmetricParams.offset, 0);
  EXPECT_NEAR(asymmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-2.0; 5.0] to int [-128; 127].
  asymmetricParams =
      chooseQuantizationParams(-2.0, 5.0, quantization::Schema::Asymmetric);
  // Scale: (5.0 - (-2.0)) / (127 - (-128)) == 7.0 / 255.0
  // Offset from min: scale(-128 - offset) == -2.0
  //                  7.0 / 255.0 * (-128 - offset) == -2.0
  //                  -128 - offset == -2.0 * 255.0 / 7.0
  //                  offset == 2.0 * 255.0 / 7.0 - 128
  //                  offset == ~-55
  EXPECT_EQ(asymmetricParams.offset, (int32_t)(2.0 * 255 / 7.0 - 128));
  EXPECT_NEAR(asymmetricParams.scale, 7.0 / 255, 0.001);

  // Map float [2.0; 5.0] to int [-128; 127].
  // Make sure we extend the range to include 0.0, i.e.,
  // we really map [0.0; 5.0] to int [-128; 127].
  asymmetricParams =
      chooseQuantizationParams(2.0, 5.0, quantization::Schema::Asymmetric);
  // Scale: (5.0 - (0.0)) / (127 - (-128)) == 5.0 / 255.0
  // Offset from min: scale(-128 - offset) == 0.0
  EXPECT_EQ(asymmetricParams.offset, -128);
  EXPECT_NEAR(asymmetricParams.scale, 5.0 / 255, 0.001);

  // Map float [-8.0; -2.0] to int [-128; 127].
  // Make sure we extend the range to include 0.0, i.e.,
  // we really map [-8.0; 0.0] to int [-128; 127].
  asymmetricParams =
      chooseQuantizationParams(-8.0, -2.0, quantization::Schema::Asymmetric);
  // Scale: (0.0 - (-8.0)) / (127 - (-128)) == 8.0 / 255.0
  // Offset from min: scale(127 - offset) == 0.0
  EXPECT_EQ(asymmetricParams.offset, 127);
  EXPECT_NEAR(asymmetricParams.scale, 8.0 / 255, 0.001);
}

/// Check that our symmetric quantization schema produces
/// the expected scales and offsets for various ranges.
TEST(Quantization, chooseQuantizationSymmetric) {
  // Map float [0.0; 6.0] to int [-128; 127].
  // With symmetric mapping, we basically map [-6.0; 6.0]
  TensorQuantizationParams symmetricParams =
      chooseQuantizationParams(0.0, 6.0, quantization::Schema::Symmetric);
  // With symmetric mapping offset should always be zero.
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 12.0 / 255, 0.001);

  // Map float [-3.0; 3.0] to int [-128; 127].
  symmetricParams =
      chooseQuantizationParams(-3.0, 3.0, quantization::Schema::Symmetric);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-2.0; 5.0] to int [-128; 127].
  // => [-5.0; 5.0] range for symmetric mode.
  symmetricParams =
      chooseQuantizationParams(-2.0, 5.0, quantization::Schema::Symmetric);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 10.0 / 255, 0.001);

  // Map float [2.0; 5.0] to int [-128; 127].
  // Ranges are extended to include 0.
  // => [0.0; 5.0] range for symmetric mode.
  symmetricParams =
      chooseQuantizationParams(2.0, 5.0, quantization::Schema::Symmetric);
  // Scale: (5.0 - (0.0)) / (127 - (-128)) == 5.0 / 255.0
  // Offset from min: scale(-128 - offset) == 0.0
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 10.0 / 255, 0.001);

  // Map float [-8.0; -2.0] to int [-128; 127].
  // => [-8.0; 8.0] range for symmetric mode.
  symmetricParams =
      chooseQuantizationParams(-8.0, -2.0, quantization::Schema::Symmetric);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 16.0 / 255, 0.001);
}

/// Check quantization symmetry in presence of infinities.
TEST(Quantization, chooseQuantizationSymmetricInf) {
  auto sym = quantization::Schema::Symmetric;
  EXPECT_EQ(chooseQuantizationParams(-INFINITY, INFINITY, sym).offset, 0);
  EXPECT_EQ(chooseQuantizationParams(INFINITY, INFINITY, sym).offset, 0);
  EXPECT_EQ(chooseQuantizationParams(-INFINITY, -INFINITY, sym).offset, 0);
  EXPECT_EQ(chooseQuantizationParams(-INFINITY, 1.0f, sym).offset, 0);
  EXPECT_EQ(chooseQuantizationParams(-INFINITY, -1.0f, sym).offset, 0);
  EXPECT_EQ(chooseQuantizationParams(-1.0f, INFINITY, sym).offset, 0);
  EXPECT_EQ(chooseQuantizationParams(1.0f, INFINITY, sym).offset, 0);
}

/// Check that Relu can use our symmetric quantization schema.
TEST(Quantization, reluCanUseSymmetricSchema) {
  PlaceholderBindings bindings;
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Placeholder *input =
      mod.createPlaceholder(ElemKind::FloatTy, {10}, "input", false);
  auto *inputTensor = bindings.allocate(input);
  auto IH = inputTensor->getHandle<float>();
  for (size_t i = 0; i < 10; i++) {
    IH.at({i}) = (i % 2 == 0) ? 5 : -5;
  }

  // Create symmetric params that will be used for Relu.
  TensorQuantizationParams reluParams =
      chooseQuantizationParams(0.0, 10.0, quantization::Schema::Symmetric);
  TypeRef reluTy = mod.uniqueType(ElemKind::Int8QTy, {10}, reluParams.scale,
                                  reluParams.offset);
  TensorQuantizationParams inputParams =
      chooseQuantizationParams(-10.0, 10.0, quantization::Schema::Symmetric);

  QuantizeNode *QN =
      F->createQuantize("quant", input,
                        mod.uniqueType(ElemKind::Int8QTy, {10},
                                       inputParams.scale, inputParams.offset));
  ReluNode *RN = F->createRELU("relu", QN, reluTy);
  DequantizeNode *DN = F->createDequantize("dequantize", RN);
  SaveNode *SN = F->createSave("save", DN);
  auto *res = bindings.allocate(SN->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);
  EE.run(bindings);

  // Verify all negative values were correctly set to zero.
  auto RH = res->getHandle();
  for (size_t i = 0; i < 10; i++) {
    if (i % 2 == 0) {
      EXPECT_NEAR(RH.at({i}), 5, 0.05);
    } else {
      EXPECT_EQ(RH.at({i}), 0);
    }
  }
}

/// Check that our symmetric with uint8 quantization schema produces
/// the expected scales and offsets for various ranges.
TEST(Quantization, chooseQuantizationSymmetricWithUInt8) {
  // Map float [0.0; 6.0] to int [-128; 127].
  // With symmetric with uint8 mapping, we basically map [0.0; 6.0]
  TensorQuantizationParams symmetricParams = chooseQuantizationParams(
      0.0, 6.0, quantization::Schema::SymmetricWithUnsigned);
  // Given this is a purely positive range, we should use uint8,
  // thus int8 - (-128).
  EXPECT_EQ(symmetricParams.offset, -128);
  EXPECT_NEAR(symmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-3.0; 3.0] to int [-128; 127].
  symmetricParams = chooseQuantizationParams(
      -3.0, 3.0, quantization::Schema::SymmetricWithUnsigned);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-2.0; 5.0] to int [-128; 127].
  // This has negative value, thus we fall back to purely symmetric.
  // => [-5.0; 5.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      -2.0, 5.0, quantization::Schema::SymmetricWithUnsigned);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 10.0 / 255, 0.001);

  // Map float [0; 0] to int [-128; 127].
  symmetricParams = chooseQuantizationParams(
      0.0, 0.0, quantization::Schema::SymmetricWithUnsigned);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 0.1, 0.001);

  // Map float [2.0; 5.0] to int [-128; 127].
  // All positive, using uint8.
  // However, our quantization schemas always include zero.
  // => [0.0; 5.0] range for uint8 mode.
  symmetricParams = chooseQuantizationParams(
      2.0, 5.0, quantization::Schema::SymmetricWithUnsigned);
  // Scale: (5.0 - (0.0)) / (127 - (-128)) == 5.0 / 255.0
  // Offset from min: scale(-128 - offset) == 0.0
  EXPECT_EQ(symmetricParams.offset, -128);
  EXPECT_NEAR(symmetricParams.scale, 5.0 / 255, 0.001);

  // Map float [-8.0; -2.0] to int [-128; 127].
  // => [-8.0; 8.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      -8.0, -2.0, quantization::Schema::SymmetricWithUnsigned);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 16.0 / 255, 0.001);
}

/// Check that LRN and Softmax are quantized.
TEST(Quantization, quantizeSoftmaxAndLRN) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  EE.setBackend(new MockQuantBackend());

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10}, "input", true);
  auto *selected =
      mod.createPlaceholder(ElemKind::Int64ITy, {1, 10}, "selected", true);
  auto *LRN =
      F->createLocalResponseNormalization("LRN", input, 2, 1.0, 0.0001, 0.75);
  auto *SM = F->createSoftMax("softmax", LRN, selected);
  auto *SN = F->createSave("ret", SM);

  quantization::QuantizationConfiguration quantConfig{
      {{NodeQuantizationInfo::generateNodeOutputName(input->getName()),
        {0.2f, 0}},
       {NodeQuantizationInfo::generateNodeOutputName(LRN->getName()),
        {0.3f, 0}},
       {NodeQuantizationInfo::generateNodeOutputName(SM->getName()), {0.4f, 0}},
       {NodeQuantizationInfo::generateNodeOutputName(SN->getName()),
        {0.4f, 0}}}};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  auto qLRNIt = std::find_if(
      F->getNodes().begin(), F->getNodes().end(), [](const Node &node) -> bool {
        return llvm::isa<LocalResponseNormalizationNode>(&node) &&
               node.getNthResult(LocalResponseNormalizationNode::ResultIdx)
                   .getType()
                   ->isQuantizedType();
      });
  ASSERT_NE(qLRNIt, F->getNodes().end());
  auto qSMIt = std::find_if(F->getNodes().begin(), F->getNodes().end(),
                            [](const Node &node) -> bool {
                              return llvm::isa<SoftMaxNode>(&node) &&
                                     node.getNthResult(SoftMaxNode::ResultIdx)
                                         .getType()
                                         ->isQuantizedType();
                            });
  ASSERT_NE(qSMIt, F->getNodes().end());

  // Make sure that SaveNode is not quantized.
  for (const auto &node : F->getNodes()) {
    if (auto *saveNode = llvm::dyn_cast<SaveNode>(&node)) {
      EXPECT_FALSE(saveNode->getInput().getType()->isQuantizedType());
    }
  }
}

/// Check that Select is quantized.
TEST(Quantization, quantizeSelect) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  EE.setBackend(new MockQuantBackend());

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {1, 10}, "LHS", false);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {1, 10}, "RHS", false);
  auto *cond = mod.createPlaceholder(ElemKind::BoolTy, {1, 10}, "cond", false);
  auto *select = F->createSelect("select", cond, LHS, RHS);
  F->createSave("save", select);

  TensorQuantizationParams LHSQP = {0.5f, 0};
  TensorQuantizationParams RHSQP = {0.3f, 0};
  TensorQuantizationParams selectQP = {0.4f, 0};

  quantization::QuantizationConfiguration quantConfig{
      {{NodeQuantizationInfo::generateNodeOutputName(LHS->getName()), LHSQP},
       {NodeQuantizationInfo::generateNodeOutputName(RHS->getName()), RHSQP},
       {NodeQuantizationInfo::generateNodeOutputName(select->getName()),
        selectQP}}};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  auto it = std::find_if(
      F->getNodes().begin(), F->getNodes().end(),
      [](const Node &node) -> bool { return llvm::isa<SelectNode>(&node); });
  ASSERT_NE(it, F->getNodes().end());

  SelectNode *qSelect = llvm::cast<SelectNode>(&(*it));
  TypeRef qSelectTy = qSelect->getResult().getType();
  TypeRef qLHSTy = qSelect->getLHS().getType();
  TypeRef qRHSTy = qSelect->getRHS().getType();

  ASSERT_TRUE(qSelectTy->isQuantizedType());
  EXPECT_EQ(qSelectTy->getScale(), selectQP.scale);
  EXPECT_EQ(qSelectTy->getOffset(), selectQP.offset);
  EXPECT_EQ(qLHSTy->getScale(), LHSQP.scale);
  EXPECT_EQ(qLHSTy->getOffset(), LHSQP.offset);
  EXPECT_EQ(qRHSTy->getScale(), RHSQP.scale);
  EXPECT_EQ(qRHSTy->getOffset(), RHSQP.offset);
}

/// Check that AvgPool is quantized, and its input and output have different
/// scale and offset.
TEST(Quantization, quantizeAvgPool) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  EE.setBackend(new MockQuantBackend());

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", true);
  auto *pool = F->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *s = F->createSave("save", pool);

  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.2f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(pool->getName()),
       {0.3f, 1}},
      {NodeQuantizationInfo::generateNodeOutputName(s->getName()), {0.4f, 0}},
  }};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  auto qPool = std::find_if(F->getNodes().begin(), F->getNodes().end(),
                            [](const Node &node) -> bool {
                              return llvm::isa<AvgPoolNode>(&node) &&
                                     node.getNthResult(AvgPoolNode::ResultIdx)
                                         .getType()
                                         ->isQuantizedType();
                            });
  ASSERT_NE(qPool, F->getNodes().end());
  auto *avgPool = llvm::cast<AvgPoolNode>(qPool);
  ASSERT_NE(avgPool->getInput().getType()->getScale(),
            avgPool->getResult().getType()->getScale());
  ASSERT_NE(avgPool->getInput().getType()->getOffset(),
            avgPool->getResult().getType()->getOffset());
}

/// Test option to disable quantization of specific node kinds in the graph.
TEST(Quantization, quantizeGraphPartially) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", true);
  bindings.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(RHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *MMN = F->createMatMul("matmul", LHS, RHS);
  auto *TN = F->createTanh("tanh", MMN);
  auto *save = F->createSave("ret", TN);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  // Note that we are creating quantization info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(LHS->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(RHS->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(MMN->getName()), {0.6f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(TN->getName()), {0.5f, 0}},
  }};

  // Do not quantize any tanh nodes.
  KindSet doNotQuantizeKinds;
  doNotQuantizeKinds.insert(Kinded::Kind::TanhNodeKind);

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend(),
                                 /* loweredMap */ {}, doNotQuantizeKinds);

  // Make sure that graph can be compiled and run.
  ::glow::convertPlaceholdersToConstants(F, bindings, {result});

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  // Do not perform any compile-time constant folding.
  cctx.optimizationOpts.enableConstantFolding = false;
  EE.compile(F, cctx);

  EE.run(bindings);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the tanh is not quantized.
    auto *TN = llvm::dyn_cast<TanhNode>(SN->getInput());
    ASSERT_TRUE(TN);
    EXPECT_TRUE(!TN->getResult().getType()->isQuantizedType());

    // Verify that the input to the tanh is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(TN->getInput());
    ASSERT_TRUE(DN);

    // Verify that the matmul is quantized.
    auto *MMN = llvm::dyn_cast<MatMulNode>(DN->getInput());
    ASSERT_TRUE(MMN);
    EXPECT_TRUE(MMN->getResult().getType()->isQuantizedType());

    // Verify that the variable inputs to the matmul are quantized.
    auto *LHS = llvm::dyn_cast<Constant>(MMN->getLHS());
    ASSERT_TRUE(LHS);
    EXPECT_TRUE(LHS->getType()->isQuantizedType());

    auto *RHS = llvm::dyn_cast<Constant>(MMN->getRHS());
    ASSERT_TRUE(RHS);
    EXPECT_TRUE(RHS->getType()->isQuantizedType());
  }
}

/// Test option to disable quantization of specific node kinds in the graph,
/// where there are multiple of that node kind.
TEST(Quantization, quantizeGraphPartiallyMultipleNodes) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", true);
  bindings.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(RHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *TNLHS = F->createTanh("tanh", LHS);
  auto *MMN = F->createMatMul("matmul", TNLHS, RHS);
  auto *TN = F->createTanh("tanh", MMN);
  auto *save = F->createSave("ret", TN);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  // Note that we are creating quantization info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(LHS->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(TNLHS->getName()),
       {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(RHS->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(MMN->getName()), {0.6f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(TN->getName()), {0.5f, 0}},
  }};

  // Do not quantize any tanh nodes.
  KindSet doNotQuantizeKinds;
  doNotQuantizeKinds.insert(Kinded::Kind::TanhNodeKind);

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend(),
                                 /* loweredMap */ {}, doNotQuantizeKinds);

  // Make sure that graph can be compiled and run.
  ::glow::convertPlaceholdersToConstants(F, bindings, {result});

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  // Do not perform any compile-time constant folding.
  cctx.optimizationOpts.enableConstantFolding = false;
  EE.compile(F, cctx);

  EE.run(bindings);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the tanh is not quantized.
    auto *TN1 = llvm::dyn_cast<TanhNode>(SN->getInput());
    ASSERT_TRUE(TN1);
    EXPECT_TRUE(!TN1->getResult().getType()->isQuantizedType());

    // Verify that the input to the tanh is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(TN1->getInput());
    ASSERT_TRUE(DN);

    // Verify that the matmul is quantized.
    auto *MMN = llvm::dyn_cast<MatMulNode>(DN->getInput());
    ASSERT_TRUE(MMN);
    EXPECT_TRUE(MMN->getResult().getType()->isQuantizedType());

    // Verify that the LHS input is a quantize node.
    auto *QN = llvm::dyn_cast<QuantizeNode>(MMN->getLHS());
    ASSERT_TRUE(QN);

    // Verify that the second tanh node is also not quantized.
    auto *TN2 = llvm::dyn_cast<TanhNode>(QN->getInput());
    ASSERT_TRUE(TN2);
    EXPECT_TRUE(!TN2->getResult().getType()->isQuantizedType());

    // Verify that the input variable to the tanh is not quantized.
    auto *varTN2 = llvm::dyn_cast<Constant>(TN2->getInput());
    ASSERT_TRUE(varTN2);
    EXPECT_TRUE(!varTN2->getType()->isQuantizedType());

    // Verify that the RHS input to the matmul is a quantized variable.
    auto *RHS = llvm::dyn_cast<Constant>(MMN->getRHS());
    ASSERT_TRUE(RHS);
    EXPECT_TRUE(RHS->getType()->isQuantizedType());
  }
}

/// Test option to disable quantization of multiple specific node kinds in the
/// graph.
TEST(Quantization, quantizeGraphPartiallyMultipleKinds) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", true);
  bindings.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(RHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *MMN = F->createMatMul("matmul", LHS, RHS);
  auto *CN = F->createAdd("concat", LHS, MMN);
  auto *TN = F->createTanh("tanh", CN);
  auto *save = F->createSave("ret", TN);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  // Note that we are creating quantization info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(LHS->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(RHS->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(MMN->getName()), {0.6f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(CN->getName()), {0.6f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(TN->getName()), {0.5f, 0}},
  }};

  // Do not quantize any tanh or add nodes.
  KindSet doNotQuantizeKinds;
  doNotQuantizeKinds.insert(Kinded::Kind::TanhNodeKind);
  doNotQuantizeKinds.insert(Kinded::Kind::AddNodeKind);

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend(),
                                 /* loweredMap */ {}, doNotQuantizeKinds);

  // Make sure that graph can be compiled and run.
  ::glow::convertPlaceholdersToConstants(F, bindings, {result});

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  // Do not perform any compile-time constant folding.
  cctx.optimizationOpts.enableConstantFolding = false;
  EE.compile(F, cctx);

  EE.run(bindings);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the tanh is not quantized.
    auto *TN = llvm::dyn_cast<TanhNode>(SN->getInput());
    ASSERT_TRUE(TN);
    EXPECT_TRUE(!TN->getResult().getType()->isQuantizedType());

    // Verify that the input to the tanh is a non-quantized add node.
    auto *AN = llvm::dyn_cast<AddNode>(TN->getInput());
    ASSERT_TRUE(AN);
    EXPECT_TRUE(!TN->getResult().getType()->isQuantizedType());

    // Verify that the LHS input to the AddNode is an unquantized variable.
    auto varANLHS = llvm::dyn_cast<Constant>(AN->getLHS());
    ASSERT_TRUE(varANLHS);
    EXPECT_TRUE(!varANLHS->getType()->isQuantizedType());

    // Verify that the RHS input to the AddNode is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(AN->getRHS());
    ASSERT_TRUE(DN);

    // Verify that the matmul is quantized.
    auto *MMN = llvm::dyn_cast<MatMulNode>(DN->getInput());
    ASSERT_TRUE(MMN);
    EXPECT_TRUE(MMN->getResult().getType()->isQuantizedType());

    // Verify that the variable inputs to the matmul are quantized.
    auto *LHS = llvm::dyn_cast<Constant>(MMN->getLHS());
    ASSERT_TRUE(LHS);
    EXPECT_TRUE(LHS->getType()->isQuantizedType());

    auto *RHS = llvm::dyn_cast<Constant>(MMN->getRHS());
    ASSERT_TRUE(RHS);
    EXPECT_TRUE(RHS->getType()->isQuantizedType());
  }
}

/// Check that quantizeFunction directly converts the constants
/// instead of leaving quantize node around.
TEST(Quantization, quantizeFunctionConvertConstant) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createConstant(ElemKind::FloatTy, {3, 3}, "rhs");
  bindings.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  RHS->getPayloadMutable().init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *MMN = F->createMatMul("matmul", LHS, RHS);
  auto *save = F->createSave("ret", MMN);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  // Note that we are creating quantization info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(LHS->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(RHS->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(MMN->getName()), {0.6f, 0}},
  }};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  optimize(F, CompilationMode::Infer);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the input to save is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(SN->getInput());
    ASSERT_TRUE(DN);

    // Verify that the matmul is quantized.
    auto *MMN = llvm::dyn_cast<MatMulNode>(DN->getInput());
    ASSERT_TRUE(MMN);
    EXPECT_TRUE(MMN->getResult().getType()->isQuantizedType());

    // Verify that the variable inputs to the matmul are quantized.
    auto *LHSQuantize = llvm::dyn_cast<QuantizeNode>(MMN->getLHS());
    ASSERT_TRUE(LHSQuantize);
    EXPECT_EQ(LHSQuantize->getInput().getNode(), LHS);

    auto *RHS = llvm::dyn_cast<Constant>(MMN->getRHS());
    ASSERT_TRUE(RHS);
    EXPECT_TRUE(RHS->getType()->isQuantizedType());
  }

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
}

/// Check that the slice node doesn't change the quantization parameters between
/// its input and output.
TEST(Quantization, quantizeSlice) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {4}, "input", true);
  bindings.allocate(input)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *slice = F->createSlice("slice", input, {2}, {3});
  auto *save = F->createSave("ret", slice);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(slice->getName()),
       {0.2f, -128}},
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.4f, 0}},
  }};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  optimize(F, CompilationMode::Infer);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the input to save is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(SN->getInput());
    ASSERT_TRUE(DN);

    // Verify that the slice is rescaled after being quantized.
    // The reason we need a rescale is because slicing doesn't perform rescaling
    // by itself.
    // Note: after optimization, the RescaleQuantized node created for the Slice
    // gets merged with the dequantize node.
    auto *qslice = llvm::dyn_cast<SliceNode>(DN->getInput());
    ASSERT_TRUE(qslice);
    ASSERT_TRUE(qslice->getResult().getType()->isQuantizedType());
    EXPECT_EQ(qslice->getResult().getType()->getOffset(), 0);
    EXPECT_EQ(qslice->getResult().getType()->getScale(), 0.4f);

    // Verify that the variable inputs to the matmul are quantized.
    auto *qinput = llvm::dyn_cast<QuantizeNode>(qslice->getInput());
    ASSERT_TRUE(qinput);
    EXPECT_EQ(qinput->getResult().getType()->getOffset(),
              qslice->getResult().getType()->getOffset());
    EXPECT_EQ(qinput->getResult().getType()->getScale(),
              qslice->getResult().getType()->getScale());
    EXPECT_EQ(qinput->getInput().getNode(), input);
  }

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
}

/// Check that the reshape node doesn't change the quantization parameters
/// between its input and output.
TEST(Quantization, quantizeReshape) {
  ExecutionEngine EE{};
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "input", true);
  bindings.allocate(input)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *reshape = F->createReshape("reshape", input, {9});
  auto *save = F->createSave("ret", reshape);
  auto *result = save->getPlaceholder();
  bindings.allocate(result);

  quantization::QuantizationConfiguration quantConfig{{
      {NodeQuantizationInfo::generateNodeOutputName(reshape->getName()),
       {0.2f, -128}},
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.4f, 0}},
  }};

  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  optimize(F, CompilationMode::Infer);

  {
    // Verify that the output variable is not quantized, and that it has a
    // single save node writer, which is also not quantized.
    EXPECT_TRUE(!result->getType()->isQuantizedType());
    ASSERT_EQ(result->getUsers().size(), 1);
    auto *SN = llvm::dyn_cast<SaveNode>(result->getUsers().begin()->getUser());
    ASSERT_TRUE(SN);
    EXPECT_TRUE(!SN->getOutput().getType()->isQuantizedType());

    // Verify that the input to save is a dequantize node.
    auto *DN = llvm::dyn_cast<DequantizeNode>(SN->getInput());
    ASSERT_TRUE(DN);

    // Verify that the reshape is rescaled after being quantized.
    // The reason we need a rescale is because reshaping doesn't perform
    // rescaling by itself.
    // Note: after optimization, the RescaleQuantized node created for the
    // Reshape gets merged with the dequantize node.
    auto *qreshape = llvm::dyn_cast<ReshapeNode>(DN->getInput());
    ASSERT_TRUE(qreshape);
    ASSERT_TRUE(qreshape->getResult().getType()->isQuantizedType());
    EXPECT_EQ(qreshape->getResult().getType()->getOffset(), 0);
    EXPECT_EQ(qreshape->getResult().getType()->getScale(), 0.4f);

    // Verify that the variable inputs to the matmul are quantized.
    auto *qinput = llvm::dyn_cast<QuantizeNode>(qreshape->getInput());
    ASSERT_TRUE(qinput);
    EXPECT_EQ(qinput->getResult().getType()->getOffset(),
              qreshape->getResult().getType()->getOffset());
    EXPECT_EQ(qinput->getResult().getType()->getScale(),
              qreshape->getResult().getType()->getScale());
    EXPECT_EQ(qinput->getInput().getNode(), input);
  }

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer, F);

  EE.run(bindings);
}

/// Mock backend that does not lower FC nodes.
class MockBackendUnloweredFC : public MockBackend {
  bool shouldLower(const Node *N) const override {
    if (N->getKind() == Kinded::Kind::FullyConnectedNodeKind) {
      return false;
    }
    return true;
  }
  bool isOpSupported(const NodeInfo &NI) const override { return true; }
};

/// Mock backend that does lower FC nodes.
class MockBackendLoweredFC : public MockBackend {
  bool shouldLower(const Node *N) const override { return true; }
  bool isOpSupported(const NodeInfo &NI) const override { return true; }
};

/// Create a simple network with an FC given \p bindings, \p EE, and \p F.
/// \returns the FC node.
static FullyConnectedNode *createSimpleFCNet(PlaceholderBindings &bindings,
                                             ExecutionEngine &EE, Function &F) {
  auto &mod = EE.getModule();
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *W = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "weights", true);
  auto *B = mod.createPlaceholder(ElemKind::FloatTy, {3}, "bias", true);

  bindings.allocate(input);
  bindings.allocate(W)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  bindings.allocate(B)->init(Tensor::InitKind::Broadcast, 0.1, mod.getPRNG());

  auto *FC = F.createFullyConnected("FC", input, W, B);
  auto *S = F.createSave("ret", FC);
  ::glow::convertPlaceholdersToConstants(&F, bindings,
                                         {input, S->getPlaceholder()});
  bindings.allocate(S->getPlaceholder());

  return FC;
}

/// Helper to look for a node with kind \p NodeClass in \p F. If found, \returns
/// a pointer to the node. Otherwise \returns a nullptr.
template <class NodeClass>
static NodeClass *findNodeKindOrReturnNull(Function *F) {
  auto it = std::find_if(
      F->getNodes().begin(), F->getNodes().end(),
      [](const Node &node) -> bool { return llvm::isa<NodeClass>(&node); });
  if (it == F->getNodes().end()) {
    return nullptr;
  }
  return &llvm::cast<NodeClass>(*it);
}

/// Profile and quantize a graph with an FC, and make sure that we find the
/// correct quantization parameters, whether the \p BackendClass does or does
/// not lower the FC given \p expectLoweredFC. Note that in this test we
/// replicate the logic from optimizeFunction(), wherein we lower and then call
/// profileQuantization(), in order to ensure each stage of the compilation
/// pipeline for profiling/quantization is correct.
template <class BackendClass>
static void testProfileQuantizationOfFC(bool expectLoweredFC,
                                        bool rowwiseQuantizeFC) {
  ExecutionEngine profileEE{};
  Function *profileF = profileEE.getModule().createFunction("profile");
  PlaceholderBindings profilebindings;
  FullyConnectedNode *FC =
      createSimpleFCNet(profilebindings, profileEE, *profileF);
  auto outputNameFC = NodeQuantizationInfo::generateNodeOutputName(
      FC->getName(), FullyConnectedNode::ResultIdx);
  auto weightsNameFC = NodeQuantizationInfo::generateNodeOutputName(
      FC->getWeights().getNode()->getName(), FC->getWeights().getResNo());
  auto biasNameFC = NodeQuantizationInfo::generateNodeOutputName(
      FC->getBias().getNode()->getName(), FC->getBias().getResNo());
  auto inputNameFC = NodeQuantizationInfo::generateNodeOutputName(
      FC->getInput().getNode()->getName(), FC->getInput().getResNo());

  // Lower everything and keep track of the lowered components source nodes via
  // the loweredMap.
  LoweredInfoMap loweredMapForProf;
  CompilationContext cctx(/* bindings */ nullptr, &loweredMapForProf);
  lower(profileF, cctx);

  // Check that the lowered graph only contains the lowered components of the
  // FC (MM and BA) and not the FC itself.
  auto *loweredFC = findNodeKindOrReturnNull<FullyConnectedNode>(profileF);
  auto *loweredMM = findNodeKindOrReturnNull<MatMulNode>(profileF);
  auto *loweredBA = findNodeKindOrReturnNull<BatchedAddNode>(profileF);
  ASSERT_FALSE(loweredFC);
  ASSERT_TRUE(loweredMM);
  ASSERT_TRUE(loweredBA);
  auto outputNameMM = NodeQuantizationInfo::generateNodeOutputName(
      loweredMM->getName(), MatMulNode::ResultIdx);
  auto outputNameBA = NodeQuantizationInfo::generateNodeOutputName(
      loweredBA->getName(), BatchedAddNode::ResultIdx);

  glow::profileQuantization(profilebindings, profileF);

  // Compile/run to capture profile.
  profileEE.compile(CompilationMode::Infer, profileF);
  profileEE.run(profilebindings);

  // Get quantization infos and build new quantized graph, passing in the
  // loweredMapForProf to include the unlowered components in QI.
  quantization::QuantizationConfiguration quantConfig{
      quantization::generateNodeQuantizationInfos(
          profilebindings, profileF, loweredMapForProf,
          quantization::Schema::Asymmetric, ElemKind::Int8QTy)};

  // Verify that we have node quantization infos for the FC and the lowered
  // components of the FC (MM and BA).
  NodeQuantizationInfo *FCQI = nullptr, *MMQI = nullptr, *BAQI = nullptr,
                       *FCWQI = nullptr, *FCBQI = nullptr, *FCIQI = nullptr;
  for (NodeQuantizationInfo &NQI : quantConfig.infos) {
    if (NQI.nodeOutputName_ == outputNameFC) {
      FCQI = &NQI;
    } else if (NQI.nodeOutputName_ == outputNameMM) {
      MMQI = &NQI;
    } else if (NQI.nodeOutputName_ == outputNameBA) {
      BAQI = &NQI;
    } else if (NQI.nodeOutputName_ == weightsNameFC) {
      FCWQI = &NQI;
    } else if (NQI.nodeOutputName_ == biasNameFC) {
      FCBQI = &NQI;
    } else if (NQI.nodeOutputName_ == inputNameFC) {
      FCIQI = &NQI;
    }
  }
  ASSERT_TRUE(FCQI);
  ASSERT_TRUE(MMQI);
  ASSERT_TRUE(BAQI);
  ASSERT_TRUE(FCWQI);
  ASSERT_TRUE(FCBQI);
  ASSERT_TRUE(FCIQI);

  // Now create the same original function in the backend we're testing.
  ExecutionEngine backendEE;
  BackendClass backend;
  backendEE.setBackend(&backend, /* ownsBackend */ false);
  Function *backendF = backendEE.getModule().createFunction("quantized");
  PlaceholderBindings backendbindings;
  createSimpleFCNet(backendbindings, backendEE, *backendF);

  // Lower the function given the backend's preferences for lowering.
  LoweredInfoMap loweredMapForQuant;
  CompilationContext cctx2(/* bindings */ nullptr, &loweredMapForQuant);
  lower(backendF, cctx2, backendEE.getBackend());

  // Check that the backend lowered the function as expected.
  auto *floatFC = findNodeKindOrReturnNull<FullyConnectedNode>(backendF);
  auto *floatMM = findNodeKindOrReturnNull<MatMulNode>(backendF);
  auto *floatBA = findNodeKindOrReturnNull<BatchedAddNode>(backendF);
  if (expectLoweredFC) {
    ASSERT_FALSE(floatFC);
    ASSERT_TRUE(floatMM);
    ASSERT_TRUE(floatBA);
  } else {
    ASSERT_TRUE(floatFC);
    ASSERT_FALSE(floatMM);
    ASSERT_FALSE(floatBA);
  }

  // Quantize the function given the current backend we're testing along with
  // the quantization infos gathered.
  quantConfig.enableRowwise = rowwiseQuantizeFC;
  quantConfig.assertAllNodesQuantized = true;
  quantization::quantizeFunction(backendF, quantConfig, *backendEE.getBackend(),
                                 loweredMapForQuant);

  // Optimize the graph to remove dead code and optimize away unnecessary
  // quantize nodes. Note that we do not do a full compile call here, as we have
  // already lowered.
  ::glow::optimize(backendF, CompilationMode::Infer);

  // Check that the graph is still structured as expected, and that the
  // scales/offsets are set as found in QI.
  auto *quantFC = findNodeKindOrReturnNull<FullyConnectedNode>(backendF);
  auto *quantMM = findNodeKindOrReturnNull<MatMulNode>(backendF);
  auto *quantBA = findNodeKindOrReturnNull<BatchedAddNode>(backendF);
  auto *quantRowwiseFC =
      findNodeKindOrReturnNull<RowwiseQuantizedFullyConnectedNode>(backendF);

  if (rowwiseQuantizeFC) {
    EXPECT_FALSE(quantMM);
    EXPECT_FALSE(quantBA);
    EXPECT_FALSE(quantFC);

    ASSERT_TRUE(quantRowwiseFC);
    EXPECT_EQ(quantRowwiseFC->getResult().getType()->getScale(), FCQI->Scale());
    EXPECT_EQ(quantRowwiseFC->getResult().getType()->getOffset(),
              FCQI->Offset());

    EXPECT_EQ(quantRowwiseFC->getBias().getElementType(), ElemKind::Int32QTy);
    EXPECT_EQ(quantRowwiseFC->getBias().getType()->getScale(),
              FCWQI->Scale() * FCIQI->Scale());
    EXPECT_EQ(quantRowwiseFC->getBias().getType()->getOffset(), 0);
  } else if (expectLoweredFC) {
    ASSERT_FALSE(quantFC);
    ASSERT_FALSE(quantRowwiseFC);

    ASSERT_TRUE(quantMM);
    EXPECT_EQ(quantMM->getResult().getType()->getScale(), MMQI->Scale());
    EXPECT_EQ(quantMM->getResult().getType()->getOffset(), MMQI->Offset());

    ASSERT_TRUE(quantBA);
    EXPECT_EQ(quantBA->getResult().getType()->getScale(), BAQI->Scale());
    EXPECT_EQ(quantBA->getResult().getType()->getOffset(), BAQI->Offset());

    EXPECT_EQ(quantBA->getSlice().getElementType(), ElemKind::Int32QTy);
    EXPECT_EQ(quantBA->getSlice().getType()->getScale(),
              FCWQI->Scale() * FCIQI->Scale());
    EXPECT_EQ(quantBA->getSlice().getType()->getOffset(), 0);
  } else {
    ASSERT_FALSE(quantRowwiseFC);

    ASSERT_TRUE(quantFC);
    EXPECT_EQ(quantFC->getResult().getType()->getScale(), FCQI->Scale());
    EXPECT_EQ(quantFC->getResult().getType()->getOffset(), FCQI->Offset());

    ASSERT_FALSE(quantMM);
    ASSERT_FALSE(quantBA);

    EXPECT_EQ(quantFC->getBias().getElementType(), ElemKind::Int32QTy);
    EXPECT_EQ(quantFC->getBias().getType()->getScale(),
              FCWQI->Scale() * FCIQI->Scale());
    EXPECT_EQ(quantFC->getBias().getType()->getOffset(), 0);
  }
}

/// Test that backends that do not lower FCs can find the quantization
/// parameters of their nodes.
TEST(Quantization, TestProfileQuantizationOfUnloweredFC) {
  testProfileQuantizationOfFC<MockBackendUnloweredFC>(
      /* expectLoweredFC */ false, /* rowwiseQuantizeFC */ false);
}

/// Test that backends that do lower FCs can find the quantization parameters of
/// their nodes.
TEST(Quantization, TestProfileQuantizationOfLoweredFC) {
  testProfileQuantizationOfFC<MockBackendLoweredFC>(
      /* expectLoweredFC */ true, /* rowwiseQuantizeFC */ false);
}

/// Test that backends that do not lower FCs can find the quantization
/// parameters of their nodes and correctly rowwise quantize.
TEST(Quantization, TestProfileQuantizationOfUnloweredFCRowwise) {
  testProfileQuantizationOfFC<MockBackendUnloweredFC>(
      /* expectLoweredFC */ false, /* rowwiseQuantizeFC */ true);
}

/// Test that backends that do lower FCs can find the quantization parameters of
/// their nodes and correctly rowwise quantize even when lowering the FC.
TEST(Quantization, TestProfileQuantizationOfLoweredFCRowwise) {
  testProfileQuantizationOfFC<MockBackendLoweredFC>(
      /* expectLoweredFC */ true, /* rowwiseQuantizeFC */ true);
}

/// Check that asserting quantization for the quantizer works as expected.
TEST(Quantization, CheckAssertQuantization) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *relu = F->createRELU("ReLU", input);
  PlaceholderBindings bindings;
  auto *save = F->createSave("ret", relu);
  bindings.allocate(save->getPlaceholder());

  quantization::QuantizationConfiguration quantConfig{
      {{NodeQuantizationInfo::generateNodeOutputName(input->getName()),
        {0.2f, 0}},
       {NodeQuantizationInfo::generateNodeOutputName(relu->getName()),
        {0.2f, -128}}}};
  quantConfig.precision = ElemKind::Int16QTy;
  quantConfig.assertAllNodesQuantized = true;

  // Expect this to die because quantizeFunction() is passed with
  // assertAllNodesQuantized true, and the Interpreter backend does not support
  // Int16QTy ReLU.
  Function *QF = F->clone("quant_clone1");
  EXPECT_DEATH(
      quantization::quantizeFunction(QF, quantConfig, *EE.getBackend()), "");

  {
    Function *QF = F->clone("quant_clone2");
    quantConfig.assertAllNodesQuantized = false;

    // This works fine because quantizeFunction() is passed with
    // assertAllNodesQuantized false, and so the ReLU will not be quantized as
    // the Interpreter does not support Int16QTy ReLU.
    quantization::quantizeFunction(QF, quantConfig, *EE.getBackend());

    auto *saveNode = llvm::dyn_cast<SaveNode>(QF->getNodeByName("ret"));
    ASSERT_TRUE(saveNode);
    auto *reluNode = llvm::dyn_cast<ReluNode>(saveNode->getInput().getNode());
    ASSERT_TRUE(reluNode);
    EXPECT_TRUE(!reluNode->getResult().getType()->isQuantizedType());
  }

  {
    Function *QF = F->clone("quant_clone3");
    quantConfig.assertAllNodesQuantized = true;
    KindSet doNotQuantizeKinds;
    doNotQuantizeKinds.insert(Kinded::Kind::ReluNodeKind);

    // This works fine because quantizeFunction() is passed with
    // assertAllNodesQuantized true, but we explicitly tell the quantizer to
    // keep ReLU in its original precision.
    quantization::quantizeFunction(QF, quantConfig, *EE.getBackend(),
                                   /* loweredMap */ {}, doNotQuantizeKinds);

    auto *saveNode = llvm::dyn_cast<SaveNode>(QF->getNodeByName("ret"));
    ASSERT_TRUE(saveNode);
    auto *reluNode = llvm::dyn_cast<ReluNode>(saveNode->getInput().getNode());
    ASSERT_TRUE(reluNode);
    EXPECT_TRUE(!reluNode->getResult().getType()->isQuantizedType());
  }
}

/// Check that we can quantize nodes that have some quantized outputs as unused,
/// e.g. a TopK node where values is unused but indices is.
TEST(Quantization, QuantizationZeroUsersResult) {
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  PlaceholderBindings bindings;
  Function *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {3, 1, 5}, "input", false);

  bindings.allocate(input)->getHandle() = {
      28, 4, 411, 19, 42, 0.4f, 0.4f, 0.4f, -0.4f, 0.45f, 7, 5, 9, 8, 100,
  };

  // Note we intentionally do not save the topk's values result.
  auto *TK = F->createTopK("TopK", input, 3);
  auto *SN = F->createSave("save_indices", TK->getIndices());
  bindings.allocate(SN->getPlaceholder());

  quantization::QuantizationConfiguration quantConfig{
      {{NodeQuantizationInfo::generateNodeOutputName(input->getName()),
        {0.2f, 0}},
       {NodeQuantizationInfo::generateNodeOutputName(TK->getName(),
                                                     TopKNode::ValuesIdx),
        {0.2f, 0}}}};
  quantConfig.assertAllNodesQuantized = true;

  quantization::quantizeFunction(F, quantConfig, *EE.getBackend());

  auto *qSN = llvm::dyn_cast<SaveNode>(F->getNodeByName("save_indices"));
  ASSERT_TRUE(qSN);
  auto *qTK = llvm::dyn_cast<TopKNode>(qSN->getInput().getNode());
  ASSERT_TRUE(qTK);
  EXPECT_TRUE(qTK->getValues().getType()->isQuantizedType());
}

INSTANTIATE_TEST_CASE_P(Interpreter, Quantization,
                        ::testing::Values("Interpreter"));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(CPU, Quantization, ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
    InterpAndCPUProfAndQuant, Operator,
    ::testing::Combine(::testing::Values("Interpreter", "CPU"),
                       ::testing::Values("Interpreter", "CPU")));

INSTANTIATE_TEST_CASE_P(
    InterpAndCPUProfAndQuant, InterpAndCPU,
    ::testing::Combine(::testing::Values("Interpreter", "CPU"),
                       ::testing::Values("Interpreter", "CPU")));

#else
INSTANTIATE_TEST_CASE_P(InterpreterProfAndQuant, Operator,
                        ::testing::Combine(::testing::Values("Interpreter"),
                                           ::testing::Values("Interpreter")));

INSTANTIATE_TEST_CASE_P(Interpreter, InterpAndCPU,
                        ::testing::Combine(::testing::Values("Interpreter"),
                                           ::testing::Values("Interpreter")));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(InterpProfOpenCLQuant, Operator,
                        ::testing::Combine(::testing::Values("Interpreter"),
                                           ::testing::Values("OpenCL")));
#endif // GLOW_WITH_OPENCL

} // namespace glow
