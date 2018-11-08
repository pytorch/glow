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

#include "glow/Quantization/Quantization.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Serialization.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"

namespace glow {

using llvm::cast;

class Quantization : public ::testing::TestWithParam<BackendKind> {
protected:
  ExecutionEngine interpreterEE{BackendKind::Interpreter};
  ExecutionEngine backendSpecificEE{GetParam()};
};

class Operator : public ::testing::TestWithParam<BackendKind> {
protected:
  ExecutionEngine interpreterEE{BackendKind::Interpreter};
  ExecutionEngine backendSpecificEE{GetParam()};
};

bool operator==(const NodeQuantizationInfo &lhs,
                const NodeQuantizationInfo &rhs) {
  return lhs.Scale() == rhs.Scale() && lhs.Offset() == rhs.Offset() &&
         lhs.nodeOutputName_ == rhs.nodeOutputName_;
}

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

TEST(Quantization, SerializeEmpty) {
  std::vector<NodeQuantizationInfo> expected;

  testSerialization(expected);
}

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

TEST(Quantization, quantizeTensor) {
  // Map float [0.0; 5.0] to int [-128; 127].
  // With symmetric mapping, we basically map [-5.0; 5.0]
  TensorQuantizationParams symmetricParams =
      chooseQuantizationParams(0.0, 6.0, quantization::Schema::Symmetric);

  // Create an FP32 tensor with 6 elements and initialize it with numbers from 0
  // to 5.
  Tensor inputFP32(ElemKind::FloatTy, {6});
  Handle<float> THFP32 = inputFP32.getHandle<float>();
  for (unsigned i = 0; i < 6; ++i) {
    THFP32.at({i}) = i * 1.0f;
  }

  // Quantize the tensor.
  auto quantizedFP32 = quantization::quantizeTensor(inputFP32, symmetricParams);
  // Check that the dequantized result is close to the original values before
  // the quantization.
  Handle<int8_t> THquantizedFP32 = quantizedFP32.getHandle<int8_t>();
  for (unsigned i = 0; i < 6; ++i) {
    EXPECT_NEAR(
        THFP32.at({i}),
        quantization::dequantize(THquantizedFP32.at({i}), symmetricParams),
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
  auto quantizedFP16 = quantization::quantizeTensor(inputFP16, symmetricParams);
  // Check that the dequantized result is close to the original values before
  // the quantization.
  Handle<int8_t> THquantizedFP16 = quantizedFP16.getHandle<int8_t>();
  for (unsigned i = 0; i < 6; ++i) {
    EXPECT_NEAR(
        THFP16.at({i}),
        quantization::dequantize(THquantizedFP16.at({i}), symmetricParams),
        0.05f);
  }
}

TEST(Quantization, quantizeGraph) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *W = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "weights", true);
  auto *B = mod.createPlaceholder(ElemKind::FloatTy, {3}, "bias", true);
  Context ctx;
  ctx.allocate(input);
  ctx.allocate(W)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  ctx.allocate(B)->init(Tensor::InitKind::Broadcast, 0.1, mod.getPRNG());

  auto *FC = F->createFullyConnected("FC", input, W, B);
  auto *S = F->createSave("ret", FC);
  ctx.allocate(S->getPlaceholder());

  std::vector<NodeQuantizationInfo> QI{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.2f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(W->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(B->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(FC->getName()), {0.6f, 0}},
  };

  F = quantization::quantizeFunction(EE, QI, F);

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run(ctx);
}

/// Quantize ReLU node and make sure that quantized version
/// has quantization parameters mapping to non-negative floating
/// point range.
TEST(Quantization, quantizeReLU) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 3}, "input", true);
  auto *relu = F->createRELU("ReLU", input);
  Context ctx;
  F->createSave("ret", relu);
  // Make sure that offset quantization parameter of ReLU is set
  // such that it produces non-negative floating point range.
  std::vector<NodeQuantizationInfo> QI{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.2f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(relu->getName()),
       {0.2f, -128}}};
  F = quantization::quantizeFunction(EE, QI, F);
  EE.compile(CompilationMode::Infer, F, ctx);

  auto *save = llvm::cast<SaveNode>(F->getNodeByName("ret"));
  ASSERT_TRUE(llvm::isa<DequantizeNode>(save->getInput().getNode()));
  auto *dequantize = llvm::cast<DequantizeNode>(save->getInput().getNode());
  ASSERT_TRUE(llvm::isa<MaxNode>(dequantize->getInput().getNode()));

  MaxNode *max = llvm::cast<MaxNode>(dequantize->getInput().getNode());
  ASSERT_TRUE(max->getResult().getType()->isQuantizedType());
  EXPECT_EQ(max->getResult().getType()->getOffset(), -128);
  EXPECT_EQ(max->getResult().getType()->getScale(), 0.2f);
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
static Function *createSimpleGraphForQuantization(Module *M, Context &ctx,
                                                  Placeholder *A,
                                                  Placeholder *B,
                                                  llvm::StringRef funcName) {
  Function *F = M->createFunction(funcName);

  fillStableRandomData(ctx.allocate(A)->getHandle(), 1100, 1);

  fillStableRandomData(ctx.allocate(B)->getHandle(), 2001, 1);

  ConvolutionNode *CV = F->createConv(ctx, "conv", A, 16, 5, 1, 2, 2);
  auto *bias = cast<Placeholder>(CV->getBias());
  auto *filter = cast<Placeholder>(CV->getFilter());
  fillStableRandomData(ctx.get(bias)->getHandle(), 2001, 1);
  fillStableRandomData(ctx.get(filter)->getHandle(), 1000, 1);

  auto *RL = F->createRELU("relu", CV);
  auto *MP = F->createMaxPool("maxPool", RL, 2, 2, 1);
  // Just add noop transpose.
  auto *T = F->createTranspose("transpose", MP, {0, 1, 2, 3});
  // Noop reshape, make sure conversion quantization procedure works well.
  auto *R = F->createReshape("reshape", T, T->getResult().dims());
  auto *AP = F->createAvgPool("avgPool", R, 2, 2, 1);

  FullyConnectedNode *FC = F->createFullyConnected(ctx, "fc", AP, 10);

  // Noop slice, make sure conversion quantization procedure works well.
  auto *S =
      F->createSlice("slice", FC, {0, 1},
                     {FC->getResult().dims()[0], FC->getResult().dims()[1]});
  auto *bias2 = cast<Placeholder>(FC->getBias());
  auto *filter2 = cast<Placeholder>(FC->getWeights());

  fillStableRandomData(ctx.get(bias2)->getHandle(), 3001, 1);
  fillStableRandomData(ctx.get(filter2)->getHandle(), 4000, 1);

  auto *CN = F->createConcat("concat", {S, B}, 0);
  auto *SP = F->createSplat("splat", B->getType(), 10.0);
  auto *O = F->createConcat("concat", {CN, SP}, 0);
  auto *TN = F->createTranspose("transpose", O, {1, 0});
  auto *MMN = F->createMatMul("batchedreduceadd", O, TN);
  auto *BRAN = F->createBatchedReduceAdd("batchedreduceadd", MMN, 0);
  auto *TLN = F->createTile("tile", BRAN, 2, 0);
  auto *save = F->createSave("save", TLN);
  ctx.allocate(save->getPlaceholder());
  return F;
}

TEST_P(Operator, end2end) {
  auto *mod = &interpreterEE.getModule();
  Context ctx;

  auto *A =
      mod->createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 2}, "A", false);
  auto *B = mod->createPlaceholder(ElemKind::FloatTy, {10, 9}, "B", false);

  // STEP1 - Generate the first network to record the quantization parameters.
  Function *F1 = createSimpleGraphForQuantization(mod, ctx, A, B, "main");
  Function *F2 = F1->clone("main2");
  SaveNode *result1 = cast<SaveNode>(F1->getNodeByName("save"));
  F1 = glow::profileQuantization(ctx, F1);
  interpreterEE.compile(CompilationMode::Infer, F1, ctx);

  // Run graph to capture profile.
  interpreterEE.run(ctx);

  // Get quantization infos and build new quantized graph.
  std::vector<NodeQuantizationInfo> QI =
      quantization::generateNodeQuantizationInfos(ctx, F1);

  // STEP2 - Use the profile to quantize a network.
  SaveNode *result2 = cast<SaveNode>(F2->getNodeByName("save"));

  F2 = quantization::quantizeFunction(backendSpecificEE, QI, F2);
  backendSpecificEE.compile(CompilationMode::Infer, F2, ctx);
  backendSpecificEE.run(ctx);

  // STEP3 - Compare the results of the original and quantized functions.
  auto result1Handle = ctx.get(result1->getPlaceholder())->getHandle();
  auto result2Handle = ctx.get(result2->getPlaceholder())->getHandle();

  EXPECT_EQ(result1Handle.size(), result2Handle.size());

  for (int i = 0, e = result1Handle.size(); i < e; ++i) {
    float mx = result2Handle.raw(result2Handle.minMaxArg().second);
    double diff = std::fabs(result2Handle.raw(i) - result1Handle.raw(i)) / mx;

    // Allow 3% difference.
    EXPECT_NEAR(diff, 0, 0.03);
  }
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
static Function *createGRUForQuantization(Module *M, Context &ctx,
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
  fillStableRandomData(ctx.allocate(emb)->getHandle(), 4565, 1);

  auto *input = F->getParent()->createPlaceholder(
      ElemKind::Int64ITy, {batchSize, sequenceSize}, "input", false);
  fillStableRandomIndex(ctx.allocate(input)->getHandle<int64_t>(), 7227, 10);

  auto *hiddenInit = F->getParent()->createPlaceholder(
      ElemKind::FloatTy, {batchSize, embeddingSize}, "hiddenInit", false);
  ctx.allocate(hiddenInit)->zero();
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

    auto *FCi = F->createFullyConnected(ctx, "gru.fci", reshape, hiddenSize);
    auto *biasI = cast<Placeholder>(FCi->getBias());
    auto *filterI = cast<Placeholder>(FCi->getWeights());
    fillStableRandomData(ctx.get(biasI)->getHandle(), 8877, 1);
    fillStableRandomData(ctx.get(filterI)->getHandle(), 1441, 1);

    auto *FCh = F->createFullyConnected(ctx, "gru.fch", hidden, hiddenSize);
    auto *biasH = cast<Placeholder>(FCh->getBias());
    auto *filterH = cast<Placeholder>(FCh->getWeights());
    fillStableRandomData(ctx.get(biasH)->getHandle(), 9009, 1);
    fillStableRandomData(ctx.get(filterH)->getHandle(), 1001, 1);

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
  ctx.allocate(save->getPlaceholder());
  return F;
}

TEST_P(Operator, end2endGRU) {
  // STEP1 - Generate the first network to record the quantization parameters.
  auto *mod = &interpreterEE.getModule();
  Context ctx;
  Function *F1 = createGRUForQuantization(mod, ctx, "main");
  Function *F2 = F1->clone("main2");
  SaveNode *result1 = cast<SaveNode>(F1->getNodeByName("save"));

  F1 = glow::profileQuantization(ctx, F1);
  interpreterEE.compile(CompilationMode::Infer, F1, ctx);

  // Run graph to capture profile.
  interpreterEE.run(ctx);

  // Get quantization infos and build new quantized graph.
  std::vector<NodeQuantizationInfo> QI =
      quantization::generateNodeQuantizationInfos(ctx, F1);

  // STEP2 - Use the profile to quantize a network.
  SaveNode *result2 = cast<SaveNode>(F2->getNodeByName("save"));

  F2 = quantization::quantizeFunction(backendSpecificEE, QI, F2);
  backendSpecificEE.compile(CompilationMode::Infer, F2, ctx);
  backendSpecificEE.run(ctx);

  // STEP3 - Compare the results of the original and quantized functions.
  auto result1Handle = ctx.get(result1->getPlaceholder())->getHandle();
  auto result2Handle = ctx.get(result2->getPlaceholder())->getHandle();

  EXPECT_EQ(result1Handle.size(), result2Handle.size());

  for (int i = 0, e = result1Handle.size(); i < e; ++i) {
    float mx = result2Handle.raw(result2Handle.minMaxArg().second);
    double diff = std::fabs(result2Handle.raw(i) - result1Handle.raw(i)) / mx;

    // Allow 3% difference.
    EXPECT_NEAR(diff, 0, 0.03);
  }
}

TEST(Quantization, rescaleSameType) {
  ExecutionEngine EE;
  Context ctx;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("foo");
  auto *input =
      mod.createPlaceholder(ElemKind::Int8QTy, {1, 1}, 0.5, 11, "input", true);
  ctx.allocate(input)->init(Tensor::InitKind::Broadcast, 21, mod.getPRNG());

  auto *Q = F->createRescaleQuantized(
      "rescale", input, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.5, 11));
  auto *D = F->createDequantize("dequantize", Q);
  auto *save = F->createSave("ret", D);
  auto *result = ctx.allocate(save->getPlaceholder());

  EXPECT_EQ(F->getNodes().size(), 3);
  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run(ctx);
  EXPECT_EQ(F->getNodes().size(), 2);

  auto RH = result->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 5.0, 0.001);
}

TEST(Quantization, optimizeRescaleQuantize) {
  ExecutionEngine EE;
  Context ctx;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("foo");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 1}, "input", true);
  ctx.allocate(input)->init(Tensor::InitKind::Broadcast, 21, mod.getPRNG());

  auto *Q = F->createQuantize(
      "quant", input, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.25, 4));
  auto *RS = F->createRescaleQuantized(
      "rescale", Q, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.5, 11));
  auto *D = F->createDequantize("dequantize", RS);
  auto *save = F->createSave("ret", D);
  auto *result = ctx.allocate(save->getPlaceholder());

  EXPECT_EQ(F->getNodes().size(), 4);
  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run(ctx);
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

/// Check that Relu can use our symmetric quantization schema.
TEST(Quantization, reluCanUseSymmetricSchema) {
  Context ctx;
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Placeholder *input =
      mod.createPlaceholder(ElemKind::FloatTy, {10}, "input", false);
  auto *inputTensor = ctx.allocate(input);
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
  auto *res = ctx.allocate(SN->getPlaceholder());

  EE.compile(CompilationMode::Infer, F, ctx);
  EE.run(ctx);

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
      0.0, 6.0, quantization::Schema::SymmetricWithUInt8);
  // Given this is a purely positive range, we should use uint8,
  // thus int8 - (-128).
  EXPECT_EQ(symmetricParams.offset, -128);
  EXPECT_NEAR(symmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-3.0; 3.0] to int [-128; 127].
  symmetricParams = chooseQuantizationParams(
      -3.0, 3.0, quantization::Schema::SymmetricWithUInt8);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 6.0 / 255, 0.001);

  // Map float [-2.0; 5.0] to int [-128; 127].
  // This has negative value, thus we fall back to purely symmetric.
  // => [-5.0; 5.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      -2.0, 5.0, quantization::Schema::SymmetricWithUInt8);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 10.0 / 255, 0.001);

  // Map float [0; 0] to int [-128; 127].
  symmetricParams = chooseQuantizationParams(
      0.0, 0.0, quantization::Schema::SymmetricWithUInt8);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 0.1, 0.001);

  // Map float [2.0; 5.0] to int [-128; 127].
  // All positive, using uint8.
  // However, our quantization schemas always include zero.
  // => [0.0; 5.0] range for uint8 mode.
  symmetricParams = chooseQuantizationParams(
      2.0, 5.0, quantization::Schema::SymmetricWithUInt8);
  // Scale: (5.0 - (0.0)) / (127 - (-128)) == 5.0 / 255.0
  // Offset from min: scale(-128 - offset) == 0.0
  EXPECT_EQ(symmetricParams.offset, -128);
  EXPECT_NEAR(symmetricParams.scale, 5.0 / 255, 0.001);

  // Map float [-8.0; -2.0] to int [-128; 127].
  // => [-8.0; 8.0] range for symmetric mode.
  symmetricParams = chooseQuantizationParams(
      -8.0, -2.0, quantization::Schema::SymmetricWithUInt8);
  EXPECT_EQ(symmetricParams.offset, 0);
  EXPECT_NEAR(symmetricParams.scale, 16.0 / 255, 0.001);
}

/// This is a mock backend which extended support of quantized operators.
class MockQuantBackend : public Backend {
  // The actual backend being wrapped.
  std::unique_ptr<Backend> backend_;

public:
  MockQuantBackend() {
    backend_.reset(createBackend(BackendKind::Interpreter));
  }
  std::unique_ptr<CompiledFunction> compile(Function *F,
                                            const Context &ctx) const override {
    return backend_->compile(F, ctx);
  }
  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override {
    if (opKind == Kinded::Kind::SoftMaxNodeKind ||
        opKind == Kinded::Kind::LocalResponseNormalizationNodeKind) {
      return true;
    }
    return backend_->isOpSupported(opKind, elementTy);
  }
};

/// Check that LRN and Softmax are quantized.
TEST(Quantization, quantizeSoftmaxAndLRN) {
  ExecutionEngine EE;
  Context ctx;
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
  F->createSave("ret", SM);

  std::vector<NodeQuantizationInfo> QI{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.2f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(LRN->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(SM->getName()), {0.4f, 0}},
  };

  F = quantization::quantizeFunction(EE, QI, F);

  auto qLRNIt = std::find_if(
      F->getNodes().begin(), F->getNodes().end(), [](const Node &node) -> bool {
        return llvm::isa<LocalResponseNormalizationNode>(&node) &&
               node.getNthResult(0).getType()->isQuantizedType();
      });
  ASSERT_NE(qLRNIt, F->getNodes().end());
  auto qSMIt = std::find_if(
      F->getNodes().begin(), F->getNodes().end(), [](const Node &node) -> bool {
        return llvm::isa<SoftMaxNode>(&node) &&
               node.getNthResult(0).getType()->isQuantizedType();
      });
  ASSERT_NE(qSMIt, F->getNodes().end());
}

/// Test option to disable quantization of specific node kinds in the graph.
TEST(Quantization, quantizeGraphPartially) {
  ExecutionEngine EE;
  Context ctx;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", true);
  ctx.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  ctx.allocate(RHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *MMN = F->createMatMul("matmul", LHS, RHS);
  auto *TN = F->createTanh("tanh", MMN);
  auto *save = F->createSave("ret", TN);
  auto *result = save->getPlaceholder();
  ctx.allocate(result);

  // Note that we are creating quantization info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  std::vector<NodeQuantizationInfo> QI{
      {NodeQuantizationInfo::generateNodeOutputName(LHS->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(RHS->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(MMN->getName()), {0.6f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(TN->getName()), {0.5f, 0}},
  };

  // Do not quantize any tanh nodes.
  KindSet doNotQuantize;
  doNotQuantize.insert(Kinded::Kind::TanhNodeKind);

  auto *QF =
      quantization::quantizeFunction(EE, QI, F, "_quantized", doNotQuantize);
  QF->getParent()->eraseFunction(F);
  F = QF;

  // Make sure that graph can be compiled and run.
  ::glow::convertPlaceholdersToConstants(F, ctx, {result});
  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run(ctx);

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
  ExecutionEngine EE;
  Context ctx;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", true);
  ctx.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  ctx.allocate(RHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *TNLHS = F->createTanh("tanh", LHS);
  auto *MMN = F->createMatMul("matmul", TNLHS, RHS);
  auto *TN = F->createTanh("tanh", MMN);
  auto *save = F->createSave("ret", TN);
  auto *result = save->getPlaceholder();
  ctx.allocate(result);

  // Note that we are creating quantization info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  std::vector<NodeQuantizationInfo> QI{
      {NodeQuantizationInfo::generateNodeOutputName(LHS->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(TNLHS->getName()),
       {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(RHS->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(MMN->getName()), {0.6f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(TN->getName()), {0.5f, 0}},
  };

  // Do not quantize any tanh nodes.
  KindSet doNotQuantize;
  doNotQuantize.insert(Kinded::Kind::TanhNodeKind);

  auto *QF =
      quantization::quantizeFunction(EE, QI, F, "_quantized", doNotQuantize);
  QF->getParent()->eraseFunction(F);
  F = QF;

  // Make sure that graph can be compiled and run.
  ::glow::convertPlaceholdersToConstants(F, ctx, {result});
  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run(ctx);

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
  ExecutionEngine EE;
  Context ctx;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", true);
  ctx.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  ctx.allocate(RHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *MMN = F->createMatMul("matmul", LHS, RHS);
  auto *CN = F->createAdd("concat", LHS, MMN);
  auto *TN = F->createTanh("tanh", CN);
  auto *save = F->createSave("ret", TN);
  auto *result = save->getPlaceholder();
  ctx.allocate(result);

  // Note that we are creating quantization info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  std::vector<NodeQuantizationInfo> QI{
      {NodeQuantizationInfo::generateNodeOutputName(LHS->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(RHS->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(MMN->getName()), {0.6f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(CN->getName()), {0.6f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(TN->getName()), {0.5f, 0}},
  };

  // Do not quantize any tanh or add nodes.
  KindSet doNotQuantize;
  doNotQuantize.insert(Kinded::Kind::TanhNodeKind);
  doNotQuantize.insert(Kinded::Kind::AddNodeKind);

  auto *QF =
      quantization::quantizeFunction(EE, QI, F, "_quantized", doNotQuantize);
  QF->getParent()->eraseFunction(F);
  F = QF;

  // Make sure that graph can be compiled and run.
  ::glow::convertPlaceholdersToConstants(F, ctx, {result});
  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run(ctx);

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
  ExecutionEngine EE;
  Context ctx;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", true);
  auto *RHS = mod.createConstant(ElemKind::FloatTy, {3, 3}, "rhs");
  ctx.allocate(LHS)->init(Tensor::InitKind::Xavier, 3, mod.getPRNG());
  RHS->getPayload().init(Tensor::InitKind::Xavier, 3, mod.getPRNG());

  auto *MMN = F->createMatMul("matmul", LHS, RHS);
  auto *save = F->createSave("ret", MMN);
  auto *result = save->getPlaceholder();
  ctx.allocate(result);

  // Note that we are creating quantization info even for nodes that will not be
  // quantized. This is how we expect quantizeFunction() to behave, as
  // quantization profiling will still get a profile for these nodes.
  std::vector<NodeQuantizationInfo> QI{
      {NodeQuantizationInfo::generateNodeOutputName(LHS->getName()), {0.3f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(RHS->getName()), {0.4f, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(MMN->getName()), {0.6f, 0}},
  };

  auto *QF = quantization::quantizeFunction(EE, QI, F, "_quantized");
  QF->getParent()->eraseFunction(F);
  F = QF;

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
  EE.compile(CompilationMode::Infer, F, ctx);

  EE.run(ctx);
}

INSTANTIATE_TEST_CASE_P(Interpreter, Quantization,
                        ::testing::Values(BackendKind::Interpreter));
INSTANTIATE_TEST_CASE_P(Interpreter, Operator,
                        ::testing::Values(BackendKind::Interpreter));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, Quantization, ::testing::Values(BackendKind::CPU));
INSTANTIATE_TEST_CASE_P(JIT, Operator, ::testing::Values(BackendKind::CPU));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, Operator,
                        ::testing::Values(BackendKind::OpenCL));
#endif // GLOW_WITH_CPU

} // namespace glow
