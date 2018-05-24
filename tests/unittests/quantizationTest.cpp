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

TEST(Quantization, quantizeGraph) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 3}, "input");
  auto *W = mod.createVariable(ElemKind::FloatTy, {3, 3}, "weights",
                               VisibilityKind::Private,
                               Variable::TrainKind::Xavier, 3);
  auto *B = mod.createVariable(ElemKind::FloatTy, {3}, "bias",
                               VisibilityKind::Private,
                               Variable::TrainKind::Broadcast, 0.1);
  auto *FC = F->createFullyConnected("FC", input, W, B);
  F->createSave("ret", FC);

  std::vector<NodeQuantizationInfo> QI{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.2, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(W->getName()), {0.3, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(B->getName()), {0.4, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(FC->getName()), {0.6, 0}},
  };

  F = quantization::quantizeFunction(EE, QI, F);

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});
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
static Function *createSimpleGraphForQuantization(Module *M, Variable *A,
                                                  Variable *B,
                                                  llvm::StringRef funcName) {
  Function *F = M->createFunction(funcName);

  fillStableRandomData(A->getHandle(), 1100, 1);

  fillStableRandomData(B->getHandle(), 2001, 1);

  ConvolutionNode *CV = F->createConv("conv", A, 16, 5, 1, 2, 2);
  Variable *bias = cast<Variable>(CV->getBias());
  Variable *filter = cast<Variable>(CV->getFilter());

  fillStableRandomData(bias->getPayload().getHandle(), 2001, 1);
  fillStableRandomData(filter->getPayload().getHandle(), 1000, 1);

  auto *RL = F->createRELU("relu", CV);
  auto *MP = F->createPoolMax("maxPool", RL, 2, 2, 1);
  // Just add noop transpose.
  auto *T = F->createTranspose("transpose", MP, {0, 1, 2, 3});
  // Noop reshape, make sure conversion quantization procedure works well.
  auto *R = F->createReshape("reshape", T, T->getResult().dims());
  auto *AP = F->createPoolAvg("avgPool", R, 2, 2, 1);

  FullyConnectedNode *FC = F->createFullyConnected("fc", AP, 10);

  // Noop slice, make sure conversion quantization procedure works well.
  auto *S =
      F->createSlice("slice", FC, {0, 1},
                     {FC->getResult().dims()[0], FC->getResult().dims()[1]});
  Variable *bias2 = cast<Variable>(FC->getBias());
  Variable *filter2 = cast<Variable>(FC->getWeights());

  fillStableRandomData(bias2->getPayload().getHandle(), 3001, 1);
  fillStableRandomData(filter2->getPayload().getHandle(), 4000, 1);

  auto *O = F->createConcat("concat", {S, B}, 0);
  F->createSave("save", O);
  return F;
}

TEST_P(Quantization, end2end) {
  auto *mod = &interpreterEE.getModule();

  auto *A =
      mod->createVariable(ElemKind::FloatTy, {1, 32, 32, 2}, "A",
                          VisibilityKind::Public, Variable::TrainKind::None);
  auto *B =
      mod->createVariable(ElemKind::FloatTy, {10, 9}, "B",
                          VisibilityKind::Public, Variable::TrainKind::None);

  // STEP1 - Generate the first network to record the quantization parameters.
  Function *F1 = createSimpleGraphForQuantization(mod, A, B, "main");
  Function *F2 = F1->clone("main2");
  SaveNode *result1 = cast<SaveNode>(F1->getNodeByName("save"));

  F1 = glow::profileQuantization(F1);
  interpreterEE.compile(CompilationMode::Infer, F1);

  // Run graph to capture profile.
  interpreterEE.run({}, {});

  // Get quantization infos and build new quantized graph.
  std::vector<NodeQuantizationInfo> QI =
      quantization::generateNodeQuantizationInfos(F1);

  // STEP2 - Use the profile to quantize a network.
  SaveNode *result2 = cast<SaveNode>(F2->getNodeByName("save"));

  F2 = quantization::quantizeFunction(backendSpecificEE, QI, F2);
  backendSpecificEE.compile(CompilationMode::Infer, F2);
  backendSpecificEE.run({}, {});

  // STEP3 - Compare the results of the original and quantized functions.
  auto result1Handle = result1->getVariable()->getHandle();
  auto result2Handle = result2->getVariable()->getHandle();

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
static void fillStableRandomIndex(Handle<size_t> H, size_t seed,
                                  size_t scale = 10) {
  for (size_t i = 0, e = H.size(); i < e; i++) {
    H.raw(i) = int(i * 1921 + seed) % scale;
  }
}

/// Builds a graph with two GRUs and saves output from last hidden node.
static Function *createGRUForQuantization(Module *M, llvm::StringRef funcName) {
  Function *F = M->createFunction(funcName);

  constexpr unsigned sequenceSize = 2;
  constexpr unsigned embeddingSize = 10;
  constexpr unsigned languageSize = 10;
  constexpr unsigned batchSize = 5;
  constexpr unsigned hiddenSize = 3 * embeddingSize;

  // STEP1 - Initialize inputs into GRU
  auto *emb = F->getParent()->createVariable(
      ElemKind::FloatTy, {languageSize, embeddingSize}, "embedding",
      VisibilityKind::Public, Variable::TrainKind::None);
  fillStableRandomData(emb->getHandle(), 4565, 1);

  auto *input = F->getParent()->createVariable(
      ElemKind::IndexTy, {batchSize, sequenceSize}, "input",
      VisibilityKind::Public, Variable::TrainKind::None);
  fillStableRandomIndex(input->getHandle<size_t>(), 7227, 10);

  auto *hiddenInit = F->getParent()->createVariable(
      ElemKind::FloatTy, {batchSize, embeddingSize}, "hiddenInit",
      VisibilityKind::Public, Variable::TrainKind::None);
  hiddenInit->getPayload().zero();
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

    auto *FCi = F->createFullyConnected("gru.fci", reshape, hiddenSize);
    Variable *biasI = cast<Variable>(FCi->getBias());
    Variable *filterI = cast<Variable>(FCi->getWeights());
    fillStableRandomData(biasI->getPayload().getHandle(), 8877, 1);
    fillStableRandomData(filterI->getPayload().getHandle(), 1441, 1);

    auto *FCh = F->createFullyConnected("gru.fch", hidden, hiddenSize);
    Variable *biasH = cast<Variable>(FCh->getBias());
    Variable *filterH = cast<Variable>(FCh->getWeights());
    fillStableRandomData(biasH->getPayload().getHandle(), 9009, 1);
    fillStableRandomData(filterH->getPayload().getHandle(), 1001, 1);

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

  F->createSave("save", {downsample, 0});
  return F;
}

TEST_P(Quantization, end2endGRU) {
  // STEP1 - Generate the first network to record the quantization parameters.
  auto *mod = &interpreterEE.getModule();
  Function *F1 = createGRUForQuantization(mod, "main");
  Function *F2 = F1->clone("main2");
  SaveNode *result1 = cast<SaveNode>(F1->getNodeByName("save"));

  F1 = glow::profileQuantization(F1);
  interpreterEE.compile(CompilationMode::Infer, F1);

  // Run graph to capture profile.
  interpreterEE.run({}, {});

  // Get quantization infos and build new quantized graph.
  std::vector<NodeQuantizationInfo> QI =
      quantization::generateNodeQuantizationInfos(F1);

  // STEP2 - Use the profile to quantize a network.
  SaveNode *result2 = cast<SaveNode>(F2->getNodeByName("save"));

  F2 = quantization::quantizeFunction(backendSpecificEE, QI, F2);
  backendSpecificEE.compile(CompilationMode::Infer, F2);
  backendSpecificEE.run({}, {});

  // STEP3 - Compare the results of the original and quantized functions.
  auto result1Handle = result1->getVariable()->getHandle();
  auto result2Handle = result2->getVariable()->getHandle();

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
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("foo");
  auto *input = mod.createVariable(ElemKind::Int8QTy, {1, 1}, 0.5, 11, "input",
                                   VisibilityKind::Public,
                                   Variable::TrainKind::Broadcast, 21);
  auto *Q = F->createRescaleQuantized(
      "rescale", input, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.5, 11));
  auto *D = F->createDequantize("dequantize", Q);
  auto *result = F->createSave("ret", D);

  EXPECT_EQ(F->getNodes().size(), 3);
  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});
  EXPECT_EQ(F->getNodes().size(), 2);

  auto RH = result->getVariable()->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 5.0, 0.001);
}

TEST(Quantization, optimizeRescaleQuantize) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("foo");
  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 1}, "input",
                                   VisibilityKind::Public,
                                   Variable::TrainKind::Broadcast, 21);
  auto *Q = F->createQuantize(
      "quant", input, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.25, 4));
  auto *RS = F->createRescaleQuantized(
      "rescale", Q, mod.uniqueType(ElemKind::Int8QTy, {1, 1}, 0.5, 11));
  auto *D = F->createDequantize("dequantize", RS);
  auto *result = F->createSave("ret", D);

  EXPECT_EQ(F->getNodes().size(), 4);
  EE.compile(CompilationMode::Infer, F);
  EE.run({}, {});
  EXPECT_EQ(F->getNodes().size(), 1);

  auto RH = result->getVariable()->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 21.0, 0.001);
}

INSTANTIATE_TEST_CASE_P(Interpreter, Quantization,
                        ::testing::Values(BackendKind::Interpreter));

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(JIT, Quantization, ::testing::Values(BackendKind::CPU));
#endif // GLOW_WITH_CPU

} // namespace glow
