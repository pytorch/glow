// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Quantization/Quantization.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Quantization/Serialization.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"

namespace glow {

using llvm::cast;

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
                               Variable::VisibilityKind::Private,
                               Variable::TrainKind::Xavier, 3);
  auto *B = mod.createVariable(ElemKind::FloatTy, {3}, "bias",
                               Variable::VisibilityKind::Private,
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

  quantization::generateQuantizedGraph(F, QI);

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
static std::pair<Function *, SaveNode *>
createSimpleGraphForQuantization(Module *M) {
  Function *F = M->createFunction("main");

  auto *A = F->getParent()->createVariable(
      ElemKind::FloatTy, {1, 32, 32, 3}, "A", Variable::VisibilityKind::Public,
      Variable::TrainKind::None);
  fillStableRandomData(A->getHandle(), 1100, 1);

  auto *B = F->getParent()->createVariable(ElemKind::FloatTy, {10, 10}, "B",
                                           Variable::VisibilityKind::Public,
                                           Variable::TrainKind::None);
  fillStableRandomData(B->getHandle(), 2001, 1);

  ConvolutionNode *CV = F->createConv("conv", A, 16, 5, 1, 2);
  Variable *bias = cast<Variable>(CV->getBias());
  Variable *filter = cast<Variable>(CV->getFilter());

  fillStableRandomData(bias->getPayload().getHandle(), 2001, 1);
  fillStableRandomData(filter->getPayload().getHandle(), 1000, 1);

  auto *RL = F->createRELU("relu", CV);
  auto *MP = F->createPoolMax("maxPool", RL, 2, 2, 1);
  // Just add noop transpose.
  auto *T = F->createTranspose("transpose", MP, {0, 1, 2, 3});
  // Noop reshape.
  auto *R = F->createReshape("reshape", T, T->getResult().dims());
  auto *AP = F->createPoolAvg("avgPool", R, 2, 2, 1);

  FullyConnectedNode *FC = F->createFullyConnected("fc", AP, 10);
  Variable *bias2 = cast<Variable>(FC->getBias());
  Variable *filter2 = cast<Variable>(FC->getWeights());

  fillStableRandomData(bias2->getPayload().getHandle(), 3001, 1);
  fillStableRandomData(filter2->getPayload().getHandle(), 4000, 1);

  auto *O = F->createConcat("concat", {FC, B}, 0);
  SaveNode *SN = F->createSave("save", O);
  return {F, SN};
}

TEST(Quantization, end2end) {
  // STEP1 - Generate the first network to record the quantization parameters.
  ExecutionEngine EE;
  auto res = createSimpleGraphForQuantization(&EE.getModule());
  Function *F1 = res.first;
  SaveNode *result1 = res.second;

  glow::profileQuantization(F1);
  EE.compile(CompilationMode::Infer, F1);

  // Run graph to capture profile.
  EE.run({}, {});

  // Get quantization infos and build new quantized graph.
  std::vector<NodeQuantizationInfo> QI =
      quantization::generateNodeQuantizationInfos(F1);

  // STEP2 - Use the profile to quantize a network.
  ExecutionEngine EE2;
  auto res2 = createSimpleGraphForQuantization(&EE2.getModule());
  Function *F2 = res2.first;
  SaveNode *result2 = res2.second;

  quantization::generateQuantizedGraph(F2, QI);
  EE2.compile(CompilationMode::Infer, F2);
  EE2.run({}, {});

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
                                   Variable::VisibilityKind::Public,
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
                                   Variable::VisibilityKind::Public,
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

} // namespace glow
