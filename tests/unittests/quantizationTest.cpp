// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Quantization/Quantization.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Quantization/Serialization.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"

namespace glow {

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

      auto TR = quantizeScaleOffset32To8(scale, 0);
      int32_t computed = TR.transform(sum32num);

      EXPECT_NEAR(input, computed, 1);
    }
  }
}

TEST(Quantization, quantizeGraph) {
  ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto &G = *mod.createFunction("main");

  auto *input = mod.createVariable(ElemKind::FloatTy, {1, 3}, "input");
  auto *W = mod.createVariable(ElemKind::FloatTy, {3, 3}, "weights",
                               Variable::VisibilityKind::Private,
                               Variable::TrainKind::Xavier, 3);
  auto *B = mod.createVariable(ElemKind::FloatTy, {3}, "bias",
                               Variable::VisibilityKind::Private,
                               Variable::TrainKind::Broadcast, 0.1);
  auto *FC = G.createFullyConnected("FC", input, W, B);
  G.createSave("ret", FC);

  std::vector<NodeQuantizationInfo> QI{
      {NodeQuantizationInfo::generateNodeOutputName(input->getName()),
       {0.2, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(W->getName()), {0.3, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(B->getName()), {0.4, 0}},
      {NodeQuantizationInfo::generateNodeOutputName(FC->getName()), {0.6, 0}},
  };

  glow::generateQuantizedGraph(G, QI);

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer, &G);
  EE.run({}, {});
}

/// Builds a simple graph, returns back input var and save node through refs.
void createSimpleGraphForQuantization(Function &G, Variable *&input,
                                      SaveNode *&saveNode, Variable *W,
                                      Variable *B) {
  auto *A = G.getParent().createVariable(ElemKind::FloatTy, {1, 32, 32, 3}, "A",
                                         Variable::VisibilityKind::Public,
                                         Variable::TrainKind::None);
  input = A;
  auto *CV = G.createConv("conv", A, 16, 5, 1, 2);
  auto *RL = G.createRELU("relu", CV);
  auto *AP = G.createPool("pool", RL, PoolNode::Mode::Avg, 2, 2, 0);

  Node *O = G.createFullyConnected("fc", AP, W, B);
  saveNode = G.createSave("save", O);
}

TEST(Quantization, end2end) {
  Tensor inputs(ElemKind::FloatTy, {1, 32, 32, 3});
  inputs.getHandle().randomize(0, 2.0);

  ExecutionEngine E1, E2;
  SaveNode *result1, *result2;
  Variable *input1, *input2;

  auto &mod1 = E1.getModule();
  auto &mod2 = E2.getModule();

  auto &G1 = *mod1.createFunction("collect_profile");
  auto &G2 = *mod2.createFunction("use_profile");

  auto *W1 = mod1.createVariable(ElemKind::FloatTy, {4096, 2}, "weights",
                                 Variable::VisibilityKind::Private,
                                 Variable::TrainKind::Xavier, 1);
  auto *B1 = mod1.createVariable(ElemKind::FloatTy, {2}, "bias",
                                 Variable::VisibilityKind::Private,
                                 Variable::TrainKind::Xavier, 1);
  createSimpleGraphForQuantization(G1, input1, result1, W1, B1);

  glow::profileQuantization(G1);
  E1.compile(CompilationMode::Infer, &G1);

  // Run graph to capture profile.
  E1.run({input1}, {&inputs});

  // Get quantization infos and build new quantized graph.
  std::vector<NodeQuantizationInfo> QI = generateNodeQuantizationInfos(G1);
  auto *W2 = mod2.createVariable(ElemKind::FloatTy, {4096, 2}, "weights",
                                 Variable::VisibilityKind::Private,
                                 Variable::TrainKind::Xavier, 1);
  auto *B2 = mod2.createVariable(ElemKind::FloatTy, {2}, "bias",
                                 Variable::VisibilityKind::Private,
                                 Variable::TrainKind::Xavier, 1);

  // Make sure we are testing with the same input tensors.
  W2->getPayload().copyFrom(&W1->getPayload());
  B2->getPayload().copyFrom(&B1->getPayload());
  createSimpleGraphForQuantization(G2, input2, result2, W2, B2);

  glow::generateQuantizedGraph(G2, QI);
  E2.compile(CompilationMode::Infer, &G2);
  E2.run({input2}, {&inputs});

  auto result2Handle = result2->getVariable()->getHandle();
  auto result1Handle = result1->getVariable()->getHandle();

  EXPECT_EQ(result1Handle.size(), result2Handle.size());
  for (int i = 0, e = result1Handle.size(); i < e; ++i) {
    double diff = std::fabs(result2Handle.raw(i) - result1Handle.raw(i)) /
                  result1Handle.raw(i);

    // Allow 3% difference.
    EXPECT_NEAR(diff, 0, 0.03);
  }
}

} // namespace glow
