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

#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::dyn_cast;

/// Helper which replaces all uses of \p oldNV with \p newNV, and also
/// optionally maps from \p newNV to \p oldNV in \p loweredMap. This map can be
/// used to determine the NodeOutputNames of NodeValues that were already
/// created.
static void replaceAllUsesOfWith(LoweredInfoMap *loweredMap, NodeValue oldNV,
                                 NodeValue newNV) {
  oldNV.replaceAllUsesOfWith(newNV);

  if (loweredMap == nullptr) {
    return;
  }

  std::string newOutputName = NodeQuantizationInfo::generateNodeOutputName(
      newNV.getNode()->getName(), newNV.getResNo());
  (*loweredMap)[newOutputName].insert(
      NodeNameAndKind(oldNV.getNode()->getName(), oldNV.getResNo(),
                      oldNV.getNode()->getKind()));
}

static void lowerAddGradNode(Function *F, CompilationContext &cctx,
                             const AddGradNode &node) {
  /// The chain rule for addition:
  /// delta(LHS) = dF/dLHS * delta(OUT) = 1 * delta(OUT)
  /// delta(RHS) = dF/dRHS * delta(OUT) = 1 * delta(OUT)

  LOG_SCOPE(F->getLogContext(), "lowerAddGradNode")

  auto outG = node.getGradOfOriginalOutputNamedResult();
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedLHS(),
                       outG);
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedRHS(),
                       outG);
}

static void lowerMulGradNode(Function *F, CompilationContext &cctx,
                             const MulGradNode &node) {
  /// The chain rule for multiplication:
  /// delta(LHS) = dF/dLHS * delta(OUT) = RHS * delta(OUT)
  /// delta(RHS) = dF/dRHS * delta(OUT) = LHS * delta(OUT)

  LOG_SCOPE(F->getLogContext(), "lowerMulGradNode")

  auto outG = node.getGradOfOriginalOutputNamedResult();
  NodeValue LHS = node.getLHS();
  NodeValue RHS = node.getRHS();

  auto *lhsResult = F->createMul("mul.grad.rhs", outG, RHS);
  auto *rhsResult = F->createMul("mul.grad.lhs", outG, LHS);
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedLHS(),
                       lhsResult);
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedRHS(),
                       rhsResult);
}

static void lowerSubGradNode(Function *F, CompilationContext &cctx,
                             const SubGradNode &node) {
  /// The chain rule for subtraction:
  /// delta(LHS) = dF/dLHS * delta(OUT) = 1 * delta(OUT)
  /// delta(RHS) = dF/dRHS * delta(OUT) = -1 * delta(OUT)

  LOG_SCOPE(F->getLogContext(), "lowerSubGradNode")

  auto outG = node.getGradOfOriginalOutputNamedResult();
  auto *zero = F->createSplat("zero", outG.getType(), 0);
  auto *sub = F->createSub("sub.grad", zero, outG);
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedLHS(),
                       outG);
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedRHS(), sub);
}
static void lowerDivGradNode(Function *F, CompilationContext &cctx,
                             const DivGradNode &node) {
  /// The chain rule for division:
  /// delta(LHS) = dF/dLHS * delta(OUT) = (1 / RHS) * delta(OUT)
  /// delta(RHS) = dF/dRHS * delta(OUT) = (-LHS / (RHS ^ 2)) * delta(OUT)

  LOG_SCOPE(F->getLogContext(), "lowerDivGradNode")

  auto outG = node.getGradOfOriginalOutputNamedResult();
  NodeValue LHS = node.getLHS();
  NodeValue RHS = node.getRHS();

  auto *lhsResult = F->createDiv("div.grad.rhs", outG, RHS);

  auto *zero = F->createSplat("zero", outG.getType(), 0);
  auto *subGrad = F->createSub("sub.grad", zero, outG);
  auto *mulLhsGrad = F->createMul("mul.sub.grad.lhs", subGrad, LHS);

  auto *squareRhs = F->createMul("square.rhs", RHS, RHS);
  auto *rhsResult = F->createDiv("div.grad", mulLhsGrad, squareRhs);

  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedLHS(),
                       lhsResult);
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedRHS(),
                       rhsResult);
}

static void lowerRegressionNode(CompilationContext &cctx,
                                const RegressionNode &node) {
  auto input = node.getInput();
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getResult(), input);
}

static void lowerRegressionGradNode(Function *F, CompilationContext &cctx,
                                    const RegressionGradNode &node) {
  LOG_SCOPE(F->getLogContext(), "lowerRegressionGradNode")

  auto outG = node.getInput();

  auto *inputG = F->createSub("rgn.grad", node.getInput(), node.getExpected());
  auto *expG = F->createSplat("exp.grad", node.getExpected().getType(), 0);

  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedInput(),
                       inputG);
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedExpected(),
                       expG);
}

static void lowerFullyConnectedNode(Function *F, CompilationContext &cctx,
                                    const FullyConnectedNode &FC) {
  LOG_SCOPE(F->getLogContext(), "lowerFullyConnectedNode")

  auto W = FC.getWeights();
  TypeRef OT = FC.getResult().getType();
  auto *mul = F->createMatMul("fc.dot", OT, FC.getInput(), W);
  auto *add = F->createBatchedAdd("fc.add.bias", OT, mul, FC.getBias());
  replaceAllUsesOfWith(cctx.loweredInfoMap, FC.getResult(), add);

  if (FC.hasPredicate()) {
    add->setPredicate(FC.getPredicate());
    mul->setPredicate(FC.getPredicate());
  }
}

static void lowerFullyConnectedGradNode(Function *F, CompilationContext &cctx,
                                        const FullyConnectedGradNode &FCG) {
  // Follow the lowering from here:
  // https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/layers.py#L53

  LOG_SCOPE(F->getLogContext(), "lowerFullyConnectedGradNode")

  auto dout = FCG.getGradOfOriginalOutputNamedResult();

  // dx = dout * w.T
  auto *wT = F->createTranspose("fcg.wT", FCG.getWeights(), {1, 0});
  auto *dx2 = F->createMatMul("fcg.dot", dout, wT);
  auto *dx = F->createReshape("fcg.inG", dx2, FCG.getInput().getType()->dims());
  replaceAllUsesOfWith(cctx.loweredInfoMap, FCG.getGradOfInputNamedInput(), dx);

  // dw = xT * dout.
  Node *x2 = F->createFlatten("fcg.x", FCG.getInput(), 1);
  auto *x2T = F->createTranspose("fcg.xT", x2, {1, 0});
  auto *dw = F->createMatMul("fcg.dot", x2T, dout);
  replaceAllUsesOfWith(cctx.loweredInfoMap, FCG.getGradOfInputNamedWeights(),
                       dw);

  // db = reduce(dout).
  auto *db = F->createBatchedReduceAdd("fc.bias.reduce", dout, /* axis */ 0);
  replaceAllUsesOfWith(cctx.loweredInfoMap, FCG.getGradOfInputNamedBias(), db);
}

static void lowerReluGradNode(Function *F, CompilationContext &cctx,
                              const ReluGradNode &RG) {
  // ReluGrad: if the input value is greater than zero then let the gradient
  // pass.

  LOG_SCOPE(F->getLogContext(), "lowerReluGradNode")

  auto *zero = F->createSplat("zero", RG.getInput().getType(), 0.0);
  auto *cond =
      F->createCmpLTE("relugrad", RG.getOriginalOutputForResult(), zero);
  auto *res = F->createSelect("relugrad", cond, zero,
                              RG.getGradOfOriginalOutputNamedResult());
  replaceAllUsesOfWith(cctx.loweredInfoMap, RG.getGradOfInputNamedInput(), res);
}

static void lowerTanhGradNode(Function *F, CompilationContext &cctx,
                              const TanhGradNode &THG) {
  // Tanh grad is calculated as:
  // inG = (1 - outW * outW) * outG

  LOG_SCOPE(F->getLogContext(), "lowerTanhGradNode")

  // (W * W)
  auto outW = THG.getOriginalOutputForResult();
  auto *sq = F->createMul("tanh.in2", outW, outW);

  auto *one = F->createSplat("tanh.one", THG.getInput().getType(), 1.0);
  // (1 - W * W)
  auto *oneSubsq = F->createSub("tanh.one.sq", one, sq);

  auto *grad = F->createMul("tanh.one.sq", oneSubsq,
                            THG.getGradOfOriginalOutputNamedResult());
  replaceAllUsesOfWith(cctx.loweredInfoMap, THG.getGradOfInputNamedInput(),
                       grad);
}

static void lowerSigmoidGradNode(Function *F, CompilationContext &cctx,
                                 const SigmoidGradNode &THG) {
  // Sigmoid grad is calculated as:
  // inG = outW * (1 - outW) * outG;

  LOG_SCOPE(F->getLogContext(), "lowerSigmoidGradNode")

  auto outW = THG.getOriginalOutputForResult();
  auto *one = F->createSplat("one", THG.getInput().getType(), 1.0);

  // (1 - W)
  auto *onew = F->createSub("sig.1w", one, outW);

  // (1 - W) * W
  auto *expr1 = F->createMul("sig.1ww", onew, outW);

  auto *grad = F->createMul("sigg.one.sq", expr1,
                            THG.getGradOfOriginalOutputNamedResult());
  replaceAllUsesOfWith(cctx.loweredInfoMap, THG.getGradOfInputNamedInput(),
                       grad);
}

static void lowerReluNode(Function *F, CompilationContext &cctx,
                          const ReluNode &R) {
  LOG_SCOPE(F->getLogContext(), "lowerReluNode")

  // Relu is a max between zero and the input value.
  SplatNode *zero = F->createSplat("zero", R.getResult().getType(), 0.0);
  auto *relu =
      F->createMax("relu", R.getResult().getType(), zero, R.getInput());
  replaceAllUsesOfWith(cctx.loweredInfoMap, R.getResult(), relu);
}

static void lowerPReluNode(Function *F, CompilationContext &cctx,
                           const PReluNode &R) {
  // PRelu is :
  // slope * x    if x < 0
  // x            if x >= 0
  // where slope is an input from a different node.

  LOG_SCOPE(F->getLogContext(), "lowerPReluNode")

  auto *zeroSplat = F->createSplat("zeroSplat", R.getResult().getType(), 0.0);
  auto *cmplgt = F->createCmpLTE("cmplgt", zeroSplat, R.getInput());
  auto *mul = F->createMul("mul", R.getSlope(), R.getInput());
  auto *prelu = F->createSelect("prelu", cmplgt, R.getInput(), mul);

  replaceAllUsesOfWith(cctx.loweredInfoMap, R.getResult(), prelu);
}

static void lowerPadNode(Function *F, CompilationContext &cctx,
                         const PadNode &P) {
  LOG_SCOPE(F->getLogContext(), "lowerPadNode")

  auto *outputType = P.getResult().getType();
  auto dims = outputType->dims();
  auto numDims = dims.size();
  auto pads = P.getPads();

  // Sanity checks: Pad needs to have the 'constant' mode and can't be lowered
  // when pads are negative.
  assert((P.getMode() == PaddingMode::CONSTANT) &&
         "only 'constant' padding is supported at lowering.");
  for (auto p : pads) {
    (void)p; // Avoids a warning in release mode.
    assert((p >= 0) && "negative pads not supported at lowering.");
  }

  SplatNode *constant = F->createSplat("pad.const", outputType, P.getValue());

  std::vector<size_t> orig(numDims);
  for (size_t i = 0; i < numDims; i++) {
    orig[i] = size_t(pads[i]);
  }

  auto *insert =
      F->createInsertTensor(P.getName(), constant, P.getInput(), orig);
  replaceAllUsesOfWith(cctx.loweredInfoMap, P.getResult(), insert);
}

static void lowerSGDNode(Function *F, CompilationContext &cctx,
                         const SGDNode &SGD) {
  LOG_SCOPE(F->getLogContext(), "lowerSGDNode")

  NodeValue W = SGD.getWeight();
  NodeValue G = SGD.getGradient();

  /// Described in the paper: Alex Krizhevsky [2014]
  // "One weird trick for parallelizing convolutional neural networks"

  float momentum = SGD.getMomentum();

  assert(W.dims() == G.dims() && "Invalid weight/gradient sizes for SGDNode");

  float L1Decay = SGD.getL1Decay();
  float L2Decay = SGD.getL2Decay();
  float learningRate = SGD.getLearningRate();
  float batchSize = SGD.getBatchSize();

  // All computations here are within the same type.
  auto type = G.getType();

  NodeValue gij = G;
  if (L1Decay != 0.0f) {
    auto *L1DecaySplat = F->createSplat("L1DecaySplat", type, L1Decay);
    auto *zeroSplat = F->createSplat("zeroSplat", type, 0);
    auto *oneSplat = F->createSplat("oneSplat", type, 1);
    auto *minusOneSplat = F->createSplat("minusOneSplat", type, -1);

    auto *Wcmp = F->createCmpLTE("Wcmp", zeroSplat, W);
    auto *Wdir = F->createSelect("Wdir", Wcmp, oneSplat, minusOneSplat);
    auto *L1Grad = F->createMul("L1Grad", L1DecaySplat, Wdir);

    gij = F->createAdd("gij_with_l1", gij, L1Grad);
  }
  if (L2Decay != 0.0f) {
    auto *L2DecaySplat = F->createSplat("L2DecaySplat", type, L2Decay);

    auto *L2Grad = F->createMul("L2Grad", L2DecaySplat, W);

    gij = F->createAdd("gij_with_l2", gij, L2Grad);
  }
  if (batchSize > 1) {
    auto *batchSizeSplat = F->createSplat("batchSizeSplat", type, batchSize);
    gij = F->createDiv("gij_div_batchSz", gij, batchSizeSplat);
  }

  auto *negLearningRateSplat =
      F->createSplat("learningRateSplat", type, -learningRate);
  Node *dx = F->createMul("dx", negLearningRateSplat, gij);

  // Use the momentum to improve the gradient descent:
  // http://ufldl.stanford.edu/tutorial/supervised/
  // OptimizationStochasticGradientDescent/
  if (momentum > 0.0) {
    Placeholder *Gsum =
        F->getParent()->createPlaceholder(W.getType(), "gsum", false);
    Gsum->setAllocZero();

    auto *momentumSplat = F->createSplat("learningRateSplat", type, momentum);
    auto *GsumMult = F->createMul("GsumMult", momentumSplat, Gsum);

    dx = F->createAdd("dx_with_momentum", GsumMult, dx);
    F->createSave("save.gsum", dx, Gsum);
  }

  auto *newW = F->createAdd("newW", W, dx);
  replaceAllUsesOfWith(cctx.loweredInfoMap, SGD.getUpdatedWeight(), newW);
}

static void lowerBatchNormalizationNode(Function *F, CompilationContext &cctx,
                                        const BatchNormalizationNode &BN) {
  LOG_SCOPE(F->getLogContext(), "lowerBatchNormalizationNode")

  auto in = BN.getInput();
  auto out = BN.getResult();

  auto beta = BN.getBias();
  auto gamma = BN.getScale();
  auto var = BN.getVar();
  auto mean = BN.getMean();

  // http://cthorey.github.io/backpropagation/
  //
  // mu = 1/N*np.sum(h,axis =0)
  // sigma2 = 1/N*np.sum((h-mu)**2)
  // hath = (h-mu)*(sigma2+epsilon)**(-1./2.)
  // y = gamma*hath+beta

  // In inference mode just apply the transformation:
  // y[i] = (x - mu) * gamma / stdvar + beta;

  auto channelIdx = BN.getChannelIdx();
  auto epsilon = BN.getEpsilon();

  auto *epsilonSplat = F->createSplat("epsSplat", var.getType(), epsilon);
  Node *coef = F->createAdd("var_plus_eps", var, epsilonSplat);
  coef = F->createPow("sqrt_var_plus_eps", coef, 0.5);
  coef = F->createDiv("inverse_sqrt_var_plus_eps", gamma, coef);

  // Apply: out := (in - mean) * coef + beta
  // in and out are of the same size, while others must be broadcasted.
  auto *meanB =
      F->createBroadcast("muBroadcasted", mean, in.dims(), channelIdx);
  auto *coefB =
      F->createBroadcast("coefBroadcasted", coef, in.dims(), channelIdx);
  auto *betaB =
      F->createBroadcast("betaBroadcasted", beta, in.dims(), channelIdx);

  Node *newResult = F->createSub("in_minus_mean", in, meanB);
  newResult = F->createMul("mul_coef", newResult, coefB);
  newResult = F->createAdd("result", newResult, betaB);

  replaceAllUsesOfWith(cctx.loweredInfoMap, BN.getResult(), newResult);
}

static void lowerMeanVarNormalizationNode(Function *F, CompilationContext &cctx,
                                          const MeanVarNormalizationNode &MVN) {
  LOG_SCOPE(F->getLogContext(), "lowerMeanVarNormalizationNode")

  auto in = MVN.getInput();

  auto inMean = MVN.getMean();
  auto inVar = MVN.getVar();

  auto channelIdx = MVN.getChannelIdx();
  auto momentum = MVN.getMomentum();

  // The number of different channels.
  const size_t numChannels = in.dims()[channelIdx];
  // The number of elements that each channel holds.
  const size_t samplesPerChannel = in.getType()->size() / numChannels;

  // Input tensor can have arbitrary shape:
  // {d_1, d_2, ..., d[channelIdx], ... d_n}
  // We need to compute Mean and Variance for each channel, 2 tensors of shape:
  // {d[channelIdx]} (which is {numChannels})
  // That is, some sort of aggregation needs to happen for each channel.
  NodeValue inPrep = in;
  if (channelIdx + 1 != in.dims().size()) {
    // If channelIdx is not the last, transform input tensor to shape:
    // {d_1, d_2, ... d_n, numChannels}
    std::vector<unsigned_t> perm(in.dims().size());
    for (size_t i = 0; i < perm.size(); i++) {
      perm[i] = i;
    }
    std::swap(perm[channelIdx], perm[perm.size() - 1]);
    inPrep = F->createTranspose("in.transpose", in, perm);
  }
  // Reshape input tensor to form:
  // {samplesPerChannel, numChannels}
  ReshapeNode *inFlat =
      F->createReshape("in.flat", inPrep, {samplesPerChannel, numChannels});

  // Calculate Mean:

  // sum(in[i])
  // reduce the tensor by the first dimension, to get {numChannels}
  auto *batchedAdd = F->createBatchedReduceAdd("in.sum", inFlat, /* axis */ 0);
  // Mean = sum(in[i]) / N
  auto samplesPerChannelSplat =
      F->createSplat("samplesPerChannelSplat",
                     batchedAdd->getResult().getType(), samplesPerChannel);
  DivNode *localMean =
      F->createDiv("localMean", batchedAdd, samplesPerChannelSplat);

  // Calculate Variance:
  // sum((x - mu) ^ 2)
  auto *localMeanB = F->createBroadcast("new_mean_broadcasted", localMean,
                                        inFlat->getResult().dims(), 1);

  Node *localVar = F->createSub("x_mu", inFlat, localMeanB);
  localVar = F->createPow("x_mu2", localVar, 2);
  localVar = F->createBatchedReduceAdd("x_mu2.sum", localVar, /* axis */ 0);
  // Var = sum((x - mu) ^ 2) / N
  localVar = F->createDiv("localVar", localVar, samplesPerChannelSplat);

  // Update the global variance and mean:
  auto *momentumSplat = F->createSplat(
      "momentumSplat", localMean->getResult().getType(), momentum);
  auto *oneMinusMomentumSplat = F->createSplat(
      "oneMinusMomentumSplat", localMean->getResult().getType(), 1 - momentum);

  // newMean := P * localMean + (1 - P) * oldMean
  auto *newMean = F->createAdd(
      "newMean",
      F->createMul("momentum_by_localMean", momentumSplat, localMean),
      F->createMul("1_momentum_by_oldMean", oneMinusMomentumSplat, inMean));
  // newVar := P * localVar + (1 - P) * oldVar
  auto *newVar = F->createAdd(
      "newVar", F->createMul("momentum_by_localVar", momentumSplat, localVar),
      F->createMul("1_momentum_by_oldVar", oneMinusMomentumSplat, inVar));

  replaceAllUsesOfWith(cctx.loweredInfoMap, MVN.getNewMean(), newMean);
  replaceAllUsesOfWith(cctx.loweredInfoMap, MVN.getNewVar(), newVar);
}

static void lowerBatchNormalizationGradNode(Function *F,
                                            CompilationContext &cctx,
                                            BatchNormalizationGradNode &BNG) {
  LOG_SCOPE(F->getLogContext(), "lowerBatchNormalizationGradNode")

  auto inW = BNG.getInput();
  auto outG = BNG.getGradOfOriginalOutputNamedResult();

  auto gamma = BNG.getScale();

  auto var = BNG.getVar();
  auto mean = BNG.getMean();

  auto channelIdx = BNG.getChannelIdx();
  auto epsilon = BNG.getEpsilon();

  // The number of different channels.
  const size_t numChannels = inW.dims()[channelIdx];
  // The number of elements that each channel holds.
  const size_t samplesPerChannel = inW.getType()->size() / numChannels;

  // Calculate: sum(dy * (h - mu))
  auto *meanB =
      F->createBroadcast("mean_broadcasted", mean, inW.dims(), channelIdx);
  auto *hmu = F->createSub("x_minus_mean", inW, meanB);
  NodeValue sumDyhmu = F->createMul("dy_mul_h_minus_mu", outG, hmu);

  // Calculate: sum(dy)
  NodeValue sumDy = outG;

  // TODO: consider adding this functionality to the main operator set.
  if (channelIdx + 1 != inW.dims().size()) {
    std::vector<unsigned_t> perm(inW.dims().size());
    for (size_t i = 0; i < perm.size(); i++) {
      perm[i] = i;
    }
    std::swap(perm[channelIdx], perm[perm.size() - 1]);

    sumDyhmu = F->createTranspose("sumDyhmu.transpose", sumDyhmu, perm);
    sumDy = F->createTranspose("sumDy.transpose", sumDy, perm);
  }
  sumDyhmu = F->createReshape("sumDyhmu.flat", sumDyhmu,
                              {samplesPerChannel, numChannels});
  sumDy =
      F->createReshape("sumDy.flat", sumDy, {samplesPerChannel, numChannels});
  sumDyhmu =
      F->createBatchedReduceAdd("sumDyhmu.reduced", sumDyhmu, /* axis */ 0);
  sumDy = F->createBatchedReduceAdd("sumDy.reduced", sumDy, /* axis */ 0);

  // http://cthorey.github.io./backpropagation/
  //
  // mu = 1./N*np.sum(h)
  // var = 1./N*np.sum((h-mu)**2)
  // dbeta = np.sum(dy)
  // dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy)
  // dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) *
  //     (N * dy - np.sum(dy) - (h - mu) * 1/(var + eps) *
  //     np.sum(dy * (h - mu)))
  //

  auto *epsilonSplat = F->createSplat("epsSplat", var.getType(), epsilon);
  auto *oneSplat = F->createSplat("oneSplat", var.getType(), 1.0);
  auto *invNSplat =
      F->createSplat("invNSplat", var.getType(), 1.0 / samplesPerChannel);
  Node *invVar = F->createAdd("var_plus_eps", var, epsilonSplat);
  invVar = F->createDiv("inverse_var_plus_eps", oneSplat, invVar);
  Node *invVarSqrt = F->createPow("invVarSqrt", invVar, 0.5);

  Node *coef1 =
      F->createMul("invN_gamma_invVarSqrt",
                   F->createMul("invN_gamma", invNSplat, gamma), invVarSqrt);
  Node *coef2 = F->createMul("invVar_sumDyhmu", invVar, sumDyhmu);

  // Apply:
  // inG := Bcast(coef1) * (NSplat * outG - Bcast(sumDy) - hmu * Bcast(coef2))

  coef1 =
      F->createBroadcast("coef1_broadcasted", coef1, inW.dims(), channelIdx);
  coef2 =
      F->createBroadcast("coef2_broadcasted", coef2, inW.dims(), channelIdx);
  auto *sumDyB =
      F->createBroadcast("sumDy_broadcasted", sumDy, inW.dims(), channelIdx);
  auto *NSplat = F->createSplat("oneSplat", inW.getType(), samplesPerChannel);
  Node *inBrackets = F->createMul("NSplat_outG", NSplat, outG);
  inBrackets = F->createSub("inBrackets",
                            F->createSub("inBrackets_2ops", inBrackets, sumDyB),
                            F->createMul("hmu_coef2", hmu, coef2));

  auto *inG = F->createMul("inG", coef1, inBrackets);
  replaceAllUsesOfWith(cctx.loweredInfoMap, BNG.getGradOfInputNamedInput(),
                       inG);

  replaceAllUsesOfWith(cctx.loweredInfoMap, BNG.getGradOfInputNamedBias(),
                       sumDy);

  auto *gammaG = F->createMul("gammaG", sumDyhmu, invVarSqrt);
  replaceAllUsesOfWith(cctx.loweredInfoMap, BNG.getGradOfInputNamedScale(),
                       gammaG);

  auto *zeroSplat = F->createSplat("zeroSplat", var.getType(), 0);
  replaceAllUsesOfWith(cctx.loweredInfoMap, BNG.getGradOfInputNamedMean(),
                       zeroSplat);
  replaceAllUsesOfWith(cctx.loweredInfoMap, BNG.getGradOfInputNamedVar(),
                       zeroSplat);
}

static void lowerGroupConvolutionNode(Function *F, CompilationContext &cctx,
                                      const ConvolutionNode &BNG) {
  // When Group parameter is more than 1, ConvolutionNode can be represented as
  // a Concatenation of smaller dimension Convolutions. Input channels will be
  // divided into equal groups of consecutive channels. These will be separately
  // convolved each with its own filter (and bias), and then concatenated.
  // This will result in 4 * Group + 1 nodes.

  LOG_SCOPE(F->getLogContext(), "lowerGroupConvolutionNode")

  llvm::ArrayRef<unsigned_t> kernels = BNG.getKernels();
  llvm::ArrayRef<unsigned_t> pads = BNG.getPads();
  llvm::ArrayRef<unsigned_t> strides = BNG.getStrides();
  unsigned_t group = BNG.getGroup();
  auto in = BNG.getInput();
  auto filter = BNG.getFilter();
  auto bias = BNG.getBias();

  ShapeNHWC idim = ShapeNHWC(in.dims());
  ShapeHW kdim(kernels);
  unsigned inCperG = idim.c / group;
  unsigned outCperG = filter.dims()[0] / group;

  auto outDims = BNG.getResult().dims().vec();
  outDims[3] = outCperG;
  auto outTy = F->getParent()->uniqueTypeWithNewShape(BNG.getResult().getType(),
                                                      outDims);

  std::vector<NodeValue> convs;
  for (unsigned_t groupId = 0; groupId < group; groupId++) {
    auto *in_slice =
        F->createSlice(BNG.getName(), in, {0, 0, 0, groupId * inCperG},
                       {idim.n, idim.h, idim.w, (groupId + 1) * inCperG});
    auto *filter_slice = F->createSlice(
        BNG.getName(), filter, {groupId * outCperG, 0, 0, 0},
        {(groupId + 1) * outCperG, kdim.height, kdim.width, inCperG});
    auto *bias_slice = F->createSlice(BNG.getName(), bias, {groupId * outCperG},
                                      {(groupId + 1) * outCperG});
    convs.push_back(F->createConv(BNG.getName(), in_slice, filter_slice,
                                  bias_slice, outTy, kernels, strides, pads, 1,
                                  BNG.getDilation()));
  }
  auto *result = F->createConcat(BNG.getName(), convs, 3);
  replaceAllUsesOfWith(cctx.loweredInfoMap, BNG.getResult(), result);
}

static void lowerChannelwiseQuantizedConvolutionNode(
    Function *F, CompilationContext &cctx,
    const ChannelwiseQuantizedConvolutionNode &CQC) {
  // ChannelwiseQuantizedConvolutionNode can be represented as a Concatenation
  // of smaller dimension quantized Convolutions each with their own qparams.
  // Input channels will be divided into equal groups of consecutive channels.
  // These will be separately convolved each with its own filter and bias.
  // This will result in 4 * Group + 1 nodes.

  // Only lowering of groupwise, not channelwise is supported so far.
  if (!CQC.getGroupwise()) {
    return;
  }

  llvm::ArrayRef<unsigned_t> kernels = CQC.getKernels();
  llvm::ArrayRef<unsigned_t> pads = CQC.getPads();
  llvm::ArrayRef<unsigned_t> strides = CQC.getStrides();
  unsigned_t group = CQC.getGroup();
  auto in = CQC.getInput();

  Constant *filter = llvm::cast<Constant>(CQC.getFilter());
  Constant *bias = llvm::cast<Constant>(CQC.getBias());
  Constant *scales = llvm::cast<Constant>(CQC.getScales());
  Constant *offsets = llvm::cast<Constant>(CQC.getOffsets());

  ShapeNHWC idim = ShapeNHWC(in.dims());
  ShapeHW kdim(kernels);
  unsigned inCperG = idim.c / group;
  unsigned outCperG = filter->dims()[0] / group;

  auto convOutDims = CQC.getResult().dims().vec();
  convOutDims[3] = outCperG;

  auto filterDims = filter->dims().vec();
  filterDims[0] = outCperG;
  filterDims[3] = inCperG;

  // Final output type of each convolution after rescaling
  auto finalConvOutTy = F->getParent()->uniqueTypeWithNewShape(
      CQC.getResult().getType(), convOutDims);

  auto scalesHandle = scales->getHandle<float>();
  auto offsetsHandle = offsets->getHandle<int32_t>();

  Module *M = F->getParent();

  std::vector<NodeValue> branches;
  for (unsigned_t groupId = 0; groupId < group; groupId++) {
    float filterScale = scalesHandle.raw(groupId);
    int32_t filterOffset = offsetsHandle.raw(groupId);

    SliceNode *inSlice =
        F->createSlice(CQC.getName(), in, {0, 0, 0, groupId * inCperG},
                       {idim.n, idim.h, idim.w, (groupId + 1) * inCperG});

    // Create quantized filter sliced Constant.
    Tensor slicedFilterTensor = filter->getPayload().getOwnedSlice(
        filterDims, {outCperG * groupId, 0, 0, 0});
    auto quantizedFilterType = slicedFilterTensor.getType();
    quantizedFilterType.scale_ = filterScale;
    quantizedFilterType.offset_ = filterOffset;
    slicedFilterTensor.setType(&quantizedFilterType);
    Constant *slicedFilterConst =
        M->createConstant(strFormat("%s_filter", CQC.getName().data()),
                          std::move(slicedFilterTensor));

    // Create bias sliced Constant. Bias's scale should always be inputScale *
    // filterScale.
    TensorQuantizationParams tqp;
    tqp.offset = 0;
    tqp.scale = inSlice->getInput().getType()->getScale() * filterScale;
    Tensor slicedBiasTensor =
        bias->getPayload().getOwnedSlice({outCperG}, {outCperG * groupId});
    Tensor quantizedSlicedBiasTensor =
        quantization::quantizeTensor(slicedBiasTensor, tqp, ElemKind::Int32QTy);
    Constant *slicedBias =
        M->createConstant(CQC.getName(), std::move(quantizedSlicedBiasTensor));

    ConvolutionNode *convNode = F->createConv(
        strFormat("%s_bias", CQC.getName().data()), inSlice, slicedFilterConst,
        slicedBias, finalConvOutTy, kernels, strides, pads,
        /* group */ 1);
    branches.push_back(convNode);
  }
  auto *result = F->createConcat(CQC.getName(), branches, /* dimension */ 3);
  replaceAllUsesOfWith(cctx.loweredInfoMap, CQC.getResult(), result);
}

static void lowerSigmoidCrossEntropyWithLogitsNode(
    Function *F, CompilationContext &cctx,
    const SigmoidCrossEntropyWithLogitsNode &SCEL) {
  // Following Caffe2 implementation closely to lower this Node.
  // https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc

  LOG_SCOPE(F->getLogContext(), "lowerSigmoidCrossEntropyWithLogitsNode")

  auto lgt = SCEL.getLogits();
  auto tgt = SCEL.getTargets();

  // Element-wise transformation:
  // max(lgt, 0) - lgt * tgt + log(1 + exp(-abs(x)))

  auto *zeroSplat = F->createSplat("zeroSplat", lgt.getType(), 0.0);
  auto *oneSplat = F->createSplat("oneSplat", lgt.getType(), 1.0);
  auto *expSplat = F->createSplat("oneSplat", lgt.getType(), exp(1.0));

  auto *cmp0lgt = F->createCmpLTE("cmp.0.lgt", zeroSplat, lgt);

  // (tgt - (lgt >= 0))
  auto *coeff = F->createSelect("select", cmp0lgt,
                                F->createSub("tgt.m1", tgt, oneSplat), tgt);

  // exp(lgt >= 0 ? -lgt : lgt)
  auto *expArg = F->createSelect("exp.arg", cmp0lgt,
                                 F->createSub("neg.lgt", zeroSplat, lgt), lgt);

  // (1 + exp(expArg))
  auto *logArg =
      F->createAdd("log.arg", oneSplat, F->createPow("exp", expSplat, expArg));

  // log(logArg) - lgt * coeff
  auto *sigmoidXent =
      F->createSub("sigmoid.xent.pointwise", F->createLog("log", logArg),
                   F->createMul("lhs", lgt, coeff));

  auto *reducedSigmoidXent = F->createBatchedReduceMean(
      "sigmoid.xent.lowered", sigmoidXent, lgt.dims().size() - 1);

  replaceAllUsesOfWith(cctx.loweredInfoMap, SCEL.getResult(),
                       reducedSigmoidXent);
}

/// Lower Tile nodes to InsertTensor nodes with correct axis and count.
static void lowerTileNode(Function *F, CompilationContext &cctx,
                          const TileNode &TN) {
  LOG_SCOPE(F->getLogContext(), "lowerTileNode")

  auto input = TN.getInput();

  // Use a zero splat as the Big node input for InsertTensor.
  auto *zero = F->createSplat("zero", TN.getResult().getType(), 0);
  // Insert at the beginning of the splat.
  auto start = std::vector<size_t>(input.dims().size(), 0);

  auto *IN = F->createInsertTensor(TN.getName(), zero, input, start,
                                   TN.getCount(), TN.getAxis());
  replaceAllUsesOfWith(cctx.loweredInfoMap, TN.getResult(), IN);
}

static void lowerChannelShuffleNode(Function *F, CompilationContext &cctx,
                                    const ChannelShuffleNode &CSN) {
  LOG_SCOPE(F->getLogContext(), "lowerChannelShuffleNode")

  auto input = CSN.getInput();
  auto group = CSN.getGroup();
  auto kernel = CSN.getKernel();

  auto inDims = input.dims();
  assert(kernel < inDims.size());

  ShapeVector dims(inDims.begin(), inDims.end());
  auto D = dims[kernel];
  assert(D % group == 0);

  dims.erase(dims.begin() + kernel);
  // Reshape {D1, ... D_k, ... D_n} -> {D1, ... group, D_k / group, ... D_n}
  dims.insert(dims.begin() + kernel, D / group);
  dims.insert(dims.begin() + kernel, group);
  auto *R1 = F->createReshape(CSN.getName().str() + ".reshape1", input, dims);

  std::vector<unsigned_t> transpose(dims.size());
  for (size_t i = 0; i < transpose.size(); i++) {
    transpose[i] = i;
  }
  std::swap(transpose[kernel], transpose[kernel + 1]);
  auto *T =
      F->createTranspose(CSN.getName().str() + ".transpose", R1, transpose);

  auto *R2 = F->createReshape(CSN.getName().str() + ".reshape2", T, inDims);
  replaceAllUsesOfWith(cctx.loweredInfoMap, CSN.getResult(), R2);
}

static void lowerBatchReduceMeanNode(Function *F, CompilationContext &cctx,
                                     const BatchedReduceMeanNode &BRM) {
  LOG_SCOPE(F->getLogContext(), "lowerBatchReduceMeanNode")

  auto input = BRM.getBatch();

  assert((BRM.getAxes().size() == 1) && "Only supporting single reduction.");

  auto axis = BRM.getAxes()[0];

  assert(axis < input.dims().size() &&
         "Axis to remove must fit inside dimensions of the provided dims.");

  ShapeVector redDims(input.dims().begin(), input.dims().end());
  redDims.erase(redDims.begin() + axis);

  auto outTy = F->getParent()->uniqueTypeWithNewShape(BRM.getResult().getType(),
                                                      redDims);

  const size_t outNumElements = input.getType()->size() / input.dims()[axis];
  (void)outNumElements;
  assert(outTy->size() == outNumElements &&
         "Incorrect number of elements in the output type.");

  // Create a batched add to sum up the values in the provided axis.
  auto outTyBRA =
      F->getParent()->uniqueTypeWithNewShape(input.getType(), redDims);

  auto *BRA = F->createBatchedReduceAdd(BRM.getName().str() + ".reduceAdd",
                                        outTyBRA, input, axis);

  // Create a splat of the same output type as the BRA, with value of the size
  // of the original dimensions of the axis, to divide the BRA by.
  auto *SN = F->createSplat(llvm::StringRef(BRM.getName().str() + ".splat"),
                            outTyBRA, input.dims()[axis]);

  // Element-wise divide to produce the reduced mean with outTy provided.
  auto *DN = F->createDiv(BRM.getName().str() + ".div", outTy, BRA, SN);

  replaceAllUsesOfWith(cctx.loweredInfoMap, BRM.getResult(), DN);
}

/// Implement ReplaceNaN via a Select node with the input of \p RN as one of the
/// inputs, a Splat node created using value from \p RN as the other input, and
/// an IsNaN node as the comparator input.
static void lowerReplaceNaNNode(Function *F, CompilationContext &cctx,
                                const ReplaceNaNNode &RN) {
  LOG_SCOPE(F->getLogContext(), "lowerReplaceNaNNode")

  // Create IsNaN node.
  auto *INN = F->createIsNaN(RN.getName().str() + ".isNaN", RN.getInput());

  // Create Splat node.
  auto *S = F->createSplat(RN.getName().str() + ".splat",
                           RN.getInput().getType(), RN.getValue());

  // Create Select node to pick between original and replacement values.
  auto *SN =
      F->createSelect(RN.getName().str() + ".select", INN, S, RN.getInput());

  replaceAllUsesOfWith(cctx.loweredInfoMap, RN.getResult(), SN);
}

/// Implement BatchMatMul \p BMMN in \p F via a series of Slices, MatMuls, and a
/// final Concat.
static void lowerBatchMatMulNode(Function *F, CompilationContext &cctx,
                                 const BatchMatMulNode &BMMN) {
  LOG_SCOPE(F->getLogContext(), "lowerBatchMatMulNode")

  auto name = BMMN.getName();
  NodeValue lhs = BMMN.getLHS();
  NodeValue rhs = BMMN.getRHS();
  NodeValue dest = BMMN.getResult();

  // LHS = {numBatches, N, M}
  // RHS = {numBatches, M, P}
  const size_t numBatches = lhs.dims()[0];
  const size_t N = lhs.dims()[1];
  const size_t M = lhs.dims()[2];
  const size_t P = rhs.dims()[2];

  // Lower to Slices, MatMuls, and a final Concat. Multiply i-th LHS matrix
  // {N, M} by i-th RHS matrix {M, P} to get final matrix {numBatches, N, P}.
  std::vector<NodeValue> MMS(numBatches);
  for (size_t i = 0; i < numBatches; i++) {
    SliceNode *sliceA =
        F->createSlice(name.str() + ".sliceA." + std::to_string(i), lhs,
                       {i, 0, 0}, {i + 1, N, M});
    SliceNode *sliceB =
        F->createSlice(name.str() + ".sliceB." + std::to_string(i), rhs,
                       {i, 0, 0}, {i + 1, M, P});
    ReshapeNode *reshapeA =
        F->createReshape(sliceA->getName().str() + ".reshape", sliceA, {N, M});
    ReshapeNode *reshapeB =
        F->createReshape(sliceB->getName().str() + ".reshape", sliceB, {M, P});
    MMS[i] = F->createMatMul(name.str() + ".MatMul." + std::to_string(i),
                             reshapeA, reshapeB);
  }
  // Concat all the resulting MatMuls back together.
  ConcatNode *CN = F->createConcat(name.str() + ".concat", MMS, /* axis */ 0);

  // Reshape the result back to the expected batch output shape, with the
  // first dimension the number of batches.
  ReshapeNode *RN =
      F->createReshape(name.str() + ".reshapeResult", CN, {numBatches, N, P});

  replaceAllUsesOfWith(cctx.loweredInfoMap, BMMN.getResult(), RN);
}

static void lowerSparseLengthsSumNode(Function *F, CompilationContext &cctx,
                                      const SparseLengthsSumNode &SLSN) {
  LOG_SCOPE(F->getLogContext(), "lowerSparseLengthsSumNode")

  auto ty = F->getParent()->uniqueTypeWithNewShape(
      SLSN.getData().getType(), {SLSN.getIndices().dims()[0]});
  auto *ones = F->createSplat(SLSN.getName().str() + ".ones", ty, 1.0);
  auto *SLWSN = F->createSparseLengthsWeightedSum(
      SLSN.getName().str(), SLSN.getData(), ones, SLSN.getIndices(),
      SLSN.getLengths());

  replaceAllUsesOfWith(cctx.loweredInfoMap, SLSN.getResult(), SLWSN);
}

static void lowerFusedRowwiseQuantizedSparseLengthsSumNode(
    Function *F, CompilationContext &cctx,
    const FusedRowwiseQuantizedSparseLengthsSumNode &FRQSLSN) {
  auto ty = F->getParent()->uniqueType(ElemKind::FloatTy,
                                       {FRQSLSN.getIndices().dims()[0]});
  auto *ones = F->createSplat(FRQSLSN.getName().str() + ".ones", ty, 1.0);
  auto *FRQSLWSN = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      FRQSLSN.getName().str(), FRQSLSN.getData(), ones, FRQSLSN.getIndices(),
      FRQSLSN.getLengths());

  replaceAllUsesOfWith(cctx.loweredInfoMap, FRQSLSN.getResult(), FRQSLWSN);
}

/// Lowers \p node given Function \p. \p cctx contains a mapping of loweredMap
/// that will log the lowering info of what was replaced by what via output
/// names.
static void lowerNode(Function *F, Node *node, CompilationContext &cctx) {
  if (auto *RN = dyn_cast<RegressionNode>(node)) {
    lowerRegressionNode(cctx, *RN);
  } else if (auto *RGN = dyn_cast<RegressionGradNode>(node)) {
    lowerRegressionGradNode(F, cctx, *RGN);
  } else if (auto *EMG = dyn_cast<AddGradNode>(node)) {
    lowerAddGradNode(F, cctx, *EMG);
  } else if (auto *EMG = dyn_cast<MulGradNode>(node)) {
    lowerMulGradNode(F, cctx, *EMG);
  } else if (auto *EMG = dyn_cast<SubGradNode>(node)) {
    lowerSubGradNode(F, cctx, *EMG);
  } else if (auto *EMG = dyn_cast<DivGradNode>(node)) {
    lowerDivGradNode(F, cctx, *EMG);
  } else if (auto *FC = dyn_cast<FullyConnectedNode>(node)) {
    lowerFullyConnectedNode(F, cctx, *FC);
  } else if (auto *FCG = dyn_cast<FullyConnectedGradNode>(node)) {
    lowerFullyConnectedGradNode(F, cctx, *FCG);
  } else if (auto *RG = dyn_cast<ReluGradNode>(node)) {
    lowerReluGradNode(F, cctx, *RG);
  } else if (auto *R = dyn_cast<ReluNode>(node)) {
    lowerReluNode(F, cctx, *R);
  } else if (auto *R = dyn_cast<PReluNode>(node)) {
    lowerPReluNode(F, cctx, *R);
  } else if (auto *P = dyn_cast<PadNode>(node)) {
    lowerPadNode(F, cctx, *P);
  } else if (auto *THG = dyn_cast<TanhGradNode>(node)) {
    lowerTanhGradNode(F, cctx, *THG);
  } else if (auto *SG = dyn_cast<SigmoidGradNode>(node)) {
    lowerSigmoidGradNode(F, cctx, *SG);
  } else if (auto *SGD = dyn_cast<SGDNode>(node)) {
    lowerSGDNode(F, cctx, *SGD);
  } else if (auto *BN = dyn_cast<BatchNormalizationNode>(node)) {
    lowerBatchNormalizationNode(F, cctx, *BN);
  } else if (auto *MVN = dyn_cast<MeanVarNormalizationNode>(node)) {
    lowerMeanVarNormalizationNode(F, cctx, *MVN);
  } else if (auto *BNG = dyn_cast<BatchNormalizationGradNode>(node)) {
    lowerBatchNormalizationGradNode(F, cctx, *BNG);
  } else if (auto *SCEL = dyn_cast<SigmoidCrossEntropyWithLogitsNode>(node)) {
    lowerSigmoidCrossEntropyWithLogitsNode(F, cctx, *SCEL);
  } else if (auto *RMN = dyn_cast<BatchedReduceMeanNode>(node)) {
    lowerBatchReduceMeanNode(F, cctx, *RMN);
  } else if (auto *CN = dyn_cast<ConvolutionNode>(node)) {
    if (CN->getGroup() > 1) {
      lowerGroupConvolutionNode(F, cctx, *CN);
    }
  } else if (auto *CQC = dyn_cast<ChannelwiseQuantizedConvolutionNode>(node)) {
    lowerChannelwiseQuantizedConvolutionNode(F, cctx, *CQC);
  } else if (auto *TN = dyn_cast<TileNode>(node)) {
    lowerTileNode(F, cctx, *TN);
  } else if (auto *CSN = dyn_cast<ChannelShuffleNode>(node)) {
    lowerChannelShuffleNode(F, cctx, *CSN);
  } else if (auto *RN = dyn_cast<ReplaceNaNNode>(node)) {
    lowerReplaceNaNNode(F, cctx, *RN);
  } else if (auto *BMMN = dyn_cast<BatchMatMulNode>(node)) {
    lowerBatchMatMulNode(F, cctx, *BMMN);
  } else if (auto *SLSN = dyn_cast<SparseLengthsSumNode>(node)) {
    lowerSparseLengthsSumNode(F, cctx, *SLSN);
  } else if (auto *FQSLSN =
                 dyn_cast<FusedRowwiseQuantizedSparseLengthsSumNode>(node)) {
    lowerFusedRowwiseQuantizedSparseLengthsSumNode(F, cctx, *FQSLSN);
  }
}

void glow::lower(Function *F, CompilationContext &cctx, const Backend *B,
                 const KindSet &doNotLowerKinds) {
  LOG_SCOPE(F->getLogContext(), "glow::lower")

  auto &nodes = F->getNodes();
  for (auto &N : nodes) {
    if (B && !B->shouldLower(&N)) {
      continue;
    }
    if (doNotLowerKinds.count(N.getKind())) {
      continue;
    }
    lowerNode(F, &N, cctx);
  }

  for (auto it = F->getNodes().begin(), e = F->getNodes().end(); it != e;) {
    auto cur = &*(it++);
    if (dyn_cast<SGDNode>(cur)) {
      F->eraseNode(cur);
    }
  }

  // Remove nodes that were lowered.
  DCE().run(F);
}
