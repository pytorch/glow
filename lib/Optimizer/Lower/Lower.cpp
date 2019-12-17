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

#include "glow/Optimizer/Lower/Lower.h"

#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/TensorLayout.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

#include <numeric>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;

#define DECORATE_NODE_NAME(Node, ...)                                          \
  llvm::join_items("_", Node.getName(), __VA_ARGS__)

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

  auto *lhsResult =
      F->createMul(DECORATE_NODE_NAME(node, "grad", "rhs"), outG, RHS);
  auto *rhsResult =
      F->createMul(DECORATE_NODE_NAME(node, "grad", "lhs"), outG, LHS);
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
  auto *sub = F->createSub(DECORATE_NODE_NAME(node, "grad"), zero, outG);
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

  auto *lhsResult =
      F->createDiv(DECORATE_NODE_NAME(node, "grad", "lhs"), outG, RHS);

  auto *zero = F->createSplat("zero", outG.getType(), 0);
  auto *subGrad = F->createSub(DECORATE_NODE_NAME(node, "grad"), zero, outG);
  auto *mulLhsGrad = F->createMul(
      DECORATE_NODE_NAME(node, "grad", "mul", "lhs"), subGrad, LHS);

  auto *squareRhs =
      F->createMul(DECORATE_NODE_NAME(node, "grad", "square", "rhs"), RHS, RHS);
  auto *rhsResult = F->createDiv(DECORATE_NODE_NAME(node, "grad", "div", "rhs"),
                                 mulLhsGrad, squareRhs);

  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedLHS(),
                       lhsResult);
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getGradOfInputNamedRHS(),
                       rhsResult);
}

static void lowerRegressionNode(Function *, CompilationContext &cctx,
                                const RegressionNode &node) {
  auto input = node.getInput();
  replaceAllUsesOfWith(cctx.loweredInfoMap, node.getResult(), input);
}

static void lowerRegressionGradNode(Function *F, CompilationContext &cctx,
                                    const RegressionGradNode &node) {
  LOG_SCOPE(F->getLogContext(), "lowerRegressionGradNode")

  auto outG = node.getInput();

  auto *inputG = F->createSub(DECORATE_NODE_NAME(node, "grad"), node.getInput(),
                              node.getExpected());
  auto *expG = F->createSplat(DECORATE_NODE_NAME(node, "grad", "exp"),
                              node.getExpected().getType(), 0);

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
  auto *mul =
      F->createMatMul(DECORATE_NODE_NAME(FC, "dot"), OT, FC.getInput(), W);
  auto *add = F->createBatchedAdd(DECORATE_NODE_NAME(FC, "bias"), OT, mul,
                                  FC.getBias());
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
  auto *wT = F->createTranspose(DECORATE_NODE_NAME(FCG, "weight", "transpose"),
                                FCG.getWeights(), {1, 0});
  auto *dx2 =
      F->createMatMul(DECORATE_NODE_NAME(FCG, "weight", "dot"), dout, wT);
  auto *dx = F->createReshape(
      DECORATE_NODE_NAME(FCG, "weight", "reshape"), dx2,
      FCG.getInput().getType()->dims(),
      CanonicalTensorLayout::getInstance().getNthInputLayoutRequirements(
          &FCG, FullyConnectedGradNode::InputIdx));
  replaceAllUsesOfWith(cctx.loweredInfoMap, FCG.getGradOfInputNamedInput(), dx);

  // dw = xT * dout.
  Node *x2 = F->createFlatten(DECORATE_NODE_NAME(FCG, "output", "flatten"),
                              FCG.getInput(), 1);
  auto *x2T = F->createTranspose(DECORATE_NODE_NAME(FCG, "output", "transpose"),
                                 x2, {1, 0});
  auto *dw =
      F->createMatMul(DECORATE_NODE_NAME(FCG, "output", "dot"), x2T, dout);
  replaceAllUsesOfWith(cctx.loweredInfoMap, FCG.getGradOfInputNamedWeights(),
                       dw);

  // db = reduce(dout).
  auto *db = F->createBatchedReduceAdd(
      DECORATE_NODE_NAME(FCG, "bias", "reduce"), dout, /* axis */ 0);
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
  auto *relu = F->createMax(DECORATE_NODE_NAME(R, "max"),
                            R.getResult().getType(), zero, R.getInput());
  replaceAllUsesOfWith(cctx.loweredInfoMap, R.getResult(), relu);
}

static void lowerPReluNode(Function *F, CompilationContext &cctx,
                           const PReluNode &R) {
  // PRelu is :
  // slope * x    if x < 0
  // x            if x >= 0
  // where slope is an input from a different node.

  LOG_SCOPE(F->getLogContext(), "lowerPReluNode")

  auto *zeroSplat = F->createSplat("zero", R.getResult().getType(), 0.0);
  auto *cmplgt =
      F->createCmpLTE(DECORATE_NODE_NAME(R, "cmplte"), zeroSplat, R.getInput());
  auto *mul =
      F->createMul(DECORATE_NODE_NAME(R, "mul"), R.getSlope(), R.getInput());
  auto *prelu = F->createSelect(DECORATE_NODE_NAME(R, "select"), cmplgt,
                                R.getInput(), mul);

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

  SplatNode *constant = F->createSplat(DECORATE_NODE_NAME(P, "pad", "const"),
                                       outputType, P.getValue());

  std::vector<dim_t> orig(numDims);
  for (dim_t i = 0; i < numDims; i++) {
    orig[i] = size_t(pads[i]);
  }

  auto *insert = F->createInsertTensor(DECORATE_NODE_NAME(P, "insert"),
                                       constant, P.getInput(), orig);
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
    auto *L1DecaySplat = F->createSplat(
        DECORATE_NODE_NAME(SGD, "splat", "l1_decay"), type, L1Decay);
    auto *zeroSplat = F->createSplat("zero", type, 0);
    auto *oneSplat = F->createSplat("one", type, 1);
    auto *minusOneSplat = F->createSplat("minusOne", type, -1);

    auto *Wcmp = F->createCmpLTE(DECORATE_NODE_NAME(SGD, "Wcmp"), zeroSplat, W);
    auto *Wdir = F->createSelect(DECORATE_NODE_NAME(SGD, "Wdir"), Wcmp,
                                 oneSplat, minusOneSplat);
    auto *L1Grad =
        F->createMul(DECORATE_NODE_NAME(SGD, "L1Grad"), L1DecaySplat, Wdir);

    gij = F->createAdd(DECORATE_NODE_NAME(SGD, "gij_with_l1"), gij, L1Grad);
  }
  if (L2Decay != 0.0f) {
    auto *L2DecaySplat =
        F->createSplat(DECORATE_NODE_NAME(SGD, "L2DecaySplat"), type, L2Decay);

    auto *L2Grad =
        F->createMul(DECORATE_NODE_NAME(SGD, "L2Grad"), L2DecaySplat, W);

    gij = F->createAdd(DECORATE_NODE_NAME(SGD, "gij_with_l2"), gij, L2Grad);
  }
  if (batchSize > 1) {
    auto *batchSizeSplat = F->createSplat(
        DECORATE_NODE_NAME(SGD, "batchSizeSplat"), type, batchSize);
    gij = F->createDiv(DECORATE_NODE_NAME(SGD, "gij_div_batchSz"), gij,
                       batchSizeSplat);
  }

  auto *negLearningRateSplat = F->createSplat(
      DECORATE_NODE_NAME(SGD, "learningRateSplat"), type, -learningRate);
  Node *dx =
      F->createMul(DECORATE_NODE_NAME(SGD, "dx"), negLearningRateSplat, gij);

  // Use the momentum to improve the gradient descent:
  // http://ufldl.stanford.edu/tutorial/supervised/
  // OptimizationStochasticGradientDescent/
  if (momentum > 0.0) {
    Placeholder *Gsum = F->getParent()->createPlaceholder(
        W.getType(), DECORATE_NODE_NAME(SGD, "gsum"), false);
    Gsum->setAllocZero();

    auto *momentumSplat = F->createSplat(
        DECORATE_NODE_NAME(SGD, "momentumSplat"), type, momentum);
    auto *GsumMult =
        F->createMul(DECORATE_NODE_NAME(SGD, "GsumMult"), momentumSplat, Gsum);

    dx =
        F->createAdd(DECORATE_NODE_NAME(SGD, "dx_with_momentum"), GsumMult, dx);
    F->createSave(DECORATE_NODE_NAME(SGD, "save", "gsum"), dx, Gsum);
  }

  auto *newW = F->createAdd(DECORATE_NODE_NAME(SGD, "weight", "update"), W, dx);
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

  auto *epsilonSplat =
      F->createSplat(DECORATE_NODE_NAME(BN, "epsilon"), var.getType(), epsilon);
  Node *coef =
      F->createAdd(DECORATE_NODE_NAME(BN, "var_plus_eps"), var, epsilonSplat);
  coef = F->createPow(DECORATE_NODE_NAME(BN, "sqrt_var_plus_eps"), coef, 0.5);
  coef = F->createDiv(DECORATE_NODE_NAME(BN, "inverse_sqrt_var_plus_eps"),
                      gamma, coef);

  // Apply: out := (in - mean) * coef + beta
  // in and out are of the same size, while others must be broadcasted.
  auto *meanB = F->createBroadcast(DECORATE_NODE_NAME(BN, "muBroadcasted"),
                                   mean, in.dims(), channelIdx);
  auto *coefB = F->createBroadcast(DECORATE_NODE_NAME(BN, "coefBroadcasted"),
                                   coef, in.dims(), channelIdx);
  auto *betaB = F->createBroadcast(DECORATE_NODE_NAME(BN, "betaBroadcasted"),
                                   beta, in.dims(), channelIdx);

  Node *newResult =
      F->createSub(DECORATE_NODE_NAME(BN, "in_minus_mean"), in, meanB);
  newResult =
      F->createMul(DECORATE_NODE_NAME(BN, "mul_coef"), newResult, coefB);
  newResult = F->createAdd(DECORATE_NODE_NAME(BN, "result"), newResult, betaB);

  replaceAllUsesOfWith(cctx.loweredInfoMap, BN.getResult(), newResult);
}

static void lowerLayerNormalizationNode(Function *F, CompilationContext &cctx,
                                        const LayerNormalizationNode &LN) {
  LOG_SCOPE(F->getLogContext(), "lowerLayerNormalizationNode")

  auto in = LN.getInput();
  auto out = LN.getResult();

  auto gamma = LN.getScale();
  auto beta = LN.getBias();
  float epsilon = LN.getEpsilon();

  auto nodeName = LN.getName();

  // input shape -> {M, N} where N is the size of each layer to be normalized.
  dim_t N = std::accumulate(gamma.dims().begin(), gamma.dims().end(), 1,
                            std::multiplies<size_t>());
  dim_t M = in.getType()->size() / N;
  in = F->createReshape(DECORATE_NODE_NAME(LN, "reshape"), in, {M, N})
           ->getResult();

  // Compute mean and standard deviation for each layer using the formula from
  // https://pytorch.org/docs/stable/nn.html#torch.nn.LayerNorm

  // {M, N} -> {M}
  auto mean =
      F->createBatchedReduceAdd(DECORATE_NODE_NAME(LN, "mean", "add"), in,
                                /*axes*/ {1})
          ->getResult();

  // {M, N} -> {M}
  auto inSquared =
      F->createMul(DECORATE_NODE_NAME(LN, "squared"), in, in)->getResult();
  auto stdDev = F->createBatchedReduceAdd(
                     DECORATE_NODE_NAME(LN, "stddev", "add"), inSquared,
                     /*axes*/ {1})
                    ->getResult();

  // {M}
  auto nSplat = F->createSplat(DECORATE_NODE_NAME(LN, "n"), mean.getType(), N)
                    ->getResult();
  auto epsSplat =
      F->createSplat(DECORATE_NODE_NAME(LN, "epsilon"), mean.getType(), epsilon)
          ->getResult();
  auto oneSplat = F->createSplat("one", mean.getType(), 1.0)->getResult();
  auto negOneSplat =
      F->createSplat("minusOne", mean.getType(), -1.0)->getResult();

  // {M}
  mean = F->createDiv(DECORATE_NODE_NAME(LN, "mean", "div"), mean, nSplat)
             ->getResult();

  // {M}
  stdDev = F->createDiv(DECORATE_NODE_NAME(LN, "stddev", "div"), stdDev, nSplat)
               ->getResult();

  // {M}
  auto meanSquared =
      F->createMul(DECORATE_NODE_NAME(LN, "mean", "squared"), mean, mean)
          ->getResult();
  stdDev =
      F->createSub(DECORATE_NODE_NAME(LN, "stddev", "sub"), stdDev, meanSquared)
          ->getResult();
  stdDev = F->createRELU(DECORATE_NODE_NAME(LN, "stddev", "relu"), stdDev)
               ->getResult();
  stdDev =
      F->createAdd(DECORATE_NODE_NAME(LN, "stddev", "add"), stdDev, epsSplat)
          ->getResult();
  stdDev = F->createPow(DECORATE_NODE_NAME(LN, "stddev", "pow"), stdDev, 0.5)
               ->getResult();
  stdDev = F->createDiv(DECORATE_NODE_NAME(LN, "stddev", "reciprocal"),
                        oneSplat, stdDev)
               ->getResult();

  // {M}
  auto scale = stdDev;
  auto bias =
      F->createMul(DECORATE_NODE_NAME(LN, "bias", "mul"), stdDev, negOneSplat)
          ->getResult();
  bias = F->createMul(DECORATE_NODE_NAME(LN, "bias", "mul"), bias, mean)
             ->getResult();

  // Broadcast mean and std deviation to the size of each batch
  // {M} -> {M, N}
  scale = F->createReshape(DECORATE_NODE_NAME(LN, "scale", "reshape"), scale,
                           {M, 1})
              ->getResult();
  bias =
      F->createReshape(DECORATE_NODE_NAME(LN, "bias", "reshape"), bias, {M, 1})
          ->getResult();
  scale = F->createTile(DECORATE_NODE_NAME(LN, "scale", "tile"), scale, N, 1)
              ->getResult();
  bias = F->createTile(DECORATE_NODE_NAME(LN, "bias", "tile"), bias, N, 1)
             ->getResult();

  // Broadcast beta and gamma across batches
  // {N} -> {M, N}
  beta =
      F->createReshape(DECORATE_NODE_NAME(LN, "beta", "reshape"), beta, {1, N})
          ->getResult();
  gamma = F->createReshape(DECORATE_NODE_NAME(LN, "gamma", "reshape"), gamma,
                           {1, N})
              ->getResult();
  beta = F->createTile(DECORATE_NODE_NAME(LN, "beta", "tile"), beta, M, 0)
             ->getResult();
  gamma = F->createTile(DECORATE_NODE_NAME(LN, "gamma", "tile"), gamma, M, 0)
              ->getResult();

  // Normalize layers
  // {M, N}
  auto output =
      F->createMul(DECORATE_NODE_NAME(LN, "output", "scale"), in, scale)
          ->getResult();
  output = F->createAdd(DECORATE_NODE_NAME(LN, "output", "bias"), output, bias)
               ->getResult();
  output =
      F->createMul(DECORATE_NODE_NAME(LN, "output", "gamma"), output, gamma)
          ->getResult();
  output = F->createAdd(DECORATE_NODE_NAME(LN, "output", "beta"), output, beta)
               ->getResult();

  // {M, N} -> output shape
  output = F->createReshape(nodeName, output, out.getType()->dims());

  replaceAllUsesOfWith(cctx.loweredInfoMap, out, output);
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
  const dim_t numChannels = in.dims()[channelIdx];
  // The number of elements that each channel holds.
  const dim_t samplesPerChannel = in.getType()->size() / numChannels;

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
    inPrep = F->createTranspose(DECORATE_NODE_NAME(MVN, "in", "transpose"), in,
                                perm);
  }
  // Reshape input tensor to form:
  // {samplesPerChannel, numChannels}
  ReshapeNode *inFlat =
      F->createReshape(DECORATE_NODE_NAME(MVN, "in", "flat"), inPrep,
                       {samplesPerChannel, numChannels});

  // Calculate Mean:

  // sum(in[i])
  // reduce the tensor by the first dimension, to get {numChannels}
  auto *batchedAdd = F->createBatchedReduceAdd(
      DECORATE_NODE_NAME(MVN, "in", "sum"), inFlat, /* axis */ 0);
  // Mean = sum(in[i]) / N
  auto samplesPerChannelSplat =
      F->createSplat(DECORATE_NODE_NAME(MVN, "samplesPerChannelSplat"),
                     batchedAdd->getResult().getType(), samplesPerChannel);
  DivNode *localMean = F->createDiv(DECORATE_NODE_NAME(MVN, "localMean"),
                                    batchedAdd, samplesPerChannelSplat);

  // Calculate Variance:
  // sum((x - mu) ^ 2)
  auto *localMeanB =
      F->createBroadcast(DECORATE_NODE_NAME(MVN, "new_mean_broadcasted"),
                         localMean, inFlat->getResult().dims(), 1);

  Node *localVar =
      F->createSub(DECORATE_NODE_NAME(MVN, "x_mu"), inFlat, localMeanB);
  localVar = F->createPow(DECORATE_NODE_NAME(MVN, "x_mu2"), localVar, 2);
  localVar = F->createBatchedReduceAdd(DECORATE_NODE_NAME(MVN, "x_mu2_sum"),
                                       localVar, /* axis */ 0);
  // Var = sum((x - mu) ^ 2) / N
  localVar = F->createDiv(DECORATE_NODE_NAME(MVN, "localVar"), localVar,
                          samplesPerChannelSplat);

  // Update the global variance and mean:
  auto *momentumSplat =
      F->createSplat(DECORATE_NODE_NAME(MVN, "momentumSplat"),
                     localMean->getResult().getType(), momentum);
  auto *oneMinusMomentumSplat =
      F->createSplat(DECORATE_NODE_NAME(MVN, "oneMinusMomentumSplat"),
                     localMean->getResult().getType(), 1 - momentum);

  // newMean := P * localMean + (1 - P) * oldMean
  auto *newMean = F->createAdd(
      DECORATE_NODE_NAME(MVN, "newMean"),
      F->createMul(DECORATE_NODE_NAME(MVN, "momentum_by_localMean"),
                   momentumSplat, localMean),
      F->createMul(DECORATE_NODE_NAME(MVN, "1_momentum_by_oldMean"),
                   oneMinusMomentumSplat, inMean));
  // newVar := P * localVar + (1 - P) * oldVar
  auto *newVar =
      F->createAdd(DECORATE_NODE_NAME(MVN, "newVar"),
                   F->createMul(DECORATE_NODE_NAME(MVN, "momentum_by_localVar"),
                                momentumSplat, localVar),
                   F->createMul(DECORATE_NODE_NAME(MVN, "1_momentum_by_oldVar"),
                                oneMinusMomentumSplat, inVar));

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
  const dim_t numChannels = inW.dims()[channelIdx];
  // The number of elements that each channel holds.
  const dim_t samplesPerChannel = inW.getType()->size() / numChannels;

  // Calculate: sum(dy * (h - mu))
  auto *meanB = F->createBroadcast(DECORATE_NODE_NAME(BNG, "mean_broadcasted"),
                                   mean, inW.dims(), channelIdx);
  auto *hmu = F->createSub(DECORATE_NODE_NAME(BNG, "x_minus_mean"), inW, meanB);
  NodeValue sumDyhmu =
      F->createMul(DECORATE_NODE_NAME(BNG, "dy_mul_h_minus_mu"), outG, hmu);

  // Calculate: sum(dy)
  NodeValue sumDy = outG;

  // TODO: consider adding this functionality to the main operator set.
  if (channelIdx + 1 != inW.dims().size()) {
    std::vector<unsigned_t> perm(inW.dims().size());
    for (size_t i = 0; i < perm.size(); i++) {
      perm[i] = i;
    }
    std::swap(perm[channelIdx], perm[perm.size() - 1]);

    sumDyhmu = F->createTranspose(
        DECORATE_NODE_NAME(BNG, "sumDyhmu", "transpose"), sumDyhmu, perm);
    sumDy = F->createTranspose(DECORATE_NODE_NAME(BNG, "sumDy", "transpose"),
                               sumDy, perm);
  }
  sumDyhmu = F->createReshape(DECORATE_NODE_NAME(BNG, "sumDyhmu", "flat"),
                              sumDyhmu, {samplesPerChannel, numChannels});
  sumDy = F->createReshape(DECORATE_NODE_NAME(BNG, "sumDy", "flat"), sumDy,
                           {samplesPerChannel, numChannels});
  sumDyhmu = F->createBatchedReduceAdd(
      DECORATE_NODE_NAME(BNG, "sumDyhmu", "reduced"), sumDyhmu, /* axis */ 0);
  sumDy = F->createBatchedReduceAdd(DECORATE_NODE_NAME(BNG, "sumDy", "reduced"),
                                    sumDy, /* axis */ 0);

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

  auto *epsilonSplat = F->createSplat(DECORATE_NODE_NAME(BNG, "epsilon"),
                                      var.getType(), epsilon);
  auto *oneSplat = F->createSplat("one", var.getType(), 1.0);
  auto *invNSplat = F->createSplat(DECORATE_NODE_NAME(BNG, "invNSplat"),
                                   var.getType(), 1.0 / samplesPerChannel);
  Node *invVar =
      F->createAdd(DECORATE_NODE_NAME(BNG, "var_plus_eps"), var, epsilonSplat);
  invVar = F->createDiv(DECORATE_NODE_NAME(BNG, "inverse_var_plus_eps"),
                        oneSplat, invVar);
  Node *invVarSqrt =
      F->createPow(DECORATE_NODE_NAME(BNG, "invVarSqrt"), invVar, 0.5);

  Node *coef1 = F->createMul(
      DECORATE_NODE_NAME(BNG, "invN_gamma_invVarSqrt"),
      F->createMul(DECORATE_NODE_NAME(BNG, "invN_gamma"), invNSplat, gamma),
      invVarSqrt);
  Node *coef2 = F->createMul(DECORATE_NODE_NAME(BNG, "invVar_sumDyhmu"), invVar,
                             sumDyhmu);

  // Apply:
  // inG := Bcast(coef1) * (NSplat * outG - Bcast(sumDy) - hmu * Bcast(coef2))

  coef1 = F->createBroadcast(DECORATE_NODE_NAME(BNG, "coef1_broadcasted"),
                             coef1, inW.dims(), channelIdx);
  coef2 = F->createBroadcast(DECORATE_NODE_NAME(BNG, "coef2_broadcasted"),
                             coef2, inW.dims(), channelIdx);
  auto *sumDyB =
      F->createBroadcast(DECORATE_NODE_NAME(BNG, "sumDy_broadcasted"), sumDy,
                         inW.dims(), channelIdx);
  auto *NSplat = F->createSplat(DECORATE_NODE_NAME(BNG, "samplesPerChannel"),
                                inW.getType(), samplesPerChannel);
  Node *inBrackets =
      F->createMul(DECORATE_NODE_NAME(BNG, "NSplat_outG"), NSplat, outG);
  inBrackets = F->createSub(
      DECORATE_NODE_NAME(BNG, "inBrackets"),
      F->createSub(DECORATE_NODE_NAME(BNG, "inBrackets_2ops"), inBrackets,
                   sumDyB),
      F->createMul(DECORATE_NODE_NAME(BNG, "hmu_coef2"), hmu, coef2));

  auto *inG = F->createMul(DECORATE_NODE_NAME(BNG, "inG"), coef1, inBrackets);
  replaceAllUsesOfWith(cctx.loweredInfoMap, BNG.getGradOfInputNamedInput(),
                       inG);

  replaceAllUsesOfWith(cctx.loweredInfoMap, BNG.getGradOfInputNamedBias(),
                       sumDy);

  auto *gammaG =
      F->createMul(DECORATE_NODE_NAME(BNG, "gammaG"), sumDyhmu, invVarSqrt);
  replaceAllUsesOfWith(cctx.loweredInfoMap, BNG.getGradOfInputNamedScale(),
                       gammaG);

  auto *zeroSplat = F->createSplat("zero", var.getType(), 0);
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

static void lowerBucketizeNode(Function *F, CompilationContext &cctx,
                               const BucketizeNode &B) {
  // Bucketize is:
  // Boundaries[i-1] < x <= Boundaries[i]
  // If 'x' is beyond the bounds of Boundaries,
  // 0 or len(Boundaries) is returned as appropriate.
  // The node is lowered as follows:
  // 1. For each value node in the input, broadcast it #buckets
  // 2. Compare the broadcasted value to the buckets
  // 3. Count the number of buckets smaller than the current value:
  // 3.1 If they are all bigger = output is zero
  // 3.2 Else if they are all smaller = output is len(Boundaries)
  // 3.2 Else output is Boundaries[i-1] < x <= Boundaries[i]
  // 4. Gather the all 'x's and create a new tensor to replace the node with
  dim_t numOfBuckets = (dim_t)B.getBoundaries().size();
  const std::string &baseStr = B.getName().str();
  auto *boundariesConst = F->getParent()->createConstant(
      ElemKind::FloatTy, {numOfBuckets}, baseStr + ".const");
  boundariesConst->getPayloadMutable().getHandle<float>() = B.getBoundaries();
  auto *zeroSplat =
      F->createSplat("zeroSplat", boundariesConst->getType(), 0.0);
  auto *oneSplat = F->createSplat("oneSplat", boundariesConst->getType(), 1.0);
  auto *reshapedInput =
      F->createReshape(baseStr + ".reshape.input", B.getInput(),
                       {B.getInput().getType()->size()}, "N");
  std::vector<NodeValue> results;
  for (size_t i = 0, e = reshapedInput->getResult().getType()->size(); i < e;
       i++) {
    std::string currBaseStr = baseStr + "." + std::to_string(i);
    auto *slicedInput =
        F->createSlice(currBaseStr + ".slice", reshapedInput, i, i + 1);
    auto *broadcastedInput = F->createBroadcast(
        currBaseStr + ".input", slicedInput, {numOfBuckets}, /* axis */ 0);
    auto *cmpBoundaryLgt = F->createCmpLTE(currBaseStr + ".cmpLTE",
                                           boundariesConst, broadcastedInput);
    auto *selectBucket = F->createSelect(currBaseStr + ".select",
                                         cmpBoundaryLgt, oneSplat, zeroSplat);
    auto *reduceAdd = F->createBatchedReduceAdd(
        currBaseStr + ".reduceAdd", slicedInput->getResult().getType(),
        selectBucket, /* axis */ 0);
    results.push_back(reduceAdd);
  }
  auto *resultConcat =
      F->createConcat(baseStr + ".concat", results, /* axis */ 0);
  auto *resultReshape = F->createReshape(baseStr + ".reshape.output",
                                         resultConcat, B.getResult().dims());
  auto *convertToIndex = F->createConvertTo(
      baseStr + ".convertTo", resultReshape, B.getResult().getType());
  replaceAllUsesOfWith(cctx.loweredInfoMap, B.getResult(), convertToIndex);
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
  auto start = std::vector<dim_t>(input.dims().size(), 0);

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
  auto *T = F->createTranspose(CSN.getName().str() + ".transpose", R1,
                               transpose, R1->getLayout());

  auto *R2 = F->createReshape(CSN.getName().str() + ".reshape2", T, inDims,
                              T->getLayout());
  replaceAllUsesOfWith(cctx.loweredInfoMap, CSN.getResult(), R2);
}

static void lowerBatchedReduceMeanNode(Function *F, CompilationContext &cctx,
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
  const dim_t numBatches = lhs.dims()[0];
  const dim_t N = lhs.dims()[1];
  const dim_t M = lhs.dims()[2];
  const dim_t P = rhs.dims()[2];

  // Lower to Slices, MatMuls, and a final Concat. Multiply i-th LHS matrix
  // {N, M} by i-th RHS matrix {M, P} to get final matrix {numBatches, N, P}.
  std::vector<NodeValue> MMS(numBatches);
  // Use the original quantization parameters from the BMM for each MM.
  const TypeRef outTy = F->getParent()->uniqueTypeWithNewShape(
      BMMN.getResult().getType(), {N, P});
  for (dim_t i = 0; i < numBatches; i++) {
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
    MMS[i] = F->createMatMul(name.str() + ".MatMul." + std::to_string(i), outTy,
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
  auto ty = F->getParent()->uniqueType(
      FRQSLSN.getResult().getType()->getElementType(),
      {FRQSLSN.getIndices().dims()[0]});
  auto *ones = F->createSplat(FRQSLSN.getName().str() + ".ones", ty, 1.0);
  auto *FRQSLWSN = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      FRQSLSN.getName().str(), FRQSLSN.getData(), ones, FRQSLSN.getIndices(),
      FRQSLSN.getLengths(), FRQSLSN.getUseFP16Accumulation());

  replaceAllUsesOfWith(cctx.loweredInfoMap, FRQSLSN.getResult(), FRQSLWSN);
}

static void lowerBatchBoxCoxNode(Function *F, CompilationContext &cctx,
                                 const BatchBoxCoxNode &BBCN) {
  auto name = BBCN.getName();
  auto data = BBCN.getInput();
  auto lambda1 = BBCN.getLambda1();
  auto lambda2 = BBCN.getLambda2();

  // Broadcast lambda1 and lambda2 so that they are both the same size as the
  // data.
  auto *BL1 =
      F->createBroadcast(name.str() + ".broadcast", lambda1, data.dims(),
                         /*axis=*/1);
  auto *BL2 =
      F->createBroadcast(name.str() + ".broadcast", lambda2, data.dims(),
                         /*axis=*/1);

  // Broadcast is usually implemented via a Tile node returned from
  // createBroadcast(). However, if the Broadcast was a noop then there is a
  // Reshape instead of a Tile returned. Thus, get the index here to use based
  // on the returned kinds from createBroadcast() above.
  assert((llvm::isa<TileNode>(BL1) || llvm::isa<ReshapeNode>(BL1)) &&
         "Broadcast is assumed to be either implemented via Tile or Reshape.");
  TypeRef typeBL1 = llvm::isa<TileNode>(BL1)
                        ? BL1->getType(TileNode::ResultIdx)
                        : BL1->getType(ReshapeNode::ResultIdx);

  // Add a small epsilon to lambda1 so that we can avoid dividing by zero
  // later. It doesn't matter that this is technically incorrect because the
  // final Select will discard the results of this computation.
  auto *eps = F->createSplat(name.str() + ".eps", typeBL1, BBCN.getEpsilon());
  auto *EBL1 = F->createAdd(name.str() + ".lambda1eps", BL1, eps);

  // Compute data + BL2, which is needed regardless of whether
  // lambda1 is 0 or not.
  auto *AN = F->createAdd(name.str() + ".add", data, BL2);

  // Take the max of data + BL2 and 1e-6 to void exponentiating or taking the
  // logarithm of too small a number.
  auto *minArg = F->createSplat(name.str() + ".logpowmin",
                                AN->getResult().getType(), 1e-6);
  auto *MN = F->createMax(name.str() + ".max", AN, minArg);

  // Compute the Box-Cox transform for the lambda1 == 0 case:
  //    y = ln(max(x + lambda2, 1e-6))

  auto *LN = F->createLog(name.str() + ".log", MN);

  // Compute the Box-Cox transform for the lambda1 != 0 case:
  //    y = (max(x + lambda2, 1e-6)^lambda1 - 1)/lambda1
  auto *PN = F->createPow(name.str() + ".pow", MN, BL1);
  auto *ones =
      F->createSplat(name.str() + ".ones", PN->getResult().getType(), 1.0f);
  auto *SN = F->createSub(name.str() + ".sub", PN, ones);
  // Divide by EBL1, not BL1 to avoid a divide-by-zero exception.
  auto *DN = F->createDiv(name.str() + ".div", SN, EBL1);

  // Compute predicates for selecting between the two cases above.
  auto *zeroes = F->createSplat(name.str() + ".zeroes", typeBL1, 0.0f);
  auto *predicate = F->createCmpEQ(name.str() + ".cmpeq", BL1, zeroes);

  // Create Select to pick between the two Box-Cox cases.
  auto *select = F->createSelect(name.str() + ".select", predicate, LN, DN);
  replaceAllUsesOfWith(cctx.loweredInfoMap, BBCN.getResult(), select);
}

static void lowerClipNode(Function *F, CompilationContext &cctx,
                          const ClipNode &CN) {
  auto const &name = CN.getName();
  auto min = CN.getMin();
  auto max = CN.getMax();
  auto type = CN.getResult().getType();
  auto *minSplat = F->createSplat(name.str() + ".minSplat", type, min);
  auto *minClipped =
      F->createMax(name.str() + ".minClip", CN.getInput(), minSplat);
  auto *maxSplat = F->createSplat(name.str() + ".maxSplat", type, max);
  auto result = F->createMin(name.str(), minClipped, maxSplat);
  replaceAllUsesOfWith(cctx.loweredInfoMap, CN.getResult(), result);
}

bool glow::lowerNode(Function *F, Node *node, CompilationContext &cctx) {
#define CASE_LOWER(NODE_NAME_)                                                 \
  case Kinded::Kind::NODE_NAME_##NodeKind:                                     \
    lower##NODE_NAME_##Node(F, cctx, *cast<NODE_NAME_##Node>(node));           \
    return true;

  switch (node->getKind()) {
    CASE_LOWER(Regression);
    CASE_LOWER(RegressionGrad);
    CASE_LOWER(AddGrad);
    CASE_LOWER(MulGrad);
    CASE_LOWER(SubGrad);
    CASE_LOWER(DivGrad);
    CASE_LOWER(FullyConnected);
    CASE_LOWER(FullyConnectedGrad);
    CASE_LOWER(Relu);
    CASE_LOWER(ReluGrad);
    CASE_LOWER(PRelu);
    CASE_LOWER(Pad);
    CASE_LOWER(TanhGrad);
    CASE_LOWER(SigmoidGrad);
    CASE_LOWER(SGD);
    CASE_LOWER(BatchNormalization);
    CASE_LOWER(LayerNormalization);
    CASE_LOWER(MeanVarNormalization);
    CASE_LOWER(BatchNormalizationGrad);
    CASE_LOWER(SigmoidCrossEntropyWithLogits);
    CASE_LOWER(BatchedReduceMean);
    CASE_LOWER(Bucketize);
    CASE_LOWER(ChannelwiseQuantizedConvolution);
    CASE_LOWER(ChannelShuffle);
    CASE_LOWER(Tile);
    CASE_LOWER(ReplaceNaN);
    CASE_LOWER(BatchMatMul);
    CASE_LOWER(SparseLengthsSum);
    CASE_LOWER(FusedRowwiseQuantizedSparseLengthsSum);
    CASE_LOWER(BatchBoxCox);
    CASE_LOWER(Clip);
  case Kinded::Kind::ConvolutionNodeKind: {
    ConvolutionNode *CN = cast<ConvolutionNode>(node);
    if (CN->getGroup() > 1 && CN->hasFusedActivation()) {
      lowerGroupConvolutionNode(F, cctx, *CN);
      return true;
    }
  }
    return false;
  default:
    return false;
  }
}
