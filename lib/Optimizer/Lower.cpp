// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::dyn_cast;

void lowerAddGradNode(Function *F, AddGradNode &node) {
  /// The chain rule for addition:
  /// delta(LHS) = dF/dLHS * delta(OUT) = 1 * delta(OUT)
  /// delta(RHS) = dF/dRHS * delta(OUT) = 1 * delta(OUT)
  auto outG = node.getGradOfOriginalOutputNamedResult();
  node.getGradOfInputNamedLHS().replaceAllUsesOfWith(outG);
  node.getGradOfInputNamedRHS().replaceAllUsesOfWith(outG);
}
void lowerMulGradNode(Function *F, MulGradNode &node) {
  /// The chain rule for multiplication:
  /// delta(LHS) = dF/dLHS * delta(OUT) = RHS * delta(OUT)
  /// delta(RHS) = dF/dRHS * delta(OUT) = LHS * delta(OUT)
  auto outG = node.getGradOfOriginalOutputNamedResult();
  NodeValue LHS = node.getLHS();
  NodeValue RHS = node.getRHS();

  auto lhsResult = F->createMul("mul.grad.rhs", outG, RHS);
  auto rhsResult = F->createMul("mul.grad.lhs", outG, LHS);
  node.getGradOfInputNamedLHS().replaceAllUsesOfWith(lhsResult);
  node.getGradOfInputNamedRHS().replaceAllUsesOfWith(rhsResult);
}
void lowerSubGradNode(Function *F, SubGradNode &node) {
  /// The chain rule for subtraction:
  /// delta(LHS) = dF/dLHS * delta(OUT) = 1 * delta(OUT)
  /// delta(RHS) = dF/dRHS * delta(OUT) = -1 * delta(OUT)
  auto outG = node.getGradOfOriginalOutputNamedResult();
  auto zero = F->createSplat("zero", outG.getType(), 0);
  auto sub = F->createSub("sub.grad", zero, outG);
  node.getGradOfInputNamedLHS().replaceAllUsesOfWith(outG);
  node.getGradOfInputNamedRHS().replaceAllUsesOfWith(sub);
}
void lowerDivGradNode(Function *F, DivGradNode &node) {
  /// The chain rule for division:
  /// delta(LHS) = dF/dLHS * delta(OUT) = (1 / RHS) * delta(OUT)
  /// delta(RHS) = dF/dRHS * delta(OUT) = (-LHS / (RHS ^ 2)) * delta(OUT)
  auto outG = node.getGradOfOriginalOutputNamedResult();
  NodeValue LHS = node.getLHS();
  NodeValue RHS = node.getRHS();

  auto lhsResult = F->createDiv("div.grad.rhs", outG, RHS);

  auto zero = F->createSplat("zero", outG.getType(), 0);
  auto subGrad = F->createSub("sub.grad", zero, outG);
  auto mulLhsGrad = F->createMul("mul.sub.grad.lhs", subGrad, LHS);

  auto squareRhs = F->createMul("square.rhs", RHS, RHS);
  auto rhsResult = F->createDiv("div.grad", mulLhsGrad, squareRhs);

  node.getGradOfInputNamedLHS().replaceAllUsesOfWith(lhsResult);
  node.getGradOfInputNamedRHS().replaceAllUsesOfWith(rhsResult);
}

void lowerRegressionNode(RegressionNode &node) {
  auto outG = node.getInput();
  node.getResult().replaceAllUsesOfWith(outG);
}

void lowerRegressionGradNode(Function *F, RegressionGradNode &node) {
  auto outG = node.getInput();

  auto inputG = F->createSub("rgn.grad", node.getInput(), node.getExpected());
  auto expG = F->createSplat("exp.grad", node.getExpected().getType(), 0);

  node.getGradOfInputNamedInput().replaceAllUsesOfWith(inputG);
  node.getGradOfInputNamedExpected().replaceAllUsesOfWith(expG);
}

void lowerFullyConnectedNode(Function *F, FullyConnectedNode &FC) {
  auto xDim = flattenCdr(FC.getInput().getType()->dims());
  auto wDim = FC.getWeights().dims();
  auto *X =
      F->createReshape("fc.1X", FC.getInput(), {1, xDim.first, xDim.second});
  Node *W = F->createReshape("fc.1W", FC.getWeights(), {1, wDim[0], wDim[1]});

  TypeRef outTy = F->getParent()->uniqueTypeWithNewShape(
      FC.getResult()->getType(), {1, xDim.first, wDim[1]});
  auto *mul = F->createBatchedMatMul("fc.dot", outTy, X, W);

  auto *mulFlat = F->createReshape("fc.cast2", mul, {xDim.first, wDim[1]});
  auto add = F->createBatchedAdd("fc.add.bias", FC.getResult()->getType(),
                                 mulFlat, FC.getBias());
  FC.getResult().replaceAllUsesOfWith(add);
}

void lowerFullyConnectedGradNode(Function *F, FullyConnectedGradNode &FCG) {
  // Follow the lowering from here:
  // https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/layers.py#L53
  auto out = FCG.getGradOfOriginalOutputNamedResult();
  auto xDims = flattenCdr(FCG.getInput().dims());
  auto outDims = out.dims();
  auto fDims = FCG.getWeights().dims();

  // dx = dout * w.T
  auto dout = F->createReshape("fcg.outG", out, {1, outDims[0], outDims[1]});
  auto *w =
      F->createReshape("fcg.w", FCG.getWeights(), {1, fDims[0], fDims[1]});
  auto *wT = F->createTranspose("fcg.wT", w, {0, 2, 1});
  auto *dx2 = F->createBatchedMatMul("fcg.dot", dout, wT);
  auto *dx = F->createReshape("fcg.inG", dx2, FCG.getInput().getType()->dims());
  FCG.getGradOfInputNamedInput().replaceAllUsesOfWith(dx);

  // dw = xT * dout.
  Node *x2 =
      F->createReshape("fcg.x", FCG.getInput(), {1, xDims.first, xDims.second});
  auto *x2T = F->createTranspose("fcg.xT", x2, {0, 2, 1});
  auto *dw = F->createBatchedMatMul("fcg.dot", x2T, dout);
  Node *dw2 = F->createReshape("fcg.dw2", dw, fDims);
  FCG.getGradOfInputNamedWeights().replaceAllUsesOfWith(dw2);

  // db = reduce(dout).
  auto *db = F->createBatchedReduceAdd("fc.bias.reduce", out);
  FCG.getGradOfInputNamedBias().replaceAllUsesOfWith(db);
}

void lowerReluGradNode(Function *F, ReluGradNode &RG) {
  // ReluGrad: if the input value is greater than zero then let the gradient
  // pass.
  auto *zero = F->createSplat("zero", RG.getInput().getType(), 0.0);
  auto *cond =
      F->createCmpLTE("relugrad", RG.getOriginalOutputForResult(), zero);
  auto *res = F->createSelect("relugrad", cond, zero,
                              RG.getGradOfOriginalOutputNamedResult());
  RG.getGradOfInputNamedInput().replaceAllUsesOfWith(res);
}

void lowerTanhGradNode(Function *F, TanhGradNode &THG) {
  // Tanh grad is calculated as:
  // inG = (1 - outW * outW) * outG

  // (W * W)
  auto outW = THG.getOriginalOutputForResult();
  auto *sq = F->createMul("tanh.in2", outW, outW);

  auto *one = F->createSplat("tanh.one", THG.getInput().getType(), 1.0);
  // (1 - W * W)
  auto *oneSubsq = F->createSub("tanh.one.sq", one, sq);

  auto *grad = F->createMul("tanh.one.sq", oneSubsq,
                            THG.getGradOfOriginalOutputNamedResult());
  THG.getGradOfInputNamedInput().replaceAllUsesOfWith(grad);
}

void lowerSigmoidGradNode(Function *F, SigmoidGradNode &THG) {
  // Sigmoid grad is calculated as:
  // inG = outW * (1 - outW) * outG;

  auto outW = THG.getOriginalOutputForResult();
  auto *one = F->createSplat("one", THG.getInput().getType(), 1.0);

  // (1 - W)
  auto *onew = F->createSub("sig.1w", one, outW);

  // (1 - W) * W
  auto *expr1 = F->createMul("sig.1ww", onew, outW);

  auto *grad = F->createMul("sigg.one.sq", expr1,
                            THG.getGradOfOriginalOutputNamedResult());
  THG.getGradOfInputNamedInput().replaceAllUsesOfWith(grad);
}

void lowerReluNode(Function *F, ReluNode &R) {
  // Relu is a max between zero and the input value.
  SplatNode *zero = F->createSplat("zero", R.getType(), 0.0);
  auto *relu = F->createMax("relu", zero, R.getInput());
  R.getResult().replaceAllUsesOfWith(relu);
}

void lowerSGDNode(Function *F, SGDNode &SGD) {
  assert(SGD.getUsers().size() == 0 && "SGDNode must not have users");

  NodeValue W = SGD.getWeight();
  NodeValue G = SGD.getGradient();
  NodeValue Gsum = SGD.getGsum();

  /// Described in the paper: Alex Krizhevsky [2014]
  // "One weird trick for parallelizing convolutional neural networks"

  float momentum = SGD.getMomentum();

  assert(W.dims() == G.dims() && "Invalid variables sizes for SGDNode");

  float L1Decay = SGD.getL1Decay();
  float L2Decay = SGD.getL2Decay();
  float learningRate = SGD.getLearningRate();
  float batchSize = SGD.getBatchSize();

  // All computations here are within the same type.
  auto type = G.getType();

  NodeValue gij = G;
  if (L1Decay) {
    auto *L1DecaySplat = F->createSplat("L1DecaySplat", type, L1Decay);
    auto *zeroSplat = F->createSplat("zeroSplat", type, 0);
    auto *oneSplat = F->createSplat("oneSplat", type, 1);
    auto *minusOneSplat = F->createSplat("minusOneSplat", type, -1);

    auto *Wcmp = F->createCmpLTE("Wcmp", zeroSplat, W);
    auto *Wdir = F->createSelect("Wdir", Wcmp, oneSplat, minusOneSplat);
    auto *L1Grad = F->createMul("L1Grad", L1DecaySplat, Wdir);

    gij = F->createAdd("gij_with_l1", gij, L1Grad);
  }
  if (L2Decay) {
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
    auto *momentumSplat = F->createSplat("learningRateSplat", type, momentum);
    auto *GsumMult = F->createMul("GsumMult", momentumSplat, Gsum);

    dx = F->createAdd("dx_with_momentum", GsumMult, dx);
    F->createSave("saveGsum", dx, llvm::cast<Variable>(Gsum.getNode()));
  }

  auto *newW = F->createAdd("newW", W, dx);
  F->createSave("saveW", newW, llvm::cast<Variable>(W.getNode()));
}

void lowerBatchNormalizationNodeForInference(Function *F,
                                             BatchNormalizationNode &BN) {
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

  auto epsilonSplat = F->createSplat("epsSplat", var.getType(), epsilon);
  Node *coef = F->createAdd("var_plus_eps", var, epsilonSplat);
  coef = F->createPow("sqrt_var_plus_eps", coef, 0.5);
  coef = F->createDiv("inverse_sqrt_var_plus_eps", gamma, coef);

  // Apply: out := (in - mean) * coef + beta
  // in and out are of the same size, while others must be broadcasted.
  auto meanB = F->createBroadcast("muBroadcasted", mean, in.dims(), channelIdx);
  auto coefB =
      F->createBroadcast("coefBroadcasted", coef, in.dims(), channelIdx);
  auto betaB =
      F->createBroadcast("betaBroadcasted", beta, in.dims(), channelIdx);

  Node *newResult = F->createSub("in_minus_mean", in, meanB);
  newResult = F->createMul("mul_coef", newResult, coefB);
  newResult = F->createAdd("result", newResult, betaB);

  BN.getResult().replaceAllUsesOfWith(newResult);
}

void computeBatchNormalizationWeights(Function *F, BatchNormalizationNode &BN) {
  auto in = BN.getInput();

  auto mean = BN.getMean();
  auto var = BN.getVar();

  auto channelIdx = BN.getChannelIdx();
  auto momentum = BN.getMomentum();

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
    std::vector<unsigned> perm(in.dims().size());
    for (size_t i = 0; i < perm.size(); i++)
      perm[i] = i;
    std::swap(perm[channelIdx], perm[perm.size() - 1]);
    inPrep = F->createTranspose("in.transpose", in, perm);
  }
  // Reshape input tensor to form:
  // {samplesPerChannel, numChannels}
  Node *inFlat =
      F->createReshape("in.flat", inPrep, {samplesPerChannel, numChannels});

  // Calculate Mean:

  // sum(in[i])
  // reduce the tensor by the first dimension, to get {numChannels}
  Node *localMean = F->createBatchedReduceAdd("in.sum", inFlat);
  // Mean = sum(in[i]) / N
  auto samplesPerChannelSplat = F->createSplat(
      "samplesPerChannelSplat", localMean->getType(), samplesPerChannel);
  localMean = F->createDiv("localMean", localMean, samplesPerChannelSplat);

  // Calculate Variance:

  // sum((x - mu) ^ 2)
  auto localMeanB =
      F->createBroadcast("new_mean_broadcasted", localMean, inFlat->dims(), 1);

  Node *localVar = F->createSub("x_mu", inFlat, localMeanB);
  localVar = F->createPow("x_mu2", localVar, 2);
  localVar = F->createBatchedReduceAdd("x_mu2.sum", localVar);
  // Var = sum((x - mu) ^ 2) / N
  localVar = F->createDiv("localVar", localVar, samplesPerChannelSplat);

  // Update the global variance and mean:
  auto momentumSplat =
      F->createSplat("momentumSplat", localMean->getType(), momentum);
  auto oneMinusMomentumSplat = F->createSplat(
      "oneMinusMomentumSplat", localMean->getType(), 1 - momentum);

  // newMean := P * localMean + (1 - P) * oldMean
  auto newMean = F->createAdd(
      "newMean",
      F->createMul("momentum_by_localMean", momentumSplat, localMean),
      F->createMul("1_momentum_by_oldMean", oneMinusMomentumSplat, mean));
  // newVar := P * localVar + (1 - P) * oldVar
  auto newVar = F->createAdd(
      "newVar", F->createMul("momentum_by_localVar", momentumSplat, localVar),
      F->createMul("1_momentum_by_oldVar", oneMinusMomentumSplat, var));

  // TODO: don't rely on operands' indices
  assert(BN.getInputName(3) == "Mean");
  assert(BN.getInputName(4) == "Var");
  BN.getNthInput(3).setOperand(newMean, 0);
  BN.getNthInput(4).setOperand(newVar, 0);
  // TODO: also consider updating corresponding BatchNormalizationGradNode

  F->createSave("saveMean", newMean, llvm::cast<Variable>(mean.getNode()));
  F->createSave("saveVar", newVar, llvm::cast<Variable>(var.getNode()));
}

void glow::lower(Function *F, CompilationMode mode) {
  auto &nodes = F->getNodes();

  for (auto const &node : nodes) {
    if (auto *RN = dyn_cast<RegressionNode>(node)) {
      lowerRegressionNode(*RN);
    } else if (auto *RGN = dyn_cast<RegressionGradNode>(node)) {
      lowerRegressionGradNode(F, *RGN);
    } else if (auto *EMG = dyn_cast<AddGradNode>(node)) {
      lowerAddGradNode(F, *EMG);
    } else if (auto *EMG = dyn_cast<MulGradNode>(node)) {
      lowerMulGradNode(F, *EMG);
    } else if (auto *EMG = dyn_cast<SubGradNode>(node)) {
      lowerSubGradNode(F, *EMG);
    } else if (auto *EMG = dyn_cast<DivGradNode>(node)) {
      lowerDivGradNode(F, *EMG);
    } else if (auto *FC = dyn_cast<FullyConnectedNode>(node)) {
      lowerFullyConnectedNode(F, *FC);
    } else if (auto *FCG = dyn_cast<FullyConnectedGradNode>(node)) {
      lowerFullyConnectedGradNode(F, *FCG);
    } else if (auto *RG = dyn_cast<ReluGradNode>(node)) {
      lowerReluGradNode(F, *RG);
    } else if (auto *R = dyn_cast<ReluNode>(node)) {
      lowerReluNode(F, *R);
    } else if (auto *THG = dyn_cast<TanhGradNode>(node)) {
      lowerTanhGradNode(F, *THG);
    } else if (auto *SG = dyn_cast<SigmoidGradNode>(node)) {
      lowerSigmoidGradNode(F, *SG);
    } else if (auto *SGD = dyn_cast<SGDNode>(node)) {
      lowerSGDNode(F, *SGD);
    } else if (auto *BN = dyn_cast<BatchNormalizationNode>(node)) {
      if (mode == CompilationMode::Train)
        computeBatchNormalizationWeights(F, *BN);
      lowerBatchNormalizationNodeForInference(F, *BN);
    }
  }

  for (auto it = F->getNodes().begin(), e = F->getNodes().end(); it != e;) {
    auto cur = *(it++);
    if (dyn_cast<SGDNode>(cur))
      F->eraseNode(cur);
  }
}
