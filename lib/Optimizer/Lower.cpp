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

#include "glow/Backends/Backend.h"
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
  auto input = node.getInput();
  node.getResult().replaceAllUsesOfWith(input);
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
  auto *X = F->createReshape("fc.1X", FC.getInput(), {xDim.first, xDim.second});

  TypeRef outTy = F->getParent()->uniqueTypeWithNewShape(
      FC.getResult()->getType(), {xDim.first, wDim[1]});
  auto *mul = F->createMatMul("fc.dot", outTy, X, FC.getWeights());

  auto add = F->createBatchedAdd("fc.add.bias", FC.getResult()->getType(), mul,
                                 FC.getBias());
  FC.getResult().replaceAllUsesOfWith(add);

  if (FC.hasPredicate()) {
    add->setPredicate(FC.getPredicate());
    mul->setPredicate(FC.getPredicate());
  }
}

void lowerFullyConnectedGradNode(Function *F, FullyConnectedGradNode &FCG) {
  // Follow the lowering from here:
  // https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/layers.py#L53
  auto dout = FCG.getGradOfOriginalOutputNamedResult();
  auto xDims = flattenCdr(FCG.getInput().dims());

  // dx = dout * w.T
  auto *wT = F->createTranspose("fcg.wT", FCG.getWeights(), {1, 0});
  auto *dx2 = F->createMatMul("fcg.dot", dout, wT);
  auto *dx = F->createReshape("fcg.inG", dx2, FCG.getInput().getType()->dims());
  FCG.getGradOfInputNamedInput().replaceAllUsesOfWith(dx);

  // dw = xT * dout.
  Node *x2 =
      F->createReshape("fcg.x", FCG.getInput(), {xDims.first, xDims.second});
  auto *x2T = F->createTranspose("fcg.xT", x2, {1, 0});
  auto *dw = F->createMatMul("fcg.dot", x2T, dout);
  FCG.getGradOfInputNamedWeights().replaceAllUsesOfWith(dw);

  // db = reduce(dout).
  auto *db = F->createBatchedReduceAdd("fc.bias.reduce", dout);
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

  // Cannot use mean.replaceAllUsesOfWith(newMean) here, because newMean depends
  // on mean. Essentially, we need to replace x with f(x). It means that such
  // replacement would also find `f(x)` itself in the graph, and create a cycle.
  for (auto N : F->getNodes())
    if (llvm::isa<BatchNormalizationNode>(N) ||
        llvm::isa<BatchNormalizationGradNode>(N))
      for (size_t i = 0; i < N->getNumInputs(); i++) {
        if (N->getNthInput(i) == mean) {
          N->getNthInput(i) = newMean;
        }
        if (N->getNthInput(i) == var) {
          N->getNthInput(i) = newVar;
        }
      }

  F->createSave("saveMean", newMean, llvm::cast<Variable>(mean.getNode()));
  F->createSave("saveVar", newVar, llvm::cast<Variable>(var.getNode()));
}

void lowerBatchNormalizationGradNode(Function *F,
                                     BatchNormalizationGradNode &BNG) {
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
  auto meanB =
      F->createBroadcast("mean_broadcasted", mean, inW.dims(), channelIdx);
  auto hmu = F->createSub("x_minus_mean", inW, meanB);
  NodeValue sumDyhmu = F->createMul("dy_mul_h_minus_mu", outG, hmu);

  // Calculate: sum(dy)
  NodeValue sumDy = outG;

  // TODO: consider adding this functionality to the main operator set.
  if (channelIdx + 1 != inW.dims().size()) {
    std::vector<unsigned> perm(inW.dims().size());
    for (size_t i = 0; i < perm.size(); i++)
      perm[i] = i;
    std::swap(perm[channelIdx], perm[perm.size() - 1]);

    sumDyhmu = F->createTranspose("sumDyhmu.transpose", sumDyhmu, perm);
    sumDy = F->createTranspose("sumDy.transpose", sumDy, perm);
  }
  sumDyhmu = F->createReshape("sumDyhmu.flat", sumDyhmu,
                              {samplesPerChannel, numChannels});
  sumDy =
      F->createReshape("sumDy.flat", sumDy, {samplesPerChannel, numChannels});
  sumDyhmu = F->createBatchedReduceAdd("sumDyhmu.reduced", sumDyhmu);
  sumDy = F->createBatchedReduceAdd("sumDy.reduced", sumDy);

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

  auto epsilonSplat = F->createSplat("epsSplat", var.getType(), epsilon);
  auto oneSplat = F->createSplat("oneSplat", var.getType(), 1.0);
  auto invNSplat =
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
  auto sumDyB =
      F->createBroadcast("sumDy_broadcasted", sumDy, inW.dims(), channelIdx);
  auto NSplat = F->createSplat("oneSplat", inW.getType(), samplesPerChannel);
  Node *inBrackets = F->createMul("NSplat_outG", NSplat, outG);
  inBrackets = F->createSub("inBrackets",
                            F->createSub("inBrackets_2ops", inBrackets, sumDyB),
                            F->createMul("hmu_coef2", hmu, coef2));

  auto inG = F->createMul("inG", coef1, inBrackets);
  BNG.getGradOfInputNamedInput().replaceAllUsesOfWith(inG);

  BNG.getGradOfInputNamedBias().replaceAllUsesOfWith(sumDy);

  auto gammaG = F->createMul("gammaG", sumDyhmu, invVarSqrt);
  BNG.getGradOfInputNamedScale().replaceAllUsesOfWith(gammaG);

  auto zeroSplat = F->createSplat("zeroSplat", var.getType(), 0);
  BNG.getGradOfInputNamedMean().replaceAllUsesOfWith(zeroSplat);
  BNG.getGradOfInputNamedVar().replaceAllUsesOfWith(zeroSplat);
}

void lowerGroupConvolutionNode(Function *F, ConvolutionNode &BNG) {
  // When Group parameter is more than 1, ConvolutionNode can be represented as
  // a Concatenation of smaller dimension Convolutions. Input channels will be
  // divided into equal groups of consecutive channels. These will be separately
  // convolved each with its own filter (and bias), and then concatenated.
  // This will result in 4 * Group + 1 nodes.
  unsigned kernel = BNG.getKernel();
  unsigned pad = BNG.getPad();
  unsigned stride = BNG.getStride();
  unsigned group = BNG.getGroup();
  auto in = BNG.getInput();
  auto filter = BNG.getFilter();
  auto bias = BNG.getBias();

  ShapeNHWC idim = ShapeNHWC(in.dims());

  unsigned inCperG = idim.c / group;
  unsigned outCperG = filter.dims()[0] / group;

  auto outDims = BNG.getResult().dims().vec();
  outDims[3] = outCperG;
  auto outTy = F->getParent()->uniqueTypeWithNewShape(
      BNG.getResult()->getType(), outDims);

  std::vector<NodeValue> convs;
  for (unsigned groupId = 0; groupId < group; groupId++) {
    auto *in_slice =
        F->createSlice(BNG.getName(), in, {0, 0, 0, groupId * inCperG},
                       {idim.n, idim.h, idim.w, (groupId + 1) * inCperG});
    auto *filter_slice =
        F->createSlice(BNG.getName(), filter, {groupId * outCperG, 0, 0, 0},
                       {(groupId + 1) * outCperG, kernel, kernel, inCperG});
    auto *bias_slice = F->createSlice(BNG.getName(), bias, {groupId * outCperG},
                                      {(groupId + 1) * outCperG});
    convs.push_back(F->createConv(BNG.getName(), in_slice, filter_slice,
                                  bias_slice, outTy, kernel, stride, pad, 1));
  }
  auto result = F->createConcat(BNG.getName(), convs, 3);
  BNG.getResult().replaceAllUsesOfWith(result);
}

void glow::lower(Function *F, CompilationMode mode, const Backend &B) {
  auto &nodes = F->getNodes();

  for (auto const &node : nodes) {
    if (!B.shouldLower(node)) {
      continue;
    }
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
    } else if (auto *BNG = dyn_cast<BatchNormalizationGradNode>(node)) {
      lowerBatchNormalizationGradNode(F, *BNG);
    } else if (auto *CN = dyn_cast<ConvolutionNode>(node)) {
      if (CN->getGroup() > 1)
        lowerGroupConvolutionNode(F, *CN);
    }
  }

  for (auto it = F->getNodes().begin(), e = F->getNodes().end(); it != e;) {
    auto cur = *(it++);
    if (dyn_cast<SGDNode>(cur))
      F->eraseNode(cur);
  }
}
