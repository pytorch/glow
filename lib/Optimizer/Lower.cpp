// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::dyn_cast;

void lowerArithmeticNode(Function *F, ArithmeticGradNode &node) {
  switch (node.getMode()) {
  case ArithmeticGradNode::Mode::Add: {
    /// The chain rule for addition:
    /// delta(LHS) = dF/dLHS * delta(OUT) = 1 * delta(OUT)
    /// delta(RHS) = dF/dRHS * delta(OUT) = 1 * delta(OUT)
    auto outG = node.getGradOfOriginalOutputNamedResult();
    node.getGradOfInputNamedLHS().replaceAllUsesOfWith(outG);
    node.getGradOfInputNamedRHS().replaceAllUsesOfWith(outG);
    break;
  }

  case ArithmeticGradNode::Mode::Mul: {
    /// The chain rule for multiplication:
    /// delta(LHS) = dF/dLHS * delta(OUT) = RHS * delta(OUT)
    /// delta(RHS) = dF/dRHS * delta(OUT) = LHS * delta(OUT)
    auto outG = node.getGradOfOriginalOutputNamedResult();
    NodeValue LHS = node.getLHS();
    NodeValue RHS = node.getRHS();

    auto lhsResult = F->createArithmetic("mul.grad.rhs", outG, RHS,
                                         ArithmeticNode::Mode::Mul);
    auto rhsResult = F->createArithmetic("mul.grad.lhs", outG, LHS,
                                         ArithmeticNode::Mode::Mul);
    node.getGradOfInputNamedLHS().replaceAllUsesOfWith(lhsResult);
    node.getGradOfInputNamedRHS().replaceAllUsesOfWith(rhsResult);
    break;
  }

  case ArithmeticGradNode::Mode::Sub: {
    /// The chain rule for subtraction:
    /// delta(LHS) = dF/dLHS * delta(OUT) = 1 * delta(OUT)
    /// delta(RHS) = dF/dRHS * delta(OUT) = -1 * delta(OUT)
    auto outG = node.getGradOfOriginalOutputNamedResult();
    auto zero = F->createSplat("zero", outG.getType(), 0);
    auto sub =
        F->createArithmetic("sub.grad", zero, outG, ArithmeticNode::Mode::Sub);
    node.getGradOfInputNamedLHS().replaceAllUsesOfWith(outG);
    node.getGradOfInputNamedRHS().replaceAllUsesOfWith(sub);
    break;
  }

  case ArithmeticGradNode::Mode::Div: {
    /// The chain rule for division:
    /// delta(LHS) = dF/dLHS * delta(OUT) = (1 / RHS) * delta(OUT)
    /// delta(RHS) = dF/dRHS * delta(OUT) = (-LHS / (RHS ^ 2)) * delta(OUT)
    auto outG = node.getGradOfOriginalOutputNamedResult();
    NodeValue LHS = node.getLHS();
    NodeValue RHS = node.getRHS();

    auto lhsResult = F->createArithmetic("div.grad.rhs", outG, RHS,
                                         ArithmeticNode::Mode::Div);

    auto zero = F->createSplat("zero", outG.getType(), 0);
    auto subGrad =
        F->createArithmetic("sub.grad", zero, outG, ArithmeticNode::Mode::Sub);
    auto mulLhsGrad = F->createArithmetic("mul.sub.grad.lhs", subGrad, LHS,
                                          ArithmeticNode::Mode::Mul);

    auto squareRhs =
        F->createArithmetic("square.rhs", RHS, RHS, ArithmeticNode::Mode::Mul);
    auto rhsResult = F->createArithmetic("div.grad", mulLhsGrad, squareRhs,
                                         ArithmeticNode::Mode::Div);

    node.getGradOfInputNamedLHS().replaceAllUsesOfWith(lhsResult);
    node.getGradOfInputNamedRHS().replaceAllUsesOfWith(rhsResult);
    break;
  }
  case ArithmeticGradNode::Mode::CmpLTE: {
    llvm_unreachable("Unable to differentiate the CmpLT function");
  }
  case ArithmeticGradNode::Mode::Max: {
    llvm_unreachable("Unable to differentiate the Max function");
  }
  case ArithmeticGradNode::Mode::Min: {
    llvm_unreachable("Unable to differentiate the Min function");
  }
  }
}

void lowerRegressionNode(RegressionNode &node) {
  auto outG = node.getInput();
  node.getResult().replaceAllUsesOfWith(outG);
}

void lowerRegressionGradNode(Function *F, RegressionGradNode &node) {
  auto outG = node.getInput();

  auto inputG =
      F->createArithmetic("rgn.grad", node.getInput(), node.getExpected(),
                          ArithmeticNode::Mode::Sub);
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
  auto *db = F->createBatchedReduce("fc.bias.reduce",
                                    BatchedReduceNode::Mode::Add, out);
  FCG.getGradOfInputNamedBias().replaceAllUsesOfWith(db);
}

void lowerReluGradNode(Function *F, ReluGradNode &RG) {
  // ReluGrad: if the input value is greater than zero then let the gradient
  // pass.
  auto *zero = F->createSplat("zero", RG.getInput().getType(), 0.0);
  auto *cond = F->createArithmetic("relugrad", RG.getOriginalOutputForResult(),
                                   zero, ArithmeticNode::Mode::CmpLTE);
  auto *res = F->createSelect("relugrad", cond, zero,
                              RG.getGradOfOriginalOutputNamedResult());
  RG.getGradOfInputNamedInput().replaceAllUsesOfWith(res);
}

void lowerTanhGradNode(Function *F, TanhGradNode &THG) {
  // Tanh grad is calculated as:
  // inG = (1 - outW * outW) * outG

  // (W * W)
  auto outW = THG.getOriginalOutputForResult();
  auto *sq =
      F->createArithmetic("tanh.in2", outW, outW, ArithmeticNode::Mode::Mul);

  auto *one = F->createSplat("tanh.one", THG.getInput().getType(), 1.0);
  // (1 - W * W)
  auto *oneSubsq =
      F->createArithmetic("tanh.one.sq", one, sq, ArithmeticNode::Mode::Sub);

  auto *grad = F->createArithmetic("tanh.one.sq", oneSubsq,
                                   THG.getGradOfOriginalOutputNamedResult(),
                                   ArithmeticNode::Mode::Mul);
  THG.getGradOfInputNamedInput().replaceAllUsesOfWith(grad);
}

void lowerSigmoidGradNode(Function *F, SigmoidGradNode &THG) {
  // Sigmoid grad is calculated as:
  // inG = outW * (1 - outW) * outG;

  auto outW = THG.getOriginalOutputForResult();
  auto *one = F->createSplat("one", THG.getInput().getType(), 1.0);

  // (1 - W)
  auto *onew =
      F->createArithmetic("sig.1w", one, outW, ArithmeticNode::Mode::Sub);

  // (1 - W) * W
  auto *expr1 =
      F->createArithmetic("sig.1ww", onew, outW, ArithmeticNode::Mode::Mul);

  auto *grad = F->createArithmetic("sigg.one.sq", expr1,
                                   THG.getGradOfOriginalOutputNamedResult(),
                                   ArithmeticNode::Mode::Mul);
  THG.getGradOfInputNamedInput().replaceAllUsesOfWith(grad);
}

void lowerReluNode(Function *F, ReluNode &R) {
  // Relu is a max between zero and the input value.
  SplatNode *zero = F->createSplat("zero", R.getType(), 0.0);
  auto *relu = F->createArithmetic("relu", zero, R.getInput(),
                                   ArithmeticNode::Mode::Max);
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
    auto L1DecaySplat = F->createSplat("L1DecaySplat", type, L1Decay);
    auto zeroSplat = F->createSplat("zeroSplat", type, 0);
    auto oneSplat = F->createSplat("oneSplat", type, 1);
    auto minusOneSplat = F->createSplat("minusOneSplat", type, -1);

    auto Wcmp =
        F->createArithmetic("Wcmp", zeroSplat, W, ArithmeticNode::Mode::CmpLTE);
    auto Wdir = F->createSelect("Wdir", Wcmp, oneSplat, minusOneSplat);
    auto L1Grad = F->createArithmetic("L1Grad", L1DecaySplat, Wdir,
                                      ArithmeticNode::Mode::Mul);

    gij = F->createArithmetic("gij_with_l1", gij, L1Grad,
                              ArithmeticNode::Mode::Add);
  }
  if (L2Decay) {
    auto L2DecaySplat = F->createSplat("L2DecaySplat", type, L2Decay);

    auto L2Grad = F->createArithmetic("L2Grad", L2DecaySplat, W,
                                      ArithmeticNode::Mode::Mul);

    gij = F->createArithmetic("gij_with_l2", gij, L2Grad,
                              ArithmeticNode::Mode::Add);
  }
  if (batchSize > 1) {
    auto batchSizeSplat = F->createSplat("batchSizeSplat", type, batchSize);
    gij = F->createArithmetic("gij_div_batchSz", gij, batchSizeSplat,
                              ArithmeticNode::Mode::Div);
  }

  auto negLearningRateSplat =
      F->createSplat("learningRateSplat", type, -learningRate);
  auto dx = F->createArithmetic("dx", negLearningRateSplat, gij,
                                ArithmeticNode::Mode::Mul);

  // Use the momentum to improve the gradient descent:
  // http://ufldl.stanford.edu/tutorial/supervised/
  // OptimizationStochasticGradientDescent/
  if (momentum > 0.0) {
    auto momentumSplat = F->createSplat("learningRateSplat", type, momentum);
    auto GsumMult = F->createArithmetic("GsumMult", momentumSplat, Gsum,
                                        ArithmeticNode::Mode::Mul);

    dx = F->createArithmetic("dx_with_momentum", GsumMult, dx,
                             ArithmeticNode::Mode::Add);
    F->createSave("saveGsum", dx, llvm::cast<Variable>(Gsum.getNode()));
  }

  auto newW = F->createArithmetic("newW", W, dx, ArithmeticNode::Mode::Add);
  F->createSave("saveW", newW, llvm::cast<Variable>(W.getNode()));
}

void glow::lower(Function *F, CompilationMode mode) {
  auto &nodes = F->getNodes();

  for (auto const &node : nodes) {
    if (auto *RN = dyn_cast<RegressionNode>(node)) {
      lowerRegressionNode(*RN);
    } else if (auto *RGN = dyn_cast<RegressionGradNode>(node)) {
      lowerRegressionGradNode(F, *RGN);
    } else if (auto *EMG = dyn_cast<ArithmeticGradNode>(node)) {
      lowerArithmeticNode(F, *EMG);
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
    }
  }

  for (auto it = F->getNodes().begin(), e = F->getNodes().end(); it != e;) {
    auto cur = *(it++);
    if (dyn_cast<SGDNode>(cur))
      F->eraseNode(cur);
  }
}
