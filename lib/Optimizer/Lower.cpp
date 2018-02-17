// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::dyn_cast;

void lowerArithmeticNode(Function &graph, ArithmeticGradNode &node) {
  switch (node.getMode()) {
  case ArithmeticGradNode::Mode::Add: {
    /// The chain rule for addition:
    /// LHS' = OUT'
    /// RHS' = OUT'
    auto outG = node.getGradOfOriginalOutputNamedResult();
    node.getGradOfInputNamedLHS().replaceAllUsesOfWith(outG);
    node.getGradOfInputNamedRHS().replaceAllUsesOfWith(outG);
    break;
  }

  case ArithmeticGradNode::Mode::Mul: {
    /// The chain rule for multiplication:
    /// LHS' = RHS * OUT'
    /// RHS' = LHS * OUT'
    auto outG = node.getGradOfOriginalOutputNamedResult();
    NodeValue LHS = node.getLHS();
    NodeValue RHS = node.getRHS();

    auto lhsResult = graph.createArithmetic("mul.grad.rhs", outG, RHS,
                                            ArithmeticNode::Mode::Mul);
    auto rhsResult = graph.createArithmetic("mul.grad.lhs", outG, LHS,
                                            ArithmeticNode::Mode::Mul);
    node.getGradOfInputNamedLHS().replaceAllUsesOfWith(lhsResult);
    node.getGradOfInputNamedRHS().replaceAllUsesOfWith(rhsResult);
    break;
  }

  case ArithmeticGradNode::Mode::Sub: {
    /// The chain rule for subtraction:
    /// LHS' = OUT'
    /// RHS' = -OUT'
    auto outG = node.getGradOfOriginalOutputNamedResult();
    auto zero = graph.createSplat("zero", outG.getType(), 0);
    auto sub = graph.createArithmetic("sub.grad", zero, outG,
                                      ArithmeticNode::Mode::Sub);
    node.getGradOfInputNamedLHS().replaceAllUsesOfWith(outG);
    node.getGradOfInputNamedRHS().replaceAllUsesOfWith(sub);
    break;
  }

  case ArithmeticGradNode::Mode::Div: {
    /// The chain rule for division:
    /// LHS' = OUT' / RHS
    /// RHS' = - LHS * OUT' / (RHS * RHS)
    auto outG = node.getGradOfOriginalOutputNamedResult();
    NodeValue LHS = node.getLHS();
    NodeValue RHS = node.getRHS();

    auto lhsResult = graph.createArithmetic("div.grad.rhs", outG, RHS,
                                            ArithmeticNode::Mode::Div);

    auto zero = graph.createSplat("zero", outG.getType(), 0);
    auto subGrad = graph.createArithmetic("sub.grad", zero, outG,
                                          ArithmeticNode::Mode::Sub);
    auto mulLhsGrad = graph.createArithmetic("mul.sub.grad.lhs", subGrad, LHS,
                                             ArithmeticNode::Mode::Mul);

    auto squareRhs = graph.createArithmetic("square.rhs", RHS, RHS,
                                            ArithmeticNode::Mode::Mul);
    auto rhsResult = graph.createArithmetic("div.grad", mulLhsGrad, squareRhs,
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

void lowerRegressionGradNode(Function &graph, RegressionGradNode &node) {
  auto outG = node.getInput();

  auto inputG =
      graph.createArithmetic("rgn.grad", node.getInput(), node.getExpected(),
                             ArithmeticNode::Mode::Sub);
  auto expG = graph.createSplat("exp.grad", node.getExpected().getType(), 0);

  node.getGradOfInputNamedInput().replaceAllUsesOfWith(inputG);
  node.getGradOfInputNamedExpected().replaceAllUsesOfWith(expG);
}

void lowerFullyConnectedNode(Function &graph, FullyConnectedNode &FC) {
  auto xDim = flattenCdr(FC.getInput().getType()->dims());
  auto wDim = FC.getWeights().dims();
  auto *X =
      graph.createReshape("fc.1X", FC.getInput(), {1, xDim.first, xDim.second});
  Node *W =
      graph.createReshape("fc.1W", FC.getWeights(), {1, wDim[0], wDim[1]});

  auto elemTy = W->getType()->getElementType();

  TypeRef outTy = nullptr;
  if (W->getType()->isQuantizedType()) {
    // We use the scale and offset from the output of the FC for both the matrix
    // multiplication node and the batched-add node.
    auto FCT = FC.getResult()->getType();
    outTy = graph.getParent().uniqueType(elemTy, {1, xDim.first, wDim[1]},
                                         FCT->getScale(), FCT->getOffset());
  } else {
    outTy = graph.getParent().uniqueType(elemTy, {1, xDim.first, wDim[1]});
  }
  auto *mul = graph.createBatchedMatMul("fc.dot", outTy, X, W);

  auto *mulFlat = graph.createReshape("fc.cast2", mul, {xDim.first, wDim[1]});
  auto add = graph.createBatchedArithmetic(
      "fc.add.bias", FC.getResult()->getType(),
      BatchedArithmeticNode::Mode::Add, mulFlat, FC.getBias());
  FC.getResult().replaceAllUsesOfWith(add);
}

void lowerFullyConnectedGradNode(Function &graph, FullyConnectedGradNode &FCG) {
  // Follow the lowering from here:
  // https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/layers.py#L53
  auto out = FCG.getGradOfOriginalOutputNamedResult();
  auto xDims = flattenCdr(FCG.getInput().dims());
  auto outDims = out.dims();
  auto fDims = FCG.getWeights().dims();

  // dx = dout * w.T
  auto dout = graph.createReshape("fcg.outG", out, {1, outDims[0], outDims[1]});
  auto *w =
      graph.createReshape("fcg.w", FCG.getWeights(), {1, fDims[0], fDims[1]});
  auto *wT = graph.createTranspose("fcg.wT", w, {0, 2, 1});
  auto *dx2 = graph.createBatchedMatMul("fcg.dot", dout, wT);
  auto *dx =
      graph.createReshape("fcg.inG", dx2, FCG.getInput().getType()->dims());
  FCG.getGradOfInputNamedInput().replaceAllUsesOfWith(dx);

  // dw = xT * dout.
  Node *x2 = graph.createReshape("fcg.x", FCG.getInput(),
                                 {1, xDims.first, xDims.second});
  auto *x2T = graph.createTranspose("fcg.xT", x2, {0, 2, 1});
  auto *dw = graph.createBatchedMatMul("fcg.dot", x2T, dout);
  Node *dw2 = graph.createReshape("fcg.dw2", dw, fDims);
  FCG.getGradOfInputNamedWeights().replaceAllUsesOfWith(dw2);

  // db = reduce(dout).
  auto *db = graph.createBatchedReduce("fc.bias.reduce",
                                       BatchedReduceNode::Mode::Add, out);
  FCG.getGradOfInputNamedBias().replaceAllUsesOfWith(db);
}

void lowerReluGradNode(Function &graph, ReluGradNode &RG) {
  // ReluGrad: if the input value is greater than zero then let the gradient
  // pass.
  auto *zero = graph.createSplat("zero", RG.getInput().getType(), 0.0);
  auto *cond =
      graph.createArithmetic("relugrad", RG.getOriginalOutputForResult(), zero,
                             ArithmeticNode::Mode::CmpLTE);
  auto *res = graph.createSelect("relugrad", cond, zero,
                                 RG.getGradOfOriginalOutputNamedResult());
  RG.getGradOfInputNamedInput().replaceAllUsesOfWith(res);
}

void lowerTanhGradNode(Function &graph, TanhGradNode &THG) {
  // Tanh grad is calculated as:
  // inG = (1 - outW * outW) * outG

  // (W * W)
  auto outW = THG.getOriginalOutputForResult();
  auto *sq =
      graph.createArithmetic("tanh.in2", outW, outW, ArithmeticNode::Mode::Mul);

  auto *one = graph.createSplat("tanh.one", THG.getInput().getType(), 1.0);
  // (1 - W * W)
  auto *oneSubsq =
      graph.createArithmetic("tanh.one.sq", one, sq, ArithmeticNode::Mode::Sub);

  auto *grad = graph.createArithmetic("tanh.one.sq", oneSubsq,
                                      THG.getGradOfOriginalOutputNamedResult(),
                                      ArithmeticNode::Mode::Mul);
  THG.getGradOfInputNamedInput().replaceAllUsesOfWith(grad);
}

void lowerSigmoidGradNode(Function &graph, SigmoidGradNode &THG) {
  // Sigmoid grad is calculated as:
  // inG = outW * (1 - outW) * outG;

  auto outW = THG.getOriginalOutputForResult();
  auto *one = graph.createSplat("one", THG.getInput().getType(), 1.0);

  // (1 - W)
  auto *onew =
      graph.createArithmetic("sig.1w", one, outW, ArithmeticNode::Mode::Sub);

  // (1 - W) * W
  auto *expr1 =
      graph.createArithmetic("sig.1ww", onew, outW, ArithmeticNode::Mode::Mul);

  auto *grad = graph.createArithmetic("sigg.one.sq", expr1,
                                      THG.getGradOfOriginalOutputNamedResult(),
                                      ArithmeticNode::Mode::Mul);
  THG.getGradOfInputNamedInput().replaceAllUsesOfWith(grad);
}

void lowerReluNode(Function &graph, ReluNode &R) {
  // Relu is a max between zero and the input value.
  SplatNode *zero;
  if (R.getType()->isQuantizedType()) {
    // Offset represents zero for quantized type.
    // floating value = scale * (X - Offset). X must be equal to offset in order
    // to get 0f.
    zero = graph.createSplat("zero", R.getType(), R.getType()->getOffset());
  } else {
    zero = graph.createSplat("zero", R.getType(), 0.0);
  }

  auto *relu = graph.createArithmetic("relu", zero, R.getInput(),
                                      ArithmeticNode::Mode::Max);
  R.getResult().replaceAllUsesOfWith(relu);
}

void glow::lower(Function &G, CompilationMode mode) {
  auto &nodes = G.getNodes();

  for (auto const &node : nodes) {
    if (auto *RN = dyn_cast<RegressionNode>(node)) {
      lowerRegressionNode(*RN);
    } else if (auto *RGN = dyn_cast<RegressionGradNode>(node)) {
      lowerRegressionGradNode(G, *RGN);
    } else if (auto *EMG = dyn_cast<ArithmeticGradNode>(node)) {
      lowerArithmeticNode(G, *EMG);
    } else if (auto *FC = dyn_cast<FullyConnectedNode>(node)) {
      lowerFullyConnectedNode(G, *FC);
    } else if (auto *FCG = dyn_cast<FullyConnectedGradNode>(node)) {
      lowerFullyConnectedGradNode(G, *FCG);
    } else if (auto *RG = dyn_cast<ReluGradNode>(node)) {
      lowerReluGradNode(G, *RG);
    } else if (auto *R = dyn_cast<ReluNode>(node)) {
      lowerReluNode(G, *R);
    } else if (auto *THG = dyn_cast<TanhGradNode>(node)) {
      lowerTanhGradNode(G, *THG);
    } else if (auto *SG = dyn_cast<SigmoidGradNode>(node)) {
      lowerSigmoidGradNode(G, *SG);
    }
  }
}
