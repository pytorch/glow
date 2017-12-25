// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::dyn_cast;

void lowerArithmeticNode(Graph &graph, ArithmeticGradNode &node) {
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
  }
}

void lowerRegressionNode(RegressionNode &node) {
  auto outG = node.getInput();
  node.getResult().replaceAllUsesOfWith(outG);
}

void lowerRegressionGradNode(Graph &graph, RegressionGradNode &node) {
  auto outG = node.getInput();

  auto inputG =
      graph.createArithmetic("rgn.grad", node.getInput(), node.getExpected(),
                             ArithmeticNode::Mode::Sub);
  auto expG = graph.createSplat("exp.grad", node.getExpected().getType(), 0);

  node.getGradOfInputNamedInput().replaceAllUsesOfWith(inputG);
  node.getGradOfInputNamedExpected().replaceAllUsesOfWith(expG);
}

void lowerFullyConnectedNode(Graph &graph, FullyConnectedNode &FC) {
  TypeRef T = FC.getInput().getType();
  auto idim = flattenCdr(T->dims());

  auto *batch = graph.createReshape("fc.cast", FC.getInput(),
                                    {idim.first, idim.second, 1});

  auto *mul = graph.createBatchedMatMul("fc.dot", batch, FC.getFilter());

  auto *mulFlat =
      graph.createReshape("fc.cast2", mul, {idim.first, FC.getDepth()});

  auto add = graph.createBatchedArithmetic(
      "fc.add.bias", BatchedArithmeticNode::Mode::Add, mulFlat, FC.getBias());

  FC.getOutput().replaceAllUsesOfWith(add);
}

void glow::lower(Graph &G, CompilationMode mode) {
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
    }
  }

}
