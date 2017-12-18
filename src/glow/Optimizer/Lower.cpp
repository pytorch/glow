// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;

void glow::lower(Graph &G, CompilationMode mode) {
  auto &nodes = G.getNodes();

  // For each node:
  for (auto const &node : nodes) {
    // Lower the RegressionNode node:
    if (auto *RN = dyn_cast<RegressionNode>(node)) {
      auto outG = RN->getInput();
      RN->getResult()->replaceAllUsesOfWith(outG);
      continue;
    }
    // Lower the RegressionGradNode node:
    if (auto *RGN = dyn_cast<RegressionGradNode>(node)) {
      auto outG = RGN->getInput();

      auto inputG =
          G.createArithmetic("rgn.grad", RGN->getInput(), RGN->getExpected(),
                             ArithmeticNode::Mode::Sub);
      auto expG = G.createZero("exp.grad", RGN->getExpected().getType());

      RGN->getGradOfInputNamedInput()->replaceAllUsesOfWith(inputG);
      RGN->getGradOfInputNamedExpected()->replaceAllUsesOfWith(expG);
      continue;
    }

    // Lower the ArithmeticGradNode node:
    if (auto *EMG = dyn_cast<ArithmeticGradNode>(node)) {
      switch (EMG->getMode()) {
      case ArithmeticGradNode::Mode::Add: {
        /// The chain rule for Addition:
        /// LHS' = OUT'
        /// RHS' = OUT'
        auto outG = EMG->getGradOfOriginalOutputNamedResult();
        EMG->getGradOfInputNamedLHS()->replaceAllUsesOfWith(outG);
        EMG->getGradOfInputNamedRHS()->replaceAllUsesOfWith(outG);
        continue;
      }

      case ArithmeticGradNode::Mode::Mul: {
        auto outG = EMG->getGradOfOriginalOutputNamedResult();
        NodeValue LHS = EMG->getLHS();
        NodeValue RHS = EMG->getRHS();
        /// The chain rule for multiplication:
        /// LHS' = RHS' * OUT'
        /// RHS' = LHS' * OUT'
        auto LHS1 = G.createArithmetic("mul.grad.lhs", RHS, outG,
                                       ArithmeticNode::Mode::Mul);
        auto RHS1 = G.createArithmetic("mul.grad.rhs", LHS, outG,
                                       ArithmeticNode::Mode::Mul);
        EMG->getGradOfInputNamedLHS()->replaceAllUsesOfWith(LHS1);
        EMG->getGradOfInputNamedRHS()->replaceAllUsesOfWith(RHS1);
        continue;
      }

      case ArithmeticGradNode::Mode::Sub: {
        /// The chain rule for subtraction:
        /// LHS' =  OUT'
        /// RHS' =  -OUT'
        auto outG = EMG->getGradOfOriginalOutputNamedResult();
        EMG->getGradOfInputNamedLHS()->replaceAllUsesOfWith(outG);
        auto zero = G.createZero("zero", outG.getType());
        auto sub = G.createArithmetic("sub.grad", zero, outG,
                                      ArithmeticNode::Mode::Sub);
        EMG->getGradOfInputNamedRHS()->replaceAllUsesOfWith(sub);
        continue;
      }
      }
    } // Arithmetic Grad.
  }   // For all nodes.
}
