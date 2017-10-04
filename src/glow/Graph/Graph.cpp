// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include <fstream>
#include <iostream>

using namespace glow;

Graph::~Graph() {
  // Delete all of the nodes and the variables.
  for (auto *N : nodes_) {
    delete N;
  }
  for (auto *V : vars_) {
    delete V;
  }
}

//===----------------------------------------------------------------------===//
//                       Node builders
//===----------------------------------------------------------------------===//

Variable *Graph::createVariable(TypeRef T, llvm::StringRef name,
                                WeightVar::InitKind initKind, float val) {
  return addVar(new Variable(name, T, initKind, val));
}

Variable *Graph::createVariable(ElemKind T, llvm::ArrayRef<size_t> dims,
                                llvm::StringRef name,
                                WeightVar::InitKind initKind, float val) {
  auto FT = M_.uniqueType(T, dims);
  return createVariable(FT, name, initKind, val);
}

ConvolutionNode *Graph::createConv(Node *input, size_t depth, size_t kernel,
                                   size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  // Calculate the size and allocate the output buffer.
  auto outSz =
      ConvolutionInst::calculateOutputDims(idim.h, idim.w, pad, kernel, stride);

  std::vector<size_t> outDims = {idim.n, outSz.first, outSz.second, depth};

  // Allocate the Filter and Bias tensors.
  std::vector<size_t> filterDim = {depth, kernel, kernel, idim.c};
  size_t fanIn = kernel * kernel * idim.c;
  auto *filter = createVariable(ElemKind::FloatTy, filterDim, "filter",
                                WeightVar::InitKind::Xavier, fanIn);

  auto *bias = createVariable(ElemKind::FloatTy, {depth}, "filter",
                              WeightVar::InitKind::Broadcast, 0.1);

  auto OT = M_.uniqueType(ElemKind::FloatTy, outDims);

  return addNode(new ConvolutionNode(input, OT, "Conv", filter, bias, kernel,
                                     stride, pad, depth));
}

PoolNode *Graph::createPool(Node *input, PoolInst::OpKind kind, size_t kernel,
                            size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz =
      ConvolutionInst::calculateOutputDims(idim.h, idim.w, pad, kernel, stride);

  auto OT = M_.uniqueType(ElemKind::FloatTy,
                          {idim.n, outSz.first, outSz.second, idim.c});

  return addNode(new PoolNode(input, OT, "pool", kind, kernel, stride, pad));
}

FullyConnectedNode *Graph::createFullyConnected(Node *input, size_t outDepth) {
  TypeRef T = input->getType();
  auto idim = flattenCdr(input->dims());

  size_t fanIn = idim.second;

  auto *W = createVariable(T->getElementType(), {outDepth, idim.second},
                           "weights", WeightVar::InitKind::Xavier, fanIn);

  auto *B = createVariable(T->getElementType(), {outDepth}, "bias",
                           WeightVar::InitKind::Xavier, 0.1);

  auto OT = M_.uniqueType(T->getElementType(), {idim.first, outDepth});
  return addNode(
      new FullyConnectedNode(input, OT, "Fullyconnected", W, B, outDepth));
}

ReluNode *Graph::createRELU(Node *input) {
  return addNode(new ReluNode(input, "relu"));
}

SigmoidNode *Graph::createSigmoid(Node *input) {
  return addNode(new SigmoidNode(input, "Sigmoid"));
}

TanhNode *Graph::createTanh(Node *input) {
  return addNode(new TanhNode(input, "Tanh"));
}

SoftMaxNode *Graph::createSoftMax(Node *input, Node *selected) {
  return addNode(new SoftMaxNode(input, "SoftMax", selected));
}

RegressionNode *Graph::createRegression(Node *input, Node *expected) {
  return addNode(new RegressionNode(input, "Regression", expected));
}

ReshapeNode *Graph::createReshape(Node *input, llvm::ArrayRef<size_t> shape) {
  return addNode(new ReshapeNode(input, "Reshape", shape));
}

TransposeNode *Graph::createTranspose(Node *input,
                                      llvm::ArrayRef<unsigned> shuffle) {
  std::vector<size_t> shape;
  auto dims = input->dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto NT = M_.uniqueType(input->getElementType(), shape);
  return addNode(new TransposeNode(input, NT, "Transpose", shuffle));
}

ConcatNode *Graph::createConcat(llvm::ArrayRef<Node *> inputs,
                                unsigned dimension) {
  auto inDim = inputs[0]->dims();

  for (auto in : inputs) {
    (void)in;
    assert(in->dims() == inDim && "Invalid input shape");
  }

  std::vector<size_t> shape(inDim.begin(), inDim.end());
  // We are stacking the tensors along a specific dimension. This means that we
  // increase the size of the tensor along this dimension.
  shape[dimension] *= inputs.size();

  auto NT = M_.uniqueType(inputs[0]->getElementType(), shape);
  return addNode(new ConcatNode(inputs, NT, "Concat", dimension));
}

BatchNormalizationNode *Graph::createBatchNormalization(Node *input,
                                                        size_t channelIdx,
                                                        float epsilon,
                                                        float momentum) {
  // Figure out how many channels are in the tensor.
  size_t channels = input->dims()[channelIdx];

  // Allocate the learnable parameters beta and gamma.
  auto *beta = createVariable(ElemKind::FloatTy, {channels}, "beta",
                              WeightVar::InitKind::Broadcast, 0.);
  auto *gamma = createVariable(ElemKind::FloatTy, {channels}, "gamma",
                               WeightVar::InitKind::Broadcast, 1.0);

  auto *mean = createVariable(ElemKind::FloatTy, {channels}, "mean",
                              WeightVar::InitKind::Broadcast, 0.0);
  auto *variance = createVariable(ElemKind::FloatTy, {channels}, "variance",
                                  WeightVar::InitKind::Broadcast, 0.0);

  return addNode(new BatchNormalizationNode(input, "Norm", gamma, beta, mean,
                                            variance, channelIdx, epsilon,
                                            momentum));
}

LocalResponseNormalizationNode *
Graph::createLocalResponseNormalization(Node *input, size_t halfWindowSize,
                                        float alpha, float beta, float k) {
  auto Ty = input->getType();
  auto *scale =
      createVariable(Ty, "scale", WeightVar::InitKind::Broadcast, 0.0);

  // The output tensor is of the same shape as the input tensor.
  return addNode(new LocalResponseNormalizationNode(
      input, "LRN", scale, halfWindowSize, alpha, beta, k));
}

ArithmeticNode *Graph::createArithmetic(Node *LHS, Node *RHS,
                                        ArithmeticInst::OpKind op) {
  assert(LHS->dims() == RHS->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  return addNode(new ArithmeticNode("Arithmetic", LHS, RHS, op));
}

//===----------------------------------------------------------------------===//
//                       Graph Utilities
//===----------------------------------------------------------------------===//

void Graph::dump() {
  std::cout << "Graph structure:\n";
  for (auto v : vars_) {
    std::cout << v->getDebugDesc() << "\n";
  }

  for (auto n : nodes_) {
    std::cout << n->getDebugDesc() << "\n";
  }
}

/// A helper class for visiting and generating the dotty file from the graph.
struct DottyPrinterPass : NodeVisitor {
  using edgeTy = std::pair<Node *, Node *>;
  std::vector<edgeTy> nodeEdges{};

public:
  // Don't revisit visited nodes.
  bool shouldVisit(Node *parent, Node *N) override {
    edgeTy e = {parent, N};
    return std::find(nodeEdges.begin(), nodeEdges.end(), e) == nodeEdges.end();
  }

  DottyPrinterPass() = default;

  void pre(Node *parent, Node *N) override { nodeEdges.push_back({parent, N}); }

  std::string nodeDescr(Node *N) {
    if (!N) {
      return "";
    }
    // Print a node descriptor that looks like this:
    // Format: "node12" [ label = "0xf7fc43e01" shape = "record" ];
    std::string sb;
    sb += quote(std::to_string((void *)N)) + "[\n";
    std::string repr = escapeDottyString(N->getDebugDesc());
    sb += "\tlabel = " + repr + "\n";
    sb += "\tshape = \"record\"\n";
    sb += "];\n\n";
    return sb;
  }

  std::string quote(std::string in) { return '"' + in + '"'; }
  std::string getDottyString() {
    std::string sb;

    sb += "digraph finite_state_machine {\n\trankdir=TD;\n";

    // Assign a unique name to each one of the nodes:
    for (auto &e : nodeEdges) {
      if (e.first) {
        sb += quote(std::to_string(e.second)) + " -> " +
              quote(std::to_string(e.first)) + ";\n";
      }
    }

    // Assign a unique name to each one of the nodes:
    for (auto &e : nodeEdges) {
      sb += nodeDescr(e.first);
      sb += nodeDescr(e.second);
    }

    sb += "}";
    return sb;
  }
};

void Graph::dumpDAG() {
  DottyPrinterPass DP;

  for (auto &N : nodes_) {
    N->visit(nullptr, &DP);
  }

  std::string filename = "dotty_graph_dump_" + std::to_string(this) + ".dot";
  std::cout << "Writing dotty graph to: " << filename << '\n';

  std::string rep = DP.getDottyString();

  std::ofstream myfile;
  myfile.open(filename);
  myfile << rep;
  myfile.close();
}
