// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

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

TypeRef Graph::uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims) {
  return uniqueType(Type(elemTy, dims));
}

TypeRef Graph::uniqueType(const Type &T) {
  for (auto &tp : types_) {
    if (T.isEqual(tp)) {
      return &tp;
    }
  }

  return &*types_.insert(types_.begin(), T);
}

TypeRef Graph::getVoidTy() { return uniqueType(Type()); }

//===----------------------------------------------------------------------===//
//                       Node builders
//===----------------------------------------------------------------------===//

Variable *Graph::createVariable(TypeRef T, llvm::StringRef name,
                                Variable::InitKind initKind, float val) {
  return addVar(new Variable(name, T, initKind, val));
}

Variable *Graph::createVariable(ElemKind T, llvm::ArrayRef<size_t> dims,
                                llvm::StringRef name,
                                Variable::InitKind initKind, float val) {
  auto FT = uniqueType(T, dims);
  return createVariable(FT, name, initKind, val);
}

ConvolutionNode *Graph::createConv(llvm::StringRef name, Node *input,
                                   size_t depth, size_t kernel, size_t stride,
                                   size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

  std::array<size_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};

  // Allocate the Filter and Bias tensors.
  std::array<size_t, 4> filterDim = {{depth, kernel, kernel, idim.c}};
  size_t fanIn = kernel * kernel * idim.c;
  auto *filter = createVariable(ElemKind::FloatTy, filterDim, "filter",
                                Variable::InitKind::Xavier, fanIn);

  auto *bias = createVariable(ElemKind::FloatTy, {depth}, "bias",
                              Variable::InitKind::Broadcast, 0.1);

  auto OT = uniqueType(ElemKind::FloatTy, outDims);

  return addNode(new ConvolutionNode(name, OT, input, filter, bias, kernel,
                                     stride, pad, depth));
}

PoolNode *Graph::createPool(llvm::StringRef name, Node *input,
                            PoolNode::Mode mode, size_t kernel, size_t stride,
                            size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input->dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

  auto OT = uniqueType(ElemKind::FloatTy,
                       {idim.n, outSz.first, outSz.second, idim.c});

  return addNode(new PoolNode(name, OT, mode, input, kernel, stride, pad));
}

FullyConnectedNode *Graph::createFullyConnected(llvm::StringRef name,
                                                Node *input, size_t outDepth) {
  TypeRef T = input->getType();
  auto idim = flattenCdr(input->dims());

  size_t fanIn = idim.second;

  auto *W = createVariable(T->getElementType(), {outDepth, idim.second},
                           "weights", Variable::InitKind::Xavier, fanIn);

  auto *B = createVariable(T->getElementType(), {outDepth}, "bias",
                           Variable::InitKind::Broadcast, 0.1);

  auto OT = uniqueType(T->getElementType(), {idim.first, outDepth});
  return addNode(new FullyConnectedNode(name, OT, input, W, B, outDepth));
}

ReluNode *Graph::createRELU(llvm::StringRef name, Node *input) {
  return addNode(new ReluNode(name, input));
}

SigmoidNode *Graph::createSigmoid(llvm::StringRef name, Node *input) {
  return addNode(new SigmoidNode(name, input));
}

TanhNode *Graph::createTanh(llvm::StringRef name, Node *input) {
  return addNode(new TanhNode(name, input));
}

SoftMaxNode *Graph::createSoftMax(llvm::StringRef name, Node *input,
                                  Node *selected) {
  return addNode(new SoftMaxNode(name, input, selected));
}

RegressionNode *Graph::createRegression(llvm::StringRef name, Node *input,
                                        Node *expected) {
  return addNode(new RegressionNode(name, input, expected));
}

ReshapeNode *Graph::createReshape(llvm::StringRef name, Node *input,
                                  llvm::ArrayRef<size_t> shape) {
  auto TR = uniqueType(input->getType()->getElementType(), shape);

  return addNode(new ReshapeNode(name, TR, input, shape.vec()));
}

TransposeNode *Graph::createTranspose(llvm::StringRef name, Node *input,
                                      llvm::ArrayRef<unsigned> shuffle) {
  llvm::SmallVector<size_t, 6> shape;
  auto dims = input->dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto NT = uniqueType(input->getElementType(), shape);
  return addNode(new TransposeNode(name, NT, input, shuffle.vec()));
}

/// \returns true if \p T1 and T2 has the exact same type except for dimension
/// \p dim.
static bool sameSameShapeExceptDim(TypeRef T1, TypeRef T2, unsigned dim) {
  if (T1->getElementType() != T2->getElementType()) {
    return false;
  }

  auto D1 = T1->dims();
  auto D2 = T2->dims();

  if (D1.size() != D2.size()) {
    return false;
  }

  for (int i = 0, e = D1.size(); i < e; i++) {
    // Ignore the dimension \p dim.
    if (i == dim) {
      continue;
    }

    if (D1[i] != D2[i]) {
      return false;
    }
  }

  return true;
}

ConcatNode *Graph::createConcat(llvm::StringRef name,
                                llvm::ArrayRef<Node *> inputs,
                                unsigned dimension) {
  for (int i = 0, e = inputs.size(); i < e; i++) {
    assert(sameSameShapeExceptDim(inputs[i]->getType(), inputs[0]->getType(),
                                  dimension) &&
           "Invalid type");
    (void)sameSameShapeExceptDim;
  }
  auto inDim = inputs[0]->dims();

  llvm::SmallVector<size_t, 6> shape(inDim.begin(), inDim.end());

  // We are stacking the tensors along a specific dimension. This means that we
  // increase the size of the tensor along this dimension.
  shape[dimension] = 0;
  for (auto I : inputs) {
    shape[dimension] += I->getType()->dims()[dimension];
  }

  auto NT = uniqueType(inputs[0]->getElementType(), shape);
  std::vector<NodeValue> ops;
  ops.reserve(inputs.size());
  for (auto &I : inputs) {
    ops.emplace_back(I);
  }
  return addNode(new ConcatNode(name, NT, ops, dimension));
}

SliceNode *Graph::createSlice(llvm::StringRef name, Node *input,
                              llvm::ArrayRef<size_t> begin,
                              llvm::ArrayRef<size_t> end) {

  std::vector<size_t> begin_v, shape;
  auto dims = input->dims();
  assert(begin.size() == end.size() && "Begin and End dimensions should match");
  assert(begin.size() == dims.size() &&
         "Begin and Input dimensions should match");
  for (int i = 0; i < dims.size(); i++) {
    size_t begin_i = begin[i];
    size_t end_i = end[i];
    size_t dim_i = dims[i];(void) dim_i;
    (void) dim_i;
    assert(begin_i >= 0 && "Illegal Begin  indices");
    assert(end_i > 0 && "Illegal End indices");
    assert(begin_i < dim_i && "Illegal Begin  indices");
    assert(end_i <= dim_i && "Illegal End indices");
    assert(end_i > begin_i && "Illegal Begin and End indices");
    begin_v.push_back(begin_i);
    shape.push_back(end_i - begin_i);
  }
  auto NT = uniqueType(input->getType()->getElementType(), shape);

  return addNode(new SliceNode(name, NT, input, begin_v));
}

BatchNormalizationNode *Graph::createBatchNormalization(llvm::StringRef name,
                                                        Node *input,
                                                        size_t channelIdx,
                                                        float epsilon,
                                                        float momentum) {
  // Figure out how many channels are in the tensor.
  size_t channels = input->dims()[channelIdx];

  // Allocate the learnable parameters beta and gamma.
  auto *beta = createVariable(ElemKind::FloatTy, {channels}, "beta",
                              Variable::InitKind::Broadcast, 0.);
  auto *gamma = createVariable(ElemKind::FloatTy, {channels}, "gamma",
                               Variable::InitKind::Broadcast, 1.0);

  auto *mean = createVariable(ElemKind::FloatTy, {channels}, "mean",
                              Variable::InitKind::Broadcast, 0.0);
  auto *variance = createVariable(ElemKind::FloatTy, {channels}, "variance",
                                  Variable::InitKind::Broadcast, 0.0);

  return createBatchNormalization(name, input, beta, gamma, mean, variance,
                                  channelIdx, epsilon, momentum);
}

BatchNormalizationNode *Graph::createBatchNormalization(
    llvm::StringRef name, Node *input, Node *beta, Node *gamma, Node *mean,
    Node *var, size_t channelIdx, float epsilon, float momentum) {
  return addNode(new BatchNormalizationNode(name, input, gamma, beta, mean, var,
                                            channelIdx, epsilon, momentum));
}

LocalResponseNormalizationNode *
Graph::createLocalResponseNormalization(llvm::StringRef name, Node *input,
                                        size_t halfWindowSize, float alpha,
                                        float beta, float k) {
  auto Ty = input->getType();
  auto *scale = createVariable(Ty, "scale", Variable::InitKind::Broadcast, 0.0);

  // The output tensor is of the same shape as the input tensor.
  return addNode(new LocalResponseNormalizationNode(
      name, input, scale, halfWindowSize, alpha, beta, k));
}

ArithmeticNode *Graph::createArithmetic(llvm::StringRef name, Node *LHS,
                                        Node *RHS, ArithmeticNode::Mode op) {
  assert(LHS->dims() == RHS->dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  return addNode(new ArithmeticNode(name, op, LHS, RHS));
}

SaveNode *Graph::createSave(llvm::StringRef name, Node *input) {
  auto *dest = createVariable(input->getType(), "saved",
                              Variable::InitKind::Broadcast, 0);

  return addNode(new SaveNode(name, input, dest));
}

SaveNode *Graph::createSave(llvm::StringRef name, Node *input,
                            Variable *output) {
  return addNode(new SaveNode(name, input, output));
}

//===----------------------------------------------------------------------===//
//                   Graph dumping and printing
//===----------------------------------------------------------------------===//

void Graph::dump() {
  llvm::outs() << "Graph structure:\n";
  for (auto v : vars_) {
    llvm::outs() << v->getDebugDesc() << "\n";
  }

  for (auto n : nodes_) {
    llvm::outs() << n->getDebugDesc() << "\n";
  }
}

/// A helper class for visiting and generating the dotty file from the graph.
struct DottyPrinterPass : NodeVisitor {
  using edgeTy = std::pair<Node *, Node *>;
  // A set of ordered visited edges.
  std::vector<edgeTy> nodeEdges{};
  // The output stream for writing the dotty descriptor.
  std::ostream &os_;

public:
  // Don't revisit visited nodes.
  bool shouldVisit(Node *parent, Node *N) override {
    edgeTy e = {parent, N};
    return std::find(nodeEdges.begin(), nodeEdges.end(), e) == nodeEdges.end();
  }

  explicit DottyPrinterPass(std::ostream &os) : os_(os) {}

  void pre(Node *parent, Node *N) override {
    nodeEdges.emplace_back(parent, N);
  }

  void dumpDesc(Node *N) {
    if (!N) {
      return;
    }
    // Print a node descriptor that looks like this:
    // Format: "node12" [ label = "0xf7fc43e01" shape = "record" ];
    os_ << quote(std::to_string((void *)N)) << "[\n";
    std::string repr = escapeDottyString(N->getDebugDesc());
    os_ << "\tlabel = " + quote(repr) + "\n";
    os_ << "\tshape = \"record\"\n";
    if (llvm::isa<Variable>(N)) {
      os_ << "\tfillcolor=pink,style=filled\n";
    }
    os_ << "];\n\n";
  }

  std::string quote(std::string in) { return '"' + in + '"'; }
  void dump() {
    os_ << "digraph finite_state_machine {\n\trankdir=TD;\n";

    // Assign a unique name to each one of the nodes:
    for (auto &e : nodeEdges) {
      if (e.first) {
        os_ << quote(std::to_string(e.second)) << " -> "
            << quote(std::to_string(e.first)) << ";\n";
      }
    }

    // Assign a unique name to each one of the nodes:
    for (auto &e : nodeEdges) {
      dumpDesc(e.first);
      dumpDesc(e.second);
    }

    os_ << "}";
  }
};

void Graph::dumpDAG(const char *dotFilename) {
  std::string filename =
      !dotFilename ? "dotty_graph_dump_" + std::to_string(this) + ".dot"
                   : dotFilename;
  llvm::outs() << "Writing dotty graph to: " << filename << '\n';

  std::ofstream myfile;
  myfile.open(filename);

  DottyPrinterPass DP(myfile);

  for (auto &N : nodes_) {
    N->visit(nullptr, &DP);
  }

  DP.dump();
  myfile.close();
}
