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
#include <unordered_set>

using namespace glow;
using llvm::dyn_cast;

Graph::~Graph() {
  // Delete all of the nodes and the variables.
  for (auto *N : nodes_) {
    eraseNode(N);
  }
  for (auto *V : vars_) {
    eraseNode(V);
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

/// Form a unique name based on the original non-uniqued \p Name.
///
/// This is done by taking the original non-uniqued name
/// (i.e. the part of the name before the first occurrence of "__")
/// and concatenating it with "__N", where N is a unique numeric
/// suffix.
///
/// The "__" suffix is used as a delimeter and therefore it should
/// not be used by names of user-defined variables.
///
/// If the compiler needs to auto-generate some node names, it should
/// never add any suffix anywhere after "__", because it will get
/// stripped by uniqueName. Instead, all such auto-generated pieces of
/// a name should be added somewhere before "__", e.g. as a prefix.
std::string Graph::uniqueName(llvm::StringRef name) {
  // First, remove everything starting with the __ delimiter.
  auto delimPos = name.find("__", 0);
  if (delimPos != llvm::StringRef::npos) {
    name = name.substr(0, delimPos);
  }
  std::string UniqueName{name};
  UniqueName += "__";
  UniqueName += std::to_string(uniqueIdx_);
  uniqueIdx_++;
  return UniqueName;
}

void Graph::uniqueNames(Node *N) { N->setName(uniqueName(N->getName())); }

void Graph::addGradientVariable(Variable *V, Variable *GradV) {
  grads_.push_back({V, GradV});
}

Variable *Graph::getGradientVariable(Variable *V) {
  for (auto &p : grads_) {
    if (p.first == V) {
      return p.second;
    }
  }
  return nullptr;
}

ConvolutionNode *Graph::createConv(llvm::StringRef name, NodeValue input,
                                   size_t depth, size_t kernel, size_t stride,
                                   size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
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

PoolNode *Graph::createPool(llvm::StringRef name, NodeValue input,
                            PoolNode::Mode mode, size_t kernel, size_t stride,
                            size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, pad, kernel, stride);

  auto OT = uniqueType(ElemKind::FloatTy,
                       {idim.n, outSz.first, outSz.second, idim.c});

  return addNode(new PoolNode(name, OT, mode, input, kernel, stride, pad));
}

FullyConnectedNode *Graph::createFullyConnected(llvm::StringRef name,
                                                NodeValue input, Variable *W,
                                                Variable *B, size_t outDepth) {
  TypeRef T = input.getType();
  auto idim = flattenCdr(input.dims());

  auto OT = uniqueType(T->getElementType(), {idim.first, outDepth});
  return addNode(new FullyConnectedNode(name, OT, input, W, B, outDepth));
}

FullyConnectedNode *Graph::createFullyConnected(llvm::StringRef name,
                                                NodeValue input,
                                                size_t outDepth) {
  TypeRef T = input.getType();
  auto idim = flattenCdr(input.dims());

  size_t fanIn = idim.second;

  auto *W = createVariable(T->getElementType(), {outDepth, idim.second},
                           "weights", Variable::InitKind::Xavier, fanIn);

  auto *B = createVariable(T->getElementType(), {outDepth}, "bias",
                           Variable::InitKind::Broadcast, 0.1);

  auto OT = uniqueType(T->getElementType(), {idim.first, outDepth});
  return addNode(new FullyConnectedNode(name, OT, input, W, B, outDepth));
}

ReluNode *Graph::createRELU(llvm::StringRef name, NodeValue input) {
  return addNode(new ReluNode(name, input));
}

SigmoidNode *Graph::createSigmoid(llvm::StringRef name, NodeValue input) {
  return addNode(new SigmoidNode(name, input));
}

TanhNode *Graph::createTanh(llvm::StringRef name, NodeValue input) {
  return addNode(new TanhNode(name, input));
}

SoftMaxNode *Graph::createSoftMax(llvm::StringRef name, NodeValue input,
                                  NodeValue selected) {
  return addNode(new SoftMaxNode(name, input, selected));
}

RegressionNode *Graph::createRegression(llvm::StringRef name, NodeValue input,
                                        NodeValue expected) {
  return addNode(new RegressionNode(name, input, expected));
}

ReshapeNode *Graph::createReshape(llvm::StringRef name, NodeValue input,
                                  llvm::ArrayRef<size_t> shape) {
  auto TR = uniqueType(input.getType()->getElementType(), shape);

  return addNode(new ReshapeNode(name, TR, input, shape.vec()));
}

TransposeNode *Graph::createTranspose(llvm::StringRef name, NodeValue input,
                                      llvm::ArrayRef<unsigned> shuffle) {
  llvm::SmallVector<size_t, 6> shape;
  auto dims = input.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto NT = uniqueType(input.getElementType(), shape);
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

  for (unsigned i = 0, e = D1.size(); i < e; i++) {
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

SliceNode *Graph::createSlice(llvm::StringRef name, NodeValue input,
                              llvm::ArrayRef<size_t> begin,
                              llvm::ArrayRef<size_t> end) {

  std::vector<size_t> begin_v, shape;
  auto dims = input.dims();
  assert(begin.size() == end.size() && "Begin and End dimensions should match");
  assert(begin.size() == dims.size() &&
         "Begin and Input dimensions should match");
  for (unsigned i = 0; i < dims.size(); i++) {
    size_t begin_i = begin[i];
    size_t end_i = end[i];
    size_t dim_i = dims[i];
    (void)dim_i;
    assert(begin_i >= 0 && "Illegal Begin  indices");
    assert(end_i > 0 && "Illegal End indices");
    assert(begin_i < dim_i && "Illegal Begin  indices");
    assert(end_i <= dim_i && "Illegal End indices");
    assert(end_i > begin_i && "Illegal Begin and End indices");
    begin_v.push_back(begin_i);
    shape.push_back(end_i - begin_i);
  }
  auto NT = uniqueType(input.getType()->getElementType(), shape);

  return addNode(new SliceNode(name, NT, input, begin_v));
}

BatchNormalizationNode *Graph::createBatchNormalization(llvm::StringRef name,
                                                        NodeValue input,
                                                        size_t channelIdx,
                                                        float epsilon,
                                                        float momentum) {
  // Figure out how many channels are in the tensor.
  size_t channels = input.dims()[channelIdx];

  // Allocate the learnable parameters beta and gamma.
  auto *beta = createVariable(ElemKind::FloatTy, {channels}, "beta",
                              Variable::InitKind::Broadcast, 0.);
  auto *gamma = createVariable(ElemKind::FloatTy, {channels}, "gamma",
                               Variable::InitKind::Broadcast, 1.0);

  auto *mean = createVariable(ElemKind::FloatTy, {channels}, "mean",
                              Variable::InitKind::Extern);
  auto *variance = createVariable(ElemKind::FloatTy, {channels}, "variance",
                                  Variable::InitKind::Extern);

  return createBatchNormalization(name, input, beta, gamma, mean, variance,
                                  channelIdx, epsilon, momentum);
}

BatchNormalizationNode *
Graph::createBatchNormalization(llvm::StringRef name, NodeValue input,
                                NodeValue beta, NodeValue gamma, NodeValue mean,
                                NodeValue var, size_t channelIdx, float epsilon,
                                float momentum) {
  return addNode(new BatchNormalizationNode(name, input, gamma, beta, mean, var,
                                            channelIdx, epsilon, momentum));
}

LocalResponseNormalizationNode *
Graph::createLocalResponseNormalization(llvm::StringRef name, NodeValue input,
                                        size_t halfWindowSize, float alpha,
                                        float beta, float k) {
  // The output tensor is of the same shape as the input tensor.
  return addNode(new LocalResponseNormalizationNode(name, input, halfWindowSize,
                                                    alpha, beta, k));
}

ArithmeticNode *Graph::createArithmetic(llvm::StringRef name, NodeValue LHS,
                                        NodeValue RHS,
                                        ArithmeticNode::Mode op) {
  assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");
  // The output tensor is of the same shape as the input tensor.
  return addNode(new ArithmeticNode(name, op, LHS, RHS));
}

SplatNode *Graph::createSplat(llvm::StringRef name, TypeRef ty, float value) {
  return addNode(new SplatNode(name, ty, value));
}

BatchedMatMulNode *Graph::createBatchedMatMul(llvm::StringRef name,
                                              NodeValue batch,
                                              NodeValue filter) {
  auto BT = batch.getType();
  auto FT = filter.getType();

  assert(BT->dims().size() == 3);
  assert(FT->dims().size() == 2);

  size_t a1 = BT->dims()[1];
  size_t a2 = BT->dims()[2];
  size_t b1 = FT->dims()[0];
  size_t b2 = FT->dims()[1];
  assert(a2 == b1 && "Column of A is not equal to the row of A.");
  (void)a1;
  (void)a2;
  (void)b1;
  (void)b2;
  auto RT = Type(BT->getElementType(), {BT->dims()[0], a1, b2});
  return addNode(new BatchedMatMulNode(name, uniqueType(RT), batch, filter));
}

BatchedReduceNode *Graph::createBatchedReduce(llvm::StringRef name,
                                              BatchedReduceNode::Mode mode,
                                              NodeValue batch) {
  auto BT = batch.getType();
  auto RT = Type(BT->getElementType(), BT->dims().drop_front());
  return addNode(new BatchedReduceNode(name, uniqueType(RT), mode, batch));
}

BatchedArithmeticNode *
Graph::createBatchedArithmetic(llvm::StringRef name,
                               BatchedArithmeticNode::Mode mode,
                               NodeValue batch, NodeValue sample) {
  return addNode(new BatchedArithmeticNode(name, mode, batch, sample));
}

SaveNode *Graph::createSave(llvm::StringRef name, NodeValue input) {
  auto *dest =
      createVariable(input.getType(), name, Variable::InitKind::Extern);

  std::string nodeName{"_save_"};
  nodeName += name;
  return addNode(new SaveNode(nodeName, input, dest));
}

SaveNode *Graph::createSave(llvm::StringRef name, NodeValue input,
                            Variable *output) {
  return addNode(new SaveNode(name, input, output));
}

//===----------------------------------------------------------------------===//
//                   Graph dumping and printing
//===----------------------------------------------------------------------===//

void Graph::dump() const {
  llvm::outs() << "Graph structure " << getName() << ":\n";
  for (auto v : vars_) {
    llvm::outs() << v->getDebugDesc() << "\n";
  }

  for (auto n : nodes_) {
    llvm::outs() << n->getDebugDesc() << "\n";
  }
}

/// A helper class for visiting and generating the dotty file from the graph.
/// We can't use NodeWalker here, because it ignores result indices, which
/// are critical in generating detailed debug output.
class DottyPrinterPass {
  // The output stream for writing the dotty descriptor.
  std::ostream &os_;
  // A set of already visited (during graph walk) nodes.
  std::unordered_set<Node *> visitedNodes_{};
  // List of generated edges.
  std::vector<std::string> nodeEdges_{};

  /// Dumps label for a input/output row, given port names.
  /// E.g. {"LHS", "RHS"} will produce {<LHS>LHS|<RHS>RHS}
  void dumpLabelForRow(llvm::ArrayRef<std::string> names) {
    os_ << "{";
    for (size_t i = 0; i < names.size(); i++) {
      if (i) {
        os_ << "|";
      }
      os_ << "<" << names[i] << ">" << names[i];
    }
    os_ << "}";
  }

  void dumpLabel(Node *N) {
    os_ << "{";
    if (N->getNumInputs()) {
      std::vector<std::string> names(N->getNumInputs());
      for (size_t i = 0; i < names.size(); i++) {
        names[i] = N->getInputName(i).str();
      }
      dumpLabelForRow(names);
      os_ << "|";
    }
    os_ << "{" << escapeDottyString(N->getDebugDesc()) << "}";
    if (N->getNumRes()) {
      os_ << "|";
      std::vector<std::string> names(N->getNumRes());
      for (size_t i = 0; i < names.size(); i++) {
        names[i] = N->getOutputName(i).str();
      }
      dumpLabelForRow(names);
    }
    os_ << "}";
  }

  void dumpNode(Node *N) {
    if (!N) {
      return;
    }
    // Print a node descriptor that looks like this:
    // "0xf7fc43e01" [ shape = "record" label = "{...}" ];
    // where 0xf7fc43e01 is address of node.
    os_ << uniqueNodeName(N) << "[\n";
    os_ << "\tlabel = \"";
    dumpLabel(N);
    os_ << "\"\n";
    os_ << "\tshape = \"record\"\n";
    if (llvm::isa<Variable>(N)) {
      os_ << "\tfillcolor=pink,style=filled\n";
    }
    os_ << "];\n\n";
  }

  std::string quote(std::string in) { return '"' + in + '"'; }
  std::string uniqueNodeName(Node *N) {
    return quote(std::to_string((void *)N));
  }

  /// Recursively traverses inputs of node \p N using Deep First Search.
  /// Each node will be visited no more than once. The method also dumps
  /// edges with their port identifiers in dotty format.
  void visitNode(Node *N) {
    if (visitedNodes_.find(N) != visitedNodes_.end())
      return;
    visitedNodes_.insert(N);

    for (size_t i = 0; i < N->getNumInputs(); i++) {
      Node *to = N->getInputNode(i).getNode();
      size_t resNo = N->getInputNode(i).getResNo();

      std::ostringstream edge;
      edge << uniqueNodeName(to) << ":" << to->getOutputName(resNo).str()
           << " -> " << uniqueNodeName(N) << ":" << N->getInputName(i).str();
      nodeEdges_.push_back(edge.str());

      visitNode(to);
    }
  }

public:
  explicit DottyPrinterPass(std::ostream &os) : os_(os) {}

  void visitGraph(Graph *G) {
    for (auto N : G->getNodes()) {
      visitNode(N);
    }
  }

  void dumpAll() {
    os_ << "digraph finite_state_machine {\n\trankdir=TB;\n";

    // Dump nodes:
    for (auto e : visitedNodes_) {
      dumpNode(e);
    }

    // Dump edges:
    for (auto &e : nodeEdges_) {
      os_ << e << ";\n";
    }

    os_ << "}";
  }
};

void Graph::dumpDAG() {
  std::string filename = "dotty_graph_dump_" + std::to_string(this) + ".dot";
  dumpDAG(filename.c_str());
}

void Graph::dumpDAG(const char *dotFilename) {
  std::string filename = dotFilename;
  llvm::outs() << "Writing dotty graph to: " << filename << '\n';

  std::ofstream myfile;
  myfile.open(filename);

  DottyPrinterPass DP(myfile);

  DP.visitGraph(this);

  DP.dumpAll();
  myfile.close();
}

void Graph::eraseVariable(VariablesList::iterator I) {
  if (I == vars_.end())
    return;
  delete *I;
  vars_.erase(I);
}

void Graph::eraseNode(NodesList::iterator I) {
  Node *N = *I;
  switch (N->getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind: {                                      \
    delete static_cast<CLASS *>(N);                                            \
    break;                                                                     \
  }
#include "AutoGenNodes.def"
  default:
    llvm_unreachable("Unhandled node");
  }

  nodes_.erase(I);
}

Variable *Graph::getVariableByName(llvm::StringRef name) {
  for (auto *V : getVars()) {
    if (V->getName() == name)
      return V;
  }
  return nullptr;
}

void Graph::eraseVariable(Variable *N) {
  auto I = std::find(vars_.begin(), vars_.end(), N);
  eraseVariable(I);
}

void Graph::eraseNode(Node *N) {
  auto I = std::find(nodes_.begin(), nodes_.end(), N);
  if (Variable *V = dyn_cast<Variable>(N)) {
    return eraseVariable(V);
  }
  assert(I != nodes_.end() && "Could not find node to delete!");
  eraseNode(I);
}

void Graph::verify() const {
  std::unordered_map<std::string, Node *> NameToNode;

  for (auto *V : vars_) {
    if (NameToNode.insert({V->getName(), V}).second)
      continue;
    /// Output extra information helping to find the error.
    llvm::errs() << "The var with name '" << V->getName()
                 << "' conflicts with a previous definition:\n";
    llvm::errs() << "Current definition: " << V->getDebugDesc() << "\n";
    llvm::errs() << "Previous definition: "
                 << NameToNode[V->getName()]->getDebugDesc() << "\n";
    dump();
    llvm_unreachable("Multiple nodes with the same name");
  }

  for (auto *N : nodes_) {
    if (NameToNode.insert({N->getName(), N}).second)
      continue;
    /// Output extra information helping to find the error.
    llvm::outs() << "The node with name '" << N->getName()
                 << "' conflicts with a previous definition:\n";
    llvm::errs() << "Current definition: " << N->getDebugDesc() << "\n";
    llvm::errs() << "Previous definition: "
                 << NameToNode[N->getName()]->getDebugDesc() << "\n";
    dump();
    llvm_unreachable("Multiple nodes with the same name");
  }
}
