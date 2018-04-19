// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <unordered_set>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

bool Module::hasFunction(llvm::StringRef name) { return getFunction(name); }

Function *Module::getFunction(llvm::StringRef name) {
  for (auto *F : functions_) {
    if (F->getName() == name) {
      return F;
    }
  }
  return nullptr;
}

Function *Module::createFunction(llvm::StringRef name) {
  assert(!hasFunction(name) && "A function with this name already exists");
  Function *F = new Function(this, name);
  functions_.push_back(F);
  return F;
}

Module::~Module() {
  for (auto *F : functions_) {
    delete F;
  }

  for (auto it = vars_.begin(), e = vars_.end(); it != e;) {
    auto cur = it++;
    eraseVariable(*cur);
  }
}

void Module::verify() const {
  for (auto *F : functions_) {
    F->verify();
  }
}

void Module::dump() const {
  llvm::outs() << "Module structure:\n";
  for (auto v : getVars()) {
    llvm::outs() << v->getDebugDesc() << "\n";
  }

  for (auto f : functions_) {
    llvm::outs() << "Function:" << f->getName() << "\n";
  }
}

/// A helper class for visiting and generating the dotty graph file.
class AbstractDottyPrinter {
protected:
  // List of generated vertices.
  std::vector<std::string> vertices_{};
  // List of generated edges.
  std::unordered_set<std::string> edges_{};

  /// Dumps label for a input/output row, given port names.
  /// E.g. {"LHS", "RHS"} will produce {<LHS>LHS|<RHS>RHS}
  void dumpLabelForRow(llvm::ArrayRef<std::string> names, std::ostream &os) {
    os << "{";
    for (size_t i = 0; i < names.size(); i++) {
      if (i) {
        os << "|";
      }
      os << "<" << names[i] << ">" << names[i];
    }
    os << "}";
  }

  void dumpLabel(Node *N, std::ostream &os) {
    os << "{";
    if (N->getNumInputs()) {
      std::vector<std::string> names(N->getNumInputs());
      for (size_t i = 0; i < names.size(); i++) {
        names[i] = N->getInputName(i).str();
      }
      dumpLabelForRow(names, os);
      os << "|";
    }
    os << "{" << escapeDottyString(N->getDebugDesc()) << "}";
    if (N->getNumResults()) {
      os << "|";
      std::vector<std::string> names(N->getNumResults());
      for (size_t i = 0; i < names.size(); i++) {
        names[i] = N->getOutputName(i).str();
      }
      dumpLabelForRow(names, os);
    }
    os << "}";
  }

  void dumpNode(Node *N) {
    if (!N) {
      return;
    }
    std::ostringstream os;
    // Print a node descriptor that looks like this:
    // "0xf7fc43e01" [ shape = "record" label = "{...}" ];
    // where 0xf7fc43e01 is address of node.
    os << uniqueVertexName(N) << "[\n";
    os << "\tlabel = \"";
    dumpLabel(N, os);
    os << "\"\n";
    os << "\tshape = \"record\"\n";
    os << "\tstyle=\"filled,rounded\"\n";

    // Pick a color based on the node kind.
    unsigned colorIdx = llvm::hash_value(llvm::StringRef(N->getKindName()));

    static const char *colorNames[] = {
        "AliceBlue",      "CadetBlue1",   "Coral",      "DarkOliveGreen1",
        "DarkSeaGreen1",  "GhostWhite",   "Khaki1",     "LavenderBlush1",
        "LemonChiffon1",  "LightSkyBlue", "MistyRose1", "MistyRose2",
        "PaleTurquoise2", "PeachPuff1",   "PowderBlue", "Salmon",
        "Thistle1",       "Thistle3",     "Wheat1",     "Yellow2",
    };
    unsigned arrayLen = sizeof(colorNames) / sizeof(colorNames[0]);
    auto nodeColor = colorNames[colorIdx % arrayLen];

    if (auto V = llvm::dyn_cast<Variable>(N)) {
      if (V->getVisibilityKind() == Variable::VisibilityKind::Public) {
        os << "\tfillcolor=Snow2 color=DarkOliveGreen4\n";
      } else {
        os << "\tfillcolor=Snow3 color=DeepSkyBlue4\n";
      }
    } else {
      os << "\tfillcolor=" << nodeColor << "\n";
    }
    os << "penwidth = 2];\n";

    vertices_.push_back(os.str());
  }

  void dumpEdgeStyle(Node *N, size_t i, Node *to, std::ostream &os) {
    if (N->isOverwrittenNthInput(i)) {
      os << " [dir=\"both\"]";
    }
  }

  std::string uniqueVertexName(void *N) {
    std::string buffer;
    llvm::raw_string_ostream stream(buffer);
    stream << '"' << N << '"';
    return stream.str();
  }

public:
  void dumpAll(std::ostream &os) {
    os << "digraph DAG {\n\trankdir=TB;\n";

    // Dump vertices:
    for (auto &v : vertices_) {
      os << v << "\n";
    }

    // Dump edges:
    for (auto &e : edges_) {
      os << e << ";\n";
    }

    os << "}";
  }
};

class ModuleDottyPrinter : public AbstractDottyPrinter {
  /// Dump Function as a vertix. Then iterate through Variables, used in the
  /// function, and create corresponding edges.
  void visitFunction(Function *F) {
    std::ostringstream os;
    // Print a Function descriptor that looks like this:
    // "0xf7fc43e01" [ label = "{...}" ];
    // where 0xf7fc43e01 is address of Function.
    os << uniqueVertexName(F) << "[\n"
       << "\tlabel = \"Function\\l"
       << "name : " << F->getName().str() << "\\l"
       << "node count : " << F->getNodes().size() << "\"\n"
       << "\tshape = box\n"
       << "\tfillcolor=gray89, style=\"filled,rounded\"\n"
       << "\t\n"
       << "];\n";
    vertices_.push_back(os.str());

    for (Node *N : F->getNodes()) {
      for (size_t i = 0; i < N->getNumInputs(); i++) {
        Node *to = N->getNthInput(i).getNode();
        size_t resNo = N->getNthInput(i).getResNo();

        if (!isa<Variable>(to))
          continue;

        std::ostringstream edge;
        edge << uniqueVertexName(to) << ":" << to->getOutputName(resNo).str()
             << " -> " << uniqueVertexName(F);
        dumpEdgeStyle(N, i, to, edge);
        edges_.insert(edge.str());
      }
    }
  }

public:
  void visitModule(Module *M) {
    for (auto N : M->getVars()) {
      dumpNode(N);
    }

    for (auto F : M->getFunctions()) {
      visitFunction(F);
    }
  }
};

// TODO: consider refactoring boilerplate code to new trait: DottyPrintable<ADP>
void Module::dumpDAG() {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  stream << "dotty_graph_dump_" << this << ".dot";
  dumpDAG(stream.str().c_str());
}

void Module::dumpDAG(const char *dotFilename) {
  std::string filename = dotFilename;
  llvm::outs() << "Writing dotty graph for Module to: " << filename << '\n';

  ModuleDottyPrinter DP;

  DP.visitModule(this);

  std::ofstream myfile;
  myfile.open(filename);
  DP.dumpAll(myfile);
  myfile.close();
}

Function::~Function() {
  // Delete all of the nodes and the variables.
  for (auto it = nodes_.begin(), e = nodes_.end(); it != e;) {
    auto cur = it++;
    eraseNode(*cur);
  }
}

TypeRef Module::uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims) {
  return uniqueType(Type(elemTy, dims));
}

TypeRef Module::uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims,
                           float scale, int32_t offset) {
  return uniqueType(Type(elemTy, dims, scale, offset));
}

TypeRef Module::uniqueTypeWithNewShape(TypeRef T, llvm::ArrayRef<size_t> dims) {
  return uniqueType(Type::newShape(*T, dims));
}

TypeRef Module::uniqueType(const Type &T) {
  for (auto &tp : types_) {
    if (T.isEqual(tp)) {
      return &tp;
    }
  }

  return &*types_.insert(types_.begin(), T);
}

TypeRef Module::getVoidTy() { return uniqueType(Type()); }

//===----------------------------------------------------------------------===//
//                       Node builders
//===----------------------------------------------------------------------===//

Variable *Module::createVariable(TypeRef T, llvm::StringRef name,
                                 Variable::VisibilityKind visibility,
                                 Variable::TrainKind train, float val) {
  auto FT = uniqueType(*T);
  return addVar(new Variable(name, FT, visibility, train, val));
}

Variable *Module::createVariable(ElemKind T, llvm::ArrayRef<size_t> dims,
                                 llvm::StringRef name,
                                 Variable::VisibilityKind visibility,
                                 Variable::TrainKind train, float val) {
  auto FT = uniqueType(T, dims);
  return createVariable(FT, name, visibility, train, val);
}

Variable *Module::createVariable(ElemKind T, llvm::ArrayRef<size_t> dims,
                                 float scale, int32_t offset,
                                 llvm::StringRef name,
                                 Variable::VisibilityKind visibility,
                                 Variable::TrainKind train, float val) {
  auto FT = uniqueType(T, dims, scale, offset);
  return createVariable(FT, name, visibility, train, val);
}

llvm::StringRef Module::uniqueName(llvm::StringRef name) {
  std::string legalName;

  // Legalize the name.
  for (const char c : name) {
    bool legal = isalpha(c) || isdigit(c) || c == '_';
    legalName.push_back(legal ? c : '_');
  }

  // Names must start with some alphabetic character or underscore and can't be
  // empty.
  if (legalName.empty() || isdigit(legalName[0])) {
    legalName = "A" + legalName;
  }

  auto it = uniqueNames_.insert(legalName);
  if (it.second) {
    // This name is already unique!
    return it.first->first();
  }

  for (unsigned i = 1; i < 10000; i++) {
    auto suffix = std::to_string(i);

    auto it = uniqueNames_.insert(legalName + suffix);
    if (it.second) {
      // Found a unique name!
      return it.first->first();
    }
  }

  llvm_unreachable("Unable to find a unique a name.");
}

void Module::assignUniqueName(Node *N) { N->setName(uniqueName(N->getName())); }

ConvolutionNode *Function::createConv(llvm::StringRef name, NodeValue input,
                                      size_t depth, size_t kernel,
                                      size_t stride, size_t pad, size_t group) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");
  assert(idim.c % group == 0 && "channels number must be divisible by groups");
  assert(depth % group == 0 && "depth must be divisible by groups");

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);

  std::array<size_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};

  // Allocate the Filter and Bias tensors.
  std::array<size_t, 4> filterDim = {{depth, kernel, kernel, idim.c / group}};
  size_t fanIn = kernel * kernel * idim.c;
  auto *filter = getParent()->createVariable(
      ElemKind::FloatTy, filterDim, "filter", Variable::VisibilityKind::Private,
      Variable::TrainKind::Xavier, fanIn);

  auto *bias = getParent()->createVariable(ElemKind::FloatTy, {depth}, "bias",
                                           Variable::VisibilityKind::Private,
                                           Variable::TrainKind::Broadcast, 0.1);

  auto OT = getParent()->uniqueType(ElemKind::FloatTy, outDims);

  return addNode(new ConvolutionNode(name, OT, input, filter, bias, kernel,
                                     stride, pad, group));
}

/// Check that the dimensions that are passed in when the convolution is
/// constructed are correct.
static void assertConvDims(NodeValue input, NodeValue filter, NodeValue bias,
                           size_t kernel, size_t stride, size_t pad,
                           size_t group) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");
  assert(idim.c % group == 0 && "channels number must be divisible by groups");
  (void)idim;

  auto filterDims = filter->dims();
  assert(filterDims[0] % group == 0 && filterDims[1] == kernel &&
         filterDims[2] == kernel && filterDims[3] == idim.c / group &&
         "Invalid filter dims");
  (void)filterDims;

  assert(bias->getType()->size() == filterDims[0] && "Invalid bias size");
}

ConvolutionNode *Function::createConv(llvm::StringRef name, NodeValue input,
                                      NodeValue filter, NodeValue bias,
                                      TypeRef outTy, size_t kernel,
                                      size_t stride, size_t pad, size_t group) {
  assertConvDims(input, filter, bias, kernel, stride, pad, group);
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new ConvolutionNode(name, OT, input, filter, bias, kernel,
                                     stride, pad, group));
}

PoolMaxNode *Function::createPoolMax(llvm::StringRef name, NodeValue input,
                                     size_t kernel, size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  auto OT = getParent()->uniqueTypeWithNewShape(
      input->getType(), {idim.n, outSz.first, outSz.second, idim.c});

  return addNode(new PoolMaxNode(name, OT, input, kernel, stride, pad));
}

PoolAvgNode *Function::createPoolAvg(llvm::StringRef name, NodeValue input,
                                     size_t kernel, size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  auto OT = getParent()->uniqueTypeWithNewShape(
      input->getType(), {idim.n, outSz.first, outSz.second, idim.c});

  return addNode(new PoolAvgNode(name, OT, input, kernel, stride, pad));
}

FullyConnectedNode *Function::createFullyConnected(llvm::StringRef name,
                                                   NodeValue input, Variable *W,
                                                   Variable *B) {
  TypeRef T = input.getType();
  TypeRef OT = getParent()->uniqueTypeWithNewShape(
      T, {input.dims()[0], B->getType()->dims()[0]});

  return addNode(new FullyConnectedNode(name, OT, input, W, B));
}

FullyConnectedNode *Function::createFullyConnected(llvm::StringRef name,
                                                   NodeValue input, Node *W,
                                                   Node *B, TypeRef outTy) {
  assert(outTy->dims().size() == 2 && "Invalid number of dimensions");
  assert(outTy->dims()[0] == input.dims()[0] && "Invalid dimensions");

  return addNode(new FullyConnectedNode(name, outTy, input, W, B));
}

FullyConnectedNode *Function::createFullyConnected(llvm::StringRef name,
                                                   NodeValue input,
                                                   size_t outDepth) {
  TypeRef T = input.getType();
  auto idim = flattenCdr(input.dims());

  size_t fanIn = idim.second;

  auto *W = getParent()->createVariable(
      T->getElementType(), {idim.second, outDepth}, "weights",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier, fanIn);

  auto *B = getParent()->createVariable(T->getElementType(), {outDepth}, "bias",
                                        Variable::VisibilityKind::Private,
                                        Variable::TrainKind::Broadcast, 0.1);

  auto OT =
      getParent()->uniqueType(T->getElementType(), {idim.first, outDepth});
  return addNode(new FullyConnectedNode(name, OT, input, W, B));
}

ReluNode *Function::createRELU(llvm::StringRef name, NodeValue input) {
  return addNode(new ReluNode(name, input));
}

SigmoidNode *Function::createSigmoid(llvm::StringRef name, NodeValue input) {
  return addNode(new SigmoidNode(name, input));
}

TanhNode *Function::createTanh(llvm::StringRef name, NodeValue input) {
  return addNode(new TanhNode(name, input));
}

SoftMaxNode *Function::createSoftMax(llvm::StringRef name, NodeValue input,
                                     NodeValue selected) {
  return addNode(new SoftMaxNode(name, input, selected));
}

CrossEntropyLossNode *Function::createCrossEntropyLoss(llvm::StringRef name,
                                                       NodeValue input,
                                                       NodeValue labels) {
  auto ty = getParent()->uniqueTypeWithNewShape(input.getType(), {1});
  return addNode(new CrossEntropyLossNode(name, ty, input, labels));
}

RegressionNode *Function::createRegression(llvm::StringRef name,
                                           NodeValue input,
                                           NodeValue expected) {
  return addNode(new RegressionNode(name, input, expected));
}

ReshapeNode *Function::createReshape(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<size_t> shape) {
  auto TR = getParent()->uniqueTypeWithNewShape(input.getType(), shape);
  assert(TR->size() == input.getType()->size() &&
         "Reshape to a different size");
  return addNode(new ReshapeNode(name, TR, input, shape.vec()));
}

TransposeNode *Function::createTranspose(llvm::StringRef name, NodeValue input,
                                         llvm::ArrayRef<unsigned> shuffle) {
  ShapeVector shape;
  auto dims = input.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto NT = getParent()->uniqueTypeWithNewShape(input.getType(), shape);
  return addNode(new TransposeNode(name, NT, input, shuffle.vec()));
}

BroadcastNode *Function::createBroadcast(llvm::StringRef name, NodeValue input,
                                         llvm::ArrayRef<size_t> shape,
                                         unsigned axis) {
  auto TR = getParent()->uniqueType(input.getType()->getElementType(), shape);
  return addNode(new BroadcastNode(name, TR, input, shape.vec(), axis));
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

ConcatNode *Function::createConcat(llvm::StringRef name,
                                   llvm::ArrayRef<Node *> inputs,
                                   unsigned dimension) {
  for (int i = 1, e = inputs.size(); i < e; i++) {
    assert(sameSameShapeExceptDim(inputs[i]->getType(), inputs[0]->getType(),
                                  dimension) &&
           "Invalid type");
    (void)sameSameShapeExceptDim;
  }
  auto inDim = inputs[0]->dims();

  ShapeVector shape(inDim.begin(), inDim.end());

  // We are stacking the tensors along a specific dimension. This means that we
  // increase the size of the tensor along this dimension.
  shape[dimension] = 0;
  for (auto I : inputs) {
    shape[dimension] += I->getType()->dims()[dimension];
  }

  auto NT = getParent()->uniqueTypeWithNewShape(inputs[0]->getType(), shape);
  std::vector<NodeValue> ops;
  ops.reserve(inputs.size());
  for (auto I : inputs) {
    ops.emplace_back(I);
  }
  return addNode(new ConcatNode(name, NT, ops, dimension));
}

ConcatNode *Function::createConcat(llvm::StringRef name,
                                   llvm::ArrayRef<Node *> inputs,
                                   unsigned dimension, TypeRef outTy) {
  std::vector<NodeValue> ops;
  ops.reserve(inputs.size());
  for (auto I : inputs) {
    ops.emplace_back(I);
  }

  return addNode(new ConcatNode(name, outTy, ops, dimension));
}

SliceNode *Function::createSlice(llvm::StringRef name, NodeValue input,
                                 llvm::ArrayRef<size_t> start, TypeRef outTy) {
  assert(input.dims().size() == start.size() &&
         "Start and input dims should match");
  assert(outTy->dims().size() == start.size() &&
         "Output and start dims should match");

  for (unsigned i = 0, e = input.dims().size(); i < e; i++) {
    assert(start[i] < input.dims()[i] && "Invalid start position");
    assert(outTy->dims()[i] == start[i] + input.dims()[i] &&
           "Input/Output/Start dims mismatch");
  }

  return addNode(new SliceNode(name, outTy, input, start));
}

SliceNode *Function::createSlice(llvm::StringRef name, NodeValue input,
                                 llvm::ArrayRef<size_t> begin,
                                 llvm::ArrayRef<size_t> end) {

  std::vector<size_t> beginV, shape;
  auto dims = input.dims();
  assert(begin.size() == end.size() && "Begin and End dimensions should match");
  assert(begin.size() == dims.size() &&
         "Begin and Input dimensions should match");
  for (unsigned i = 0; i < dims.size(); i++) {
    size_t beginI = begin[i];
    size_t endI = end[i];
    size_t dimI = dims[i];
    (void)dimI;
    assert(beginI >= 0 && "Illegal Begin indices");
    assert(endI > 0 && "Illegal End indices");
    assert(beginI < dimI && "Illegal Begin indices");
    assert(endI <= dimI && "Illegal End indices");
    assert(endI > beginI && "Illegal Begin and End indices");
    beginV.push_back(beginI);
    shape.push_back(endI - beginI);
  }

  auto NT = getParent()->uniqueTypeWithNewShape(input.getType(), shape);
  return addNode(new SliceNode(name, NT, input, beginV));
}

Node *Function::createChannelShuffle(llvm::StringRef name, NodeValue input,
                                     size_t group, size_t kernel) {
  auto inDims = input.dims();
  assert(kernel < inDims.size());

  ShapeVector dims(inDims.begin(), inDims.end());
  auto D = dims[kernel];
  assert(D % group == 0);

  dims.erase(dims.begin() + kernel);
  // Reshape {D1, ... D_k, ... D_n} -> {D1, ... group, D_k / group, ... D_n}
  dims.insert(dims.begin() + kernel, D / group);
  dims.insert(dims.begin() + kernel, group);
  Node *R1 = createReshape(name.str() + ".reshape1", input, dims);

  std::vector<unsigned> transpose(dims.size());
  for (size_t i = 0; i < transpose.size(); i++)
    transpose[i] = i;
  std::swap(transpose[kernel], transpose[kernel + 1]);
  Node *T = createTranspose(name.str() + ".transpose", R1, transpose);

  return createReshape(name.str() + ".reshape2", T, inDims);
}

BatchNormalizationNode *Function::createBatchNormalization(llvm::StringRef name,
                                                           NodeValue input,
                                                           size_t channelIdx,
                                                           float epsilon,
                                                           float momentum) {
  // Figure out how many channels are in the tensor.
  size_t channels = input.dims()[channelIdx];

  // Allocate the learnable parameters beta and gamma.
  auto *beta = getParent()->createVariable(
      ElemKind::FloatTy, {channels}, "beta", Variable::VisibilityKind::Private,
      Variable::TrainKind::Broadcast, 0.);
  auto *gamma = getParent()->createVariable(
      ElemKind::FloatTy, {channels}, "gamma", Variable::VisibilityKind::Private,
      Variable::TrainKind::Broadcast, 1.0);

  auto *mean = getParent()->createVariable(
      ElemKind::FloatTy, {channels}, "mean", Variable::VisibilityKind::Private,
      Variable::TrainKind::None);
  auto *variance = getParent()->createVariable(
      ElemKind::FloatTy, {channels}, "variance",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);

  return createBatchNormalization(name, input, beta, gamma, mean, variance,
                                  channelIdx, epsilon, momentum);
}

BatchNormalizationNode *Function::createBatchNormalization(
    llvm::StringRef name, NodeValue input, NodeValue beta, NodeValue gamma,
    NodeValue mean, NodeValue var, size_t channelIdx, float epsilon,
    float momentum) {
  return addNode(new BatchNormalizationNode(name, input, gamma, beta, mean, var,
                                            channelIdx, epsilon, momentum));
}

LocalResponseNormalizationNode *Function::createLocalResponseNormalization(
    llvm::StringRef name, NodeValue input, size_t halfWindowSize, float alpha,
    float beta, float k) {
  // The output tensor is of the same shape as the input tensor.
  return addNode(new LocalResponseNormalizationNode(name, input, halfWindowSize,
                                                    alpha, beta, k));
}

#define ARITHMETIC_FUN_DEF(NODE_NAME_)                                         \
  NODE_NAME_##Node *Function::create##NODE_NAME_(                              \
      llvm::StringRef name, NodeValue LHS, NodeValue RHS) {                    \
    return create##NODE_NAME_(name, LHS.getType(), LHS, RHS);                  \
  }                                                                            \
  NODE_NAME_##Node *Function::create##NODE_NAME_(                              \
      llvm::StringRef name, TypeRef T, NodeValue LHS, NodeValue RHS) {         \
    assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");              \
    return addNode(new NODE_NAME_##Node(name, T, LHS, RHS));                   \
  }

ARITHMETIC_FUN_DEF(Add);
ARITHMETIC_FUN_DEF(Mul);
ARITHMETIC_FUN_DEF(Sub);
ARITHMETIC_FUN_DEF(Div);
ARITHMETIC_FUN_DEF(Max);
ARITHMETIC_FUN_DEF(Min);
#undef ARITHMETIC_FUN_DEF

// For the quantized CmpLTE instruction, we require that the scale params be
// (1.0, 0), so that the actual value and comparison value match.
CmpLTENode *Function::createCmpLTE(llvm::StringRef name, NodeValue LHS,
                                   NodeValue RHS) {
  assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");
  TypeRef OT;
  if (LHS.getType()->isQuantizedType()) {
    OT = getParent()->uniqueType(LHS.getType()->getElementType(), LHS.dims(),
                                 1.0, 0);
  } else {
    OT = getParent()->uniqueType(*LHS.getType());
  }
  return addNode(new CmpLTENode(name, OT, LHS, RHS));
}

PowNode *Function::createPow(llvm::StringRef name, NodeValue Base, float exp) {
  return addNode(new PowNode(name, Base.getType(), Base, exp));
}

SelectNode *Function::createSelect(llvm::StringRef name, TypeRef outTy,
                                   NodeValue Cond, NodeValue LHS,
                                   NodeValue RHS) {
  assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");
  assert(LHS.dims() == Cond.dims() && "Invalid operand shapes");
  assert(LHS.dims() == outTy->dims() && "Invalid result shape");
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new SelectNode(name, OT, Cond, LHS, RHS));
}

SelectNode *Function::createSelect(llvm::StringRef name, NodeValue Cond,
                                   NodeValue LHS, NodeValue RHS) {
  auto inDims = LHS.dims();
  assert(inDims.size() > 0);
  ShapeVector outDims(inDims.begin(), inDims.end());
  auto OT = getParent()->uniqueType(LHS->getElementType(), outDims);
  return createSelect(name, OT, Cond, LHS, RHS);
}

SplatNode *Function::createSplat(llvm::StringRef name, TypeRef ty,
                                 float value) {
  return addNode(new SplatNode(name, getParent()->uniqueType(*ty), value));
}

MatMulNode *Function::createMatMul(llvm::StringRef name, TypeRef outTy,
                                   NodeValue lhs, NodeValue rhs) {
  return addNode(
      new MatMulNode(name, getParent()->uniqueType(*outTy), lhs, rhs));
}

MatMulNode *Function::createMatMul(llvm::StringRef name, NodeValue lhs,
                                   NodeValue rhs) {
  auto LT = lhs.getType();
  auto RT = rhs.getType();
  auto LDims = LT->dims();
  auto RDims = RT->dims();
  assert(lhs.getType()->getElementType() == rhs.getType()->getElementType());

  auto ty =
      getParent()->uniqueTypeWithNewShape(lhs.getType(), {LDims[0], RDims[1]});
  return createMatMul(name, ty, lhs, rhs);
}

BatchedReduceAddNode *Function::createBatchedReduceAdd(llvm::StringRef name,
                                                       TypeRef outTy,
                                                       NodeValue batch) {
  assert(outTy->size() == flattenCdr(batch.dims()).second);
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new BatchedReduceAddNode(name, OT, batch));
}

BatchedReduceAddNode *Function::createBatchedReduceAdd(llvm::StringRef name,
                                                       NodeValue batch) {
  auto BT = batch.getType();
  auto OT =
      getParent()->uniqueType(BT->getElementType(), BT->dims().drop_front());
  return createBatchedReduceAdd(name, OT, batch);
}

BatchedAddNode *Function::createBatchedAdd(llvm::StringRef name,
                                           NodeValue batch, NodeValue sample) {
  return addNode(new BatchedAddNode(name, batch.getType(), batch, sample));
}

BatchedAddNode *Function::createBatchedAdd(llvm::StringRef name, TypeRef outTy,
                                           NodeValue batch, NodeValue sample) {
  return addNode(
      new BatchedAddNode(name, getParent()->uniqueType(*outTy), batch, sample));
}

SaveNode *Function::createSave(llvm::StringRef name, NodeValue input) {
  auto *dest = getParent()->createVariable(input.getType(), name,
                                           Variable::VisibilityKind::Public,
                                           Variable::TrainKind::None);

  std::string nodeName{"_save_"};
  nodeName += name;
  return addNode(new SaveNode(nodeName, input, dest));
}

SaveNode *Function::createSave(llvm::StringRef name, NodeValue input,
                               Variable *output) {
  return addNode(new SaveNode(name, input, output));
}

QuantizationProfileNode *
Function::createQuantizationProfile(llvm::StringRef name, NodeValue input) {
  // TODO: this size is going to be refined. Just a placeholder now.
  const size_t numberOfBuckets = 2000U;
  auto *histogram = getParent()->createVariable(
      ElemKind::FloatTy, {numberOfBuckets}, "histogram",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  // Intermediate data used for histogram calculations.
  // Min tensor value seen so far is kept on the first position.
  // Max tensor value seen so far is kept on the second position.
  auto *computationInfo = getParent()->createVariable(
      ElemKind::FloatTy, {2}, "computationInfo",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);

  return addNode(new QuantizationProfileNode(
      name, input, histogram, computationInfo, input->getName().str()));
}

TopKNode *Function::createTopK(llvm::StringRef name, NodeValue input,
                               size_t k) {
  auto inDims = input.dims();
  assert(inDims.size() > 0);
  assert(k <= inDims.back());
  ShapeVector outDims(inDims.begin(), inDims.end());
  outDims.back() = k;
  auto OT = getParent()->uniqueTypeWithNewShape(input.getType(), outDims);
  return addNode(new TopKNode(
      name, OT, getParent()->uniqueType(ElemKind::IndexTy, outDims), input, k));
}

GatherNode *Function::createGather(llvm::StringRef name, NodeValue data,
                                   NodeValue indices) {
  auto dDims = data.dims();
  auto iDims = indices.dims();
  assert(dDims.size() > 0);
  ShapeVector outDims(iDims.begin(), iDims.end());
  outDims.insert(outDims.end(), dDims.begin() + 1, dDims.end());
  return addNode(new GatherNode(
      name, getParent()->uniqueTypeWithNewShape(data->getType(), outDims), data,
      indices));
}

QuantizeNode *Function::createQuantize(llvm::StringRef name, NodeValue input,
                                       TypeRef outTy) {
  assert(input.getElementType() == ElemKind::FloatTy &&
         "Input must be a floating type");
  assert(outTy->getElementType() == ElemKind::Int8QTy &&
         "Output must be a quantized type");
  assert(input->dims().equals(outTy->dims()) &&
         "Different dimensions for input and output");

  return addNode(
      new QuantizeNode(name, getParent()->uniqueType(*outTy), input));
}

DequantizeNode *Function::createDequantize(llvm::StringRef name,
                                           NodeValue input) {
  assert(input.getElementType() == ElemKind::Int8QTy &&
         "Input must be a quantized type");
  TypeRef outTy =
      getParent()->uniqueType(Type(ElemKind::FloatTy, input.dims()));
  return addNode(
      new DequantizeNode(name, getParent()->uniqueType(*outTy), input));
}

RescaleQuantizedNode *Function::createRescaleQuantized(llvm::StringRef name,
                                                       NodeValue input,
                                                       TypeRef outTy) {
  assert(input.getElementType() == ElemKind::Int8QTy &&
         "Input must be a quantized type");
  assert(outTy->getElementType() == ElemKind::Int8QTy &&
         "Output must be a quantized type");
  assert(input->dims().equals(outTy->dims()) &&
         "Different dimensions for input and output");

  return addNode(
      new RescaleQuantizedNode(name, getParent()->uniqueType(*outTy), input));
}

void Function::createSimpleRNN(llvm::StringRef namePrefix,
                               llvm::ArrayRef<Node *> inputs,
                               unsigned batchSize, unsigned hiddenSize,
                               unsigned outputSize,
                               std::vector<Node *> &outputs) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front()->dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the state to zero.
  auto *HInit = getParent()->createVariable(
      ElemKind::FloatTy, {batchSize, hiddenSize}, nameBase + ".initial_state",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  HInit->getPayload().zero();
  Node *Ht = HInit;

  float b = 0.1;
  auto *Whh = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whh",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *Bhh = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bhh",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);
  auto *Wxh = getParent()->createVariable(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxh",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      inputSize);
  auto *Bxh = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bxh",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);
  auto *Why = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, outputSize}, nameBase + ".Why",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *Bhy = getParent()->createVariable(
      ElemKind::FloatTy, {outputSize}, nameBase + ".Bhy",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);

  // Un-roll backpropogation through time as a loop with the shared parameters.
  for (unsigned t = 0; t < timeSteps; t++) {
    auto fc1Name = nameBase + ".fc1." + std::to_string(t);
    auto *FC1 = createFullyConnected(fc1Name, Ht, Whh, Bhh);
    auto fc2Name = nameBase + ".fc2." + std::to_string(t);
    auto *FC2 = createFullyConnected(fc2Name, inputs[t], Wxh, Bxh);
    auto aName = nameBase + ".add." + std::to_string(t);
    auto *A = createAdd(aName, FC1, FC2);
    auto tanhName = nameBase + ".tanh." + std::to_string(t);
    auto *H = createTanh(tanhName, A);
    auto outName = nameBase + ".out." + std::to_string(t);
    auto *O = createFullyConnected(outName, H, Why, Bhy);
    outputs.push_back(O);

    Ht = H;
  };
}

void Function::createGRU(llvm::StringRef namePrefix,
                         llvm::ArrayRef<Node *> inputs, unsigned batchSize,
                         unsigned hiddenSize, unsigned outputSize,
                         std::vector<Node *> &outputs) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front()->dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the state to zero.
  auto *HInit = getParent()->createVariable(
      ElemKind::FloatTy, {batchSize, hiddenSize}, "initial_state",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);

  HInit->getPayload().zero();
  Node *Ht = HInit;

  // Update gate:
  //    Z <- sigmoid(Wxz * x + Whz * h + bz)
  // Reset gate:
  //    R <- sigmoid(Wxr * x + Whr * h + br)
  // Hidden state:
  //    h <- Z . h + (1 - Z) tanh (Wxh * x + Whh * (R . h) + bh)

  // update gate
  float bUpdate = 0.1;
  auto *Wxz = getParent()->createVariable(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxz",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      inputSize);
  auto *Whz = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whz",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *Bz1 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bz1",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bUpdate);
  auto *Bz2 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bz2",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bUpdate);
  float bReset = -1.0;
  // reset gate
  auto *Wxr = getParent()->createVariable(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxr",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      inputSize);
  auto *Whr = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whr",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *Br1 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".br1",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bReset);
  auto *Br2 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".br2",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bReset);

  // hidden state
  float b = 0.1;
  auto *Wxh = getParent()->createVariable(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxh",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      inputSize);
  auto *Whh = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whh",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *Bh1 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bh1",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);
  auto *Bh2 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bh2",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);

  // output layer
  auto *Why = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, outputSize}, nameBase + ".Why",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *By = getParent()->createVariable(
      ElemKind::FloatTy, {outputSize}, nameBase + ".by",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);

  auto *Ones = getParent()->createVariable(
      ElemKind::FloatTy, {batchSize, hiddenSize}, nameBase + ".ones",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);

  Ones->getPayload().getHandle().clear(1.0);

  std::vector<Node *> outputNodes;
  for (unsigned t = 0; t < timeSteps; t++) {
    auto fc1Name = nameBase + ".fc1." + std::to_string(t);
    auto fc2Name = nameBase + ".fc2." + std::to_string(t);
    auto add1Name = nameBase + ".add1." + std::to_string(t);
    auto sigmoid1Name = nameBase + ".sigmoid1." + std::to_string(t);

    auto *Zt = createSigmoid(
        sigmoid1Name,
        createAdd(add1Name, createFullyConnected(fc1Name, Ht, Whz, Bz1),
                  createFullyConnected(fc2Name, inputs[t], Wxz, Bz2)));

    auto fc3Name = nameBase + ".fc3." + std::to_string(t);
    auto fc4Name = nameBase + ".fc4." + std::to_string(t);
    auto add2Name = nameBase + ".add2." + std::to_string(t);
    auto sigmoid2Name = nameBase + ".sigmoid2." + std::to_string(t);

    auto *Rt = createSigmoid(
        sigmoid2Name,
        createAdd(add2Name, createFullyConnected(fc3Name, Ht, Whr, Br1),
                  createFullyConnected(fc4Name, inputs[t], Wxr, Br2)));

    auto zhtName = nameBase + ".zh." + std::to_string(t);
    auto *ZHt = createMul(zhtName, Zt, Ht);

    auto oneMinusZtName = nameBase + ".1-z." + std::to_string(t);
    auto *OneMinusZt = createSub(oneMinusZtName, Ones, Zt);

    auto rhtName = nameBase + ".rh." + std::to_string(t);
    auto *RHt = createMul(rhtName, Rt, Ht);

    auto fc5Name = nameBase + ".fc5." + std::to_string(t);
    auto fc6Name = nameBase + ".fc6." + std::to_string(t);
    auto add3Name = nameBase + ".add3." + std::to_string(t);
    auto tanh1Name = nameBase + ".tanh1." + std::to_string(t);

    auto *Ut = createTanh(
        tanh1Name,
        createAdd(add3Name, createFullyConnected(fc5Name, RHt, Whh, Bh1),
                  createFullyConnected(fc6Name, inputs[t], Wxh, Bh2)));

    auto oneMinusZtUtName = nameBase + "1.-zu." + std::to_string(t);
    auto *OneMinusZtUt = createMul(oneMinusZtUtName, OneMinusZt, Ut);

    auto htName = nameBase + ".H." + std::to_string(t);
    Ht = createAdd(htName, ZHt, OneMinusZtUt);

    auto outName = nameBase + ".out." + std::to_string(t);
    auto *O = createFullyConnected(outName, Ht, Why, By);
    outputs.push_back(O);
  }
};

void Function::createLSTM(llvm::StringRef namePrefix,
                          llvm::ArrayRef<Node *> inputs, unsigned batchSize,
                          unsigned hiddenSize, unsigned outputSize,
                          std::vector<Node *> &outputs) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front()->dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the hidden and cell states to zero.
  auto *HInit = getParent()->createVariable(
      ElemKind::FloatTy, {batchSize, hiddenSize}, "initial_hidden_state",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  HInit->getPayload().zero();
  Node *Ht = HInit;

  auto *CInit = getParent()->createVariable(
      ElemKind::FloatTy, {batchSize, hiddenSize}, "initial_cell_state",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  CInit->getPayload().zero();
  Node *Ct = CInit;

  // Forget gate:
  //    F <- sigmoid(Wxf * x + Whf * h + bf)
  // Input gate:
  //    I <- sigmoid(Wxi * x + Whi * h + bi)
  // Output gate:
  //    O <- sigmoid(Wxo * x + Who * h + bi)
  // Cell state:
  //    C <- F . C + I . tanh(Wxc  * x + Whc * h + bc)
  // Hidden state:
  //    h <- O . tanh(C)

  // forget gate
  float bForget = 1.0;
  auto *Wxf = getParent()->createVariable(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxf",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      inputSize);
  auto *Whf = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whf",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *Bf1 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bf1",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bForget);
  auto *Bf2 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bf2",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bForget);
  // input gate
  float bInput = 0.1;
  auto *Wxi = getParent()->createVariable(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxi",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      inputSize);
  auto *Whi = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whi",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *Bi1 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bi1",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bInput);
  auto *Bi2 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bi2",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bInput);

  // output gate
  float bOutput = 0.1;
  auto *Wxo = getParent()->createVariable(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxo",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      inputSize);
  auto *Who = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Who",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *Bo1 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bo1",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bOutput);
  auto *Bo2 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bo2",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast,
      bOutput);

  // cell state
  float bCell = 0.1;
  auto *Wxc = getParent()->createVariable(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxc",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      inputSize);
  auto *Whc = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whc",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *Bc1 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bc1",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, bCell);
  auto *Bc2 = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bc2",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, bCell);

  // output layer
  float b = 0.1;
  auto *Why = getParent()->createVariable(
      ElemKind::FloatTy, {hiddenSize, outputSize}, nameBase + ".Why",
      Variable::VisibilityKind::Private, Variable::TrainKind::Xavier,
      hiddenSize);
  auto *By = getParent()->createVariable(
      ElemKind::FloatTy, {outputSize}, nameBase + ".by",
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);

  std::vector<Node *> outputNodes;
  for (unsigned t = 0; t < timeSteps; t++) {
    auto fc1Name = nameBase + ".fc1." + std::to_string(t);
    auto fc2Name = nameBase + ".fc2." + std::to_string(t);
    auto add1Name = nameBase + ".add1." + std::to_string(t);
    auto sigmoid1Name = nameBase + ".sigmoid1." + std::to_string(t);

    auto *Ft = createSigmoid(
        sigmoid1Name,
        createAdd(add1Name, createFullyConnected(fc1Name, Ht, Whf, Bf1),
                  createFullyConnected(fc2Name, inputs[t], Wxf, Bf2)));

    auto fc3Name = nameBase + ".fc3." + std::to_string(t);
    auto fc4Name = nameBase + ".fc4." + std::to_string(t);
    auto add2Name = nameBase + ".add2." + std::to_string(t);
    auto sigmoid2Name = nameBase + ".sigmoid2." + std::to_string(t);

    auto *It = createSigmoid(
        sigmoid2Name,
        createAdd(add2Name, createFullyConnected(fc3Name, Ht, Whi, Bi1),
                  createFullyConnected(fc4Name, inputs[t], Wxi, Bi2)));

    auto fc5Name = nameBase + ".fc5." + std::to_string(t);
    auto fc6Name = nameBase + ".fc6." + std::to_string(t);
    auto add3Name = nameBase + ".add3." + std::to_string(t);
    auto sigmoid3Name = nameBase + ".sigmoid3." + std::to_string(t);

    auto *Ot = createSigmoid(
        sigmoid3Name,
        createAdd(add3Name, createFullyConnected(fc5Name, Ht, Who, Bo1),
                  createFullyConnected(fc6Name, inputs[t], Wxo, Bo2)));

    auto fc7Name = nameBase + ".fc7." + std::to_string(t);
    auto fc8Name = nameBase + ".fc8." + std::to_string(t);
    auto add4Name = nameBase + ".add4." + std::to_string(t);
    auto tanh1Name = nameBase + ".tanh1." + std::to_string(t);

    auto *CRt = createTanh(
        tanh1Name,
        createAdd(add4Name, createFullyConnected(fc7Name, Ht, Whc, Bc1),
                  createFullyConnected(fc8Name, inputs[t], Wxc, Bc2)));

    auto mul1Name = nameBase + ".mul1." + std::to_string(t);
    auto mul2Name = nameBase + ".mul2." + std::to_string(t);
    Ct = createAdd(nameBase + ".C." + std::to_string(t),
                   createMul(mul1Name, Ft, Ct), createMul(mul2Name, It, CRt));

    auto htName = nameBase + ".H." + std::to_string(t);
    auto tanh2Name = nameBase + ".tanh2." + std::to_string(t);
    Ht = createMul(htName, Ot, createTanh(tanh2Name, Ct));

    auto outName = nameBase + ".out." + std::to_string(t);
    auto *O = createFullyConnected(outName, Ht, Why, By);
    outputs.push_back(O);
  }
};

//===----------------------------------------------------------------------===//
//                   Graph dumping and printing
//===----------------------------------------------------------------------===//

void Function::dump() const {
  llvm::outs() << "Graph structure " << getName() << ":\n";
  for (auto n : nodes_) {
    llvm::outs() << n->getDebugDesc() << "\n";
  }
}

/// We can't use NodeWalker here, because it ignores result indices, which
/// are critical in generating detailed debug output.
class FunctionDottyPrinter : public AbstractDottyPrinter {
  // A set of already visited (during graph walk) nodes.
  std::unordered_set<Node *> visitedNodes_{};

  /// Recursively traverses inputs of node \p N using Deep First Search.
  /// Each node will be visited no more than once. The method also dumps
  /// edges with their port identifiers in dotty format.
  void visitNode(Node *N) {
    if (visitedNodes_.find(N) != visitedNodes_.end())
      return;
    visitedNodes_.insert(N);

    // Print edges for the predicate field, if it's used.
    if (N->hasPredicate()) {
      auto pred = N->getPredicate();
      size_t resNo = pred.getResNo();
      std::ostringstream edge;
      edge << uniqueVertexName(pred) << ":" << pred->getOutputName(resNo).str()
           << " -> " << uniqueVertexName(N) << ":w";
      dumpEdgeStyle(N, 0, pred, edge);
      edges_.insert(edge.str());
      visitNode(pred);
    }

    for (size_t i = 0; i < N->getNumInputs(); i++) {
      Node *to = N->getNthInput(i).getNode();
      size_t resNo = N->getNthInput(i).getResNo();

      std::ostringstream edge;
      edge << uniqueVertexName(to) << ":" << to->getOutputName(resNo).str()
           << " -> " << uniqueVertexName(N) << ":" << N->getInputName(i).str();
      dumpEdgeStyle(N, i, to, edge);
      edges_.insert(edge.str());

      visitNode(to);
    }
  }

public:
  void visitGraph(Function *F) {
    for (auto N : F->getNodes()) {
      visitNode(N);
    }

    for (auto N : visitedNodes_) {
      dumpNode(N);
    }
  }
};

void Function::dumpDAG() {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  stream << "dotty_graph_dump_" << this << ".dot";
  dumpDAG(stream.str().c_str());
}

void Function::dumpDAG(const char *dotFilename) {
  std::string filename = dotFilename;
  llvm::outs() << "Writing dotty graph for Function to: " << filename << '\n';

  FunctionDottyPrinter DP;

  DP.visitGraph(this);

  std::ofstream myfile;
  myfile.open(filename);
  DP.dumpAll(myfile);
  myfile.close();
}

void Module::eraseVariable(VariablesList::iterator I) {
  if (I == vars_.end())
    return;
  delete *I;
  vars_.erase(I);
}

void Function::eraseNode(NodesList::iterator I) {
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

Variable *Module::getVariableByName(llvm::StringRef name) {
  for (auto *V : getVars()) {
    if (V->getName() == name)
      return V;
  }
  return nullptr;
}

void Module::eraseVariable(Variable *N) {
  auto vars = getVars();
  auto I = std::find(vars.begin(), vars.end(), N);
  eraseVariable(I);
}

void Function::eraseNode(Node *N) {
  if (Variable *V = dyn_cast<Variable>(N)) {
    return getParent()->eraseVariable(V);
  }
  auto I = std::find(nodes_.begin(), nodes_.end(), N);
  assert(I != nodes_.end() && "Could not find node to delete!");
  eraseNode(I);
}

Function *Function::clone(llvm::StringRef newName,
                          llvm::DenseMap<Node *, Node *> *map) {
  Module *M = getParent();
  auto *newF = M->createFunction(newName);

  // Maps current nodes to new nodes.
  llvm::DenseMap<Node *, Node *> currToNew;

  // Clone all of the nodes in the function.
  for (auto *N : getNodes()) {
    Node *copy = N->clone();
    // Record the copy relationship between the graphs.
    currToNew[N] = copy;
    newF->addNode(copy);
  }

  // At this point we have a new invalid function that points into nodes in the
  // original function. Here we update the links between the nodes in the new
  // function.
  for (auto *N : newF->getNodes()) {
    // Fix each one of the inputs of this node.
    for (unsigned inp = 0, e = N->getNumInputs(); inp < e; inp++) {
      NodeValue &input = N->getNthInput(inp);

      auto it = currToNew.find(input.getNode());
      if (it == currToNew.end()) {
        assert(isa<Variable>(input.getNode()) &&
               "Could not find a mapping for some node!");
        continue;
      }

      // Update the node with the edge to the current graph.
      input.setOperand(it->second, input.getResNo());
    }
  }

  // Record the node mapping into the external map.
  if (map) {
    assert(map->empty() && "The external map must be empty");
    for (auto it : currToNew) {
      map->insert(it);
    }
  }

  assert(newF->getNodes().size() == getNodes().size() && "Invalid func size");
  return newF;
}

void Function::verify() const {
  std::unordered_map<std::string, Node *> NameToNode;

  for (auto *V : getParent()->getVars()) {
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

  auto vars = getParent()->getVars();

  // Any node referenced by one of the graph nodes should be part of the Graph.
  for (auto *N : nodes_) {
    for (size_t idx = 0, e = N->getNumInputs(); idx < e; ++idx) {
      assert((std::find(nodes_.begin(), nodes_.end(), N->getNthInput(idx)) !=
                  nodes_.end() ||
              std::find(vars.begin(), vars.end(), N->getNthInput(idx)) !=
                  vars.end()) &&
             "Every node referenced by one of the graph"
             " nodes should be part of the graph");
    }
  }

  for (const auto *N : nodes_) {
    N->verify();
  }
}
