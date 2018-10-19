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
#include "glow/Graph/Graph.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Nodes.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
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

void Module::clear() {
  eraseFunctions();

  for (auto it = constants_.begin(), e = constants_.end(); it != e; it++) {
    Constant *v = *it;
    delete v;
  }
  for (auto it = placeholders_.begin(), e = placeholders_.end(); it != e;
       it++) {
    Placeholder *p = *it;
    delete p;
  }
  constants_.clear();
  placeholders_.clear();
}

Module::~Module() { clear(); }
void Module::verify() const {
  for (auto *F : functions_) {
    F->verify();
  }
}

void Module::dump() const {
  llvm::outs() << "Module structure:\n";
  for (auto v : getConstants()) {
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
        names[i] = N->getInputName(i);
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

    if (isa<Constant>(N)) {
      os << "\tfillcolor=Snow3 color=DeepSkyBlue4\n";
    } else {
      os << "\tfillcolor=" << nodeColor << "\n";
    }
    os << "penwidth = 2];\n";

    vertices_.push_back(os.str());
  }

  void dumpEdgeStyle(const Node *N, size_t i, Node *to, std::ostream &os) {
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
  /// Dump Function as a vertix. Then iterate through constants, used in the
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

    for (auto &N : F->getNodes()) {
      for (size_t i = 0; i < N.getNumInputs(); i++) {
        Node *to = N.getNthInput(i).getNode();
        size_t resNo = N.getNthInput(i).getResNo();

        if (!isa<Constant>(to))
          continue;

        std::ostringstream edge;
        edge << uniqueVertexName(to) << ":" << to->getOutputName(resNo).str()
             << " -> " << uniqueVertexName(F);
        dumpEdgeStyle(&N, i, to, edge);
        edges_.insert(edge.str());
      }
    }
  }

public:
  void visitModule(Module *M) {
    for (auto N : M->getConstants()) {
      dumpNode(N);
    }

    for (auto F : M->getFunctions()) {
      visitFunction(F);
    }
  }
};

// TODO: consider refactoring boilerplate code to new trait: DottyPrintable<ADP>
void Module::dumpDAG() {
  llvm::SmallString<64> dotPath;
  llvm::sys::fs::createTemporaryFile("dotty_graph_dump", "dot", dotPath);
  dumpDAG(dotPath);
}

void Module::dumpDAG(llvm::StringRef dotFilename) {
  llvm::outs() << "Writing dotty graph for Module to: " << dotFilename << '\n';

  ModuleDottyPrinter DP;

  DP.visitModule(this);

  std::ofstream myfile;
  myfile.open(dotFilename);
  DP.dumpAll(myfile);
  myfile.close();
}

void Module::dumpDAG(const char *dotFilename) {
  dumpDAG(llvm::StringRef(dotFilename));
}

void Module::eraseFunctions() {
  while (!functions_.empty()) {
    eraseFunction(*functions_.begin());
  }
}

void Module::eraseFunction(Function *F) {
  auto it = std::find(functions_.begin(), functions_.end(), F);
  assert(it != functions_.end() && "Function is not part of a module");
  functions_.erase(it);
  delete F;
}

Function::~Function() {
  // Delete all of the nodes.
  for (auto it = nodes_.begin(), e = nodes_.end(); it != e;) {
    auto cur = it++;
    eraseNode(&*cur);
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

/// \returns a ShapeVector of rank 1 less than the input \p dims, where the
/// provided \p axis dimension is removed from the shape.
static ShapeVector getNewShapeWithoutAxis(llvm::ArrayRef<size_t> dims,
                                          size_t axis) {
  assert(axis < dims.size() &&
         "Axis to remove must fit inside dimensions of the provided dims.");
  ShapeVector newDims(dims.begin(), dims.end());
  newDims.erase(newDims.begin() + axis);
  return newDims;
}

//===----------------------------------------------------------------------===//
//                       Node builders
//===----------------------------------------------------------------------===//

Placeholder *Module::createPlaceholder(TypeRef T, llvm::StringRef name,
                                       bool isTrainable) {
  auto FT = uniqueType(*T);
  return addPlaceholder(new Placeholder(name, FT, isTrainable));
}

Placeholder *Module::createPlaceholder(ElemKind T, llvm::ArrayRef<size_t> dims,
                                       llvm::StringRef name, bool isTrainable) {
  auto FT = uniqueType(T, dims);
  return createPlaceholder(FT, name, isTrainable);
}

Placeholder *Module::createPlaceholder(ElemKind T, llvm::ArrayRef<size_t> dims,
                                       float scale, int32_t offset,
                                       llvm::StringRef name, bool isTrainable) {
  auto FT = uniqueType(T, dims, scale, offset);
  return createPlaceholder(FT, name, isTrainable);
}

Constant *Module::createConstant(TypeRef T, llvm::StringRef name) {
  auto FT = uniqueType(*T);
  return addConstant(new Constant(name, FT));
}

Constant *Module::createConstant(ElemKind T, llvm::ArrayRef<size_t> dims,
                                 llvm::StringRef name) {
  auto FT = uniqueType(T, dims);
  return createConstant(FT, name);
}

Constant *Module::createConstant(ElemKind T, llvm::ArrayRef<size_t> dims,
                                 float scale, int32_t offset,
                                 llvm::StringRef name) {
  auto FT = uniqueType(T, dims, scale, offset);
  return createConstant(FT, name);
}

Constant *Module::createConstant(llvm::StringRef name, const Tensor &tensor) {
  auto *V = createConstant(tensor.getElementType(), tensor.dims(), name);
  V->assign(&tensor);
  return V;
}

llvm::StringRef Module::uniqueName(llvm::StringRef name,
                                   llvm::StringSet<> &stringTable) {
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

  auto it = stringTable.insert(legalName);
  if (it.second) {
    // This name is already unique!
    return it.first->first();
  }

  for (unsigned i = 1; i < 10000; i++) {
    auto suffix = std::to_string(i);

    auto it = stringTable.insert(legalName + suffix);
    if (it.second) {
      // Found a unique name!
      return it.first->first();
    }
  }

  llvm_unreachable("Unable to find a unique a name.");
}

Constant *Module::addConstant(Constant *V) {
  V->setName(uniqueName(V->getName(), uniqueVariableNames_));
  constants_.push_back(V);
  return V;
}

Placeholder *Module::addPlaceholder(Placeholder *ph) {
  ph->setName(uniqueName(ph->getName(), uniqueVariableNames_));
  placeholders_.push_back(ph);
  return ph;
}

/// Check that the dimensions that are passed in when the convolution is
/// constructed are correct.
static void assertConvDims(NodeValue input, NodeValue filter, NodeValue bias,
                           llvm::ArrayRef<unsigned_t> kernels,
                           llvm::ArrayRef<unsigned_t> strides,
                           llvm::ArrayRef<unsigned_t> pads, unsigned_t group) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  PaddingTLBR pdim(pads);
  (void)pdim;
  ShapeHW kdim(kernels);
  (void)kdim;
  assert((idim.w + pdim.left + pdim.right) >= kdim.width &&
         (idim.h + pdim.top + pdim.bottom) >= kdim.height &&
         "buffer too small for selected stride");
  assert(idim.c % group == 0 && "channels number must be divisible by groups");
  (void)idim;

  auto filterDims = filter.dims();
  assert(filterDims[0] % group == 0 && filterDims[1] == kdim.height &&
         filterDims[2] == kdim.width && filterDims[3] == idim.c / group &&
         "Invalid filter dims");
  (void)filterDims;

  assert(bias.getType()->size() == filterDims[0] && "Invalid bias size");
}

ConvolutionNode *Function::createConv(llvm::StringRef name, NodeValue input,
                                      NodeValue filter, NodeValue bias,
                                      TypeRef outTy,
                                      llvm::ArrayRef<unsigned_t> kernels,
                                      llvm::ArrayRef<unsigned_t> strides,
                                      llvm::ArrayRef<unsigned_t> pads,
                                      unsigned_t group) {
  assertConvDims(input, filter, bias, kernels, strides, pads, group);
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new ConvolutionNode(name, OT, input, filter, bias, kernels,
                                     strides, pads, group));
}

ConvolutionNode *Function::createConv(llvm::StringRef name, NodeValue input,
                                      NodeValue filter, NodeValue bias,
                                      TypeRef outTy, unsigned_t kernel,
                                      unsigned_t stride, unsigned_t pad,
                                      unsigned_t group) {
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  return createConv(name, input, filter, bias, outTy, kernels, strides, pads,
                    group);
}

MaxPoolNode *Function::createMaxPool(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<unsigned_t> kernels,
                                     llvm::ArrayRef<unsigned_t> strides,
                                     llvm::ArrayRef<unsigned_t> pads) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  PaddingTLBR pdim(pads);
  (void)pdim;
  ShapeHW kdim(kernels);
  (void)kdim;
  assert((idim.w + pdim.left + pdim.right) >= kdim.width &&
         (idim.h + pdim.top + pdim.bottom) >= kdim.height &&
         "buffer too small for selected stride");

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  auto OT = getParent()->uniqueTypeWithNewShape(
      input.getType(), {idim.n, outSz.first, outSz.second, idim.c});

  return addNode(new MaxPoolNode(name, OT, input, kernels, strides, pads));
}

MaxPoolNode *Function::createMaxPool(llvm::StringRef name, NodeValue input,
                                     unsigned_t kernel, unsigned_t stride,
                                     unsigned_t pad) {
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  return createMaxPool(name, input, kernels, strides, pads);
}

AvgPoolNode *Function::createAvgPool(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<unsigned_t> kernels,
                                     llvm::ArrayRef<unsigned_t> strides,
                                     llvm::ArrayRef<unsigned_t> pads) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  PaddingTLBR pdim(pads);
  (void)pdim;
  ShapeHW kdim(kernels);
  (void)kdim;
  assert((idim.w + pdim.left + pdim.right) >= kdim.width &&
         (idim.h + pdim.top + pdim.bottom) >= kdim.height &&
         "buffer too small for selected stride");

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  auto OT = getParent()->uniqueTypeWithNewShape(
      input.getType(), {idim.n, outSz.first, outSz.second, idim.c});

  return addNode(new AvgPoolNode(name, OT, input, kernels, strides, pads));
}

AvgPoolNode *Function::createAvgPool(llvm::StringRef name, NodeValue input,
                                     unsigned_t kernel, unsigned_t stride,
                                     unsigned_t pad) {
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  return createAvgPool(name, input, kernels, strides, pads);
}

FullyConnectedNode *Function::createFullyConnected(llvm::StringRef name,
                                                   NodeValue input, Storage *W,
                                                   Storage *B) {
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

  TypeRef OT = getParent()->uniqueType(*outTy);
  return addNode(new FullyConnectedNode(name, OT, input, W, B));
}

RowwiseQuantizedFullyConnectedNode *
Function::createRowwiseQuantizedFullyConnected(llvm::StringRef name,
                                               NodeValue input, Constant *W,
                                               Node *B, TypeRef outTy) {
  // Since W is constant, quantize it in compilation time.
  // The quantized data is in qWeights, the scale of each row is in scales,
  // and the offset of each row is in offsets.
  Constant *weights = llvm::cast<Constant>(W);
  size_t numRows = W->getType()->dims()[0];

  // So far, if we want to create a storage with Int8QTy/Int16QTy/Int32QTy,
  // it is assumed to be quantized data and the scale and offset should be
  // provided. But for rowwise quantization, the scales and offsets are stored
  // in vectors separately, we add the dummy scale and offset here.
  auto *qWeights = getParent()->createConstant(ElemKind::Int8QTy, W->dims(),
                                               0.0, 0, "weights.rwqfc");
  auto *scales =
      getParent()->createConstant(ElemKind::FloatTy, {numRows}, "scales.rwqfc");
  auto *offsets = getParent()->createConstant(ElemKind::Int32QTy, {numRows},
                                              0.0, 0, "offsets.rwqfc");

  quantization::tensorRowwiseQuantization(
      weights->getPayload(), qWeights->getPayload(), scales->getPayload(),
      offsets->getPayload());

  return addNode(new RowwiseQuantizedFullyConnectedNode(
      name, outTy, input, qWeights, scales, offsets, B));
}

ReluNode *Function::createRELU(llvm::StringRef name, NodeValue input,
                               TypeRef outTy) {
  return addNode(new ReluNode(name, outTy, input));
}

ReluNode *Function::createRELU(llvm::StringRef name, NodeValue input) {
  return addNode(new ReluNode(name, input.getType(), input));
}

SigmoidNode *Function::createSigmoid(llvm::StringRef name, NodeValue input) {
  return addNode(new SigmoidNode(name, input));
}

TanhNode *Function::createTanh(llvm::StringRef name, NodeValue input) {
  return addNode(new TanhNode(name, input));
}

SoftMaxNode *Function::createSoftMax(llvm::StringRef name, NodeValue input,
                                     NodeValue selected, TypeRef outTy) {
  // By default, pick the input type
  if (!outTy) {
    outTy = getParent()->uniqueType(*input.getType());
  }
  return addNode(new SoftMaxNode(name, outTy, input, selected));
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

SigmoidCrossEntropyWithLogitsNode *
Function::createSigmoidCrossEntropyWithLogits(llvm::StringRef name,
                                              NodeValue logits,
                                              NodeValue targets) {
  assert(logits.dims().size() > 1);
  std::vector<size_t> outDims(logits.dims().begin(), logits.dims().end() - 1);
  auto ty = getParent()->uniqueTypeWithNewShape(logits.getType(), outDims);
  return addNode(
      new SigmoidCrossEntropyWithLogitsNode(name, ty, logits, targets));
}

ReshapeNode *Function::createReshape(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<size_t> shape) {
  auto TR = getParent()->uniqueTypeWithNewShape(input.getType(), shape);
  assert(TR->size() == input.getType()->size() &&
         "Reshape to a different size");
  return addNode(new ReshapeNode(name, TR, input, shape.vec()));
}

TransposeNode *Function::createTranspose(llvm::StringRef name, NodeValue input,
                                         llvm::ArrayRef<unsigned_t> shuffle) {
  ShapeVector shape;
  auto dims = input.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto NT = getParent()->uniqueTypeWithNewShape(input.getType(), shape);
  return addNode(new TransposeNode(name, NT, input, shuffle.vec()));
}

Node *Function::createBroadcast(llvm::StringRef name, NodeValue input,
                                llvm::ArrayRef<size_t> newShape,
                                unsigned_t axis) {
  const auto &origDims = input.dims();

  assert(axis + origDims.size() <= newShape.size() &&
         "Axis must fit inside the newShape.");

  // Iterate over the new shape; if the original shape had a dimension here
  // (when considering the axis) then verify the dimension either matches the
  // new shape (no action taken) or == 1 (broadcast in that direction). Else
  // the original shape had no dimensions here (after considering axis), so
  // add the new dimension and broadcast in that direction.
  size_t reshapeDims[max_tensor_dimensions];
  for (size_t i = 0; i < newShape.size(); i++) {
    if (i >= axis && i < origDims.size() + axis) {
      const int origIdx = i - axis;
      if (origDims[origIdx] == newShape[i]) {
        // Keep original dimensions; they are compatible.
        reshapeDims[i] = origDims[origIdx];
      } else if (origDims[origIdx] == 1) {
        // Will broadcast this dimension to size from newShape.
        reshapeDims[i] = 1;
      } else {
        // Incompatible dimensions for broadcasting
        llvm_unreachable("Cannot broadcast with these dimensions.");
      }
    } else {
      // Will broadcast this dimension to size from newShape.
      reshapeDims[i] = 1;
    }
  }

  // Reshape the input node to same number of dimensions as new shape, but with
  // 1s in place of to-be-brodacasted dimensions.
  Node *currNode =
      createReshape(name.str() + ".reshape", input,
                    llvm::ArrayRef<size_t>(reshapeDims, newShape.size()));

  // Create a Tile (which is really a Concat) in each direction that needs to be
  // broadcasted.
  for (size_t i = 0; i < newShape.size(); i++) {
    if (reshapeDims[i] == 1 && newShape[i] != 1) {
      currNode = createTile(name.str() + ".tile" + std::to_string(i), currNode,
                            newShape[i], i);
    }
  }

  return currNode;
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
                                   llvm::ArrayRef<NodeValue> inputs,
                                   unsigned_t dimension) {
  for (int i = 1, e = inputs.size(); i < e; i++) {
    assert(sameSameShapeExceptDim(inputs[i].getType(), inputs[0].getType(),
                                  dimension) &&
           "Invalid type");
    (void)sameSameShapeExceptDim;
  }
  auto inDim = inputs[0].dims();

  ShapeVector shape(inDim.begin(), inDim.end());

  // We are stacking the tensors along a specific dimension. This means that we
  // increase the size of the tensor along this dimension.
  shape[dimension] = 0;
  for (auto I : inputs) {
    shape[dimension] += I.getType()->dims()[dimension];
  }

  auto NT = getParent()->uniqueTypeWithNewShape(inputs[0].getType(), shape);
  std::vector<NodeValue> ops;
  ops.reserve(inputs.size());
  for (auto I : inputs) {
    ops.emplace_back(I);
  }
  return addNode(new ConcatNode(name, NT, ops, dimension));
}

ConcatNode *Function::createConcat(llvm::StringRef name,
                                   llvm::ArrayRef<NodeValue> inputs,
                                   unsigned_t dimension, TypeRef outTy) {
  std::vector<NodeValue> ops;
  ops.reserve(inputs.size());
  for (auto I : inputs) {
    ops.emplace_back(I);
  }

  TypeRef OT = getParent()->uniqueType(*outTy);
  return addNode(new ConcatNode(name, OT, ops, dimension));
}

TileNode *Function::createTile(llvm::StringRef name, NodeValue input,
                               unsigned_t tiles, unsigned_t axis,
                               TypeRef outTy) {
  assert(tiles > 0 && "Tiles must be non-zero.");
  assert(axis >= 0 && axis < input.dims().size() &&
         "Axis must fall in range of source dims.");

  if (outTy == nullptr) {
    ShapeVector outShape(input.dims().begin(), input.dims().end());
    outShape[axis] *= tiles;
    outTy = getParent()->uniqueTypeWithNewShape(input.getType(), outShape);
  }

  return addNode(new TileNode(name, outTy, input, tiles, axis));
}

InsertTensorNode *Function::createInsertTensor(llvm::StringRef name,
                                               NodeValue big, NodeValue small,
                                               llvm::ArrayRef<size_t> start,
                                               unsigned_t count,
                                               unsigned_t axis) {
  return addNode(new InsertTensorNode(name, big, small, start, count, axis));
}

SliceNode *Function::createSlice(llvm::StringRef name, NodeValue input,
                                 llvm::ArrayRef<size_t> start, TypeRef outTy) {
  assert(input.dims().size() == start.size() &&
         "Start and input dims should match");
  assert(outTy->dims().size() == start.size() &&
         "Output and start dims should match");

  for (unsigned i = 0, e = input.dims().size(); i < e; i++) {
    assert(start[i] + outTy->dims()[i] <= input.dims()[i] &&
           "Input/Output/Start dims mismatch");
  }

  TypeRef OT = getParent()->uniqueType(*outTy);
  return addNode(new SliceNode(name, OT, input, start));
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

  std::vector<unsigned_t> transpose(dims.size());
  for (size_t i = 0; i < transpose.size(); i++)
    transpose[i] = i;
  std::swap(transpose[kernel], transpose[kernel + 1]);
  Node *T = createTranspose(name.str() + ".transpose", R1, transpose);

  return createReshape(name.str() + ".reshape2", T, inDims);
}

ReshapeNode *Function::createSqueeze(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<size_t> axes) {
  assert(!axes.empty() && "Parameter `axes` must be provided.");

  ShapeVector shapeAxes(axes.begin(), axes.end());

  // Sort and unique the values in axes to
  // 1. make sure each dim is only removed once;
  // 2. check if the size and value of dimensions to squeeze are valid.
  std::sort(shapeAxes.begin(), shapeAxes.end());
  shapeAxes.erase(std::unique(shapeAxes.begin(), shapeAxes.end()),
                  shapeAxes.end());
  auto inDims = input.dims();
  assert(shapeAxes.back() < inDims.size() && "The size and value of dimensions "
                                             "to squeeze must be less than the "
                                             "input size.");

  ShapeVector newDims;
  size_t j = 0;
  for (size_t i = 0, e = inDims.size(); i < e; i++) {
    if (j < shapeAxes.size() && shapeAxes[j] == i) {
      assert(inDims[i] == 1 && "The dimension to squeeze must be 1.");
      j++;
    } else {
      newDims.push_back(inDims[i]);
    }
  }
  return createReshape(name.str() + ".reshape", input, newDims);
}

ReshapeNode *Function::createExpandDims(llvm::StringRef name, NodeValue input,
                                        llvm::ArrayRef<size_t> axes) {
  assert(!axes.empty() && "Parameter `axes` must be provided.");

  // Dimensions provided in axes are for the output tensor, so we sort them and
  // unique them to make sure they are processed correctly and in the right
  // order.
  ShapeVector shapeAxes(axes.begin(), axes.end());
  std::sort(shapeAxes.begin(), shapeAxes.end());
  shapeAxes.erase(std::unique(shapeAxes.begin(), shapeAxes.end()),
                  shapeAxes.end());

  const auto inDims = input.dims();

  // The total number of dimensions in the new shape is equal to the original
  // shape size plus the uniqued new shape axes, which represents where to
  // insert dimensions of 1 into the output tensor's shape.
  const size_t totalNumNewDims = shapeAxes.size() + inDims.size();
  assert(totalNumNewDims <= max_tensor_dimensions &&
         "New expanded shape has too many dimensions.");
  assert(shapeAxes.back() < totalNumNewDims &&
         "Specified axis expands outside size of output tensor shape.");
  ShapeVector newDims;
  for (size_t i = 0, j = 0, k = 0; k < totalNumNewDims; k++) {
    if (j < shapeAxes.size() && shapeAxes[j] == k) {
      newDims.push_back(1);
      j++;
    } else {
      assert(i < inDims.size() && "Somehow overflowing inDims.");
      newDims.push_back(inDims[i]);
      i++;
    }
  }

  // Create a reshape of the original data with the newly determined dimensions.
  return createReshape(name.str() + ".expanddims", input, newDims);
}

ReshapeNode *Function::createFlatten(llvm::StringRef name, NodeValue input,
                                     unsigned_t axis) {
  auto xDim = flattenCdr(input.getType()->dims(), axis);
  return createReshape(name, input, {xDim.first, xDim.second});
}

void Function::createSplit(llvm::StringRef name, NodeValue input,
                           unsigned_t outputNum, unsigned_t axis,
                           llvm::ArrayRef<size_t> split,
                           std::vector<Node *> &outputs) {
  auto inDims = input.dims();
  if (split.empty()) {
    assert(inDims[axis] % outputNum == 0 &&
           "Dimension to split must be divisible by outputs number.");
  } else {
    assert(outputNum == split.size() &&
           "Number of splits must be divisible by outputs number.");
  }

  ShapeVector start(inDims.size(), 0);
  ShapeVector end(inDims.begin(), inDims.end());
  end[axis] = 0;

  outputs.resize(outputNum);
  for (size_t i = 0; i < outputNum; i++) {
    size_t curLength = split.empty() ? inDims[axis] / outputNum : split[i];
    end[axis] += curLength;
    outputs[i] =
        createSlice(name.str() + ".out" + std::to_string(i), input, start, end);
    start[axis] = end[axis];
  }

  assert(end[axis] == inDims[axis] &&
         "Total size of results must be equal to input size.");
}

BatchNormalizationNode *Function::createBatchNormalization(
    llvm::StringRef name, NodeValue input, NodeValue beta, NodeValue gamma,
    NodeValue mean, NodeValue var, unsigned_t channelIdx, float epsilon,
    float momentum) {
  return addNode(new BatchNormalizationNode(name, input, gamma, beta, mean, var,
                                            channelIdx, epsilon, momentum));
}

LocalResponseNormalizationNode *Function::createLocalResponseNormalization(
    llvm::StringRef name, NodeValue input, unsigned_t halfWindowSize,
    float alpha, float beta, float k) {
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
    TypeRef OT = getParent()->uniqueType(*T);                                  \
    return addNode(new NODE_NAME_##Node(name, OT, LHS, RHS));                  \
  }

ARITHMETIC_FUN_DEF(Add);
ARITHMETIC_FUN_DEF(Mul);
ARITHMETIC_FUN_DEF(Sub);
ARITHMETIC_FUN_DEF(Div);
ARITHMETIC_FUN_DEF(Max);
ARITHMETIC_FUN_DEF(Min);
ARITHMETIC_FUN_DEF(Pow);
#undef ARITHMETIC_FUN_DEF

/// This helper function is used only by the functions creating boolean
/// operations like createCmpEQ and createCmpLTE to compute their output types.
/// The output type is computed based on the type \p T of the operand of the
/// logical operation. In case of a quantized type we provide concrete scale and
/// offset for the type as we know that the output of a boolean operation could
/// be only 0 or 1.
///
/// \returns the output type for a boolean operation.
static TypeRef getResultTypeOfBooleanOp(Module &M, TypeRef T) {
  TypeRef OT;
  if (T->isQuantizedType()) {
    OT = M.uniqueType(T->getElementType(), T->dims(), 1.0, 0);
  } else {
    OT = M.uniqueType(*T);
  }
  return OT;
}

// For the quantized CmpLTE instruction, we require that the scale params be
// (1.0, 0), so that the actual value and comparison value match.
CmpLTENode *Function::createCmpLTE(llvm::StringRef name, NodeValue LHS,
                                   NodeValue RHS) {
  assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");
  TypeRef OT = getResultTypeOfBooleanOp(*getParent(), LHS.getType());
  return addNode(new CmpLTENode(name, OT, LHS, RHS));
}

CmpEQNode *Function::createCmpEQ(llvm::StringRef name, NodeValue LHS,
                                 NodeValue RHS) {
  assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");
  TypeRef OT = getResultTypeOfBooleanOp(*getParent(), LHS.getType());
  return addNode(new CmpEQNode(name, OT, LHS, RHS));
}

IsNaNNode *Function::createIsNaN(llvm::StringRef name, NodeValue input) {
  TypeRef OT = getResultTypeOfBooleanOp(*getParent(), input.getType());
  return addNode(new IsNaNNode(name, OT, input));
}

Node *Function::createReplaceNaN(llvm::StringRef name, NodeValue input,
                                 float value) {
  // Create IsNaN node.
  auto *INN = createIsNaN(name.str() + ".isNaN", input);

  // Create Splat node.
  auto *S = createSplat(name.str() + ".splat", input.getType(), value);

  // Create Select node to pick between original and replacement values.
  auto *SN = createSelect(name.str() + ".select", INN, S, input);

  return SN;
}

PowNode *Function::createPow(llvm::StringRef name, NodeValue base, float exp) {
  auto *SP = createSplat(name, base.getType(), exp);
  return createPow(name, base, SP);
}

LogNode *Function::createLog(llvm::StringRef name, NodeValue input,
                             TypeRef outTy) {
  return addNode(new LogNode(name, outTy ? outTy : input.getType(), input));
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
  auto OT = getParent()->uniqueType(LHS.getElementType(), outDims);
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

Node *Function::createBroadcastedBatchMatMul(llvm::StringRef name,
                                             NodeValue lhs, NodeValue rhs) {
  assert(lhs.dims().size() == 3 && rhs.dims().size() == 2 &&
         "Only supporting lhs 3d, rhs 2d for Broadcasted BatchMatMul.");

  // LHS = {numBatches, N, M}
  // RHS = {M, P}
  // Multiply each LHS matrix {N, M} by RHS {M, P} to get final matrix
  // {numBatches, N, P}
  const auto numBatches = lhs.dims()[0];
  const auto N = lhs.dims()[1];
  const auto M = lhs.dims()[2];
  const auto P = rhs.dims()[1];
  assert((rhs.dims()[0] == M) && "Batch matmul dimensions are invalid.");

  // Reshape the LHS to be a two-dimensional matrix, where each batch
  // is essentially concatenated onto itself in the 0th dimension.
  auto *reshapeLHS =
      createReshape(name.str() + ".reshapeLHS", lhs, {numBatches * N, M});

  // Perform a normal matmul, implementing the batch matmul.
  auto *MMN = createMatMul(name, reshapeLHS, rhs);

  assert(MMN->getResult().dims()[0] == (numBatches * N) &&
         "Incorrect resulting dimension for batch matmul");
  assert(MMN->getResult().dims()[1] == P &&
         "Incorrect resulting dimension for batch matmul");

  // Reshape the result back to the expected batch output shape, with the first
  // dimension the number of batches.
  return createReshape(name.str() + ".reshapeResult", MMN, {numBatches, N, P});
}

Node *Function::createParallelBatchMatMul(llvm::StringRef name, NodeValue lhs,
                                          NodeValue rhs) {
  assert(lhs.dims().size() == 3 && rhs.dims().size() == 3 &&
         "Only supporting lhs 3d, rhs 3d for Parallel BatchMatMul.");

  // LHS = {numBatches, N, M}
  // RHS = {numBatches, M, P}
  // Multiply i-th LHS matrix {N, M} by i-th RHS matrix {M, P} to get final
  // matrix
  // {numBatches, N, P}
  const auto numBatches = lhs.dims()[0];
  const auto N = lhs.dims()[1];
  const auto M = lhs.dims()[2];
  const auto P = rhs.dims()[2];
  assert((rhs.dims()[0] == numBatches) &&
         "Batch matmul dimensions are invalid.");
  assert((rhs.dims()[1] == M) && "Batch matmul dimensions are invalid.");

  std::vector<NodeValue> MMS(numBatches);
  for (size_t i = 0; i < numBatches; i++) {
    auto *sliceA = createSlice(name.str() + ".sliceA." + std::to_string(i), lhs,
                               {i, 0, 0}, {i + 1, N, M});
    auto *sliceB = createSlice(name.str() + ".sliceB." + std::to_string(i), rhs,
                               {i, 0, 0}, {i + 1, M, P});
    auto *reshapeA =
        createReshape(sliceA->getName().str() + ".reshape", sliceA, {N, M});
    auto *reshapeB =
        createReshape(sliceB->getName().str() + ".reshape", sliceB, {M, P});
    MMS[i] = createMatMul(name.str() + ".MatMul." + std::to_string(i), reshapeA,
                          reshapeB);
  }

  auto *concat = createConcat(name.str() + ".concat", MMS, 0);
  return createReshape(name.str() + ".FinalReshape", concat,
                       {numBatches, N, P});
}

BatchedReduceAddNode *Function::createBatchedReduceAdd(llvm::StringRef name,
                                                       TypeRef outTy,
                                                       NodeValue batch,
                                                       unsigned_t axis) {
  // Calculate the expected total number of elements in the output tensor based
  // on the number of elements in the batch divided by the axis dimension.
  const size_t outNumElements = batch.getType()->size() / batch.dims()[axis];
  (void)outNumElements;
  assert(outTy->size() == outNumElements &&
         "Incorrect number of elements in the output type.");
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new BatchedReduceAddNode(name, OT, batch, axis));
}

BatchedReduceAddNode *Function::createBatchedReduceAdd(llvm::StringRef name,
                                                       NodeValue batch,
                                                       unsigned_t axis) {
  auto outDims = getNewShapeWithoutAxis(batch.dims(), axis);
  auto OT = getParent()->uniqueType(batch.getType()->getElementType(), outDims);
  return createBatchedReduceAdd(name, OT, batch, axis);
}

DivNode *Function::createBatchedReduceMean(llvm::StringRef name, TypeRef outTy,
                                           NodeValue batch, unsigned_t axis) {
  // Use the same output type for both the batched reduce add and the
  // denominator splat. Only use outTy as the output of the final div.
  auto redDims = getNewShapeWithoutAxis(batch.dims(), axis);
  auto outTyBRA = getParent()->uniqueTypeWithNewShape(batch.getType(), redDims);

  // Create a batched add to sum up the values in the provided axis.
  auto *BRA = createBatchedReduceAdd(name, outTyBRA, batch, axis);

  // Create a splat of the same output type as the BRA, with value of the size
  // of the original dimensions of the axis, to divide the BRA by.
  auto *SN = createSplat(name.str() + ".denom", outTyBRA, batch.dims()[axis]);

  // Element-wise divide to produce the reduced mean with outTy provided.
  return createDiv(name.str() + ".res", outTy, BRA, SN);
}

DivNode *Function::createBatchedReduceMean(llvm::StringRef name,
                                           NodeValue batch, unsigned_t axis) {
  auto redDims = getNewShapeWithoutAxis(batch.dims(), axis);
  auto outTy = getParent()->uniqueTypeWithNewShape(batch.getType(), redDims);
  return createBatchedReduceMean(name, outTy, batch, axis);
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

SparseLengthsWeightedSumNode *
Function::createSparseLengthsSum(llvm::StringRef name, NodeValue data,
                                 NodeValue indices, NodeValue lengths) {
  auto ty =
      getParent()->uniqueTypeWithNewShape(data.getType(), {indices.dims()[0]});
  auto ones = createSplat(name.str() + ".ones", ty, 1.0);
  return createSparseLengthsWeightedSum(name, data, ones, indices, lengths);
}

SparseLengthsWeightedSumNode *
Function::createSparseLengthsWeightedSum(llvm::StringRef name, NodeValue data,
                                         NodeValue weights, NodeValue indices,
                                         NodeValue lengths) {
  auto inDims = data.dims();
  ShapeVector outDims(inDims.begin(), inDims.end());
  outDims[0] = lengths.dims()[0];
  auto outTy = getParent()->uniqueTypeWithNewShape(data.getType(), outDims);
  return addNode(new SparseLengthsWeightedSumNode(name, outTy, data, weights,
                                                  indices, lengths));
}

LengthsToRangesNode *Function::createLengthsToRanges(llvm::StringRef name,
                                                     NodeValue lengths) {
  ShapeVector outDims({lengths.dims()[0], 2});
  auto outTy = getParent()->uniqueTypeWithNewShape(lengths.getType(), outDims);
  return addNode(new LengthsToRangesNode(name, outTy, lengths));
}

SaveNode *Function::createSave(llvm::StringRef name, NodeValue input) {
  auto *dest = getParent()->createPlaceholder(input.getType(), name, false);

  return addNode(new SaveNode(name, input, dest));
}

SaveNode *Function::createSave(llvm::StringRef name, NodeValue input,
                               Placeholder *output) {
  return addNode(new SaveNode(name, input, output));
}

QuantizationProfileNode *
Function::createQuantizationProfile(Context &ctx, llvm::StringRef name,
                                    NodeValue input) {
  // TODO: this size is going to be refined. Just a placeholder now.
  const size_t numberOfBuckets = 2000U;
  auto *histogram = getParent()->createPlaceholder(
      ElemKind::FloatTy, {numberOfBuckets}, "histogram", false);
  ctx.allocate(histogram);
  // Intermediate data used for histogram calculations.
  // Min tensor value seen so far is kept on the first position.
  // Max tensor value seen so far is kept on the second position.
  auto *computationInfo = getParent()->createPlaceholder(
      ElemKind::FloatTy, {2}, "computationInfo", false);
  ctx.allocate(computationInfo);
  return addNode(new QuantizationProfileNode(
      name, input, histogram, computationInfo, input.getNode()->getName().str(),
      input.getResNo()));
}

IntLookupTableNode *
Function::createIntLookupTable(llvm::StringRef name, NodeValue input,
                               llvm::ArrayRef<int8_t> initValues,
                               TypeRef outTy) {
  auto *mapping = getParent()->createConstant(
      ElemKind::Int8QTy, {initValues.size()}, outTy->getScale(),
      outTy->getOffset(), "mapping");
  mapping->getHandle<int8_t>() = initValues;

  return addNode(new IntLookupTableNode(name, outTy, input, mapping));
}

IntLookupTableNode *Function::createIntTanh(llvm::StringRef name,
                                            NodeValue input, TypeRef outTy) {
  static int8_t mapping[] = {
      -128, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126,
      -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125,
      -125, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124,
      -124, -124, -123, -123, -123, -123, -123, -123, -122, -122, -122, -122,
      -121, -121, -121, -120, -120, -120, -120, -119, -119, -118, -118, -118,
      -117, -117, -116, -116, -115, -115, -114, -114, -113, -112, -112, -111,
      -110, -109, -109, -108, -107, -106, -105, -104, -103, -102, -101, -100,
      -99,  -98,  -96,  -95,  -94,  -92,  -91,  -89,  -88,  -86,  -85,  -83,
      -81,  -79,  -77,  -76,  -74,  -72,  -69,  -67,  -65,  -63,  -61,  -58,
      -56,  -53,  -51,  -48,  -46,  -43,  -41,  -38,  -35,  -32,  -29,  -27,
      -24,  -21,  -18,  -15,  -12,  -9,   -6,   -3,   0,    3,    6,    9,
      12,   15,   18,   21,   24,   27,   29,   32,   35,   38,   41,   43,
      46,   48,   51,   53,   56,   58,   61,   63,   65,   67,   69,   72,
      74,   76,   77,   79,   81,   83,   85,   86,   88,   89,   91,   92,
      94,   95,   96,   98,   99,   100,  101,  102,  103,  104,  105,  106,
      107,  108,  109,  109,  110,  111,  112,  112,  113,  114,  114,  115,
      115,  116,  116,  117,  117,  118,  118,  118,  119,  119,  120,  120,
      120,  120,  121,  121,  121,  122,  122,  122,  122,  123,  123,  123,
      123,  123,  123,  124,  124,  124,  124,  124,  124,  124,  125,  125,
      125,  125,  125,  125,  125,  125,  125,  125,  125,  126,  126,  126,
      126,  126,  126,  126,  126,  126,  126,  126,  126,  126,  126,  126,
      126,  126,  126,  127};

  return createIntLookupTable(name, input, mapping, outTy);
}

IntLookupTableNode *Function::createIntSigmoid(llvm::StringRef name,
                                               NodeValue input, TypeRef outTy) {
  static int8_t mapping[] = {
      -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
      -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126,
      -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125,
      -125, -125, -124, -124, -124, -124, -124, -123, -123, -123, -123, -122,
      -122, -122, -122, -121, -121, -121, -120, -120, -120, -119, -119, -118,
      -118, -118, -117, -117, -116, -115, -115, -114, -114, -113, -112, -112,
      -111, -110, -109, -109, -108, -107, -106, -105, -104, -103, -102, -101,
      -99,  -98,  -97,  -96,  -94,  -93,  -91,  -90,  -88,  -87,  -85,  -83,
      -82,  -80,  -78,  -76,  -74,  -72,  -70,  -68,  -66,  -63,  -61,  -59,
      -56,  -54,  -51,  -49,  -46,  -44,  -41,  -38,  -36,  -33,  -30,  -27,
      -24,  -21,  -18,  -15,  -12,  -9,   -6,   -3,   -1,   2,    5,    8,
      11,   14,   17,   20,   23,   26,   29,   32,   35,   37,   40,   43,
      45,   48,   50,   53,   55,   58,   60,   62,   65,   67,   69,   71,
      73,   75,   77,   79,   81,   82,   84,   86,   87,   89,   90,   92,
      93,   95,   96,   97,   98,   100,  101,  102,  103,  104,  105,  106,
      107,  108,  108,  109,  110,  111,  111,  112,  113,  113,  114,  114,
      115,  116,  116,  117,  117,  117,  118,  118,  119,  119,  119,  120,
      120,  120,  121,  121,  121,  121,  122,  122,  122,  122,  123,  123,
      123,  123,  123,  124,  124,  124,  124,  124,  124,  124,  124,  125,
      125,  125,  125,  125,  125,  125,  125,  125,  125,  125,  126,  126,
      126,  126,  126,  126,  126,  126,  126,  126,  126,  126,  126,  126,
      126,  126,  126,  127};

  return createIntLookupTable(name, input, mapping, outTy);
}

TopKNode *Function::createTopK(llvm::StringRef name, NodeValue input,
                               unsigned_t k) {
  auto inDims = input.dims();
  assert(inDims.size() > 0);
  assert(k <= inDims.back());
  ShapeVector outDims(inDims.begin(), inDims.end());
  outDims.back() = k;
  auto OT = getParent()->uniqueTypeWithNewShape(input.getType(), outDims);
  return addNode(new TopKNode(
      name, OT, getParent()->uniqueType(ElemKind::Int64ITy, outDims), input,
      k));
}

GatherNode *Function::createGather(llvm::StringRef name, NodeValue data,
                                   NodeValue indices, unsigned_t batchDims) {

  auto dDims = data.dims();
  auto iDims = indices.dims();
  assert(dDims.size() > batchDims);
  ShapeVector outDims;
  outDims.insert(outDims.end(), dDims.begin(), dDims.begin() + batchDims);
  outDims.insert(outDims.end(), iDims.begin(), iDims.end());
  outDims.insert(outDims.end(), dDims.begin() + batchDims + 1, dDims.end());
  return addNode(new GatherNode(
      name, getParent()->uniqueTypeWithNewShape(data.getType(), outDims), data,
      indices, batchDims));
}

ScatterAssignNode *Function::createScatterAssign(llvm::StringRef name,
                                                 NodeValue data,
                                                 NodeValue indices,
                                                 NodeValue slices) {
  return addNode(new ScatterAssignNode(name, data, indices, slices));
}

QuantizeNode *Function::createQuantize(llvm::StringRef name, NodeValue input,
                                       TypeRef outTy) {
  assert(input.getElementType() == ElemKind::FloatTy &&
         "Input must be a floating type");
  assert(outTy->getElementType() == ElemKind::Int8QTy &&
         "Output must be a quantized type");
  assert(input.dims().equals(outTy->dims()) &&
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
  return addNode(new DequantizeNode(name, outTy, input));
}

RescaleQuantizedNode *Function::createRescaleQuantized(llvm::StringRef name,
                                                       NodeValue input,
                                                       TypeRef outTy) {
  assert(input.getElementType() == ElemKind::Int8QTy &&
         "Input must be a quantized type");
  assert(outTy->getElementType() == ElemKind::Int8QTy &&
         "Output must be a quantized type");
  assert(input.dims().equals(outTy->dims()) &&
         "Different dimensions for input and output");

  return addNode(
      new RescaleQuantizedNode(name, getParent()->uniqueType(*outTy), input));
}

Node *Function::createWeightedSum(llvm::StringRef name,
                                  llvm::ArrayRef<NodeValue> data,
                                  llvm::ArrayRef<NodeValue> weights) {
  assert(data.size() == weights.size() &&
         "Must have same number of data and weights.");
  assert(data.size() > 0 && "No inputs provided.");

  const auto *outTy = data[0].getType();

  // Create a zero splat to bootstrap the adding chain.
  Node *currAdd = createSplat(name.str() + ".splat", outTy, 0.);

  for (size_t i = 0, e = data.size(); i < e; i++) {
    assert(weights[i].getType()->size() == 1 &&
           "Each provided weight node must be size 1.");
    assert(outTy == data[i].getType() &&
           "All data nodes must have the same type.");

    // Broadcast the current weight to same shape as the data.
    auto *bcastW =
        createBroadcast(name.str() + ".bcastWeight" + std::to_string(i),
                        weights[i], outTy->dims(), /* axis */ 0);

    // Element-wise multiply the broadcasted weight by the data.
    auto *scaledD =
        createMul(name.str() + ".mul" + std::to_string(i), bcastW, data[i]);

    // Element-wise add the scaled data to the running total.
    currAdd =
        createAdd(name.str() + ".add" + std::to_string(i), scaledD, currAdd);
  }

  // Return the final weighted sum via the last add in the chain.
  return currAdd;
}

Node *Function::createBatchBoxCox(llvm::StringRef name, NodeValue data,
                                  NodeValue lambda1, NodeValue lambda2) {
  assert((lambda1.dims() == lambda2.dims()) &&
         "lambda1 and lambda2 must have the same shape");
  assert((lambda1.getType()->getElementType() == lambda2.getElementType()) &&
         "lambda1 and lambda2 must have the same element type");
  assert((lambda1.getType()->getElementType() == data.getElementType()) &&
         "data and lambdas must have the same element type");
  assert((lambda1.dims().size() == 1) && "lambda1 and lambda2 must be vectors");
  assert((data.dims().size() == 2) && "data must be a matrix");
  assert((data.dims()[1] == lambda1.dims()[0]) &&
         "data, lambda1 and lambda2 must have the same number of rows");

  // Broadcast lambda1 and lambda2 so that they are both the same size as the
  // data.
  auto *BL1 = createBroadcast(name.str() + ".broadcast", lambda1, data.dims(),
                              /*axis=*/1);
  auto *BL2 = createBroadcast(name.str() + ".broadcast", lambda2, data.dims(),
                              /*axis=*/1);

  // Add a small epsilon to lambda1 so that we can avoid dividing by zero later.
  // It doesn't matter that this is technically incorrect because the final
  // Select will discard the results of this computation.
  auto *eps = createSplat(name.str() + ".eps", BL1->getNthResult(0).getType(),
                          std::numeric_limits<float>::min());
  auto *EBL1 = createAdd(name.str() + ".lambda1eps", BL1, eps);

  // Compute data + BL2, which is needed regardless of whether
  // lambda1 is 0 or not.
  auto *AN = createAdd(name.str() + ".add", data, BL2);

  // Take the max of data + BL2 and 1e-6 to void exponentiating or taking the
  // logarithm of too small a number.
  auto *minArg =
      createSplat(name.str() + ".logpowmin", AN->getResult().getType(), 1e-6);
  auto *MN = createMax(name.str() + ".max", AN, minArg);

  // Compute the Box-Cox transform for the lambda1 == 0 case:
  //    y = ln(max(x + lambda2, 1e-6))

  auto *LN = createLog(name.str() + ".log", MN);

  // Compute the Box-Cox transform for the lambda1 != 0 case:
  //    y = (max(x + lambda2, 1e-6)^lambda1 - 1)/lambda1
  auto *PN = createPow(name.str() + ".pow", MN, BL1);
  auto *ones =
      createSplat(name.str() + ".ones", PN->getResult().getType(), 1.0f);
  auto *SN = createSub(name.str() + ".sub", PN, ones);
  // Divide by EBL1, not BL1 to avoid a divide-by-zero exception.
  auto *DN = createDiv(name.str() + ".div", SN, EBL1);

  // Compute predicates for selecting between the two cases above.
  auto *zeroes =
      createSplat(name.str() + ".zeroes", BL1->getNthResult(0).getType(), 0.0f);
  auto *predicate = createCmpEQ(name.str() + ".cmpeq", BL1, zeroes);

  // Create Select to pick between the two Box-Cox cases.
  return createSelect(name.str() + ".select", predicate, LN, DN);
}

Node *Function::createClip(llvm::StringRef name, NodeValue input, float min,
                           float max) {
  auto *minSplat = createSplat(name.str() + ".minSplat", input.getType(), min);
  auto *minClipped = createMax(name.str() + ".minClip", input, minSplat);
  auto *maxSplat = createSplat(name.str() + ".maxSplat", input.getType(), max);
  auto result = createMin(name.str(), minClipped, maxSplat);
  return result;
}

//===----------------------------------------------------------------------===//
//                   Placeholder-builder methods.
//===----------------------------------------------------------------------===//

BatchNormalizationNode *
Function::createBatchNormalization(Context &ctx, llvm::StringRef name,
                                   NodeValue input, unsigned_t channelIdx,
                                   float epsilon, float momentum) {
  // Figure out how many channels are in the tensor.
  size_t channels = input.dims()[channelIdx];

  ElemKind inputTy = input.getType()->getElementType();

  // Allocate the learnable parameters beta and gamma.
  auto *beta =
      getParent()->createPlaceholder(inputTy, {channels}, "beta", true);
  ctx.allocate(beta)->init(glow::Tensor::InitKind::Zero, 0, getPRNG());

  auto *gamma =
      getParent()->createPlaceholder(inputTy, {channels}, "gamma", true);

  ctx.allocate(gamma)->init(glow::Tensor::InitKind::Broadcast, 1.0, getPRNG());

  auto *mean =
      getParent()->createPlaceholder(inputTy, {channels}, "mean", false);
  ctx.allocate(mean)->zero();

  auto *variance =
      getParent()->createPlaceholder(inputTy, {channels}, "variance", false);
  ctx.allocate(variance)->zero();

  return createBatchNormalization(name, input, beta, gamma, mean, variance,
                                  channelIdx, epsilon, momentum);
}

ConvolutionNode *Function::createConv(Context &ctx, llvm::StringRef name,
                                      NodeValue input, size_t depth,
                                      llvm::ArrayRef<unsigned_t> kernels,
                                      llvm::ArrayRef<unsigned_t> strides,
                                      llvm::ArrayRef<unsigned_t> pads,
                                      unsigned_t group) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  ShapeHW kdim(kernels);
  PaddingTLBR pdim(pads);
  (void)pdim;
  assert((idim.w + pdim.left + pdim.right) >= kdim.width &&
         (idim.h + pdim.top + pdim.bottom) >= kdim.height &&
         "buffer too small for selected stride");

  assert(group > 0 && "group should be larger than 0");
  assert(idim.c % group == 0 && "channels number must be divisible by groups");
  assert(depth % group == 0 && "depth must be divisible by groups");

  // Calculate the size and allocate the output buffer.
  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);

  std::array<size_t, 4> outDims = {{idim.n, outSz.first, outSz.second, depth}};

  // Allocate the Filter and Bias tensors.
  std::array<size_t, 4> filterDim = {
      {depth, kdim.height, kdim.width, idim.c / group}};
  size_t fanIn = kdim.height * kdim.width * idim.c;
  ElemKind inputTy = input.getType()->getElementType();
  assert((inputTy == ElemKind::FloatTy || inputTy == ElemKind::Float16Ty) &&
         "Convolution on non-floating point type?");
  auto *filter =
      getParent()->createPlaceholder(inputTy, filterDim, "filter", true);
  ctx.allocate(filter)->init(glow::Tensor::InitKind::Xavier, fanIn, getPRNG());

  auto *bias = getParent()->createPlaceholder(inputTy, {depth}, "bias", true);
  ctx.allocate(bias)->init(glow::Tensor::InitKind::Broadcast, 0.1, getPRNG());

  auto OT = getParent()->uniqueType(inputTy, outDims);

  return addNode(new ConvolutionNode(name, OT, input, filter, bias, kernels,
                                     strides, pads, group));
}

ConvolutionNode *Function::createConv(Context &ctx, llvm::StringRef name,
                                      NodeValue input, size_t depth,
                                      unsigned_t kernel, unsigned_t stride,
                                      unsigned_t pad, unsigned_t group) {
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  return createConv(ctx, name, input, depth, kernels, strides, pads, group);
}

ConvertToNode *Function::createConvertTo(llvm::StringRef name, NodeValue input,
                                         TypeRef outTy) {
  return addNode(new ConvertToNode(name, outTy, input));
}

FullyConnectedNode *Function::createFullyConnected(Context &ctx,
                                                   llvm::StringRef name,
                                                   NodeValue input,
                                                   size_t outDepth) {
  TypeRef T = input.getType();
  auto idim = flattenCdr(input.dims());
  size_t fanIn = idim.second;

  auto *W = getParent()->createPlaceholder(
      T->getElementType(), {idim.second, outDepth}, "weights", true);
  auto *B = getParent()->createPlaceholder(T->getElementType(), {outDepth},
                                           "bias", true);

  ctx.allocate(W)->init(Tensor::InitKind::Xavier, fanIn, getPRNG());
  ctx.allocate(B)->init(Tensor::InitKind::Broadcast, .1, getPRNG());

  auto OT =
      getParent()->uniqueType(T->getElementType(), {idim.first, outDepth});
  return addNode(new FullyConnectedNode(name, OT, input, W, B));
}

Node *Function::createDotProduct(llvm::StringRef name, NodeValue X,
                                 NodeValue Y) {
  auto XDimsSize = X.dims().size();
  (void)XDimsSize;

  assert(X.dims() == Y.dims() && "X and Y must have the same shape");
  assert(((XDimsSize == 1) || (XDimsSize == 2)) && "X and Y must be 1D or 2D");

  // Create Mul node.
  auto *MN = createMul(name.str() + ".mul", X, Y);

  // If X and Y are 1D, the BatchedReduceAdd node is not needed.
  if (XDimsSize == 1) {
    return MN;
  }

  // Create and return BatchedReduceAdd node.
  return createBatchedReduceAdd(name.str() + ".bra", MN, 1);
}

void Function::createGRU(Context &ctx, llvm::StringRef namePrefix,
                         llvm::ArrayRef<Node *> inputs, unsigned batchSize,
                         unsigned hiddenSize, unsigned outputSize,
                         std::vector<NodeValue> &outputs) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front()->dims(0).back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the state to zero.
  Placeholder *HInit = getParent()->createPlaceholder(
      ElemKind::FloatTy, {batchSize, hiddenSize}, "initial_state", false);
  ctx.allocate(HInit)->zero();
  Node *Ht = HInit;

  // Update gate:
  //    Z <- sigmoid(Wxz * x + Whz * h + bz)
  // Reset gate:
  //    R <- sigmoid(Wxr * x + Whr * h + br)
  // Hidden state:
  //    h <- Z . h + (1 - Z) tanh (Wxh * x + Whh * (R . h) + bh)

  // update gate
  float bUpdate = 0.1;
  Placeholder *Wxz = getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxz", true);
  Placeholder *Whz = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whz", true);
  Placeholder *Bz1 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bz1", true);
  Placeholder *Bz2 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bz2", true);

  ctx.allocate(Wxz)->init(glow::Tensor::InitKind::Xavier, inputSize, getPRNG());
  ctx.allocate(Whz)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(Bz1)->init(glow::Tensor::InitKind::Broadcast, bUpdate,
                          getPRNG());
  ctx.allocate(Bz2)->init(glow::Tensor::InitKind::Broadcast, bUpdate,
                          getPRNG());

  // Reset gate.
  float bReset = -1.0;
  Placeholder *Wxr = getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxr", true);
  Placeholder *Whr = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whr", true);
  Placeholder *Br1 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".br1", true);
  Placeholder *Br2 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".br2", true);

  ctx.allocate(Wxr)->init(glow::Tensor::InitKind::Xavier, inputSize, getPRNG());
  ctx.allocate(Whr)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(Br1)->init(glow::Tensor::InitKind::Broadcast, bReset, getPRNG());
  ctx.allocate(Br2)->init(glow::Tensor::InitKind::Broadcast, bReset, getPRNG());

  // hidden state
  float b = 0.1;
  Placeholder *Wxh = getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxh", true);
  Placeholder *Whh = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whh", true);
  Placeholder *Bh1 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bh1", true);
  Placeholder *Bh2 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bh2", true);

  ctx.allocate(Wxh)->init(glow::Tensor::InitKind::Xavier, inputSize, getPRNG());
  ctx.allocate(Whh)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(Bh1)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());
  ctx.allocate(Bh2)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());

  // Output Layer.
  Placeholder *Why = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, outputSize}, nameBase + ".Why", true);
  Placeholder *By = getParent()->createPlaceholder(
      ElemKind::FloatTy, {outputSize}, nameBase + ".by", true);

  ctx.allocate(Why)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(By)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());

  auto ty = getParent()->uniqueType(ElemKind::FloatTy, {batchSize, hiddenSize});
  auto *Ones = createSplat(nameBase + ".ones", ty, 1.0);

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
}

void Function::createSimpleRNN(Context &ctx, llvm::StringRef namePrefix,
                               llvm::ArrayRef<Node *> inputs,
                               unsigned batchSize, unsigned hiddenSize,
                               unsigned outputSize,
                               std::vector<NodeValue> &outputs) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front()->dims(0).back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the state to zero.
  Placeholder *HInit =
      getParent()->createPlaceholder(ElemKind::FloatTy, {batchSize, hiddenSize},
                                     nameBase + ".initial_state", false);
  ctx.allocate(HInit)->zero();
  Node *Ht = HInit;

  float b = 0.1;
  Placeholder *Whh = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whh", true);
  Placeholder *Bhh = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bhh", true);
  Placeholder *Wxh = getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxh", true);

  Placeholder *Bxh = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bxh", true);
  Placeholder *Why = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, outputSize}, nameBase + ".Why", true);
  Placeholder *Bhy = getParent()->createPlaceholder(
      ElemKind::FloatTy, {outputSize}, nameBase + ".Bhy", true);

  ctx.allocate(Whh)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(Bhh)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());
  ctx.allocate(Wxh)->init(glow::Tensor::InitKind::Xavier, inputSize, getPRNG());
  ctx.allocate(Bxh)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());
  ctx.allocate(Why)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(Bhy)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());

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

void Function::createLSTM(Context &ctx, llvm::StringRef namePrefix,
                          llvm::ArrayRef<Node *> inputs, unsigned batchSize,
                          unsigned hiddenSize, unsigned outputSize,
                          std::vector<NodeValue> &outputs) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front()->dims(0).back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the hidden and cell states to zero.
  Placeholder *HInit =
      getParent()->createPlaceholder(ElemKind::FloatTy, {batchSize, hiddenSize},
                                     "initial_hidden_state", false);
  ctx.allocate(HInit)->zero();
  Node *Ht = HInit;

  Placeholder *CInit = getParent()->createPlaceholder(
      ElemKind::FloatTy, {batchSize, hiddenSize}, "initial_cell_state", false);
  ctx.allocate(CInit)->zero();
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
  Placeholder *Wxf = getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxf", true);
  Placeholder *Whf = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whf", true);
  Placeholder *Bf1 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bf1", true);
  Placeholder *Bf2 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bf2", true);
  ctx.allocate(Wxf)->init(glow::Tensor::InitKind::Xavier, inputSize, getPRNG());
  ctx.allocate(Whf)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(Bf1)->init(glow::Tensor::InitKind::Broadcast, bForget,
                          getPRNG());
  ctx.allocate(Bf2)->init(glow::Tensor::InitKind::Broadcast, bForget,
                          getPRNG());

  // input gate
  float bInput = 0.1;
  Placeholder *Wxi = getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxi", true);
  Placeholder *Whi = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whi", true);
  Placeholder *Bi1 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bi1", true);
  Placeholder *Bi2 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bi2", true);

  ctx.allocate(Wxi)->init(glow::Tensor::InitKind::Xavier, inputSize, getPRNG());
  ctx.allocate(Whi)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(Bi1)->init(glow::Tensor::InitKind::Broadcast, bInput, getPRNG());
  ctx.allocate(Bi2)->init(glow::Tensor::InitKind::Broadcast, bInput, getPRNG());

  // output gate
  float bOutput = 0.1;
  Placeholder *Wxo = getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxo", true);
  Placeholder *Who = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Who", true);
  Placeholder *Bo1 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bo1", true);
  Placeholder *Bo2 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bo2", true);

  ctx.allocate(Wxo)->init(glow::Tensor::InitKind::Xavier, inputSize, getPRNG());
  ctx.allocate(Who)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(Bo1)->init(glow::Tensor::InitKind::Broadcast, bOutput,
                          getPRNG());
  ctx.allocate(Bo2)->init(glow::Tensor::InitKind::Broadcast, bOutput,
                          getPRNG());

  // cell state
  float bCell = 0.1;
  Placeholder *Wxc = getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wxc", true);
  Placeholder *Whc = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whc", true);
  Placeholder *Bc1 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bc1", true);
  Placeholder *Bc2 = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".bc2", true);

  ctx.allocate(Wxc)->init(glow::Tensor::InitKind::Xavier, inputSize, getPRNG());
  ctx.allocate(Whc)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(Bc1)->init(glow::Tensor::InitKind::Broadcast, bCell, getPRNG());
  ctx.allocate(Bc2)->init(glow::Tensor::InitKind::Broadcast, bCell, getPRNG());

  // output layer
  float b = 0.1;
  Placeholder *Why = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, outputSize}, nameBase + ".Why", true);
  Placeholder *By = getParent()->createPlaceholder(
      ElemKind::FloatTy, {outputSize}, nameBase + ".by", true);

  ctx.allocate(Why)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                          getPRNG());
  ctx.allocate(By)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());

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
  for (auto &n : nodes_) {
    llvm::outs() << n.getDebugDesc() << "\n";
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
      edge << uniqueVertexName(pred) << ":"
           << pred.getNode()->getOutputName(resNo).str() << " -> "
           << uniqueVertexName(N) << ":w";
      dumpEdgeStyle(N, 0, pred, edge);
      edges_.insert(edge.str());
      visitNode(pred);
    }

    for (size_t i = 0; i < N->getNumInputs(); i++) {
      Node *to = N->getNthInput(i).getNode();
      size_t resNo = N->getNthInput(i).getResNo();

      std::ostringstream edge;
      edge << uniqueVertexName(to) << ":" << to->getOutputName(resNo).str()
           << " -> " << uniqueVertexName(N) << ":" << N->getInputName(i);
      dumpEdgeStyle(N, i, to, edge);
      edges_.insert(edge.str());

      visitNode(to);
    }
  }

public:
  void visitGraph(Function *F) {
    for (auto &N : F->getNodes()) {
      visitNode(&N);
    }

    for (auto N : visitedNodes_) {
      dumpNode(N);
    }
  }
};

void Function::dumpDAG() {
  llvm::SmallString<64> dotPath;
  llvm::sys::fs::createTemporaryFile("dotty_graph_dump", "dot", dotPath);
  dumpDAG(dotPath);
}

void Function::dumpDAG(llvm::StringRef dotFilename) {
  llvm::outs() << "Writing dotty graph for Function to: " << dotFilename
               << '\n';

  FunctionDottyPrinter DP;

  DP.visitGraph(this);

  std::ofstream myfile;
  myfile.open(dotFilename);
  DP.dumpAll(myfile);
  myfile.close();
}

void Function::dumpDAG(const char *dotFilename) {
  dumpDAG(llvm::StringRef(dotFilename));
}

Node *Function::getNodeByName(llvm::StringRef name) {
  for (auto &N : getNodes()) {
    if (N.getName().equals(name)) {
      return &N;
    }
  }
  return nullptr;
}

void Module::eraseConstant(ConstList::iterator I) {
  if (I == constants_.end())
    return;
  delete *I;
  constants_.erase(I);
}

void Function::eraseNode(NodesList::iterator I) { nodes_.erase(I); }

Constant *Module::getConstantByName(llvm::StringRef name) {
  for (auto *V : getConstants()) {
    if (V->getName() == name)
      return V;
  }
  return nullptr;
}

Placeholder *Module::getPlaceholderByName(llvm::StringRef name) {
  for (auto *P : getPlaceholders()) {
    if (P->getName() == name) {
      return P;
    }
  }

  return nullptr;
}

void Module::eraseConstant(Constant *N) {
  auto &vars = getConstants();
  auto I = std::find(vars.begin(), vars.end(), N);
  eraseConstant(I);
}

void Function::eraseNode(Node *N) {
  if (Constant *V = dyn_cast<Constant>(N)) {
    return getParent()->eraseConstant(V);
  }
  assert(std::find_if(nodes_.begin(), nodes_.end(),
                      [N](const Node &node) -> bool { return &node == N; }) !=
             nodes_.end() &&
         "Could not find node to delete!");
  eraseNode(N->getIterator());
}

Function *Function::clone(llvm::StringRef newName,
                          llvm::DenseMap<Node *, Node *> *map) {
  Module *M = getParent();
  auto *newF = M->createFunction(newName);

  // Maps current nodes to new nodes.
  llvm::DenseMap<Node *, Node *> currToNew;

  // Clone all of the nodes in the function.
  for (auto &N : getNodes()) {
    Node *copy = N.clone();
    // Record the copy relationship between the graphs.
    currToNew[&N] = copy;
    newF->addNode(copy);
  }

  // At this point we have a new invalid function that points into nodes in the
  // original function. Here we update the links between the nodes in the new
  // function.
  for (auto &N : newF->getNodes()) {
    // Fix each one of the inputs of this node.
    for (unsigned inp = 0, e = N.getNumInputs(); inp < e; inp++) {
      auto input = N.getNthInput(inp);

      auto it = currToNew.find(input.getNode());
      if (it == currToNew.end()) {
        assert(isa<Storage>(input.getNode()) &&
               "Could not find a mapping for some node!");
        continue;
      }

      // Update the node with the edge to the current graph.
      N.setNthInput(inp, NodeValue(it->second, input.getResNo()));
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

/// Verify the input \p idx of a node \p N. Check that the node \p N is in the
/// use-list of the corresponding input node.
static void verifyNodeInput(const Node &N, size_t idx) {
  auto input = N.getNthInput(idx);
  auto *refN = input.getNode();
  // Check that N is in the use-list of the input node and there is a proper
  // entry for it.
  for (auto &U : refN->getUsers()) {
    if (U.getUser() == &N && *U.get() == input) {
      return;
    }
  }
  llvm_unreachable(
      "Any node referencing another node N be in the use-list of the node N");
}

/// \returns True if \p n is a storage node (constant or placeholder) of the
/// function \p F.
static bool isGraphStorageNode(Node *n, const Function *F) {
  auto &vars = F->getParent()->getConstants();
  auto &placeholders = F->getParent()->getPlaceholders();

  if (Constant *V = dyn_cast<Constant>(n)) {
    return std::find(vars.begin(), vars.end(), V) != vars.end();
  }

  if (Placeholder *P = dyn_cast<Placeholder>(n)) {
    return std::find(placeholders.begin(), placeholders.end(), P) !=
           placeholders.end();
  }

  return false;
}

void Function::verify() const {
  std::unordered_map<std::string, const Node *> NameToNode;

  for (auto *V : getParent()->getConstants()) {
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

  NameToNode.clear();

  for (auto &N : nodes_) {
    if (NameToNode.insert({N.getName(), &N}).second)
      continue;
    /// Output extra information helping to find the error.
    llvm::outs() << "The node with name '" << N.getName()
                 << "' conflicts with a previous definition:\n";
    llvm::errs() << "Current definition: " << N.getDebugDesc() << "\n";
    llvm::errs() << "Previous definition: "
                 << NameToNode[N.getName()]->getDebugDesc() << "\n";
    dump();
    llvm_unreachable("Multiple nodes with the same name");
  }

  // Any node referenced by one of the graph nodes should be part of the Graph.
  for (const auto &N : nodes_) {
    for (size_t idx = 0, e = N.getNumInputs(); idx < e; ++idx) {
      auto &input = N.getNthInput(idx);
      (void)input;
      // Verify each input of N.
      verifyNodeInput(N, idx);
      bool foundNode =
          std::find(nodes_.begin(), nodes_.end(), *input) != nodes_.end();
      (void)foundNode;
      (void)isGraphStorageNode;
      assert((foundNode || isGraphStorageNode(input, this)) &&
             "Every node referenced by one of the graph nodes should be part of"
             "the graph");
    }
  }

  // Check that all uses of each node refer to this node.
  for (const auto &N : nodes_) {
    for (const auto &U : N.getUsers()) {
      (void)U;
      assert(U.get()->getNode() == &N &&
             "All uses of a node should refer to this node");
    }
  }

  std::unordered_map<const Placeholder *, const Node *> placeholderWrittenTo;
  for (const auto &N : nodes_) {
    assert(N.getParent() == this &&
           "Node is not linked to the function it belongs");
    N.verify();
    // Make sure all the placeholders are at most written once, and that
    // constants are never written to.
    for (size_t idx = 0, e = N.getNumInputs(); idx < e; ++idx) {
      // Placeholders and Constants have no input, so they can only be
      // written to via overwritten inputs.
      if (!N.isOverwrittenNthInput(idx)) {
        continue;
      }

      const Node *nthInputNode = N.getNthInput(idx).getNode();
      assert(!isa<Constant>(nthInputNode) &&
             "Constants can never be used as an overwritten input.");

      // Unlike Constants, Placeholders can be used at most once as overwritten
      // inputs. Keep a map of Placeholders to Nodes that used them as
      // overwritten inputs, which is also used later to check for
      // read-after-write dependence violations.
      const auto *ph = dyn_cast<Placeholder>(nthInputNode);
      if (!ph) {
        continue;
      }
      auto varToFirstDef = placeholderWrittenTo.find(ph);
      if (varToFirstDef != placeholderWrittenTo.end()) {
        llvm::errs() << "Placeholder " << ph->getDebugDesc() << '\n';
        llvm::errs() << "has more than one write:\n";
        llvm::errs() << N.getDebugDesc() << '\n';
        llvm::errs() << varToFirstDef->second->getDebugDesc() << '\n';
      }
      assert(varToFirstDef == placeholderWrittenTo.end() &&
             "Placeholder has more than one write");
      placeholderWrittenTo[ph] = &N;
    }
  }

  // Now check that the placeholders that are written to are either:
  // - Written by a save node, or
  // - Are only used by the node that writes them
  // If this check fails, that means we have implicit memory
  // dependencies that may not be honored by the scheduler.
  // Either the input IR is incorrect or the scheduler needs
  // fixing.
  for (const std::pair<const Placeholder *, const Node *> &varToWrite :
       placeholderWrittenTo) {
    if (isa<SaveNode>(varToWrite.second)) {
      continue;
    }
    for (const NodeUse &use : varToWrite.first->getUsers()) {
      const Node *user = use.getUser();
      // Ignore users outside this function.
      if (user->getParent() != this) {
        continue;
      }
      assert(user == varToWrite.second &&
             "Implicit read after write memory dependency may not be honored");
    }
  }
}
