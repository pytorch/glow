/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#include "glow/Backend/Backend.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Graph/TensorLayout.h"
#include "glow/Graph/VerifierHelper.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#ifdef WIN32
#include <corecrt_math_defines.h>
#endif
#include <fstream>
#include <unordered_set>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace {
/// A helper function to log the deletion of constant/placeholder \p s of a
/// module into the log context of given functions \p functions.
/// Note: The reason we don't log the deletion of constants in the function that
/// ueses or creates it, is that constants/placeholders do not have a function
/// parent (we can't utilize its user's function also because its users might be
/// removed) such that it's best to log the constants/placeholders in a Module
/// level log context and copy over to its all functions.
void logStorageDeletion(std::list<Function *> functions, Storage *s) {
  for (auto *F : functions) {
    F->getLogContext()->logNodeDeletion(*s);
  }
  if (functions.size() > 0) {
    auto *F = *(functions.begin());
    F->getLogContext()->logNodeDeletion(*s, /* logIntoModule */ true);
  }
}

/// A helper function to log the creation of constant/placeholder \p s of a
/// module into the log context of given functions \p functions.
/// Same note as for logStorageDeletion().
void logStorageCreation(std::list<Function *> functions, Storage *s) {
  for (auto *F : functions) {
    F->getLogContext()->logNodeCreation(*s);
  }
  if (functions.size() > 0) {
    auto *F = *(functions.begin());
    F->getLogContext()->logNodeCreation(*s, /* logIntoModule */ true);
  }
}
} // namespace

/// Merge shape \p shape into \p mergeShape, following multidirectional
/// broadcasting rules.
static void mergeMultidirectionalBroadcastHelper(std::vector<dim_t> &mergeShape,
                                                 llvm::ArrayRef<dim_t> shape) {
  size_t shift = mergeShape.size() - shape.size();
  for (size_t i = 0, e = shape.size(); i < e; i++) {
    if (shape[i] == 1) {
      // Just leave mergeShape[i] as it is.
      continue;
    }

    assert(
        ((shape[i] == mergeShape[shift + i]) || (mergeShape[shift + i] == 1)) &&
        "Incompatible dimension for the broadcast");
    mergeShape[shift + i] = shape[i];
  }
}

/// Utility function which computes the resulting shape in case of
/// multidirectional broadcasting.
static std::vector<dim_t>
computeMultidirectionalBroadcastHelper(llvm::ArrayRef<dim_t> shape0,
                                       llvm::ArrayRef<dim_t> shape1) {
  size_t numDims0 = shape0.size();
  size_t numDims1 = shape1.size();
  size_t newNumDims = std::max(numDims0, numDims1);
  std::vector<dim_t> reshapeDims(newNumDims, 1);

  mergeMultidirectionalBroadcastHelper(reshapeDims, shape0);
  mergeMultidirectionalBroadcastHelper(reshapeDims, shape1);

  return reshapeDims;
}

std::vector<NodeValue>
Function::broadcastInputs(int axis, const llvm::ArrayRef<NodeValue> inputs) {
  dim_t numInputs = inputs.size();

  if (axis > -1) {
    assert(
        numInputs == 2 &&
        "If axis is specified, not -1, unidirectional broadcast will be used, "
        "input size must be 2.");
    return {inputs[0],
            createBroadcast("broadcast_" + inputs[1].getNode()->getName().str(),
                            inputs[1], inputs[0].dims(), axis)};
  }

  assert(numInputs >= 2 && "Invalid input passed in to commonCreateBroadcast.");

  std::vector<dim_t> targetDim = computeMultidirectionalBroadcastHelper(
      inputs[0].dims(), inputs[1].dims());

  for (size_t i = 2; i < numInputs; ++i) {
    targetDim =
        computeMultidirectionalBroadcastHelper(targetDim, inputs[i].dims());
  }

  std::vector<NodeValue> out(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    NodeValue n = inputs[i];
    auto dims = n.dims();
    if (dims != llvm::ArrayRef<dim_t>(targetDim)) {
      unsigned axis = targetDim.size() - dims.size();
      out[i] = createBroadcast("broadcast_" + n.getNode()->getName().str(), n,
                               targetDim, axis);
    } else {
      out[i] = inputs[i];
    }
  }
  return out;
}

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

void Module::strip() {
  for (auto it = constants_.begin(), e = constants_.end(); it != e; it++) {
    Constant *v = *it;
    v->clearPayload();
  }
}

void Module::clear() {
  for (auto it = constants_.begin(), e = constants_.end(); it != e; it++) {
    Constant *v = *it;
    logStorageDeletion(functions_, v);
    delete v;
  }

  constants_.clear();

  for (auto it = placeholders_.begin(), e = placeholders_.end(); it != e;
       it++) {
    Placeholder *p = *it;
    logStorageDeletion(functions_, p);
    delete p;
  }

  eraseFunctions();

  placeholders_.clear();
}

Module::~Module() { clear(); }
bool Module::verify() const {
  bool isValid = true;
  for (auto *F : functions_) {
    isValid &= F->verify();
  }
  return isValid;
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

std::string Module::toString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dump(os);
  return os.str();
}

void Module::dump(llvm::raw_ostream &os) const {
  os << "Module structure:\n";
  for (auto v : getConstants()) {
    os << v->getDebugDesc() << "\n";
  }

  for (auto f : functions_) {
    os << "Function:" << f->getName() << "\n";
  }
}

/// A helper class for visiting and generating the dotty graph file.
class AbstractDottyPrinter {
protected:
  // List of generated vertices.
  std::vector<std::string> vertices_{};
  // List of generated edges.
  std::unordered_set<std::string> edges_{};
  // Map node addresses to unique numbers.
  using VertexNumberMap = std::unordered_map<void *, unsigned>;
  VertexNumberMap vertex_numbers{};

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

  void dumpNode(Node *N, bool uniqueNames) {
    if (!N) {
      return;
    }
    std::ostringstream os;
    // Print a node descriptor that looks like this:
    if (uniqueNames) {
      // vNNNN [ shape = "record" label = "{...}" ];
      os << uniqueVertexName(N) << "[\n";
    } else {
      // <name> [ shape = "record" label = "{...}" ];
      os << N->getName().str() << "[\n";
    }
    os << "\tlabel = \"";
    dumpLabel(N, os);
    os << "\"\n";
    os << "\tshape = \"record\"\n";
    os << "\tstyle=\"filled,rounded\"\n";

    // Pick a color based on the node kind.
    unsigned colorIdx = llvm::hash_value(llvm::StringRef(N->getKindName()));
    auto nodeColor = getDotFileNodeColor(colorIdx);

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
    VertexNumberMap::iterator i;
    bool inserted;
    std::tie(i, inserted) = vertex_numbers.insert(std::make_pair(N, 0u));
    if (inserted) {
      i->second = vertex_numbers.size() - 1;
    }

    std::string buffer;
    llvm::raw_string_ostream stream(buffer);
    stream << llvm::format("v%04u", i->second);
    return stream.str();
  }

public:
  void dumpAll(std::ostream &os) {
    CHECK(os) << "Failed to create file for to dump Graph";

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
    // vNNNN [ label = "{...}" ];
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
      dumpNode(N, true);
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

uint64_t Module::getConstantsSize() {
  uint64_t size = 0;
  for (auto *constant : constants_) {
    size += constant->getPayload().getSizeInBytes();
  }
  return size;
}

Function::~Function() {
  // Delete all of the nodes.
  for (auto it = nodes_.begin(), e = nodes_.end(); it != e;) {
    auto cur = it++;
    eraseNode(&*cur);
  }
}

TypeRef Module::uniqueType(ElemKind elemTy, llvm::ArrayRef<dim_t> dims) {
  return uniqueType(Type(elemTy, dims));
}

TypeRef Module::uniqueType(ElemKind elemTy, llvm::ArrayRef<dim_t> dims,
                           float scale, int32_t offset) {
  return uniqueType(Type(elemTy, dims, scale, offset));
}

TypeRef Module::uniqueTypeWithNewShape(TypeRef T, llvm::ArrayRef<dim_t> dims) {
  return uniqueType(Type::newShape(*T, dims));
}

TypeRef Module::uniqueTypeWithNewShape(TypeRef T, llvm::ArrayRef<dim_t> dims,
                                       llvm::ArrayRef<dim_t> alignments) {
  return uniqueType(Type::newShape(*T, dims, alignments));
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

/// \returns a ShapeVector of rank axes.size() less than the input \p dims,
/// where the provided \p axes dimensions are removed from the shape.
static ShapeVector getNewShapeWithoutAxes(llvm::ArrayRef<dim_t> dims,
                                          llvm::ArrayRef<unsigned_t> axes) {
  assert(axes.size() <= dims.size() &&
         "Cannot remove more dimensions than exist.");
  ShapeVector newDims(dims.begin(), dims.end());
  ShapeVector shapeAxes(axes.begin(), axes.end());

  // Sort so that looping erase below doesn't fail.
  std::sort(shapeAxes.rbegin(), shapeAxes.rend());

  for (const auto &axis : shapeAxes) {
    assert(axis <= dims.size() &&
           "Axis to remove must fit inside dimensions of the provided dims.");
    newDims.erase(newDims.begin() + axis);
  }
  return newDims;
}

//===----------------------------------------------------------------------===//
//                       Node builders
//===----------------------------------------------------------------------===//

Placeholder *Module::createPlaceholder(TypeRef T, llvm::StringRef name,
                                       bool isTrainable,
                                       const std::string &layout) {
  auto FT = uniqueType(*T);
  auto *ph = new Placeholder(name, FT, isTrainable, layout);
  ph->setName(uniqueName(ph->getName(), usedNodeNames_, usedStorageNames_));
  placeholders_.push_back(ph);
  logStorageCreation(functions_, ph);
  return ph;
}

Placeholder *Module::createPlaceholder(ElemKind T, llvm::ArrayRef<dim_t> dims,
                                       llvm::StringRef name, bool isTrainable,
                                       const std::string &layout) {
  auto FT = uniqueType(T, dims);
  return createPlaceholder(FT, name, isTrainable, layout);
}

Placeholder *Module::createPlaceholder(ElemKind T, llvm::ArrayRef<dim_t> dims,
                                       float scale, int32_t offset,
                                       llvm::StringRef name, bool isTrainable,
                                       const std::string &layout) {
  auto FT = uniqueType(T, dims, scale, offset);
  return createPlaceholder(FT, name, isTrainable, layout);
}

Constant *Module::createConstant(TypeRef T, llvm::StringRef name,
                                 const std::string &layout) {
  auto FT = uniqueType(*T);
  return addConstant(new Constant(name, FT, layout));
}

Constant *Module::createConstant(ElemKind T, llvm::ArrayRef<dim_t> dims,
                                 llvm::StringRef name,
                                 const std::string &layout) {
  auto FT = uniqueType(T, dims);
  return createConstant(FT, name, layout);
}

Constant *Module::createConstant(ElemKind T, llvm::ArrayRef<dim_t> dims,
                                 float scale, int32_t offset,
                                 llvm::StringRef name,
                                 const std::string &layout) {
  auto FT = uniqueType(T, dims, scale, offset);
  return createConstant(FT, name, layout);
}

Constant *Module::createConstant(llvm::StringRef name, const Tensor &tensor,
                                 const std::string &layout) {
  auto *V = createConstant(&tensor.getType(), name, layout);
  V->assign(&tensor);
  return V;
}

Constant *Module::createConstant(llvm::StringRef name, Tensor &&tensor,
                                 const std::string &layout) {
  return addConstant(new Constant(name, std::move(tensor), layout));
}

std::string Module::getPrefix(llvm::StringRef name) {
  std::string prefix = name;
  size_t delim = name.rfind("__");
  if (delim != std::string::npos &&
      std::all_of(name.begin() + (delim + 2), name.end(),
                  [](unsigned char c) { return ::isdigit(c); })) {
    prefix = prefix.substr(0, delim);
  }
  return prefix;
}

llvm::StringRef Module::uniqueName(llvm::StringRef name,
                                   const llvm::StringSet<> &stringTable,
                                   llvm::StringSet<> &updateTable) {
  std::string legalName = legalizeName(name);
  if (stringTable.find(legalName) == stringTable.end()) {
    auto it = updateTable.insert(legalName);
    if (it.second) {
      return it.first->first();
    }
  }

  std::string prefix = Module::getPrefix(legalName);
  for (unsigned i = 1; i < 10000; i++) {
    auto suffix = std::to_string(i);
    std::string fullName = prefix + "__" + suffix;
    if (stringTable.find(fullName) != stringTable.end()) {
      continue;
    }

    auto it = updateTable.insert(fullName);
    if (it.second) {
      return it.first->first();
    }
  }
  llvm_unreachable("Unable to find a unique a name.");
}

Constant *Module::addConstant(Constant *V) {
  V->setName(uniqueName(V->getName(), usedNodeNames_, usedStorageNames_));
  // Replace the Constant's output type with the equivalent unique type for
  // this Module to maintain the invariant that each type in the Module is
  // unique.
  V->setType(Constant::ResultIndices::OutputIdx, uniqueType(*V->getType()));
  constants_.push_back(V);
  logStorageCreation(functions_, V);
  return V;
}

/// Check if the 'pads' array has the right size.
static void assertPadsSize(NodeValue input, llvm::ArrayRef<int> pads) {
  assert((pads.size() == 2 * input.dims().size()) &&
         "the pads array must contain 2 values per dimensions");
}

PadNode *Function::createPad(llvm::StringRef name, NodeValue input,
                             TypeRef outTy, unsigned_t mode,
                             llvm::ArrayRef<int> pads, float value) {
  assertPadsSize(input, pads);
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new PadNode(name, OT, input, mode, pads, value));
}

/// Check the kernel size for Conv/Pooling ops.
static void checkKernelSize(ShapeNHWC idim, llvm::ArrayRef<unsigned_t> kernels,
                            llvm::ArrayRef<unsigned_t> pads) {
  PaddingTLBR pdim(pads);
  (void)pdim;
  ShapeHW kdim(kernels);
  (void)kdim;
  assert((idim.w + pdim.left + pdim.right) >= kdim.width &&
         (idim.h + pdim.top + pdim.bottom) >= kdim.height &&
         "Kernel size is too large");
}

/// Check the kernel size for 3D Conv/Pooling ops.
static void check3DKernelSize(ShapeNHWDC idim,
                              llvm::ArrayRef<unsigned_t> kernels,
                              llvm::ArrayRef<unsigned_t> pads) {
  PaddingTLNBRF pdim(pads);
  (void)pdim;
  ShapeHWD kdim(kernels);
  (void)kdim;
  assert((idim.w + pdim.left + pdim.right) >= kdim.width &&
         (idim.h + pdim.top + pdim.bottom) >= kdim.height &&
         (idim.d + pdim.near + pdim.far) >= kdim.depth &&
         "Kernel size is too large");
}

/// Check that the dimensions that are passed in when the convolution is
/// constructed are correct.
static void assertConvDims(NodeValue input, NodeValue filter, NodeValue bias,
                           llvm::ArrayRef<unsigned_t> kernels,
                           llvm::ArrayRef<unsigned_t> strides,
                           llvm::ArrayRef<unsigned_t> pads, unsigned_t group) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  ShapeHW kdim(kernels);
  (void)kdim;
  checkKernelSize(idim, kernels, pads);
  assert(idim.c % group == 0 && "channels number must be divisible by groups");

  // NOTE: here the N in NHWC is abnormal because it is the number of filters
  // (and therefore the number of output channels of the conv) and not the
  // batch size. The rest of the dimensions are representative of the input
  // dimensions to the convolution.
  ShapeNHWC filterDims(filter.dims());
  (void)filterDims;

  assert(filterDims.n % group == 0 && filterDims.h == kdim.height &&
         filterDims.w == kdim.width && filterDims.c == idim.c / group &&
         "Invalid filter dims");

  assert(bias.getType()->size() == filterDims.n && "Invalid bias size");
}

/// Check that the dimensions that are passed in when the 3D convolution is
/// constructed are correct.
static void assertConv3DDims(NodeValue input, NodeValue filter, NodeValue bias,
                             llvm::ArrayRef<unsigned_t> kernels,
                             llvm::ArrayRef<unsigned_t> strides,
                             llvm::ArrayRef<unsigned_t> pads,
                             unsigned_t group) {
  ShapeNHWDC idim(input.dims());
  ShapeHWD kdim(kernels);
  (void)kdim;
  check3DKernelSize(idim, kernels, pads);
  assert(idim.c % group == 0 && "channels number must be divisible by groups");

  // NOTE: here the N in NHWDC is abnormal because it is the number of filters
  // (and therefore the number of output channels of the 3d conv) and not the
  // batch size. The rest of the dimensions are representative of the input
  // dimensions to the convolution.
  ShapeNHWDC filterDims(filter.dims());
  (void)filterDims;

  assert(filterDims.n % group == 0 && filterDims.h == kdim.height &&
         filterDims.w == kdim.width && filterDims.d == kdim.depth &&
         filterDims.c == idim.c / group && "Invalid filter dims");

  assert(bias.getType()->size() == filterDims.n && "Invalid bias size");
}

ConvolutionNode *Function::createConv(
    llvm::StringRef name, NodeValue input, NodeValue filter, NodeValue bias,
    TypeRef outTy, llvm::ArrayRef<unsigned_t> kernels,
    llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
    unsigned_t group, unsigned_t dilation, ConvolutionLayout layout) {
  assertConvDims(input, filter, bias, kernels, strides, pads, group);
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new ConvolutionNode(name, OT, input, filter, bias, kernels,
                                     strides, pads, group, dilation, layout,
                                     FusedActivation::NONE));
}

ConvolutionNode *Function::createConv(llvm::StringRef name, NodeValue input,
                                      NodeValue filter, NodeValue bias,
                                      TypeRef outTy, unsigned_t kernel,
                                      unsigned_t stride, unsigned_t pad,
                                      unsigned_t group, unsigned_t dilation,
                                      ConvolutionLayout layout) {
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  return createConv(name, input, filter, bias, outTy, kernels, strides, pads,
                    group, dilation, layout);
}

Convolution3DNode *Function::createConv3D(llvm::StringRef name, NodeValue input,
                                          NodeValue filter, NodeValue bias,
                                          TypeRef outTy,
                                          llvm::ArrayRef<unsigned_t> kernels,
                                          llvm::ArrayRef<unsigned_t> strides,
                                          llvm::ArrayRef<unsigned_t> pads,
                                          unsigned_t group) {
  assertConv3DDims(input, filter, bias, kernels, strides, pads, group);
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new Convolution3DNode(name, OT, input, filter, bias, kernels,
                                       strides, pads, group));
}

Convolution3DNode *Function::createConv3D(llvm::StringRef name, NodeValue input,
                                          NodeValue filter, NodeValue bias,
                                          TypeRef outTy, unsigned_t kernel,
                                          unsigned_t stride, unsigned_t pad,
                                          unsigned_t group) {
  llvm::SmallVector<unsigned_t, 6> pads = {pad, pad, pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 3> strides = {stride, stride, stride};
  llvm::SmallVector<unsigned_t, 3> kernels = {kernel, kernel, kernel};
  return createConv3D(name, input, filter, bias, outTy, kernels, strides, pads,
                      group);
}

MaxPoolNode *Function::createMaxPool(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<unsigned_t> kernels,
                                     llvm::ArrayRef<unsigned_t> strides,
                                     llvm::ArrayRef<unsigned_t> pads,
                                     ConvolutionLayout layout) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  checkKernelSize(idim, kernels, pads);

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  auto OT = getParent()->uniqueTypeWithNewShape(
      input.getType(), {idim.n, outSz.first, outSz.second, idim.c});
  auto AMT = getParent()->uniqueType(
      IndexElemKind, {idim.n, outSz.first, outSz.second, idim.c});

  return addNode(
      new MaxPoolNode(name, OT, AMT, input, kernels, strides, pads, layout));
}

MaxPoolNode *Function::createMaxPool(llvm::StringRef name, NodeValue input,
                                     unsigned_t kernel, unsigned_t stride,
                                     unsigned_t pad, ConvolutionLayout layout) {
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  return createMaxPool(name, input, kernels, strides, pads, layout);
}

AvgPoolNode *Function::createAvgPool(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<unsigned_t> kernels,
                                     llvm::ArrayRef<unsigned_t> strides,
                                     llvm::ArrayRef<unsigned_t> pads,
                                     ConvolutionLayout layout) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  checkKernelSize(idim, kernels, pads);

  auto outSz =
      calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
  auto OT = getParent()->uniqueTypeWithNewShape(
      input.getType(), {idim.n, outSz.first, outSz.second, idim.c});

  return addNode(
      new AvgPoolNode(name, OT, input, kernels, strides, pads, layout));
}

AvgPoolNode *Function::createAvgPool(llvm::StringRef name, NodeValue input,
                                     TypeRef outTy,
                                     llvm::ArrayRef<unsigned_t> kernels,
                                     llvm::ArrayRef<unsigned_t> strides,
                                     llvm::ArrayRef<unsigned_t> pads,
                                     ConvolutionLayout layout) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  ShapeHW kdim(kernels);
  (void)kdim;
  checkKernelSize(idim, kernels, pads);
  return addNode(
      new AvgPoolNode(name, outTy, input, kernels, strides, pads, layout));
}

AvgPoolNode *Function::createAvgPool(llvm::StringRef name, NodeValue input,
                                     unsigned_t kernel, unsigned_t stride,
                                     unsigned_t pad, ConvolutionLayout layout) {
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  return createAvgPool(name, input, kernels, strides, pads, layout);
}

AdaptiveAvgPoolNode *Function::createAdaptiveAvgPool(llvm::StringRef name,
                                                     NodeValue input,
                                                     TypeRef outTy) {
  return addNode(new AdaptiveAvgPoolNode(name, outTy, input));
}

FullyConnectedNode *Function::createFullyConnected(llvm::StringRef name,
                                                   NodeValue input, Storage *W,
                                                   Storage *B,
                                                   unsigned_t axis) {
  TypeRef T = input.getType();
  TypeRef OT = getParent()->uniqueTypeWithNewShape(
      T, {input.dims()[0], B->getType()->dims()[0]});

  return createFullyConnected(name, input, W, B, OT, axis);
}

FullyConnectedNode *Function::createFullyConnected(llvm::StringRef name,
                                                   NodeValue input, NodeValue W,
                                                   NodeValue B,
                                                   unsigned_t axis) {
  TypeRef T = input.getType();
  TypeRef OT =
      getParent()->uniqueTypeWithNewShape(T, {input.dims()[0], B.dims()[0]});

  return createFullyConnected(name, input, W, B, OT, axis);
}

FullyConnectedNode *Function::createFullyConnected(llvm::StringRef name,
                                                   NodeValue input, NodeValue W,
                                                   NodeValue B, TypeRef outTy,
                                                   unsigned_t axis) {
  assert(outTy->dims().size() == 2 && "Invalid number of dimensions");
  assert(outTy->dims()[0] == input.dims()[0] && "Invalid dimensions");

  // FC always uses 2D input; flatten if necessary.
  if (input.dims().size() != 2) {
    input = createFlatten(name.str() + ".reshape2D", input, axis);
  }

  TypeRef OT = getParent()->uniqueType(*outTy);
  return addNode(new FullyConnectedNode(name, OT, input, W, B));
}

RowwiseQuantizedFullyConnectedNode *
Function::createRowwiseQuantizedFullyConnected(llvm::StringRef name,
                                               NodeValue input, Constant *W,
                                               Constant *scales,
                                               Constant *offsets, NodeValue B,
                                               TypeRef outTy) {
  return addNode(new RowwiseQuantizedFullyConnectedNode(name, outTy, input, W,
                                                        scales, offsets, B));
}

RowwiseQuantizedFullyConnectedNode *
Function::createRowwiseQuantizedFullyConnected(llvm::StringRef name,
                                               NodeValue input, Constant *W,
                                               NodeValue B, TypeRef outTy,
                                               quantization::Schema schema,
                                               bool transposeWeight) {
  // Since W is constant, quantize it in compilation time.
  // The quantized data is in qWeights, the scale of each row is in scales,
  // and the offset of each row is in offsets.
  Constant *weights = llvm::cast<Constant>(W);
  dim_t numRows =
      transposeWeight ? W->getType()->dims()[1] : W->getType()->dims()[0];
  dim_t numCols =
      transposeWeight ? W->getType()->dims()[0] : W->getType()->dims()[1];

  // So far, if we want to create a storage with Int8QTy/Int16QTy,
  // it is assumed to be quantized data and the scale and offset should be
  // provided. But for rowwise quantization, the scales and offsets are stored
  // in vectors separately, we add the dummy scale and offset here.
  auto *qWeights = getParent()->createConstant(
      ElemKind::Int8QTy, {numRows, numCols}, 0.0, 0, "weights.rwqfc");
  auto *scales =
      getParent()->createConstant(ElemKind::FloatTy, {numRows}, "scales.rwqfc");
  auto *offsets = getParent()->createConstant(ElemKind::Int32ITy, {numRows},
                                              "offsets.rwqfc");

  Tensor wt;
  if (transposeWeight) {
    // This happens when the RowwiseQuantizedFullyConnected node is converted
    // from a quantized FullyConnected node in Glow's quantization procedure.
    // Since in FC, the weights is stored as transposed (i.e. I * W + B), but
    // in RowwiseQuantizedFullyConnected, the weights is stored as it is (i.e.
    // I * W(T) + B).
    weights->getPayloadMutable().transpose(&wt, {1, 0});
  } else {
    wt.assign(&(weights->getPayload()));
  }

  // Note: Using int32_t offset here as that is what RWQ-FC expects.
  quantization::tensorRowwiseQuantization<float, int32_t, int8_t>(
      wt, qWeights->getPayloadMutable(), scales->getPayloadMutable(),
      offsets->getPayloadMutable(), schema);

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

Node *Function::createGELU(llvm::StringRef name, NodeValue input) {
  auto outTy = input.getType();

  Node *alphaSplat =
      createSplat(name.str() + ".alpha", outTy, M_2_SQRTPI * M_SQRT1_2);
  Node *splat = createSplat(name.str() + ".splat", outTy, 0.044715);
  Node *splatHalf = createSplat(name.str() + ".splatHalf", outTy, 0.5);
  Node *splat1 = createSplat(name.str() + ".splat3", outTy, 1.0);
  Node *splat3 = createSplat(name.str() + ".splat3", outTy, 3.0);

  // pow(x, 3)
  Node *pow = createPow(name.str() + ".pow", input, splat3);

  // pow(x, 3) * 0.044715
  Node *mul = createMul(name.str() + ".mul", pow, splat);

  // x + pow(x, 3) * 0.044715
  Node *add = createAdd(name.str() + ".add", input, mul);

  // (x * pow(x, 3) * 0.044715) * alpha
  Node *mul2 = createMul(name.str() + ".mul2", add, alphaSplat);

  // tanh((x * pow(x, 3) * 0.044715) * alpha)
  Node *tanh = createTanh(name.str() + ".tanh", mul2);

  // tanh((x * pow(x, 3) * 0.044715) * alpha) + 1
  Node *add2 = createAdd(name.str() + ".add2", tanh, splat1);

  // (tanh((x * pow(x, 3) * 0.044715) * alpha) + 1) * 0.5
  Node *mul3 = createMul(name.str() + ".mul3", splatHalf, add2);

  // (tanh((x * pow(x, 3) * 0.044715) * alpha) + 1) * 0.5 * x
  return createMul(name.str() + ".mul4", mul3, input);
}

PReluNode *Function::createPRELU(llvm::StringRef name, NodeValue input,
                                 NodeValue slope, TypeRef outTy) {
  return addNode(new PReluNode(name, outTy, input, slope));
}

PReluNode *Function::createPRELU(llvm::StringRef name, NodeValue input,
                                 NodeValue slope) {
  return addNode(new PReluNode(name, input.getType(), input, slope));
}

SigmoidNode *Function::createSigmoid(llvm::StringRef name, TypeRef outTy,
                                     NodeValue input) {
  return addNode(new SigmoidNode(name, outTy, input));
}

SigmoidNode *Function::createSigmoid(llvm::StringRef name, NodeValue input) {
  return createSigmoid(name, input.getType(), input);
}

TanhNode *Function::createTanh(llvm::StringRef name, TypeRef outTy,
                               NodeValue input) {
  return addNode(new TanhNode(name, outTy, input));
}

TanhNode *Function::createTanh(llvm::StringRef name, NodeValue input) {
  return createTanh(name, input.getType(), input);
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
  std::vector<dim_t> outDims(logits.dims().begin(), logits.dims().end() - 1);
  auto ty = getParent()->uniqueTypeWithNewShape(logits.getType(), outDims);
  return addNode(
      new SigmoidCrossEntropyWithLogitsNode(name, ty, logits, targets));
}

ReshapeNode *Function::createReshape(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<dim_t> shape,
                                     llvm::StringRef layout) {
  auto TR = getParent()->uniqueTypeWithNewShape(input.getType(), shape);
  DCHECK_EQ(TR->size(), input.getType()->size())
      << "Reshape to a different size";
  return addNode(new ReshapeNode(name, TR, input, shape.vec(), layout));
}

TransposeNode *Function::createTranspose(llvm::StringRef name, NodeValue input,
                                         llvm::ArrayRef<unsigned_t> shuffle,
                                         const std::string &layout) {
  ShapeVector shape;
  auto dims = input.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  // If the layout is known, check that it matches the shuffle:
  auto compareShuffle = [&](const std::vector<unsigned_t> targetShuffle) {
    auto shuffleVec = shuffle.vec();
    return targetShuffle.size() == dims.size() &&
           std::equal(shuffleVec.begin(), shuffleVec.end(),
                      targetShuffle.begin());
  };

  auto currLayout = layout;
  if (currLayout == ANY_LAYOUT) {
    // If layout got a default value, change it based on shuffle:
    // TODO: remove the shuffle and replace it with layout.
    if (compareShuffle(NCHW2NHWC) || compareShuffle(HWCN2NHWC)) {
      currLayout = "NHWC";
    } else if (compareShuffle(NHWC2NCHW)) {
      currLayout = "NCHW";
    } else if (compareShuffle(NHWC2HWNC)) {
      currLayout = "HWNC";
    }
  }

  auto NT = getParent()->uniqueTypeWithNewShape(input.getType(), shape);
  return addNode(new TransposeNode(name, NT, input, shuffle.vec(), currLayout));
}

Node *Function::createBroadcast(llvm::StringRef name, NodeValue input,
                                UnsignedArrayRef newShape, unsigned_t axis) {
  const auto &origDims = input.dims();

  assert(axis + origDims.size() <= newShape.size() &&
         "Axis must fit inside the newShape.");

  // Iterate over the new shape; if the original shape had a dimension here
  // (when considering the axis) then verify the dimension either matches the
  // new shape (no action taken) or == 1 (broadcast in that direction). Else
  // the original shape had no dimensions here (after considering axis), so
  // add the new dimension and broadcast in that direction.
  dim_t reshapeDims[max_tensor_dimensions];
  for (dim_t i = 0; i < newShape.size(); i++) {
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

  // Reshape the input node to same number of dimensions as new shape, but
  // with 1s in place of to-be-broadcasted dimensions.
  Node *currNode =
      createReshape(name.str() + ".reshape", input,
                    llvm::ArrayRef<dim_t>(reshapeDims, newShape.size()));

  // Create a Tile (which is really a Concat) in each direction that needs to
  // be broadcasted.
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

  // We are stacking the tensors along a specific dimension. This means that
  // we increase the size of the tensor along this dimension.
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
                                               llvm::ArrayRef<dim_t> start,
                                               unsigned_t count,
                                               unsigned_t axis) {
  return addNode(new InsertTensorNode(name, big, small, start, count, axis));
}

SliceNode *Function::createSlice(llvm::StringRef name, NodeValue input,
                                 llvm::ArrayRef<dim_t> start, TypeRef outTy) {
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
                                 llvm::ArrayRef<dim_t> begin,
                                 llvm::ArrayRef<dim_t> end) {
  std::vector<dim_t> beginV, shape;
  auto dims = input.dims();
  assert(begin.size() == end.size() && "Begin and End dimensions should match");
  assert(begin.size() == dims.size() &&
         "Begin and Input dimensions should match");
  for (unsigned i = 0; i < dims.size(); i++) {
    dim_t beginI = begin[i];
    dim_t endI = end[i];
    dim_t dimI = dims[i];
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
  return addNode(
      new ChannelShuffleNode(name, input.getType(), input, group, kernel));
}

ReshapeNode *Function::createSqueeze(llvm::StringRef name, NodeValue input,
                                     llvm::ArrayRef<dim_t> axes) {
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
                                        llvm::ArrayRef<dim_t> axes) {
  assert(!axes.empty() && "Parameter `axes` must be provided.");

  // Dimensions provided in axes are for the output tensor, so we sort them
  // and unique them to make sure they are processed correctly and in the
  // right order.
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

  // Create a reshape of the original data with the newly determined
  // dimensions.
  return createReshape(name.str() + ".expanddims", input, newDims);
}

ReshapeNode *Function::createFlatten(llvm::StringRef name, NodeValue input,
                                     unsigned_t axis) {
  auto xDim = flattenCdr(input.getType()->dims(), axis);
  return createReshape(name, input, {xDim.first, xDim.second});
}

void Function::createSplit(llvm::StringRef name, NodeValue input,
                           unsigned_t outputNum, unsigned_t axis,
                           llvm::ArrayRef<dim_t> split,
                           std::vector<SliceNode *> &outputs) {
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
    llvm::StringRef name, NodeValue input, NodeValue beta, NodeValue scale,
    NodeValue mean, NodeValue var, unsigned_t channelIdx, float epsilon,
    float momentum) {
  return addNode(new BatchNormalizationNode(name, input, scale, beta, mean, var,
                                            channelIdx, epsilon, momentum));
}

LayerNormalizationNode *Function::createLayerNormalization(llvm::StringRef name,
                                                           NodeValue input,
                                                           NodeValue scale,
                                                           NodeValue bias,
                                                           float epsilon) {
  return addNode(new LayerNormalizationNode(name, input, scale, bias, epsilon));
}

BucketizeNode *Function::createBucketizeNode(llvm::StringRef name,
                                             NodeValue input,
                                             llvm::ArrayRef<float> boundaries) {
  auto OT = getParent()->uniqueType(ElemKind::Int32ITy, input.dims());
  return addNode(new BucketizeNode(name, OT, input, boundaries));
}

LocalResponseNormalizationNode *Function::createLocalResponseNormalization(
    llvm::StringRef name, NodeValue input, unsigned_t halfWindowSize,
    float alpha, float beta, float k) {
  // The output tensor is of the same shape as the input tensor.
  return addNode(new LocalResponseNormalizationNode(name, input, halfWindowSize,
                                                    alpha, beta, k));
}

ModuloNode *Function::createModulo(llvm::StringRef name, NodeValue input,
                                   int64_t divisor, bool signFollowDivisor) {
  // The output tensor is of the same shape as the input tensor.
  auto OT = getParent()->uniqueType(*input.getType());
  return addNode(new ModuloNode(name, OT, input, divisor, signFollowDivisor));
}

#define ARITHMETIC_FUN_DEF(NODE_NAME_)                                         \
  NODE_NAME_##Node *Function::create##NODE_NAME_(                              \
      llvm::StringRef name, NodeValue LHS, NodeValue RHS) {                    \
    return create##NODE_NAME_(name, LHS.getType(), LHS, RHS);                  \
  }                                                                            \
  NODE_NAME_##Node *Function::create##NODE_NAME_(                              \
      llvm::StringRef name, TypeRef T, NodeValue LHS, NodeValue RHS) {         \
    DCHECK(LHS.dims() == RHS.dims())                                           \
        << "Invalid operand shapes " << LHS.dims() << " vs " << RHS.dims();    \
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

CmpLTENode *Function::createCmpLTE(llvm::StringRef name, NodeValue LHS,
                                   NodeValue RHS) {
  assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");
  TypeRef OT = getParent()->uniqueType(ElemKind::BoolTy, LHS.dims());
  return addNode(new CmpLTENode(name, OT, LHS, RHS));
}

CmpLTNode *Function::createCmpLT(llvm::StringRef name, NodeValue LHS,
                                 NodeValue RHS) {
  assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");
  TypeRef OT = getParent()->uniqueType(ElemKind::BoolTy, LHS.dims());
  return addNode(new CmpLTNode(name, OT, LHS, RHS));
}

CmpEQNode *Function::createCmpEQ(llvm::StringRef name, NodeValue LHS,
                                 NodeValue RHS) {
  assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");
  TypeRef OT = getParent()->uniqueType(ElemKind::BoolTy, LHS.dims());
  return addNode(new CmpEQNode(name, OT, LHS, RHS));
}

IsNaNNode *Function::createIsNaN(llvm::StringRef name, NodeValue input) {
  TypeRef OT = getParent()->uniqueType(ElemKind::BoolTy, input.dims());
  return addNode(new IsNaNNode(name, OT, input));
}

ReplaceNaNNode *Function::createReplaceNaN(llvm::StringRef name,
                                           NodeValue input, float value) {
  return addNode(new ReplaceNaNNode(name, input.getType(), input, value));
}

PowNode *Function::createPow(llvm::StringRef name, NodeValue base, float exp) {
  auto *SP = createSplat(name, base.getType(), exp);
  return createPow(name, base, SP);
}

LogNode *Function::createLog(llvm::StringRef name, NodeValue input,
                             TypeRef outTy) {
  return addNode(new LogNode(name, outTy ? outTy : input.getType(), input));
}

ExpNode *Function::createExp(llvm::StringRef name, NodeValue input) {
  return addNode(new ExpNode(name, input.getType(), input));
}

Node *Function::createLogit(llvm::StringRef name, NodeValue input, float eps) {
  assert(eps > 0.0f && "Clamping parameter eps must be strictly positive.");
  assert(eps < 0.5f && "Clamping parameter eps must be less than 0.5.");

  // Compute clamped x using clip(x, eps, 1 - eps).
  auto epsComplement = 1.0f - eps;
  auto *MaxN = createClip(name.str() + ".clip", input, eps, epsComplement);

  // Compute the logit transform of clamped x,
  // log(numerator / denominator),
  // where numerator = clamped x = MaxN,
  // and denominator = 1 - clamped x = 1 - MaxN.

  // Compute denominator = 1 - clamped x.
  auto *onesSplat =
      createSplat(name.str() + ".onesSplat", input.getType(), 1.0f);

  auto *SN = createSub(name.str() + ".sub", onesSplat, MaxN);

  // Compute the quotient = numerator / denominator.
  auto *DN = createDiv(name.str() + ".div", MaxN, SN);

  // Compute and return the logit transform (the final node).
  return createLog(name.str() + ".log", DN);
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

BatchMatMulNode *Function::createBatchMatMul(llvm::StringRef name,
                                             NodeValue LHS, NodeValue RHS) {
  const size_t numDimsRHS = RHS.dims().size();
  assert(LHS.dims().size() == 3 && "LHS must be 3 dimensional.");
  assert((numDimsRHS == 2 || numDimsRHS == 3) &&
         "RHS must be 2 or 3 dimensional.");

  // If necessary, expand the RHS input to be 3D by adding initial leading
  // dim.
  if (numDimsRHS == 2) {
    RHS = createExpandDims(name.str() + ".reshapeRHS", RHS, {0});
  }
  // If necessary, Tile the RHS input so it matches the numBatches of LHS.
  if (RHS.dims()[0] == 1 && LHS.dims()[0] != 1) {
    RHS = createTile(name.str() + ".tileRHS", RHS, LHS.dims()[0], /*axis */ 0);
  }

  // LHS = {numBatches, N, M}
  // RHS = {numBatches, M, P}
  // Result = {numBatches, N, P}
  const dim_t numBatches = LHS.dims()[0];
  const dim_t N = LHS.dims()[1];
  const dim_t M = LHS.dims()[2];
  (void)M;
  const dim_t P = RHS.dims()[2];
  assert((RHS.dims()[0] == numBatches) && "Batch sizes are invalid.");
  assert((RHS.dims()[1] == M) && "Batch matmul dimensions are invalid.");

  auto OT =
      getParent()->uniqueTypeWithNewShape(LHS.getType(), {numBatches, N, P});
  return addNode(new BatchMatMulNode(name, OT, LHS, RHS));
}

BatchedReduceAddNode *
Function::createBatchedReduceAdd(llvm::StringRef name, TypeRef outTy,
                                 NodeValue batch,
                                 llvm::ArrayRef<unsigned_t> axes) {
  assert(axes.size() == 1 && "Only supporting single reduction for now.");
  auto axis = axes[0];

  // Calculate the expected total number of elements in the output tensor
  // based on the number of elements in the batch divided by the axis
  // dimension.
  const size_t outNumElements = batch.getType()->size() / batch.dims()[axis];
  (void)outNumElements;
  assert(outTy->size() == outNumElements &&
         "Incorrect number of elements in the output type.");
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new BatchedReduceAddNode(name, OT, batch, axis));
}

BatchedReduceAddNode *
Function::createBatchedReduceAdd(llvm::StringRef name, NodeValue batch,
                                 llvm::ArrayRef<unsigned_t> axes) {
  auto outDims = getNewShapeWithoutAxes(batch.dims(), axes);
  auto OT = getParent()->uniqueTypeWithNewShape(batch.getType(), outDims);
  return createBatchedReduceAdd(name, OT, batch, axes);
}

BatchedReduceMeanNode *
Function::createBatchedReduceMean(llvm::StringRef name, TypeRef outTy,
                                  NodeValue batch,
                                  llvm::ArrayRef<unsigned_t> axes) {
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new BatchedReduceMeanNode(name, OT, batch, axes));
}

BatchedReduceMeanNode *
Function::createBatchedReduceMean(llvm::StringRef name, NodeValue batch,
                                  llvm::ArrayRef<unsigned_t> axes) {
  // Create new shape with specified dimensions either reduced or removed.
  auto outDims = getNewShapeWithoutAxes(batch.dims(), axes);
  auto OT = getParent()->uniqueType(batch.getType()->getElementType(), outDims);
  return createBatchedReduceMean(name, OT, batch, axes);
}

BatchedReduceMinNode *
Function::createBatchedReduceMin(llvm::StringRef name, NodeValue batch,
                                 llvm::ArrayRef<unsigned_t> axes) {
  // Create new shape with specified dimensions either reduced or removed.
  auto outDims = getNewShapeWithoutAxes(batch.dims(), axes);
  auto OT = getParent()->uniqueType(batch.getType()->getElementType(), outDims);
  return addNode(new BatchedReduceMinNode(name, OT, batch, axes));
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

LengthsSumNode *Function::createLengthsSum(llvm::StringRef name, NodeValue data,
                                           NodeValue lengths) {
  ShapeVector outDims(data.dims().begin(), data.dims().end());
  outDims[0] = lengths.dims()[0];
  auto outTy = getParent()->uniqueTypeWithNewShape(data.getType(), outDims);
  return addNode(new LengthsSumNode(name, outTy, data, lengths));
}

SparseLengthsSumNode *Function::createSparseLengthsSum(llvm::StringRef name,
                                                       NodeValue data,
                                                       NodeValue indices,
                                                       NodeValue lengths) {
  auto inDims = data.dims();
  ShapeVector outDims(inDims.begin(), inDims.end());
  outDims[0] = lengths.dims()[0];
  auto outTy = getParent()->uniqueTypeWithNewShape(data.getType(), outDims);
  return addNode(new SparseLengthsSumNode(name, outTy, data, indices, lengths));
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

SparseLengthsWeightedSumNode *
Function::createSparseLengthsWeightedSum(llvm::StringRef name, TypeRef outTy,
                                         NodeValue data, NodeValue weights,
                                         NodeValue indices, NodeValue lengths) {
  return addNode(new SparseLengthsWeightedSumNode(name, outTy, data, weights,
                                                  indices, lengths));
}

RowwiseQuantizedSparseLengthsWeightedSumNode *
Function::createRowwiseQuantizedSparseLengthsWeightedSum(
    llvm::StringRef name, Storage *data, Constant *scales, Constant *offsets,
    NodeValue weights, NodeValue indices, NodeValue lengths, ElemKind precision,
    bool useFP16Accumulation) {
  auto inDims = data->dims();
  ShapeVector outDims(inDims.begin(), inDims.end());
  outDims[0] = lengths.dims()[0];
  auto outTy = getParent()->uniqueType(precision, outDims);
  return addNode(new RowwiseQuantizedSparseLengthsWeightedSumNode(
      name, outTy, data, scales, offsets, weights, indices, lengths,
      useFP16Accumulation));
}

RowwiseQuantizedSparseLengthsWeightedSumNode *
Function::createRowwiseQuantizedSparseLengthsSum(
    llvm::StringRef name, Storage *data, Constant *scales, Constant *offsets,
    NodeValue indices, NodeValue lengths, ElemKind precision,
    bool useFP16Accumulation) {
  auto ty = getParent()->uniqueType(precision, {indices.dims()[0]});
  auto ones = createSplat(name.str() + ".ones", ty, 1.0);
  return createRowwiseQuantizedSparseLengthsWeightedSum(
      name, data, scales, offsets, ones, indices, lengths, precision,
      useFP16Accumulation);
}

/// Helper to create a RowwiseQuantizedSparseLengthsWeightedSumNode in the
/// Function \p F with \p name, using \ data, \p weights, \p indices, and \p
/// lengths as inputs. The provided float data in \p Tensor is rowwise
/// quantized, creating Constants for the rowwise quantized data as well as
/// Scales and Offsets, in the Module containing \p F.
static RowwiseQuantizedSparseLengthsWeightedSumNode *
quantizeDataAndCreateRowwiseQuantizedSparseLengthsWeightedSum(
    Function *F, llvm::StringRef name, Tensor &data, NodeValue weights,
    NodeValue indices, NodeValue lengths, quantization::Schema schema,
    ElemKind precision, bool useFP16Accumulation) {
  auto inDims = data.dims();

  // Note: In rwqData, we are using a quantized type, however the scale/offset
  // are set to dummy values 0.0/0. This is because the actually used
  // scale/offset come from dataScales and dataOffsets.
  Constant *rwqData = F->getParent()->createConstant(ElemKind::UInt8QTy, inDims,
                                                     0.0, 0, "data");
  Constant *dataScales =
      F->getParent()->createConstant(precision, {inDims[0]}, "dataScales");
  Constant *dataOffsets =
      F->getParent()->createConstant(precision, {inDims[0]}, "dataOffsets");

  // Note: Using floating point offset here as that is what RWQ-SLWS expects.
  switch (precision) {
  case ElemKind::FloatTy:
    quantization::tensorRowwiseQuantization<float, float, uint8_t>(
        data, rwqData->getPayloadMutable(), dataScales->getPayloadMutable(),
        dataOffsets->getPayloadMutable(), schema);
    break;
  case ElemKind::Float16Ty:
    quantization::tensorRowwiseQuantization<float16_t, float16_t, uint8_t>(
        data, rwqData->getPayloadMutable(), dataScales->getPayloadMutable(),
        dataOffsets->getPayloadMutable(), schema);
    break;
  default:
    LOG(FATAL) << "Unsupported precision for RWQ-SLWS.";
  }
  return F->createRowwiseQuantizedSparseLengthsWeightedSum(
      name, rwqData, dataScales, dataOffsets, weights, indices, lengths,
      precision, useFP16Accumulation);
}

RowwiseQuantizedSparseLengthsWeightedSumNode *
Function::createRowwiseQuantizedSparseLengthsWeightedSum(
    llvm::StringRef name, Tensor &data, NodeValue weights, NodeValue indices,
    NodeValue lengths, quantization::Schema schema, ElemKind precision,
    bool useFP16Accumulation) {
  return quantizeDataAndCreateRowwiseQuantizedSparseLengthsWeightedSum(
      this, name, data, weights, indices, lengths, schema, precision,
      useFP16Accumulation);
}

RowwiseQuantizedSparseLengthsWeightedSumNode *
Function::createRowwiseQuantizedSparseLengthsSum(
    llvm::StringRef name, Tensor &data, NodeValue indices, NodeValue lengths,
    quantization::Schema schema, ElemKind precision, bool useFP16Accumulation) {
  auto ty = getParent()->uniqueType(precision, {indices.dims()[0]});
  auto ones = createSplat(name.str() + ".ones", ty, 1.0);
  return quantizeDataAndCreateRowwiseQuantizedSparseLengthsWeightedSum(
      this, name, data, ones, indices, lengths, schema, precision,
      useFP16Accumulation);
}

/// Helper used to get specific output type required for
/// createRowwiseQuantizedSparseLengthsSum,
/// createRowwiseQuantizedSparseLengthsWeightedSum, and
/// EmbeddingBagByteRowwiseOffsets. Function \p F is used to get the specific
/// type, using inputs \p data and \p segmentsDim to compute output dimensions.
static TypeRef
getOutputTypeOfFusedRowwiseQuantizedSLS(Function *F, NodeValue data,
                                        llvm::ArrayRef<dim_t> segmentsDim) {
  ShapeVector outDims(data.dims().begin(), data.dims().end());
  outDims[0] = segmentsDim[0];
  // The output column count is the same as the input column count, but
  // without the extra bytes for the fused scale/offset, as the output is not
  // fused.
  CHECK(isFusedQuantizedElemKind(data.getElementType()))
      << "Must use a fused ElemKind for data.";
  outDims[1] -= 2 * ((data.getElementType() == ElemKind::UInt8FusedQTy)
                         ? sizeof(float)
                         : sizeof(float16_t));
  // If using 4-bit quantization, then the input data has packed two 4-bit
  // elements into one byte, so we need to double the outDims.
  if (data.getElementType() == ElemKind::UInt4FusedFP16QTy) {
    outDims[1] *= 2;
  }
  const ElemKind outputK = (data.getElementType() == ElemKind::UInt8FusedQTy)
                               ? ElemKind::FloatTy
                               : ElemKind::Float16Ty;
  return F->getParent()->uniqueType(outputK, outDims);
}

FusedRowwiseQuantizedSparseLengthsWeightedSumNode *
Function::createFusedRowwiseQuantizedSparseLengthsWeightedSum(
    llvm::StringRef name, NodeValue data, NodeValue weights, NodeValue indices,
    NodeValue lengths, bool useFP16Accumulation) {
  auto outTy =
      getOutputTypeOfFusedRowwiseQuantizedSLS(this, data, lengths.dims());
  return addNode(new FusedRowwiseQuantizedSparseLengthsWeightedSumNode(
      name, outTy, data, weights, indices, lengths, useFP16Accumulation));
}

FusedRowwiseQuantizedSparseLengthsSumNode *
Function::createFusedRowwiseQuantizedSparseLengthsSum(
    llvm::StringRef name, Storage *data, NodeValue indices, NodeValue lengths,
    bool useFP16Accumulation) {
  auto outTy =
      getOutputTypeOfFusedRowwiseQuantizedSLS(this, data, lengths.dims());
  return addNode(new FusedRowwiseQuantizedSparseLengthsSumNode(
      name, outTy, data, indices, lengths, useFP16Accumulation));
}

/// Helper to get quantized data required for
/// RowwiseQuantizedSparseLengthsWeightedSumNode and
/// RowwiseQuantizedSparseLengthsSumNode. Function \p F uses float Tensor \p
/// data to create a rowwise qunatized Constant \p rwqData, which contains fused
/// scales and offsets.
static Constant *quantizeDataForFusedRowwiseQuantizedSparseLengthsWeightedSum(
    Function *F, Tensor &data, ElemKind precision) {
  // For fused rowwise quantization, we must have a two-dimensional input. If
  // passed in a single dimensional data Tensor then add an extra dimension.
  const auto fDims = flattenCdr(data.dims());
  Tensor fData = data.getUnowned({fDims.first, fDims.second});

  // Note: In rwqData, we are using a quantized type, however the scale/offset
  // are set to dummy values 0.0/0. This is because the actually used
  // scale/offset are fused inline with each row. Also, we expand the second
  // dimension to include space for the scale/offset, each 4 bytes
  // (float/int32_t).
  switch (precision) {
  case ElemKind::UInt8FusedQTy: {
    Constant *rwqData = F->getParent()->createConstant(
        precision, {fDims.first, fDims.second + 2 * (dim_t)sizeof(float)}, 0.0,
        0, "data");
    quantization::tensorFusedRowwiseQuantization<float>(
        fData, rwqData->getPayloadMutable());
    return rwqData;
  }
  case ElemKind::UInt8FusedFP16QTy: {
    Constant *rwqData = F->getParent()->createConstant(
        precision, {fDims.first, fDims.second + 2 * (dim_t)sizeof(float16_t)},
        0.0, 0, "data");
    quantization::tensorFusedRowwiseQuantization<float16_t>(
        fData, rwqData->getPayloadMutable());
    return rwqData;
  }
  case ElemKind::UInt4FusedFP16QTy: {
    // We pack 4-bit values into bytes, so given the input size in float we
    // divide by two and take the ceiling to make sure we have enough space for
    // all elements.
    const dim_t outerDim =
        std::ceil(((float)fDims.second) / 2) + 2 * sizeof(float16_t);
    Constant *rwqData = F->getParent()->createConstant(
        precision, {fDims.first, outerDim}, 0.0, 0, "data");
    quantization::tensorFusedRowwiseQuantization<float16_t>(
        fData, rwqData->getPayloadMutable());
    return rwqData;
  }
  default:
    LOG(FATAL) << "Invalid type for FusedRowwiswQuantization.";
  }
}

FusedRowwiseQuantizedSparseLengthsWeightedSumNode *
Function::createFusedRowwiseQuantizedSparseLengthsWeightedSum(
    llvm::StringRef name, Tensor &data, NodeValue weights, NodeValue indices,
    NodeValue lengths, ElemKind fusedElemKind, bool useFP16Accumulation) {
  Constant *rwqData =
      quantizeDataForFusedRowwiseQuantizedSparseLengthsWeightedSum(
          this, data, fusedElemKind);
  return createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      name, rwqData, weights, indices, lengths, useFP16Accumulation);
}

FusedRowwiseQuantizedSparseLengthsSumNode *
Function::createFusedRowwiseQuantizedSparseLengthsSum(
    llvm::StringRef name, Tensor &data, NodeValue indices, NodeValue lengths,
    ElemKind fusedElemKind, bool useFP16Accumulation) {
  Constant *rwqData =
      quantizeDataForFusedRowwiseQuantizedSparseLengthsWeightedSum(
          this, data, fusedElemKind);
  return this->createFusedRowwiseQuantizedSparseLengthsSum(
      name, rwqData, indices, lengths, useFP16Accumulation);
}

EmbeddingBagNode *
Function::createEmbeddingBag(llvm::StringRef name, NodeValue data,
                             NodeValue weights, NodeValue indices,
                             NodeValue offsets, bool hasEndOffset) {
  auto inDims = data.dims();
  ShapeVector outDims(inDims.begin(), inDims.end());
  outDims[0] = hasEndOffset ? offsets.dims()[0] - 1 : offsets.dims()[0];
  auto outTy = getParent()->uniqueTypeWithNewShape(data.getType(), outDims);
  return addNode(new EmbeddingBagNode(name, outTy, data, weights, indices,
                                      offsets, hasEndOffset));
}

EmbeddingBagByteRowwiseOffsetsNode *
Function::createEmbeddingBagByteRowwiseOffsets(
    llvm::StringRef name, Tensor &data, NodeValue weights, NodeValue indices,
    NodeValue offsets, ElemKind fusedElemKind, bool useFP16Accumulation,
    bool hasEndOffset) {
  Constant *rwqData =
      quantizeDataForFusedRowwiseQuantizedSparseLengthsWeightedSum(
          this, data, fusedElemKind);
  return createEmbeddingBagByteRowwiseOffsets(name, rwqData, weights, indices,
                                              offsets, useFP16Accumulation,
                                              hasEndOffset);
}

EmbeddingBagByteRowwiseOffsetsNode *
Function::createEmbeddingBagByteRowwiseOffsets(
    llvm::StringRef name, NodeValue data, NodeValue weights, NodeValue indices,
    NodeValue offsets, bool useFP16Accumulation, bool hasEndOffset) {
  std::vector<dim_t> segmentDims(offsets.dims().begin(), offsets.dims().end());
  // If hasEndOffset the last offset is just for marking the end of the last
  // segment.
  if (hasEndOffset) {
    segmentDims[0] -= 1;
  }
  auto outTy = getOutputTypeOfFusedRowwiseQuantizedSLS(this, data, segmentDims);
  return addNode(new EmbeddingBagByteRowwiseOffsetsNode(
      name, outTy, data, weights, indices, offsets, useFP16Accumulation,
      hasEndOffset));
}

LengthsToRangesNode *Function::createLengthsToRanges(llvm::StringRef name,
                                                     NodeValue lengths) {
  ShapeVector outDims({lengths.dims()[0], 2});
  auto outTy = getParent()->uniqueTypeWithNewShape(lengths.getType(), outDims);
  return addNode(new LengthsToRangesNode(name, outTy, lengths));
}

LengthsRangeFillNode *
Function::createLengthsRangeFill(llvm::StringRef name, NodeValue lengths,
                                 unsigned_t maxOutputSize) {
  auto outTy =
      getParent()->uniqueTypeWithNewShape(lengths.getType(), {maxOutputSize});
  return addNode(new LengthsRangeFillNode(name, outTy, lengths));
}

SparseToDenseNode *Function::createSparseToDense(llvm::StringRef name,
                                                 NodeValue indices,
                                                 NodeValue values,
                                                 NodeValue dataToInferDim) {
  // The dimensions of the output are the same as the values tensor except for
  // the first dimension, which should match that of dataToInferDim.
  ShapeVector outDims(values.dims().begin(), values.dims().end());
  outDims[0] = dataToInferDim.dims()[0];
  auto outTy = getParent()->uniqueTypeWithNewShape(values.getType(), outDims);
  return addNode(new SparseToDenseNode(name, outTy, indices, values));
}

SparseToDenseMaskNode *Function::createSparseToDenseMask(
    llvm::StringRef name, NodeValue indices, NodeValue values,
    NodeValue defaultValue, NodeValue lengths, llvm::ArrayRef<dim_t> mask) {
  auto lengthsDims = lengths.dims();
  auto valueDims = defaultValue.dims();
  ShapeVector outDims = {(dim_t)mask.size()};
  // If lengths is 0-dimensional tensor, then there is no batch dimension.
  if (lengthsDims.size() > 0) {
    outDims.insert(outDims.begin(), lengthsDims[0]);
  }
  outDims.insert(outDims.end(), valueDims.begin(), valueDims.end());
  auto outTy = getParent()->uniqueTypeWithNewShape(values.getType(), outDims);
  return addNode(new SparseToDenseMaskNode(name, outTy, indices, values,
                                           defaultValue, lengths, mask));
}

SaveNode *Function::createSave(llvm::StringRef name, NodeValue input) {
  auto *dest = getParent()->createPlaceholder(input.getType(), name, false);
  std::string nodeName = (name + "_save").str();
  return addNode(new SaveNode(nodeName, input, dest));
}

SaveNode *Function::createSave(llvm::StringRef name, NodeValue input,
                               Placeholder *output) {
  return addNode(new SaveNode(name, input, output));
}

QuantizationProfileNode *
Function::createQuantizationProfile(PlaceholderBindings &bindings,
                                    llvm::StringRef name, NodeValue input) {
  // TODO: this size is going to be refined. Just a placeholder now.
  const size_t numberOfBuckets = 2000U;
  auto *histogram = getParent()->createPlaceholder(
      ElemKind::FloatTy, {numberOfBuckets}, "histogram_" + name.str(), false);
  bindings.allocate(histogram)->zero();
  // Intermediate data used for histogram calculations.
  // Min tensor value seen so far is kept on the first position.
  // Max tensor value seen so far is kept on the second position.
  auto *computationInfoPH = getParent()->createPlaceholder(
      ElemKind::FloatTy, {2}, "CI_" + name.str(), false);
  bindings.allocate(computationInfoPH);
  auto *computationInfoTensor = bindings.get(computationInfoPH);
  auto handle = computationInfoTensor->getHandle<float>();
  handle.raw(0) = std::numeric_limits<float>::max();
  handle.raw(1) = std::numeric_limits<float>::lowest();

  return addNode(new QuantizationProfileNode(
      "QI_" + name.str(), input, histogram, computationInfoPH,
      input.getNode()->getName().str(), input.getResNo()));
}

IntLookupTableNode *
Function::createIntLookupTable(llvm::StringRef name, NodeValue input,
                               llvm::ArrayRef<int8_t> initValues,
                               TypeRef outTy) {
  auto *mapping = getParent()->createConstant(
      ElemKind::Int8QTy, {(dim_t)initValues.size()}, outTy->getScale(),
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
      name, OT, getParent()->uniqueType(IndexElemKind, outDims), input, k));
}

ArgMaxNode *Function::createArgMax(llvm::StringRef name, NodeValue input,
                                   unsigned_t axis, bool keepDims) {
  auto inDims = input.dims();
  ShapeVector newDims;
  for (size_t i = 0, e = inDims.size(); i < e; i++) {
    if (i == axis && !keepDims) {
      continue;
    } else {
      newDims.push_back(i == axis ? 1 : inDims[i]);
    }
  }
  auto TR = getParent()->uniqueType(IndexElemKind, newDims);
  return addNode(new ArgMaxNode(name, TR, input, axis, keepDims));
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

GatherRangesNode *Function::createGatherRanges(llvm::StringRef name,
                                               NodeValue data, NodeValue ranges,
                                               unsigned_t maxOutputSize) {
  auto numRanges = ranges.dims()[0];
  return addNode(new GatherRangesNode(
      name,
      /*OutputTy=*/
      getParent()->uniqueTypeWithNewShape(data.getType(), {maxOutputSize}),
      /*LengthsTy=*/
      getParent()->uniqueTypeWithNewShape(ranges.getType(), numRanges), data,
      ranges));
}

ScatterDataNode *Function::createScatterData(llvm::StringRef name,
                                             NodeValue data, NodeValue indices,
                                             NodeValue slices,
                                             bool cumulative) {
  return addNode(new ScatterDataNode(name, data, indices, slices, cumulative));
}

BatchOneHotNode *Function::createBatchOneHot(llvm::StringRef name,
                                             NodeValue data, NodeValue lengths,
                                             NodeValue values) {
  auto outTy = getParent()->uniqueTypeWithNewShape(
      data.getType(), {data.dims()[0], values.dims()[0]});
  return addNode(new BatchOneHotNode(name, outTy, data, lengths, values));
}

SpaceToDepthNode *Function::createSpaceToDepth(llvm::StringRef name,
                                               NodeValue input,
                                               unsigned blockSize) {
  assert(blockSize > 0 && "BlockSize must be >= 1.");

  auto inputDim = input.dims();
  assert(inputDim.size() == 4 && "Dimension size of 4 is expected.");
  assert((inputDim[1] % blockSize == 0 && inputDim[2] % blockSize == 0) &&
         "Height and Width needs to be multiple of blockSize.");
  std::vector<dim_t> newDim = {inputDim[0], inputDim[1] / blockSize,
                               inputDim[2] / blockSize,
                               inputDim[3] * blockSize * blockSize};
  auto outTy = getParent()->uniqueTypeWithNewShape(input.getType(), newDim);
  return addNode(new SpaceToDepthNode(name, outTy, input, blockSize));
}

ResizeNearestNode *Function::createResizeNearest(llvm::StringRef name,
                                                 NodeValue input,
                                                 float heightScale,
                                                 float widthScale) {
  auto inputDim = input.dims();
  DCHECK_EQ(inputDim.size(), 4)
      << "Dimension size: " << inputDim.size() << ", size of 4 is expected.";
  DCHECK_GT(heightScale, 0.0) << "Height scale: " << heightScale
                              << ", Scale larger than 0 is expected.";
  DCHECK_GT(widthScale, 0.0)
      << "Width scale: " << widthScale << ", Scale larger than 0 is expected.";
  dim_t newH = std::floor(inputDim[1] * heightScale);
  DCHECK_GT(newH, 0) << "Scaled height is " << newH
                     << ", Scaled value needs to be larger than 0.";
  dim_t newW = std::floor(inputDim[2] * widthScale);
  DCHECK_GT(newW, 0) << "Scaled width is " << newW
                     << ", Scaled value needs to be larger than 0.";
  std::vector<dim_t> newDim = {inputDim[0], newH, newW, inputDim[3]};
  auto outTy = getParent()->uniqueTypeWithNewShape(input.getType(), newDim);
  return addNode(
      new ResizeNearestNode(name, outTy, input, heightScale, widthScale));
}

QuantizeNode *Function::createQuantize(llvm::StringRef name, NodeValue input,
                                       TypeRef outTy) {
  assert(input.getType()->isFPType() && "Input must be a floating type");
  assert(outTy->isQuantizedType() && "Output must be a quantized type");
  assert(input.dims().equals(outTy->dims()) &&
         "Different dimensions for input and output");

  return addNode(
      new QuantizeNode(name, getParent()->uniqueType(*outTy), input));
}

DequantizeNode *Function::createDequantize(llvm::StringRef name,
                                           NodeValue input) {
  assert(input.getType()->isQuantizedType() &&
         "Input must be a quantized type");
  TypeRef outTy =
      getParent()->uniqueType(Type(ElemKind::FloatTy, input.dims()));
  return createDequantize(name, input, outTy);
}

DequantizeNode *Function::createDequantize(llvm::StringRef name,
                                           NodeValue input, TypeRef outTy) {
  assert(input.getType()->isQuantizedType() &&
         "Input must be a quantized type");
  assert(outTy->isFPType() && "Output should be an FP type");
  return addNode(new DequantizeNode(name, outTy, input));
}

RescaleQuantizedNode *Function::createRescaleQuantized(llvm::StringRef name,
                                                       NodeValue input,
                                                       TypeRef outTy) {
  assert(input.getType()->isQuantizedType() &&
         "Input must be a quantized type");
  assert(outTy->isQuantizedType() && "Output must be a quantized type");
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
                                  NodeValue lambda1, NodeValue lambda2,
                                  float epsilon) {
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

  return addNode(new BatchBoxCoxNode(name, data, lambda1, lambda2, epsilon));
}

ClipNode *Function::createClip(llvm::StringRef name, NodeValue input,
                               TypeRef outTy, float min, float max) {
  return addNode(new ClipNode(name, outTy, input, min, max));
}

ClipNode *Function::createClip(llvm::StringRef name, NodeValue input, float min,
                               float max) {
  return addNode(new ClipNode(name, input.getType(), input, min, max));
}

ClipNode *Function::createClipMinMaxFP16(llvm::StringRef name,
                                         NodeValue input) {
  constexpr float float16Min = -65504.0f;
  constexpr float float16Max = 65504.0f;
  return createClip(name, input, float16Min, float16Max);
}

//===----------------------------------------------------------------------===//
//                   Placeholder-builder methods.
//===----------------------------------------------------------------------===//

BatchNormalizationNode *Function::createBatchNormalization(
    PlaceholderBindings &bindings, llvm::StringRef name, NodeValue input,
    unsigned_t channelIdx, float epsilon, float momentum) {
  // Figure out how many channels are in the tensor.
  dim_t channels = input.dims()[channelIdx];

  ElemKind inputTy = input.getType()->getElementType();

  // Allocate the learnable parameters beta and gamma.
  auto *beta =
      getParent()->createPlaceholder(inputTy, {channels}, "beta", true);
  bindings.allocate(beta)->init(Tensor::InitKind::Broadcast, 0.1, getPRNG());

  auto *scale =
      getParent()->createPlaceholder(inputTy, {channels}, "scale", true);
  bindings.allocate(scale)->init(Tensor::InitKind::Broadcast, 0.001, getPRNG());

  auto *mean =
      getParent()->createPlaceholder(inputTy, {channels}, "mean", false);
  bindings.allocate(mean)->zero();

  auto *variance =
      getParent()->createPlaceholder(inputTy, {channels}, "variance", false);
  bindings.allocate(variance)->init(Tensor::InitKind::Broadcast, 1.0,
                                    getPRNG());

  return createBatchNormalization(name, input, beta, scale, mean, variance,
                                  channelIdx, epsilon, momentum);
}

ConvolutionNode *Function::createConv(
    PlaceholderBindings &bindings, llvm::StringRef name, NodeValue input,
    dim_t outChannels, llvm::ArrayRef<unsigned_t> kernels,
    llvm::ArrayRef<unsigned_t> strides, llvm::ArrayRef<unsigned_t> pads,
    unsigned_t group, unsigned_t dilation, ConvolutionLayout layout) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  ShapeHW kdim(kernels);
  PaddingTLBR pdim(pads);
  (void)pdim;
  assert((idim.w + pdim.left + pdim.right) >= kdim.width &&
         (idim.h + pdim.top + pdim.bottom) >= kdim.height &&
         "buffer too small for selected stride");

  assert(group > 0 && "group should be larger than 0");
  assert(idim.c % group == 0 && "channels number must be divisible by groups");
  assert(outChannels % group == 0 && "outChannels must be divisible by groups");

  // Calculate the size and allocate the output buffer.
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides,
                                           pads, dilation);

  std::array<dim_t, 4> outDims = {
      {idim.n, outSz.first, outSz.second, outChannels}};

  // Allocate the Filter and Bias tensors.
  std::array<dim_t, 4> filterDim = {
      {outChannels, kdim.height, kdim.width, idim.c / group}};
  size_t fanIn = kdim.height * kdim.width * idim.c;
  ElemKind inputTy = input.getType()->getElementType();
  assert((inputTy == ElemKind::FloatTy || inputTy == ElemKind::Float16Ty) &&
         "Convolution on non-floating point type?");
  auto *filter =
      getParent()->createPlaceholder(inputTy, filterDim, "filter", true);
  bindings.allocate(filter)->init(glow::Tensor::InitKind::Xavier, fanIn,
                                  getPRNG());

  auto *bias =
      getParent()->createPlaceholder(inputTy, {outChannels}, "bias", true);
  bindings.allocate(bias)->init(glow::Tensor::InitKind::Broadcast, 0.1,
                                getPRNG());

  auto OT = getParent()->uniqueType(inputTy, outDims);

  return addNode(new ConvolutionNode(name, OT, input, filter, bias, kernels,
                                     strides, pads, group, dilation, layout,
                                     FusedActivation::NONE));
}

ConvolutionNode *Function::createConv(PlaceholderBindings &bindings,
                                      llvm::StringRef name, NodeValue input,
                                      dim_t outChannels, unsigned_t kernel,
                                      unsigned_t stride, unsigned_t pad,
                                      unsigned_t group, unsigned_t dilation,
                                      ConvolutionLayout layout) {
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  return createConv(bindings, name, input, outChannels, kernels, strides, pads,
                    group, dilation, layout);
}

Convolution3DNode *Function::createConv3D(PlaceholderBindings &bindings,
                                          llvm::StringRef name, NodeValue input,
                                          dim_t outChannels,
                                          llvm::ArrayRef<unsigned_t> kernels,
                                          llvm::ArrayRef<unsigned_t> strides,
                                          llvm::ArrayRef<unsigned_t> pads,
                                          unsigned_t group) {
  ShapeNHWDC idim(input.dims());
  ShapeHWD kdim(kernels);

  assert(group > 0 && "group should be larger than 0");
  assert(idim.c % group == 0 && "channels number must be divisible by groups");
  assert(outChannels % group == 0 && "outChannels must be divisible by groups");

  // Calculate the size and allocate the output buffer.
  auto outSz = calculate3DConvPoolOutputDims(idim.h, idim.w, idim.d, kernels,
                                             strides, pads);

  std::array<dim_t, 5> outDims = {
      {idim.n, outSz.height, outSz.width, outSz.depth, outChannels}};

  // Allocate the Filter and Bias tensors.
  std::array<dim_t, 5> filterDim = {
      {outChannels, kdim.height, kdim.width, kdim.depth, idim.c / group}};

  dim_t fanIn = kdim.height * kdim.width * kdim.depth * idim.c;
  ElemKind inputTy = input.getType()->getElementType();
  assert((inputTy == ElemKind::FloatTy || inputTy == ElemKind::Float16Ty) &&
         "Convolution3D on non-floating point type?");
  auto *filter =
      getParent()->createPlaceholder(inputTy, filterDim, "filter", true);
  bindings.allocate(filter)->init(glow::Tensor::InitKind::Xavier, fanIn,
                                  getPRNG());

  auto *bias =
      getParent()->createPlaceholder(inputTy, {outChannels}, "bias", true);
  bindings.allocate(bias)->init(glow::Tensor::InitKind::Broadcast, 0.1,
                                getPRNG());

  auto OT = getParent()->uniqueType(inputTy, outDims);

  assertConv3DDims(input, filter, bias, kernels, strides, pads, group);

  return addNode(new Convolution3DNode(name, OT, input, filter, bias, kernels,
                                       strides, pads, group));
}

Convolution3DNode *Function::createConv3D(PlaceholderBindings &bindings,
                                          llvm::StringRef name, NodeValue input,
                                          size_t outChannels, unsigned_t kernel,
                                          unsigned_t stride, unsigned_t pad,
                                          unsigned_t group) {
  llvm::SmallVector<unsigned_t, 6> pads = {pad, pad, pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 3> strides = {stride, stride, stride};
  llvm::SmallVector<unsigned_t, 3> kernels = {kernel, kernel, kernel};
  return createConv3D(bindings, name, input, outChannels, kernels, strides,
                      pads, group);
}

ChannelwiseQuantizedConvolutionNode *Function::createChannelwiseQuantizedConv(
    llvm::StringRef name, NodeValue input, Constant *filter, Constant *bias,
    Constant *scales, Constant *offsets, TypeRef outTy,
    llvm::ArrayRef<unsigned_t> kernels, llvm::ArrayRef<unsigned_t> strides,
    llvm::ArrayRef<unsigned_t> pads, unsigned_t group) {
  assertConvDims(input, filter, bias, kernels, strides, pads, group);
  auto OT = getParent()->uniqueType(*outTy);
  return addNode(new ChannelwiseQuantizedConvolutionNode(
      name, OT, input, filter, bias, scales, offsets, kernels, strides, pads,
      group, /*Groupwise*/ true));
}

ConvertToNode *Function::createConvertTo(llvm::StringRef name, NodeValue input,
                                         TypeRef outTy) {
  return addNode(new ConvertToNode(name, outTy, input));
}

ConvertToNode *Function::createConvertTo(llvm::StringRef name, NodeValue input,
                                         ElemKind k) {
  auto OT = getParent()->uniqueType(k, input.dims());
  return addNode(new ConvertToNode(name, OT, input));
}

FullyConnectedNode *
Function::createFullyConnected(PlaceholderBindings &bindings,
                               llvm::StringRef name, NodeValue input,
                               dim_t outDepth, unsigned_t axis) {
  const ElemKind k = input.getType()->getElementType();

  // FC always uses 2D input; flatten if necessary.
  if (input.dims().size() != 2) {
    input = createFlatten(name.str() + ".reshape2D", input, axis);
  }
  auto *W = getParent()->createPlaceholder(k, {input.dims()[1], outDepth},
                                           "weights", true);
  auto *B = getParent()->createPlaceholder(k, {outDepth}, "bias", true);

  bindings.allocate(W)->init(Tensor::InitKind::Xavier, input.dims()[1],
                             getPRNG());
  bindings.allocate(B)->init(Tensor::InitKind::Broadcast, .1, getPRNG());

  auto OT = getParent()->uniqueType(k, {input.dims()[0], outDepth});
  return createFullyConnected(name, input, W, B, OT, axis);
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

Node *Function::createElementwiseLinear(llvm::StringRef name, NodeValue X,
                                        NodeValue w, NodeValue b,
                                        unsigned axis) {
  auto XDims = X.dims();
  auto wDims = w.dims();
  auto bDims = b.dims();

  // Suppress release mode unused variable warnings.
  (void)wDims;
  (void)bDims;

  // Check that the inputs are sensible.
  assert(XDims.size() == 2 && "X must be 2D");
  assert((axis == 0 || axis == 1) && "axis must be 0 or 1");
  assert(wDims.size() == 1 && "w must be 1D");
  assert(bDims.size() == 1 && "b must be 1D");
  assert(wDims[0] == XDims[axis] &&
         "size of w must match input dimension of X");
  assert(bDims[0] == XDims[axis] &&
         "size of b must match input dimension of X");

  // Broadcast w and b so that they have the same dimensions as X.
  auto *broadcastW =
      createBroadcast(name.str() + ".broadcastW", w, XDims, axis);
  auto *broadcastB =
      createBroadcast(name.str() + ".broadcastB", b, XDims, axis);

  // Implement the elementwise linear operation by multiplying X elementwise
  // with broadcasted w and adding broadcasted b elementwise.
  auto *wX = createMul(name.str() + ".mul", broadcastW, X);
  auto *out = createAdd(name.str() + ".add", wX, broadcastB);

  return out;
}

void Function::createGRU(PlaceholderBindings &bindings,
                         llvm::StringRef namePrefix,
                         llvm::ArrayRef<NodeValue> inputs, unsigned batchSize,
                         unsigned hiddenSize, unsigned outputSize,
                         std::vector<NodeValue> &outputs) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front().dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the state to zero.
  Placeholder *HInit = getParent()->createPlaceholder(
      ElemKind::FloatTy, {batchSize, hiddenSize}, "initial_state", false);
  bindings.allocate(HInit)->zero();
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

  bindings.allocate(Wxz)->init(glow::Tensor::InitKind::Xavier, inputSize,
                               getPRNG());
  bindings.allocate(Whz)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(Bz1)->init(glow::Tensor::InitKind::Broadcast, bUpdate,
                               getPRNG());
  bindings.allocate(Bz2)->init(glow::Tensor::InitKind::Broadcast, bUpdate,
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

  bindings.allocate(Wxr)->init(glow::Tensor::InitKind::Xavier, inputSize,
                               getPRNG());
  bindings.allocate(Whr)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(Br1)->init(glow::Tensor::InitKind::Broadcast, bReset,
                               getPRNG());
  bindings.allocate(Br2)->init(glow::Tensor::InitKind::Broadcast, bReset,
                               getPRNG());

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

  bindings.allocate(Wxh)->init(glow::Tensor::InitKind::Xavier, inputSize,
                               getPRNG());
  bindings.allocate(Whh)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(Bh1)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());
  bindings.allocate(Bh2)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());

  // Output Layer.
  Placeholder *Why = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, outputSize}, nameBase + ".Why", true);
  Placeholder *By = getParent()->createPlaceholder(
      ElemKind::FloatTy, {outputSize}, nameBase + ".by", true);

  bindings.allocate(Why)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(By)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());

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

void Function::createSimpleRNN(PlaceholderBindings &bindings,
                               llvm::StringRef namePrefix,
                               llvm::ArrayRef<NodeValue> inputs,
                               unsigned batchSize, unsigned hiddenSize,
                               unsigned outputSize,
                               std::vector<NodeValue> &outputs) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front().dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the state to zero.
  Placeholder *HInit =
      getParent()->createPlaceholder(ElemKind::FloatTy, {batchSize, hiddenSize},
                                     nameBase + ".initial_state", false);
  bindings.allocate(HInit)->zero();
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

  bindings.allocate(Whh)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(Bhh)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());
  bindings.allocate(Wxh)->init(glow::Tensor::InitKind::Xavier, inputSize,
                               getPRNG());
  bindings.allocate(Bxh)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());
  bindings.allocate(Why)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(Bhy)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());

  // Un-roll backpropogation through time as a loop with the shared
  // parameters.
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

void Function::createLSTM(PlaceholderBindings &bindings,
                          llvm::StringRef namePrefix,
                          llvm::ArrayRef<NodeValue> inputs, unsigned batchSize,
                          unsigned hiddenSize, unsigned outputSize,
                          std::vector<NodeValue> &outputs) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front().dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the hidden and cell states to zero.
  Placeholder *HInit =
      getParent()->createPlaceholder(ElemKind::FloatTy, {batchSize, hiddenSize},
                                     "initial_hidden_state", false);
  bindings.allocate(HInit)->zero();
  Node *Ht = HInit;

  Placeholder *CInit = getParent()->createPlaceholder(
      ElemKind::FloatTy, {batchSize, hiddenSize}, "initial_cell_state", false);
  bindings.allocate(CInit)->zero();
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
  bindings.allocate(Wxf)->init(glow::Tensor::InitKind::Xavier, inputSize,
                               getPRNG());
  bindings.allocate(Whf)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(Bf1)->init(glow::Tensor::InitKind::Broadcast, bForget,
                               getPRNG());
  bindings.allocate(Bf2)->init(glow::Tensor::InitKind::Broadcast, bForget,
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

  bindings.allocate(Wxi)->init(glow::Tensor::InitKind::Xavier, inputSize,
                               getPRNG());
  bindings.allocate(Whi)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(Bi1)->init(glow::Tensor::InitKind::Broadcast, bInput,
                               getPRNG());
  bindings.allocate(Bi2)->init(glow::Tensor::InitKind::Broadcast, bInput,
                               getPRNG());

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

  bindings.allocate(Wxo)->init(glow::Tensor::InitKind::Xavier, inputSize,
                               getPRNG());
  bindings.allocate(Who)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(Bo1)->init(glow::Tensor::InitKind::Broadcast, bOutput,
                               getPRNG());
  bindings.allocate(Bo2)->init(glow::Tensor::InitKind::Broadcast, bOutput,
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

  bindings.allocate(Wxc)->init(glow::Tensor::InitKind::Xavier, inputSize,
                               getPRNG());
  bindings.allocate(Whc)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(Bc1)->init(glow::Tensor::InitKind::Broadcast, bCell,
                               getPRNG());
  bindings.allocate(Bc2)->init(glow::Tensor::InitKind::Broadcast, bCell,
                               getPRNG());

  // output layer
  float b = 0.1;
  Placeholder *Why = getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, outputSize}, nameBase + ".Why", true);
  Placeholder *By = getParent()->createPlaceholder(
      ElemKind::FloatTy, {outputSize}, nameBase + ".by", true);

  bindings.allocate(Why)->init(glow::Tensor::InitKind::Xavier, hiddenSize,
                               getPRNG());
  bindings.allocate(By)->init(glow::Tensor::InitKind::Broadcast, b, getPRNG());

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

void Function::createOnnxRNN(llvm::StringRef namePrefix, NodeValue X,
                             NodeValue W, NodeValue R, NodeValue B,
                             NodeValue initial_h, NodeValue &Y, NodeValue &Y_h,
                             unsigned hiddenSize, RnnDirection direction,
                             std::vector<RnnActivation> &activations) {

#define RNN_X_SLICE_RANGE(idx)                                                 \
  {idx + 0, 0, 0}, { idx + 1, batchSize, inputSize }
#define RNN_W_SLICE_RANGE(idx0, idx1)                                          \
  {idx0, idx1 * hiddenSize, 0}, { idx0 + 1, (idx1 + 1) * hiddenSize, inputSize }
#define RNN_R_SLICE_RANGE(idx0, idx1)                                          \
  {idx0, idx1 * hiddenSize, 0}, {                                              \
    idx0 + 1, (idx1 + 1) * hiddenSize, hiddenSize                              \
  }
#define RNN_B_SLICE_RANGE(idx0, idx1)                                          \
  {idx0, idx1 * hiddenSize}, { idx0 + 1, (idx1 + 1) * hiddenSize }
#define RNN_H_SLICE_RANGE(idx)                                                 \
  {idx + 0, 0, 0}, { idx + 1, batchSize, hiddenSize }
#define RNN_CREATE_FC(name, LHS, RHS, BIAS)                                    \
  BIAS ? (Node *)createFullyConnected(name, LHS, RHS, BIAS)                    \
       : (Node *)createMatMul(name, LHS, RHS)

  // Operator name.
  const std::string &opName = namePrefix.str();

  // Get all size parameters.
  dim_t numDirections = (direction == RnnDirection::Bidirectional) ? 2 : 1;
  assert(X.dims().size() == 3 &&
         "ONNX RNN input 'X' should have 3 dimensions!");
  dim_t seqLength = X.dims()[0];
  dim_t batchSize = X.dims()[1];
  dim_t inputSize = X.dims()[2];

  // Validate W size.
  assert(W.dims().size() == 3 &&
         "ONNX RNN input 'W' should have 3 dimensions!");
  assert(W.dims()[0] == numDirections && W.dims()[1] == hiddenSize &&
         W.dims()[2] == inputSize && "ONNX RNN 'W' tensor size invalid!");

  // Validate R size.
  assert(R.dims().size() == 3 &&
         "ONNX RNN input 'R' should have 3 dimensions!");
  assert(R.dims()[0] == numDirections && R.dims()[1] == hiddenSize &&
         R.dims()[2] == hiddenSize && "ONNX RNN 'R' tensor size invalid!");

  // Validate B size.
  if (B.getNode()) {
    assert(B.dims().size() == 2 &&
           "ONNX RNN input 'B' should have 2 dimensions!");
    assert(B.dims()[0] == numDirections && B.dims()[1] == 2 * hiddenSize &&
           "ONNX RNN 'B' tensor size invalid!");
  }

  // Validate initial_h size.
  assert(initial_h.getNode() &&
         "ONNX RNN input 'initial_h' is mandatory. Null provided!");
  assert(initial_h.dims().size() == 3 &&
         "ONNX RNN input 'initial_h' should have 2 dimensions!");
  assert(initial_h.dims()[0] == numDirections &&
         initial_h.dims()[1] == batchSize &&
         initial_h.dims()[2] == hiddenSize &&
         "ONNX RNN 'initial_h' tensor size invalid!");

  // Validate number of activations.
  assert(activations.size() == numDirections * 1 &&
         "ONNX RNN activations vector invalid!");

  // Create X slices.
  std::vector<Node *> Xslices;
  for (dim_t t = 0; t < seqLength; t++) {
    auto XsliceName = opName + ".X" + std::to_string(t) + ".slice";
    Node *Xt = createSlice(XsliceName, X, RNN_X_SLICE_RANGE(t));
    auto XreshapeName = opName + ".X" + std::to_string(t) + ".reshape";
    Xt = createReshape(XreshapeName, Xt, {batchSize, inputSize});
    Xslices.push_back(Xt);
  }

  // Lambda to load forward/backward RNN cell.
  auto loadRNNCell = [&](bool forward, std::vector<NodeValue> &Yslices,
                         NodeValue &Hslice) {
    // Name prefix.
    std::string dirLabel = forward ? ".fw" : ".bw";
    std::string prefix = opName + ((numDirections > 1) ? dirLabel : "");

    // Slice index used for creating weights slices.
    dim_t sliceIdx0 = 0;
    if (direction == RnnDirection::Bidirectional) {
      sliceIdx0 = forward ? 0 : 1;
    }

    // Activations.
    size_t activationOffset = sliceIdx0 * 1;
    auto activationF = activations[activationOffset + 0];

    // Create W slice (Required).
    NodeValue Wi =
        createSlice(prefix + ".Wi.", W, RNN_W_SLICE_RANGE(sliceIdx0, 0));
    Wi = createReshape(prefix + ".Wi.reshape", Wi, {hiddenSize, inputSize});
    Wi = createTranspose(prefix + ".Wi.transp", Wi, {1, 0});

    // Create R slice (Required).
    NodeValue Ri =
        createSlice(prefix + ".Ri.", R, RNN_R_SLICE_RANGE(sliceIdx0, 0));
    Ri = createReshape(prefix + ".Ri.reshape", Ri, {hiddenSize, hiddenSize});
    Ri = createTranspose(prefix + ".Ri.transp", Ri, {1, 0});

    // Create B slices (optional).
    NodeValue bWi = nullptr;
    NodeValue bRi = nullptr;

    if (B) {

      bWi = createSlice(prefix + ".bWi.", B, RNN_B_SLICE_RANGE(sliceIdx0, 0));
      bRi = createSlice(prefix + ".bRi.", B, RNN_B_SLICE_RANGE(sliceIdx0, 1));

      bWi = createReshape(prefix + ".bWi.reshape", bWi, {hiddenSize});
      bRi = createReshape(prefix + ".bRi.reshape", bRi, {hiddenSize});
    }

    // Create H slice for this direction.
    Node *Hinit = createSlice(prefix + ".H.slice", initial_h,
                              RNN_H_SLICE_RANGE(sliceIdx0));
    Hinit =
        createReshape(prefix + ".H.reshape", Hinit, {batchSize, hiddenSize});

    // Initialize.
    Node *Ht = Hinit;

    // Unroll RNN cell for all time steps.
    for (size_t t = 0; t < seqLength; t++) {

      // Input for current time step.
      // For the reverse RNN cell the inputs are provided in reverse order.
      Node *Xt = forward ? Xslices[t] : Xslices[seqLength - 1 - t];

      // Hidden state update: Ht = f(Xt * Wi + bWi + Ht-1 * Ri + bRi).
      Ht = createAdd(prefix + ".H.add",
                     RNN_CREATE_FC(prefix + ".H.fc1", Xt, Wi, bWi),
                     RNN_CREATE_FC(prefix + ".H.fc2", Ht, Ri, bRi));
      Ht = activationF(prefix + ".H.act", Ht);

      // Output.
      Yslices.push_back(Ht);
    }

    // Updated states nodes.
    Hslice = Ht;
  }; // End of local lambda "loadRNNCell".

  bool forwardEnabled = ((direction == RnnDirection::Forward) ||
                         (direction == RnnDirection::Bidirectional));
  bool backwardEnabled = ((direction == RnnDirection::Reverse) ||
                          (direction == RnnDirection::Bidirectional));

  std::vector<NodeValue> YSlices;
  std::vector<NodeValue> Hslices;

  // Load forward RNN.
  std::vector<NodeValue> forwardYslices;
  if (forwardEnabled) {
    NodeValue forwardHslice;
    loadRNNCell(/* forward */ true, forwardYslices, forwardHslice);
    Hslices.push_back(forwardHslice);
  }

  // Load backward RNN.
  std::vector<NodeValue> backwardYslices;
  if (backwardEnabled) {
    NodeValue backwardHslice;
    loadRNNCell(/* forward */ false, backwardYslices, backwardHslice);
    Hslices.push_back(backwardHslice);
  }

  // Gather Y slices.
  for (size_t t = 0; t < seqLength; t++) {
    if (forwardEnabled) {
      YSlices.push_back(forwardYslices[t]);
    }
    if (backwardEnabled) {
      YSlices.push_back(backwardYslices[seqLength - 1 - t]);
    }
  }

  // Concatenate Y slices.
  // Y size is [seqLength, numDirections, batchSize, hiddenSize].
  Y = createReshape(opName + ".Y.reshape",
                    createConcat(opName + ".Y.concat", YSlices, 0),
                    {seqLength, numDirections, batchSize, hiddenSize});

  // Concatenate Y_h slices.
  // Y_h size is [numDirections, batchSize, hiddenSize].
  Y_h = createReshape(opName + ".Y_h.reshape",
                      createConcat(opName + ".Y_h.concat", Hslices, 0),
                      {numDirections, batchSize, hiddenSize});

#undef RNN_X_SLICE_RANGE
#undef RNN_W_SLICE_RANGE
#undef RNN_R_SLICE_RANGE
#undef RNN_B_SLICE_RANGE
#undef RNN_H_SLICE_RANGE
#undef RNN_CREATE_FC
}

void Function::createOnnxGRU(llvm::StringRef namePrefix, NodeValue X,
                             NodeValue W, NodeValue R, NodeValue B,
                             NodeValue initial_h, NodeValue &Y, NodeValue &Y_h,
                             unsigned hiddenSize, RnnDirection direction,
                             std::vector<RnnActivation> &activations,
                             bool linearBeforeReset) {

#define GRU_X_SLICE_RANGE(idx)                                                 \
  {idx + 0, 0, 0}, { idx + 1, batchSize, inputSize }
#define GRU_W_SLICE_RANGE(idx0, idx1)                                          \
  {idx0, idx1 * hiddenSize, 0}, { idx0 + 1, (idx1 + 1) * hiddenSize, inputSize }
#define GRU_R_SLICE_RANGE(idx0, idx1)                                          \
  {idx0, idx1 * hiddenSize, 0}, {                                              \
    idx0 + 1, (idx1 + 1) * hiddenSize, hiddenSize                              \
  }
#define GRU_B_SLICE_RANGE(idx0, idx1)                                          \
  {idx0, idx1 * hiddenSize}, { idx0 + 1, (idx1 + 1) * hiddenSize }
#define GRU_H_SLICE_RANGE(idx)                                                 \
  {idx + 0, 0, 0}, { idx + 1, batchSize, hiddenSize }
#define GRU_CREATE_FC(name, LHS, RHS, BIAS)                                    \
  BIAS ? (Node *)createFullyConnected(name, LHS, RHS, BIAS)                    \
       : (Node *)createMatMul(name, LHS, RHS)

  // Operator name.
  const std::string &opName = namePrefix.str();

  // Get all size parameters.
  dim_t numDirections = (direction == RnnDirection::Bidirectional) ? 2 : 1;
  assert(X.dims().size() == 3 &&
         "ONNX GRU input 'X' should have 3 dimensions!");
  dim_t seqLength = X.dims()[0];
  dim_t batchSize = X.dims()[1];
  dim_t inputSize = X.dims()[2];

  // Validate W size.
  assert(W.dims().size() == 3 &&
         "ONNX GRU input 'W' should have 3 dimensions!");
  assert(W.dims()[0] == numDirections && W.dims()[1] == 3 * hiddenSize &&
         W.dims()[2] == inputSize && "ONNX GRU 'W' tensor size invalid!");

  // Validate R size.
  assert(R.dims().size() == 3 &&
         "ONNX GRU input 'R' should have 3 dimensions!");
  assert(R.dims()[0] == numDirections && R.dims()[1] == 3 * hiddenSize &&
         R.dims()[2] == hiddenSize && "ONNX GRU 'R' tensor size invalid!");

  // Validate B size.
  if (B.getNode()) {
    assert(B.dims().size() == 2 &&
           "ONNX GRU input 'B' should have 2 dimensions!");
    assert(B.dims()[0] == numDirections && B.dims()[1] == 6 * hiddenSize &&
           "ONNX GRU 'B' tensor size invalid!");
  }

  // Validate initial_h size.
  assert(initial_h.getNode() &&
         "ONNX GRU input 'initial_h' is mandatory. Null provided!");
  assert(initial_h.dims().size() == 3 &&
         "ONNX GRU input 'initial_h' should have 2 dimensions!");
  assert(initial_h.dims()[0] == numDirections &&
         initial_h.dims()[1] == batchSize &&
         initial_h.dims()[2] == hiddenSize &&
         "ONNX GRU 'initial_h' tensor size invalid!");

  // Validate number of activations.
  assert(activations.size() == numDirections * 2 &&
         "ONNX GRU activations vector invalid!");

  // Create X slices.
  std::vector<Node *> Xslices;
  for (dim_t t = 0; t < seqLength; t++) {
    auto XsliceName = opName + ".X" + std::to_string(t) + ".slice";
    Node *Xt = createSlice(XsliceName, X, GRU_X_SLICE_RANGE(t));
    auto XreshapeName = opName + ".X" + std::to_string(t) + ".reshape";
    Xt = createReshape(XreshapeName, Xt, {batchSize, inputSize});
    Xslices.push_back(Xt);
  }

  // Lambda to load forward/backward GRU cell.
  auto loadGRUCell = [&](bool forward, std::vector<NodeValue> &Yslices,
                         NodeValue &Hslice) {
    // Name prefix.
    std::string dirLabel = forward ? ".fw" : ".bw";
    std::string prefix = opName + ((numDirections > 1) ? dirLabel : "");

    // Slice index used for creating weights slices.
    dim_t sliceIdx0 = 0;
    if (direction == RnnDirection::Bidirectional) {
      sliceIdx0 = forward ? 0 : 1;
    }

    // Activations.
    size_t activationOffset = sliceIdx0 * 2;
    auto activationF = activations[activationOffset + 0];
    auto activationG = activations[activationOffset + 1];

    // Create W slices (Required).
    NodeValue Wz =
        createSlice(prefix + ".Wz.", W, GRU_W_SLICE_RANGE(sliceIdx0, 0));
    NodeValue Wr =
        createSlice(prefix + ".Wr.", W, GRU_W_SLICE_RANGE(sliceIdx0, 1));
    NodeValue Wh =
        createSlice(prefix + ".Wh.", W, GRU_W_SLICE_RANGE(sliceIdx0, 2));

    Wz = createReshape(prefix + ".Wz.reshape", Wz, {hiddenSize, inputSize});
    Wr = createReshape(prefix + ".Wr.reshape", Wr, {hiddenSize, inputSize});
    Wh = createReshape(prefix + ".Wh.reshape", Wh, {hiddenSize, inputSize});

    Wz = createTranspose(prefix + ".Wz.transp", Wz, {1, 0});
    Wr = createTranspose(prefix + ".Wr.transp", Wr, {1, 0});
    Wh = createTranspose(prefix + ".Wh.transp", Wh, {1, 0});

    // Create R slices (Required).
    NodeValue Rz =
        createSlice(prefix + ".Rz.", R, GRU_R_SLICE_RANGE(sliceIdx0, 0));
    NodeValue Rr =
        createSlice(prefix + ".Rr.", R, GRU_R_SLICE_RANGE(sliceIdx0, 1));
    NodeValue Rh =
        createSlice(prefix + ".Rh.", R, GRU_R_SLICE_RANGE(sliceIdx0, 2));

    Rz = createReshape(prefix + ".Rz.reshape", Rz, {hiddenSize, hiddenSize});
    Rr = createReshape(prefix + ".Rr.reshape", Rr, {hiddenSize, hiddenSize});
    Rh = createReshape(prefix + ".Rh.reshape", Rh, {hiddenSize, hiddenSize});

    Rz = createTranspose(prefix + ".Rz.transp", Rz, {1, 0});
    Rr = createTranspose(prefix + ".Rr.transp", Rr, {1, 0});
    Rh = createTranspose(prefix + ".Rh.transp", Rh, {1, 0});

    // Create B slices (optional).
    NodeValue bWz = nullptr;
    NodeValue bWr = nullptr;
    NodeValue bWh = nullptr;
    NodeValue bRz = nullptr;
    NodeValue bRr = nullptr;
    NodeValue bRh = nullptr;

    if (B) {

      bWz = createSlice(prefix + ".bWz.", B, GRU_B_SLICE_RANGE(sliceIdx0, 0));
      bWr = createSlice(prefix + ".bWr.", B, GRU_B_SLICE_RANGE(sliceIdx0, 1));
      bWh = createSlice(prefix + ".bWh.", B, GRU_B_SLICE_RANGE(sliceIdx0, 2));
      bRz = createSlice(prefix + ".bRz.", B, GRU_B_SLICE_RANGE(sliceIdx0, 3));
      bRr = createSlice(prefix + ".bRr.", B, GRU_B_SLICE_RANGE(sliceIdx0, 4));
      bRh = createSlice(prefix + ".bRh.", B, GRU_B_SLICE_RANGE(sliceIdx0, 5));

      bWz = createReshape(prefix + ".bWz.reshape", bWz, {hiddenSize});
      bWr = createReshape(prefix + ".bWr.reshape", bWr, {hiddenSize});
      bWh = createReshape(prefix + ".bWh.reshape", bWh, {hiddenSize});
      bRz = createReshape(prefix + ".bRz.reshape", bRz, {hiddenSize});
      bRr = createReshape(prefix + ".bRr.reshape", bRr, {hiddenSize});
      bRh = createReshape(prefix + ".bRh.reshape", bRh, {hiddenSize});
    }

    // Create H slice for this direction.
    Node *Hinit = createSlice(prefix + ".H.slice", initial_h,
                              GRU_H_SLICE_RANGE(sliceIdx0));
    Hinit =
        createReshape(prefix + ".H.reshape", Hinit, {batchSize, hiddenSize});

    // Initialize.
    Node *Ht = Hinit;

    // Unroll GRU cell for all time steps.
    for (size_t t = 0; t < seqLength; t++) {

      // Input for current time step.
      // For the reverse GRU cell the inputs are provided in reverse order.
      Node *Xt = forward ? Xslices[t] : Xslices[seqLength - 1 - t];

      // Update gate: zt = f(Xt * Wz + bWz + Ht-1 * Rz + bRz).
      Node *zt = createAdd(prefix + ".Z.add1",
                           GRU_CREATE_FC(prefix + ".Z.fc1", Xt, Wz, bWz),
                           GRU_CREATE_FC(prefix + ".Z.fc2", Ht, Rz, bRz));
      zt = activationF(prefix + ".Z.act", zt);

      // Reset gate: rt = f(Xt * Wr + bWr + Ht-1 * Rr + bRr).
      Node *rt = createAdd(prefix + ".R.add1",
                           GRU_CREATE_FC(prefix + ".R.fc1", Xt, Wr, bWr),
                           GRU_CREATE_FC(prefix + ".R.fc2", Ht, Rr, bRr));
      rt = activationF(prefix + ".R.act", rt);

      // Hidden gate:
      // For linearBeforeReset = true:
      //   htild = g(Xt * Wh + bWh + rt . (Ht-1 * Rh + bRh)).
      // For linearBeforeReset = false:
      //   htild = g(Xt * Wh + bWh + (rt . Ht-1) * Rh + bRh).
      Node *htild;
      if (linearBeforeReset) {
        htild = createAdd(
            prefix + ".Htild.add",
            GRU_CREATE_FC(prefix + ".Htild.fc1", Xt, Wh, bWh),
            createMul(prefix + ".Htild.reset", rt,
                      GRU_CREATE_FC(prefix + ".Htild.fc2", Ht, Rh, bRh)));
      } else {
        htild = createAdd(
            prefix + ".Htild.add",
            GRU_CREATE_FC(prefix + ".Htild.fc1", Xt, Wh, bWh),
            GRU_CREATE_FC(prefix + ".Htild.fc2",
                          createMul(prefix + ".Htild.reset", rt, Ht), Rh, bRh));
      }
      htild = activationG(prefix + ".Htild.act", htild);

      // Hidden state update:
      // Ht = (1 - zt) . htild + zt . Ht-1 = htild - zt . htild + zt . Ht-1.
      Ht = createAdd(prefix + ".H.add",
                     createSub(prefix + ".H.sub", htild,
                               createMul(prefix + ".H.mult1", zt, htild)),
                     createMul(prefix + ".H.mult2", zt, Ht));

      // Output.
      Yslices.push_back(Ht);
    }

    // Updated states nodes.
    Hslice = Ht;
  }; // End of local lambda "loadGRUCell".

  bool forwardEnabled = ((direction == RnnDirection::Forward) ||
                         (direction == RnnDirection::Bidirectional));
  bool backwardEnabled = ((direction == RnnDirection::Reverse) ||
                          (direction == RnnDirection::Bidirectional));

  std::vector<NodeValue> YSlices;
  std::vector<NodeValue> Hslices;

  // Load forward GRU.
  std::vector<NodeValue> forwardYslices;
  if (forwardEnabled) {
    NodeValue forwardHslice;
    loadGRUCell(/* forward */ true, forwardYslices, forwardHslice);
    Hslices.push_back(forwardHslice);
  }

  // Load backward GRU.
  std::vector<NodeValue> backwardYslices;
  if (backwardEnabled) {
    NodeValue backwardHslice;
    loadGRUCell(/* forward */ false, backwardYslices, backwardHslice);
    Hslices.push_back(backwardHslice);
  }

  // Gather Y slices.
  for (size_t t = 0; t < seqLength; t++) {
    if (forwardEnabled) {
      YSlices.push_back(forwardYslices[t]);
    }
    if (backwardEnabled) {
      YSlices.push_back(backwardYslices[seqLength - 1 - t]);
    }
  }

  // Concatenate Y slices.
  // Y size is [seqLength, numDirections, batchSize, hiddenSize].
  Y = createReshape(opName + ".Y.reshape",
                    createConcat(opName + ".Y.concat", YSlices, 0),
                    {seqLength, numDirections, batchSize, hiddenSize});

  // Concatenate Y_h slices.
  // Y_h size is [numDirections, batchSize, hiddenSize].
  Y_h = createReshape(opName + ".Y_h.reshape",
                      createConcat(opName + ".Y_h.concat", Hslices, 0),
                      {numDirections, batchSize, hiddenSize});

#undef GRU_X_SLICE_RANGE
#undef GRU_W_SLICE_RANGE
#undef GRU_R_SLICE_RANGE
#undef GRU_B_SLICE_RANGE
#undef GRU_H_SLICE_RANGE
#undef GRU_CREATE_FC
}

void Function::createOnnxLSTM(llvm::StringRef namePrefix, NodeValue X,
                              NodeValue W, NodeValue R, NodeValue B,
                              NodeValue initial_h, NodeValue initial_c,
                              NodeValue P, NodeValue &Y, NodeValue &Y_h,
                              NodeValue &Y_c, unsigned hiddenSize,
                              RnnDirection direction,
                              std::vector<RnnActivation> &activations,
                              bool inputForget) {

#define LSTM_X_SLICE_RANGE(idx)                                                \
  {idx + 0, 0, 0}, { idx + 1, batchSize, inputSize }
#define LSTM_H_SLICE_RANGE(idx)                                                \
  {idx + 0, 0, 0}, { idx + 1, batchSize, hiddenSize }
#define LSTM_C_SLICE_RANGE(idx)                                                \
  {idx + 0, 0, 0}, { idx + 1, batchSize, hiddenSize }
#define LSTM_W_SLICE_RANGE(idx0, idx1)                                         \
  {idx0, idx1 * hiddenSize, 0}, { idx0 + 1, (idx1 + 1) * hiddenSize, inputSize }
#define LSTM_R_SLICE_RANGE(idx0, idx1)                                         \
  {idx0, idx1 * hiddenSize, 0}, {                                              \
    idx0 + 1, (idx1 + 1) * hiddenSize, hiddenSize                              \
  }
#define LSTM_B_SLICE_RANGE(idx0, idx1)                                         \
  {idx0, idx1 * hiddenSize}, { idx0 + 1, (idx1 + 1) * hiddenSize }
#define LSTM_P_SLICE_RANGE(idx0, idx1)                                         \
  {idx0, idx1 * hiddenSize}, { idx0 + 1, (idx1 + 1) * hiddenSize }
#define LSTM_CREATE_FC(name, LHS, RHS, BIAS)                                   \
  BIAS ? (Node *)createFullyConnected(name, LHS, RHS, BIAS)                    \
       : (Node *)createMatMul(name, LHS, RHS)

  // Operator name.
  const std::string &opName = namePrefix.str();

  // Get all size parameters.
  dim_t numDirections = (direction == RnnDirection::Bidirectional) ? 2 : 1;
  assert(X.dims().size() == 3 &&
         "ONNX LSTM input 'X' should have 3 dimensions!");
  dim_t seqLength = X.dims()[0];
  dim_t batchSize = X.dims()[1];
  dim_t inputSize = X.dims()[2];

  // Validate W size.
  assert(W.dims().size() == 3 &&
         "ONNX LSTM input 'W' should have 3 dimensions!");
  assert(W.dims()[0] == numDirections && W.dims()[1] == 4 * hiddenSize &&
         W.dims()[2] == inputSize && "ONNX LSTM 'W' tensor size invalid!");

  // Validate R size.
  assert(R.dims().size() == 3 &&
         "ONNX LSTM input 'R' should have 3 dimensions!");
  assert(R.dims()[0] == numDirections && R.dims()[1] == 4 * hiddenSize &&
         R.dims()[2] == hiddenSize && "ONNX LSTM 'R' tensor size invalid!");

  // Validate B size.
  if (B.getNode()) {
    assert(B.dims().size() == 2 &&
           "ONNX LSTM input 'B' should have 2 dimensions!");
    assert(B.dims()[0] == numDirections && B.dims()[1] == 8 * hiddenSize &&
           "ONNX LSTM 'B' tensor size invalid!");
  }

  // Validate initial_h size.
  assert(initial_h.getNode() &&
         "ONNX LSTM input 'initial_h' is mandatory. Null provided!");
  assert(initial_h.dims().size() == 3 &&
         "ONNX LSTM input 'initial_h' should have 2 dimensions!");
  assert(initial_h.dims()[0] == numDirections &&
         initial_h.dims()[1] == batchSize &&
         initial_h.dims()[2] == hiddenSize &&
         "ONNX LSTM 'initial_h' tensor size invalid!");

  // Validate initial_c size.
  assert(initial_c.getNode() &&
         "ONNX LSTM input 'initial_c' is mandatory. Null provided!");
  assert(initial_c.dims().size() == 3 &&
         "ONNX LSTM input 'initial_c' should have 2 dimensions!");
  assert(initial_c.dims()[0] == numDirections &&
         initial_c.dims()[1] == batchSize &&
         initial_c.dims()[2] == hiddenSize &&
         "ONNX LSTM 'initial_c' tensor size invalid!");

  // Validate P size.
  if (P.getNode()) {
    assert(P.dims().size() == 2 &&
           "ONNX LSTM input 'P' should have 2 dimensions!");
    assert(P.dims()[0] == numDirections && P.dims()[1] == 3 * hiddenSize &&
           "ONNX LSTM 'P' tensor size invalid!");
  }

  // Validate number of activations.
  assert(activations.size() == numDirections * 3 &&
         "ONNX LSTM activations vector invalid!");

  // Create X slices.
  std::vector<Node *> Xslices;
  for (dim_t t = 0; t < seqLength; t++) {
    auto XsliceName = opName + ".X" + std::to_string(t) + ".slice";
    Node *Xt = createSlice(XsliceName, X, LSTM_X_SLICE_RANGE(t));
    auto XreshapeName = opName + ".X" + std::to_string(t) + ".reshape";
    Xt = createReshape(XreshapeName, Xt, {batchSize, inputSize});
    Xslices.push_back(Xt);
  }

  // Lambda to load forward/backward LSTM cell.
  auto loadLSTMCell = [&](bool forward, std::vector<NodeValue> &Yslices,
                          NodeValue &Hslice, NodeValue &Cslice) {
    // Name prefix.
    std::string dirLabel = forward ? ".fw" : ".bw";
    std::string prefix = opName + ((numDirections > 1) ? dirLabel : "");

    // Slice index used for creating weights slices.
    dim_t sliceIdx0 = 0;
    if (direction == RnnDirection::Bidirectional) {
      sliceIdx0 = forward ? 0 : 1;
    }

    // Activations.
    size_t activationOffset = sliceIdx0 * 3;
    auto activationF = activations[activationOffset + 0];
    auto activationG = activations[activationOffset + 1];
    auto activationH = activations[activationOffset + 2];

    // Create W slices (Required).
    NodeValue Wi =
        createSlice(prefix + ".Wi.", W, LSTM_W_SLICE_RANGE(sliceIdx0, 0));
    NodeValue Wo =
        createSlice(prefix + ".Wo.", W, LSTM_W_SLICE_RANGE(sliceIdx0, 1));
    NodeValue Wf =
        createSlice(prefix + ".Wf.", W, LSTM_W_SLICE_RANGE(sliceIdx0, 2));
    NodeValue Wc =
        createSlice(prefix + ".Wc.", W, LSTM_W_SLICE_RANGE(sliceIdx0, 3));

    Wi = createReshape(prefix + ".Wi.reshape", Wi, {hiddenSize, inputSize});
    Wo = createReshape(prefix + ".Wo.reshape", Wo, {hiddenSize, inputSize});
    Wf = createReshape(prefix + ".Wf.reshape", Wf, {hiddenSize, inputSize});
    Wc = createReshape(prefix + ".Wc.reshape", Wc, {hiddenSize, inputSize});

    Wi = createTranspose(prefix + ".Wi.transp", Wi, {1, 0});
    Wo = createTranspose(prefix + ".Wo.transp", Wo, {1, 0});
    Wf = createTranspose(prefix + ".Wf.transp", Wf, {1, 0});
    Wc = createTranspose(prefix + ".Wc.transp", Wc, {1, 0});

    // Create R slices (Required).
    NodeValue Ri =
        createSlice(prefix + ".Ri.", R, LSTM_R_SLICE_RANGE(sliceIdx0, 0));
    NodeValue Ro =
        createSlice(prefix + ".Ro.", R, LSTM_R_SLICE_RANGE(sliceIdx0, 1));
    NodeValue Rf =
        createSlice(prefix + ".Rf.", R, LSTM_R_SLICE_RANGE(sliceIdx0, 2));
    NodeValue Rc =
        createSlice(prefix + ".Rc.", R, LSTM_R_SLICE_RANGE(sliceIdx0, 3));

    Ri = createReshape(prefix + ".Ri.reshape", Ri, {hiddenSize, hiddenSize});
    Ro = createReshape(prefix + ".Ro.reshape", Ro, {hiddenSize, hiddenSize});
    Rf = createReshape(prefix + ".Rf.reshape", Rf, {hiddenSize, hiddenSize});
    Rc = createReshape(prefix + ".Rc.reshape", Rc, {hiddenSize, hiddenSize});

    Ri = createTranspose(prefix + ".Ri.transp", Ri, {1, 0});
    Ro = createTranspose(prefix + ".Ro.transp", Ro, {1, 0});
    Rf = createTranspose(prefix + ".Rf.transp", Rf, {1, 0});
    Rc = createTranspose(prefix + ".Rc.transp", Rc, {1, 0});

    // Create B slices (optional).
    NodeValue bWi = nullptr;
    NodeValue bWo = nullptr;
    NodeValue bWf = nullptr;
    NodeValue bWc = nullptr;
    NodeValue bRi = nullptr;
    NodeValue bRo = nullptr;
    NodeValue bRf = nullptr;
    NodeValue bRc = nullptr;

    if (B) {

      bWi = createSlice(prefix + ".bWi.", B, LSTM_B_SLICE_RANGE(sliceIdx0, 0));
      bWo = createSlice(prefix + ".bWo.", B, LSTM_B_SLICE_RANGE(sliceIdx0, 1));
      bWf = createSlice(prefix + ".bWf.", B, LSTM_B_SLICE_RANGE(sliceIdx0, 2));
      bWc = createSlice(prefix + ".bWc.", B, LSTM_B_SLICE_RANGE(sliceIdx0, 3));
      bRi = createSlice(prefix + ".bRi.", B, LSTM_B_SLICE_RANGE(sliceIdx0, 4));
      bRo = createSlice(prefix + ".bRo.", B, LSTM_B_SLICE_RANGE(sliceIdx0, 5));
      bRf = createSlice(prefix + ".bRf.", B, LSTM_B_SLICE_RANGE(sliceIdx0, 6));
      bRc = createSlice(prefix + ".bRc.", B, LSTM_B_SLICE_RANGE(sliceIdx0, 7));

      bWi = createReshape(prefix + ".bWi.reshape", bWi, {hiddenSize});
      bWo = createReshape(prefix + ".bWo.reshape", bWo, {hiddenSize});
      bWf = createReshape(prefix + ".bWf.reshape", bWf, {hiddenSize});
      bWc = createReshape(prefix + ".bWc.reshape", bWc, {hiddenSize});
      bRi = createReshape(prefix + ".bRi.reshape", bRi, {hiddenSize});
      bRo = createReshape(prefix + ".bRo.reshape", bRo, {hiddenSize});
      bRf = createReshape(prefix + ".bRf.reshape", bRf, {hiddenSize});
      bRc = createReshape(prefix + ".bRc.reshape", bRc, {hiddenSize});
    }

    // Create P slices (optional).
    NodeValue Pi = nullptr;
    NodeValue Po = nullptr;
    NodeValue Pf = nullptr;

    if (P) {

      Pi = createSlice(prefix + ".Pi.", P, LSTM_P_SLICE_RANGE(sliceIdx0, 0));
      Po = createSlice(prefix + ".Po.", P, LSTM_P_SLICE_RANGE(sliceIdx0, 1));
      Pf = createSlice(prefix + ".Pf.", P, LSTM_P_SLICE_RANGE(sliceIdx0, 2));

      // Repeat P slices to match [batchSize, hiddenSize].
      Pi = createTile(prefix + ".Pi.repeat", Pi, batchSize, 0);
      Po = createTile(prefix + ".Po.repeat", Po, batchSize, 0);
      Pf = createTile(prefix + ".Pf.repeat", Pf, batchSize, 0);
    }

    // Create H slice for this direction.
    Node *Hinit = createSlice(prefix + ".H.slice", initial_h,
                              LSTM_H_SLICE_RANGE(sliceIdx0));
    Hinit =
        createReshape(prefix + ".H.reshape", Hinit, {batchSize, hiddenSize});

    // Create C slice for this direction.
    Node *Cinit = createSlice(prefix + ".C.slice", initial_c,
                              LSTM_C_SLICE_RANGE(sliceIdx0));
    Cinit =
        createReshape(prefix + ".C.reshape", Cinit, {batchSize, hiddenSize});

    // Initialize.
    Node *Ht = Hinit;
    Node *Ct = Cinit;

    // Unroll LSTM cell for all time steps.
    for (size_t t = 0; t < seqLength; t++) {

      // Input for current time step.
      // For the reverse LSTM cell the inputs are provided in reverse order.
      Node *Xt = forward ? Xslices[t] : Xslices[seqLength - 1 - t];

      // Forget gate: ft = f(Xt * Wf + bWf + Ht-1 * Rf + bRf + Pf . Ct-1).
      Node *ft = createAdd(prefix + ".F.add1",
                           LSTM_CREATE_FC(prefix + ".F.fc1", Xt, Wf, bWf),
                           LSTM_CREATE_FC(prefix + ".F.fc2", Ht, Rf, bRf));
      if (Pf) {
        ft = createAdd(prefix + ".F.add2", ft,
                       createMul(prefix + ".F.mult", Pf, Ct));
      }
      ft = activationF(prefix + ".F.act", ft);

      // Cell state candidate: ctild = g(Xt * Wc + bWc + Ht-1 * Rc + bRc).
      Node *ctild =
          createAdd(prefix + ".Ctild.add",
                    LSTM_CREATE_FC(prefix + ".Ctild.fc1", Xt, Wc, bWc),
                    LSTM_CREATE_FC(prefix + ".Ctild.fc2", Ht, Rc, bRc));
      ctild = activationG(prefix + ".Ctild.act", ctild);

      // Input gate:
      // For inputForget == true:
      //   it = 1 - ft.
      // For inputForget == false:
      //   it = f(Xt * Wi + bWi + Ht-1 * Ri + bRi + Pi . Ct-1).
      Node *it;
      if (inputForget) {
        auto splatTy = ft->getNthResult(0).getType();
        it = createSub(prefix + ".I.sub",
                       createSplat(prefix + ".I.splat", splatTy, 1.0), ft);
      } else {
        it = createAdd(prefix + ".I.add1",
                       LSTM_CREATE_FC(prefix + ".I.fc1", Xt, Wi, bWi),
                       LSTM_CREATE_FC(prefix + ".I.fc2", Ht, Ri, bRi));
        if (Pi) {
          it = createAdd(prefix + ".I.add2", it,
                         createMul(prefix + ".I.mult", Pi, Ct));
        }
        it = activationF(prefix + ".I.act", it);
      }

      // Cell state update: Ct = ft . Ct-1 + it . ctild.
      Ct = createAdd(prefix + ".C.add", createMul(prefix + ".C.mult1", ft, Ct),
                     createMul(prefix + ".C.mult2", it, ctild));

      // Output gate: ot = f(Xt * Wo + bWo + Ht-1 * Ro + bRo + Po . Ct).
      Node *ot = createAdd(prefix + ".O.add1",
                           LSTM_CREATE_FC(prefix + ".O.fc1", Xt, Wo, bWo),
                           LSTM_CREATE_FC(prefix + ".O.fc2", Ht, Ro, bRo));
      if (Po) {
        ot = createAdd(prefix + ".O.add2", ot,
                       createMul(prefix + ".O.mult", Po, Ct));
      }
      ot = activationF(prefix + ".O.act", ot);

      // Hidden state update: Ht = ot . h(Ct).
      Ht =
          createMul(prefix + ".H.mult", ot, activationH(prefix + ".H.act", Ct));

      // Output.
      Yslices.push_back(Ht);
    }

    // Updated states nodes.
    Hslice = Ht;
    Cslice = Ct;
  }; // End of local lambda "loadLSTMCell".

  bool forwardEnabled = ((direction == RnnDirection::Forward) ||
                         (direction == RnnDirection::Bidirectional));
  bool backwardEnabled = ((direction == RnnDirection::Reverse) ||
                          (direction == RnnDirection::Bidirectional));

  std::vector<NodeValue> YSlices;
  std::vector<NodeValue> Hslices;
  std::vector<NodeValue> Cslices;

  // Load forward LSTM.
  std::vector<NodeValue> forwardYslices;
  if (forwardEnabled) {
    NodeValue forwardHslice;
    NodeValue forwardCslice;
    loadLSTMCell(/* forward */ true, forwardYslices, forwardHslice,
                 forwardCslice);
    Hslices.push_back(forwardHslice);
    Cslices.push_back(forwardCslice);
  }

  // Load backward LSTM.
  std::vector<NodeValue> backwardYslices;
  if (backwardEnabled) {
    NodeValue backwardHslice;
    NodeValue backwardCslice;
    loadLSTMCell(/* forward */ false, backwardYslices, backwardHslice,
                 backwardCslice);
    Hslices.push_back(backwardHslice);
    Cslices.push_back(backwardCslice);
  }

  // Gather Y slices.
  for (size_t t = 0; t < seqLength; t++) {
    if (forwardEnabled) {
      YSlices.push_back(forwardYslices[t]);
    }
    if (backwardEnabled) {
      YSlices.push_back(backwardYslices[seqLength - 1 - t]);
    }
  }

  // Concatenate Y slices.
  // Y size is [seqLength, numDirections, batchSize, hiddenSize].
  Y = createReshape(opName + ".Y.reshape",
                    createConcat(opName + ".Y.concat", YSlices, 0),
                    {seqLength, numDirections, batchSize, hiddenSize});

  // Concatenate Y_h slices.
  // Y_h size is [numDirections, batchSize, hiddenSize].
  Y_h = createReshape(opName + ".Y_h.reshape",
                      createConcat(opName + ".Y_h.concat", Hslices, 0),
                      {numDirections, batchSize, hiddenSize});

  // Concatenate Y_c slices.
  // Y_c size is [numDirections, batchSize, hiddenSize].
  Y_c = createReshape(opName + ".Y_c.reshape",
                      createConcat(opName + ".Y_c.concat", Cslices, 0),
                      {numDirections, batchSize, hiddenSize});

#undef LSTM_X_SLICE_RANGE
#undef LSTM_H_SLICE_RANGE
#undef LSTM_C_SLICE_RANGE
#undef LSTM_W_SLICE_RANGE
#undef LSTM_R_SLICE_RANGE
#undef LSTM_B_SLICE_RANGE
#undef LSTM_P_SLICE_RANGE
#undef LSTM_CREATE_FC
}

TraceEventNode *Function::createTraceEvent(llvm::StringRef eventName,
                                           llvm::StringRef eventType,
                                           Node *data, unsigned index) {
  std::string name = (getName() + "_" + eventName + "_instrumentation").str();
  return addNode(new TraceEventNode(name, data, eventName, eventType, index));
}

//===----------------------------------------------------------------------===//
//                   Graph dumping and printing
//===----------------------------------------------------------------------===//

void Function::dump() const {
  llvm::outs() << "Graph structure " << getName() << ":\n";
  for (auto &n : nodes_) {
    llvm::outs() << n.getDebugDesc();
  }
}

std::string Function::toString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dump(os);
  return os.str();
}

void Function::dump(llvm::raw_ostream &os) const {
  os << "Graph structure " << getName() << ":\n";
  for (auto &n : nodes_) {
    os << n.getDebugDesc();
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

    dumpNode(N, false);

    // Print edges for the predicate field, if it's used.
    if (N->hasPredicate()) {
      auto pred = N->getPredicate();
      size_t resNo = pred.getResNo();
      std::ostringstream edge;
      edge << pred.getNode()->getName().str() << ":"
           << pred.getNode()->getOutputName(resNo).str() << " -> "
           << N->getName().str() << ":w";
      dumpEdgeStyle(N, 0, pred, edge);
      edges_.insert(edge.str());
      visitNode(pred);
    }

    for (size_t i = 0; i < N->getNumInputs(); i++) {
      Node *to = N->getNthInput(i).getNode();
      size_t resNo = N->getNthInput(i).getResNo();

      std::ostringstream edge;
      edge << to->getName().str() << ":" << to->getOutputName(resNo).str()
           << " -> " << N->getName().str() << ":" << N->getInputName(i);
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
  }
};

std::string Function::dumpDAG() {
  llvm::SmallString<64> dotPath;
  llvm::sys::fs::createTemporaryFile("dotty_graph_dump", "dot", dotPath);
  dumpDAG(dotPath);

  return std::string(dotPath.begin(), dotPath.end());
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
  logStorageDeletion(functions_, *I);
  delete *I;
  constants_.erase(I);
}

void Module::erasePlaceholder(PlaceholderList::iterator I) {
  if (I == placeholders_.end()) {
    return;
  }

  logStorageDeletion(functions_, *I);
  delete *I;
  placeholders_.erase(I);
}

void Function::eraseNode(NodesList::iterator I) {
  // Log node deletion.
  logCtx_->logNodeDeletion(*I);

  nodes_.erase(I);
}

Constant *Module::getConstantByName(llvm::StringRef name) const {
  for (auto *V : getConstants()) {
    if (V->getName() == name)
      return V;
  }
  return nullptr;
}

Placeholder *Module::getPlaceholderByName(llvm::StringRef name) const {
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

PlaceholderList Function::findPlaceholders() {
  PlaceholderList list;
  for (auto &PH : parent_->getPlaceholders()) {
    for (auto &user : PH->getUsers()) {
      if (user.getUser()->getParent() == this) {
        list.push_back(PH);
        break;
      }
    }
  }
  return list;
}

PlaceholderList Function::findPlaceholders() const {
  PlaceholderList list;
  for (auto &PH : parent_->getPlaceholders()) {
    for (auto &user : PH->getUsers()) {
      if (user.getUser()->getParent() == this) {
        list.push_back(PH);
        break;
      }
    }
  }
  return list;
}

ConstList Function::findConstants() {
  ConstList list;
  for (auto &constant : parent_->getConstants()) {
    for (auto &user : constant->getUsers()) {
      if (user.getUser()->getParent() == this) {
        list.push_back(constant);
        break;
      }
    }
  }
  return list;
}

ConstList Function::findConstants() const {
  ConstList list;
  for (auto &constant : parent_->getConstants()) {
    for (auto &user : constant->getUsers()) {
      if (user.getUser()->getParent() == this) {
        list.push_back(constant);
        break;
      }
    }
  }
  return list;
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
    if (N.hasPredicate()) {
      copy->setPredicate(N.getPredicate());
    }
  }

  // At this point we have a new invalid function that points into nodes in
  // the original function. Here we update the links between the nodes in the
  // new function.
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

    if (N.hasPredicate()) {
      auto it = currToNew.find(N.getPredicate().getNode());
      if (it != currToNew.end()) {
        N.setPredicate(NodeValue(it->second, N.getPredicate().getResNo()));
      }
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
static bool verifyNodeInput(const Node &N, size_t idx) {
  auto input = N.getNthInput(idx);
  auto *refN = input.getNode();
  // Check that N is in the use-list of the input node and there is a proper
  // entry for it.
  for (auto &U : refN->getUsers()) {
    if (U.getUser() == &N && *U.get() == input) {
      return true;
    }
  }

  report("Any node referencing another node N must be in the use-list of the "
         "node N");
  return false;
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

/// Insert \p node in \p nameToNode and report an error if the insertion fails.
/// \returns True if \p node was inserted into \p nameToNode. False otherwise.
/// When true is returned that means that \p nameToNode had no other nodes
/// registered under \p node.getName().
static bool
insertAndReport(std::unordered_map<std::string, const Node *> &nameToNode,
                const Node &node, const Function &function) {
  bool inserted = expectCompareTrue(
      "Node is not unique", nameToNode.insert({node.getName(), &node}).second,
      true, &function);
  if (!inserted) {
    std::string storage;
    llvm::raw_string_ostream msg(storage);
    /// Output extra information helping to find the error.
    msg << "The node with name '" << node.getName()
        << "' conflicts with a previous definition:\n";
    msg << "Current definition: " << node.getDebugDesc() << "\n";
    msg << "Previous definition: "
        << nameToNode[node.getName()]->getDebugDesc();
    report(msg.str().c_str());
    return false;
  }
  return true;
}

bool Function::verify(const Backend *backend) const {
  bool isValid = true;
  if (backend) {
    if (backend->getTensorLayoutRequirements().isEnabled()) {
      isValid &= expectCompareTrue(
          "Expected correct backend-specific layouts for the graph",
          verifyLayouts(*this, backend->getTensorLayoutRequirements()), true,
          this);
    }
  } else {
    // Always run verification pre-lowering / when we don't have backend:
    isValid &= expectCompareTrue(
        "Expected correct Glow canonical layouts for the graph",
        verifyLayouts(*this, CanonicalTensorLayout::getInstance()), true, this);
  }
  std::unordered_map<std::string, const Node *> nameToNode;

  for (auto *V : getParent()->getConstants()) {
    isValid &= insertAndReport(nameToNode, *V, *this);
    isValid &= expectCompareTrue("Constant and its payload must have same type",
                                 *V->getType(), V->getPayload().getType(), V);
  }

  nameToNode.clear();
  for (const auto &N : nodes_) {
    isValid &= insertAndReport(nameToNode, N, *this);
  }

  // Any node referenced by one of the graph nodes should be part of the
  // Graph.
  for (const auto &N : nodes_) {
    for (size_t idx = 0, e = N.getNumInputs(); idx < e; ++idx) {
      auto &input = N.getNthInput(idx);
      // Verify each input of N.
      isValid &= verifyNodeInput(N, idx);
      bool foundNode =
          std::find(nodes_.begin(), nodes_.end(), *input) != nodes_.end();
      isValid &= expectCompareTrue(
          "Every node referenced by one of the graph nodes should be part of "
          "the graph",
          foundNode || isGraphStorageNode(input, this), true, &N);
    }
  }

  // Check that all uses of each node refer to this node.
  for (const auto &N : nodes_) {
    for (const auto &U : N.getUsers()) {
      isValid &= expectCompareTrue<const Node *>(
          "All uses of a node should refer to this node", U.get()->getNode(),
          &N, &N);
      ;
    }
  }

  std::unordered_map<const Placeholder *, const Node *> placeholderWrittenTo;
  for (const auto &N : nodes_) {
    isValid &=
        expectCompareTrue("Node is not linked to the function it belongs",
                          N.getParent(), this, &N);
    isValid &= N.verify();
    // Make sure all the placeholders are at most written once, and that
    // constants are never written to.
    for (size_t idx = 0, e = N.getNumInputs(); idx < e; ++idx) {
      // Placeholders and Constants have no input, so they can only be
      // written to via overwritten inputs.
      if (!N.isOverwrittenNthInput(idx)) {
        continue;
      }

      const Node *nthInputNode = N.getNthInput(idx).getNode();
      isValid &= expectCompareTrue(
          "Constants can never be used as an overwritten input",
          isa<Constant>(nthInputNode), false, nthInputNode);

      // Unlike Constants, Placeholders can be used at most once as
      // overwritten inputs. Keep a map of Placeholders to Nodes that used
      // them as overwritten inputs, which is also used later to check for
      // read-after-write dependence violations.
      const auto *ph = dyn_cast<Placeholder>(nthInputNode);
      if (!ph) {
        continue;
      }
      auto varToFirstDef = placeholderWrittenTo.find(ph);
      bool writtenOnce = expectCompareTrue(
          "Placeholder has more than one write",
          varToFirstDef == placeholderWrittenTo.end(), true, ph);
      if (!writtenOnce) {
        isValid = false;
        std::string storage;
        llvm::raw_string_ostream msg(storage);

        msg << "Placeholder " << ph->getDebugDesc() << '\n';
        msg << "has more than one write; second writer found:\n";
        msg << N.getDebugDesc() << '\n';
        msg << varToFirstDef->second->getDebugDesc() << '\n';

        report(msg.str().c_str());
      }

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
      isValid &= expectCompareTrue(
          "Implicit read after write memory dependency may not be honored",
          user, varToWrite.second, user);
    }
  }
  return isValid;
}

SaveNode *glow::getOutputSave(Function *F, Placeholder *PH) {
  // if parent is set for PH, check if it is the same as provided Function.
  auto *PHP = PH->getParent();
  if (PHP != nullptr && F != PHP) {
    return nullptr;
  }
  for (auto &use : PH->getUsers()) {
    if (auto *save = llvm::dyn_cast<SaveNode>(use.getUser())) {
      if (save->getParent() == F && save->getPlaceholder() == PH) {
        return save;
      }
    }
  }
  return nullptr;
}

Node *glow::recursiveClone(Function *newF, Node *node, NodeMap &currToNew) {
  Node *copy = node->clone();
  currToNew[node] = copy;
  newF->addNode(copy);
  for (unsigned inp = 0, e = copy->getNumInputs(); inp < e; inp++) {
    auto input = copy->getNthInput(inp);
    auto it = currToNew.find(input.getNode());
    Node *newInput;
    if (it != currToNew.end()) {
      newInput = it->second;
    } else if (llvm::isa<Storage>(input.getNode())) {
      continue;
    } else {
      newInput = recursiveClone(newF, input.getNode(), currToNew);
    }
    copy->setNthInput(inp, NodeValue(newInput, input.getResNo()));
  }
  return copy;
}

namespace glow {
/// If \p PH is an output placeholder, \returns true.
/// This is determined by checking if the PH has a user which uses the PH as an
/// overwritten input.
bool isOutput(const Placeholder *PH, const Function &F) {
  for (const auto &use : PH->getUsers()) {
    // Look through the inputs of the PH's users. If an input is overwritten
    // check if it's the PH, if it is return true.
    auto *user = use.getUser();
    // Consider only users inside the same function.
    if (user->getParent() != &F) {
      continue;
    }
    for (unsigned i = 0, numInputs = user->getNumInputs(); i < numInputs; i++) {
      // If the input is not overwritten we can continue.
      if (!user->isOverwrittenNthInput(i)) {
        continue;
      }
      auto input = use.getUser()->getNthInput(i);
      if (input.getNode() == PH) {
        return true;
      }
    }
  }
  return false;
}

/// If \p PH is an input placeholder, \returns true.
bool isInput(const Placeholder *PH, const Function &F) {
  // Check that the PH is the input to a saveNode or is used by a non saveNode.
  for (const auto &use : PH->getUsers()) {
    // Consider only users inside the same function.
    if (use.getUser()->getParent() != &F) {
      continue;
    }
    // Check if PH is an input to a saveNode.
    if (auto *save = dyn_cast<SaveNode>(use.getUser())) {
      auto input = save->getInput();
      // If the PH is not an input to the saveNode we keep looking.
      if (input.getNode() != PH) {
        continue;
      }
    }
    return true;
  }
  return false;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Module &mod) {
  mod.dump(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Module *mod) {
  assert(mod != nullptr && "Null Pointer.");
  mod->dump(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Function &F) {
  F.dump(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Function *F) {
  assert(F != nullptr && "Null Pointer.");
  F->dump(os);
  return os;
}
} // namespace glow
