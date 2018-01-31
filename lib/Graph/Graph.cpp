// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <unordered_set>

using namespace glow;
using llvm::dyn_cast;
using llvm::isa;

Graph::~Graph() {
  // Delete all of the nodes and the variables.
  for (auto it = nodes_.begin(), e = nodes_.end(); it != e;) {
    auto cur = it++;
    eraseNode(*cur);
  }

  for (auto it = vars_.begin(), e = vars_.end(); it != e;) {
    auto cur = it++;
    eraseVariable(*cur);
  }
}

TypeRef Graph::uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims) {
  return uniqueType(Type(elemTy, dims));
}

TypeRef Graph::uniqueType(ElemKind elemTy, llvm::ArrayRef<size_t> dims,
                          float scale, float offset) {
  return uniqueType(Type(elemTy, dims, scale, offset));
}

TypeRef Graph::uniqueTypeWithNewShape(TypeRef T, llvm::ArrayRef<size_t> dims) {
  if (T->isQuantizedType()) {
    return uniqueType(
        Type(T->getElementType(), dims, T->getScale(), T->getOffset()));

  } else {
    return uniqueType(Type(T->getElementType(), dims));
  }
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
                                Variable::VisibilityKind visibility,
                                Variable::TrainKind train, float val) {
  auto FT = uniqueType(*T);
  return addVar(new Variable(name, FT, visibility, train, val));
}

Variable *Graph::createVariable(ElemKind T, llvm::ArrayRef<size_t> dims,
                                llvm::StringRef name,
                                Variable::VisibilityKind visibility,
                                Variable::TrainKind train, float val) {
  auto FT = uniqueType(T, dims);
  return createVariable(FT, name, visibility, train, val);
}

Variable *Graph::createVariable(ElemKind T, llvm::ArrayRef<size_t> dims,
                                float scale, float offset, llvm::StringRef name,
                                Variable::VisibilityKind visibility,
                                Variable::TrainKind train, float val) {
  auto FT = uniqueType(T, dims, scale, offset);
  return createVariable(FT, name, visibility, train, val);
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
  advanceState(State::Differentiated);
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
                                Variable::VisibilityKind::Private,
                                Variable::TrainKind::Xavier, fanIn);

  auto *bias = createVariable(ElemKind::FloatTy, {depth}, "bias",
                              Variable::VisibilityKind::Private,
                              Variable::TrainKind::Broadcast, 0.1);

  auto OT = uniqueType(ElemKind::FloatTy, outDims);

  return addNode(new ConvolutionNode(name, OT, input, filter, bias, kernel,
                                     stride, pad, depth));
}

/// Check that the dimensions that are passed in when the convolution is
/// constructed are correct.
static void assertConvDims(NodeValue input, NodeValue filter, NodeValue bias,
                           size_t depth, size_t kernel, size_t stride,
                           size_t pad) {
  ShapeNHWC idim = ShapeNHWC(input.dims());
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");
  (void)idim;

  auto filterDims = filter->dims();
  assert(filterDims[0] == depth && filterDims[1] == kernel &&
         filterDims[2] == kernel && filterDims[3] == idim.c &&
         "Invalid filter dims");
  (void)filterDims;

  assert(bias->getType()->size() == depth && "Invalid bias size");
}

ConvolutionNode *Graph::createConv(llvm::StringRef name, NodeValue input,
                                   NodeValue filter, NodeValue bias,
                                   TypeRef outTy, size_t depth, size_t kernel,
                                   size_t stride, size_t pad) {
  assertConvDims(input, filter, bias, depth, kernel, stride, pad);
  return addNode(new ConvolutionNode(name, outTy, input, filter, bias, kernel,
                                     stride, pad, depth));
}

ConvolutionQNode *Graph::createConvQ(llvm::StringRef name, NodeValue input,
                                     NodeValue filter, NodeValue bias,
                                     TypeRef outTy, size_t depth, size_t kernel,
                                     size_t stride, size_t pad, float scale,
                                     float offset) {
  assertConvDims(input, filter, bias, depth, kernel, stride, pad);
  return addNode(new ConvolutionQNode(name, outTy, input, filter, bias, kernel,
                                      stride, pad, depth, scale, offset));
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
  TypeRef OT = getVoidTy();

  // if \p input is of type void, we cannot calculate the dimensions
  if (T != getVoidTy()) {
    auto idim = flattenCdr(input.dims());
    OT = uniqueTypeWithNewShape(T, {idim.first, outDepth});
  }

  return addNode(new FullyConnectedNode(name, OT, input, W, B, outDepth));
}

FullyConnectedNode *Graph::createFullyConnected(llvm::StringRef name,
                                                NodeValue input,
                                                size_t outDepth) {
  TypeRef T = input.getType();
  assert(
      T != getVoidTy() &&
      "cannot initialize untrained fully connected node with void input type");
  auto idim = flattenCdr(input.dims());

  size_t fanIn = idim.second;

  auto *W = createVariable(T->getElementType(), {idim.second, outDepth},
                           "weights", Variable::VisibilityKind::Private,
                           Variable::TrainKind::Xavier, fanIn);

  auto *B = createVariable(T->getElementType(), {outDepth}, "bias",
                           Variable::VisibilityKind::Private,
                           Variable::TrainKind::Broadcast, 0.1);

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
  auto TR = uniqueTypeWithNewShape(input.getType(), shape);
  assert(TR->size() == input.getType()->size() &&
         "Reshape to a different size");
  return addNode(new ReshapeNode(name, TR, input, shape.vec()));
}

TransposeNode *Graph::createTranspose(llvm::StringRef name, NodeValue input,
                                      llvm::ArrayRef<unsigned> shuffle) {
  llvm::SmallVector<size_t, 6> shape;
  auto dims = input.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  auto NT = uniqueTypeWithNewShape(input.getType(), shape);
  return addNode(new TransposeNode(name, NT, input, shuffle.vec()));
}

BroadcastNode *Graph::createBroadcast(llvm::StringRef name, NodeValue input,
                                      llvm::ArrayRef<size_t> shape,
                                      unsigned axis) {
  auto TR = uniqueType(input.getType()->getElementType(), shape);
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

IntrinsicNode *Graph::createIntrinsicNode(llvm::StringRef name,
                                          llvm::StringRef identifier,
                                          llvm::ArrayRef<Node *> inputs,
                                          llvm::ArrayRef<TypeRef> outputs,
                                          void *saved) {
  std::vector<NodeValue> ops;
  ops.reserve(inputs.size());
  for (auto &I : inputs) {
    ops.emplace_back(I);
  }
  return addNode(new IntrinsicNode(name, outputs, ops, identifier, saved));
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

  auto NT = uniqueTypeWithNewShape(input.getType(), shape);
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
                              Variable::VisibilityKind::Private,
                              Variable::TrainKind::Broadcast, 0.);
  auto *gamma = createVariable(ElemKind::FloatTy, {channels}, "gamma",
                               Variable::VisibilityKind::Private,
                               Variable::TrainKind::Broadcast, 1.0);

  auto *mean = createVariable(ElemKind::FloatTy, {channels}, "mean",
                              Variable::VisibilityKind::Private,
                              Variable::TrainKind::None);
  auto *variance = createVariable(ElemKind::FloatTy, {channels}, "variance",
                                  Variable::VisibilityKind::Private,
                                  Variable::TrainKind::None);

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

SelectNode *Graph::createSelect(llvm::StringRef name, NodeValue Cond,
                                NodeValue LHS, NodeValue RHS) {
  assert(LHS.dims() == RHS.dims() && "Invalid operand shapes");
  assert(Cond.dims() == RHS.dims() && "Invalid operand shapes");
  return addNode(new SelectNode(name, Cond, LHS, RHS));
}

SplatNode *Graph::createSplat(llvm::StringRef name, TypeRef ty, float value) {
  return addNode(new SplatNode(name, ty, value));
}

BatchedMatMulNode *Graph::createBatchedMatMul(llvm::StringRef name,
                                              NodeValue LHS, NodeValue RHS) {
  auto LT = LHS.getType();
  auto RT = RHS.getType();
  auto LDims = LT->dims();
  auto RDims = RT->dims();

  assert(LDims.size() == 3);
  assert(RDims.size() == 3);
  assert(LHS.getType()->getElementType() == RHS->getElementType());

  size_t a0 = LDims[0];
  size_t a1 = LDims[1];
  size_t a2 = LDims[2];

  size_t b0 = RDims[0];
  size_t b1 = RDims[1];
  size_t b2 = RDims[2];

  assert(((a0 == 1) || (b0 == 1) || (a0 == b0)) &&
         "Batch size must be broadcasted or identical");

  // Select the batch size. If the left operand is broadcast (value 1), select
  // the RHS.
  size_t N = (a0 != 1 ? a0 : b0);

  assert(a2 == b1 && "Row size of LHS is not equal to the column size of RHS.");
  (void)a1;
  (void)a2;
  (void)b1;
  (void)b2;

  auto Ty = uniqueTypeWithNewShape(LHS.getType(), {N, a1, b2});
  return addNode(new BatchedMatMulNode(name, Ty, LHS, RHS));
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
      createVariable(input.getType(), name, Variable::VisibilityKind::Private,
                     Variable::TrainKind::None);

  std::string nodeName{"_save_"};
  nodeName += name;
  return addNode(new SaveNode(nodeName, input, dest));
}

SaveNode *Graph::createSave(llvm::StringRef name, NodeValue input,
                            Variable *output) {
  return addNode(new SaveNode(name, input, output));
}

QuantizationProfileNode *Graph::createQuantizationProfile(llvm::StringRef name,
                                                          NodeValue input) {
  // TODO: this size is going to be refined. Just a placeholder now.
  const size_t numberOfBuckets = 2000U;
  auto *histogram = createVariable(
      ElemKind::FloatTy, {numberOfBuckets}, "histogram",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);
  // Intermediate data used for histogram calculations.
  // Min tensor value seen so far is kept on the first position.
  // Max tensor value seen so far is kept on the second position.
  auto *computationInfo = createVariable(
      ElemKind::FloatTy, {2}, "computationInfo",
      Variable::VisibilityKind::Private, Variable::TrainKind::None);

  return addNode(
      new QuantizationProfileNode(name, input, histogram, computationInfo));
}

TopKNode *Graph::createTopK(llvm::StringRef name, NodeValue input, size_t k) {
  auto inDims = input.dims();
  assert(inDims.size() > 0);
  assert(k <= inDims.back());
  llvm::SmallVector<size_t, 6> outDims(inDims.begin(), inDims.end());
  outDims.back() = k;
  return addNode(
      new TopKNode(name, uniqueType(input->getElementType(), outDims),
                   uniqueType(ElemKind::IndexTy, outDims), input, k));
}

void Graph::createSimpleRNN(llvm::StringRef namePrefix,
                            llvm::ArrayRef<Node *> inputs, unsigned batchSize,
                            unsigned hiddenSize, unsigned outputSize,
                            std::vector<Node *> &outputs) {
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front()->dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the state to zero.
  auto *HInit = createVariable(ElemKind::FloatTy, {batchSize, hiddenSize},
                               (namePrefix + ".initial_state").str(),
                               Variable::VisibilityKind::Public,
                               Variable::TrainKind::None);
  HInit->getPayload().zero();
  Node *Ht = HInit;

  float b = 0.1;
  auto *Whh = createVariable(ElemKind::FloatTy, {hiddenSize, hiddenSize},
                             (namePrefix + ".Whh").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *Bhh = createVariable(
      ElemKind::FloatTy, {hiddenSize}, (namePrefix + ".Bhh").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);
  auto *Wxh = createVariable(ElemKind::FloatTy, {inputSize, hiddenSize},
                             (namePrefix + ".Wxh").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, inputSize);
  auto *Bxh = createVariable(
      ElemKind::FloatTy, {hiddenSize}, (namePrefix + ".Bxh").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);
  auto *Why = createVariable(ElemKind::FloatTy, {hiddenSize, outputSize},
                             (namePrefix + ".Why").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *Bhy = createVariable(
      ElemKind::FloatTy, {outputSize}, (namePrefix + ".Bhy").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);

  // Un-roll backpropogation through time as a loop with the shared parameters.
  for (unsigned t = 0; t < timeSteps; t++) {
    auto fc1Name = (namePrefix + ".fc1." + std::to_string(t)).str();
    auto *FC1 = createFullyConnected(fc1Name, Ht, Whh, Bhh, hiddenSize);
    auto fc2Name = (namePrefix + ".fc2." + std::to_string(t)).str();
    auto *FC2 = createFullyConnected(fc2Name, inputs[t], Wxh, Bxh, hiddenSize);
    auto aName = (namePrefix + ".add." + std::to_string(t)).str();
    auto *A = createArithmetic(aName, FC1, FC2, ArithmeticNode::Mode::Add);
    auto tanhName = (namePrefix + ".tanh." + std::to_string(t)).str();
    auto *H = createTanh(tanhName, A);
    auto outName = (namePrefix + ".out." + std::to_string(t)).str();
    auto *O = createFullyConnected(outName, H, Why, Bhy, outputSize);
    outputs.push_back(O);

    Ht = H;
  };
}

void Graph::createGRU(llvm::StringRef namePrefix, llvm::ArrayRef<Node *> inputs,
                      unsigned batchSize, unsigned hiddenSize,
                      unsigned outputSize, std::vector<Node *> &outputs) {
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front()->dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the state to zero.
  auto *HInit = createVariable(
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
  auto *Wxz = createVariable(ElemKind::FloatTy, {inputSize, hiddenSize},
                             (namePrefix + ".Wxz").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, inputSize);
  auto *Whz = createVariable(ElemKind::FloatTy, {hiddenSize, hiddenSize},
                             (namePrefix + ".Whz").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *Bz1 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".bz1").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bUpdate);
  auto *Bz2 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".bz2").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bUpdate);
  float bReset = -1.0;
  // reset gate
  auto *Wxr = createVariable(ElemKind::FloatTy, {inputSize, hiddenSize},
                             (namePrefix + ".Wxr").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, inputSize);
  auto *Whr = createVariable(ElemKind::FloatTy, {hiddenSize, hiddenSize},
                             (namePrefix + ".Whr").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *Br1 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".br1").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bReset);
  auto *Br2 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".br2").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bReset);

  // hidden state
  float b = 0.1;
  auto *Wxh = createVariable(ElemKind::FloatTy, {inputSize, hiddenSize},
                             (namePrefix + ".Wxh").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, inputSize);
  auto *Whh = createVariable(ElemKind::FloatTy, {hiddenSize, hiddenSize},
                             (namePrefix + ".Whh").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *Bh1 = createVariable(
      ElemKind::FloatTy, {hiddenSize}, (namePrefix + ".bh1").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);
  auto *Bh2 = createVariable(
      ElemKind::FloatTy, {hiddenSize}, (namePrefix + ".bh2").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);

  // output layer
  auto *Why = createVariable(ElemKind::FloatTy, {hiddenSize, outputSize},
                             (namePrefix + ".Why").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *By = createVariable(
      ElemKind::FloatTy, {outputSize}, (namePrefix + ".by").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);

  auto *Ones = createVariable(
      ElemKind::FloatTy, {batchSize, hiddenSize}, (namePrefix + ".ones").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::None);

  Ones->getPayload().getHandle().clear(1.0);

  std::vector<Node *> outputNodes;
  for (unsigned t = 0; t < timeSteps; t++) {
    auto fc1Name = (namePrefix + ".fc1." + std::to_string(t)).str();
    auto fc2Name = (namePrefix + ".fc2." + std::to_string(t)).str();
    auto add1Name = (namePrefix + ".add1." + std::to_string(t)).str();
    auto sigmoid1Name = (namePrefix + ".sigmoid1." + std::to_string(t)).str();

    auto *Zt = createSigmoid(
        sigmoid1Name,
        createArithmetic(
            add1Name, createFullyConnected(fc1Name, Ht, Whz, Bz1, hiddenSize),
            createFullyConnected(fc2Name, inputs[t], Wxz, Bz2, hiddenSize),
            ArithmeticNode::Mode::Add));

    auto fc3Name = (namePrefix + ".fc3." + std::to_string(t)).str();
    auto fc4Name = (namePrefix + ".fc4." + std::to_string(t)).str();
    auto add2Name = (namePrefix + ".add2." + std::to_string(t)).str();
    auto sigmoid2Name = (namePrefix + ".sigmoid2." + std::to_string(t)).str();

    auto *Rt = createSigmoid(
        sigmoid2Name,
        createArithmetic(
            add2Name, createFullyConnected(fc3Name, Ht, Whr, Br1, hiddenSize),
            createFullyConnected(fc4Name, inputs[t], Wxr, Br2, hiddenSize),
            ArithmeticNode::Mode::Add));

    auto zhtName = (namePrefix + ".zh." + std::to_string(t)).str();
    auto *ZHt = createArithmetic(zhtName, Zt, Ht, ArithmeticNode::Mode::Mul);

    auto oneMinusZtName = (namePrefix + ".1-z." + std::to_string(t)).str();
    auto *OneMinusZt =
        createArithmetic(oneMinusZtName, Ones, Zt, ArithmeticNode::Mode::Sub);

    auto rhtName = (namePrefix + ".rh." + std::to_string(t)).str();
    auto *RHt = createArithmetic(rhtName, Rt, Ht, ArithmeticNode::Mode::Mul);

    auto fc5Name = (namePrefix + ".fc5." + std::to_string(t)).str();
    auto fc6Name = (namePrefix + ".fc6." + std::to_string(t)).str();
    auto add3Name = (namePrefix + ".add3." + std::to_string(t)).str();
    auto tanh1Name = (namePrefix + ".tanh1." + std::to_string(t)).str();

    auto *Ut = createTanh(
        tanh1Name,
        createArithmetic(
            add3Name, createFullyConnected(fc5Name, RHt, Whh, Bh1, hiddenSize),
            createFullyConnected(fc6Name, inputs[t], Wxh, Bh2, hiddenSize),
            ArithmeticNode::Mode::Add));

    auto oneMinusZtUtName = (namePrefix + "1.-zu." + std::to_string(t)).str();
    auto *OneMinusZtUt = createArithmetic(oneMinusZtUtName, OneMinusZt, Ut,
                                          ArithmeticNode::Mode::Mul);

    auto htName = (namePrefix + ".H." + std::to_string(t)).str();
    Ht = createArithmetic(htName, ZHt, OneMinusZtUt, ArithmeticNode::Mode::Add);

    auto outName = (namePrefix + ".out." + std::to_string(t)).str();
    auto *O = createFullyConnected(outName, Ht, Why, By, outputSize);
    outputs.push_back(O);
  }
};

void Graph::createLSTM(llvm::StringRef namePrefix,
                       llvm::ArrayRef<Node *> inputs, unsigned batchSize,
                       unsigned hiddenSize, unsigned outputSize,
                       std::vector<Node *> &outputs) {
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front()->dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the hidden and cell states to zero.
  auto *HInit = createVariable(
      ElemKind::FloatTy, {batchSize, hiddenSize}, "initial_hidden_state",
      Variable::VisibilityKind::Public, Variable::TrainKind::None);
  HInit->getPayload().zero();
  Node *Ht = HInit;

  auto *CInit = createVariable(
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
  //    C <- F. C + i . sigmoid(Wxc  * x + Whc * h + bc)
  // Hidden state:
  //    h <- O . tanh(C)

  // forget gate
  float bForget = 1.0;
  auto *Wxf = createVariable(ElemKind::FloatTy, {inputSize, hiddenSize},
                             (namePrefix + ".Wxf").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, inputSize);
  auto *Whf = createVariable(ElemKind::FloatTy, {hiddenSize, hiddenSize},
                             (namePrefix + ".Whf").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *Bf1 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".bf1").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bForget);
  auto *Bf2 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".bf2").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bForget);
  // input gate
  float bInput = 0.1;
  auto *Wxi = createVariable(ElemKind::FloatTy, {inputSize, hiddenSize},
                             (namePrefix + ".Wxi").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, inputSize);
  auto *Whi = createVariable(ElemKind::FloatTy, {hiddenSize, hiddenSize},
                             (namePrefix + ".Whi").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *Bi1 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".bi1").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bInput);
  auto *Bi2 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".bi2").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bInput);

  // output gate
  float bOutput = 0.1;
  auto *Wxo = createVariable(ElemKind::FloatTy, {inputSize, hiddenSize},
                             (namePrefix + ".Wxo").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, inputSize);
  auto *Who = createVariable(ElemKind::FloatTy, {hiddenSize, hiddenSize},
                             (namePrefix + ".Who").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *Bo1 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".bo1").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bOutput);
  auto *Bo2 = createVariable(ElemKind::FloatTy, {hiddenSize},
                             (namePrefix + ".bo2").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Broadcast, bOutput);

  // cell state
  float bCell = 0.1;
  auto *Wxc = createVariable(ElemKind::FloatTy, {inputSize, hiddenSize},
                             (namePrefix + ".Wxc").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, inputSize);
  auto *Whc = createVariable(ElemKind::FloatTy, {hiddenSize, hiddenSize},
                             (namePrefix + ".Whc").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *Bc1 = createVariable(
      ElemKind::FloatTy, {hiddenSize}, (namePrefix + ".bc1").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, bCell);
  auto *Bc2 = createVariable(
      ElemKind::FloatTy, {hiddenSize}, (namePrefix + ".bc2").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, bCell);

  // output layer
  float b = 0.1;
  auto *Why = createVariable(ElemKind::FloatTy, {hiddenSize, outputSize},
                             (namePrefix + ".Why").str(),
                             Variable::VisibilityKind::Private,
                             Variable::TrainKind::Xavier, hiddenSize);
  auto *By = createVariable(
      ElemKind::FloatTy, {outputSize}, (namePrefix + ".by").str(),
      Variable::VisibilityKind::Private, Variable::TrainKind::Broadcast, b);

  std::vector<Node *> outputNodes;
  for (unsigned t = 0; t < timeSteps; t++) {
    auto fc1Name = (namePrefix + ".fc1." + std::to_string(t)).str();
    auto fc2Name = (namePrefix + ".fc2." + std::to_string(t)).str();
    auto add1Name = (namePrefix + ".add1." + std::to_string(t)).str();
    auto sigmoid1Name = (namePrefix + ".sigmoid1." + std::to_string(t)).str();

    auto *Ft = createSigmoid(
        sigmoid1Name,
        createArithmetic(
            add1Name, createFullyConnected(fc1Name, Ht, Whf, Bf1, hiddenSize),
            createFullyConnected(fc2Name, inputs[t], Wxf, Bf2, hiddenSize),
            ArithmeticNode::Mode::Add));

    auto fc3Name = (namePrefix + ".fc3." + std::to_string(t)).str();
    auto fc4Name = (namePrefix + ".fc4." + std::to_string(t)).str();
    auto add2Name = (namePrefix + ".add2." + std::to_string(t)).str();
    auto sigmoid2Name = (namePrefix + ".sigmoid2." + std::to_string(t)).str();

    auto *It = createSigmoid(
        sigmoid2Name,
        createArithmetic(
            add2Name, createFullyConnected(fc3Name, Ht, Whi, Bi1, hiddenSize),
            createFullyConnected(fc4Name, inputs[t], Wxi, Bi2, hiddenSize),
            ArithmeticNode::Mode::Add));

    auto fc5Name = (namePrefix + ".fc5." + std::to_string(t)).str();
    auto fc6Name = (namePrefix + ".fc6." + std::to_string(t)).str();
    auto add3Name = (namePrefix + ".add3." + std::to_string(t)).str();
    auto sigmoid3Name = (namePrefix + ".sigmoid3." + std::to_string(t)).str();

    auto *Ot = createSigmoid(
        sigmoid3Name,
        createArithmetic(
            add3Name, createFullyConnected(fc5Name, Ht, Who, Bo1, hiddenSize),
            createFullyConnected(fc6Name, inputs[t], Wxo, Bo2, hiddenSize),
            ArithmeticNode::Mode::Add));

    auto fc7Name = (namePrefix + ".fc7." + std::to_string(t)).str();
    auto fc8Name = (namePrefix + ".fc8." + std::to_string(t)).str();
    auto add4Name = (namePrefix + ".add4." + std::to_string(t)).str();
    auto tanh1Name = (namePrefix + ".tanh1." + std::to_string(t)).str();

    auto *CRt = createTanh(
        tanh1Name,
        createArithmetic(
            add4Name, createFullyConnected(fc7Name, Ht, Whc, Bc1, hiddenSize),
            createFullyConnected(fc8Name, inputs[t], Wxc, Bc2, hiddenSize),
            ArithmeticNode::Mode::Add));

    auto mul1Name = (namePrefix + ".mul1." + std::to_string(t)).str();
    auto mul2Name = (namePrefix + ".mul2." + std::to_string(t)).str();
    Ct = createArithmetic(
        (namePrefix + ".C." + std::to_string(t)).str(),
        createArithmetic(mul1Name, Ft, Ct, ArithmeticNode::Mode::Mul),
        createArithmetic(mul2Name, It, CRt, ArithmeticNode::Mode::Mul),
        ArithmeticNode::Mode::Add);

    auto htName = (namePrefix + ".H." + std::to_string(t)).str();
    auto tanh2Name = (namePrefix + ".tanh2." + std::to_string(t)).str();
    Ht = createArithmetic(htName, Ot, createTanh(tanh2Name, Ct),
                          ArithmeticNode::Mode::Mul);

    auto outName = (namePrefix + ".out." + std::to_string(t)).str();
    auto *O = createFullyConnected(outName, Ht, Why, By, outputSize);
    outputs.push_back(O);
  }
};

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
    if (N->getNumResults()) {
      os_ << "|";
      std::vector<std::string> names(N->getNumResults());
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

  std::string uniqueNodeName(Node *N) {
    std::string buffer;
    llvm::raw_string_ostream stream(buffer);
    stream << '"' << N << '"';
    return stream.str();
  }

  /// Recursively traverses inputs of node \p N using Deep First Search.
  /// Each node will be visited no more than once. The method also dumps
  /// edges with their port identifiers in dotty format.
  void visitNode(Node *N) {
    if (visitedNodes_.find(N) != visitedNodes_.end())
      return;
    visitedNodes_.insert(N);

    for (size_t i = 0; i < N->getNumInputs(); i++) {
      Node *to = N->getNthInput(i).getNode();
      size_t resNo = N->getNthInput(i).getResNo();

      std::ostringstream edge;
      edge << uniqueNodeName(to) << ":" << to->getOutputName(resNo).str()
           << " -> " << uniqueNodeName(N) << ":" << N->getInputName(i).str();
      if (isa<Variable>(to)) {
        edge << "[style=bold, color=pink]";
      }
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
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  stream << "dotty_graph_dump_" << this << ".dot";
  dumpDAG(stream.str().c_str());
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
  if (Variable *V = dyn_cast<Variable>(N)) {
    return eraseVariable(V);
  }
  auto I = std::find(nodes_.begin(), nodes_.end(), N);
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

  // Any node referenced by one of the graph nodes should be part of the Graph.
  for (auto *N : nodes_) {
    for (size_t idx = 0, e = N->getNumInputs(); idx < e; ++idx) {
      assert((std::find(nodes_.begin(), nodes_.end(), N->getNthInput(idx)) !=
                  nodes_.end() ||
              std::find(vars_.begin(), vars_.end(), N->getNthInput(idx)) !=
                  vars_.end()) &&
             "Every node referenced by one of the graph"
             " nodes should be part of the graph");
    }
  }
}

void Graph::resetState() { state_ = State::Created; }

void Graph::advanceState(State s) {
  assert(state_ <= s && "Wrong order of actions with a graph.");
  state_ = s;
}
