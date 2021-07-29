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

#include <ctype.h>
#include <memory>
#include <sstream>

#include <glog/logging.h>

#include "glow/Graph/Graph.h"
#include "glow/Graph/TensorLayout.h"
#include "glow/Graph/VerifierHelper.h"

using namespace glow;

/// Checks if two layout descriptions \p lhs and \p rhs describe the same layout
/// for a value of the type \p ty \returns true if layouts are the same.
bool glow::checkSameLayout(llvm::StringRef srcLayoutStr,
                           llvm::StringRef destLayoutStr, TypeRef ty,
                           const Node *parent, const std::string &prefix,
                           const TensorLayoutCommon &TLC, bool verbose) {
  auto srcLayout = TensorLayoutDescription(srcLayoutStr.str());
  auto destLayout = TensorLayoutDescription(destLayoutStr.str());
  // Are layouts literally the same?
  if (srcLayout.isSameLayout(destLayout)) {
    return true;
  }
  // Does the type satisfy the dest layout?
  if (TLC.isSatisfiedBy(ty, destLayout, &srcLayout)) {
    return true;
  }
  if (verbose) {
    report("\n\n\n");
    reportContext(parent);
    report("\n");
    report(prefix);
    report("\n");
    report(parent->getDebugDesc());
    report("\nMismatching layouts:\n");
    report("Provided layout\n");
    report(srcLayout.getDebugDesc());
    report("\n");
    report("Expected layout\n");
    report(destLayout.getDebugDesc());
    report("\n");
  }
  return false;
}

/// Verifies the correctness of tensor layouts in the function \p F using layout
/// requirements interface \p TLC.
bool glow::verifyLayouts(const Function &F, TensorLayoutCommon &TLC,
                         bool verbose) {
  bool isValid = true;
  for (const auto &N : F.getNodes()) {
    for (unsigned idx = 0, e = N.getNumInputs(); idx < e; ++idx) {
      auto input = N.getNthInput(idx);
      auto producerLayout =
          TLC.getNthResultLayoutRequirements(input.getNode(), input.getResNo());
      auto consumerLayout = TLC.getNthInputLayoutRequirements(&N, idx);
      std::string inputName = strFormat("input %d", idx);
      isValid &= checkSameLayout(producerLayout, consumerLayout,
                                 input.getType(), &N, inputName, TLC, verbose);
    }
  }
  return isValid;
}

TensorLayoutDescription::TensorLayoutDescription(const std::string &layoutStr) {
  if (layoutStr.empty()) {
    // 0-D output
    numDims_ = 0;
    return;
  }
  parse(layoutStr);
}

static bool isCustomExtension(llvm::StringRef text) {
  auto nsPos = text.find(':');
  if (nsPos == llvm::StringRef::npos) {
    return false;
  }
  auto bracketPos = text.find(']');
  assert(bracketPos != llvm::StringRef::npos && "Expected a closing bracket.");
  return (bracketPos > nsPos);
}

// Serialization format -
// The form for each dimension is as follows:
// 1. (mandatory) one char representing the current dimension. Either an
// alphabetic letter or '*'.
// 2. (optional) token for the start of optional dimension information: '['
// 3. (optional, must have 2. in place) namespace of the extension followed by
// ':'. must be provided for non-official backends. example: ocl:<information>
// 4. (optional,  must have 2. in place) end of the current default extension
// ']'
// 5. (optional) go to 2.
// NOTE: To add alignment information, the format is: a=<size_t>
// Example: N[a=32][namespace_for_unsupported:<bla>]HWC would represent 4-D
// tensor wherein N needs an alignment of 32 + some closed-backend requirements
// we don't know about. HWC have no restrictions.
// NOTES:
// 1. For each dimension, the identifier can be either a single english alphabet
// letter, either upper or lower case, or the star symbol.
// 2. We assume that a single letter is enough for each dimension, it makes
// parsing easier and avoids adding delimiters in the serialized format,
// however, we do have a constructor that (theoretically) accepts multi-letter
// dimensions. If we decide to expand the current support, we will need to add
// delimiters to the serialized form.
void TensorLayoutDescription::parse(llvm::StringRef text) {
  unsigned idx = 0;
  while (!text.empty()) {
    char curr = text.front();
    text = text.drop_front();
    if (curr == '\0' || isblank(curr)) {
      continue;
    }
    switch (curr) {
    case '[': {
      assert(idx > 0 && "Expected at least one parsed entry.");
      if (isCustomExtension(text)) {
        parseCustomExtensions(text, idx - 1);
      } else {
        parseOfficialExtensions(text, idx - 1);
      }
      break;
    }
    default: {
      DCHECK(isalpha(curr) || curr == '*')
          << "Expected an alphabetic letter or '*'., got: " << curr
          << " in string: " << text.str();
      std::string currStr(1, curr);
      dims_[idx].append(currStr);
      serializedLayout_.append(dims_[idx]);
      ++idx;
      assert(idx <= max_tensor_dimensions && "Too many tensor dimensions");
      break;
    }
    }
  }
  numDims_ = idx;
}

void TensorLayoutDescription::parseCustomExtensions(llvm::StringRef &text,
                                                    unsigned idx) {
  char curr = '[';
  dims_[idx].append("[");
  for (curr = text.front(); curr != ']' && !text.empty(); curr = text.front()) {
    dims_[idx].append(std::string(1, curr));
    text = text.drop_front();
  }
  assert(curr == ']' && "Expected closing ']' bracket.");
  text = text.drop_front();
  dims_[idx].append("]");
}

void TensorLayoutDescription::parseOfficialExtensions(llvm::StringRef &text,
                                                      unsigned idx) {
  // Only alignment so far - very simple parser:
  if (!text.consume_front("a=")) {
    llvm_unreachable("Unsupported layout extension.");
  }
  size_t align;
  if (text.consumeInteger(10, align)) {
    llvm_unreachable("Expected alignment info.");
  }
  if (!text.consume_front("]")) {
    llvm_unreachable("Expected closing ']'");
  }
  dims_[idx].append("[a=");
  dims_[idx].append(std::to_string(align));
  dims_[idx].append("]");
}

TensorLayoutDescription::TensorLayoutDescription(
    llvm::ArrayRef<std::string> dims) {
  assert(dims.size() <= max_tensor_dimensions && "Too many tensor dimensions");
  numDims_ = dims.size();
  for (unsigned idx = 0; idx < numDims_; ++idx) {
    dims_[idx] = dims[idx];
    serializedLayout_.append(dims_[idx]);
  }
}

const llvm::StringRef
TensorLayoutDescription::getNthDimDescription(size_t n) const {
  assert(n < numDims_ && "Wrong dimension number");
  return dims_[n];
}

size_t TensorLayoutDescription::getAlignment(size_t n) const {
  assert(n < numDims_ && "Wrong dimension number");
  return getAlignment(dims_[n]);
}

size_t TensorLayoutDescription::getAlignment(const std::string &s) const {
  std::string alignPrefix = "a=";
  size_t pos = s.find(alignPrefix);
  if (pos == std::string::npos) {
    // Default alignment:
    return 1;
  }
  auto align = s.substr(pos + alignPrefix.size());
  size_t ret;
  std::istringstream(align) >> ret;
  return ret;
}

/// \returns the position of ']' for extension at \p pos.
static size_t getEndOFExtension(llvm::StringRef dimStr, size_t pos) {
  size_t posEnd = pos;
  pos = pos - 1;
  assert(dimStr[pos] == '[' && "Expected start of align extension.");
  while (dimStr[posEnd] != ']') {
    ++posEnd;
    assert(posEnd < dimStr.size() && "Expected to find closing bracket.");
  }
  return posEnd;
}

void TensorLayoutDescription::removeAttribute(const std::string &name,
                                              std::string &dimStr) {
  size_t pos = dimStr.find(name);
  if (pos != std::string::npos) {
    size_t posEnd = getEndOFExtension(dimStr, pos);
    dimStr = dimStr.substr(0, pos - 1) + dimStr.substr(posEnd + 1);
  }
}

void TensorLayoutDescription::reconstructSerialized() {
  serializedLayout_ = "";
  for (size_t i = 0; i < numDims_; ++i) {
    serializedLayout_.append(dims_[i]);
  }
}

llvm::StringRef TensorLayoutDescription::setAlignment(size_t n, size_t align) {
  assert(n < numDims_ && "Wrong dimension number");
  return setAttribute(n, "a=", std::to_string(align));
}

llvm::StringRef TensorLayoutDescription::setAttribute(size_t n,
                                                      llvm::StringRef name,
                                                      llvm::StringRef value) {
  assert(n < numDims_ && "Wrong dimension number");
  auto &dimStr = dims_[n];
  // If we have a current name - remove it.
  removeAttribute(name.str(), dimStr);
  // Add new name information to dim:
  dimStr.append("[");
  dimStr.append(name.str());
  dimStr.append(value.str());
  dimStr.append("]");
  reconstructSerialized();
  return dimStr;
}

std::string TensorLayoutDescription::getAttribute(size_t n,
                                                  llvm::StringRef name) const {
  assert(n < numDims_ && "Wrong dimension number");
  size_t pos = dims_[n].find(name.str());
  if (pos == std::string::npos) {
    return "";
  }
  size_t posEnd = getEndOFExtension(dims_[n], pos);
  auto nameSZ = name.size();
  return dims_[n].substr(pos + nameSZ, posEnd - nameSZ - pos);
}

llvm::ArrayRef<std::string> TensorLayoutDescription::getDims() const {
  return llvm::makeArrayRef(dims_, numDims_);
}

std::string TensorLayoutDescription::getDebugDesc() const {
  std::string desc = "Layout: " + getSerializedLayout() + " [";
  for (unsigned idx = 0; idx < numDims_; idx++) {
    if (idx > 0) {
      desc += ", ";
    }
    desc += "name = ";
    desc += dims_[idx];
    desc += " : alignment = ";
    desc += std::to_string(getAlignment(idx));
    desc += " : index = ";
    desc += std::to_string(idx);
  }
  desc += "]";
  return desc;
}

bool TensorLayoutDescription::isSameLayout(
    const TensorLayoutDescription &rhs) const {
  if (numDims_ != rhs.numDims_) {
    return false;
  }
  if (serializedLayout_ != rhs.serializedLayout_) {
    return false;
  }
  return true;
}

static bool isAnyHelper(llvm::StringRef layout) {
  for (unsigned idx = 0, e = layout.size(); idx < e; ++idx) {
    if (layout[idx] != '*') {
      return false;
    }
  }
  return true;
}

bool TensorLayoutDescription::isAnyLayout() {
  return (isAnyHelper(getSerializedLayout()));
}

/// Definitions of different tensor layouts.
static std::string dimsNHWC[] = {
    {"N"},
    {"H"},
    {"W"},
    {"C"},
};
static std::string dimsNCHW[] = {
    {"N"},
    {"C"},
    {"H"},
    {"W"},
};
static std::string dimsHWNC[] = {
    {"H"},
    {"W"},
    {"N"},
    {"C"},
};
static std::string dimsCNHW[] = {
    {"C"},
    {"N"},
    {"H"},
    {"W"},
};
static std::string dims0D[]{
    {""},
};
static std::string dims1D[] = {
    {"N"},
};
static std::string dims2D[] = {
    {"*"},
    {"*"},
};
static std::string dims3D[] = {
    {"*"},
    {"*"},
    {"*"},
};
static std::string dims4D[] = {
    {"*"},
    {"*"},
    {"*"},
    {"*"},
};
static std::string dims5D[] = {
    {"*"}, {"*"}, {"*"}, {"*"}, {"*"},
};
static std::string dims6D[] = {
    {"*"}, {"*"}, {"*"}, {"*"}, {"*"}, {"*"},
};

static TensorLayoutDescription layoutNHWC(dimsNHWC);
static TensorLayoutDescription layoutNCHW(dimsNCHW);
static TensorLayoutDescription layoutHWNC(dimsHWNC);
static TensorLayoutDescription layoutCNHW(dimsCNHW);
static TensorLayoutDescription layout0D(dims0D);
static TensorLayoutDescription layout1D(dims1D);
static TensorLayoutDescription layout2D(dims2D);
static TensorLayoutDescription layout3D(dims3D);
static TensorLayoutDescription layout4D(dims4D);
static TensorLayoutDescription layout5D(dims5D);
static TensorLayoutDescription layout6D(dims6D);

/// Glow layouts for any specific number of dimensions.
static TensorLayoutDescription layoutsForDims[] = {
    layout0D, layout1D, layout2D, layout3D, layout4D, layout5D, layout6D,
};

TensorLayoutCommon::TensorLayoutCommon() : enabled_(false) {}

TensorLayoutCommon::TensorLayoutCommon(TensorLayoutCommon *ctxTensorLayout)
    : TensorLayoutCommon() {
  ctxTensorLayout_ = ctxTensorLayout;
}

TensorLayoutCommon::~TensorLayoutCommon() {}

LayoutNameToLayoutDescriptionTy &
TensorLayoutCommon::getLayoutNameToLayoutDescription() const {
  if (ctxTensorLayout_) {
    return ctxTensorLayout_->getLayoutNameToLayoutDescription();
  }
  return layoutNameToLayoutDescription_;
}

llvm::ArrayRef<TensorLayoutDescription>
TensorLayoutCommon::getLayoutsForDims() const {
  if (ctxTensorLayout_) {
    return ctxTensorLayout_->getLayoutsForDims();
  }
  return llvm::makeArrayRef(layoutsForDims);
}

static LayoutNameToLayoutDescriptionTy initLayoutNameToDescription() {
  LayoutNameToLayoutDescriptionTy map;
  map.insert(std::make_pair(
      "NCHW", glow::make_unique<TensorLayoutDescription>("NCHW")));
  map.insert(std::make_pair(
      "NHWC", glow::make_unique<TensorLayoutDescription>("NHWC")));
  map.insert(std::make_pair(
      "HWNC", glow::make_unique<TensorLayoutDescription>("HWNC")));
  map.insert(std::make_pair(
      "CNHW", glow::make_unique<TensorLayoutDescription>("CNHW")));
  map.insert(
      std::make_pair("N", glow::make_unique<TensorLayoutDescription>("N")));
  return map;
}

LayoutNameToLayoutDescriptionTy
    TensorLayoutCommon::layoutNameToLayoutDescription_ =
        initLayoutNameToDescription();

static TensorLayoutDescription *getLayoutFromName(
    const std::string &name,
    LayoutNameToLayoutDescriptionTy &layoutNameToLayoutDescription) {
  if (isAnyHelper(name)) {
    return nullptr;
  }
  auto it = layoutNameToLayoutDescription.find(name);
  if (it != layoutNameToLayoutDescription.end()) {
    return it->second.get();
  }
  // Add new layout to map:
  auto *ret = new TensorLayoutDescription(name);
  if (ret->getNumDims() == 0) {
    // empty / any layout.
    delete ret;
    ret = nullptr;
  }
  layoutNameToLayoutDescription.insert(
      std::make_pair(name, std::unique_ptr<TensorLayoutDescription>(ret)));
  return ret;
}

std::string TensorLayoutCommon::getDefaultNDLayout(unsigned dims) const {
  DCHECK_LE(dims, max_tensor_dimensions) << "Too many dimensions";
  return getLayoutsForDims()[dims].getSerializedLayout();
}

std::string
TensorLayoutCommon::getNthInputLayoutRequirementsImpl(const Node *node,
                                                      size_t n) {
  if (ctxTensorLayout_) {
    return ctxTensorLayout_->getNthInputLayoutRequirementsImpl(node, n);
  }
  return getNthInputLayoutRequirements(node, n);
}

std::string TensorLayoutCommon::getNthInputLayoutRequirements(const Node *node,
                                                              size_t n) {
  DCHECK_LT(n, node->getNumInputs()) << "Wrong input number";
  auto dims = node->getNthInput(n).getType()->dims();
  DCHECK_LE(dims.size(), max_tensor_dimensions) << "Too many dimensions";
  if (const auto *TN = llvm::dyn_cast<TransposeNode>(node)) {
    // The layout for the input of transpose is the same as the layout of the
    // operation's result producing this input.
    auto input = TN->getInput();
    return getNthResultLayoutRequirementsImpl(input.getNode(),
                                              input.getResNo());
  }
  if (const auto *QN = llvm::dyn_cast<QuantizeNode>(node)) {
    auto input = QN->getInput();
    return getNthResultLayoutRequirementsImpl(input.getNode(),
                                              input.getResNo());
  }
  if (const auto *CTN = llvm::dyn_cast<ConvertToNode>(node)) {
    auto input = CTN->getInput();
    return getNthResultLayoutRequirementsImpl(input.getNode(),
                                              input.getResNo());
  }
  if (const auto *QPN = llvm::dyn_cast<QuantizationProfileNode>(node)) {
    switch (n) {
    case QuantizationProfileNode::InputIndices::InputIdx: {
      auto input = QPN->getInput();
      return getNthResultLayoutRequirementsImpl(input.getNode(),
                                                input.getResNo());
    }
    default:
      return getLayoutsForDims()[dims.size()].getSerializedLayout();
    }
  }
  return getLayoutsForDims()[dims.size()].getSerializedLayout();
}

/// \returns The index of node \p N input \p in. NumInputs if not found.
static unsigned getInputIdx(const Node *N, NodeValue in) {
  for (unsigned idx = 0, e = N->getNumInputs(); idx < e; ++idx) {
    if (N->getNthInput(idx) == in) {
      return idx;
    }
  }
  return N->getNumInputs();
}

/// \returns true if getting the input's layout would cause an infinite loop.
static bool inputDoesNotKnowRequirements(const Node *node) {
  switch (node->getKind()) {
  case Kinded::Kind::TransposeNodeKind:
  case Kinded::Kind::QuantizeNodeKind:
  case Kinded::Kind::QuantizationProfileNodeKind:
  case Kinded::Kind::ConvertToNodeKind:
    return true;
  default:
    return false;
  }
}

std::string
TensorLayoutCommon::getNthResultLayoutRequirementsImpl(const Node *node,
                                                       size_t n) {
  if (ctxTensorLayout_) {
    return ctxTensorLayout_->getNthResultLayoutRequirementsImpl(node, n);
  }
  return getNthResultLayoutRequirements(node, n);
}

std::string TensorLayoutCommon::getNthResultLayoutRequirements(const Node *node,
                                                               size_t n) {
  DCHECK_LT(n, node->getNumResults()) << "Wrong output number";
  auto dims = node->getNthResult(n).getType()->dims();
  DCHECK_LE(dims.size(), max_tensor_dimensions) << "Too many dimensions";
  if (auto *TN = llvm::dyn_cast<TransposeNode>(node)) {
    // If the result of Transpose is a concrete layout, try to use this specific
    // layout.
    if (auto *layout = getLayoutFromName(TN->getLayout(),
                                         getLayoutNameToLayoutDescription())) {
      return layout->getSerializedLayout();
    }
    // Dynamically form the layout description for transposes.
    auto input = TN->getInput();
    while (inputDoesNotKnowRequirements(input)) {
      input = input.getNode()->getNthInput(0);
    }
    auto inputLayout =
        getNthInputLayoutRequirementsImpl(node, TransposeNode::InputIdx);
    auto inputLayoutHelper = TensorLayoutDescription(inputLayout);
    llvm::SmallVector<std::string, max_tensor_dimensions> dims(
        input.dims().size());
    auto shuffle = TN->getShuffle();
    for (unsigned idx = 0, e = inputLayoutHelper.getNumDims(); idx < e; ++idx) {
      dims[shuffle[idx]] = inputLayoutHelper.getNthDimDescription(idx).str();
    }
    TensorLayoutDescription tld(dims);
    return tld.getSerializedLayout();
  }
  if (auto *C = llvm::dyn_cast<Constant>(node)) {
    if (auto *layout = getLayoutFromName(C->getLayout(),
                                         getLayoutNameToLayoutDescription())) {
      return layout->getSerializedLayout();
    }
  }
  if (auto *PH = llvm::dyn_cast<Placeholder>(node)) {
    if (auto *layout = getLayoutFromName(PH->getLayout(),
                                         getLayoutNameToLayoutDescription())) {
      return layout->getSerializedLayout();
    }
  }
  if (auto *RN = llvm::dyn_cast<ReshapeNode>(node)) {
    if (auto *layout = getLayoutFromName(RN->getLayout(),
                                         getLayoutNameToLayoutDescription())) {
      return layout->getSerializedLayout();
    }
    auto result = node->getNthResult(n);
    auto *user = (*result.getUsers().begin()).getUser();
    unsigned inputIdx = getInputIdx(user, result);
    if (inputDoesNotKnowRequirements(user) ||
        inputIdx >= user->getNumInputs() || llvm::isa<TransposeNode>(user)) {
      return getLayoutsForDims()[dims.size()].getSerializedLayout();
    }
    auto layout = getNthInputLayoutRequirementsImpl(user, inputIdx);
    if (auto *layoutDesc =
            getLayoutFromName(layout, getLayoutNameToLayoutDescription())) {
      return layoutDesc->getSerializedLayout();
    }
  }
  return getLayoutsForDims()[dims.size()].getSerializedLayout();
}

bool TensorLayoutCommon::isSatisfiedBy(
    TypeRef ty, const TensorLayoutDescription &destLayout,
    const TensorLayoutDescription *srcLayout) const {
  // Strides of the type (in elements).
  auto strides = ty->strides();
  if (strides.size() != destLayout.getNumDims()) {
    return false;
  }
  unsigned idx = 0;
  for (const auto &dim : destLayout.getDims()) {
    // dim.alignment is in bytes, but strides are in elements.
    if (strides[idx] * ty->getElementSize() % destLayout.getAlignment(dim) !=
        0) {
      return false;
    }
    idx++;
  }
  if (!srcLayout) {
    return true;
  }
  if (destLayout.getNumDims() != srcLayout->getNumDims()) {
    return false;
  }
  // Names should be compatible. * is compatible to anything.
  if (srcLayout->getSerializedLayout().size() !=
      destLayout.getSerializedLayout().size()) {
    return false;
  }
  for (unsigned idx = 0, e = destLayout.getSerializedLayout().size(); idx < e;
       ++idx) {
    // '*' is compatible with anything.
    if (destLayout.getSerializedLayout()[idx] == '*' ||
        srcLayout->getSerializedLayout()[idx] == '*') {
      continue;
    }
    // Non-'*' are only compatible with themselves.
    if (srcLayout->getSerializedLayout()[idx] ==
        destLayout.getSerializedLayout()[idx]) {
      continue;
    }
    return false;
  }
  return true;
}

static std::string returnBaseReqOrNHWC(std::string baseReq, const Node *node) {
  auto baseReqHelper = TensorLayoutDescription(baseReq);
  if (!baseReqHelper.isSameLayout(
          CanonicalTensorLayout::getInstance().getLayoutsForDims()[4])) {
    return baseReq;
  }
  if (CanonicalTensorLayout::getInstance().acceptsAnyLayout(node)) {
    // These nodes accept any 4-D layout.
    return baseReqHelper.getSerializedLayout();
  }
  // NHWC is the canonical default
  return CanonicalTensorLayout::getInstance().getDefaultNDLayout(4);
}

std::string
CanonicalTensorLayout::getNthInputLayoutRequirements(const Node *node,
                                                     size_t n) {
  auto baseReq = TensorLayoutCommon::getNthInputLayoutRequirements(node, n);
  if (acceptsAnyLayout(node)) {
    return baseReq;
  }
  return returnBaseReqOrNHWC(baseReq, node);
}

std::string
CanonicalTensorLayout::getNthResultLayoutRequirements(const Node *node,
                                                      size_t n) {
  auto baseReq = TensorLayoutCommon::getNthResultLayoutRequirements(node, n);
  return returnBaseReqOrNHWC(baseReq, node);
}

std::string CanonicalTensorLayout::getDefaultNDLayout(unsigned dims) const {
  if (dims == 4) {
    return layoutNHWC.getSerializedLayout();
  }
  return TensorLayoutCommon::getDefaultNDLayout(dims);
}

static bool acceptsAnyInputLayout(const glow::Node *node) {
  switch (node->getKind()) {
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::BatchedReduceMeanNodeKind:
  case Kinded::Kind::BatchedAddNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
  case Kinded::Kind::BatchedReduceMinNodeKind:
  case Kinded::Kind::BatchedReduceMaxNodeKind:
  case Kinded::Kind::BatchNormalizationNodeKind:
  case Kinded::Kind::InstanceNormalizationNodeKind:
  case Kinded::Kind::BatchNormalizationGradNodeKind:
  case Kinded::Kind::PadNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
  case Kinded::Kind::MeanVarNormalizationNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::FlipNodeKind:
  case Kinded::Kind::SliceNodeKind:
  case Kinded::Kind::TileNodeKind:
  case Kinded::Kind::InsertTensorNodeKind:
  case Kinded::Kind::SGDNodeKind:
  case Kinded::Kind::BroadcastNodeKind:
  case Kinded::Kind::GaussianFillNodeKind:
  case Kinded::Kind::SpaceToDepthNodeKind:
  case Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind:
    return true;
  default:
    return false;
  }
}

bool CanonicalTensorLayout::acceptsAnyLayout(const Node *node) const {
  if (node->isDataParallel()) {
    return true;
  }
  // In the canonical representation, some nodes are input layout agnostic even
  // if they are not necessarily data parallel:
  return acceptsAnyInputLayout(node);
}
