/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "Importer.h"
#include "DebugMacros.h"
#include "NNPI.h"
#include "glow/Flags/Flags.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Quantization.h"
#include "nnpi_transformer.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>

#include "llvm/Support/CommandLine.h"

using namespace glow;

const std::string NNPIImporter::internalName_("_NNPI_");

static std::string nodeValueName(const glow::NodeValue &nv) {
  if (nv.getNode()->getKind() == glow::Kinded::Kind::PlaceholderKind) {
    return nv.getNode()->getName();
  } else if (nv.getNode()->getKind() == glow::Kinded::Kind::ConstantKind) {
    return std::string(nv.getNode()->getName()) + std::string("__const");
  }
  return std::string(nv.getNode()->getName()) + std::string("__res_") +
         std::to_string(nv.getResNo());
}

NNPIErrorCode glow::NNPIImporter::convertLengthsModeToLengthType(
    glow::LengthsMode mode, NNPI_LENGTH_TYPE &lengthType) {
  if (!GlowNNPISpecializeAllOneSLS) {
    mode = LengthsMode::Variable;
  }
  switch (mode) {
  case LengthsMode::Variable:
    lengthType = NNPI_LENGTH_VARIABLE;
    break;
  case LengthsMode::AllOne:
    lengthType = NNPI_LENGTH_ALL_ONE;
    break;
  default:
    return NNPI_INVALID_PARAM;
  }
  return NNPI_NO_ERROR;
}

glow::NNPIImporter::NNPIImporter(const NNPICompilationOptions &compileOptions)
    : internalNameCounter_(0), network_(NNPI_INVALID_NNPIHANDLE),
      compileOptions_(compileOptions) {
  ASSERT_LOG_NNPI_ERROR(nnpiNetworkCreate(&network_),
                        "Failed to create NNPI network");
}

/// Destructor.
glow::NNPIImporter::~NNPIImporter() {
  if (network_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_IF_ERROR(nnpiNetworkDestroy(network_),
                      "Failed to destroy NNPI network");
  }
}

NNPIErrorCode glow::NNPIImporter::addTensor(std::string name,
                                            bool alternativeLayout,
                                            const std::string &scaleTensor,
                                            const std::string &offsetTensor,
                                            bool forceSymlowp) {
  LOG_AND_RETURN_IF_NOT(
      ERROR, constants_.count(name),
      strFormat("Could not find Constants for tensor %s", name.c_str()),
      NNPI_INVALID_PARAM);
  const Tensor *t = constants_.at(name);

  NNPITensorDesc desc;
  desc.attributes.value = 0;
  desc.attributes.constant = 1;
  const auto &dims = t->dims();
  desc.numDims = dims.size();
  updateDescQuantFromGlow(t->getType(), desc, scaleTensor, offsetTensor,
                          forceSymlowp || compileOptions_.useSymlowp);
  updateDescDimsFromGlow(dims, desc, alternativeLayout);

  const void *pRawData(nullptr);
  int32_t *pDataInt32(nullptr); // Used for converting int64_t to int32_t
  switch (t->getType().getElementType()) {
  case glow::ElemKind::FloatTy:
  case glow::ElemKind::Float16Ty:
  case glow::ElemKind::Int8QTy:
  case glow::ElemKind::UInt8QTy:
  case glow::ElemKind::UInt8FusedQTy:
  case glow::ElemKind::UInt8FusedFP16QTy:
  case glow::ElemKind::UInt4FusedFP16QTy:
  case glow::ElemKind::Int32ITy:
  case glow::ElemKind::Int32QTy:
  case glow::ElemKind::BoolTy:
    pRawData = t->getUnsafePtr();
    break;
  case glow::ElemKind::Int64ITy: {
    auto *pDataInt64 = &(t->getHandle<int64_t>().raw(0));
    const size_t numElements(t->size());
    pDataInt32 = new int32_t[numElements];
    LOG_AND_RETURN_IF_NOT(
        ERROR, pDataInt32,
        "Failed to allocate temporary storage for Int64 tensor",
        NNPI_INVALID_PARAM);
    for (size_t i = 0; i < numElements; i++) {
      pDataInt32[i] = static_cast<int32_t>(pDataInt64[i]);
    }
    pRawData = static_cast<void *>(pDataInt32);
  } break;
  default:
    LOG_AND_RETURN_IF_NOT(ERROR, 0, "Unhandled tensor data type",
                          NNPI_INVALID_PARAM);
    break;
  }
  auto res = nnpiNetworkAddTensor(network_, name.c_str(), &desc, pRawData);

  if (pDataInt32) {
    delete[] pDataInt32;
  }
  return res;
}

NNPIErrorCode glow::NNPIImporter::addTensor(std::string name,
                                            const NNPITensorDesc &desc,
                                            const void *pData) {
  auto res = nnpiNetworkAddTensor(network_, name.c_str(), &desc, pData);
  return res;
}

NNPIErrorCode glow::NNPIImporter::addValueIfTensor(Value *v) {
  LOG_AND_RETURN_IF_NOT(ERROR, v, "Trying to add NULL value",
                        NNPI_INVALID_PARAM);
  auto *weight = llvm::dyn_cast<WeightVar>(v);
  if (weight &&
      weight->getMutability() == WeightVar::MutabilityKind::Constant &&
      constants_.count(v->getName())) {
    // Add a tensor.
    return addTensor(v->getName().begin());
  }
  return NNPI_NO_ERROR;
}

NNPIErrorCode glow::NNPIImporter::addValue(
    std::string name, const glow::Type *vType, bool alternativeLayout,
    bool input, bool output, const std::string &scaleTensor,
    const std::string &offsetTensor, bool forceSymlowp) {
  if (definedTensors_.count(name) && !alternativeLayout && !forceSymlowp &&
      !input && !output) {
    // The value was already defined and unless we're forcing a change in
    // layout/input/output/etc. Don't redefine it.
    return NNPI_NO_ERROR;
  } else {
    definedTensors_.insert(name);
  }

  NNPITensorDesc desc;
  desc.attributes.value = 0;
  desc.attributes.input = input;
  desc.attributes.output = output;
  updateDescQuantFromGlow(*vType, desc, scaleTensor, offsetTensor,
                          forceSymlowp || compileOptions_.useSymlowp);
  updateDescDimsFromGlow(vType->dims(), desc, alternativeLayout);

  const void *pRawData(nullptr);
  if (constants_.count(name)) {
    desc.attributes.constant = 1;
    const Tensor *t = constants_.at(name);
    switch (t->getType().getElementType()) {
    case glow::ElemKind::FloatTy:
      pRawData = &(t->getHandle<float>().raw(0));
      break;
    case glow::ElemKind::Float16Ty:
      pRawData = &(t->getHandle<float16_t>().raw(0));
      break;
    case glow::ElemKind::Int64ITy:
      pRawData = &(t->getHandle<int64_t>().raw(0));
      break;
    case glow::ElemKind::Int8QTy:
      pRawData = &(t->getHandle<int8_t>().raw(0));
      break;
    case glow::ElemKind::BoolTy:
      pRawData = &(t->getHandle<uint8_t>().raw(0));
      break;
    case glow::ElemKind::Int32QTy:
    default:
      LOG_AND_RETURN_IF_NOT(ERROR, 0, "Unhandled tensor data type",
                            NNPI_INVALID_PARAM);
      break;
    }
  }

  return nnpiNetworkAddTensor(network_, name.c_str(), &desc, pRawData);
}

void glow::NNPIImporter::updateDescDimsFromGlow(
    const llvm::ArrayRef<size_t> glowDims, NNPITensorDesc &desc,
    bool alternativeLayout) {
  desc.numDims = glowDims.size();
  for (size_t d = 0; d < desc.numDims; d++) {
    desc.dims[d] = glowDims[d];
  }
  switch (desc.numDims) {
  case 6:
    desc.layout = NNPI_LAYOUT_ANY;
    break;
  case 5:
    desc.layout = alternativeLayout ? NNPI_LAYOUT_NDHWC : NNPI_LAYOUT_ANY;
    break;
  case 4:
    desc.layout = alternativeLayout ? NNPI_LAYOUT_NHWC : NNPI_LAYOUT_ANY;
    break;
  case 3:
    desc.layout = NNPI_LAYOUT_CHW;
    break;
  case 2:
    desc.layout = alternativeLayout ? NNPI_LAYOUT_CN : NNPI_LAYOUT_NC;
    break;
  case 1:
    desc.layout = NNPI_LAYOUT_C;
    break;
  case 0: // Special case for Caffe/Pytorch scalar.
    desc.layout = NNPI_LAYOUT_C;
    desc.numDims = 1;
    desc.dims[0] = 1;
    break;
  default:
    LOG(ERROR) << "Invalid number of dims";
    break;
  }
}

template <class T> bool isBufferZero(T *buffer, size_t s) {
  for (size_t i = 0; i < s; i++) {
    if (!(buffer[i] == (T)0))
      return false;
  }
  return true;
}

bool glow::NNPIImporter::zeroes(const std::string &name) const {
  LOG_AND_RETURN_IF_NOT(ERROR, constants_.count(name), "Can't find tensor",
                        false);
  const Tensor *t = constants_.at(name);
  switch (t->getType().getElementType()) {
  case glow::ElemKind::FloatTy:
    return t->getHandle<float>().isZero();
  case glow::ElemKind::Float16Ty: {
    // Using isZero here leads to ambiguous overload for operator*
    // for now manually check if the buffer is all zeros.
    const auto dims = t->dims();
    size_t s = 1;
    for (size_t i = 0, e = dims.size(); i < e; i++) {
      s *= dims[i];
    }
    const float16_t *pRawData = reinterpret_cast<const float16_t *>(
        &(t->getHandle<float16_t>().raw(0)));
    for (size_t i = 0; i < s; i++) {
      if (!(pRawData[i] == (float16_t)0)) {
        return false;
      }
    }
    return true;
  }
  case glow::ElemKind::Int64ITy:
    return t->getHandle<int64_t>().isZero();
  case glow::ElemKind::Int32ITy:
    return t->getHandle<int32_t>().isZero();
  default:
    LOG_AND_RETURN_IF_NOT(ERROR, 0, "Unhandled tensor data type", false);
    break;
  }

  return false;
}

void glow::NNPIImporter::updateDescQuantFromGlow(
    const glow::Type &t, NNPITensorDesc &desc, const std::string &scaleTensor,
    const std::string &offsetTensor, bool forceSymlowp) {
  // Start with blanket defaults.
  desc.quantParams.params.gemlowp.scale = 1.f;
  desc.quantParams.params.gemlowp.offset = 0;
  switch (t.getElementType()) {
  case glow::ElemKind::FloatTy:
    LOG_ERROR_IF_NOT((scaleTensor.empty() && offsetTensor.empty()))
        << "Scales and offsets provided for Float";
    desc.quantParams.precision = NNPI_PRECISION_FLOAT32;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case glow::ElemKind::Float16Ty:
    LOG_ERROR_IF_NOT((scaleTensor.empty() && offsetTensor.empty()))
        << "Scales and offsets provided for Float16";
    desc.quantParams.precision = NNPI_PRECISION_FLOAT16;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case glow::ElemKind::Int64ITy:
    LOG_ERROR_IF_NOT((scaleTensor.empty() && offsetTensor.empty()))
        << "Scales and offsets provided for Int64";
    desc.quantParams.precision = NNPI_PRECISION_INT32;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case glow::ElemKind::Int8QTy:
    desc.quantParams.precision = NNPI_PRECISION_INT8;
    // If we have scales tensor, this is PCQ case.
    if (!scaleTensor.empty()) {
      // If there is no offsets, or Symlowp workaround is used and all offsets
      // are zero, the quantization type is SYMLOWP_PCQ.
      if (offsetTensor.empty() || (forceSymlowp && zeroes(offsetTensor))) {
        desc.quantParams.type = NNPI_QUANTIZATION_SYMLOWP_PCQ;
        std::strncpy(desc.quantParams.params.symlowpPCQ.scalesTensor,
                     scaleTensor.c_str(),
                     sizeof(desc.quantParams.params.symlowpPCQ.scalesTensor));
      } else { // Both scales and offsets are present.
        desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ;
        std::strncpy(desc.quantParams.params.gemmlowpPCQ.scalesTensor,
                     scaleTensor.c_str(),
                     sizeof(desc.quantParams.params.gemmlowpPCQ.scalesTensor));
        std::strncpy(desc.quantParams.params.gemmlowpPCQ.offsetsTensor,
                     offsetTensor.c_str(),
                     sizeof(desc.quantParams.params.gemmlowpPCQ.offsetsTensor));
      }
    } else {
      desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP;
      desc.quantParams.params.gemlowp.scale = t.getScale();
      desc.quantParams.params.gemlowp.offset = t.getOffset();
      if (forceSymlowp && (t.getOffset() == 0)) {
        // WA use SYMLOWP for zero offset tensors.
        DBG("SYMLOWP WA");
        desc.quantParams.type = NNPI_QUANTIZATION_SYMLOWP;
        desc.quantParams.params.symlowp.scale = t.getScale();
      }
    }

    break;
  case glow::ElemKind::UInt8QTy:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    if (!scaleTensor.empty()) {
      desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ;
      std::strncpy(desc.quantParams.params.gemmlowpPCQ.scalesTensor,
                   scaleTensor.c_str(),
                   sizeof(desc.quantParams.params.gemmlowpPCQ.scalesTensor));
      std::strncpy(desc.quantParams.params.gemmlowpPCQ.offsetsTensor,
                   offsetTensor.c_str(),
                   sizeof(desc.quantParams.params.gemmlowpPCQ.offsetsTensor));
    } else {
      desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP;
      desc.quantParams.params.gemlowp.scale = t.getScale();
      desc.quantParams.params.gemlowp.offset = t.getOffset();
    }
    break;
  case glow::ElemKind::UInt8FusedQTy:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ_FUSED;
    break;
  case glow::ElemKind::UInt8FusedFP16QTy:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ_FUSED_FP16;
    break;
  case glow::ElemKind::UInt4FusedFP16QTy:
    desc.quantParams.precision = NNPI_PRECISION_UINT8;
    desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP_PCQ_4BIT_FUSED_FP16;
    break;
  case glow::ElemKind::Int32ITy:
    desc.quantParams.precision = NNPI_PRECISION_INT32;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  case glow::ElemKind::Int32QTy:
    desc.quantParams.precision = NNPI_PRECISION_INT32;
    if (forceSymlowp && t.getOffset() == 0) {
      desc.quantParams.type = NNPI_QUANTIZATION_SYMLOWP;
    } else {
      desc.quantParams.type = NNPI_QUANTIZATION_GEMMLOWP;
    }
    desc.quantParams.params.gemlowp.scale = t.getScale();
    desc.quantParams.params.gemlowp.offset = t.getOffset();
    // This will be overwritten in addTensor for Int32QTy->Int8QTy WA.
    break;
  case glow::ElemKind::BoolTy:
    desc.quantParams.precision = NNPI_PRECISION_BOOLEAN;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    break;
  default:
    LOG(ERROR) << "Unhandled tensor data type";
    break;
  }
}

bool glow::NNPIImporter::isVariableUsingAlternativeLayout(Storage *v) {
  for (const auto &user : v->getUsers()) {
    switch (user.getUser()->getKind()) {
    case Kinded::Kind::ConvolutionNodeKind:
    case Kinded::Kind::Convolution3DNodeKind:
    case Kinded::Kind::AvgPoolNodeKind:
    case Kinded::Kind::MaxPoolNodeKind:
      return true;
    case Kinded::Kind::FullyConnectedNodeKind:
      return (v->getType()->dims().size() == 4);
    default: // Do nothing.
      break;
    }
  }
  return false;
}

NNPIErrorCode
glow::NNPIImporter::addIAExtentionPath(const std::string &extPath) {
  LOG_AND_RETURN_IF(ERROR, extPath.empty(), "Check if empty IA extension path.",
                    NNPI_INVALID_PARAM);
  std::ifstream extensionFile(extPath.c_str());
  LOG_AND_RETURN_IF_NOT(ERROR, extensionFile, "IA extension path not found.",
                        NNPI_INVALID_RESOURCE_NAME);
  iaExtensionPaths_.push_back(extPath);
  return NNPI_NO_ERROR;
}

NNPINetwork glow::NNPIImporter::importFunction(Function *F,
                                               const BackendOptions &opts) {
  if (compileOptions_.normalizeLayerNames) {
    std::map<std::string, uint32_t> type2count;
    std::map<std::string, glow::Node *> nodes;
    for (auto &N : F->getNodes()) {
      nodes[N.getName()] = &N;
    }
    auto *module = F->getParent();
    std::string prefix;

    if (module->getFunctions().size() > 1) {
      uint32_t netID = 0;
      for (const auto &function : module->getFunctions()) {
        if (function == F) {
          break;
        }
        netID++;
      }
      prefix = std::string("Net") + std::to_string(netID) + "_";
    }
    for (auto &pN : nodes) {
      std::string kindStr = pN.second->getKindName();
      if (type2count.count(kindStr) == 0) {
        type2count[kindStr] = 0;
      }
      auto counter = type2count[kindStr]++;
      auto newName = prefix + kindStr + "_" + std::to_string(counter);
      pN.second->setName(newName);
    }
  }

  // Clear internals.
  constants_.clear();
  readTensors_.clear();
  writeTensors_.clear();
  definedTensors_.clear();
  DBG_MEM_USAGE("ImportFunction <<");
  // Add constants.
  for (const auto &c : F->getParent()->getConstants()) {
    DBG("Importing Constant: " << c->getName().str() << " ("
                               << nodeValueName(c->getOutput()) << ") ["
                               << c->getKindName() << "]");
    std::string name = nodeValueName(c->getOutput());
    constants_.emplace(name, &c->getPayload());
    DBG_MEM_USAGE("ImportFunction: Add Constant Tensor: " << name);
    LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
        addTensor(nodeValueName(c->getOutput())), "Failed to add constant");
  }

  // Per node handling.
  for (auto &N : F->getNodes()) {
    // Check this type is handled.
    if (nodeImporters_.count(N.getKindName()) == 0) {
      DBG("-------------------------------------------------");
      DBG("Unhandled node type: " << N.getKindName());
      N.dump();
      DBG("-------------------------------------------------");
      return NNPI_INVALID_NNPIHANDLE;
    }

    DBG("Importing Node: " << N.getName().str() << " (" << N.getKindName()
                           << ")");
    // Set node inputs and outputs.
    for (unsigned i = 0, e = N.getNumInputs(); i < e; i++) {
      auto inVal = N.getNthInput(i);
      DBG("  Input: " << nodeValueName(inVal));
    }
    for (unsigned r = 0, e = N.getNumResults(); r < e; r++) {
      auto resVal = N.getNthResult(r);
      LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
          addValue(nodeValueName(resVal), resVal.getType()),
          "Failed to add intermediate");
      DBG("  Output: " << nodeValueName(resVal));
    }
    DBG_MEM_USAGE("ImportFunction import node: " << N.getKindName());
    // Import node.
    LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
        nodeImporters_.at(N.getKindName())->importNode(&N, *this),
        "Failed to import node");
  }

  // Handle placeholders (inputs/outputs).
  for (auto *v : F->getParent()->getPlaceholders()) {
    bool inputVar(readTensors_.count(v->getName()) &&
                  !writeTensors_.count(v->getName()));
    bool outputVar(!readTensors_.count(v->getName()) &&
                   writeTensors_.count(v->getName()));
    if (inputVar || outputVar) {
      LOG_NNPI_IF_ERROR_RETURN_INVALID_HANDLE(
          addValue(v->getName(), v->getType(),
                   isVariableUsingAlternativeLayout(v), inputVar, outputVar),
          "Failed to add placeholder");
      DBG("[--IO--] Setting IO variable: " << v->getName().str() << ", R:"
                                           << inputVar << ", W:" << outputVar
                                           << ", U:" << v->getNumUsers());
    } else {
      DBG("[--IO--] Unused Placeholder: " << v->getName().str());
    }
  }

  DBG_MEM_USAGE("ImportFunction call nnpiNetworkBuild");
  // Build network.
  NNPINetwork net;
  NNPIErrorCode res = nnpiNetworkBuild(network_);
  if (res != NNPI_NO_ERROR) {
    LOG(INFO) << "Failed to build network";
    LOG_NNPI_IF_ERROR(nnpiNetworkDestroy(network_),
                      "Failed to destroy NNPI network");
    net = NNPI_INVALID_NNPIHANDLE;
  } else {
    net = network_;
  }

  // Detach network from importer (if failed to build then it's already
  // destroyed, otherwise relinquish ownership to the backend).
  network_ = NNPI_INVALID_NNPIHANDLE;
  DBG_MEM_USAGE("ImportFunction done >>");
  return net;
}

// Node Importers ////////////////////////////////////////////////////////
template <class ConvType = ConvolutionNode, size_t convDims = 2>
class ConvolutionNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowConv = llvm::dyn_cast<ConvType>(n);

    std::string convStr = (convDims == 2) ? "Conv" : "Conv3D";
    LOG_AND_RETURN_IF_NOT(ERROR, glowConv, "Bad node type", NNPI_INVALID_PARAM);

    LOG_AND_RETURN_IF_NOT(ERROR, glowConv->getKernels().size() == convDims,
                          "[" + convStr + "] Invalid number of kernel sizes",
                          NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, glowConv->getPads().size() == 2 * convDims,
                          "[" + convStr + "] Invalid number of pads",
                          NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, glowConv->getStrides().size() == convDims,
                          "[" + convStr + "] Invalid number of strides",
                          NNPI_INVALID_PARAM);

    uint32_t kernel[convDims];
    uint32_t paddingStart[convDims];
    uint32_t paddingEnd[convDims];
    uint32_t stride[convDims];
    uint32_t dilation[convDims];

    ConvolutionNode *conv2DNode = llvm::dyn_cast<ConvolutionNode>(glowConv);
    for (size_t i = 0; i < convDims; i++) {
      kernel[i] = glowConv->getKernels()[i];
      stride[i] = glowConv->getStrides()[i];
      if (conv2DNode) {
        paddingStart[i] = glowConv->getPads()[i];
        paddingEnd[i] = glowConv->getPads()[convDims + i];
        dilation[i] = conv2DNode->getDilation();
      } else {
        paddingStart[i] = glowConv->getPads()[i * 2];
        paddingEnd[i] = glowConv->getPads()[i * 2 + 1];
        dilation[i] = 1;
      }
    }

    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowConv->getFilter()),
                           /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowConv->getBias())),
        "Failed to add tensor to NNPI");

    // Overwrite input/output values for layout.
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowConv->getInput()),
                          glowConv->getInput().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowConv->getResult()),
                          glowConv->getResult().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");

    importer.setUsedTensors({nodeValueName(glowConv->getInput()),
                             nodeValueName(glowConv->getFilter()),
                             nodeValueName(glowConv->getBias())},
                            {nodeValueName(glowConv->getResult())});

    return nnpiNetworkAddConvolutionOp(
        importer.getNetwork(), glowConv->getName().begin(),
        nodeValueName(glowConv->getInput()).c_str(),
        nodeValueName(glowConv->getResult()).c_str(),
        nodeValueName(glowConv->getFilter()).c_str(),
        glowConv->getBias() ? nodeValueName(glowConv->getBias()).c_str()
                            : nullptr,
        kernel, paddingStart, paddingEnd, stride, dilation, convDims,
        glowConv->getGroup());
  }
};

class TransposeNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowTranspose = llvm::dyn_cast<TransposeNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowTranspose, "Bad node type",
                          NNPI_INVALID_PARAM);

    const auto &glowOrder = glowTranspose->getShuffle();
    LOG_AND_RETURN_IF_NOT(ERROR, glowOrder.size() <= NNPI_MAX_DIMS,
                          "Bad dimansion", NNPI_INVALID_DIMS);

    uint32_t nnpiOrder[NNPI_MAX_DIMS];
    for (size_t i = 0, e = glowOrder.size(); i < e; i++) {
      nnpiOrder[i] = glowOrder[i];
    }

    importer.setUsedTensors({nodeValueName(glowTranspose->getInput())},
                            {nodeValueName(glowTranspose->getResult())});

    return nnpiNetworkAddTransposeOp(
        importer.getNetwork(), glowTranspose->getName().begin(),
        nodeValueName(glowTranspose->getInput()).c_str(),
        nodeValueName(glowTranspose->getResult()).c_str(), nnpiOrder,
        glowOrder.size());
  }
};

template <class PoolNodeType, NNPI_POOLING_TYPE poolType>
class PoolNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowPool = llvm::dyn_cast<PoolNodeType>(n);
    int inputDimension = glowPool->getInput().dims().size();
    int numDims = inputDimension - 2;
    LOG_AND_RETURN_IF_NOT(ERROR, numDims == 2 || numDims == 3,
                          "Input dimension is incorrect", NNPI_INVALID_PARAM);

    std::string poolStr = (numDims == 2) ? "Pool" : "Pool3D";
    LOG_AND_RETURN_IF_NOT(ERROR, glowPool, "Bad node type", NNPI_INVALID_PARAM);

    LOG_AND_RETURN_IF_NOT(ERROR, glowPool->getKernels().size() == numDims,
                          "[" + poolStr + "] Invalid number of kernel sizes",
                          NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, glowPool->getPads().size() == 2 * numDims,
                          "[" + poolStr + "] Invalid number of pads",
                          NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, glowPool->getStrides().size() == numDims,
                          "[" + poolStr + "] Invalid number of strides",
                          NNPI_INVALID_PARAM);

    std::vector<uint32_t> kernel(numDims);
    std::vector<uint32_t> paddingStart(numDims);
    std::vector<uint32_t> paddingEnd(numDims);
    std::vector<uint32_t> stride(numDims);
    bool countIncludePads = 1;
    if (auto *APN = llvm::dyn_cast<AvgPoolNode>(glowPool)) {
      countIncludePads = APN->getCountIncludePads();
    }

    for (size_t i = 0; i < numDims; i++) {
      kernel[i] = glowPool->getKernels()[i];
      stride[i] = glowPool->getStrides()[i];
      if (numDims == 2) {
        paddingStart[i] = glowPool->getPads()[i];
        paddingEnd[i] = glowPool->getPads()[numDims + i];
      } else {
        paddingStart[i] = glowPool->getPads()[i * 2];
        paddingEnd[i] = glowPool->getPads()[i * 2 + 1];
      }
    }

    // Overwrite input/output values for layout.
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowPool->getInput()),
                          glowPool->getInput().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowPool->getResult()),
                          glowPool->getResult().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");

    importer.setUsedTensors({nodeValueName(glowPool->getInput())},
                            {nodeValueName(glowPool->getResult())});

    return nnpiNetworkAddPoolingOp(
        importer.getNetwork(), glowPool->getName().begin(),
        nodeValueName(glowPool->getInput()).c_str(),
        nodeValueName(glowPool->getResult()).c_str(), NULL, kernel.data(),
        paddingStart.data(), paddingEnd.data(), stride.data(), numDims,
        poolType, !countIncludePads, 0);
  }
};

template <class AdaptivePoolNodeType, NNPI_POOLING_TYPE poolType>
class AdaptivePoolNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowPool = llvm::dyn_cast<AdaptivePoolNodeType>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowPool, "Bad node type", NNPI_INVALID_PARAM);

    // Overwrite input/output values for layout.
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowPool->getInput()),
                          glowPool->getInput().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowPool->getResult()),
                          glowPool->getResult().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");

    importer.setUsedTensors({nodeValueName(glowPool->getInput())},
                            {nodeValueName(glowPool->getResult())});

    return nnpiNetworkAddAdaptivePoolingOp(
        importer.getNetwork(), glowPool->getName().begin(),
        nodeValueName(glowPool->getInput()).c_str(),
        nodeValueName(glowPool->getResult()).c_str(), poolType);
  }
};

class FullyConnectedNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowFC = llvm::dyn_cast<FullyConnectedNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowFC, "Bad node type", NNPI_INVALID_PARAM);

    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowFC->getWeights()),
                           /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");

    // Overwrite input/output values for layout.
    const auto *input = glowFC->getInput().getNode();
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(input->getName(), input->getType(0),
                          input->getType(0)->dims().size() == 4),
        "Failed to add tensor to NNPI");
    const auto *result = glowFC->getResult().getNode();
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(result->getName(), result->getType(0),
                          result->getType(0)->dims().size() == 4),
        "Failed to add tensor to NNPI");

    importer.setUsedTensors({nodeValueName(glowFC->getInput()),
                             nodeValueName(glowFC->getWeights()),
                             nodeValueName(glowFC->getBias())},
                            {nodeValueName(glowFC->getResult())});

    return nnpiNetworkAddFullyConnectedOp(
        importer.getNetwork(), glowFC->getName().begin(),
        nodeValueName(glowFC->getInput()).c_str(),
        nodeValueName(glowFC->getResult()).c_str(),
        nodeValueName(glowFC->getWeights()).c_str(),
        glowFC->getBias() ? nodeValueName(glowFC->getBias()).c_str() : nullptr);
  }
};

class SoftMaxNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSM = llvm::dyn_cast<SoftMaxNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSM, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowSM->getInput())},
                            {nodeValueName(glowSM->getResult())});

    return nnpiNetworkAddSoftmaxOp(importer.getNetwork(),
                                   glowSM->getName().begin(),
                                   nodeValueName(glowSM->getInput()).c_str(),
                                   nodeValueName(glowSM->getResult()).c_str(),
                                   1); // Defaulting to axis 1 (C).
  }
};

class SaveNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSave = llvm::dyn_cast<SaveNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSave, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowSave->getInput())},
                            {nodeValueName(glowSave->getOutput())});

    return nnpiNetworkAddCopyOp(importer.getNetwork(),
                                glowSave->getName().begin(),
                                nodeValueName(glowSave->getInput()).c_str(),
                                nodeValueName(glowSave->getOutput()).c_str());
  }
};

class ReluNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowRelu = llvm::dyn_cast<ReluNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowRelu, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowRelu->getInput())},
                            {nodeValueName(glowRelu->getResult())});

    return nnpiNetworkAddReluOp(importer.getNetwork(),
                                glowRelu->getName().begin(),
                                nodeValueName(glowRelu->getInput()).c_str(),
                                nodeValueName(glowRelu->getResult()).c_str());
  }
};

class PReluNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowPRelu = llvm::dyn_cast<PReluNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowPRelu, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowPRelu->getInput()),
                             nodeValueName(glowPRelu->getSlope())},
                            {nodeValueName(glowPRelu->getResult())});

    return nnpiNetworkAddPReluOp(importer.getNetwork(),
                                 glowPRelu->getName().begin(),
                                 nodeValueName(glowPRelu->getInput()).c_str(),
                                 nodeValueName(glowPRelu->getResult()).c_str(),
                                 nodeValueName(glowPRelu->getSlope()).c_str());
  }
};

class GeluNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowGelu = llvm::dyn_cast<GeluNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowGelu, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowGelu->getInput())},
                            {nodeValueName(glowGelu->getResult())});

    return nnpiNetworkAddGeluOp(importer.getNetwork(),
                                glowGelu->getName().begin(),
                                nodeValueName(glowGelu->getInput()).c_str(),
                                nodeValueName(glowGelu->getResult()).c_str());
  }
};

template <class EltwiseNodeType, NNPI_ELTWISE_TYPE eltwiseType>
class BinaryEltwiseNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowEltwise = llvm::dyn_cast<EltwiseNodeType>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowEltwise, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowEltwise->getRHS()),
                             nodeValueName(glowEltwise->getLHS())},
                            {nodeValueName(glowEltwise->getResult())});

    NNPIObjectName inputNames[2];
    snprintf(inputNames[0], NNPI_MAX_STRING_LEN, "%s",
             nodeValueName(glowEltwise->getLHS()).c_str());
    snprintf(inputNames[1], NNPI_MAX_STRING_LEN, "%s",
             nodeValueName(glowEltwise->getRHS()).c_str());
    return nnpiNetworkAddElementwiseOp(
        importer.getNetwork(), glowEltwise->getName().begin(), inputNames, 2,
        nodeValueName(glowEltwise->getResult()).c_str(), eltwiseType);
  }
};

template <class EltwiseNodeType, NNPI_ELTWISE_TYPE eltwiseType>
class UnaryEltwiseNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowEltwise = llvm::dyn_cast<EltwiseNodeType>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowEltwise, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {
            nodeValueName(glowEltwise->getInput()),
        },
        {nodeValueName(glowEltwise->getResult())});

    NNPIObjectName inputNames[1];
    snprintf(inputNames[0], NNPI_MAX_STRING_LEN, "%s",
             nodeValueName(glowEltwise->getInput()).c_str());
    return nnpiNetworkAddElementwiseOp(
        importer.getNetwork(), glowEltwise->getName().begin(), inputNames, 1,
        nodeValueName(glowEltwise->getResult()).c_str(), eltwiseType);
  }
};

class ReshapeNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowReshape = llvm::dyn_cast<ReshapeNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowReshape, "Bad node type",
                          NNPI_INVALID_PARAM);

    NNPITensorDesc desc;
    importer.updateDescDimsFromGlow(glowReshape->getResult().getType()->dims(),
                                    desc);

    importer.setUsedTensors({nodeValueName(glowReshape->getInput())},
                            {nodeValueName(glowReshape->getResult())});

    return nnpiNetworkAddReshapeOp(
        importer.getNetwork(), glowReshape->getName().begin(),
        nodeValueName(glowReshape->getInput()).c_str(),
        nodeValueName(glowReshape->getResult()).c_str(), &desc);
  }
};

template <typename TypedNode>
class ConvertNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowTypedNode = llvm::dyn_cast<TypedNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowTypedNode, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowTypedNode->getInput())},
                            {nodeValueName(glowTypedNode->getResult())});

    return nnpiNetworkAddConvertOp(
        importer.getNetwork(), glowTypedNode->getName().begin(),
        nodeValueName(glowTypedNode->getInput()).c_str(),
        nodeValueName(glowTypedNode->getResult()).c_str());
  }
};

template <class MatMulNodeType>
class MatMulNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowMatMul = llvm::dyn_cast<MatMulNodeType>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowMatMul, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowMatMul->getLHS()),
                             nodeValueName(glowMatMul->getRHS())},
                            {nodeValueName(glowMatMul->getResult())});

    return nnpiNetworkAddMatMulOp(
        importer.getNetwork(), glowMatMul->getName().begin(),
        nodeValueName(glowMatMul->getLHS()).c_str(),
        nodeValueName(glowMatMul->getRHS()).c_str(),
        nodeValueName(glowMatMul->getResult()).c_str());
  }
};

class SliceNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSlice = llvm::dyn_cast<SliceNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSlice, "Bad node type",
                          NNPI_INVALID_PARAM);

    const auto &sliceOffset = glowSlice->getStart();
    int32_t startOffset[NNPI_MAX_DIMS] = {0};
    int32_t endOffset[NNPI_MAX_DIMS] = {0};
    auto *srcType = glowSlice->getInput().getType();
    auto *dstType = glowSlice->getResult().getType();
    LOG_AND_RETURN_IF_NOT(ERROR,
                          srcType->dims().size() == dstType->dims().size(),
                          "Bad dimansion", NNPI_INVALID_DIMS);
    LOG_AND_RETURN_IF_NOT(ERROR, srcType->dims().size() == sliceOffset.size(),
                          "Bad dimansion", NNPI_INVALID_DIMS);

    for (size_t i = 0, e = sliceOffset.size(); i < e; i++) {
      startOffset[i] = sliceOffset[i];
      endOffset[i] = startOffset[i] + dstType->dims()[i];
    }

    importer.setUsedTensors({nodeValueName(glowSlice->getInput())},
                            {nodeValueName(glowSlice->getResult())});

    return nnpiNetworkAddSliceOp(
        importer.getNetwork(), glowSlice->getName().begin(),
        nodeValueName(glowSlice->getInput()).c_str(),
        nodeValueName(glowSlice->getResult()).c_str(), startOffset, endOffset,
        nullptr, uint32_t(sliceOffset.size()));
  }
};

class SigmoidNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSigmoid = llvm::dyn_cast<SigmoidNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSigmoid, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowSigmoid->getInput())},
                            {nodeValueName(glowSigmoid->getResult())});

    return nnpiNetworkAddSigmoidOp(
        importer.getNetwork(), glowSigmoid->getName().begin(),
        nodeValueName(glowSigmoid->getInput()).c_str(),
        nodeValueName(glowSigmoid->getResult()).c_str());
  }
};

class SwishNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSwish = llvm::dyn_cast<SwishNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSwish, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowSwish->getInput())},
                            {nodeValueName(glowSwish->getResult())});

    return nnpiNetworkAddSwishOp(importer.getNetwork(),
                                 glowSwish->getName().begin(),
                                 nodeValueName(glowSwish->getInput()).c_str(),
                                 nodeValueName(glowSwish->getResult()).c_str());
  }
};

class TanhNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowTanh = llvm::dyn_cast<TanhNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowTanh, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowTanh->getInput())},
                            {nodeValueName(glowTanh->getResult())});

    return nnpiNetworkAddTanhOp(importer.getNetwork(),
                                glowTanh->getName().begin(),
                                nodeValueName(glowTanh->getInput()).c_str(),
                                nodeValueName(glowTanh->getResult()).c_str());
  }
};

class TopkNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowTopk = llvm::dyn_cast<TopKNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowTopk, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowTopk->getInput())},
                            {nodeValueName(glowTopk->getValues()),
                             nodeValueName(glowTopk->getIndices())});
    return nnpiNetworkAddTopkOp(
        importer.getNetwork(), glowTopk->getName().begin(),
        nodeValueName(glowTopk->getInput()).c_str(),
        nodeValueName(glowTopk->getValues()).c_str(),
        nodeValueName(glowTopk->getIndices()).c_str(), glowTopk->getK(),
        -1); // No Axis parameter in Glow - using -1 by default.
  }
};

class ConcatNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowConcat = llvm::dyn_cast<ConcatNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowConcat, "Bad node type",
                          NNPI_INVALID_PARAM);

    auto numInputs = glowConcat->getNumInputs();
    NNPIObjectName *inputs = new NNPIObjectName[numInputs];
    LOG_AND_RETURN_IF_NOT(ERROR, inputs, "No inputs", NNPI_INVALID_PARAM);
    std::unordered_set<std::string> inputTensors;

    for (unsigned i = 0; i < numInputs; i++) {
      auto nvName = nodeValueName(glowConcat->getNthInput(i));
      strncpy(inputs[i], nvName.c_str(), sizeof(NNPIObjectName));
      inputTensors.insert(nvName);
    }

    importer.setUsedTensors(inputTensors,
                            {nodeValueName(glowConcat->getResult())});

    auto res = nnpiNetworkAddConcatOp(
        importer.getNetwork(), glowConcat->getName().begin(), inputs, numInputs,
        nodeValueName(glowConcat->getResult()).c_str(), glowConcat->getDim());
    delete[] inputs;
    return res;
  }
};

class TileNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowTile = llvm::dyn_cast<TileNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowTile, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowTile->getInput())},
                            {nodeValueName(glowTile->getResult())});

    auto numDims = glowTile->getInput().getType()->dims().size();
    std::vector<int32_t> repeats(numDims, 1);
    auto axis = glowTile->getAxis();
    LOG_AND_RETURN_IF_NOT(ERROR, axis >= 0 && axis < numDims,
                          "tile axis is invalid", NNPI_INVALID_PARAM);
    repeats[axis] = glowTile->getCount();
    NNPITensorDesc desc;
    desc.attributes.value = 0;
    desc.attributes.constant = 1;
    desc.numDims = 1;
    desc.dims[0] = numDims;
    desc.quantParams.precision = NNPI_PRECISION_INT32;
    desc.quantParams.type = NNPI_QUANTIZATION_NONE;
    desc.layout = NNPI_LAYOUT_ANY;

    auto repeatsTensorName = glowTile->getName().str() + "_repeats";

    importer.addTensor(repeatsTensorName, desc, repeats.data());

    return nnpiNetworkAddTileOp(
        importer.getNetwork(), glowTile->getName().begin(),
        nodeValueName(glowTile->getInput()).c_str(), repeatsTensorName.c_str(),
        nodeValueName(glowTile->getResult()).c_str());
  }
};

class GatherNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowGather = llvm::dyn_cast<GatherNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowGather, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowGather->getData()),
                             nodeValueName(glowGather->getIndices())},
                            {nodeValueName(glowGather->getResult())});

    return nnpiNetworkAddGatherOp(
        importer.getNetwork(), glowGather->getName().begin(),
        nodeValueName(glowGather->getData()).c_str(),
        nodeValueName(glowGather->getIndices()).c_str(),
        nodeValueName(glowGather->getResult()).c_str(),
        glowGather->getBatchDims());
  }
};

class ArgMaxNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowArgMax = llvm::dyn_cast<ArgMaxNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowArgMax, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowArgMax->getInput())},
                            {nodeValueName(glowArgMax->getResult())});

    uint32_t axis = glowArgMax->getAxis();
    auto keepDims = glowArgMax->getKeepDims() ? 1 : 0;
    return nnpiNetworkAddReduceOp(
        importer.getNetwork(), glowArgMax->getName().begin(),
        nodeValueName(glowArgMax->getInput()).c_str(),
        nodeValueName(glowArgMax->getResult()).c_str(), NNPI_REDUCE_ARG_MAX,
        &axis, 1, keepDims);
  }
};

class ReduceAddNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowReduce = llvm::dyn_cast<BatchedReduceAddNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowReduce, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowReduce->getBatch())},
                            {nodeValueName(glowReduce->getResult())});

    uint32_t axis = glowReduce->getAxis();
    return nnpiNetworkAddReduceOp(
        importer.getNetwork(), glowReduce->getName().begin(),
        nodeValueName(glowReduce->getBatch()).c_str(),
        nodeValueName(glowReduce->getResult()).c_str(), NNPI_REDUCE_SUM, &axis,
        1, 0);
  }
};

template <class ReduceNodeType, NNPI_REDUCE_TYPE reduceType>
class ReduceMultAxesNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowReduce = llvm::dyn_cast<ReduceNodeType>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowReduce, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowReduce->getBatch())},
                            {nodeValueName(glowReduce->getResult())});

    LOG_AND_RETURN_IF_NOT(ERROR, glowReduce->getAxes().size() == 1,
                          "Bad axis value", NNPI_INVALID_PARAM);
    uint32_t axis = glowReduce->getAxes()[0];
    return nnpiNetworkAddReduceOp(
        importer.getNetwork(), glowReduce->getName().begin(),
        nodeValueName(glowReduce->getBatch()).c_str(),
        nodeValueName(glowReduce->getResult()).c_str(), reduceType, &axis, 1,
        0);
  }
};

class LogNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowLog = llvm::dyn_cast<LogNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowLog, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowLog->getInput())},
                            {nodeValueName(glowLog->getResult())});

    return nnpiNetworkAddLogOp(importer.getNetwork(),
                               glowLog->getName().begin(),
                               nodeValueName(glowLog->getInput()).c_str(),
                               nodeValueName(glowLog->getResult()).c_str());
  }
};

class SplatNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSplat = llvm::dyn_cast<SplatNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSplat, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({}, {nodeValueName(glowSplat->getResult())});
    auto *destType = glowSplat->getResult().getType();
    int32_t numDims = static_cast<int32_t>(destType->dims().size());
    float glowSplatValue = glowSplat->getValue();

    std::vector<dim_t> finalShapeFilledWithOnes(numDims, 1);

    auto tileInputTensorName = NNPIImporter::internalName_ +
                               glowSplat->getName().str() + "_Tile_input";

    if (destType->getElementType() != ElemKind::FloatTy) {
      NNPITensorDesc convertInputDesc;
      convertInputDesc.attributes.value = 0;
      convertInputDesc.attributes.constant = 1;
      convertInputDesc.quantParams.precision = NNPI_PRECISION_FLOAT32;
      convertInputDesc.quantParams.type = NNPI_QUANTIZATION_NONE;
      importer.updateDescDimsFromGlow(finalShapeFilledWithOnes,
                                      convertInputDesc);

      auto convertInputTensorName = NNPIImporter::internalName_ +
                                    glowSplat->getName().str() +
                                    "_Tile_Convert_input";
      LOG_NNPI_IF_ERROR_RETURN_VALUE(importer.addTensor(convertInputTensorName,
                                                        convertInputDesc,
                                                        &glowSplatValue),
                                     "Failed to add tensor");

      auto convertName = NNPIImporter::internalName_ +
                         glowSplat->getName().str() + "_Tile_Convert";
      Type convertOutputType =
          Type::newShape(*destType, finalShapeFilledWithOnes);
      LOG_NNPI_IF_ERROR_RETURN_VALUE(
          importer.addValue(tileInputTensorName, &convertOutputType),
          "Failed to add value");

      LOG_NNPI_IF_ERROR_RETURN_VALUE(
          nnpiNetworkAddConvertOp(importer.getNetwork(), convertName.c_str(),
                                  convertInputTensorName.c_str(),
                                  tileInputTensorName.c_str()),
          "Failed to add layer");
    } else {
      NNPITensorDesc tileInputDesc;
      tileInputDesc.attributes.value = 0;
      tileInputDesc.attributes.constant = 1;
      tileInputDesc.quantParams.precision = NNPI_PRECISION_FLOAT32;
      tileInputDesc.quantParams.type = NNPI_QUANTIZATION_NONE;
      importer.updateDescDimsFromGlow(finalShapeFilledWithOnes, tileInputDesc);

      LOG_NNPI_IF_ERROR_RETURN_VALUE(importer.addTensor(tileInputTensorName,
                                                        tileInputDesc,
                                                        &glowSplatValue),
                                     "Failed to add tensor");
    }

    NNPITensorDesc repeatsDesc;
    repeatsDesc.attributes.value = 0;
    repeatsDesc.attributes.constant = 1;
    repeatsDesc.quantParams.precision = NNPI_PRECISION_INT32;
    repeatsDesc.quantParams.type = NNPI_QUANTIZATION_NONE;
    importer.updateDescDimsFromGlow({destType->dims().size()}, repeatsDesc);
    auto repeatsTensorName = NNPIImporter::internalName_ +
                             glowSplat->getName().str() + "_Tile_repeats";
    std::vector<int32_t> dims;
    for (int i = 0; i < numDims; i++) {
      dims.push_back(destType->dims()[i]);
    }

    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addTensor(repeatsTensorName, repeatsDesc, dims.data()),
        "Failed to add tensor");

    auto tileNodeName =
        NNPIImporter::internalName_ + glowSplat->getName().str() + "_Tile";

    return nnpiNetworkAddTileOp(importer.getNetwork(), tileNodeName.c_str(),
                                tileInputTensorName.c_str(),
                                repeatsTensorName.c_str(),
                                nodeValueName(glowSplat->getResult()).c_str());
  }
};

class SLSNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSLS = llvm::dyn_cast<SparseLengthsSumNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSLS, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {
            nodeValueName(glowSLS->getData()),
            nodeValueName(glowSLS->getIndices()),
            nodeValueName(glowSLS->getLengths()),
        },
        {nodeValueName(glowSLS->getResult())});

    NNPI_LENGTH_TYPE lengthType;
    LOG_AND_RETURN_IF_NOT(ERROR,
                          NNPIImporter::convertLengthsModeToLengthType(
                              glowSLS->getLengthsMode(), lengthType) ==
                              NNPI_NO_ERROR,
                          "Unhandled SLS length type", NNPI_INVALID_PARAM);

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLS->getName().begin(),
        nodeValueName(glowSLS->getData()).c_str(),
        nodeValueName(glowSLS->getResult()).c_str(), NULL,
        nodeValueName(glowSLS->getIndices()).c_str(),
        nodeValueName(glowSLS->getLengths()).c_str(), false, false,
        glowSLS->getAvgLength(), lengthType);
  }
};

class SLWSNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSLWS = llvm::dyn_cast<SparseLengthsWeightedSumNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSLWS, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {
            nodeValueName(glowSLWS->getData()),
            nodeValueName(glowSLWS->getWeights()),
            nodeValueName(glowSLWS->getIndices()),
            nodeValueName(glowSLWS->getLengths()),
        },
        {nodeValueName(glowSLWS->getResult())});

    NNPI_LENGTH_TYPE lengthType;
    LOG_AND_RETURN_IF_NOT(ERROR,
                          NNPIImporter::convertLengthsModeToLengthType(
                              glowSLWS->getLengthsMode(), lengthType) ==
                              NNPI_NO_ERROR,
                          "Unhandled SLS length type", NNPI_INVALID_PARAM);

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLWS->getName().begin(),
        nodeValueName(glowSLWS->getData()).c_str(),
        nodeValueName(glowSLWS->getResult()).c_str(),
        nodeValueName(glowSLWS->getWeights()).c_str(),
        nodeValueName(glowSLWS->getIndices()).c_str(),
        nodeValueName(glowSLWS->getLengths()).c_str(), false, false,
        glowSLWS->getAvgLength(), lengthType);
  }
};

class EmbeddingBagNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowEmbeddingBag = llvm::dyn_cast<EmbeddingBagNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowEmbeddingBag, "Bad node type",
                          NNPI_INVALID_PARAM);

    bool hasEndOffset = glowEmbeddingBag->getHasEndOffset();
    LOG_AND_RETURN_IF_NOT(ERROR, hasEndOffset,
                          "[EmbeddingBag] hasEndOffset must be true",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {
            nodeValueName(glowEmbeddingBag->getData()),
            nodeValueName(glowEmbeddingBag->getWeights()),
            nodeValueName(glowEmbeddingBag->getIndices()),
            nodeValueName(glowEmbeddingBag->getOffsets()),
        },
        {nodeValueName(glowEmbeddingBag->getResult())});

    NNPI_LENGTH_TYPE lengthType;
    LOG_AND_RETURN_IF_NOT(ERROR,
                          NNPIImporter::convertLengthsModeToLengthType(
                              glowEmbeddingBag->getLengthsMode(), lengthType) ==
                              NNPI_NO_ERROR,
                          "Unhandled SLS length type", NNPI_INVALID_PARAM);

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowEmbeddingBag->getName().begin(),
        nodeValueName(glowEmbeddingBag->getData()).c_str(),
        nodeValueName(glowEmbeddingBag->getResult()).c_str(),
        nodeValueName(glowEmbeddingBag->getWeights()).c_str(),
        nodeValueName(glowEmbeddingBag->getIndices()).c_str(),
        nodeValueName(glowEmbeddingBag->getOffsets()).c_str(), false, true,
        glowEmbeddingBag->getAvgLength(), lengthType);
  }
};

class EmbeddingBagByteRowwiseOffsetsNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowEBBRO = llvm::dyn_cast<EmbeddingBagByteRowwiseOffsetsNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowEBBRO, "Bad node type",
                          NNPI_INVALID_PARAM);

    bool hasEndOffset = glowEBBRO->getHasEndOffset();
    LOG_AND_RETURN_IF_NOT(ERROR, hasEndOffset,
                          "[EmbeddingBag] hasEndOffset must be true",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {
            nodeValueName(glowEBBRO->getData()),
            nodeValueName(glowEBBRO->getWeights()),
            nodeValueName(glowEBBRO->getIndices()),
            nodeValueName(glowEBBRO->getOffsets()),
        },
        {nodeValueName(glowEBBRO->getResult())});

    bool usFp32Accum = !(glowEBBRO->getUseFP16Accumulation() &&
                         (glowEBBRO->getResult().getType()->getElementType() ==
                          glow::ElemKind::Float16Ty));

    NNPI_LENGTH_TYPE lengthType;
    LOG_AND_RETURN_IF_NOT(ERROR,
                          NNPIImporter::convertLengthsModeToLengthType(
                              glowEBBRO->getLengthsMode(), lengthType) ==
                              NNPI_NO_ERROR,
                          "Unhandled SLS length type", NNPI_INVALID_PARAM);

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowEBBRO->getName().begin(),
        nodeValueName(glowEBBRO->getData()).c_str(),
        nodeValueName(glowEBBRO->getResult()).c_str(),
        nodeValueName(glowEBBRO->getWeights()).c_str(),
        nodeValueName(glowEBBRO->getIndices()).c_str(),
        nodeValueName(glowEBBRO->getOffsets()).c_str(), usFp32Accum, true,
        glowEBBRO->getAvgLength(), lengthType);
  }
};

class SelectNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSelect = llvm::dyn_cast<SelectNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSelect, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {
            nodeValueName(glowSelect->getLHS()),
            nodeValueName(glowSelect->getRHS()),
            nodeValueName(glowSelect->getCond()),
        },
        {nodeValueName(glowSelect->getResult())});
    return nnpiNetworkAddSelectOp(
        importer.getNetwork(), glowSelect->getName().begin(),
        nodeValueName(glowSelect->getLHS()).c_str(), // True values.
        nodeValueName(glowSelect->getRHS()).c_str(), // False values.
        nodeValueName(glowSelect->getCond()).c_str(),
        nodeValueName(glowSelect->getResult()).c_str());
  }
};

class LRNNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowLRN = llvm::dyn_cast<LocalResponseNormalizationNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowLRN, "Bad node type", NNPI_INVALID_PARAM);

    // Overwrite input/output values for layout.
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowLRN->getInput()),
                          glowLRN->getInput().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowLRN->getResult()),
                          glowLRN->getResult().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");

    importer.setUsedTensors({nodeValueName(glowLRN->getInput())},
                            {nodeValueName(glowLRN->getResult())});

    return nnpiNetworkAddLRNOp(
        importer.getNetwork(), glowLRN->getName().begin(),
        nodeValueName(glowLRN->getInput()).c_str(),
        nodeValueName(glowLRN->getResult()).c_str(),
        NNPI_LRN_TYPE::NNPI_LRN_ACROSS_CHANNELS, glowLRN->getAlpha(),
        glowLRN->getBeta(), glowLRN->getK(),
        (2 * glowLRN->getHalfWindowSize() + 1));
  }
};

class RQFCNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowRowwiseFC = llvm::dyn_cast<RowwiseQuantizedFullyConnectedNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowRowwiseFC, "Bad node type",
                          NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(
        ERROR,
        !(glowRowwiseFC->getOffsets()) ||
            importer.zeroes(nodeValueName(glowRowwiseFC->getOffsets()).c_str()),
        "Bad offset value", NNPI_INVALID_PARAM);

    // Create the weights with no offset tensor.
    // Assert weights & biases have no offset or all zeroes.

    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowRowwiseFC->getWeights()),
                           /* alternativeLayout */ false,
                           nodeValueName(glowRowwiseFC->getScales()),
                           nodeValueName(glowRowwiseFC->getOffsets()),
                           /* forceSymlowp */ true),
        "Failed to add tensor to NNPI");

    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowRowwiseFC->getBias()),
                           /* alternativeLayout */ false, {}, {},
                           /* forceSymlowp */ true),
        "Failed to add tensor to NNPI");

    // Overwrite input/output values for layout.
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowRowwiseFC->getInput()),
                          glowRowwiseFC->getInput().getType(),
                          glowRowwiseFC->getInput().getType()->dims().size() ==
                              4),
        "Failed to add tensor to NNPI");
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowRowwiseFC->getResult()),
                          glowRowwiseFC->getResult().getType(),
                          glowRowwiseFC->getResult().getType()->dims().size() ==
                              4),
        "Failed to add tensor to NNPI");

    importer.setUsedTensors(
        {
            nodeValueName(glowRowwiseFC->getInput()),
            nodeValueName(glowRowwiseFC->getWeights()),
            nodeValueName(glowRowwiseFC->getBias()),
        },
        {
            nodeValueName(glowRowwiseFC->getResult()),
        });
    return nnpiNetworkAddFullyConnectedOp(
        importer.getNetwork(), glowRowwiseFC->getName().begin(),
        nodeValueName(glowRowwiseFC->getInput()).c_str(),
        nodeValueName(glowRowwiseFC->getResult()).c_str(),
        nodeValueName(glowRowwiseFC->getWeights()).c_str(),
        glowRowwiseFC->getBias()
            ? nodeValueName(glowRowwiseFC->getBias()).c_str()
            : nullptr);
  }
};

class ChannelwiseQuantizedConvolutionNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {

    auto *glowChannelwiseQuantizedConv =
        llvm::dyn_cast<ChannelwiseQuantizedConvolutionNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowChannelwiseQuantizedConv, "Bad node type",
                          NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(
        ERROR,
        !(glowChannelwiseQuantizedConv->getFilterOffsets()) ||
            importer.zeroes(
                nodeValueName(glowChannelwiseQuantizedConv->getFilterOffsets())
                    .c_str()),
        "Bad offset value", NNPI_INVALID_PARAM);

    const uint32_t SPATIAL_DIMS2 = 2;
    LOG_AND_RETURN_IF_NOT(
        ERROR,
        glowChannelwiseQuantizedConv->getKernels().size() == SPATIAL_DIMS2,
        "[Conv] Invalid number of kernel sizes", NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR,
                          glowChannelwiseQuantizedConv->getPads().size() ==
                              2 * SPATIAL_DIMS2,
                          "[Conv] Invalid number of pads", NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(
        ERROR,
        glowChannelwiseQuantizedConv->getStrides().size() == SPATIAL_DIMS2,
        "[Conv] Invalid number of strides", NNPI_INVALID_PARAM);

    uint32_t kernel[SPATIAL_DIMS2] = {
        glowChannelwiseQuantizedConv->getKernels()[0],
        glowChannelwiseQuantizedConv->getKernels()[1]};
    uint32_t paddingStart[SPATIAL_DIMS2] = {
        glowChannelwiseQuantizedConv->getPads()[0],
        glowChannelwiseQuantizedConv->getPads()[1]};
    uint32_t paddingEnd[SPATIAL_DIMS2] = {
        glowChannelwiseQuantizedConv->getPads()[2],
        glowChannelwiseQuantizedConv->getPads()[3]};
    uint32_t stride[SPATIAL_DIMS2] = {
        glowChannelwiseQuantizedConv->getStrides()[0],
        glowChannelwiseQuantizedConv->getStrides()[1]};
    uint32_t dilation[SPATIAL_DIMS2] = {1, 1}; // No dilation, default values

    // Create the weights with no offset tensor.
    // Assert weights & biases have no offset or all zeroes.

    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addTensor(
            nodeValueName(glowChannelwiseQuantizedConv->getFilter()),
            /* alternativeLayout */ true,
            nodeValueName(glowChannelwiseQuantizedConv->getFilterScales()),
            nodeValueName(glowChannelwiseQuantizedConv->getFilterOffsets()),
            /* forceSymlowp */ true),
        "Failed to add tensor to NNPI");

    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addTensor(
            nodeValueName(glowChannelwiseQuantizedConv->getBias()),
            /* alternativeLayout */ false, {}, {},
            /* forceSymlowp */ true),
        "Failed to add tensor to NNPI");

    // Overwrite input/output values for layout.
    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(
            nodeValueName(glowChannelwiseQuantizedConv->getInput()),
            glowChannelwiseQuantizedConv->getInput().getType(),
            glowChannelwiseQuantizedConv->getInput().getType()->dims().size() ==
                4),
        "Failed to add tensor to NNPI");

    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addValue(
            nodeValueName(glowChannelwiseQuantizedConv->getResult()),
            glowChannelwiseQuantizedConv->getResult().getType(),
            glowChannelwiseQuantizedConv->getResult()
                    .getType()
                    ->dims()
                    .size() == 4),
        "Failed to add tensor to NNPI");

    importer.setUsedTensors(
        {
            nodeValueName(glowChannelwiseQuantizedConv->getInput()),
            nodeValueName(glowChannelwiseQuantizedConv->getFilter()),
            nodeValueName(glowChannelwiseQuantizedConv->getBias()),
        },
        {
            nodeValueName(glowChannelwiseQuantizedConv->getResult()),
        });

    return nnpiNetworkAddConvolutionOp(
        importer.getNetwork(), glowChannelwiseQuantizedConv->getName().begin(),
        nodeValueName(glowChannelwiseQuantizedConv->getInput()).c_str(),
        nodeValueName(glowChannelwiseQuantizedConv->getResult()).c_str(),
        nodeValueName(glowChannelwiseQuantizedConv->getFilter()).c_str(),
        glowChannelwiseQuantizedConv->getBias()
            ? nodeValueName(glowChannelwiseQuantizedConv->getBias()).c_str()
            : nullptr,
        kernel, paddingStart, paddingEnd, stride, dilation, SPATIAL_DIMS2,
        glowChannelwiseQuantizedConv->getGroup());
  }
};

class ReplaceNaNNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowReplaceNan = llvm::dyn_cast<ReplaceNaNNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowReplaceNan, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowReplaceNan->getInput())},
                            {nodeValueName(glowReplaceNan->getResult())});
    return nnpiNetworkAddReplaceNanOp(
        importer.getNetwork(), glowReplaceNan->getName().begin(),
        nodeValueName(glowReplaceNan->getInput()).c_str(),
        nodeValueName(glowReplaceNan->getResult()).c_str(),
        glowReplaceNan->getValue());
  }
};

class GatherRangesNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowGatherRanges = llvm::dyn_cast<GatherRangesNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowGatherRanges, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowGatherRanges->getData()),
                             nodeValueName(glowGatherRanges->getRanges())},
                            {nodeValueName(glowGatherRanges->getOutput()),
                             nodeValueName(glowGatherRanges->getLengths())});
    return nnpiNetworkAddGatherRangesOp(
        importer.getNetwork(), glowGatherRanges->getName().begin(),
        nodeValueName(glowGatherRanges->getData()).c_str(),
        nodeValueName(glowGatherRanges->getRanges()).c_str(),
        nodeValueName(glowGatherRanges->getOutput()).c_str(),
        nodeValueName(glowGatherRanges->getLengths()).c_str());
  }
};

class BatchAddNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowBatchAdd = llvm::dyn_cast<BatchedAddNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowBatchAdd, "Bad node type",
                          NNPI_INVALID_PARAM);

    NNPIObjectName inputNames[2];
    snprintf(inputNames[0], NNPI_MAX_STRING_LEN, "%s",
             nodeValueName(glowBatchAdd->getBatch()).c_str());
    snprintf(inputNames[1], NNPI_MAX_STRING_LEN, "%s",
             nodeValueName(glowBatchAdd->getSlice()).c_str());
    importer.setUsedTensors({nodeValueName(glowBatchAdd->getBatch()),
                             nodeValueName(glowBatchAdd->getSlice())},
                            {nodeValueName(glowBatchAdd->getResult())});
    return nnpiNetworkAddElementwiseOp(
        importer.getNetwork(), glowBatchAdd->getName().begin(), inputNames, 2,
        nodeValueName(glowBatchAdd->getResult()).c_str(), NNPI_ELTWISE_ADD);
  }
};

class BatchMulNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowBatchMul = llvm::dyn_cast<BatchedMulNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowBatchMul, "Bad node type",
                          NNPI_INVALID_PARAM);

    NNPIObjectName inputNames[2];
    snprintf(inputNames[0], NNPI_MAX_STRING_LEN, "%s",
             nodeValueName(glowBatchMul->getBatch()).c_str());
    snprintf(inputNames[1], NNPI_MAX_STRING_LEN, "%s",
             nodeValueName(glowBatchMul->getSlice()).c_str());
    importer.setUsedTensors({nodeValueName(glowBatchMul->getBatch()),
                             nodeValueName(glowBatchMul->getSlice())},
                            {nodeValueName(glowBatchMul->getResult())});
    return nnpiNetworkAddElementwiseOp(
        importer.getNetwork(), glowBatchMul->getName().begin(), inputNames, 2,
        nodeValueName(glowBatchMul->getResult()).c_str(), NNPI_ELTWISE_MUL);
  }
};

class RQSLWSNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSLWS =
        llvm::dyn_cast<RowwiseQuantizedSparseLengthsWeightedSumNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSLWS, "Bad node type", NNPI_INVALID_PARAM);

    LOG_NNPI_IF_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowSLWS->getData()),
                           /* alternativeLayout */ false,
                           nodeValueName(glowSLWS->getScales()),
                           nodeValueName(glowSLWS->getOffsets()),
                           /* force to IA */ false),
        "Failed to add tensor to NNPI");

    importer.setUsedTensors(
        {
            nodeValueName(glowSLWS->getData()),
            nodeValueName(glowSLWS->getWeights()),
            nodeValueName(glowSLWS->getIndices()),
            nodeValueName(glowSLWS->getLengths()),
        },
        {nodeValueName(glowSLWS->getResult())});

    bool usFp32Accum = !(glowSLWS->getUseFP16Accumulation() &&
                         (glowSLWS->getResult().getType()->getElementType() ==
                          glow::ElemKind::Float16Ty));

    NNPI_LENGTH_TYPE lengthType;
    LOG_AND_RETURN_IF_NOT(ERROR,
                          NNPIImporter::convertLengthsModeToLengthType(
                              glowSLWS->getLengthsMode(), lengthType) ==
                              NNPI_NO_ERROR,
                          "Unhandled SLS length type", NNPI_INVALID_PARAM);

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLWS->getName().begin(),
        nodeValueName(glowSLWS->getData()).c_str(),
        nodeValueName(glowSLWS->getResult()).c_str(),
        nodeValueName(glowSLWS->getWeights()).c_str(),
        nodeValueName(glowSLWS->getIndices()).c_str(),
        nodeValueName(glowSLWS->getLengths()).c_str(), usFp32Accum, false,
        glowSLWS->getAvgLength(), lengthType);
  }
};

class FRQSLSNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSLWS =
        llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsSumNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSLWS, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {
            nodeValueName(glowSLWS->getData()),
            nodeValueName(glowSLWS->getIndices()),
            nodeValueName(glowSLWS->getLengths()),
        },
        {nodeValueName(glowSLWS->getResult())});

    bool usFp32Accum = !(glowSLWS->getUseFP16Accumulation() &&
                         (glowSLWS->getResult().getType()->getElementType() ==
                          glow::ElemKind::Float16Ty));

    NNPI_LENGTH_TYPE lengthType;
    LOG_AND_RETURN_IF_NOT(ERROR,
                          NNPIImporter::convertLengthsModeToLengthType(
                              glowSLWS->getLengthsMode(), lengthType) ==
                              NNPI_NO_ERROR,
                          "Unhandled SLS length type", NNPI_INVALID_PARAM);

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLWS->getName().begin(),
        nodeValueName(glowSLWS->getData()).c_str(),
        nodeValueName(glowSLWS->getResult()).c_str(), NULL,
        nodeValueName(glowSLWS->getIndices()).c_str(),
        nodeValueName(glowSLWS->getLengths()).c_str(), usFp32Accum, false,
        glowSLWS->getAvgLength(), lengthType);
  }
};

class FRQSLWSNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSLWS =
        llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSLWS, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {
            nodeValueName(glowSLWS->getData()),
            nodeValueName(glowSLWS->getWeights()),
            nodeValueName(glowSLWS->getIndices()),
            nodeValueName(glowSLWS->getLengths()),
        },
        {nodeValueName(glowSLWS->getResult())});

    bool usFp32Accum = !(glowSLWS->getUseFP16Accumulation() &&
                         (glowSLWS->getResult().getType()->getElementType() ==
                          glow::ElemKind::Float16Ty));

    NNPI_LENGTH_TYPE lengthType;
    LOG_AND_RETURN_IF_NOT(ERROR,
                          NNPIImporter::convertLengthsModeToLengthType(
                              glowSLWS->getLengthsMode(), lengthType) ==
                              NNPI_NO_ERROR,
                          "Unhandled SLS length type", NNPI_INVALID_PARAM);

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLWS->getName().begin(),
        nodeValueName(glowSLWS->getData()).c_str(),
        nodeValueName(glowSLWS->getResult()).c_str(),
        nodeValueName(glowSLWS->getWeights()).c_str(),
        nodeValueName(glowSLWS->getIndices()).c_str(),
        nodeValueName(glowSLWS->getLengths()).c_str(), usFp32Accum, false,
        glowSLWS->getAvgLength(), lengthType);
  }
};

class LengthsRangeFillNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowLengthsRangesfill = llvm::dyn_cast<LengthsRangeFillNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowLengthsRangesfill, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {nodeValueName(glowLengthsRangesfill->getLengths())},
        {nodeValueName(glowLengthsRangesfill->getResult())});

    return nnpiNetworkAddLengthsRangeFillOp(
        importer.getNetwork(), glowLengthsRangesfill->getName().begin(),
        nodeValueName(glowLengthsRangesfill->getLengths()).c_str(),
        nodeValueName(glowLengthsRangesfill->getResult()).c_str());
  }
};

class SpaceToDepthNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSpaceToDepth = llvm::dyn_cast<SpaceToDepthNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSpaceToDepth, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowSpaceToDepth->getInput())},
                            {nodeValueName(glowSpaceToDepth->getResult())});

    return nnpiNetworkAddSpaceToDepthOp(
        importer.getNetwork(), glowSpaceToDepth->getName().begin(),
        nodeValueName(glowSpaceToDepth->getInput()).c_str(),
        nodeValueName(glowSpaceToDepth->getResult()).c_str(),
        glowSpaceToDepth->getBlockSize());
  }
};

class BatchOneHotNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowBatchOneHot = llvm::dyn_cast<BatchOneHotNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowBatchOneHot, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {
            nodeValueName(glowBatchOneHot->getData()),
            nodeValueName(glowBatchOneHot->getLengths()),
            nodeValueName(glowBatchOneHot->getValues()),
        },
        {nodeValueName(glowBatchOneHot->getResult())});

    return nnpiNetworkAddBatchOneHotOp(
        importer.getNetwork(), glowBatchOneHot->getName().begin(),
        nodeValueName(glowBatchOneHot->getData()).c_str(),
        nodeValueName(glowBatchOneHot->getLengths()).c_str(),
        nodeValueName(glowBatchOneHot->getValues()).c_str(),
        nodeValueName(glowBatchOneHot->getResult()).c_str());
  }
};

class NNPICustomIANodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowIA = llvm::dyn_cast<NNPICustomIANode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowIA, "Bad node type", NNPI_INVALID_PARAM);

    auto numInputs = glowIA->getInputs().size();
    NNPIObjectName inputs[numInputs];
    LOG_AND_RETURN_IF_NOT(ERROR, inputs, "No inputs", NNPI_INVALID_PARAM);
    std::unordered_set<std::string> inputTensors;
    uint32_t i = 0;
    for (const auto &nv : glowIA->getInputs()) {
      auto nvName = nodeValueName(nv);
      strncpy(inputs[i++], nvName.c_str(), sizeof(NNPIObjectName));
      inputTensors.insert(nvName);
    }

    uint32_t numOutputs = 1;
    NNPIObjectName outputs[numOutputs];
    LOG_AND_RETURN_IF_NOT(ERROR, outputs, "No outputs", NNPI_INVALID_PARAM);
    std::unordered_set<std::string> outputTensors;
    auto nvName = nodeValueName(glowIA->getResult());
    strncpy(outputs[0], nvName.c_str(), sizeof(NNPIObjectName));
    outputTensors.insert(nvName);

    importer.setUsedTensors(inputTensors, outputTensors);
    NNPIErrorCode error = importer.addIAExtentionPath(glowIA->getIAPath());
    LOG_AND_RETURN_IF_NOT(ERROR, error == NNPI_NO_ERROR,
                          "Failed to store IA extension", NNPI_INVALID_PARAM);

    auto res = nnpiNetworkAddCustomIAOp(
        importer.getNetwork(), glowIA->getName().begin(), numInputs, inputs,
        numOutputs, outputs, glowIA->getKernelName().c_str());
    return res;
  }
};
class NNPICustomDSPNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowDSP = llvm::dyn_cast<NNPICustomDSPNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowDSP, "Bad node type", NNPI_INVALID_PARAM);

    auto numInputs = glowDSP->getInputs().size();
    NNPIObjectName *inputs = new NNPIObjectName[numInputs];
    LOG_AND_RETURN_IF_NOT(ERROR, inputs, "No inputs", NNPI_INVALID_PARAM);
    std::unordered_set<std::string> inputTensors;
    uint32_t i = 0;
    for (const auto &nv : glowDSP->getInputs()) {
      auto nvName = nodeValueName(nv);
      strncpy(inputs[i++], nvName.c_str(), sizeof(NNPIObjectName));
      inputTensors.insert(nvName);
    }

    uint32_t numOutputs = 1;
    NNPIObjectName *outputs = new NNPIObjectName[numOutputs];
    LOG_AND_RETURN_IF_NOT(ERROR, outputs, "No outputs", NNPI_INVALID_PARAM);
    std::unordered_set<std::string> outputTensors;
    auto nvName = nodeValueName(glowDSP->getResult());
    strncpy(outputs[0], nvName.c_str(), sizeof(NNPIObjectName));
    outputTensors.insert(nvName);

    importer.setUsedTensors(inputTensors, outputTensors);

    const auto *kpConstant =
        glowDSP->getParent()->getParent()->getConstantByName(
            glowDSP->getKernelParams().getNode()->getName());
    LOG_AND_RETURN_IF_NOT(ERROR, kpConstant, "Kernel Params must be constant",
                          NNPI_INVALID_PARAM);

    const auto *wcConstant =
        glowDSP->getParent()->getParent()->getConstantByName(
            glowDSP->getWalkConfig().getNode()->getName());
    LOG_AND_RETURN_IF_NOT(ERROR, wcConstant, "Walk Config must be constant",
                          NNPI_INVALID_PARAM);

    const Tensor *kpTensor = &kpConstant->getPayload();
    const Tensor *wcTensor = &wcConstant->getPayload();
    auto res = nnpiNetworkAddCustomDspOp(
        importer.getNetwork(), glowDSP->getName().begin(), inputs, numInputs,
        outputs, numOutputs, kpTensor->getUnsafePtr(),
        kpTensor->getSizeInBytes(),
        reinterpret_cast<const NNPIWalkConfig *>(wcTensor->getUnsafePtr()),
        glowDSP->getPrivateAreaSize(), glowDSP->getKernelName().c_str(),
        reinterpret_cast<const NNPICustomDspIceRefCallback>(
            glowDSP->getICERefCallback()));
    delete[] inputs;
    delete[] outputs;
    return res;
  }
};

class ClipNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowClip = llvm::dyn_cast<ClipNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowClip, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowClip->getInput())},
                            {nodeValueName(glowClip->getResult())});

    return nnpiNetworkAddClipOp(importer.getNetwork(),
                                glowClip->getName().begin(),
                                nodeValueName(glowClip->getInput()).c_str(),
                                nodeValueName(glowClip->getResult()).c_str(),
                                glowClip->getMin(), glowClip->getMax());
  }
};

class BatchNormalizationNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowBN = llvm::dyn_cast<BatchNormalizationNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowBN, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {nodeValueName(glowBN->getInput()), nodeValueName(glowBN->getScale())},
        {nodeValueName(glowBN->getBias()), nodeValueName(glowBN->getMean()),
         nodeValueName(glowBN->getVar()), nodeValueName(glowBN->getResult())});

    return nnpiNetworkAddBatchNormOp(
        importer.getNetwork(), glowBN->getName().begin(),
        nodeValueName(glowBN->getInput()).c_str(),
        nodeValueName(glowBN->getResult()).c_str(),
        nodeValueName(glowBN->getMean()).c_str(),
        nodeValueName(glowBN->getVar()).c_str(),
        nodeValueName(glowBN->getScale()).c_str(),
        nodeValueName(glowBN->getBias()).c_str(), glowBN->getEpsilon());
  }
};

class LayerNormalizationNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowLN = llvm::dyn_cast<LayerNormalizationNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowLN, "Bad node type", NNPI_INVALID_PARAM);

    importer.setUsedTensors(
        {nodeValueName(glowLN->getInput()), nodeValueName(glowLN->getScale())},
        {nodeValueName(glowLN->getBias()), nodeValueName(glowLN->getResult())});

    llvm::SmallVector<uint32_t, 4> normShape(glowLN->getScale().dims().begin(),
                                             glowLN->getScale().dims().end());

    return nnpiNetworkAddLayerNormOp(
        importer.getNetwork(), glowLN->getName().begin(),
        nodeValueName(glowLN->getInput()).c_str(),
        nodeValueName(glowLN->getResult()).c_str(),
        nodeValueName(glowLN->getScale()).c_str(),
        nodeValueName(glowLN->getBias()).c_str(), normShape.data(),
        normShape.size(), glowLN->getEpsilon());
  }
};

class LogitNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowLogit = llvm::dyn_cast<LogitNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowLogit, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowLogit->getInput()),
                             nodeValueName(glowLogit->getResult())});

    return nnpiNetworkAddLogitOp(
        importer.getNetwork(), glowLogit->getName().begin(),
        nodeValueName(glowLogit->getInput()).c_str(),
        nodeValueName(glowLogit->getResult()).c_str(), glowLogit->getEpsilon());
  }
};

class ModuloNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowModulo = llvm::dyn_cast<ModuloNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowModulo, "Bad node type",
                          NNPI_INVALID_PARAM);

    importer.setUsedTensors({nodeValueName(glowModulo->getInput())},
                            {nodeValueName(glowModulo->getResult())});
    auto divisor = glowModulo->getDivisor();
    LOG_AND_RETURN_IF(ERROR, ((divisor < INT_MIN) || (divisor > INT_MAX)),
                      "Divisor out of int32 range!", NNPI_INVALID_PARAM);

    return nnpiNetworkAddModuloOp(
        importer.getNetwork(), glowModulo->getName().begin(),
        nodeValueName(glowModulo->getInput()).c_str(),
        nodeValueName(glowModulo->getResult()).c_str(), (int32_t)divisor,
        glowModulo->getSignFollowDivisor() ? 1 : 0);
  }
};

//////////////////////////////////////////////////////////////////////////
namespace {
std::unordered_map<
    std::string,
    std::unique_ptr<INNPINodeImporter>>::value_type importerInit[] = {
    {"", nullptr},
    {"Convolution",
     glow::make_unique<ConvolutionNodeImporter<ConvolutionNode, 2>>()},
    {"Convolution3D",
     glow::make_unique<ConvolutionNodeImporter<Convolution3DNode, 3>>()},
    {"Transpose", glow::make_unique<TransposeNodeImporter>()},
    {"MaxPool",
     glow::make_unique<PoolNodeImporter<glow::MaxPoolNode, NNPI_POOL_MAX>>()},
    {"AvgPool",
     glow::make_unique<PoolNodeImporter<glow::AvgPoolNode, NNPI_POOL_AVG>>()},
    {"AdaptiveAvgPool",
     glow::make_unique<
         AdaptivePoolNodeImporter<glow::AdaptiveAvgPoolNode, NNPI_POOL_AVG>>()},
    {"FullyConnected", glow::make_unique<FullyConnectedNodeImporter>()},
    {"SoftMax", glow::make_unique<SoftMaxNodeImporter>()},
    {"Save", glow::make_unique<SaveNodeImporter>()},
    {"Relu", glow::make_unique<ReluNodeImporter>()},
    {"PRelu", glow::make_unique<PReluNodeImporter>()},
    {"Gelu", glow::make_unique<GeluNodeImporter>()},
    {"Exp", glow::make_unique<
                UnaryEltwiseNodeImporter<glow::ExpNode, NNPI_ELTWISE_EXP>>()},
    {"Max", glow::make_unique<
                BinaryEltwiseNodeImporter<glow::MaxNode, NNPI_ELTWISE_MAX>>()},
    {"Min", glow::make_unique<
                BinaryEltwiseNodeImporter<glow::MinNode, NNPI_ELTWISE_MIN>>()},
    {"Add", glow::make_unique<
                BinaryEltwiseNodeImporter<glow::AddNode, NNPI_ELTWISE_ADD>>()},
    {"Mul", glow::make_unique<
                BinaryEltwiseNodeImporter<glow::MulNode, NNPI_ELTWISE_MUL>>()},
    {"Div", glow::make_unique<
                BinaryEltwiseNodeImporter<glow::DivNode, NNPI_ELTWISE_DIV>>()},
    {"Sub", glow::make_unique<
                BinaryEltwiseNodeImporter<glow::SubNode, NNPI_ELTWISE_SUB>>()},
    {"Pow", glow::make_unique<
                BinaryEltwiseNodeImporter<glow::PowNode, NNPI_ELTWISE_POW>>()},
    {"CmpEQ",
     glow::make_unique<
         BinaryEltwiseNodeImporter<glow::CmpEQNode, NNPI_ELTWISE_EQ>>()},
    {"CmpLTE",
     glow::make_unique<
         BinaryEltwiseNodeImporter<glow::CmpLTENode, NNPI_ELTWISE_LTE>>()},
    {"CmpLT",
     glow::make_unique<
         BinaryEltwiseNodeImporter<glow::CmpLTNode, NNPI_ELTWISE_LESS>>()},
    {"ArgMax", glow::make_unique<ArgMaxNodeImporter>()},
    {"Reshape", glow::make_unique<ReshapeNodeImporter>()},
    {"Quantize", glow::make_unique<ConvertNodeImporter<QuantizeNode>>()},
    {"Dequantize", glow::make_unique<ConvertNodeImporter<DequantizeNode>>()},
    {"RescaleQuantized",
     glow::make_unique<ConvertNodeImporter<RescaleQuantizedNode>>()},
    {"ConvertTo", glow::make_unique<ConvertNodeImporter<ConvertToNode>>()},
    {"MatMul", glow::make_unique<MatMulNodeImporter<MatMulNode>>()},
    {"BatchMatMul", glow::make_unique<MatMulNodeImporter<BatchMatMulNode>>()},
    {"Slice", glow::make_unique<SliceNodeImporter>()},
    {"Sigmoid", glow::make_unique<SigmoidNodeImporter>()},
    {"Tanh", glow::make_unique<TanhNodeImporter>()},
    {"Concat", glow::make_unique<ConcatNodeImporter>()},
    {"Tile", glow::make_unique<TileNodeImporter>()},
    {"Gather", glow::make_unique<GatherNodeImporter>()},
    {"BatchedReduceAdd", glow::make_unique<ReduceAddNodeImporter>()},
    {"Log", glow::make_unique<LogNodeImporter>()},
    {"TopK", glow::make_unique<TopkNodeImporter>()},
    {"BatchedReduceMean",
     glow::make_unique<ReduceMultAxesNodeImporter<glow::BatchedReduceMeanNode,
                                                  NNPI_REDUCE_MEAN>>()},
    {"BatchedReduceMin",
     glow::make_unique<ReduceMultAxesNodeImporter<glow::BatchedReduceMinNode,
                                                  NNPI_REDUCE_MIN>>()},
    {"Splat", glow::make_unique<SplatNodeImporter>()},
    {"SparseLengthsWeightedSum", glow::make_unique<SLWSNodeImporter>()},
    {"SparseLengthsSum", glow::make_unique<SLSNodeImporter>()},
    {"FusedRowwiseQuantizedSparseLengthsSum",
     glow::make_unique<FRQSLSNodeImporter>()},
    {"Select", glow::make_unique<SelectNodeImporter>()},
    {"LocalResponseNormalization", glow::make_unique<LRNNodeImporter>()},
    {"RowwiseQuantizedFullyConnected", glow::make_unique<RQFCNodeImporter>()},
    {"ReplaceNaN", glow::make_unique<ReplaceNaNNodeImporter>()},
    {"GatherRanges", glow::make_unique<GatherRangesNodeImporter>()},
    {"BatchedAdd", glow::make_unique<BatchAddNodeImporter>()},
    {"BatchedMul", glow::make_unique<BatchMulNodeImporter>()},
    {"RowwiseQuantizedSparseLengthsWeightedSum",
     glow::make_unique<RQSLWSNodeImporter>()},
    {"FusedRowwiseQuantizedSparseLengthsWeightedSum",
     glow::make_unique<FRQSLWSNodeImporter>()},
    {"LengthsRangeFill", glow::make_unique<LengthsRangeFillNodeImporter>()},
    {"BatchOneHot", glow::make_unique<BatchOneHotNodeImporter>()},
    {"NNPICustomDSP", glow::make_unique<NNPICustomDSPNodeImporter>()},
    {"NNPICustomIA", glow::make_unique<NNPICustomIANodeImporter>()},
    {"SpaceToDepth", glow::make_unique<SpaceToDepthNodeImporter>()},
    {"Clip", glow::make_unique<ClipNodeImporter>()},
    {"BatchNormalization", glow::make_unique<BatchNormalizationNodeImporter>()},
    {"LayerNormalization", glow::make_unique<LayerNormalizationNodeImporter>()},
    {"ChannelwiseQuantizedConvolution",
     glow::make_unique<ChannelwiseQuantizedConvolutionNodeImporter>()},
    {"EmbeddingBag", glow::make_unique<EmbeddingBagNodeImporter>()},
    {"EmbeddingBagByteRowwiseOffsets",
     glow::make_unique<EmbeddingBagByteRowwiseOffsetsNodeImporter>()},
    {"Logit", glow::make_unique<LogitNodeImporter>()},
    {"Modulo", glow::make_unique<ModuloNodeImporter>()},
    {"Swish", glow::make_unique<SwishNodeImporter>()},
};
}

const std::unordered_map<std::string, std::unique_ptr<INNPINodeImporter>>
    NNPIImporter::nodeImporters_ = {
        std::make_move_iterator(std::begin(importerInit)),
        std::make_move_iterator(std::end(importerInit))};
