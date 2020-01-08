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
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Quantization/Base/Base.h"
#include "glow/Quantization/Quantization.h"
#include "nnpi_transformer.h"
#include <cmath>
#include <cstdio>
#include <limits>

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

glow::NNPIImporter::NNPIImporter(const NNPICompilationOptions &compileOptions)
    : internalNameCounter_(0), network_(NNPI_INVALID_NNPIHANDLE),
      compileOptions_(compileOptions) {
  ASSERT_LOG_NNPI_ERROR(nnpiNetworkCreate(&network_),
                        "Failed to create NNPI network");
  // Setting the network name for testing framework purposes.
  ASSERT_LOG_NNPI_ERROR(
      nnpiNetworkSetName(network_, compileOptions_.compiledFile.get().c_str()),
      "Failed to set NNPI network name");
}

/// Destructor.
glow::NNPIImporter::~NNPIImporter() {
  if (network_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_ERROR(nnpiNetworkDestroy(network_),
                   "Failed to destroy NNPI network");
  }
}

NNPIErrorCode glow::NNPIImporter::addTensor(std::string name,
                                            bool alternativeLayout,
                                            const std::string &scaleTensor,
                                            const std::string &offsetTensor,
                                            bool forceSymlowp) {
  LOG_AND_RETURN_IF_NOT(ERROR, constants_.count(name),
                        "Could not find Constants for tensor",
                        NNPI_INVALID_PARAM);
  const Tensor *t = constants_.at(name);

  NNPITensorDesc desc;
  desc.attributes.value = 0;
  desc.attributes.constant = 1;
  const auto &dims = t->dims();
  desc.numDims = dims.size();
  updateDescQuantFromGlow(t->getType(), desc, scaleTensor, offsetTensor,
                          forceSymlowp);
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
                          forceSymlowp);
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
    desc.layout = NNPI_LAYOUT_ANY;
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
      if (forceSymlowp || offsetTensor.empty() ||
          (compileOptions_.useSymlowp && zeroes(offsetTensor))) {
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
      if (forceSymlowp || compileOptions_.useSymlowp) {
        // WA use SYMLOWP for zero offset tensors.
        if (t.getOffset() == 0) {
          DBG("SYMLOWP WA");
          desc.quantParams.type = NNPI_QUANTIZATION_SYMLOWP;
          desc.quantParams.params.symlowp.scale = t.getScale();
        }
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
    if ((forceSymlowp || compileOptions_.useSymlowp) && t.getOffset() == 0) {
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

NNPINetwork glow::NNPIImporter::importFunction(Function *F,
                                               const BackendOptions &opts) {
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
    LOG_NNPI_ERROR_RETURN_INVALID_HANDLE(
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
      LOG_NNPI_ERROR_RETURN_INVALID_HANDLE(
          addValue(nodeValueName(resVal), resVal.getType()),
          "Failed to add intermediate");
      DBG("  Output: " << nodeValueName(resVal));
    }
    DBG_MEM_USAGE("ImportFunction import node: " << N.getKindName());
    // Import node.
    LOG_NNPI_ERROR_RETURN_INVALID_HANDLE(
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
      LOG_NNPI_ERROR_RETURN_INVALID_HANDLE(
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
    LOG_NNPI_ERROR(nnpiNetworkDestroy(network_),
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
class ConvolutionNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowConv = llvm::dyn_cast<ConvolutionNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowConv, "Bad node type", NNPI_INVALID_PARAM);

    const uint32_t SPATIAL_DIMS2 = 2;
    LOG_AND_RETURN_IF_NOT(ERROR, glowConv->getKernels().size() == SPATIAL_DIMS2,
                          "[Conv] Invalid number of kernel sizes",
                          NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR,
                          glowConv->getPads().size() == 2 * SPATIAL_DIMS2,
                          "[Conv] Invalid number of pads", NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, glowConv->getStrides().size() == SPATIAL_DIMS2,
                          "[Conv] Invalid number of strides",
                          NNPI_INVALID_PARAM);

    uint32_t kernel[SPATIAL_DIMS2] = {glowConv->getKernels()[0],
                                      glowConv->getKernels()[1]};
    uint32_t paddingStart[SPATIAL_DIMS2] = {glowConv->getPads()[0],
                                            glowConv->getPads()[1]};
    uint32_t paddingEnd[SPATIAL_DIMS2] = {glowConv->getPads()[2],
                                          glowConv->getPads()[3]};
    uint32_t stride[SPATIAL_DIMS2] = {glowConv->getStrides()[0],
                                      glowConv->getStrides()[1]};
    uint32_t dilation[SPATIAL_DIMS2] = {glowConv->getDilation(),
                                        glowConv->getDilation()};

    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowConv->getFilter()),
                           /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");
    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowConv->getBias())),
        "Failed to add tensor to NNPI");

    // Overwrite input/output values for layout.
    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowConv->getInput()),
                          glowConv->getInput().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");
    LOG_NNPI_ERROR_RETURN_VALUE(
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
        kernel, paddingStart, paddingEnd, stride, dilation, SPATIAL_DIMS2,
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
    LOG_AND_RETURN_IF_NOT(ERROR, glowPool, "Bad node type", NNPI_INVALID_PARAM);

    const uint32_t SPATIAL_DIMS2 = 2;
    LOG_AND_RETURN_IF_NOT(ERROR, glowPool->getKernels().size() == SPATIAL_DIMS2,
                          "[Pool] Invalid number of kernel sizes",
                          NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR,
                          glowPool->getPads().size() == 2 * SPATIAL_DIMS2,
                          "[Pool] Invalid number of pads", NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(ERROR, glowPool->getStrides().size() == SPATIAL_DIMS2,
                          "[Pool] Invalid number of strides",
                          NNPI_INVALID_PARAM);
    uint32_t kernel[SPATIAL_DIMS2] = {glowPool->getKernels()[0],
                                      glowPool->getKernels()[1]};
    uint32_t paddingStart[SPATIAL_DIMS2] = {glowPool->getPads()[0],
                                            glowPool->getPads()[1]};
    uint32_t paddingEnd[SPATIAL_DIMS2] = {glowPool->getPads()[2],
                                          glowPool->getPads()[3]};
    uint32_t stride[SPATIAL_DIMS2] = {glowPool->getStrides()[0],
                                      glowPool->getStrides()[1]};

    // Overwrite input/output values for layout.
    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowPool->getInput()),
                          glowPool->getInput().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");
    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowPool->getResult()),
                          glowPool->getResult().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");

    importer.setUsedTensors({nodeValueName(glowPool->getInput())},
                            {nodeValueName(glowPool->getResult())});

    return nnpiNetworkAddPoolingOp(
        importer.getNetwork(), glowPool->getName().begin(),
        nodeValueName(glowPool->getInput()).c_str(),
        nodeValueName(glowPool->getResult()).c_str(), NULL, kernel,
        paddingStart, paddingEnd, stride, SPATIAL_DIMS2, poolType, 0, 0);
  }
};

class FullyConnectedNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowFC = llvm::dyn_cast<FullyConnectedNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowFC, "Bad node type", NNPI_INVALID_PARAM);

    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowFC->getWeights()),
                           /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");

    // Overwrite input/output values for layout.
    const auto *input = glowFC->getInput().getNode();
    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addValue(input->getName(), input->getType(0),
                          input->getType(0)->dims().size() == 4),
        "Failed to add tensor to NNPI");
    const auto *result = glowFC->getResult().getNode();
    LOG_NNPI_ERROR_RETURN_VALUE(
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
                            {nodeValueName(glowArgMax->getArgmax())});

    uint32_t axis = glowArgMax->getAxis();
    auto keepDims = glowArgMax->getKeepDims() ? 1 : 0;
    return nnpiNetworkAddReduceOp(
        importer.getNetwork(), glowArgMax->getName().begin(),
        nodeValueName(glowArgMax->getInput()).c_str(),
        nodeValueName(glowArgMax->getArgmax()).c_str(), NNPI_REDUCE_ARG_MAX,
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
        1);
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
        nodeValueName(glowReduce->getResult()).c_str(), reduceType, &axis, 1);
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
    // Create a constant tensor instead of a Splat (Tile) node.
    NNPITensorDesc desc;
    desc.attributes.value = 0;
    desc.attributes.constant = 1;

    importer.updateDescQuantFromGlow(*destType, desc);
    importer.updateDescDimsFromGlow(destType->dims(), desc);

    uint8_t *pData = new uint8_t[destType->getSizeInBytes()];
    uint8_t elem[8]; // Assuming no element larger than 8 bytes.
    LOG_AND_RETURN_IF_NOT(ERROR, destType->getElementSize() <= 8,
                          "Bad dimansion", NNPI_INVALID_DIMS);

    switch (destType->getElementType()) {
    case glow::ElemKind::FloatTy: {
      float val = glowSplat->getValue();
      std::memcpy(elem, &val, sizeof(float));
    } break;
    case glow::ElemKind::Float16Ty: {
      float16_t val = glowSplat->getValue();
      std::memcpy(elem, &val, sizeof(float16_t));
    } break;
    case glow::ElemKind::Int8QTy: {
      float qfVal = round((glowSplat->getValue() / destType->getScale()) +
                          destType->getOffset());
      int8_t qVal = qfVal < static_cast<float>(INT8_MIN)
                        ? INT8_MIN
                        : qfVal > static_cast<float>(INT8_MAX)
                              ? INT8_MAX
                              : static_cast<int8_t>(qfVal);
      std::memcpy(elem, &qVal, sizeof(int8_t));
    } break;
    case glow::ElemKind::BoolTy:
      elem[0] = (glowSplat->getValue() != 0);
      break;
    case glow::ElemKind::Int32ITy: {
      int32_t val = static_cast<int32_t>(glowSplat->getValue());
      std::memcpy(elem, &val, sizeof(int32_t));
    } break;
    case glow::ElemKind::Int64ITy: {
      int64_t val = static_cast<int64_t>(glowSplat->getValue());
      std::memcpy(elem, &val, sizeof(int64_t));
    } break;
    default:
      LOG_AND_RETURN_IF_NOT(ERROR, 0, "Unhandled ElemKind for Splat output.",
                            NNPI_INVALID_PARAM);
      return NNPI_NOT_IMPLEMENTED;
    }

    auto destSize(destType->size());
    auto elemSize(destType->getElementSize());
    for (size_t i = 0; i < destSize; i++) {
      for (size_t j = 0; j < elemSize; j++) {
        pData[i * elemSize + j] = elem[j];
      }
    }

    auto res =
        importer.addTensor(nodeValueName(glowSplat->getResult()), desc, pData);
    delete[] pData;
    return res;
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

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLS->getName().begin(),
        nodeValueName(glowSLS->getData()).c_str(),
        nodeValueName(glowSLS->getResult()).c_str(), NULL,
        nodeValueName(glowSLS->getIndices()).c_str(),
        nodeValueName(glowSLS->getLengths()).c_str(), false);
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

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLWS->getName().begin(),
        nodeValueName(glowSLWS->getData()).c_str(),
        nodeValueName(glowSLWS->getResult()).c_str(),
        nodeValueName(glowSLWS->getWeights()).c_str(),
        nodeValueName(glowSLWS->getIndices()).c_str(),
        nodeValueName(glowSLWS->getLengths()).c_str(), false);
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
    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowLRN->getInput()),
                          glowLRN->getInput().getType(),
                          /* alternativeLayout */ true),
        "Failed to add tensor to NNPI");
    LOG_NNPI_ERROR_RETURN_VALUE(
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
        ERROR, glowRowwiseFC->getInput().getType()->getOffset() == 0.f,
        (std::string("Bad input offset value") +
         std::to_string(glowRowwiseFC->getInput().getType()->getOffset())),
        NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(
        ERROR, glowRowwiseFC->getResult().getType()->getOffset() == 0.f,
        (std::string("Bad result offset value") +
         std::to_string(glowRowwiseFC->getResult().getType()->getOffset())),
        NNPI_INVALID_PARAM);
    LOG_AND_RETURN_IF_NOT(
        ERROR,
        !(glowRowwiseFC->getOffsets()) ||
            importer.zeroes(nodeValueName(glowRowwiseFC->getOffsets()).c_str()),
        "Bad offset value", NNPI_INVALID_PARAM);

    // Add internal tensor for Symlowp input.
    std::string symlowpInputName =
        NNPIImporter::internalName_ +
        nodeValueName(glowRowwiseFC->getInput()).c_str() + "_symlowp";
    auto *inType = glowRowwiseFC->getInput().getType();
    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addValue(symlowpInputName, inType,
                          /* alternativeLayout */ inType->dims().size() == 4,
                          /* input */ false, /* output */ false, {}, {},
                          /* forceSymlowp */ true),
        "Failed to add value");

    // Add internal tensor for Symlowp output.
    std::string symlowpOutputName =
        NNPIImporter::internalName_ +
        nodeValueName(glowRowwiseFC->getResult()).c_str() + "_symlowp";
    auto *outType = glowRowwiseFC->getResult().getType();
    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addValue(symlowpOutputName, outType,
                          /* alternativeLayout */ outType->dims().size() == 4,
                          /* input */ false, /* output */ false, {}, {},
                          /* forceSymlowp */ true),
        "Failed to add value");

    // Add convert op from Gemmlowp input to Symlowp.
    std::string convertInputName = NNPIImporter::internalName_ +
                                   glowRowwiseFC->getName().begin() +
                                   "_convert_input";
    LOG_NNPI_ERROR_RETURN_VALUE(
        nnpiNetworkAddConvertOp(
            importer.getNetwork(), convertInputName.c_str(),
            nodeValueName(glowRowwiseFC->getInput()).c_str(),
            symlowpInputName.c_str()),
        "Failed to add layer");

    // Add convert op from Symlowp output to Gemmlowp.
    std::string convertOutputName = NNPIImporter::internalName_ +
                                    glowRowwiseFC->getName().begin() +
                                    "_convert_output";
    LOG_NNPI_ERROR_RETURN_VALUE(
        nnpiNetworkAddConvertOp(
            importer.getNetwork(), convertOutputName.c_str(),
            symlowpOutputName.c_str(),
            nodeValueName(glowRowwiseFC->getResult()).c_str()),
        "Failed to add layer");

    // Create the weights with no offset tensor.
    // Assert weights & biases have no offset or all zeroes.

    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowRowwiseFC->getWeights()),
                           /* alternativeLayout */ false,
                           nodeValueName(glowRowwiseFC->getScales()),
                           nodeValueName(glowRowwiseFC->getOffsets()),
                           /* forceSymlowp */ true),
        "Failed to add tensor to NNPI");

    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowRowwiseFC->getBias()),
                           /* alternativeLayout */ false, {}, {},
                           /* forceSymlowp */ true),
        "Failed to add tensor to NNPI");

    // Overwrite input/output values for layout.
    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addValue(nodeValueName(glowRowwiseFC->getInput()),
                          glowRowwiseFC->getInput().getType(),
                          glowRowwiseFC->getInput().getType()->dims().size() ==
                              4),
        "Failed to add tensor to NNPI");
    LOG_NNPI_ERROR_RETURN_VALUE(
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
            symlowpInputName,
            symlowpOutputName,
        },
        {
            nodeValueName(glowRowwiseFC->getResult()),
            symlowpInputName,
            symlowpOutputName,
        });
    return nnpiNetworkAddFullyConnectedOp(
        importer.getNetwork(), glowRowwiseFC->getName().begin(),
        symlowpInputName.c_str(), symlowpOutputName.c_str(),
        nodeValueName(glowRowwiseFC->getWeights()).c_str(),
        glowRowwiseFC->getBias()
            ? nodeValueName(glowRowwiseFC->getBias()).c_str()
            : nullptr);
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

class RQSLWSNodeImporter : public INNPINodeImporter {
public:
  NNPIErrorCode importNode(Node *n, NNPIImporter &importer) override {
    auto *glowSLWS =
        llvm::dyn_cast<RowwiseQuantizedSparseLengthsWeightedSumNode>(n);
    LOG_AND_RETURN_IF_NOT(ERROR, glowSLWS, "Bad node type", NNPI_INVALID_PARAM);

    LOG_NNPI_ERROR_RETURN_VALUE(
        importer.addTensor(nodeValueName(glowSLWS->getData()),
                           /* alternativeLayout */ false,
                           nodeValueName(glowSLWS->getScales()),
                           nodeValueName(glowSLWS->getOffsets())),
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

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLWS->getName().begin(),
        nodeValueName(glowSLWS->getData()).c_str(),
        nodeValueName(glowSLWS->getResult()).c_str(),
        nodeValueName(glowSLWS->getWeights()).c_str(),
        nodeValueName(glowSLWS->getIndices()).c_str(),
        nodeValueName(glowSLWS->getLengths()).c_str(), usFp32Accum);
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

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLWS->getName().begin(),
        nodeValueName(glowSLWS->getData()).c_str(),
        nodeValueName(glowSLWS->getResult()).c_str(), NULL,
        nodeValueName(glowSLWS->getIndices()).c_str(),
        nodeValueName(glowSLWS->getLengths()).c_str(), usFp32Accum);
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

    return nnpiNetworkAddSparseLengthsWeightedSumOp(
        importer.getNetwork(), glowSLWS->getName().begin(),
        nodeValueName(glowSLWS->getData()).c_str(),
        nodeValueName(glowSLWS->getResult()).c_str(),
        nodeValueName(glowSLWS->getWeights()).c_str(),
        nodeValueName(glowSLWS->getIndices()).c_str(),
        nodeValueName(glowSLWS->getLengths()).c_str(), usFp32Accum);
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

//////////////////////////////////////////////////////////////////////////
namespace {
std::unordered_map<
    std::string,
    std::unique_ptr<INNPINodeImporter>>::value_type importerInit[] = {
    {"", nullptr},
    {"Convolution", std::make_unique<ConvolutionNodeImporter>()},
    {"Transpose", std::make_unique<TransposeNodeImporter>()},
    {"MaxPool",
     std::make_unique<PoolNodeImporter<glow::MaxPoolNode, NNPI_POOL_MAX>>()},
    {"AvgPool",
     std::make_unique<PoolNodeImporter<glow::AvgPoolNode, NNPI_POOL_AVG>>()},
    {"FullyConnected", std::make_unique<FullyConnectedNodeImporter>()},
    {"SoftMax", std::make_unique<SoftMaxNodeImporter>()},
    {"Save", std::make_unique<SaveNodeImporter>()},
    {"Relu", std::make_unique<ReluNodeImporter>()},
    {"PRelu", std::make_unique<PReluNodeImporter>()},
    {"Exp", std::make_unique<
                UnaryEltwiseNodeImporter<glow::ExpNode, NNPI_ELTWISE_EXP>>()},
    {"Max", std::make_unique<
                BinaryEltwiseNodeImporter<glow::MaxNode, NNPI_ELTWISE_MAX>>()},
    {"Min", std::make_unique<
                BinaryEltwiseNodeImporter<glow::MinNode, NNPI_ELTWISE_MIN>>()},
    {"Add", std::make_unique<
                BinaryEltwiseNodeImporter<glow::AddNode, NNPI_ELTWISE_ADD>>()},
    {"Mul", std::make_unique<
                BinaryEltwiseNodeImporter<glow::MulNode, NNPI_ELTWISE_MUL>>()},
    {"Div", std::make_unique<
                BinaryEltwiseNodeImporter<glow::DivNode, NNPI_ELTWISE_DIV>>()},
    {"Sub", std::make_unique<
                BinaryEltwiseNodeImporter<glow::SubNode, NNPI_ELTWISE_SUB>>()},
    {"Pow", std::make_unique<
                BinaryEltwiseNodeImporter<glow::PowNode, NNPI_ELTWISE_POW>>()},
    {"CmpEQ",
     std::make_unique<
         BinaryEltwiseNodeImporter<glow::CmpEQNode, NNPI_ELTWISE_EQ>>()},
    {"CmpLTE",
     std::make_unique<
         BinaryEltwiseNodeImporter<glow::CmpLTENode, NNPI_ELTWISE_LTE>>()},
    {"CmpLT",
     std::make_unique<
         BinaryEltwiseNodeImporter<glow::CmpLTNode, NNPI_ELTWISE_LESS>>()},
    {"ArgMax", std::make_unique<ArgMaxNodeImporter>()},
    {"Reshape", std::make_unique<ReshapeNodeImporter>()},
    {"Quantize", std::make_unique<ConvertNodeImporter<QuantizeNode>>()},
    {"Dequantize", std::make_unique<ConvertNodeImporter<DequantizeNode>>()},
    {"RescaleQuantized",
     std::make_unique<ConvertNodeImporter<RescaleQuantizedNode>>()},
    {"ConvertTo", std::make_unique<ConvertNodeImporter<ConvertToNode>>()},
    {"MatMul", std::make_unique<MatMulNodeImporter<MatMulNode>>()},
    {"BatchMatMul", std::make_unique<MatMulNodeImporter<BatchMatMulNode>>()},
    {"Slice", std::make_unique<SliceNodeImporter>()},
    {"Sigmoid", std::make_unique<SigmoidNodeImporter>()},
    {"Tanh", std::make_unique<TanhNodeImporter>()},
    {"Concat", std::make_unique<ConcatNodeImporter>()},
    {"Tile", std::make_unique<TileNodeImporter>()},
    {"Gather", std::make_unique<GatherNodeImporter>()},
    {"BatchedReduceAdd", std::make_unique<ReduceAddNodeImporter>()},
    {"Log", std::make_unique<LogNodeImporter>()},
    {"TopK", std::make_unique<TopkNodeImporter>()},
    {"BatchedReduceMean",
     std::make_unique<ReduceMultAxesNodeImporter<glow::BatchedReduceMeanNode,
                                                 NNPI_REDUCE_MEAN>>()},
    {"BatchedReduceMin",
     std::make_unique<ReduceMultAxesNodeImporter<glow::BatchedReduceMinNode,
                                                 NNPI_REDUCE_MIN>>()},
    {"Splat", std::make_unique<SplatNodeImporter>()},
    {"SparseLengthsWeightedSum", std::make_unique<SLWSNodeImporter>()},
    {"SparseLengthsSum", std::make_unique<SLSNodeImporter>()},
    {"FusedRowwiseQuantizedSparseLengthsSum",
     std::make_unique<FRQSLSNodeImporter>()},
    {"Select", std::make_unique<SelectNodeImporter>()},
    {"LocalResponseNormalization", std::make_unique<LRNNodeImporter>()},
    {"RowwiseQuantizedFullyConnected", std::make_unique<RQFCNodeImporter>()},
    {"ReplaceNaN", std::make_unique<ReplaceNaNNodeImporter>()},
    {"GatherRanges", std::make_unique<GatherRangesNodeImporter>()},
    {"BatchedAdd", std::make_unique<BatchAddNodeImporter>()},
    {"RowwiseQuantizedSparseLengthsWeightedSum",
     std::make_unique<RQSLWSNodeImporter>()},
    {"FusedRowwiseQuantizedSparseLengthsWeightedSum",
     std::make_unique<FRQSLWSNodeImporter>()},
    {"LengthsRangeFill", std::make_unique<LengthsRangeFillNodeImporter>()},
    {"BatchOneHot", std::make_unique<BatchOneHotNodeImporter>()},
    {"NNPICustomDSP", std::make_unique<NNPICustomDSPNodeImporter>()},
    {"SpaceToDepth", std::make_unique<SpaceToDepthNodeImporter>()},
    {"Clip", std::make_unique<ClipNodeImporter>()},
};
}

const std::unordered_map<std::string, std::unique_ptr<INNPINodeImporter>>
    NNPIImporter::nodeImporters_ = {
        std::make_move_iterator(std::begin(importerInit)),
        std::make_move_iterator(std::end(importerInit))};
