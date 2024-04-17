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

#include "NNPIResource.h"
#include "InferenceContext.h"
#include "NNPIAdapterContainer.h"
#include "NNPIUtils.h"
#include "nnpi_inference.h"
#include <fstream>
#include <iomanip>
#include <sstream>

static size_t CalcDescSize(const NNPIResourceDesc *desc) {
  if (desc->numDims == 0) {
    return 0;
  }

  size_t elements = 1;
  for (uint32_t d = 0; d < desc->numDims; d++) {
    elements *= desc->dims[d];
  }
  size_t elemSize = 0;
  switch (desc->dataType) {
  case NNPI_INF_PRECISION_FLOAT32:
  case NNPI_INF_PRECISION_INT32:
  case NNPI_INF_PRECISION_UINT32:
    elemSize = 4;
    break;
  case NNPI_INF_PRECISION_FLOAT16:
    elemSize = 2;
    break;
  case NNPI_INF_PRECISION_INT8:
  case NNPI_INF_PRECISION_UINT8:
    // case NNPI_INF_PRECISION_BOOLEAN://todo: uncomment once supported
    elemSize = 1;
    break;
  default:
    LOG_AND_RETURN_IF(ERROR, true, "Invalid precision", 0);
    break;
  }

  return elements * elemSize;
}

static std::string ConvertHexTextFormat(void *data, size_t size) {
  std::stringstream ss;

  uint8_t *buf = reinterpret_cast<uint8_t *>(data);

  for (size_t cl = 0; cl < size; cl += 64) {
    size_t size_to_read = std::min((size_t)64, size - cl);
    for (size_t i = 0; i < size_to_read; i++) {
      ss << std::setfill('0') << std::setw(2) << std::uppercase << std::hex
         << (uint32_t)(buf[cl + (size_to_read - 1 - i)]);
    }

    ss << "\n";
  }

  return ss.str();
}

static void DumpToFile(const std::string &filename, void *data, size_t size) {
  std::fstream fs(filename, std::ios::out | std::ios::binary);

  if (fs.is_open()) {
    auto res = ConvertHexTextFormat(data, size);
    fs << res;
    fs.close();
  } else {
    LOG(ERROR) << "cannot open the file \"" << filename << "\" for writing";
  }
}

namespace glow {
namespace runtime {

static std::shared_ptr<NNPIResource>
findResourceForDevice(const ResourceUsers &users, NNPIDeviceContext device) {
  for (auto reader : users.readers) {
    if (reader && reader->getDevice() == device) {
      return reader;
    }
  }
  for (auto writer : users.writers) {
    if (writer && writer->getDevice() == device) {
      return writer;
    }
  }
  return nullptr;
}

NNPIResource::NNPIResource() {
  pAdapter_ = nullptr;
  device_ = NNPI_INVALID_NNPIHANDLE;
  memset(name_, 0, sizeof(name_));
  memset(&desc_, 0, sizeof(desc_));
  deviceResource_ = NNPI_INVALID_NNPIHANDLE;
  hostResource_ = NNPI_INVALID_NNPIHANDLE;
  hostPtr_ = nullptr;
  copyCommand_ = NNPI_INVALID_NNPIHANDLE;
  partialSize_ = -1;
  usage_ = ResourceUsage::None;
  deviceOptions_ = nullptr;
  cmdListIdx_ = UINT32_MAX;
  ownsDeviceResource_ = true;
  ownsHostResource_ = false;
  p2pDevice_ = NNPI_INVALID_NNPIHANDLE;
  p2pDeviceResource_ = NNPI_INVALID_NNPIHANDLE;
}

NNPIResource::~NNPIResource() {
  if (copyCommand_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_INF_IF_ERROR(nnpiCopyCommandDestroy(copyCommand_),
                          "Failed to destroy NNPI copy command");
  }
  if (ownsDeviceResource_ && (deviceResource_ != NNPI_INVALID_NNPIHANDLE)) {
    LOG_NNPI_INF_IF_ERROR(nnpiDeviceResourceDestroy(deviceResource_),
                          "Failed to destroy NNPI device resource");
  }
  if (ownsHostResource_ && (hostResource_ != NNPI_INVALID_NNPIHANDLE)) {
    LOG_NNPI_INF_IF_ERROR(nnpiHostResourceDestroy(hostResource_),
                          "Failed to destroy NNPI host resource");
  }
  // Do not destroy the adapter or device context as the Resource doesn't own
  // them but only keeps reference for it's usage.
}

bool NNPIResource::init(const NNPIObjectName name,
                        std::shared_ptr<NNPIDeviceOptions> deviceOptions,
                        NNPIAdapterContainer *adapter, NNPIDeviceContext device,
                        const NNPIResourceDesc *desc,
                        NNPIResource::ResourceUsage usage,
                        PlaceholderUsageMap *phUsage) {
  if (name == nullptr || desc == nullptr || deviceOptions == nullptr) {
    return false;
  }
  if (deviceOptions->inferOnDevice &&
      (adapter->getHandle() == NNPI_INVALID_NNPIHANDLE ||
       device == NNPI_INVALID_NNPIHANDLE)) {
    return false;
  }

  std::strncpy(name_, name, sizeof(NNPIObjectName));
  device_ = device;
  pAdapter_ = adapter;
  deviceOptions_ = deviceOptions;
  usage_ = usage;
  desc_ = *desc;

  // At this point it's safe to call NNPIResource::dump(), so do so if we return
  // false, representing some issue has occurred during initialization.
  ScopeGuard dumpResourceInfo(
      [&]() { LOG(ERROR) << "Error initializing " << dump(); });

  ResourceUsers *users = nullptr;
  if (phUsage) {
    users = &(phUsage->at(name_));
    LOG_AND_RETURN_IF_NOT(ERROR, users, "Invalid resource users", false);
  }

  if (!deviceOptions_->inferOnDevice) {
    // Handle stuff for ice ref (or compile only path).
    size_t resourceSize = CalcDescSize(&desc_);
    LOG_AND_RETURN_IF(ERROR, resourceSize == 0,
                      "Failed to calculate resource size", false);
    refStorage_.resize(resourceSize);
    if (refStorage_.size() != resourceSize) {
      LOG_ERROR_IF_NOT(0) << "Failed to allocate memory for reference storage";
      return false;
    }
    hostPtr_ = &(refStorage_[0]);
    dumpResourceInfo.dismiss();
    return true;
  }

  if (deviceOptions_->disableDRT || (users && users->disableDRT)) {
    switch (usage_) {
    case ResourceUsage::DRTInput:
      usage_ = ResourceUsage::InputResource;
      break;
    case ResourceUsage::DRTOutput:
      usage_ = ResourceUsage::OutputResource;
      break;
    default:; // Do nothing.
    }
  }
  if (deviceOptions_->disableP2P || (users && users->disableP2P)) {
    switch (usage_) {
    case ResourceUsage::P2PInput:
      usage_ = ResourceUsage::InputResource;
      break;
    case ResourceUsage::P2POutput:
      usage_ = ResourceUsage::OutputResource;
      break;
    default:; // Do nothing.
    }
  }

  // Create host resource (pinned and aligned allocation).
  NNPIResourceDesc hostResDesc =
      desc_; // Make a copy of the desc to overwrite attributes.
  hostResDesc.hostAttrib.locklessExecution =
      1; // Set host resource to lockless.
  switch (usage_) {
  case ResourceUsage::InputResource:
  case ResourceUsage::OutputResource: {
    hostPtr_ = users ? users->tensor->getUnsafePtr() : nullptr;
    hostResource_ = hostPtr_ ? pAdapter_->getResourceForPtr(hostPtr_)
                             : NNPI_INVALID_NNPIHANDLE;
    if (hostResource_ != NNPI_INVALID_NNPIHANDLE) {
      // Tensor bound to placeholder was already allocated as a host resource.
      // No need to allocate another one.
      ownsHostResource_ = false;
    } else { // Allocate a private host resource for DMA.
      LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
          nnpiHostResourceCreate(pAdapter_->getHandle(), &hostResDesc,
                                 &hostResource_),
          "Failed to create NNPI host resource");

      // Query host resource for address (not locking).
      LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
          nnpiHostResourceLock(hostResource_, NNPI_LOCKLESS_QUERY_ADDRESS,
                               UINT32_MAX, &hostPtr_),
          "Failed to lock host resource");
      memset(hostPtr_, 0,
             CalcDescSize(&desc_)); // Clear host resource to zero for compile
                                    // only path (USE_ICE_T).
      ownsHostResource_ = true;
    }
  } break;
  case ResourceUsage::StaticInputResource:
  case ResourceUsage::P2PInput:
  case ResourceUsage::P2POutput:
  case ResourceUsage::DRTInput:
  case ResourceUsage::DRTOutput:
    // No host resource is needed
    break;
  default:
    LOG_AND_RETURN_IF_NOT(ERROR, 0, "Invalid usage", false);
  }

  // Create device resource.
  bool allocateDeviceResource = false;

  std::shared_ptr<NNPIResource> sharedResource = nullptr;
  if (phUsage) {
    sharedResource = findResourceForDevice(*users, device_);
  }

  switch (usage_) {
  case ResourceUsage::InputResource:
  case ResourceUsage::OutputResource:
    // Normal create.
    allocateDeviceResource = true;
    break;

  case ResourceUsage::StaticInputResource:
  case ResourceUsage::DRTInput:
    // Potential shared (allocate if doesnt exist).
    if (sharedResource) {
      // If exists on device - share it.
      deviceResource_ = sharedResource->getDeviceResource();
      ownsDeviceResource_ = false;
      allocateDeviceResource = false;
    } else {
      allocateDeviceResource = true;
    }
    break;

  case ResourceUsage::P2PInput:
    // Create p2p input.
    desc_.deviceAttrib.p2pUsage = NNPI_P2P_USAGE_DST;
    desc_.deviceAttrib.p2pDepth = 1;
    allocateDeviceResource = true;
    break;

  case ResourceUsage::P2POutput:
    // Create p2p output.
    desc_.deviceAttrib.p2pUsage = NNPI_P2P_USAGE_SRC;
    desc_.deviceAttrib.p2pDepth = 1;
    allocateDeviceResource = true;
    break;

  case ResourceUsage::DRTOutput:
    // Must be shared (creation order maintains readers allocated before
    // writers).
    LOG_AND_RETURN_IF_NOT(
        ERROR, sharedResource,
        "Missing DRT resource (should have been created already)", false);
    deviceResource_ = sharedResource->getDeviceResource();
    ownsDeviceResource_ = false;
    allocateDeviceResource = false;
    break;

  default:
    LOG_AND_RETURN_IF_NOT(ERROR, 0, "Invalid usage", false);
  }

  if (allocateDeviceResource) {
    LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
        nnpiDeviceResourceCreate(device_, &desc_, &deviceResource_),
        "Failed to create NNPI device resource");
  }

  // Create copy command.
  switch (usage_) {
  case ResourceUsage::InputResource:
    LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
        nnpiCopyCommandCreateHostToDevice(device_, deviceResource_,
                                          hostResource_, &copyCommand_),
        "Failed to create NNPI copy command (input)");
    break;
  case ResourceUsage::OutputResource:
    LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
        nnpiCopyCommandCreateDeviceToHost(device_, hostResource_,
                                          deviceResource_, &copyCommand_),
        "Failed to create NNPI copy command (output)");
    break;
  case ResourceUsage::P2POutput:
    LOG_AND_RETURN_IF_NOT(ERROR, users, "Missing resource users", false);
    for (auto reader : users->readers) {
      if (reader && reader->getDevice() != device) {
        p2pDevice_ = reader->getDevice();
        p2pDeviceResource_ = reader->getDeviceResource();
      }
    }
    LOG_AND_RETURN_IF_NOT(ERROR,
                          (p2pDevice_ != NNPI_INVALID_NNPIHANDLE) &&
                              (p2pDeviceResource_ != NNPI_INVALID_NNPIHANDLE),
                          "Can't find p2p counterpart", false);
    LOG_NNPI_INF_IF_ERROR_RETURN_FALSE(
        nnpiCopyCommandCreateDeviceToDevice(p2pDevice_, p2pDeviceResource_,
                                            device_, deviceResource_,
                                            &copyCommand_),
        "Failed to create NNPI copy command (p2p output)");
    break;
  case ResourceUsage::P2PInput:
    // The device resource is copied by the writer in the
    // preceding context.
    break;
  case ResourceUsage::StaticInputResource:
    // The device resource doesn't need to be updated before
    // inference.
    break;
  case ResourceUsage::DRTInput:
    // The device resource doesn't need to be updated before
    // inference.
    break;
  case ResourceUsage::DRTOutput:
    // The device resource doesn't need to be updated before
    // inference.
    break;
  case ResourceUsage::None:
    // Do nothing - no copy command needed.
    break;
  default:
    LOG_ERROR_IF_NOT(0) << "Invalid resource usage";
    return false;
  }

  DBG(__FUNCTION__ << dump());
  dumpResourceInfo.dismiss();
  return true;
}

NNPIInferenceErrorCode
NNPIResource::preInference(Tensor *t, bool partialTensor,
                           bool requiresLastElementPadding) {
  if (usage_ != ResourceUsage::InputResource) {
    // Nothing to do here yet.
    return NNPI_INF_NO_ERROR;
  }

  // Update the host resource from the tensor content.
  updateHostResourceFromTensor(t, partialTensor, requiresLastElementPadding);

  if (deviceOptions_->dumpIOtoFiles) {
    size_t unpaddedSize = t->getUnpaddedSizeInBytes();
    const bool downcastInt64 = t->getElementType() == glow::ElemKind::Int64ITy;
    if (downcastInt64) {
      unpaddedSize /= 2;
    }
    DumpToFile(std::string("input_") + std::string(name_) + ".txt", hostPtr_,
               unpaddedSize);
  }

  return NNPI_INF_NO_ERROR;
}

NNPIInferenceErrorCode NNPIResource::postInference(Tensor *t) {
  if (usage_ != ResourceUsage::OutputResource) {
    // Nothing to do here yet.
    return NNPI_INF_NO_ERROR;
  }

  char *tensorData = t->getUnsafePtr();
  const bool upcastInt64 = t->getElementType() == glow::ElemKind::Int64ITy;
  size_t unpaddedSize = t->getUnpaddedSizeInBytes();
  if (upcastInt64) {
    // TODO: add AVX implementation.
    const size_t outputSize = unpaddedSize / t->getType().getElementSize();
    int64_t *i64Data = reinterpret_cast<int64_t *>(tensorData);
    int32_t *i32Data = reinterpret_cast<int32_t *>(hostPtr_);
    // Convert from end to start for the case where (tensorData == hostPtr_).
    for (int64_t j = (outputSize - 1); j >= 0; j--) {
      i64Data[j] = static_cast<int64_t>(i32Data[j]);
    }
  } else {
    if (tensorData != hostPtr_) {
      std::memcpy(tensorData, hostPtr_, unpaddedSize);
    }
  }

  if (deviceOptions_->dumpIOtoFiles) {
    DumpToFile(std::string("output_") + std::string(name_) + ".txt", hostPtr_,
               unpaddedSize);
  }
  return NNPI_INF_NO_ERROR;
}

bool NNPIResource::updateResourceDescFromTensorDesc(
    NNPIResourceDesc *rDesc, const NNPITensorDesc *tDesc) {
  if (tDesc == nullptr || rDesc == nullptr) {
    return false;
  }
  memset(rDesc, 0, sizeof(NNPIResourceDesc));
  rDesc->numDims = tDesc->numDims;
  for (size_t i = 0; i < rDesc->numDims; i++) {
    rDesc->dims[i] = tDesc->dims[i];
  }
  switch (tDesc->quantParams.precision) {
  case NNPI_PRECISION_FLOAT32:
    rDesc->dataType = NNPI_INF_PRECISION_FLOAT32;
    break;
  case NNPI_PRECISION_FLOAT16:
    rDesc->dataType = NNPI_INF_PRECISION_FLOAT16;
    break;
  case NNPI_PRECISION_INT32:
    rDesc->dataType = NNPI_INF_PRECISION_INT32;
    break;
  case NNPI_PRECISION_UINT32:
    rDesc->dataType = NNPI_INF_PRECISION_UINT32;
    break;
  case NNPI_PRECISION_INT8:
    rDesc->dataType = NNPI_INF_PRECISION_INT8;
    break;
  case NNPI_PRECISION_UINT8:
    rDesc->dataType = NNPI_INF_PRECISION_UINT8;
    break;
  case NNPI_PRECISION_BOOLEAN:
    rDesc->dataType = NNPI_INF_PRECISION_UINT8;
    break;
  default:
    LOG_AND_RETURN_IF(ERROR, true, "Unhandled precision", false);
    break;
  }

  return true;
}

void NNPIResource::updateDeviceResourceFromTensor(
    Tensor *t, std::function<void(Error)> resultCB) {
  LOG_AND_FAIL_CALLBACK_IF_NOT(
      t != nullptr, "Invalid tensor used to update static input", resultCB);

  LOG_AND_FAIL_CALLBACK_IF_NOT(
      updateHostResourceFromTensor(t, /* partialTensor */ false,
                                   /* requiresLastElementPadding */ false),
      "Invalid Static placeholder", resultCB);

  if (deviceOptions_->inferOnDevice) {
    const auto sizeFromDesc = CalcDescSize(&desc_);
    LOG_AND_FAIL_CALLBACK_IF_NOT(
        sizeFromDesc == t->getSizeInBytes(),
        "Tensor size " + std::to_string(t->getSizeInBytes()) +
            " doesn't match size from description " +
            std::to_string(sizeFromDesc) + " for " + name_,
        resultCB);

    LOG_AND_CALLBACK_NNPI_INF_IF_ERROR(
        nnpiDeviceResourceSubLoad(deviceResource_, 0, t->getUnsafePtr(),
                                  t->getSizeInBytes()),
        "Failed to execute device resource sub load for " + std::string(name_) +
            ", size is " + std::to_string(t->getSizeInBytes()) +
            " bytes, tensor is " + t->toString(),
        resultCB);
  }

  resultCB(Error::success());
}

bool NNPIResource::updateHostResourceFromTensor(
    Tensor *t, bool partialTensor, bool requiresLastElementPadding) {
  // Prepare data on the host resource (for ice-ref use int32sTorage).
  char *tensorData = t->getUnsafePtr();
  const bool downcastInt64 = t->getElementType() == glow::ElemKind::Int64ITy;
  size_t paddedSize = t->getSizeInBytes();
  size_t unpaddedSize = t->getUnpaddedSizeInBytes();
  const bool partialData = (unpaddedSize != paddedSize);

  if (usage_ == ResourceUsage::StaticInputResource) {
    LOG_AND_RETURN_IF(ERROR, downcastInt64,
                      "Static placeholder not allowed to be of type Int64",
                      false);
    LOG_AND_RETURN_IF(ERROR, partialData,
                      "Static placeholders are not allowed to do partial copy",
                      false);
    if (deviceOptions_->inferOnDevice) {
      // nothing else to do for static placeholders when running on device
      return true;
    }
  }

  if (downcastInt64) {
    paddedSize /= 2;
    unpaddedSize /= 2;
  }

  // Copy or convert.
  if (downcastInt64) { // Convert
    switch (deviceOptions_->avxType) {
    case NNPI_AVX_NONE:
      convertI64toI32(reinterpret_cast<const int64_t *>(tensorData),
                      reinterpret_cast<int32_t *>(hostPtr_),
                      unpaddedSize / sizeof(int32_t));
      break;
    case NNPI_AVX_AVX512:
      convertI64toI32_AVX512(reinterpret_cast<const int64_t *>(tensorData),
                             reinterpret_cast<int32_t *>(hostPtr_),
                             unpaddedSize / sizeof(int32_t));
      break;
    default:
      LOG(ERROR) << "Invalid avxType=" << deviceOptions_->avxType;
    }
  } else if (unpaddedSize) { // Only copy if there is data. Required because
                             // tensorData cannot be null for memcpy.
    if (tensorData != hostPtr_) {
      std::memcpy(hostPtr_, tensorData, unpaddedSize);
    }
  }

  // Pad with zeros or last element if needed.
  if (partialData && !partialTensor) {
    if (!requiresLastElementPadding) {
      memset(reinterpret_cast<uint8_t *>(hostPtr_) + unpaddedSize, 0,
             paddedSize - unpaddedSize);
    } else {
      // size of elements in hostPtr_
      auto hostElementSize =
          downcastInt64 ? sizeof(int32_t) : t->getType().getElementSize();
      int numElements = unpaddedSize / hostElementSize;
      int totalElements = paddedSize / hostElementSize;
      if (hostElementSize == 1) {
        std::fill(reinterpret_cast<uint8_t *>(hostPtr_) + numElements,
                  reinterpret_cast<uint8_t *>(hostPtr_) + totalElements,
                  reinterpret_cast<const uint8_t *>(hostPtr_)[numElements - 1]);
      } else if (hostElementSize == 2) {
        std::fill(
            reinterpret_cast<uint16_t *>(hostPtr_) + numElements,
            reinterpret_cast<uint16_t *>(hostPtr_) + totalElements,
            reinterpret_cast<const uint16_t *>(hostPtr_)[numElements - 1]);
      } else if (hostElementSize == 4) {
        std::fill(
            reinterpret_cast<uint32_t *>(hostPtr_) + numElements,
            reinterpret_cast<uint32_t *>(hostPtr_) + totalElements,
            reinterpret_cast<const uint32_t *>(hostPtr_)[numElements - 1]);
      } else if (hostElementSize == 8) {
        std::fill(
            reinterpret_cast<uint64_t *>(hostPtr_) + numElements,
            reinterpret_cast<uint64_t *>(hostPtr_) + totalElements,
            reinterpret_cast<const uint64_t *>(hostPtr_)[numElements - 1]);
      } else {
        LOG(ERROR) << "Invalid Tensor type, padding is unsuccessful";
      }
    }
    partialSize_ = -1;
  } else if (partialData) {
    partialSize_ = unpaddedSize;
  } else {
    partialSize_ = -1;
  }
  return true;
}

std::string NNPIResource::dump() const {
  std::stringstream stream;
  stream << "NNPIResource: \"" << name_ << '"';
  stream << ", DescSize: " << CalcDescSize(&desc_);
  stream << ", Usage: " << static_cast<int>(usage_);
  stream << ", Adapter: " << pAdapter_;
  stream << ", Device: " << device_;
  stream << ", DeviceResource: " << deviceResource_;
  stream << ", HostResource: " << hostResource_;
  stream << ", HostPtr: " << hostPtr_;
  stream << ", CopyCommand: " << copyCommand_;
  stream << ", CommandListIndex: " << cmdListIdx_;
  stream << ", PartialSize: " << partialSize_;
  stream << ", ownsDeviceResource: " << ownsDeviceResource_;
  stream << ", p2pDevice: " << p2pDevice_;
  stream << ", p2pDeviceResource: " << p2pDeviceResource_;
  return stream.str();
}

} // namespace runtime
} // namespace glow
