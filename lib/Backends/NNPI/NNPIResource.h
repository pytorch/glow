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

#ifndef GLOW_BACKENDS_NNPI_NNPIRESOURCE_H
#define GLOW_BACKENDS_NNPI_NNPIRESOURCE_H

#include "DebugMacros.h"
#include "NNPIOptions.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "nnpi_inference.h"
#include "nnpi_transformer.h"

namespace glow {
class Tensor;

namespace runtime {

class NNPIResource;
struct ResourceUsers {
  unsigned numReaders = 0, numWriters = 0;
  std::vector<std::shared_ptr<NNPIResource>> writers;
  std::vector<std::shared_ptr<NNPIResource>> readers;
  std::unordered_set<NNPIDeviceContext> devices;
  bool disableP2P = false;
  bool disableDRT = false;
};
using PlaceholderUsageMap = std::unordered_map<std::string, ResourceUsers>;

/// This class Holds metadata for an inference resource.
class NNPIResource {
public:
  /// Usage type of the resource.
  enum class ResourceUsage {
    None,
    InputResource,
    OutputResource,
    StaticInputResource,
    P2PInput,
    P2POutput,
    DRTInput,
    DRTOutput,
  };

  /// Constructor.
  NNPIResource();
  /// Destructor.
  ~NNPIResource();
  /// Invalidating copy constructor to prevent multiple owners of the same
  /// handle.
  NNPIResource(const NNPIResource &) = delete;
  /// Update a device resource contents from a provided tensor.
  void updateDeviceResourceFromTensor(Tensor *t,
                                      std::function<void(Error)> resultCB);

  /// Initialize a resource.
  bool init(const NNPIObjectName name,
            std::shared_ptr<NNPIDeviceOptions> deviceOptions,
            NNPIAdapter adapter, NNPIDeviceContext device,
            const NNPIResourceDesc *desc, ResourceUsage usage,
            PlaceholderUsageMap *phUsage = nullptr);

  /// Pre-inference processing on the resource.
  NNPIInferenceErrorCode preInference(Tensor *t, bool partialTensor);
  /// Post-inference processing on the resource.
  NNPIInferenceErrorCode postInference(Tensor *t);

  /// Getters.
  inline const NNPIObjectName &getName() const { return name_; }
  inline NNPIDeviceContext getDevice() const { return device_; }
  inline const NNPIResourceDesc &getDesc() const { return desc_; }
  inline NNPIDeviceResource getDeviceResource() const {
    return deviceResource_;
  }
  inline NNPIHostResource getHostResource() const { return hostResource_; }
  inline void *getHostPtr() const { return hostPtr_; }
  inline NNPICopyCommand getCopyCommand() const { return copyCommand_; }
  inline uint32_t getCmdListIdx() const { return cmdListIdx_; }
  inline void setCmdListIdx(uint32_t idx) { cmdListIdx_ = idx; }
  inline uint64_t getPartialSize() const { return partialSize_; }
  inline ResourceUsage getUsage() const { return usage_; }
  inline NNPIDeviceResource getP2PDeviceResource() const {
    return p2pDeviceResource_;
  }

  /// Update a given NNPIResourceDesc struct from the data in an NNPITensorDesc
  /// struct.
  static bool updateResourceDescFromTensorDesc(NNPIResourceDesc *rDesc,
                                               const NNPITensorDesc *tDesc);
  /// Dump the state of a resource object.
  std::string dump() const;

private:
  NNPIAdapter adapter_;      // This handle isn't owned by the object.
  NNPIDeviceContext device_; // This handle isn't owned by the object.
  NNPIObjectName name_;
  NNPIResourceDesc desc_;
  NNPIDeviceResource deviceResource_;
  NNPIHostResource hostResource_;
  void *hostPtr_;
  NNPICopyCommand copyCommand_;
  uint64_t partialSize_;
  ResourceUsage usage_;
  std::shared_ptr<NNPIDeviceOptions> deviceOptions_;
  std::vector<uint8_t> refStorage_;
  uint32_t cmdListIdx_;
  bool ownsDeviceResource_; // Used for DRT (only one NNPIResource will own the
                            // device resource)
  NNPIDeviceContext p2pDevice_; // the other device used in p2p
  NNPIDeviceResource
      p2pDeviceResource_; // the resource on the other device used in p2p.

  /// Update the owned host resource with data taken from the given tensor.
  // return true when successfull, false otherwise.
  bool updateHostResourceFromTensor(Tensor *t, bool partialTensor);
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_NNPI_NNPIRESOURCE_H
