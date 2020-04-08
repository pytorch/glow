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

/// This class Holds metadata for an inference resource.
class NNPIResource {
public:
  /// Usage type of the resource.
  enum class ResourceUsage {
    None,
    InputResource,
    OutputResource,
    StaticInputResource,
  };

  /// Constructor.
  NNPIResource();
  /// Destructor.
  ~NNPIResource();
  /// Invalidating copy constructor to prevent multiple owners of the same
  /// handle.
  NNPIResource(const NNPIResource &) = delete;
  /// Update a device resource contents from a provided tensor.
  void UpdateDeviceResourceFromTensor(Tensor *t,
                                      std::function<void(Error)> resultCB);

  /// Initialize a resource.
  bool init(const NNPIObjectName name,
            std::shared_ptr<NNPIDeviceOptions> deviceOptions,
            NNPIAdapter adapter, NNPIDeviceContext device,
            const NNPIResourceDesc *desc, ResourceUsage usage);

  /// Pre-inference processing on the resource.
  NNPIInferenceErrorCode PreInference(Tensor *t, bool partialTensor);
  /// Post-inference processing on the resource.
  NNPIInferenceErrorCode PostInference(Tensor *t);

  /// Getters.
  inline const NNPIObjectName &GetName() const { return name_; }
  inline NNPIDeviceContext GetDevice() const { return device_; }
  inline const NNPIResourceDesc &GetDesc() const { return desc_; }
  inline NNPIDeviceResource GetDeviceResource() const {
    return deviceResource_;
  }
  inline NNPIHostResource GetHostResource() const { return hostResource_; }
  inline void *GetHostPtr() const { return hostPtr_; }
  inline NNPICopyCommand GetCopyCommand() const { return copyCommand_; }
  inline uint32_t GetCmdListIdx() const { return cmdListIdx_; }
  inline void SetCmdListIdx(uint32_t idx) { cmdListIdx_ = idx; }
  inline uint64_t GetPartialSize() const { return partialSize_; }
  inline ResourceUsage GetUsage() const { return usage_; }

  /// Update a given NNPIResourceDesc struct from the data in an NNPITensorDesc
  /// struct.
  static bool UpdateResourceDescFromTensorDesc(NNPIResourceDesc *rDesc,
                                               const NNPITensorDesc *tDesc);
  /// Dump the state of a resource object
  std::string Dump() const;

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

  /// Update the owned host resource with data taken from the given tensor.
  // return true when successfull, false otherwise.
  bool updateHostResourceFromTensor(Tensor *t, bool partialTensor);
};

} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_NNPI_NNPIRESOURCE_H
