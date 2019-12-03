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
#ifndef GLOW_BASE_DEVICETENSORTRANSFERMANAGER_H
#define GLOW_BASE_DEVICETENSORTRANSFERMANAGER_H

#include "glow/Base/Tensor.h"
#include "glow/Support/Error.h"

#include <functional>

namespace glow {

class Tensor;

#define GLOW_DRT_DEFAULT_CB [](Error e) { ERR_TO_VOID(std::move(e)); }

class DeviceTensorTransferManager {
public:
  virtual ~DeviceTensorTransferManager() {}
  /// Copies the contents of \p tensor from the host to the \p location address
  /// on this device. Updates the tensor residency info.
  virtual void transferToDevice(
      Tensor &tensor, void *locationContext = nullptr,
      std::function<void(Error)> resultCB = GLOW_DRT_DEFAULT_CB) = 0;

  /// Copies the device buffer associated with \p tensor to the host.
  /// The tensor must be resident on this device. If \p release is true, frees
  /// the device memory. Updates the tensor residency info.
  virtual void transferFromDevice(
      Tensor &tensor, bool release = true,
      std::function<void(Error)> resultCB = GLOW_DRT_DEFAULT_CB) = 0;

  /// Releases the device buffer associated with \p tensor.
  virtual bool releaseDeviceTensor(void *locationContext) = 0;
};

} // namespace glow

#endif // GLOW_BASE_DEVICETENSORTRANSFERMANAGER_H
