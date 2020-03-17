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

#ifndef GLOW_NNPI_ADAPTER_H
#define GLOW_NNPI_ADAPTER_H

#include "DebugMacros.h"
#include "NNPIOptions.h"
#include "glow/Backend/Backend.h"
#include "nnpi_inference.h"
#include <mutex>

namespace glow {
class NNPIAdapterContainer {
public:
  NNPIAdapterContainer() : nnpiAdapter_(NNPI_INVALID_NNPIHANDLE) {}

  ~NNPIAdapterContainer();

  NNPIAdapter get(bool inferOnDevice);

private:
  /// NNPI Adapter handle.
  NNPIAdapter nnpiAdapter_;
  /// Lock to synchronize function adding/removing to/from the device manager.
  std::mutex adapterMutex_;
};
} // namespace glow
#endif // GLOW_NNPI_ADAPTER_H
