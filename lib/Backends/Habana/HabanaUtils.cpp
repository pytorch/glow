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

#include "HabanaUtils.h"

namespace glow {
const char *statusStr(synStatus status) {
  static const char *strs[] = {
      "Success",
      "Invalid argument",
      "Command buffer full",
      "Out of host memory",
      "Out of device memory",
      "Object already initialized",
      "Object not initialized",
      "Command submission failed",
      "No device found",
      "Device type mismatch",
      "Failed to initialize command buffer",
      "Failed to free command buffer",
      "Failed to map command buffer",
      "Failed to unmap command buffer",
      "Failed to allocate device memory",
      "Failed to free device memory",
      "Not enough devices found",
      "Device reset",
      "Unsupported",
      "Wrong params file",
      "Device already acquired",
      "Failed",
  };
  static_assert(sizeof(strs) / sizeof(strs[0]) == synFail + 1);
  if (status < synSuccess || status > synFail) {
    return "Invalid status";
  }
  return strs[status];
}
} // namespace glow
