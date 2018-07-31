/*
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
#include "Base.h"

namespace glow {
namespace onnxifi {

bool Event::signal() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (fired_) {
      return false;
    }
    fired_ = true;
  }
  cond_.notify_all();
  return true;
}

void Event::wait() {
  std::unique_lock<std::mutex> guard(mutex_);
  cond_.wait(guard, [this] { return fired_; });
}

onnxStatus Graph::initGraph(const void *onnxModel, size_t onnxModelSize,
                            uint32_t weightCount,
                            const onnxTensorDescriptorV1 *weightDescriptors) {
  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus Graph::run() { return ONNXIFI_STATUS_SUCCESS; }

onnxStatus Graph::setIO(uint32_t inputsCount,
                        const onnxTensorDescriptorV1 *inputDescriptors,
                        uint32_t outputsCount,
                        const onnxTensorDescriptorV1 *outputDescriptors) {
  return ONNXIFI_STATUS_SUCCESS;
}

} // namespace onnxifi
} // namespace glow
