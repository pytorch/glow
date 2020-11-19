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

#include "NNPIAdapterContainer.h"

using namespace glow;

NNPIAdapterContainer::~NNPIAdapterContainer() {
  while (!hostResourceMap_.empty()) {
    freeHostResource(hostResourceMap_.begin()->first);
  }

  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  if (adapter_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_INF_IF_ERROR(nnpiAdapterDestroy(adapter_),
                          "Failed to destroy NNPI adapter");
    adapter_ = NNPI_INVALID_NNPIHANDLE;
  }
}

NNPIAdapter NNPIAdapterContainer::getHandle() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  if (adapter_ == NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_INF_IF_ERROR(nnpiAdapterCreate(nullptr, &adapter_),
                          "Failed to create NNPI Adapter");
  }
  return adapter_;
}

void *NNPIAdapterContainer::allocateHostResource(size_t resourceSize) {
  CHECK(adapter_ != NNPI_INVALID_NNPIHANDLE);
  NNPIHostResource hostResource = NNPI_INVALID_NNPIHANDLE;
  NNPIResourceDesc desc;
  memset(&desc, 0, sizeof(desc));
  desc.dataType = NNPI_INF_PRECISION_UINT8;
  desc.layout = NNPI_INF_LAYOUT_NC;
  desc.numDims = 2;
  desc.dims[0] = resourceSize;
  desc.dims[1] = 1;

  LOG_NNPI_INF_IF_ERROR(nnpiHostResourceCreate(adapter_, &desc, &hostResource),
                        "Failed to create NNPI host resource");
  if (hostResource == NNPI_INVALID_NNPIHANDLE) {
    return nullptr;
  }

  void *hostPtr = nullptr;
  LOG_NNPI_INF_IF_ERROR(nnpiHostResourceLock(hostResource,
                                             NNPI_LOCKLESS_QUERY_ADDRESS,
                                             UINT32_MAX, &hostPtr),
                        "Failed to lock host resource");
  if (hostPtr == nullptr) {
    return nullptr;
  }

  {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    hostResourceMap_.insert({hostPtr, hostResource});
    return hostPtr;
  }
}

void NNPIAdapterContainer::freeHostResource(void *ptr) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  if (hostResourceMap_.count(ptr)) {
    LOG_NNPI_INF_IF_ERROR(nnpiHostResourceDestroy(hostResourceMap_.at(ptr)),
                          "Failed to destroy NNPI host resource");
    hostResourceMap_.erase(ptr);
  }
}

NNPIHostResource NNPIAdapterContainer::getResourceForPtr(void *ptr) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  auto it = hostResourceMap_.find(ptr);
  if (it == hostResourceMap_.end()) {
    return NNPI_INVALID_NNPIHANDLE;
  } else {
    return it->second;
  }
}
