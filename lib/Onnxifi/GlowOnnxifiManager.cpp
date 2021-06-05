/*
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
#include "GlowOnnxifiManager.h"

#include <mutex>

namespace glow {
namespace onnxifi {
GlowOnnxifiManager &GlowOnnxifiManager::get() {
  static GlowOnnxifiManager manager;
  return manager;
}

BackendPtr GlowOnnxifiManager::createBackend(llvm::StringRef backendName,
                                             bool useOnnx,
                                             bool forQuantization) {
  std::lock_guard<std::mutex> lock(m_);

  BackendPtr backend;
  if (forQuantization) {
    backend = new onnxifi::Backend(backendName, useOnnx);
  } else {
    auto hostManager = getOrCreateHostManager(backendName);
    backend =
        new onnxifi::HostManagerBackend(hostManager, backendName, useOnnx);
  }

  auto res = backends_.insert(backend);

  (void)res;
  assert((res.second && *res.first) && "Failed to add new Backend");

  return backend;
}

EventPtr GlowOnnxifiManager::createEvent() {
  EventPtr event = new onnxifi::Event();

  std::lock_guard<std::mutex> lock(m_);

  auto res = events_.insert(event);

  (void)res;
  assert((res.second && *res.first) && "Failed to create new Event");
  return event;
}

GraphPtr GlowOnnxifiManager::createGraph(BackendPtr backend,
                                         QuantizationMode quantizationMode) {
  assert(isValid(backend));

  GraphPtr graph;

  if (quantizationMode == QuantizationMode::None) {
    graph = new onnxifi::HostManagerGraph(backend);
  } else {
    graph = new onnxifi::InlineGraph(backend, quantizationMode);
  }

  std::lock_guard<std::mutex> lock(m_);

  auto res = graphs_.insert(graph);

  (void)res;
  assert((res.second && *res.first) && "Failed to create new Graph");
  return graph;
}

std::shared_ptr<runtime::HostManager>
GlowOnnxifiManager::getOrCreateHostManager(llvm::StringRef backendName) {
  std::shared_ptr<runtime::HostManager> hostManager;

  auto it = hostManagers_.find(backendName.str());

  if (it != hostManagers_.end()) {
    hostManager = it->second.lock();
  }

  if (!hostManager) {
    hostManager = onnxifi::HostManagerBackend::createHostManager(backendName);
    assert(hostManager);
    hostManagers_[backendName.str()] = hostManager;
  }

  return hostManager;
}

bool GlowOnnxifiManager::isValid(BackendPtr backend) const {
  std::lock_guard<std::mutex> lock(m_);
  return backend && backends_.count(backend) == 1;
}

bool GlowOnnxifiManager::isValid(EventPtr event) const {
  std::lock_guard<std::mutex> lock(m_);
  return event && events_.count(event) == 1;
}

bool GlowOnnxifiManager::isValid(GraphPtr graph) const {
  std::lock_guard<std::mutex> lock(m_);
  return graph && graphs_.count(graph) == 1;
}

void GlowOnnxifiManager::release(BackendPtr backend) {
  // TODO: fix this so that a HostManager is deleted when all backends
  // holding pointers to that HostManager are deleted.
  std::lock_guard<std::mutex> lock(m_);
  size_t erased = backends_.erase(backend);

  if (erased) {
    delete backend;
  }

  if (backends_.empty()) {
    hostManagers_.clear();
  }
}

void GlowOnnxifiManager::release(EventPtr event) {
  size_t erased;
  {
    std::lock_guard<std::mutex> lock(m_);
    erased = events_.erase(event);
  }
  if (erased) {
    delete event;
  }
}

void GlowOnnxifiManager::release(GraphPtr graph) {
  size_t erased;
  {
    std::lock_guard<std::mutex> lock(m_);
    erased = graphs_.erase(graph);
  }
  if (erased) {
    delete graph;
  }
}
} // namespace onnxifi
} // namespace glow
