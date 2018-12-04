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
#include "GlowOnnxManager.h"

#include "Base.h"

#include <mutex>

namespace glow {
namespace onnxifi {
GlowOnnxManager &GlowOnnxManager::get() {
  static GlowOnnxManager manager;
  return manager;
}

void GlowOnnxManager::addBackendId(BackendIdPtr backendId) {
  assert(!isValid(backendId) && backendId != nullptr);
  {
    std::lock_guard<std::mutex> lock(m_);
    auto res = backendIds_.insert(backendId);
    assert((res.second && *res.first) && "Failed to add new BackendId");
  }
}

BackendPtr GlowOnnxManager::createBackend(BackendIdPtr backendId) {
  assert(isValid(backendId));
  BackendPtr backend = new Backend(backendId);
  {
    std::lock_guard<std::mutex> lock(m_);
    auto res = backends_.insert(backend);
    assert((res.second && *res.first) && "Failed to create new Backend");
  }
  return backend;
}

EventPtr GlowOnnxManager::createEvent() {
  EventPtr event = new glow::onnxifi::Event();
  {
    std::lock_guard<std::mutex> lock(m_);
    auto res = events_.insert(event);
    assert((res.second && *res.first) && "Failed to create new Event");
  }
  return event;
}

GraphPtr GlowOnnxManager::createGraph(BackendPtr backend) {
  assert(isValid(backend));
  GraphPtr graph = new glow::onnxifi::Graph(backend);
  {
    std::lock_guard<std::mutex> lock(m_);
    auto res = graphs_.insert(graph);
    assert((res.second && *res.first) && "Failed to create new Graph");
  }
  return graph;
}

bool GlowOnnxManager::isValid(BackendIdPtr backendId) {
  std::lock_guard<std::mutex> lock(m_);
  return backendId && backendIds_.count(backendId) == 1;
}

bool GlowOnnxManager::isValid(BackendPtr backend) {
  std::lock_guard<std::mutex> lock(m_);
  return backend && backends_.count(backend) == 1;
}

bool GlowOnnxManager::isValid(EventPtr event) {
  std::lock_guard<std::mutex> lock(m_);
  return event && events_.count(event) == 1;
}

bool GlowOnnxManager::isValid(GraphPtr graph) {
  std::lock_guard<std::mutex> lock(m_);
  return graph && graphs_.count(graph) == 1;
}

void GlowOnnxManager::release(BackendIdPtr backendId) {
  assert(isValid(backendId) && "trying to release an invalid BackendId");
  {
    std::lock_guard<std::mutex> lock(m_);
    backendIds_.erase(backendId);
  }
  delete backendId;
}

void GlowOnnxManager::release(BackendPtr backend) {
  assert(isValid(backend) && "trying to release an invalid Backend");
  {
    std::lock_guard<std::mutex> lock(m_);
    backends_.erase(backend);
  }
  delete backend;
}

void GlowOnnxManager::release(EventPtr event) {
  assert(isValid(event) && "trying to release an invalid Event");
  {
    std::lock_guard<std::mutex> lock(m_);
    events_.erase(event);
  }
  delete event;
}

void GlowOnnxManager::release(GraphPtr graph) {
  assert(isValid(graph) && "trying to release an invalid Graph");
  {
    std::lock_guard<std::mutex> lock(m_);
    graphs_.erase(graph);
  }
  delete graph;
}
} // namespace onnxifi
} // namespace glow
