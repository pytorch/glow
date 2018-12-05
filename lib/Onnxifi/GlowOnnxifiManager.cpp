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
#include "GlowOnnxifiManager.h"

#include "Base.h"

#include <mutex>

namespace glow {
namespace onnxifi {
GlowOnnxifiManager &GlowOnnxifiManager::get() {
  static GlowOnnxifiManager manager;
  return manager;
}

void GlowOnnxifiManager::addBackendId(BackendIdPtr backendId) {
  {
    std::lock_guard<std::mutex> lock(m_);
    assert(!backendIds_.count(backendId));
    auto res = backendIds_.insert(backendId);
    (void)res;
    assert((res.second && *res.first) && "Failed to add new BackendId");
  }
}

BackendPtr GlowOnnxifiManager::createBackend(BackendIdPtr backendId) {
  assert(isValid(backendId));
  BackendPtr backend = new Backend(backendId);
  {
    std::lock_guard<std::mutex> lock(m_);
    auto res = backends_.insert(backend);
    (void)res;
    assert((res.second && *res.first) && "Failed to create new Backend");
  }
  return backend;
}

EventPtr GlowOnnxifiManager::createEvent() {
  EventPtr event = new glow::onnxifi::Event();
  {
    std::lock_guard<std::mutex> lock(m_);
    auto res = events_.insert(event);
    (void)res;
    assert((res.second && *res.first) && "Failed to create new Event");
  }
  return event;
}

GraphPtr GlowOnnxifiManager::createGraph(BackendPtr backend) {
  assert(isValid(backend));
  GraphPtr graph = new glow::onnxifi::Graph(backend);
  {
    std::lock_guard<std::mutex> lock(m_);
    auto res = graphs_.insert(graph);
    (void)res;
    assert((res.second && *res.first) && "Failed to create new Graph");
  }
  return graph;
}

bool GlowOnnxifiManager::isValid(BackendIdPtr backendId) {
  std::lock_guard<std::mutex> lock(m_);
  return backendId && backendIds_.count(backendId) == 1;
}

bool GlowOnnxifiManager::isValid(BackendPtr backend) {
  std::lock_guard<std::mutex> lock(m_);
  return backend && backends_.count(backend) == 1;
}

bool GlowOnnxifiManager::isValid(EventPtr event) {
  std::lock_guard<std::mutex> lock(m_);
  return event && events_.count(event) == 1;
}

bool GlowOnnxifiManager::isValid(GraphPtr graph) {
  std::lock_guard<std::mutex> lock(m_);
  return graph && graphs_.count(graph) == 1;
}

void GlowOnnxifiManager::release(BackendIdPtr backendId) {
  assert(isValid(backendId) && "trying to release an invalid BackendId");
  {
    std::lock_guard<std::mutex> lock(m_);
    backendIds_.erase(backendId);
  }
  delete backendId;
}

void GlowOnnxifiManager::release(BackendPtr backend) {
  assert(isValid(backend) && "trying to release an invalid Backend");
  {
    std::lock_guard<std::mutex> lock(m_);
    backends_.erase(backend);
  }
  delete backend;
}

void GlowOnnxifiManager::release(EventPtr event) {
  assert(isValid(event) && "trying to release an invalid Event");
  {
    std::lock_guard<std::mutex> lock(m_);
    events_.erase(event);
  }
  delete event;
}

void GlowOnnxifiManager::release(GraphPtr graph) {
  assert(isValid(graph) && "trying to release an invalid Graph");
  {
    std::lock_guard<std::mutex> lock(m_);
    graphs_.erase(graph);
  }
  delete graph;
}
} // namespace onnxifi
} // namespace glow
