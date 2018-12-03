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

namespace glow {
namespace onnxifi {
GlowOnnxManager &GlowOnnxManager::get() {
  static GlowOnnxManager manager;
  return manager;
}

void GlowOnnxManager::addBackendId(BackendIdPtr backendId) {
  assert(!isValid(backendId));
  backendIds_.insert(backendId);
  assert(isValid(backendId));
}

BackendPtr GlowOnnxManager::createBackend(BackendIdPtr backendId) {
  assert(isValid(backendId));
  BackendPtr backend = new Backend(backendId);
  backends_.insert(backend);
  assert(isValid(backend));
  return backend;
}

EventPtr GlowOnnxManager::createEvent() {
  EventPtr event = new glow::onnxifi::Event();
  events_.insert(event);
  assert(isValid(event));
  return event;
}

GraphPtr GlowOnnxManager::createGraph(BackendPtr backend) {
  assert(isValid(backend));
  GraphPtr graph = new glow::onnxifi::Graph(backend);
  graphs_.insert(graph);
  assert(isValid(graph));
  return graph;
}

bool GlowOnnxManager::isValid(BackendIdPtr backendId) {
  return backendId && backendIds_.count(backendId) == 1;
}

bool GlowOnnxManager::isValid(BackendPtr backend) {
  return backend && backends_.count(backend) == 1;
}

bool GlowOnnxManager::isValid(EventPtr event) {
  return event && events_.count(event) == 1;
}

bool GlowOnnxManager::isValid(GraphPtr graph) {
  return graph && graphs_.count(graph) == 1;
}

void GlowOnnxManager::release(BackendIdPtr backendId) {
  assert(isValid(backendId) && "trying to release an invalid BackendId");
  backendIds_.erase(backendId);
  delete backendId;
}

void GlowOnnxManager::release(BackendPtr backend) {
  assert(isValid(backend) && "trying to release an invalid Backend");
  backends_.erase(backend);
  delete backend;
}

void GlowOnnxManager::release(EventPtr event) {
  assert(isValid(event) && "trying to release an invalid Event");
  events_.erase(event);
  delete event;
}

void GlowOnnxManager::release(GraphPtr graph) {
  assert(isValid(graph) && "trying to release an invalid Graph");
  graphs_.erase(graph);
  delete graph;
}
} // namespace onnxifi
} // namespace glow
