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
#ifndef GLOW_ONNXIFI_GLOWONNXMANAGER_H
#define GLOW_ONNXIFI_GLOWONNXMANAGER_H

#include <unordered_set>

namespace glow {
namespace onnxifi {
class BackendId;
class Backend;
class Event;
class Graph;

/// Singleton class for creating and destroying objects for the ONNX interface.
/// GlowOnnxManager tracks objects it has created and can be used to check if
/// if an ONNX interface object was created by glow or not.
class GlowOnnxManager {
public:
  /// Get a reference to the GlowOnnxManager singleton. There should only ever
  /// be one GlowOnnxManager.
  static GlowOnnxManager &get();

  // Disallow copying GlowOnnxManager to help enforce singleton pattern.
  GlowOnnxManager(const GlowOnnxManager &) = delete;
  GlowOnnxManager &operator=(const GlowOnnxManager &) = delete;

  /// Add a new glow \p backendId to the set of valid backendIds.
  void addBackendId(BackendId *backendId);

  /// Create a new glow Backend associated with \p backendId.
  Backend *createBackend(BackendId *backendId);

  /// Create a new glow Event.
  Event *createEvent();

  /// Create a new glow Graph associated with \p backend.
  Graph *createGraph(Backend *backend);

  /// Check if \p backendId is a BackendId created and managed by glow.
  bool isValid(BackendId *backendId);

  /// Check if \p backend is a Backend created and managed by glow.
  bool isValid(Backend *backend);

  /// Check if \p event is a Event created and managed by glow.
  bool isValid(Event *event);

  /// Check if \p graph is a Graph created and managed by glow.
  bool isValid(Graph *graph);

  /// Free \p backendId.
  void release(BackendId *backendId);

  /// Free \p backend.
  void release(Backend *backend);

  /// Free \p event.
  void release(Event *event);

  /// Free \p graph.
  void release(Graph *graph);

private:
  GlowOnnxManager() = default;

  /// The set of all valid glow BackendIds.
  std::unordered_set<BackendId *> backendIds_;

  /// The set of all valid glow Backends.
  std::unordered_set<Backend *> backends_;

  /// The set of all valid glow Events.
  std::unordered_set<Event *> events_;

  /// The set of all valid glow Graphs.
  std::unordered_set<Graph *> graphs_;
};

} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_GLOWONNXMANAGER_H
