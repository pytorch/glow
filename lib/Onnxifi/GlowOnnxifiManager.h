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
#ifndef GLOW_ONNXIFI_GLOWONNXIFIMANAGER_H
#define GLOW_ONNXIFI_GLOWONNXIFIMANAGER_H

#include "Base.h"

#include <mutex>
#include <unordered_set>

namespace glow {
namespace onnxifi {
/// Singleton class for creating and destroying objects for the ONNXIFI
/// interface. GlowOnnxifiManager tracks objects it has created and can be used
/// to check if an ONNXIFI interface object was created by glow or not.
class GlowOnnxifiManager final {
public:
  /// Get a reference to the GlowOnnxifiManager singleton. There should only
  /// ever be one GlowOnnxifiManager.
  static GlowOnnxifiManager &get();

  /// Disallow copying and moving GlowOnnxifiManager to help enforce singleton
  /// pattern.
  GlowOnnxifiManager(const GlowOnnxifiManager &) = delete;
  GlowOnnxifiManager(GlowOnnxifiManager &&) = delete;
  GlowOnnxifiManager &operator=(const GlowOnnxifiManager &) = delete;
  GlowOnnxifiManager &operator=(GlowOnnxifiManager &&) = delete;

  /// Add a new glow BackendId \p backendId to the set of valid BackendIds.
  /// GlowOnnxifiManager then owns this BackendId and is responsible for
  /// deallocating when it is released. Can be called safely by multiple threads
  /// concurrently.
  void addBackendId(BackendIdPtr backendId);

  /// Create a new glow Backend associated with \p backendId.
  /// Can be called safely by multiple threads concurrently.
  BackendPtr createBackend(BackendIdPtr backendId);

  /// Create a new glow Event.
  /// Can be called safely by multiple threads concurrently.
  EventPtr createEvent();

  /// Create a new glow Graph associated with \p backend.
  /// Can be called safely by multiple threads concurrently.
  GraphPtr createGraph(BackendPtr backend);

  /// Check if \p backendId is a BackendId created and managed by glow.
  /// Can be called safely by multiple threads concurrently.
  bool isValid(BackendIdPtr backendId);

  /// Check if \p backend is a Backend created and managed by glow.
  /// Can be called safely by multiple threads concurrently.
  bool isValid(BackendPtr backend);

  /// Check if \p event is a Event created and managed by glow.
  /// Can be called safely by multiple threads concurrently.
  bool isValid(EventPtr event);

  /// Check if \p graph is a Graph created and managed by glow.
  /// Can be called safely by multiple threads concurrently.
  bool isValid(GraphPtr graph);

  /// Free \p backendId.
  /// Can be called safely by multiple threads concurrently.
  void release(BackendIdPtr backendId);

  /// Free \p backend.
  /// Can be called safely by multiple threads concurrently.
  void release(BackendPtr backend);

  /// Free \p event.
  /// Can be called safely by multiple threads concurrently.
  void release(EventPtr event);

  /// Free \p graph.
  /// Can be called safely by multiple threads concurrently.
  void release(GraphPtr graph);

private:
  GlowOnnxifiManager() = default;

  /// The set of all valid glow BackendIds.
  std::unordered_set<BackendIdPtr> backendIds_;

  /// The set of all valid glow Backends.
  std::unordered_set<BackendPtr> backends_;

  /// The set of all valid glow Events.
  std::unordered_set<EventPtr> events_;

  /// The set of all valid glow Graphs.
  std::unordered_set<GraphPtr> graphs_;

  /// Mutex that protects all members of GlowOnnxifiManager.
  /// TODO: can use one mutex per set if performance becomes an issue.
  mutable std::mutex m_;
};

} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_GLOWONNXIFIMANAGER_H
