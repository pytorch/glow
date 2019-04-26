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
#include "HostManagerOnnxifi.h"
#include "InlineOnnxifi.h"

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

  /// Create a new glow BackendId for BackendKind \p kind using onnx graphs if
  /// \p useOnnx and caffe2 graphs otherwise. If \p forQuantization is true then
  /// a BackendId will be created otherwise a HostManagerBackendId will be
  /// be created.
  /// Can be called safely by multiple threads concurrently.
  BackendIdPtr createBackendId(glow::BackendKind kind, bool useOnnx,
                               bool forQuantization = false);

  /// Create a new glow Backend associated with \p backendId.
  /// Can be called safely by multiple threads concurrently.
  BackendPtr createBackend(BackendIdPtr backendId);

  /// Create a new glow Event.
  /// Can be called safely by multiple threads concurrently.
  EventPtr createEvent();

  /// Create a new glow Graph associated with \p backend.
  /// Can be called safely by multiple threads concurrently.
  GraphPtr createGraph(
      BackendPtr backend,
      OnnxifiQuantizationStep quantizationStep = OnnxifiQuantizationStep::None);

  /// Check if \p backendId is a BackendId created and managed by glow.
  /// Can be called safely by multiple threads concurrently.
  bool isValid(BackendIdPtr backendId) const;

  /// Check if \p backend is a Backend created and managed by glow.
  /// Can be called safely by multiple threads concurrently.
  bool isValid(BackendPtr backend) const;

  /// Check if \p event is a Event created and managed by glow.
  /// Can be called safely by multiple threads concurrently.
  bool isValid(EventPtr event) const;

  /// Check if \p graph is a Graph created and managed by glow.
  /// Can be called safely by multiple threads concurrently.
  bool isValid(GraphPtr graph) const;

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

  /// Create a new HostManager managing backends of kind \p backendKind or get
  /// an existing HostManager for the backendKind if one exists.
  /// NOTE: This method is not thread safe, the caller should be holding the
  /// mutex m_ when calling it!
  std::shared_ptr<runtime::HostManager>
  getOrCreateHostManager(BackendKind backendKind);

  /// The set of all valid glow BackendIds.
  std::unordered_set<BackendIdPtr> backendIds_;

  /// The set of all valid glow Backends.
  std::unordered_set<BackendPtr> backends_;

  /// The set of all valid glow Events.
  std::unordered_set<EventPtr> events_;

  /// The set of all valid glow Graphs.
  std::unordered_set<GraphPtr> graphs_;

  /// Map from BackendKind to HostManager managing devices of that kind that is
  /// shared by all BackendIds using that HostManager. HostManager is stored as
  /// weak_ptr here so that it will be destructed when the last BackendId using
  /// it is destroyed not when this singleton is destroyed.
  std::map<BackendKind, std::weak_ptr<runtime::HostManager>> hostManagers_;

  /// Mutex that protects all members of GlowOnnxifiManager.
  /// TODO: can use one mutex per set if performance becomes an issue.
  mutable std::mutex m_;
};

} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_GLOWONNXIFIMANAGER_H
