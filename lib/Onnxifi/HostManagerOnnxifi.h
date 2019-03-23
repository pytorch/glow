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
#ifndef GLOW_ONNXIFI_HOSTMANAGERONNXIFI_H
#define GLOW_ONNXIFI_HOSTMANAGERONNXIFI_H

#include "Base.h"

#include "glow/Runtime/HostManager/HostManager.h"

namespace glow {
namespace onnxifi {

class HostManagerBackendId : public BackendId {
public:
  /// Create Glow ONNXIFI backend identifier using HostManager with the
  /// given Glow backend \p kind, whether to use onnx or caffe2 for models
  /// (\p useOnnx)
  HostManagerBackendId(runtime::HostManager *hostManager,
                       glow::BackendKind kind, bool useOnnx)
      : BackendId(kind, useOnnx), hostManager_(hostManager) {}

  void runNetwork(const Graph *graph, std::unique_ptr<ExecutionContext> context,
                  runtime::ResultCBTy callback) override;

  onnxStatus addNetwork(Module *module) override;

  void removeNetwork(const Graph *graph) override;

  // \returns a unique_ptr to a new HostManager for the given BackendKind \p
  // kind.
  static std::unique_ptr<runtime::HostManager>
  createHostManager(glow::BackendKind kind);

private:
  runtime::HostManager *hostManager_;
};

class HostManagerGraph : public Graph {
public:
  using Graph::Graph;

  /// \returns a globally unique graph id.
  static size_t makeUniqueGraphId();

  /// Init Glow graph based on the ONNX model \p onnxModel and
  /// static trained weights \p weightDescriptors.
  onnxStatus
  initGraph(const void *onnxModel, size_t onnxModelSize, uint32_t weightCount,
            const onnxTensorDescriptorV1 *weightDescriptors) override;

  onnxStatus run(std::unique_ptr<ExecutionContext> ctx, EventPtr outputEvent,
                 std::unordered_map<Placeholder *, onnxTensorDescriptorV1>
                     phNameToOnnxTensorOutputs) override;

  const std::string &getName() const { return netName_; }

private:
  Module m_;
  std::string netName_;
};

} // namespace onnxifi
} // namespace glow

#endif // GLOW_ONNXIFI_HOSTMANAGERONNXIFI_H
