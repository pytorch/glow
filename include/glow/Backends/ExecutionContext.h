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
#ifndef GLOW_BACKENDS_EXECUTIONCONTEXT_H
#define GLOW_BACKENDS_EXECUTIONCONTEXT_H

#include "glow/Backends/TraceEvents.h"
#include "glow/Graph/PlaceholderBindings.h"

#include "llvm/ADT/STLExtras.h"

namespace glow {

enum class BackendKind;

/// Sub-classed per backend, this holds Device specific per-function information
/// if that is necessary on that particular backend.
class DeviceBindings {
  const glow::BackendKind backend_;

public:
  DeviceBindings(BackendKind kind) : backend_{kind} {}
  virtual ~DeviceBindings() {}

  virtual std::unique_ptr<DeviceBindings> clone() {
    return llvm::make_unique<DeviceBindings>(backend_);
  }

  BackendKind getBackendKind() { return backend_; }
};

/// The runtime context for a single execution (Inferance or Training) in the
/// the Glow Execution Engine or HostManager. This class includes the mapping
/// between Input/Output Placeholders and the materialized Tensors used for this
/// run, the set of Device specific details required to execute the function,
/// and stores TraceEvents that were generated as a result of the run.
class ExecutionContext {
  std::unique_ptr<PlaceholderBindings> placeholderBindings_;
  std::unique_ptr<DeviceBindings> deviceBindings_;

  /// Trace Events recorded during this run.
  std::vector<TraceEvent> traceEvents_;

public:
  ExecutionContext()
      : placeholderBindings_(llvm::make_unique<PlaceholderBindings>()) {}

  ExecutionContext(std::unique_ptr<PlaceholderBindings> bindings)
      : placeholderBindings_(std::move(bindings)) {}

  ExecutionContext(std::unique_ptr<PlaceholderBindings> bindings,
                   std::unique_ptr<DeviceBindings> devices)
      : placeholderBindings_(std::move(bindings)),
        deviceBindings_(std::move(devices)) {}

  /// \returns TraceEvents for the last run.
  std::vector<TraceEvent> &getTraceEvents() { return traceEvents_; }

  /// \returns a non-owning pointer to the PlaceholderBindings.
  PlaceholderBindings *getPlaceholderBindings() {
    return placeholderBindings_.get();
  }

  /// \returns a const non-owning pointer to the PlaceholderBindings.
  const PlaceholderBindings *getPlaceholderBindings() const {
    return placeholderBindings_.get();
  }

  /// \returns an owning pointer to the PlaceholderBindings.
  std::unique_ptr<PlaceholderBindings> movePlaceholderBindings() {
    return std::move(placeholderBindings_);
  }

  /// \returns a non-owning pointer to the DeviceBindings.
  DeviceBindings *getDeviceBindings() { return deviceBindings_.get(); }

  /// \returns a const non-owning pointer to the DeviceBindings.
  const DeviceBindings *getDeviceBindings() const {
    return deviceBindings_.get();
  }

  /// Sets the DeviceBindings and \returns the existing value.
  std::unique_ptr<DeviceBindings>
  setDeviceBindings(std::unique_ptr<DeviceBindings> bindings) {
    std::swap(deviceBindings_, bindings);
    return bindings;
  }

  /// Clones this ExecutionContext, but does not clone underlying Tensors.
  ExecutionContext clone() {
    if (deviceBindings_) {
      return ExecutionContext(
          llvm::make_unique<PlaceholderBindings>(placeholderBindings_->clone()),
          deviceBindings_->clone());
    } else {
      return ExecutionContext(llvm::make_unique<PlaceholderBindings>(
          placeholderBindings_->clone()));
    }
  }
};

} // namespace glow

#endif // GLOW_BACKENDS_EXECUTIONCONTEXT_H
