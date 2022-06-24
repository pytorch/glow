/**
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
#ifndef GLOW_BACKENDS_EXECUTIONCONTEXT_H
#define GLOW_BACKENDS_EXECUTIONCONTEXT_H

#include "glow/ExecutionContext/TraceEvents.h"
#include "glow/Graph/PlaceholderBindings.h"

#include "llvm/ADT/STLExtras.h"

namespace glow {
namespace runtime {
class DeviceManager;
}

/// Sub-classed per backend, this holds Device specific per-function information
/// if that is necessary on that particular backend.
class DeviceBindings {
  const std::string backend_;

public:
  DeviceBindings(llvm::StringRef backend) : backend_{backend} {}
  virtual ~DeviceBindings() {}

  virtual std::unique_ptr<DeviceBindings> clone() {
    return glow::make_unique<DeviceBindings>(backend_);
  }
};

/// The runtime context for a single execution (Inferance or Training) in the
/// the Glow Execution Engine or HostManager. This class includes the mapping
/// between Input/Output Placeholders and the materialized Tensors used for this
/// run, the set of Device specific details required to execute the function,
/// and stores TraceEvents that were generated as a result of the run.
class ExecutionContext {
  std::unique_ptr<PlaceholderBindings> placeholderBindings_;
  std::unique_ptr<DeviceBindings> deviceBindings_;

  /// Pointer to DeviceManager this context is bound to, use for P2P/DRT
  /// enablement. Unused otherwise.
  runtime::DeviceManager *boundDeviceManager_{nullptr};

  /// Trace Events recorded during this run.
  std::unique_ptr<TraceContext> traceContext_;

  /// Positional bindings for external inputs/outputs
  std::vector<std::pair<Placeholder *, Tensor>> externalIOBindings_;

  /// Perf counters (optional) recorded during this run.
  void *perfData_{nullptr};

  /// Execution state ID
  int64_t stateId_{-1};

  /// Mark if this context belongs to the last node.
  bool lastNode_{true};

public:
  ExecutionContext()
      : placeholderBindings_(glow::make_unique<PlaceholderBindings>()) {}

  ExecutionContext(std::unique_ptr<PlaceholderBindings> bindings)
      : placeholderBindings_(std::move(bindings)) {}

  ExecutionContext(std::unique_ptr<PlaceholderBindings> bindings,
                   std::unique_ptr<DeviceBindings> devices)
      : placeholderBindings_(std::move(bindings)),
        deviceBindings_(std::move(devices)) {}

  /// \returns positional bindings for external inputs
  std::vector<std::pair<Placeholder *, Tensor>> &getExternalIOBindings() {
    return externalIOBindings_;
  }

  /// \returns positional bindings for external inputs
  const std::vector<std::pair<Placeholder *, Tensor>> &
  getExternalIOBindings() const {
    return externalIOBindings_;
  }

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

  /// \returns a non-owning pointer the the deviceManager this context is bound
  /// to.
  runtime::DeviceManager *getBoundDeviceManager() {
    return boundDeviceManager_;
  }

  /// \returns a non-owning pointer to the perfData which should be
  /// cast to the correct type by the caller.
  void *getPerfData() { return perfData_; }

  /// Sets the perfData pointer to storage for whatever object is used
  /// for perf data.
  void setPerfData(void *perfData) { perfData_ = perfData; }

  /// Sets which device this context is bound to. NOTE this should not be
  /// changed once set.
  void setBoundDeviceManager(runtime::DeviceManager *device) {
    DCHECK(boundDeviceManager_ == nullptr);
    boundDeviceManager_ = device;
  }

  /// Sets the DeviceBindings and \returns the existing value.
  std::unique_ptr<DeviceBindings>
  setDeviceBindings(std::unique_ptr<DeviceBindings> bindings) {
    std::swap(deviceBindings_, bindings);
    return bindings;
  }

  /// \returns a non-owning pointer to the TraceContext.
  TraceContext *getTraceContext() { return traceContext_.get(); }

  /// \returns a const non-owning pointer to the TraceContext.
  const TraceContext *getTraceContext() const { return traceContext_.get(); }

  /// Sets the TraceContext and \returns the existing value.
  std::unique_ptr<TraceContext>
  setTraceContext(std::unique_ptr<TraceContext> traceContext) {
    std::swap(traceContext_, traceContext);
    return traceContext;
  }

  /// Clones this ExecutionContext, but does not clone underlying
  /// Tensors or the TraceContext or PerfData.
  ExecutionContext clone() {
    if (deviceBindings_) {
      return ExecutionContext(
          glow::make_unique<PlaceholderBindings>(placeholderBindings_->clone()),
          deviceBindings_->clone());
    } else {
      return ExecutionContext(glow::make_unique<PlaceholderBindings>(
          placeholderBindings_->clone()));
    }
  }

  /// A helper function to create a scoped TraceEvent builder.
  /// If there is no TraceContext, this will still create an object, but it will
  /// do nothing.
  ScopedTraceBlock scopedEvent(llvm::StringRef name, TraceLevel level) {
    return ScopedTraceBlock(getTraceContext(), level, name);
  }

  /// A helper function to log a TraceEvent at the current time, if there is a
  /// TraceContext available.
  void logTraceEvent(llvm::StringRef name, TraceLevel level,
                     char type = TraceEvent::InstantType,
                     std::map<std::string, std::string> args = {}) {
    TraceContext *traceContext = getTraceContext();
    if (traceContext) {
      traceContext->logTraceEvent(name, level, type, std::move(args));
    }
  }

  void setStateId(int64_t stateId) { stateId_ = stateId; }

  int64_t getStateId() const { return stateId_; }

  bool isLastNode() const { return lastNode_; }

  void setLastNode(bool isLastNode) { lastNode_ = isLastNode; }
};

} // namespace glow

#endif // GLOW_BACKENDS_EXECUTIONCONTEXT_H
