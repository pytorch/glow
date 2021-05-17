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

#include "glow/Runtime/TraceExporter.h"

#include <vector>

namespace glow {

void TraceExporterRegistry::registerTraceExporter(TraceExporter *exporter) {
  /// This function can be called in static init, hence do not use
  /// glog here as it may not have been initialized yet.
  std::lock_guard<std::mutex> g(mutex_);
  exporters_.push_back(exporter);
}

void TraceExporterRegistry::revokeTraceExporter(TraceExporter *exporter) {
  std::lock_guard<std::mutex> g(mutex_);
  exporters_.erase(std::remove(exporters_.begin(), exporters_.end(), exporter),
                   exporters_.end());
}

bool TraceExporterRegistry::shouldTrace() {
  bool should = false;
  DLOG(INFO) << "shouldTrace(): total exporter count = " << exporters_.size();
  for (auto const &exporter : exporters_) {
    should |= exporter->shouldTrace();
  }
  return should;
}

void TraceExporterRegistry::exportTrace(TraceContext *tcontext) {
  DLOG(INFO) << "exportTrace(): total exporter count = " << exporters_.size();
  if (!tcontext) {
    return;
  }
  for (auto const &exporter : exporters_) {
    exporter->exportTrace(tcontext);
  }
}

std::shared_ptr<TraceExporterRegistry> TraceExporterRegistry::getInstance() {
  static auto texp = std::make_shared<TraceExporterRegistry>();
  return texp;
}

} // namespace glow
