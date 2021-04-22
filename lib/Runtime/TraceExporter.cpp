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
  exporters_.push_back(exporter);
}

void TraceExporterRegistry::revokeTraceExporter(TraceExporter *exporter) {
  exporters_.erase(std::remove(exporters_.begin(), exporters_.end(), exporter),
                   exporters_.end());
}

bool TraceExporterRegistry::shouldTrace() {
  bool should = false;
  for (auto const &exporter : exporters_) {
    should |= exporter->shouldTrace();
  }
  return should;
}

void TraceExporterRegistry::exportTrace(TraceContext *tcontext) {
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
