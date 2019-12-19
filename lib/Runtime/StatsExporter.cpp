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

#include "glow/Runtime/StatsExporter.h"

#include <vector>

namespace glow {

void StatsExporterRegistry::registerStatsExporter(StatsExporter *exporter) {
  exporters_.push_back(exporter);
}

void StatsExporterRegistry::revokeStatsExporter(StatsExporter *exporter) {
  exporters_.erase(std::remove(exporters_.begin(), exporters_.end(), exporter),
                   exporters_.end());
}

void StatsExporterRegistry::addTimeSeriesValue(llvm::StringRef key,
                                               double value) {
  for (auto const &exporter : exporters_) {
    exporter->addTimeSeriesValue(key, value);
  }
}

void StatsExporterRegistry::setCounter(llvm::StringRef key, int64_t value) {
  for (auto const &exporter : exporters_) {
    exporter->setCounter(key, value);
  }
}

void StatsExporterRegistry::incrementCounter(llvm::StringRef key,
                                             int64_t value) {
  for (auto const &exporter : exporters_) {
    exporter->incrementCounter(key, value);
  }
}

std::shared_ptr<StatsExporterRegistry> StatsExporterRegistry::Stats() {
  static auto stats = std::make_shared<StatsExporterRegistry>();
  return stats;
}

} // namespace glow
