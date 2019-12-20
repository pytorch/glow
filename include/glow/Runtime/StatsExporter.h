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
#ifndef GLOW_RUNTIME_STATSEXPORTER_H
#define GLOW_RUNTIME_STATSEXPORTER_H

#include "llvm/ADT/StringRef.h"

#include <vector>

namespace glow {

/// Interface for exporting runtime statistics.  The base implementation
/// delegates to any subclass registered via `registerStatsExporter`.
class StatsExporter {
public:
  /// Dtor.
  virtual ~StatsExporter() = default;

  /// Add value to a time series.  May be called concurrently.
  virtual void addTimeSeriesValue(llvm::StringRef key, double value) = 0;

  /// Increment a counter.  May be called concurrently.
  virtual void incrementCounter(llvm::StringRef key, int64_t value = 1) = 0;

  /// Set a counter.  May be called concurrently.
  virtual void setCounter(llvm::StringRef key, int64_t value) = 0;
};

/// Registry of StatsExporters.
class StatsExporterRegistry final {
public:
  /// Add value to a time series for all registered StatsExporters.
  void addTimeSeriesValue(llvm::StringRef key, double value);

  /// Increment a counter for all registered StatsExporters.
  void incrementCounter(llvm::StringRef key, int64_t value = 1);

  /// Set a counter for all registered StatsExporters.
  void setCounter(llvm::StringRef key, int64_t value);

  /// Register a StatsExporter.
  void registerStatsExporter(StatsExporter *exporter);

  /// Revoke a StatsExporter.
  void revokeStatsExporter(StatsExporter *exporter);

  /// Static singleton StatsExporter.
  static std::shared_ptr<StatsExporterRegistry> Stats();

private:
  /// Registered StatsExporters.
  std::vector<StatsExporter *> exporters_;
};

} // namespace glow

#endif // GLOW_RUNTIME_STATSEXPORTER_H
