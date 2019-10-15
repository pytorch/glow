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

/**
 * Contributed by Xperi Corporation on August 13, 2019
 */

#ifndef X_PERF_MONITOR_H
#define X_PERF_MONITOR_H

#ifdef ENABLE_PERF_MONITORING

#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

/// Performance monitor statistics
struct PerfStatistics {
  /// Number of CPU cycles it took to run inference
  long long numCPUCycles;
  /// Number of cases processed (i.e. batch size)
  size_t numCases;
  /// The size of constant weights
  size_t constWeightsSize;
};

/// The performance data
struct PerfData {
  /// Performance statistics
  struct PerfStatistics ps;
  /// Performance event attributes (which performance events to monitor)
  struct perf_event_attr pe;
  /// Whether performance should be monitored
  int doPerfMonitoring;
  /// Performance event reader file descriptor
  int fd;
};

/// Initialize performance data \p pd.
int initPerfMonitoring(struct PerfData *pd);

/// Stop performance monitoring of events specified in \p pd
int stopPerfMonitoring(struct PerfData *pd);

/// Pause performance monitoring of events specified in \p pd
int pausePerfMonitoring(struct PerfData *pd);

/// Resume performance monitoring of events specified in \p pd
int resumePerfMonitoring(struct PerfData *pd);

/// Reset performance statistics specified in \p pd
int resetPerfStatistics(struct PerfData *pd);

/// Read performance statistics from the file specified by the file descriptor
/// in \pd
int readPerfStatistics(struct PerfData *pd);

#endif // ENABLE_PERF_MONITORING
#endif // X_PERF_MONITOR_H
