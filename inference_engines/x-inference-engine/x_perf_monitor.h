/** Copyright 2019 Xperi Corporation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License”); 
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef X_PERF_MONITOR_H
#define X_PERF_MONITOR_H

#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

struct PerfStatistics
{
    long long num_cpu_cycles;
    size_t num_cases;
    size_t const_weights_size;
};

struct PerfData
{
    struct PerfStatistics ps;
    struct perf_event_attr pe;
    int do_perf_monitoring;
    int fd;
};

int init_perf_monitoring(struct PerfData *pd);
int stop_perf_monitoring(struct PerfData *pd);
int pause_perf_monitoring(struct PerfData *pd);
int resume_perf_monitoring(struct PerfData *pd);
int reset_perf_statistics(struct PerfData *pd);
int read_perf_statistics(struct PerfData *pd);

#endif