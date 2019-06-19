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
    long long num_cases;
};

struct PerfData
{
    struct PerfStatistics ps;
    struct perf_event_attr pe;
    int fd;
};

int init_perf_monitoring(struct PerfData *pd);
int stop_perf_monitoring(struct PerfData *pd);
int pause_perf_monitoring(struct PerfData *pd);
int resume_perf_monitoring(struct PerfData *pd);
int reset_perf_statistics(struct PerfData *pd);
int read_perf_statistics(struct PerfData *pd);

#endif