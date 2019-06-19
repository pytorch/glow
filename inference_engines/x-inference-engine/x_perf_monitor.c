#include <sys/ioctl.h>
#include <string.h>

#include "x_perf_monitor.h"

int
init_perf_monitoring(struct PerfData *pd)
{
    int ret;

    memset(pd, 0x0, sizeof(struct PerfData));

    pd->pe.type = PERF_TYPE_HARDWARE;
    pd->pe.size = sizeof(struct perf_event_attr);
    pd->pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pd->pe.disabled = 1;
    pd->pe.exclude_kernel = 1;
    pd->pe.exclude_hv = 1;
    pd->pe.exclude_idle = 1;

    ret = syscall(__NR_perf_event_open, &(pd->pe), 0, -1, -1, 0);
    pd->fd = ret;

    return ret;
}

int
stop_perf_monitoring(struct PerfData *pd)
{
    if (pd->fd >= 0)
        close(pd->fd);

    return 0;
}

int
pause_perf_monitoring(struct PerfData *pd)
{
    if (pd->fd >= 0)
        ioctl(pd->fd, PERF_EVENT_IOC_DISABLE, 0);
    else
        return pd->fd;

    return 0;
}

int
resume_perf_monitoring(struct PerfData *pd)
{
    if (pd->fd >= 0)
        ioctl(pd->fd, PERF_EVENT_IOC_ENABLE, 0);
    else
        return pd->fd;

    return 0;
}

int
reset_perf_statistics(struct PerfData *pd)
{
    if (pd->fd >= 0) {
        pd->ps.num_cpu_cycles = 0;
        ioctl(pd->fd, PERF_EVENT_IOC_RESET, 0);
    } else {
        return pd->fd;
    }

    return 0;
}

int
read_perf_statistics(struct PerfData *pd)
{
    int ret = -1;

    if (pd->fd >= 0)
        ret = read(pd->fd, &(pd->ps.num_cpu_cycles), sizeof(pd->ps.num_cpu_cycles));

    return ret;
}
