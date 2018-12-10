#!/usr/bin/env python
"""
Displays OpenCL device numbers and additional information.
This is useful if you are wondering what to pass to the `-device` and
`-platform` parameters of the opencl backend.

Prerequisites: `pip install pyopencl` (and probably `brew install pybind11).
"""
from __future__ import print_function
import pyopencl

def format(indent, key, value):
    print("%s%-17s %s" % ("  "*indent, key, value))

for p, platform in enumerate(pyopencl.get_platforms()):
    format(0, "platform %s" % p, platform.name)
    for key in (
        'vendor',
        'profile',
        'version',
        'extensions'):
        format(0, key, getattr(platform, key))
    for d, device in enumerate(platform.get_devices()):
        context = pyopencl.Context([device])

        print("")
        format(1, "device %s" %d, device.name)
        format(2, "type", pyopencl.device_type.to_string(device.type))
        format(2, "driver_version", device.driver_version)
        format(2, "memory", "%s GB" % (device.global_mem_size / 2**30))
        format(2, "max_compute_units", device.max_compute_units)
