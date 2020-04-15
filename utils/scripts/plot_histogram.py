#!/usr/bin/env python

# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import argparse
import numpy as np
import yaml
import os

# Command line options.
parser = argparse.ArgumentParser(
    usage='Helper script to print the histogram from a Glow YAML profile.')
parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                    help='Profile YAML file path.')
parser.add_argument('-n', '--name', dest='name', required=True, type=str,
                    help='Node value name to plot.')
parser.add_argument('-l', '--log-scale', dest='log_scale', required=False, default=False, action='store_true',
                    help='Plot the histogram on a logarithmic scale (base 10).')
args = parser.parse_args()

# Get arguments.
profile = args.file
name = args.name
log_scale = args.log_scale

# Verify profile exists.
if not os.path.isfile(profile):
    print('File "%s" not found!' % profile)
    exit(1)

# Read YAML data.
print('Reading file "%s" ...' % profile)
data = None
with open(profile, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as err:
        print(err)

# Search YAML entry for node value.
print('Searching node value name "%s" ...' % name)
entry = None
for item in data:
    if item['nodeOutputName'] == name:
        entry = item
if not entry:
    print('Node value "%s" not found!' % name)
    exit(1)

# Extract data.
hist_min = entry['min']
hist_max = entry['max']
histogram = np.array(entry['histogram'])
num_bins = len(histogram)
bin_width = (hist_max - hist_min) / num_bins
bin_centers = [(hist_min + idx * bin_width + bin_width/2)
               for idx in range(num_bins)]
if log_scale:
    histogram = np.log10(histogram)
    histogram = np.maximum(histogram, np.zeros(histogram.shape))

# Plot histogram.
fig = plt.figure()
plt.plot(bin_centers, histogram)
plt.bar(bin_centers, histogram, bin_width)
fig.suptitle('Histogram for "%s" with range [%f, %f]' % (
    name, hist_min, hist_max))
plt.grid()
plt.xlabel('Range')
plt.ylabel('Bins [%s]' % ('Log Scale' if log_scale else 'Linear Scale'))
plt.show()
