#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
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

import sys
from PIL import Image

# This script is used to visualize memory allocations in the Glow compiler.
#
# Usage: ./visualize.py dump.txt
#
# The script will dump a sequence of bitmap files that can be combined into a
# video. Example: heap100123.bmp, heap heap100124.bmp, heap100125.bmp ...  )
#
# On mac and linux this command will generate a gif file:
#    convert -delay 10 -loop 0 *.bmp video.gif
#
# The input file should contain a list of allocation/deallocation commands.
# Allocation commands (marked with the letter 'a') report the start address and
# the size of the buffer, and the deallocation commands (marked with the letter
# 'd') report the address of the buffer. You can generate these command lists
# by inserting printf calls into the Glow memory allocator.
#
# Example input:
#    a 348864 20000
#    a 368896 20000
#    a 388928 20000
#    a 408960 200000
#    d 388928
#    d 368896
#    d 348864


content = open(sys.argv[1]).read()
lines = content.split('\n')

canvas_size = 512
pixelsize = 8

img = Image.new("RGB", (canvas_size, canvas_size), "black")
pixels = img.load()


# Use this number to assign file names to frames in the video.
filename_counter = 10000000

# Maps from address to size
sizes={}

color_index = 0
colors=[(218, 112, 214), (255, 182, 193), (250, 235, 215), (255, 250, 205),
        (210, 105, 30), (210, 180, 140), (188, 143, 143), (255, 240, 245),
        (230, 230, 250), (255, 255, 240)]

def getColor():
    global color_index
    color_index+=1
    return colors[color_index % len(colors)]

def setPixel(addr, color):
    # Don't draw out-of-bounds pixels.
    if (addr >= canvas_size * canvas_size): return
    # Only draw pixels that are aligned to the block size.
    if (addr % pixelsize != 0): return
    # Draw large pixels.
    sx = addr%canvas_size
    sy = addr/canvas_size
    sx = int(sx/pixelsize)
    sy = int(sy/pixelsize)
    for x in range(pixelsize):
        for y in range(pixelsize):
            pixels[sx*pixelsize + x, sy*pixelsize + y] = color

def saveFrame():
    global filename_counter
    filename_counter+=1
    img.save("heap" + str(filename_counter) + ".bmp")

for line in lines:
    tokens = line.split()
    if (len(tokens) < 1): break

    print(tokens)
    if (tokens[0] == 'a'):
        frm = int(tokens[1])
        sz = int(tokens[2])
        sizes[frm] = sz
        if (frm + sz >= canvas_size * canvas_size): continue
        for i in range(sz): setPixel(i + frm ,(255,255,255)) # allocate
        saveFrame()
        cc = getColor()
        for i in range(sz): setPixel(i + frm ,cc) # allocated
        saveFrame()


    if (tokens[0] == 'd'):
        frm = int(tokens[1])
        sz = sizes[frm]
        if (frm + sz >= canvas_size * canvas_size): continue
        for i in range(sz): setPixel(i + frm ,(128,0,0)) # deallocate
        saveFrame()
        for i in range(sz): setPixel(i + frm ,(15,15,15)) # previously allocated
        saveFrame()
