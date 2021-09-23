#!/usr/bin/env python3
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


import argparse
import glob
import os

# imagenet-process : Runs preprocessing of standard imagenet images
#                    to work with a pretrained model (e.g. resnet)
#                    through glow
# usage: python3 imagenet-process.py "images/*.JPEG" processed
import PIL.Image
import torchvision


parser = argparse.ArgumentParser(description="imagenet preprocessor")
parser.add_argument("input", metavar="input", help="glob to input images")
parser.add_argument(
    "output", metavar="output", default="./", help="directory to put output images"
)
parser.add_argument("--normalize", action="store_true")

args = parser.parse_args()

# create the output dir if necessary
try:
    os.makedirs(args.output, exist_ok=True)
except Exception as e:
    print(e)

for ifn in glob.glob(args.input):
    name, ext = os.path.splitext(ifn)
    name = os.path.basename(name)
    outputname = os.path.join(args.output, name + ".png")
    print("processing", name, "as", outputname)

    im = PIL.Image.open(ifn)
    im = im.convert("RGB")
    resize = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224)]
    )
    processed_im = resize(im)

    if args.normalize:
        transform_fn = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform_fn = torchvision.transforms.ToTensor()

    processed_im = transform_fn(processed_im)
    processed_im = processed_im.unsqueeze(0)

    torchvision.utils.save_image(processed_im, outputname)
