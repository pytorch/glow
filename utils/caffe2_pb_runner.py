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

# Load a pre-trained Caffe2 image classifier and run it on an image.

import argparse
import collections
import os
import time

import numpy as np
import skimage.io
from caffe2.python import workspace


print("Required modules imported.")

cmd_line_parser = argparse.ArgumentParser(
    description="Run Caffe2 using provided models and inputs."
)
cmd_line_parser.add_argument(
    "--image", "-i", required=True, help="Image to be processed by the neural network"
)
cmd_line_parser.add_argument(
    "--directory",
    "-d",
    required=True,
    help="Directory containing the network structure "
    "<predict_net.pb> and weight <init_net.pb> files. "
    "The model name is assumed to be the directory "
    "name, and should correspond to a model from the  "
    "model_props (e.g. 'resnet50', 'lenet_mnist', "
    "etc.). If the directory name is not the model "
    "name, use --model-name (-m) to specify the name "
    "of the supported model to use.",
)
cmd_line_parser.add_argument(
    "--model-name", "-m", required=False, help="Name of the model to be used"
)
cmd_line_parser.add_argument(
    "--image_mode",
    required=False,
    help="Image mode; one of '0to1', '0to256', or '128to127'",
)
cmd_line_parser.add_argument("--time", action="store_true")
cmd_line_parser.add_argument("--iterations", type=int, default=1)

args = cmd_line_parser.parse_args()

# 0to256 is the default input


def mode_0to256(x):
    return x


def mode_0to1(x):
    return x / 255


def mode_128to127(x):
    return x - 128


Model = collections.namedtuple(
    "Model", "blob_name, image_mode_op, image_size, num_color_channels"
)

model_props = dict(
    densenet121=Model("data", mode_0to1, 224, 3),
    inception_v1=Model("data", mode_128to127, 224, 3),
    inception_v2=Model("data", mode_128to127, 224, 3),  # unknown
    resnet50=Model("gpu_0/data", mode_0to1, 224, 3),
    shufflenet=Model("gpu_0/data", mode_0to1, 224, 3),
    squeezenet=Model("data", mode_128to127, 224, 3),
    vgg19=Model("data", mode_128to127, 224, 3),
    zfnet512=Model("gpu_0/data", mode_0to256, 224, 3),
    lenet_mnist=Model("data", mode_0to1, 28, 1),
    resnext=Model("data", mode_0to1, 224, 3),
)

MODEL = args.model_name
if MODEL is None:
    MODEL = os.path.basename(os.path.normpath(args.directory))

if MODEL not in list(model_props.keys()):
    print(
        "Model " + MODEL + " is not supported. Specify --model-name (-m) if "
        "it is not the base name of the directory containing pb files."
    )
    exit(1)

MODEL_ROOT = args.directory
IMAGE_LOCATION = args.image
img = skimage.img_as_ubyte(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)

image_shape = np.array(img).shape

print("Initial img shape: " + str(image_shape))

if img.shape[0] != img.shape[1] or img.shape[0] != model_props[MODEL].image_size:
    print("Invalid image dimensions for model.")
    exit(2)

num_dims = len(np.array(img).shape)
if num_dims != 3:
    img = np.expand_dims(img, axis=num_dims)

img = img[:, :, : model_props[MODEL].num_color_channels]

# Create a zero initiated image.
transposed_image = np.zeros(
    (
        1,
        model_props[MODEL].num_color_channels,
        model_props[MODEL].image_size,
        model_props[MODEL].image_size,
    )
).astype(np.float32)
for w in range(0, model_props[MODEL].image_size):
    for h in range(0, model_props[MODEL].image_size):
        for c in range(0, model_props[MODEL].num_color_channels):
            # WHC -> CWH, RGB -> BGR
            transposed_image[0][model_props[MODEL].num_color_channels - c - 1][w][h] = (
                model_props[MODEL].image_mode_op(img[w][h][c])
            )

final_image = transposed_image

print("Shape of final_image: " + str(np.array(final_image).shape))

with open(MODEL_ROOT + "/init_net.pb", "rb") as f:
    init_net = f.read()
with open(MODEL_ROOT + "/predict_net.pb", "rb") as f:
    predict_net = f.read()

workspace.ResetWorkspace()

blob_name = model_props[MODEL].blob_name
workspace.FeedBlob(blob_name, final_image)

print("The blobs in the workspace after FeedBlob: {}".format(workspace.Blobs()))

# Create a predictor using the loaded model.
p = workspace.Predictor(init_net, predict_net)

start = time.time()
for i in range(0, args.iterations):
    results = p.run([final_image])
end = time.time()
if args.time:
    print(
        "Wall time per iteration (s): {:0.4f}".format((end - start) / args.iterations)
    )

max_idx = np.argmax(results[0][0])
sum_probability = sum(results[0][0])

print("Max index is {}".format(max_idx))
print(
    "Predicted class at index {} with probability {}".format(
        max_idx, results[0][0][max_idx]
    )
)
print("Number of classes {}".format(len(results[0][0])))
print("Sum of probabilities is {}".format(sum_probability))
