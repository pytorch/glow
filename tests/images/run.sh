#!/usr/bin/env bash

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

./bin/image-classifier tests/images/imagenet/*.png -use-imagenet-normalization -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image-mode=neg128to127 -m=vgg19 -model-input-name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image-mode=neg128to127 -m=squeezenet -model-input-name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to255 -m=zfnet512 -model-input-name=gpu_0/data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to1 -m=densenet121 -model-input-name=data -compute-softmax "$@"
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to1 -m=shufflenet -model-input-name=gpu_0/data "$@"
./bin/image-classifier tests/images/mnist/*.png -image-mode=0to1 -m=lenet_mnist -model-input-name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to255 -m=inception_v1 -model-input-name=data "$@"
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to255 -m=bvlc_alexnet -model-input-name=data "$@"
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -use-imagenet-normalization -image-mode=0to1 -m=resnet50/model.onnx -model-input-name=gpu_0/data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=neg128to127 -m=vgg19/model.onnx -model-input-name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=neg128to127 -m=squeezenet/model.onnx -model-input-name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to255 -m=zfnet512/model.onnx -model-input-name=gpu_0/data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to1 -m=densenet121/model.onnx -model-input-name=data_0 -compute-softmax "$@"
done
./bin/image-classifier tests/images/imagenet/zebra_340.png -image-mode=0to1 -m=shufflenet/model.onnx -model-input-name=gpu_0/data_0 "$@"
for png_filename in tests/images/mnist/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to1 -m=mnist.onnx -model-input-name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to255 -m=inception_v1/model.onnx -model-input-name=data_0 "$@"
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to255 -m=bvlc_alexnet/model.onnx -model-input-name=data_0 "$@"
done
#Inference test for TF ONNX models
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to1 -m=googlenet_v1_slim/googlenet_v1_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
done
for png_filename in tests/images/imagenet_299/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to1 -m=googlenet_v4_slim/googlenet_v4_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
done
#Quantized Resnet50 Caffe2 model test
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to1 -m=quant_resnet50 -model-input-name=gpu_0/data_0 -use-imagenet-normalization "$@"
