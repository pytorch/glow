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
set -e

imagenetFiles=("cat_285" "dog_207" "zebra_340")
imagenetIdxValue=(281 207 340)
inceptionIdxValue=(173 124 79)

imagenet299Files=("cat_281_299" "dog_207_299" "zebra_340_299")
googlenetV1IdxValue=(281 222 340)
googlenetV4IdxValue=(281 207 340)

mnistFiles=("0_1009" "1_1008" "2_1065" "3_1020" "4_1059" "5_1087" "6_1099" "7_1055" "8_1026" "9_1088")
mnistIdxValue=(0 1 2 3 4 5 6 7 8 9)

# Default idx is a valid input
./bin/image-classifier tests/images/imagenet/${imagenetFiles[0]}.png -use-imagenet-normalization -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data "$@"

for i in ${!imagenetFiles[@]}; do
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -use-imagenet-normalization -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=neg128to127 -m=vgg19 -model-input-name=data "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=neg128to127 -m=squeezenet -model-input-name=data "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=0to255 -m=zfnet512 -model-input-name=gpu_0/data "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=0to1 -m=densenet121 -model-input-name=data -compute-softmax "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=0to1 -m=shufflenet -model-input-name=gpu_0/data "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${inceptionIdxValue[$i]} -image-mode=0to255 -m=inception_v2 -model-input-name=data "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=0to255 -m=bvlc_alexnet -model-input-name=data "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -use-imagenet-normalization -image-mode=0to1 -m=resnet50/model.onnx -model-input-name=gpu_0/data_0 "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=neg128to127 -m=vgg19/model.onnx -model-input-name=data_0 "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=neg128to127 -m=squeezenet/model.onnx -model-input-name=data_0 "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=0to255 -m=zfnet512/model.onnx -model-input-name=gpu_0/data_0 "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=0to1 -m=densenet121/model.onnx -model-input-name=data_0 -compute-softmax "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=0to1 -m=shufflenet/model.onnx -model-input-name=gpu_0/data_0 "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=0to255 -m=inception_v1/model.onnx -model-input-name=data_0 "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]} -image-mode=0to255 -m=bvlc_alexnet/model.onnx -model-input-name=data_0 "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${googlenetV1IdxValue[$i]} -image-mode=0to1 -m=googlenet_v1_slim/googlenet_v1_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
	./bin/image-classifier tests/images/imagenet/${imagenetFiles[$i]}.png -idx=${imagenetIdxValue[$i]}  -image-mode=0to1 -m=quant_resnet50 -model-input-name=gpu_0/data_0 -use-imagenet-normalization "$@"
done

for i in ${!imagenet299Files[@]}; do
	./bin/image-classifier tests/images/imagenet_299/${imagenet299Files[$i]}.png -idx=${googlenetV4IdxValue[$i]} -image-mode=0to1 -m=googlenet_v4_slim/googlenet_v4_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
done

for i in ${!mnistFiles[@]}; do
	./bin/image-classifier tests/images/mnist/${mnistFiles[$i]}.png -idx=${mnistIdxValue[$i]} -image-mode=0to1 -m=lenet_mnist -model-input-name=data "$@"
	./bin/image-classifier tests/images/mnist/${mnistFiles[$i]}.png -idx=${mnistIdxValue[$i]} -image-mode=0to1 -m=mnist.onnx -model-input-name=data_0 "$@"
done
