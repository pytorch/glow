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

imagenetFiles=("cat_285" "dog_207" "zebra_340")
imagenetIdxArray=(281 207 340)
imagenetIdxValues="281,207,340"

imagenet299Files=("cat_281_299" "dog_207_299" "zebra_340_299")
googlenetV1IdxArray=(281 222 340)
googlenetV1IdxValues="281,222,340"

googlenetV4IdxArray=(281 207 340)
googlenetV4IdxValue="281,207,340"

mnistFiles=("0_1009" "1_1008" "2_1065" "3_1020" "4_1059" "5_1087" "6_1099" "7_1055" "8_1026" "9_1088")
mnistIdxArray=(0 1 2 3 4 5 6 7 8 9)
mnistIdxValues="0,1,2,3,4,5,6,7,8,9"

# Accumulate errors
num_errors=0

# Batch processing, no indices provided
./bin/image-classifier tests/images/imagenet/*.png -use-imagenet-normalization -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -image-mode=neg128to127 -m=vgg19 -model-input-name=data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -image-mode=neg128to127 -m=squeezenet -model-input-name=data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to255 -m=zfnet512 -model-input-name=gpu_0/data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to1 -m=densenet121 -model-input-name=data -compute-softmax "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to1 -m=shufflenet -model-input-name=gpu_0/data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/mnist/*.png -image-mode=0to1 -m=lenet_mnist -model-input-name=data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to255 -m=inception_v1 -model-input-name=data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to255 -m=bvlc_alexnet -model-input-name=data "$@"
num_errors=$(($num_errors + $?))
# Batch processing, with indices provided
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -use-imagenet-normalization -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -image-mode=neg128to127 -m=vgg19 -model-input-name=data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -image-mode=neg128to127 -m=squeezenet -model-input-name=data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -image-mode=0to255 -m=zfnet512 -model-input-name=gpu_0/data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -image-mode=0to1 -m=densenet121 -model-input-name=data -compute-softmax "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -image-mode=0to1 -m=shufflenet -model-input-name=gpu_0/data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/mnist/*.png -idxs=${mnistIdxValues} -image-mode=0to1 -m=lenet_mnist -model-input-name=data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -image-mode=0to255 -m=inception_v1 -model-input-name=data "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -image-mode=0to255 -m=bvlc_alexnet -model-input-name=data "$@"
num_errors=$(($num_errors + $?))

# Single file processing in a loop, no indices provided
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -use-imagenet-normalization -image-mode=0to1 -m=resnet50/model.onnx -model-input-name=gpu_0/data_0 "$@"
  num_errors=$(($num_errors + $?))
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=neg128to127 -m=vgg19/model.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=neg128to127 -m=squeezenet/model.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to255 -m=zfnet512/model.onnx -model-input-name=gpu_0/data_0 "$@"
  num_errors=$(($num_errors + $?))
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to1 -m=densenet121/model.onnx -model-input-name=data_0 -compute-softmax "$@"
  num_errors=$(($num_errors + $?))
done

#Single file, no index
./bin/image-classifier tests/images/imagenet/zebra_340.png -image-mode=0to1 -m=shufflenet/model.onnx -model-input-name=gpu_0/data_0 "$@"
num_errors=$(($num_errors + $?))

for png_filename in tests/images/mnist/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to1 -m=mnist.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to255 -m=inception_v1/model.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
done
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to255 -m=bvlc_alexnet/model.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
done

# Single file processing in a loop, with indices provided
i=0
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${imagenetIdxArray[$i]} -use-imagenet-normalization -image-mode=0to1 -m=resnet50/model.onnx -model-input-name=gpu_0/data_0 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done
i=0
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${imagenetIdxArray[$i]} -image-mode=neg128to127 -m=vgg19/model.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done
i=0
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${imagenetIdxArray[$i]} -image-mode=neg128to127 -m=squeezenet/model.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done
i=0
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${imagenetIdxArray[$i]} -image-mode=0to255 -m=zfnet512/model.onnx -model-input-name=gpu_0/data_0 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done
i=0
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${imagenetIdxArray[$i]} -image-mode=0to1 -m=densenet121/model.onnx -model-input-name=data_0 -compute-softmax "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done

#Single file, with index
./bin/image-classifier tests/images/imagenet/zebra_340.png -idxs=${magenetIdxArray[2]} -image-mode=0to1 -m=shufflenet/model.onnx -model-input-name=gpu_0/data_0 "$@"
num_errors=$(($num_errors + $?))

i=0
for png_filename in tests/images/mnist/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${mnistIdxArray[$i]} -image-mode=0to1 -m=mnist.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done
i=0
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${imagenetIdxArray[$i]} -image-mode=0to255 -m=inception_v1/model.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done
i=0
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${imagenetIdxArray[$i]} -image-mode=0to255 -m=bvlc_alexnet/model.onnx -model-input-name=data_0 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done

#Inference test for TF ONNX models no indices
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to1 -m=googlenet_v1_slim/googlenet_v1_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
  num_errors=$(($num_errors + $?))
done

#Inference test for TF ONNX models with indices
i=0
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${googlenetV1IdxArray[$i]} -image-mode=0to1 -m=googlenet_v1_slim/googlenet_v1_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done

for png_filename in tests/images/imagenet_299/*.png; do
  ./bin/image-classifier "$png_filename" -image-mode=0to1 -m=googlenet_v4_slim/googlenet_v4_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
  num_errors=$(($num_errors + $?))
done

i=0
for png_filename in tests/images/imagenet_299/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${googlenetV4IdxArray[$i]} -image-mode=0to1 -m=googlenet_v4_slim/googlenet_v4_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done

# Quantized Resnet50 Caffe2 model test
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to1 -m=quant_resnet50 -model-input-name=gpu_0/data_0 -use-imagenet-normalization "$@"
num_errors=$(($num_errors + $?))
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -image-mode=0to1 -m=quant_resnet50 -model-input-name=gpu_0/data_0 -use-imagenet-normalization "$@"
num_errors=$(($num_errors + $?))

if [ $num_errors -gt 0 ]
then
exit 1
fi
