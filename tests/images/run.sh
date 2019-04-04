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

imagenetIdxValues="281,207,340"
imagenetIdxArray=(281 207 340)
googlenetV1IdxArray=(281 222 340)
googlenetV4IdxArray=(281 207 340)
mnistIdxValues="0,1,2,3,4,5,6,7,8,9"

# Accumulate errors
num_errors=0

function runSqueezenetModel {
  local indices=''
  if [ "$#" -ge 2 ]
  then
    indices="-idxs=$2"
  fi
  local suffix=''
  local model=''
  if [ "$#" -ge 3 ]
  then
    suffix="$3"
    model="/model.onnx"
  fi
  ./bin/image-classifier $1 $indices -image-mode=neg128to127 -m=squeezenet$model -model-input-name=data$suffix
  num_errors=$(($num_errors + $?))
}

# Different command line configurations, testing the small model (squeezenet)
runSqueezenetModel "tests/images/imagenet/*.png"
runSqueezenetModel "tests/images/imagenet/*.png" ${imagenetIdxValues}
i=0
for png_filename in tests/images/imagenet/*.png; do
  runSqueezenetModel "$png_filename" ${imagenetIdxArray[$i]} "_0"
  i=$(($i + 1))
done
runSqueezenetModel "tests/images/imagenet/zebra_340.png" ${imagenetIdxArray[2]} "_0"


# Batch processing
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
#Inference test for TF ONNX models
i=0
for png_filename in tests/images/imagenet/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${googlenetV1IdxArray[$i]} -image-mode=0to1 -m=googlenet_v1_slim/googlenet_v1_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done
i=0
for png_filename in tests/images/imagenet_299/*.png; do
  ./bin/image-classifier "$png_filename" -idxs=${googlenetV4IdxArray[$i]} -image-mode=0to1 -m=googlenet_v4_slim/googlenet_v4_slim.onnx -model-input-name=input:0 -image-layout=NHWC -label-offset=1 "$@"
  num_errors=$(($num_errors + $?))
  i=$(($i + 1))
done

# Quantized Resnet50 Caffe2 model test
./bin/image-classifier tests/images/imagenet/*.png -idxs=${imagenetIdxValues} -image-mode=0to1 -m=quant_resnet50 -model-input-name=gpu_0/data_0 -use-imagenet-normalization "$@"
num_errors=$(($num_errors + $?))

if [ $num_errors -gt 0 ]
then
exit 1
fi
