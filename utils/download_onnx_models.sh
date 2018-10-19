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

for modelname in resnet50 vgg19 squeezenet zfnet512 densenet121 shufflenet; do
  wget -nc https://s3.amazonaws.com/download.onnx/models/opset_6/$modelname.tar.gz
  tar -xzvf $modelname.tar.gz
done

for modelname in inception_v1 inception_v2 bvlc_alexnet; do
  wget -nc https://s3.amazonaws.com/download.onnx/models/opset_8/$modelname.tar.gz
  tar -xzvf $modelname.tar.gz
done

# shellcheck disable=SC2043
for modelname in lenet_mnist; do
  wget -nc http://fb-glow-assets.s3.amazonaws.com/models/$modelname.tar.gz
  tar -xzvf $modelname.tar.gz
done

for modelname in googlenet_v1_slim googlenet_v4_slim resnet50_tf; do
  mkdir $modelname
  wget -nc -P $modelname \
      http://fb-glow-assets.s3.amazonaws.com/models/$modelname.onnx
done
