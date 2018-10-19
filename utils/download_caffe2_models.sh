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

MODELS=$(cat <<EOF
densenet121
inception_v1
inception_v2
lenet_mnist
resnet50
shufflenet
squeezenet
vgg19
zfnet512
bvlc_alexnet
en2gr
EOF
)

for model in $MODELS; do
  for file in predict_net.pbtxt predict_net.pb init_net.pb; do
    wget -nc "http://fb-glow-assets.s3.amazonaws.com/models/$model/$file" -P "$model"
  done
done

for file in dst_dictionary.txt src_dictionary.txt; do
  wget -nc "http://fb-glow-assets.s3.amazonaws.com/models/en2gr/$file" -P en2gr
done
