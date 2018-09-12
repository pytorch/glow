#!/usr/bin/env bash

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
