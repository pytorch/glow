## GLOW Model loader precision configuration

This document describes the mixed precision feature which enables the network
to be able to run with the combination of operators in float, float16_t and int8
precision.

### Overview

Glow has following two options along with quantization to set precision of operators

`convert-to-fp16` - Allows running all floating point operations in fp16.
`keep-original-precision-for-nodes` - Allows running certain node kinds not to be
quantized and run in the orginal precision.

Note that the above two options are node kind based. In order to run specific instances
of operators in fp16 precision `-node-precision-info` option can be used to indicate
execution of specific nodes in fp16. The nodes to run in fp16 are specified by the name of
the first output in a yaml file.

`-node-precision-info` option can be passed along with `-load-profile` option  of building
quantized models. In such case operators not mentioned in `-node-precision-info` will run
in quantized precision (If supported by backend)

### Design details

#### `-node-precision-info` yaml schema

Precision profile can be created with a list of output names of nodes required to run
in fp16 as shown below.

  ```
    FP16NodeInstanceNames: [109, 110, 111, 112, 237]
  ```

#### How to use mixed precision feature
Generate quantization profile using the following command
```
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data -dump-profile="profile.yaml" -node-precision-info="precision_profile.yaml"
```

Use the generated quantization profile from above command along with `-node-precision-info` to run the network in mixed precision
```
./bin/image-classifier tests/images/imagenet/*.png -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data -load-profile="profile.yaml" -node-precision-info="precision_profile.yaml"
```
