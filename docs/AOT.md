# Creating standalone executable bundles

This document provides a short description about producing ahead-of-time
compiled executable bundles. The motivation for this work is to remove the cost
of compile time by allowing the users of Glow to compile the package ahead of
time.

## Overview

A bundle is a self-contained compiled network model that can be used to execute
the model in a standalone mode. After following the instructions in this
document and the [CMakeLists.txt](../examples/bundles/resnet50/CMakeLists.txt) in the example
directory you will be able to compile convolutional neural networks into small
executables. Example:

```
  $cmake -G ninja <other cmake flags> -DGLOW_WITH_BUNDLES=ON -DGLOW_WITH_CPU=ON
  ...

  $ninja ResNet50Bundle
  ...

  $./resnet50 cat.png
  Result: 285
```


## Producing a bundle

It is possible to use the Glow library to produce bundles. On the CPU, the
bundles are object files that can be linked with some executable. On other
architectures, the bundle may look completely different.

This document demonstrates how to produce a bundle for the host CPU using the
'image-classifier' tool.  We use the flag `-emit-bundle` to specify the output
directory.

```
$image-classifier image.png -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data -backend=CPU -emit-bundle build/
```

The command above would compile the neural network model described by the files
`init_net.pb` and `predict_net.pb` located in the `network_model_directory_name`
directory and generate a bundle consisting of two files in the directory
`output_directory_name`, `<network_name>.o` and `<network_name>.weights` where
`<network_name>` is by default equals to the last directory in the model path,
i.e., `resnet50` in that case, and can be changed using
`-network-name=<network_name>`.
`predict_net.pb` describes the network model using the protobuf format for the ONNX
or the caffe2 representation. `init_net.pb` contains the weights that are used by the
network using the protobuf format as well.

The first generated file is named `<network_name>.o` and contains the compiled code
of the network model. By default, this is a non-relocatable object file that
can be linked with other files in your project. It is possible to control
the relocation model with the command line option `-relocation-model=<mode>`.

This option supports two modes:
- `static`: (Default) Produce non-relocatable code.
- `pic`: Produce position independent code.

The second generated file is named `<network_name>.weights` and
contains the weights required to run the compiled model.

## APIs exposed by bundles

This section describes the APIs that the CPU bundle exposes. Other targets may
expose a completely different API.

Each bundle exposes two symbols named `<network_name>` and
`<network_name>_config`, where, again, `<network_name>` is specified by the
`-network-name` command line option.  The `<network_name>` is the name of the
auto-generated function that implements the network model. This symbol always
has the following signature:

```c++
extern "C" void network_name(uint8_t *constantWeightVars,
                             uint8_t *mutableWeightVars,
                             uint8_t *activations);
```
The parameters of this function are the base addresses of the memory areas for
constant weights variables, mutable weights variables (i.e. inputs and outputs)
and activations.

The `<network_name>_config` is a symbol that contains the configuration of
the compiled network. The type of this symbol is always the following struct:
```c++
struct BundleConfig {
  // Size of the constant weight variables memory area.
  uint64_t constantWeightVarsMemSize;
  // Size of the mutable weight variables memory area.
  uint64_t mutableWeightVarsMemSize;
  // Size of the activations memory area.
  uint64_t activationsMemSize;
  // Alignment to be used for weights and activations.
  uint64_t alignment;
  // Number of symbols in the symbol table.
  uint64_t numSymbols;
  // Symbol table.
  const SymbolTableEntry *symbolTable;
};
```
This configuration is supposed to be used by the client code to allocate the
required amounts of memory for each of the memory areas, before invoking the
`<network_name>` function to run the network.

Clients also use `BundleConfig` to perform the symbol table lookups when they
need to find information about an input or output variable.
The SymbolTableEntry always has the following structure:
```c++
struct SymbolTableEntry {
  // Name of a variable.
  const char *name;
  // Offset of the variable inside the memory area.
  uint64_t offset;
  // The number of elements inside this variable.
  uint64_t size;
  // The kind of the variable. 1 if it is a mutable variable, 0 otherwise.
  char kind;
};
```

Offsets of constants are offsets inside the memory area for constant weights.
Offsets of mutable variables are offsets inside the memory area for mutable
weights.

## How to use the bundle

This section describes the use of the CPU bundle. Other targets may have
different interfaces.

To integrate the artifacts generated by the image-classifier into your project, you
generally need to do the following:
* You need to link with the generated object file `<network_name>.o`.
* You need to allocate the memory for constant weights variables,
mutable weights variables (i.e. inputs and outputs) and activations based on the
memory area sizes provided by `<network_name>_config`.
* You need to load the content of the auto-generated `network_model_name.weights`
file into the constant weights variables memory area.
* And need to initialize the mutable weights area with inputs (e.g. image data)
* And finally, you need to invoke the `<network_name>` function with 3
parameters that are base addresses of the memory areas for constant weights variables,
mutable weights variables, and activations.
* After `<network_name>` has returned, you can find the results of the mutable weights
variables area.

## A step-by-step example of the Resnet50 network model

There are concrete examples of integrating a network model with a project located in the `examples/bundles/` directory in the Glow repository. You can enable the compilation of these bundles by invoking `cmake` with `-DGLOW_WITH_BUNDLES=ON -DGLOW_WITH_CPU=ON`.

### Floating point network
To build and run the example, you just need to execute:
* `cmake -G ninja <other cmake flags> -DGLOW_WITH_BUNDLES=ON -DGLOW_WITH_CPU=ON`
* `ninja RunResNet50Bundle`

The CMakeLists.txt provides the following targets:
* `ResNet50BundleNetFiles`: it downloads the Resnet50 network model in the Caffe2 format.
* `ResNet50BundleNet`: it generates the bundle files using the Glow image-classifier as described above.
  The concrete command line looks like this:
  `image-classifier tests/images/imagenet/cat_285.png -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data -backend=CPU -emit-bundle <build_dir>`
  It reads the network model from `resnet50` and generates the `resnet50.o`
  and `resnet50.weights` files into the `build_dir` directory.
* `ResNet50BundleMain`:  it compiles the `main.cpp` file, which is the main file of the project.
  This source file gives a good idea about how to interface with an auto-generated bundle.
  It contains the code for interfacing with the auto-generated bundle.
  *  It allocated the memory areas based on their memory sizes provided in `resnet50_config`.
  *  Then it loads the weights from the auto-generated `resnet50.weights` file.
  *  It loads the input image, pre-processes it and puts it into the mutable weight variables
     memory area.
  *  Once everything is setup, it invokes the compiled network model by calling the
     `resnet50` function from the `resnet50.o` object file.
* `ResNet50Bundle`: it links the user-defined `main.o` and auto-generated `resnet50.o` into a standalone executable file called `resnet50`
### Quantized network

All of the aforementioned targets have quantized versions in CMakeLists.txt named
`QuantizedResNet50BundleNet`, `QuantizedResNet50Bundle`.

This run performs almost the same steps as non-quantized Resnet50 version
except it emits bundle based on the quantization profile:
`image-classifier tests/images/imagenet/cat_285.png -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data -load-profile=profile.yml -backend=CPU -emit-bundle build`

The `profile.yml` itself is captured at a prior step by executing image-classifier with the `dump-profile` option:
`image-classifier tests/images/imagenet/*.png -image-mode=0to1 -m=resnet50 -model-input-name=gpu_0/data -dump-profile=profile.yml`.

See the [CMakeLists.txt](../examples/bundles/resnet50/CMakeLists.txt) for details.
