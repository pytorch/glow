# Creating standalone executable bundles

This document provides a short description about producing Ahead Of Time (AOT)
compiled executable bundles. The motivation for this work is to remove the cost
of compile time by allowing the users of Glow to compile a network model Ahead Of
Time.


## Overview

A bundle is a self-contained compiled network model that can be used to execute
the model in a standalone mode. Glow has multiple backends which can run models
but not all of them are capable of saving a compiled model.

The main backend used to generate bundles is the **CPU backend** where the
bundle exists as a self-contained object file (library) containing all the
necessary code to run a model. The bundle has a single entry function which
performs inference for the model.

After following this document you will be able to compile models into bundles.
You can view the CMake instructions (*CMakeLists.txt*) used to build the bundles
and also the applications (*main.cpp*) which integrate the bundles for the ResNet50
and LeNetMnist models here:
- [ResNet50 example](../examples/bundles/resnet50)
- [LeNetMnist example](../examples/bundles/lenet_mnist)

When building Glow you can choose to build the **ResNet50** and the **LeNetMnist**
examples by enabling the `GLOW_WITH_BUNDLES` option as below:

```
mkdir build_Debug
cd build_Debug
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGLOW_WITH_BUNDLES=ON ../glow
ninja all
```

After the build is complete, you can run the example applications using one of the
following commands (the executables are found in the `build_Debug/bundles` folder):

```
./ResNet50Bundle cat_285.png
./QuantizedResNet50Bundle cat_285.png
./LeNetMnistBundle 0_1009.png
./QuantizedLeNetMnistBundle 0_1009.png
```

You can find the sample images used in the examples above in the following directories:
- Sample images for the ResNet50 model from the IMAGENET database [here](../tests/images/imagenet)
- Sample image for the LeNetMnist model from the MNIST database [here](../tests/images/mnist)


## Compile a bundle for a floating-point model

The **model-compiler** front-end tool is the main Glow tool used to compile ONNX and
Caffe2 models into bundles. The tool is generic in the sense that it can compile
models with any number of inputs or outputs, without being limited to a particular
application.

The command used to build a bundle using the CPU backend is the following:
```
model-compiler -backend=CPU -model=<model-path> -emit-bundle=<bundle-dir>
```

- The option `backend` specifies the backend used to generated the bundle.
- The option `model` specifies the path for the Caffe2 or ONNX model:
  - For Caffe2 models the `<model-path>` is expected to point to a **directory**
  which contains two model files: `init_net.pb` and `predict_net.pb`. Some Caffe2
  model directories also contain a file named `predict_net.pbtxt` but that file
  is not required.
  - For ONNX models the `<model-path>` is expected to point to a **file** with
  the extension **onnx** or **onnxtxt**.
- The option `emit-bundle` specifies the output **directory** where all the bundle
  artifacts will be generated. If the directory does not exist, it will be created.

There is a small difference when using this tool with ONNX versus Caffe2 models:
- For **ONNX models** the tool can infer automatically the inputs of the model
since the description of the input tensors is part of the model. Therefore the tool
will be used in the form shown above:
  ```
  model-compiler -backend=CPU -model=<onnx-model-path> -emit-bundle=<bundle-dir>
  ```
- For **Caffe2 models** the user must also explicitly provide the description
of the input tensors which are not part of model. The option `model-input` will be used
to specify the name, type and shape for each model input. The model input names can be
deduced by inspecting the model with [Netron](https://lutzroeder.github.io/netron/).
The option can be used multiple times to describe each model input separately:
  ```
  model-compiler -backend=CPU -model=<caffe2-model-path> -emit-bundle=<bundle-dir>  \
    -model-input=<inputName1>,<inputType1>,<inputShape1>                            \
    -model-input=<inputName2>,<inputType2>,<inputShape2>                            \
    ....................................................
  ```
  For quantized types the format of the `-model-input` is slightly different since the
  scale and offset parameters should also be provided:
  ```
  -model-input=<name>,<type>,<scale>,<offset>,<shape>
  ```
  For example we can can provide one or more inputs with:
  ```
  -model-input=input_03_data,float,[1]
  -model-input=data_bias,int32,[1,32,32]
  -model-input=data,int8q,0.123,-13,[1,10]
  ```

After running the **model-compiler** tool, the following bundle artifacts will be generated
in the output directory:
- `<network_name>.o` - the bundle object file (code).
- `<network_name>.h` - the bundle header file (API).
- `<network_name>.weights.bin` - the model weights in binary format.
- `<network_name>.weights.txt` - the model weights in text format as C text array.

For example, to compile the **ResNet50** model provided by Glow in Caffe2 format, you can use
the following command. The name of the input tensor is *gpu_0/data*, the type is *float*
and the shape is *1 x 3 x 224 x 224* corresponding to a RGB image with NCHW layout:
```
model-compiler -backend=CPU -model=resnet50 -emit-bundle=build -model-input=gpu_0/data,float,[1,3,224,224]
```

Similarly, to compile the **LeNetMnist** model provided by Glow in Caffe2 format, you can use
the following command. The name of the input tensor is *data*, the type is *float* and the
shape is *1 x 1 x 28 x 28* corresponding to a grayscale image with NCHW layout:
```
model-compiler -backend=CPU -model=lenet_mnist -emit-bundle=build -model-input=data,float,[1,1,28,28]
```

For more information about the options of the **model-compiler** tool use:
```
model-compiler -help
```


## Compile a bundle for a quantized model

Glow support for out-of-the-box quantized models is a work in progress mainly because the
quantized tensors and operators are not well standardized in formats like Caffe2 and ONNX.

The way Glow produces a quantized bundle is by taking a floating-point model and converting
it to a quantized model using it's own internal representation before compiling to a bundle.

The procedure used for quantizing the model is called **profile-guided quantization** (more
details [here](./Quantization.md)). Before the model is quantized and compiled with the
*model-compiler* tool, the quantization profile must be acquired.

In order to compute the quantization profile, the **image-classifier** tool is used. This
application is used for image classification models only and requires a set of images to do
the inference and compute the profile:
```
image-classifier <images> <image-opts> -model=<model-path> -model-input-name=<name> -dump-profile=profile.yaml
```

- The image paths are specified one after the other, separated by space. Only images in **PNG**
format are supported (RGB or grayscale).
- `model` specifies the path for the Caffe2 or ONNX model.
- `model-input-name` specifies the name of the model input.
- The option `dump-profile` specifies the file name used to dump the profile in YAML format.

Extra options are available to specify how the images are pre-processed before they are fed
to the model during inference:
- The option `image-mode` can be used to specify how the images are normalized:
  - `neg1to1` for the range [-1, 1].
  - `0to1` for the range [0, 1].
  - `0to255` for the range [0, 255] (Default).
  - `neg128to127` for the range [-128,127].
- The option `image-channel-order` can be used to specify the order of the channels:
  - `RGB` for R,G,B channel ordering.
  - `BGR` for B,G,R channel ordering (Default).
- The option `image-layout` can be used to specify the tensor dimension ordering (layout) which must
match the layout of the model input:
  - `NCHW` for NCHW tensor layout (Default).
  - `NHWC` for NHWC tensor layout.

For more information about the options of the **image-classifier** tool use:
```
image-classifier -help
```

After the quantization profile `profile.yaml` has been generated, we can use the **model-compiler**
tool to compile the model into a bundle by loading the previously generated profile:
```
model-compiler ... -load-profile=profile.yaml
```

If a specific quantization schema is desired, same schema must be provided during both
profiling and compiling using the `quantization-schema` option. More details about the
quantization schema can be found [here](./Quantization.md).

For example, in order to profile, quantize and compile the **ResNet50** model you can use the commands:
```
image-classifier cat_285.png ... -image-mode=0to1 -model=resnet50 -model-input-name=gpu_0/data -dump-profile=profile.yml
model-compiler -backend=CPU -model=resnet50 -emit-bundle=build -model-input=gpu_0/data,float,[1,3,224,224] -load-profile=profile.yml
```

Similarly, for the **LeNetMnist** model:
```
image-classifier 0_1009.png ... -image-mode=0to1 -model=lenet_mnist -model-input-name=data -dump-profile=profile.yml
model-compiler -backend=CPU -model=lenet_mnist -emit-bundle=build -model-input=data,float,[1,1,28,28] -load-profile=profile.yml
```

It is important to note that currently the quantization of a model is performed
only for the intermediate nodes of the graph, without affecting the data types of
the model inputs and outputs. In the examples above, the data type for the model input
(image tensor) and the model output remains **float** even though the intermediate
operators and tensors use **int8** data type.

When compiling a quantized bundle you can choose to disable the quantization of some
of the graph operators which might be more susceptible to quantization by using
the option `keep-original-precision-for-nodes`. For example in order to disable the
quantization for the operators Add,Sub,Mul,Div,Exp and SoftMax you can use:
```
model-compiler ... -keep-original-precision-for-nodes=Add,Sub,Mul,Div,Exp,SoftMax
```


## Cross-compile a bundle for a specific architecture

Since the CPU backend is based on LLVM, the Glow tools can be used to
cross-compile bundles for different target architectures. To specify
the target architecture you must use the `-target` and `-mcpu` flags
(if no target flags are provided the bundle will be generated by default
for the native architecture - the one which is running Glow). Below you
have a table with examples for some of the common architecures:

|Architecture    |Option                              |
|----------------|------------------------------------|
| x86 - 32 bit   | `-target=i386`                     |
| x86 - 64 bit   | `-target=x86_64`                   |
| ARM Cortex M0  | `-target=arm -mcpu=cortex-m0`      |
| ARM Cortex M4  | `-target=arm -mcpu=cortex-m4`      |
| ARM Cortex M7  | `-target=arm -mcpu=cortex-m7`      |
| ARM Cortex M33 | `-target=arm -mcpu=cortex-m33`     |
| ARM Cortex A53 | `-target=aarch64 -mcpu=cortex-a53` |
| ARM Cortex A72 | `-target=aarch64 -mcpu=cortex-a72` |

The bundle can be cross-compiled for any target architecture supported by
LLVM. For the complete list of LLVM target architectures you can type
`llc -version` command in Linux (assuming you have LLVM installed). For
example the LLVM 8.0.1 has the following supported architectures:

```
LLVM (http://llvm.org/):
  LLVM version 8.0.1
  
  Optimized build.
  Default target: x86_64-pc-linux-gnu
  Host CPU: skylake

  Registered Targets:
    aarch64    - AArch64 (little endian)
    aarch64_be - AArch64 (big endian)
    amdgcn     - AMD GCN GPUs
    arm        - ARM
    arm64      - ARM64 (little endian)
    armeb      - ARM (big endian)
    avr        - Atmel AVR Microcontroller
    bpf        - BPF (host endian)
    bpfeb      - BPF (big endian)
    bpfel      - BPF (little endian)
    hexagon    - Hexagon
    lanai      - Lanai
    mips       - MIPS (32-bit big endian)
    mips64     - MIPS (64-bit big endian)
    mips64el   - MIPS (64-bit little endian)
    mipsel     - MIPS (32-bit little endian)
    msp430     - MSP430 [experimental]
    nvptx      - NVIDIA PTX 32-bit
    nvptx64    - NVIDIA PTX 64-bit
    ppc32      - PowerPC 32
    ppc64      - PowerPC 64
    ppc64le    - PowerPC 64 LE
    r600       - AMD GPUs HD2XXX-HD6XXX
    sparc      - Sparc
    sparcel    - Sparc LE
    sparcv9    - Sparc V9
    systemz    - SystemZ
    thumb      - Thumb
    thumbeb    - Thumb (big endian)
    wasm32     - WebAssembly 32-bit
    wasm64     - WebAssembly 64-bit
    x86        - 32-bit X86: Pentium-Pro and above
    x86-64     - 64-bit X86: EM64T and AMD64
    xcore      - XCore
```


## Extra options

- If you want to change the base name of the bundle artifacts you can use the option:
  ```
  -network-name=<network-name>
  ```
  For example by using the option `-network-name=mb2` the generated bundle artifacts
will be named *mb2.o*, *mb2.h*, *mb2.weights.bin* and *mb2.weights.txt*.

- The generated bundle object file `<network_name>.o` is non-relocatable by default.
It is possible to control the relocation model with the command line option
`-relocation-model=<mode>`. This option supports two modes:
  - `static`: (Default) Produce non-relocatable code.
  - `pic`: Produce position independent code.

- The ONNX format allows some of the tensor dimensions to be
undefined (that is, marked as symbols and not as actual sizes) within the model. In the ONNX proto
description the symbolic sizes are marked with `dim_param` while the actual sizes
are marked with `dim_value`. For example, when inspecting an image classification
model with [Netron](https://lutzroeder.github.io/netron/) one might see that the
input tensor size is defined as `None x 3 x 224 x 224` where `None` is the undefined
size (symbol) associated with the batch size of the model. Glow cannot compile a
model with undefined sizes, therefore the user must assign manually the actual
values for all the symbols. The user will be prompted with an error if there is an
undefined symbol (for example `ONNX model symbol 'None' is undefined.`). In the above
example, the user can define the symbol `None` to have the value `1` by using the
following option:
  ```
  -onnx-define-symbol=None,1
  ```
  Multiple symbols can be defined using this option, for example:
  ```
  -onnx-define-symbol=<symbol_name1>,<symbol_value1>
  -onnx-define-symbol=<symbol_name2>,<symbol_value2>
  ..................................................
  ```

- When cross-compiling bundles for some target architectures you might
be interested in generating a bundle compatible with a given float ABI
(Application Binary Interface) type (*soft* or *hard*). The LLVM backend
can be instructed to generate an object file using a specific float ABI
by using the option `-float-abi=hard` or `-float-abi=soft`.

- When compiling the bundle it is useful to view the final form of the
graph after all the transformations and optimizations performed by Glow
(which might differ from the initial model). You can generate the graph
visual representation in *.dot* format by using the `-dump-graph-DAG`
option like in this:
  ```
  -dump-graph-DAG=graph.dot
  ```
  Additionally, you can convert the *.dot* file to *.pdf* format using the
  *dot* utility available on Linux like this:
  ```
  dot -Tpdf graph.dot -o graph.pdf
  ```


## Bundle memory layout

The memory of a bundle is organized in three separate memory regions which must be
allocated by the user application code and provided through the bundle interface:

- `constantWeight` - contains the model constant weights. The user application must:
  - allocate this memory region (statically or dynamically)
  - initialize this memory region with the content of the generated weights file in
    one of two possible formats:
    - binary format (`<network_name>.weights.bin`) used to initialize this memory
      region (allocated statically or dynamically) by loading the binary file
      dynamically at run-time using standard C function like **fopen**. 
    - text format (`<network_name>.weights.txt`) used to initialize this memory
      region (only if statically allocated) by including the text file statically
      at compile-time as a C array using the **#include** pre-processor directive.
      This format is suitable for target architectures which do not have file systems
      (for example microcontrollers).
  - provide the base address of this memory region to the inference function

- `mutableWeight` - contains all the model inputs and outputs (graph placeholders).
The tensors corresponding to different inputs and outputs are identified using offsets
relative to the base address of this memory region. The user application must:
  - allocate this memory region (statically or dynamically)
  - initialize the model input tensors from this memory region with the desired input
    data before running the inference
  - provide the base address of this memory region to the inference function
  - read the model output tensors from this memory region after running the inference

- `activations` - this memory region is a scratch memory required for the bundle code
to store the intermediate results of the graph computation (activations). The user
application must:
  - allocate this memory region (statically or dynamically)
  - provide the base address of this memory region to the inference function
  - this memory region is NOT required to be initialized

The required sizes for all the memory regions described above are provided in the bundle
interface. Also all the memory regions must be allocated with a minimum alignment which
is also provided in the interface (typically 64 bytes). For example, for aligning a
statically allocated buffer one can use the following C syntax: 

```c++
__attribute__((aligned(64)))
uint8_t aligned_buffer[BUFFER_SIZE];
```


## Static bundle API

This is the default bundle API obtained by generating the bundle with the option
`-bundle-api=static`. Below is an example of how the auto-generated header file
looks like for the Lenet Mnist model:

```c++
// Placeholder address offsets within mutable buffer (bytes)
#define LENET_MNIST_data        0
#define LENET_MNIST_softmax__1  3136

// Memory sizes (bytes)
#define LENET_MNIST_CONSTANT_MEM_SIZE     1724672
#define LENET_MNIST_MUTABLE_MEM_SIZE      3200
#define LENET_MNIST_ACTIVATIONS_MEM_SIZE  57600

// Memory alignment (bytes)
#define LENET_MNIST_MEM_ALIGN  64

// Bundle entry point (inference function)
void lenet_mnist(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations);
```

The header file contains all the information required to run the bundle,
defined in a static manner using macro defines:
- the offsets of all the placeholders (graph inputs/outputs) within the
`mutableWeight` memory
- the sizes for all the memory regions
- the alignment required for allocating the memory regions
- the inference function prototype

All the definitions names (the macros and the inference function) are prefixed
with the model name, in this example with *lenet_mnist*. If you want to change
the model name you can use the command line option `-network-name`, for example
`-network-name=my_bundle`.

The auto-generated header file file also contains some extra defines to
help with writing the user application code:

```c++
// Memory alignment definition with given alignment size
// for static allocation of memory.
#define GLOW_MEM_ALIGN(size)  __attribute__((aligned(size)))

// Macro function to get the absolute address of a
// placeholder using the base address of the mutable
// weight buffer and placeholder offset definition.
#define GLOW_GET_ADDR(mutableBaseAddr, placeholderOff)  (((uint8_t*)(mutableBaseAddr)) + placeholderOff)
```

For example, in order to allocate and initialize all the memory regions, you need
to write the following in the user application (*lenet_mnist.weights.txt* is the
file containing the model weights serialized as text):

```c++
GLOW_MEM_ALIGN(LENET_MNIST_MEM_ALIGN)
uint8_t constantWeight[LENET_MNIST_CONSTANT_MEM_SIZE] = {
#include "lenet_mnist.weights.txt"
};

GLOW_MEM_ALIGN(LENET_MNIST_MEM_ALIGN)
uint8_t mutableWeight[LENET_MNIST_MUTABLE_MEM_SIZE];

GLOW_MEM_ALIGN(LENET_MNIST_MEM_ALIGN)
uint8_t activations[LENET_MNIST_ACTIVATIONS_MEM_SIZE];
```

In order to obtain the absolute addresses of the model inputs/outputs
you need to write the following in the user application:

```c++
uint8_t *inputAddr  = GLOW_GET_ADDR(mutableWeight, LENET_MNIST_data);
uint8_t *outputAddr = GLOW_GET_ADDR(mutableWeight, LENET_MNIST_softmax__1);
```


## Dynamic bundle API

This is the bundle API obtained by generating the bundle with the option
`-bundle-api=dynamic`. Below is an example of how the auto-generated header
file looks like for the Resnet50 model:

```c++
// Bundle memory configuration (memory layout)
extern BundleConfig resnet50_config;

// Bundle entry point (inference function)
void resnet50(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations);
```

This API has all the information about the memory configuration encapsulated
in a structure named `<network_name>_config`. The layout of this structure is
defined by the type `BundleConfig` which is also included in the generated
header file:

```c++
// Type describing the config of a generated bundle.
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

Similar to the static API, this structure contains:
- the sizes for all the memory regions
- the alignment required for allocating all the memory regions
- the number of symbols
- the descriptions of all the symbols as an array of symbol entries

In this case the notion of *symbol* might include not only the model
placeholders but also the model constant weights. Each symbol is
described according to the `SymbolTableEntry` structure definition
(included also in the header file):

```c++
// Type describing a symbol table entry of a generated bundle.
struct SymbolTableEntry {
  // Name of a variable.
  const char *name;
  // Offset of the variable inside the memory area.
  uint64_t offset;
  // The number of elements inside this variable.
  uint64_t size;
  // Variable kind: 1 if it is a mutable variable, 0 otherwise.
  char kind;
};
```

For each symbol the following information is registered:
- the symbol name
- the symbol kind: whether is mutable (placeholder) or not (constant)
- the size in bytes
- the offset: if the symbol is mutable this is the offset of the variable
  within the `mutableWeight` buffer, otherwise this is the offset of the
  variable within the `constantWeight` buffer

The user has to look up the symbol entries to find the model variables
(placeholders or constants) at run-time (dynamically).


## How to use the bundle

This section describes how to use the bundle produced with the CPU backend. Other backends may
have different interfaces. To integrate the bundle into your project, you need to do the following:
1. Link the object file `<network_name>.o` to your application.
2. Include the header file `<network_name>.h` in your application.
3. Allocate memory for all the required buffers: `constantWeight`, `mutableWeight`, `activations`:
   - For the **static API** the buffer sizes and alignments are given by the macro defines from the
header file `<network_name>.h` and are available both at compile-time and run-time.
   - For the **dynamic API** the buffer sizes and alignments are provided in the `<network_name>_config`
structure and are only available at run-time.
4. Initialize the content of the `constantWeight` buffer with the model weights using either the
`<network_name>.weights.bin` file or the `<network_name>.weights.txt` file.
5. Initialize the model input tensors from the `mutableWeight` buffer (e.g. image data).
6. Invoke the main entry function `<network_name>`  by providing the base addresses of the memory
regions previously allocated.
7. After the main entry function has returned, you can find the results in the corresponding model
output tensors from the `mutableWeight` buffer.


## Extra utilities

### Visualize models

A very popular tool for visualizing the original model before compiling with Glow is **Netron** which
has:
- an online browser version [here](https://lutzroeder.github.io/netron/) (drag & drop a model into the browser window).
- an offline standalone version [here](https://github.com/lutzroeder/netron) (drag & drop a model into the GUI window).

### Convert models

The Glow compiler currently has support only for Caffe2 and ONNX model formats. Since a lot of popular
models are available in other formats, for example TensorFlow, it is useful to have tools to convert
models between different formats. The most used tools for format conversion are:
- [MMdnn](https://github.com/Microsoft/MMdnn)
- [tf2onnx](https://github.com/onnx/tensorflow-onnx)

Here we demonstrate how to convert a TensorFlow model to ONNX. We will convert a MobileNet V1 image
classification model which operates on 128 x 128 RGB images and 1001 classes. You can download the
model archive in TensorFlow format from [here](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz).

In order to convert the model to ONNX format using the **MMdnn** tool:
1. Install MMdnn from [here](https://github.com/Microsoft/MMdnn).
2. Download the MobileNet V1 model archive and unzip it.
3. Run the following command to convert the TensorFlow frozen file `mobilenet_v1_0.25_128_frozen.pb`
to the ONNX model file `mobilenet_v1_0.25_128.onnx`:
   ```
   mmconvert                                              \
     --srcFramework tensorflow                            \
     --inputWeight mobilenet_v1_0.25_128_frozen.pb        \
     --inNodeName input                                   \
     --inputShape 128,128,3                               \
     --dstNodeName MobilenetV1/Predictions/Softmax        \
     --dstFramework onnx                                  \
     --outputModel mobilenet_v1_0.25_128.onnx              
   ```

In order to convert the model to ONNX format using the **tf2onnx** tool:
1. Install tf2onnx from [here](https://github.com/onnx/tensorflow-onnx).
2. Download the MobileNet V1 model archive and unzip it.
3. Run the following command to convert the TensorFlow frozen file `mobilenet_v1_0.25_128_frozen.pb`
to the ONNX model file `mobilenet_v1_0.25_128.onnx`:
   ```
   python -m tf2onnx.convert                      \
     --input mobilenet_v1_0.25_128_frozen.pb      \
     --inputs input:0                             \
     --outputs MobilenetV1/Predictions/Softmax:0  \
     --output mobilenet_v1_0.25_128.onnx           
   ```

After converting the model to ONNX you can build a bundle using the Glow **model-compiler** tool with the command:
```
model-compiler -backend=CPU -model=mobilenet_v1_0.25_128.onnx -emit-bundle=bundle
```

When converting the model with **tf2onnx** some undefined symbols might end up in the ONNX model. For example,
the input batch size might be encoded as a symbol `unk__286` which can be defined to a value of `1` by using
the extra option:
```
-onnx-define-symbol=unk__286,1
```

### Edit protocol buffers

There are times when it is useful to manually edit the [protocol buffers](https://developers.google.com/protocol-buffers)
which are used for model formats like ONNX and TensorFlow. One such example is when
one needs to modify an ONNX model in order to circumvent a limitation of the Glow
compiler (for example when an attribute is not relevant but causes compiler issues, etc).

In order modify a protocol buffer you need:
- the proto compiler application **protoc** which can be found [here](https://github.com/protocolbuffers/protobuf/releases).
- the proto definition:
  - for ONNX format the proto definition **onnx.proto** can be found [here](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto).
  - for TensorFlow format the proto definition **saved_model.proto** can be found [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saved_model.proto).

When you need to manually edit an ONNX model you will need to:
- decode the model proto to text format.
- edit the model in text format.
- encode the model back to proto format.

After you install the **protoc** application and you download the proto definitions, you can
perform the actions defined below:

- Decode an ONNX model from proto `model.onnx` to text file `model.onnxtxt`:
  ```
  protoc onnx.proto --decode=onnx.ModelProto < model.onnx > model.onnxtxt
  ```

- Encode an ONNX model from text file `model.onnxtxt` to proto `model.onnx`:
  ```
  protoc onnx.proto --encode=onnx.ModelProto < model.onnxtxt > model.onnx
  ```

- Decode an ONNX tensor from proto `tensor.pb` to text file `tensor.pbtxt`:
  ```
  protoc onnx.proto --decode=onnx.TensorProto < tensor.pb > tensor.pbtxt 
  ```

- Encode an ONNX tensor from text file `tensor.pbtxt` to proto `tensor.pb`:
  ```
  protoc onnx.proto --encode=onnx.TensorProto < tensor.pbtxt > tensor.pb
  ```

- Decode/Encode TensorFlow saved model to/from text file:
  ```
  protoc saved_model.proto --decode tensorflow.SavedModel < model.pb > model.pbtxt
  protoc saved_model.proto --encode tensorflow.SavedModel < model.pbtxt < model.pb
  ```

- Decode/Encode TensorFlow frozen graph to/from text file:
  ```
  protoc saved_model.proto --decode tensorflow.GraphDef < model.pb > model.pbtxt
  protoc saved_model.proto --encode tensorflow.GraphDef < model.pbtxt > model.pb
  ```
