# Glow

[![Build Status](https://travis-ci.org/pytorch/glow.svg?branch=master)](https://travis-ci.org/pytorch/glow)
[![Code Coverage](https://img.shields.io/badge/coverage-open-brightgreen.svg?style=flat)](https://fb-glow-assets.s3.amazonaws.com/coverage/coverage-master/index.html)

Glow is a machine learning compiler and execution engine for various hardware
targets.  It is designed to be used as a backend for high-level machine learning
frameworks.  The compiler is designed to allow state of the art compiler
optimizations and code generation of neural network graphs. This library is in
active development.

## How does it work?

Glow lowers a traditional neural network dataflow graph into a two-phase
strongly-typed [intermediate representation (IR)](./docs/IR.md). The high-level
IR allows the optimizer to perform domain-specific optimizations. The
lower-level instruction-based address-only IR allows the compiler to perform
memory-related optimizations, such as instruction scheduling, static memory
allocation and copy elimination. At the lowest level, the optimizer performs
machine-specific code generation to take advantage of specialized hardware
features. Glow features a lowering phase which enables the compiler to support a
high number of input operators as well as a large number of hardware targets by
eliminating the need to implement all operators on all targets. The lowering
phase is designed to reduce the input space and allow new hardware backends to
focus on a small number of linear algebra primitives.
The design philosophy is described in an [arXiv paper](https://arxiv.org/abs/1805.00907).

![](./docs/3LevelIR.png)

## Getting Started

### System Requirements

Glow builds and runs on macOS and Linux. The software depends on a modern C++
compiler that supports C++11, on CMake, LLVM, protocol buffers, and libpng.

#### Get Glow!

  ```bash
  git clone git@github.com:pytorch/glow.git  # or: git clone https://github.com/pytorch/glow.git
  cd glow
  ```

#### Submodules

Glow depends on a few submodules: googletest, onnx, and a library
for FP16 conversions.

To get them, from the glow directory, run:

  ```bash
  git submodule update --init --recursive
  ```

#### macOS

Install the required dependencies using [Homebrew](https://brew.sh/):

  ```bash
  brew install cmake graphviz libpng ninja protobuf wget
  brew install --with-toolchain llvm@6
  ```

Note that LLVM is installed to a non-default location (`/usr/local/opt/llvm`) to
avoid conflicts with the system's LLVM.

#### Ubuntu

On Ubuntu you would need to install a few dependencies. The following command
should install the required dependencies.

  ```bash
  sudo apt-get install graphviz clang cmake wget ninja-build llvm-5.0 \
      libprotobuf-dev protobuf-compiler libpng-dev
  ```

In order to support ONNX net serialization format, Glow requires
`protobuf >= 2.6.1`, but the above command may install older
version on older Ubuntu (e.g. 14.04). If this is the case, we suggest to look
at `utils/install_protobuf.sh` to install newer version from source.

Note, that OpenCL support is not trivial on Linux. We suggest to build without
OpenCL for the first time.

### Configure and build

To build the compiler, create a build directory and run cmake on the source
directory. It's a good idea to build two configurations (Release and Debug)
because some programs take a really long time to run in Debug mode. It's also a
good idea to build the project outside of the source directory.

  ```bash
  mkdir build_Debug
  cd build_Debug
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ../glow
  ninja all
  ```

It's possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.

### Building with dependencies (LLVM)

By default, Glow will use a system provided LLVM.  Note that Glow requires LLVM
5.0 or later.  If you have LLVM installed in a non-default location (for
example, if you installed it using Homebrew on macOS), you need to tell CMake
where to find llvm using `-DCMAKE_PREFIX_PATH`.  For example:

  ```bash
  cmake -G Ninja ../glow \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_PREFIX_PATH=/usr/local/opt/llvm
  ```

If LLVM is not available on your system you'll need to build it manually.  Run
the script '`/utils/build_llvm.sh` to clone, build and install LLVM in a local
directory. You will need to configure Glow with the flag `-DCMAKE_PREFIX_PATH`
to tell the build system where to find LLVM (e.g. the location of
`llvm_install/` if using `build_llvm.sh`).

For more platform-specific build instructions and advanced options, such as
building with Address-Sanitizers refer to this guide:
[Building the Compiler](docs/Building.md).

## Testing and Running

### Unit tests

The project has a few unit tests in the tests/unittests subdirectory. To run all
of them, simply run `ninja test`.

### C++ API examples

A few test programs that use Glow's C++ API are found under the `examples/`
subdirectory. The `mnist`, `cifar10`, `fr2en` and `ptb` programs train and run digit
recognition, image classification and language modeling benchmarks,
respectively.

To run these programs, build Glow in Release mode, then run the following commands
to download the cifar10, mnist and ptb databases.

  ```bash
  python ../glow/utils/download_test_db.py --all
  ```

Now run the examples. Note that the databases should be in the current working
directory.

  ```bash
  ./bin/mnist
  ./bin/cifar10
  ./bin/fr2en
  ./bin/ptb
  ./bin/char-rnn
  ```

If everything goes well you should see:
  * `mnist`: pictures from the mnist digits database
  * `cifar10`: image classifications that steadily improve
  * `fr2en`: an interactive French-to-English translator
  * `ptb`: decreasing perplexity on the dataset as the network trains
  * `char-rnn`: generates random text based on some document

Note that the default build mode is `Debug`, which means that the compiler
itself is easy to debug because the binary contains debug info, lots of
assertions, and the optimizations are disabled. It also means that the compiler
and runtime are very slow, and the execution time can be hundreds of times
slower than that of release builds. If you wish to benchmark the compiler, run
long benchmarks, or release the product then you should compile the compiler in
Release mode. Check the main CMake file for more details.

More details on testing and running Glow can be found in: [Testing the Glow
Compiler](docs/Testing.md).

### Ahead-of-time Compilation

Glow can be used to compile neural networks into object files containing native
code.  We provide resnet50 (both quantized and non-quantized versions) as an
example of this capability in `examples/bundles/resnet50`.  See [Creating
Standalone Executable Bundles](docs/AOT.md) for more detail.

## Contributing

To get started, please refer to the following guides:
* [Contributing](CONTRIBUTING.md)
* [Coding Standards](docs/CodingStandards.md)

### Communication

* forums: discuss implementations, research, etc. http://discuss.pytorch.org.
  Make sure to label topic with the ["glow"](https://discuss.pytorch.org/c/glow) category.
* GitHub issues: bug reports, feature requests, install issues, RFCs, thoughts, etc.

## License

Glow is licensed under the [Apache 2.0 License](LICENSE).
