# Building the compiler

This document contains detailed information about building the compiler on different
systems and with different options.

## macOS

To build the compiler on a mac you would need to install a few dependencies. You
can install these dependencies using the brew package manager:

  ```
    brew install libpng protobuf graphviz cmake wget
  ```

## Ubuntu

On Ubuntu you would need to install a few dependencies. The following command should install the required dependencies.

  ```
  sudo apt-get install graphviz clang cmake wget ninja-build llvm-5.0 libprotobuf-dev protobuf-compiler 
  ```

Note, that OpenCL support is not trivial on Linux. We suggest to build without OpenCL for the first time.

## Building with JIT/OpenCL Backends

By default Glow builds with only the interpreter backend enabled.  To enable
support for the JIT and/or OpenCL backends, pass additional options to cmake:

  ```
  -DGLOW_WITH_CPU=1 -DGLOW_WITH_OPENCL=1
  ```

### Supporting multiple targets

The JIT is able to target all environments supported by LLVM.  If the
`build_llvm.sh` script is used to build LLVM for Glow, all the currently stable
architectures will be enabled.  If you wish to control which architectures are
built, you can use the `LLVM_TARGETS_TO_BUILD` parameter to CMake, which is a
list of architectures to support, to enable the desired targets.

## Building with the Sanitizers

The clang-sanitizer project provides a number of libraries which can be used with
compiler inserted instrumentation to find a variety of bugs at runtime.  These
include memory issues due such as use-after-free or double-free.  They can also
detect other types of problems like memory leaks.  Glow can be built with the
sanitizers enabled using an additional parameter to cmake.

The following sanitizers are currently configurable:

  - Address
  - Memory
  - Undefined
  - Thread
  - Leaks

You can pass one of the above as a value to the cmake parameter
`GLOW_USE_SANITIZER`.  `Address` and `Undefined` are special in that they may be
enabled simultaneously by passing `Address;Undefined` as the value.
Additionally, the `Memory` sanitizer can also track the origin of the memory
which can be enabled by using `MemoryWithOrigins` instead of `Memory`.

## Building with clang-tidy

The clang project provides an additional utility to scan your source code for
possible issues.  Enabling `clang-tidy` checks on the source code is easy and
can be done by passing an additional cmake parameter during the configure step.

  ```
  -DCMAKE_CXX_CLANG_TIDY=$(which clang-tidy)
  ```

Adding this to the configure step will automatically run clang-tidy on the
source tree during compilation. Use the following configuration to enable
auto-fix and to enable/disable specific checks:

  ```
  -DCMAKE_CXX_CLANG_TIDY:STRING="$(which clang-tidy);-checks=...;-fix"
  ```

## Building with coverage

Glow uses gcov, lcov and genhtml to generate coverage reports for the code base.
Using this tool allows you to make sure that corner cases are covered with the
unit tests as well as keep the unit test coverage at a healthy level.
You can generate a coverage report by providing additional options to cmake:
  
  ```
  -DGLOW_USE_COVERAGE=1
  ```
and then invoke ```glow_coverage``` make target.

## Building Doxygen documentation

Building documentation can be enabled by passing an additional cmake parameter:

  ```
  -DBUILD_DOCS=ON
  ```

Output will be placed in the `docs/html` subdirectory of the build output
directory.