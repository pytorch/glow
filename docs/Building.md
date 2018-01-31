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
  sudo apt-get install graphviz clang cmake  wget ninja-build llvm ocl-icd-opencl-dev libprotobuf-dev protobuf-compiler 
  ```

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
