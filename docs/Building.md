# Building the compiler

This document contains detailed information about building the compiler on different
systems and with different options.

## Building with JIT/OpenCL Backends

By default Glow builds with only the interpreter backend enabled.  To enable
support for the JIT and/or OpenCL backends, pass additional options to cmake:

  ```
  -DGLOW_WITH_CPU=1 -DGLOW_WITH_OPENCL=1
  ```

### OpenCL on Ubuntu

If you decide to use OpenCL, the easiest way is to install
portable open source implementation of the OpenCL standard,
[pocl](https://github.com/pocl/pocl). Glow relies on pocl to run OpenCL tests
on CI. All required steps are outlined in the [install_pocl](https://github.com/pytorch/glow/blob/bd66a74eab5e7d855052221bb6715fcd499af3b6/.circleci/build.sh#L29) method.

Alternatively, you can follow these steps:

1. Install necessary packages:

  ```bash
  sudo apt-get install ocl-icd-opencl-dev ocl-icd-libopencl1 opencl-headers \
      clinfo
  ```

2. Install the appropriate runtime for your CPU/GPU. This will depend on your
hardware. If you have an Intel CPU with onboard graphics, you can navigate to
Intel's compute-runtime releases page on Github at
https://github.com/intel/compute-runtime/releases/ and follow their
instructions. You will probably want to choose the latest release and then
download and install about ~4 prebuilt packages. At the time of this writing,
the prebuilt packages of compute-runtime Release 18.45.11804 ran successfully
with Glow on an Intel Core i7-7600U running Ubuntu 16.04.1.

3. To determine if installation was successful, you can run the following
command:

  ```bash
  clinfo
  ```

This will display information about your OpenCL platforms and devices (if
found). Lastly, build Glow with the cmake flag `-DGLOW_WITH_OPENCL=ON` and run
the test `OCLTest`.

### Supporting multiple targets

The JIT is able to target any/all environments supported by LLVM.  If the
`build_llvm.sh` script is used to build LLVM for Glow, all the currently stable
architectures will be enabled.  If you wish to control which architectures are
built, you can use the `LLVM_TARGETS_TO_BUILD` cmake parameter when building
LLVM.

## Building with the Sanitizers

The clang-sanitizer project provides a number of libraries which can be used with
compiler-inserted instrumentation to find a variety of bugs at runtime.  These
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

Glow uses gcov, lcov, and genhtml to generate coverage reports for the code base.
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

The output will be placed in the `docs/html` subdirectory of the build output
directory.

## Generate dependency graph for cmake targets

Run cmake normally and then execute `dependency_graph` target. The `dependency_graph`
file contains dependencies for all project targets. It might look a bit overwhelming,
in that case you could check `dependency_graph.loader`, `dependency_graph.cifar10`, etc.
