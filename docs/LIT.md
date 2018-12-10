# LLVM Integrated Tester (LIT) in Glow

Glow supports LLVM style LIT tests. These tests provide a way to run our high
level tools through command line invocations like a user would do.

To run these test, glow needs to know where to find two tools:
* `lit`: This is the driver of the lit testsuite.
* `FileCheck`: This is the tool used to check that the test ran correctly.

To use the LIT testing, you need to have both these tools in your path before
invoking CMake. More specifically, CMake will determine whether to enable the
LIT testing based on the presence or absent of these two executables.

Note: `lit` and `lit.py` can be used indistinctly, CMake will recognize both.

When those tools are set, a typical CMake invocation supporting our LIT tests
looks like this:
  ```bash
  cmake -G Ninja <glow_src>  -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=<path_to_llvm_install>      \
        -DGLOW_MODELS_DIR=<downloaded_c2_models>
  ```

## Installing the Tools

### From LLVM source

Both tools `lit.py` and `FileCheck` are available directly from the LLVM source
and our `utils/build_llvm.sh` makes them readily available.

Alternatively, `utils/set_llvm_test_env.sh` provides a light way of exporting
the tools while not building LLVM in full.
To use it, simply run:
  ```bash
  source utils/set_llvm_test_en.sh
  ```
This will build the required tools and update your `PATH` accordingly.

### Using a Package Manager

LIT is a python package that can easily be obtained through a python package
manager.
Simply run:
  ```bash
  pip install lit
  ```

If using `brew`, `FileCheck` comes with the LLVM package that we already
described in the earlier section. I.e., `FileCheck` comes with:
  ```bash
  brew install --with-toolchain llvm@6
  ```

For `apt-get` users, `FileCheck` comes with the `llvm-tools` package.
To get it, simply run:
  ```bash
  sudo apt-get install -y llvm-6.0-tools
  ```

Make sure both `lit` and `FileCheck` are in the `PATH` and you can invoke CMake.

## Other Dependencies

Most, if not all, of our lit tests run real production models. Because of their
size, they are not directly available from a fresh glow checkout and we have to
tell CMake where to find them. Moreover, they may take a while to execute, in
particular in debug mode, hence, most of the lit tests are run only in release
mode.

### Get the Models

The models are obtained using our `utils/download_*_models.sh` scripts.
Right now, we only test C2 models, so it is enough to run only:
  ```bash
  utils/download_caffe2_models.sh
  ```

Then, we need to tell CMake where to find the models. This is done with the
`GLOW_MODELS_DIR` CMake variable.

### Running in Release Mode

To build glow in release mode and enable most of the lit tests, simply use
the CMake variable `CMAKE_BUILD_TYPE` with one of the release variant:
`Release` or `RelWithDebInfo`.


## Running the Tests

The lit tests run with all the other unittests (`ninja test` or `ninja check`).
However, it is possible to run only the lit tests, using:
  ```bash
  ninja litTests
  ```

Note: Unlike the `ninja check` target, this one won't build the command line
tools for you, so either you will test potentially outdated tools or will get
`Unsupported` because the tools are just not built.

If you want to run an individual test, you can run the following command from
your build directory:
  ```bash
  lit -s -v <glow_src>/tests/<path_to_test>
  ```

With these command line options, `lit` will print the commands it actually used
when a test fails. To see the command of all the tests, use `-a`.

## Adding a LIT Test

To add a lit test, create a `.test` file containing your test in one of the
subdirectories of `<glow_src>/tests`. That's it!

## Structure of a LIT Test

This section is a crash course on what a lit test looks like in glow. Check the
[lit pages of LLVM](http://llvm.org/docs/CommandGuide/lit.html) for more
information.

Essentially, a lit test defines four things:
* The command lines to be run, using the `RUN:` prefix.
* The pattern to look for, using the `CHECK:` prefix. For more details look at
  the [`FileCheck`
  documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)
* The required configuration for the test using the `REQUIRES:` prefix.
* The input of the test (whatever is left in the file).

Note: In fact, only the `RUN:` lines are required. If there is nothing to
pattern match, the test would be declared successful whenever the executed
command line returns 0, like any `bash` command.

## Substitutions

Glow lit environment provides several substitution macros on top of the ones
that come with stock lit (see [the lit pages for the default
substitutions](https://llvm.org/docs/TestingGuide.html#substitutions)). Those
are:
* `%text-translator`: Will be expanded in `<build_dir>/bin/text-translator`.
* `%image-classifier`: Will be expanded in `<build_dir>/bin/image-classifier`.
* `%model-runner`: Will be expanded in `<build_dir>/bin/model-runner`.
* `%models_dir`: Will be expanded in `<GLOW_MODELS_DIR>` as provided on the CMake command.

These substitutions are defined in `tests/lit.cfg.py`, which is loaded by `lit`
before running the tests.

## Test Requirements Description

By default, `lit` runs all the tests that live in specific directories. For
glow, this represents all the files ending in `.test` under the `tests`
directory and their subdirectories, as described in `tests/lit.cfg.py`.

However, some tests are specific to some CMake configuration. For instance,
we want to test our `CPU` backend only when it has been built.
This is what the lines starting with `REQUIRES:` express in a test file.
Those lines mean that the test will be run if and only if all the
requirements are fulfilled.

Right now, glow defines the following configurations to be used in `REQUIRES:`
directives:
* `cpu`: Will be run only if glow has been built with CPU support.
* `opencl`: Will be run only if glow has been built with OpenCL support.
* `release`: Will be run only if glow has been built in release mode.

Those configurations are declared in `tests/lit.cfg.py` and use the
information of a file generated by CMake, `build_dir/tests/litconfig.py`,
to set the right values.

If one of the requirement is not fulfilled, `lit` will report `Unsupported` for
that test.
An `Unsupported` test is skipped and does not count as a failure.

Finally, some directories have specific requirements to be run. For instance,
all the tests living in `tests/text-translator` should only run if both
`text-translator` is available and `GLOW_MODELS_DIR` was set on the command
line. These requirements are described in the different `lit.local.cfg` of the
related directories.
