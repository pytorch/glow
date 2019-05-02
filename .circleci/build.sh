#!/bin/bash

# This script setup an working environemnt for running glow tests and gtest driver.
# By default, we run with the enabled CPU backend and disabled OpenCL backend.
set -ex

export MAX_JOBS=8

install_opencl_drivers() {
  sudo apt-get install -y ocl-icd-opencl-dev ocl-icd-libopencl1 clinfo libhwloc-dev opencl-headers

  mkdir neo
  cd neo
  wget https://github.com/intel/compute-runtime/releases/download/18.45.11804/intel-gmmlib_18.4.0.348_amd64.deb
  wget https://github.com/intel/compute-runtime/releases/download/18.45.11804/intel-igc-core_18.44.1060_amd64.deb
  wget https://github.com/intel/compute-runtime/releases/download/18.45.11804/intel-igc-opencl_18.44.1060_amd64.deb
  wget https://github.com/intel/compute-runtime/releases/download/18.45.11804/intel-opencl_18.45.11804_amd64.deb
 
  sudo dpkg -i *.deb
  cd ..
}

# Install Glow dependencies
sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main"
sudo apt-get update
sudo apt-get install -y llvm-8 clang-8 llvm-8-dev libpng-dev libgoogle-glog-dev

# Redirect clang
sudo ln -s /usr/bin/clang-8 /usr/bin/clang
sudo ln -s /usr/bin/clang++-8 /usr/bin/clang++
sudo ln -s /usr/bin/llvm-symbolizer-8 /usr/bin/llvm-symbolizer
sudo ln -s /usr/bin/llvm-config-8 /usr/bin/llvm-config-8.0

GLOW_DIR=$PWD

# Install ninja and (newest version of) cmake through pip
sudo pip install ninja cmake
hash cmake ninja

# Build glow
cd ${GLOW_DIR}
mkdir build && cd build
CMAKE_ARGS=("-DCMAKE_CXX_COMPILER=/usr/bin/clang++-8")
CMAKE_ARGS+=("-DCMAKE_C_COMPILER=/usr/bin/clang-8")
CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER_LAUNCHER=sccache")
CMAKE_ARGS+=("-DCMAKE_C_COMPILER_LAUNCHER=sccache")
CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-Werror")
CMAKE_ARGS+=("-DGLOW_WITH_CPU=ON")
CMAKE_ARGS+=("-DGLOW_WITH_HABANA=OFF")
if [[ "${CIRCLE_JOB}" == "ASAN" ]]; then
    CMAKE_ARGS+=("-DGLOW_USE_SANITIZER='Address;Undefined'")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=OFF")
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
elif [[ "${CIRCLE_JOB}" == "TSAN" ]]; then
    CMAKE_ARGS+=("-DGLOW_USE_SANITIZER='Thread'")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=OFF")
elif [[ "$CIRCLE_JOB" == RELEASE_WITH_EXPENSIVE_TESTS ]]; then
    # Download the models and tell cmake where to find them.
    MODELS_DIR="$GLOW_DIR/downloaded_models"
    DOWNLOAD_EXE="$GLOW_DIR/utils/download_caffe2_models.sh"
    mkdir $MODELS_DIR
    (
        cd $MODELS_DIR
        $DOWNLOAD_EXE
    )
    CMAKE_ARGS+=("-DGLOW_MODELS_DIR=$MODELS_DIR")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=OFF")
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
else
    install_opencl_drivers
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Debug")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=ON")
    if [[ "${CIRCLE_JOB}" == "SHARED" ]]; then
        CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=ON")
    fi
fi

cmake -GNinja ${CMAKE_ARGS[*]} ../
ninja

# Build onnxifi test driver (Only for DEBUG mode)
if [[ "$CIRCLE_JOB" == DEBUG ]]; then
    cd ${GLOW_DIR}
    ./tests/onnxifi/build_onnxifi_tests.sh
fi

# Report sccache hit/miss stats
if hash sccache 2>/dev/null; then
    sccache --show-stats
fi
