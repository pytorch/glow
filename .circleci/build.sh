#!/bin/bash

# This script setup an working environemnt for running glow tests and gtest driver.
# By default, we run with the enabled CPU backend and disabled OpenCL backend.
set -ex

export MAX_JOBS=8

install_pocl() {
   sudo apt-get install -y ocl-icd-opencl-dev clinfo libhwloc-dev opencl-headers

   git clone https://github.com/pocl/pocl.git
   cd pocl && git checkout 4efafa82c087b5e846a9f8083d46b3cdac2f698b && cd ../
   mkdir build_pocl
   cd build_pocl
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/clang++-8 -DCMAKE_C_COMPILER=/usr/bin/clang-8 -DENABLE_ICD=ON ../pocl
   make -j`nproc`
   sudo make install

   sudo mkdir -p /etc/OpenCL/vendors/
   sudo cp /usr/local/etc/OpenCL/vendors/pocl.icd /etc/OpenCL/vendors/

   clinfo
   cd ../
}

if [ "${CIRCLE_JOB}" != "CHECK_CLANG_FORMAT" ]; then
    # Install Glow dependencies
    sudo apt-get update

    # Redirect clang
    sudo ln -s /usr/bin/clang-8 /usr/bin/clang
    sudo ln -s /usr/bin/clang++-8 /usr/bin/clang++
    sudo ln -s /usr/bin/llvm-symbolizer-8 /usr/bin/llvm-symbolizer
    sudo ln -s /usr/bin/llvm-config-8 /usr/bin/llvm-config-8.0

    sudo apt-get install -y libpng-dev libgoogle-glog-dev
else
    sudo -E apt-add-repository -y "ppa:ubuntu-toolchain-r/test"
    curl -sSL "https://build.travis-ci.org/files/gpg/llvm-toolchain-trusty-7.asc" | sudo -E apt-key add - 
    echo "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-7 main" | sudo tee -a /etc/apt/sources.list >/dev/null
    sudo apt-get update
fi

# Install ninja and (newest version of) cmake through pip
sudo pip install ninja cmake
hash cmake ninja

# Build glow
GLOW_DIR=$PWD
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
elif [[ "$CIRCLE_JOB" == "RELEASE_WITH_EXPENSIVE_TESTS" ]]; then
    # Download the models and tell cmake where to find them.
    MODELS_DIR="$GLOW_DIR/downloaded_models"
    DOWNLOAD_EXE="python $GLOW_DIR/utils/download_datasets_and_models.py  -c resnet50 en2gr"
    mkdir $MODELS_DIR
    (
        cd $MODELS_DIR
        $DOWNLOAD_EXE
    )
    CMAKE_ARGS+=("-DGLOW_MODELS_DIR=$MODELS_DIR")
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=OFF")
    CMAKE_ARGS+=("-DGLOW_WITH_BUNDLES=ON")
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
elif [[ "$CIRCLE_JOB" == "COVERAGE" ]]; then
    sudo apt-get install wget 
    sudo apt-get install -y lcov
    sudo pip install awscli --upgrade 
    ../utils/install_protobuf.sh
    CC=gcc-5 CXX=g++-5 cmake -G Ninja \
          -DCMAKE_BUILD_TYPE=Debug -DGLOW_WITH_OPENCL=OFF -DGLOW_WITH_CPU=ON \
          -DLLVM_DIR=/usr/lib/llvm-7/cmake \
          -DGLOW_USE_COVERAGE=ON \
          ../
elif [[ "$CIRCLE_JOB" == "CHECK_CLANG_FORMAT" ]]; then
    sudo apt-get install -y clang-format-7
elif [[ "$CIRCLE_JOB" == "OPENCL" ]]; then
    install_pocl
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=ON")
else
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Debug")
    if [[ "${CIRCLE_JOB}" == "SHARED" ]]; then
        CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=ON")
    fi
fi

if [ "${CIRCLE_JOB}" != "COVERAGE" ] && [ "${CIRCLE_JOB}" != "CHECK_CLANG_FORMAT" ]; then
    cmake -GNinja ${CMAKE_ARGS[*]} ../
    ninja

    # Report sccache hit/miss stats
    if hash sccache 2>/dev/null; then
        sccache --show-stats
    fi
fi
