#!/bin/bash

# This script setup an working environemnt for running glow tests and gtest driver.
# By default, we run with the enabled CPU backend and disabled OpenCL backend.
set -ex

export MAX_JOBS=8
if [ "${CIRCLE_JOB}" != "COVERAGE" ]; then
    if hash sccache 2>/dev/null; then
        SCCACHE_BIN_DIR="/tmp/sccache"
        mkdir -p "$SCCACHE_BIN_DIR"
        for compiler in cc c++ gcc g++ x86_64-linux-gnu-gcc; do
            (
                echo "#!/bin/sh"
                echo "exec $(which sccache) $(which $compiler) \"\$@\""
            ) > "$SCCACHE_BIN_DIR/$compiler"
            chmod +x "$SCCACHE_BIN_DIR/$compiler"
        done
        export PATH="$SCCACHE_BIN_DIR:$PATH"
    fi
fi

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

GLOW_DEPS="libpng-dev libgoogle-glog-dev libboost-all-dev libdouble-conversion-dev libgflags-dev libjemalloc-dev libpthread-stubs0-dev"

if [ "${CIRCLE_JOB}" == "CHECK_CLANG_AND_PEP8_FORMAT" ]; then
    sudo -E apt-add-repository -y "ppa:ubuntu-toolchain-r/test"
    curl -sSL "https://build.travis-ci.org/files/gpg/llvm-toolchain-trusty-7.asc" | sudo -E apt-key add -
    echo "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-7 main" | sudo tee -a /etc/apt/sources.list >/dev/null
    sudo apt-get update
elif [ "${CIRCLE_JOB}" == "PYTORCH" ]; then
    # Install Glow dependencies
    sudo apt-get update
    sudo apt-get install -y llvm-7
    # Redirect clang
    sudo ln -s /usr/bin/clang-7 /usr/bin/clang
    sudo ln -s /usr/bin/clang++-7 /usr/bin/clang++
    sudo ln -s /usr/bin/llvm-symbolizer-7 /usr/bin/llvm-symbolizer
    sudo ln -s /usr/bin/llvm-config-7 /usr/bin/llvm-config-7.0

    sudo apt-get install -y ${GLOW_DEPS}
else
    # Install Glow dependencies
    sudo apt-get update

    # Redirect clang
    sudo ln -s /usr/bin/clang-8 /usr/bin/clang
    sudo ln -s /usr/bin/clang++-8 /usr/bin/clang++
    sudo ln -s /usr/bin/llvm-symbolizer-8 /usr/bin/llvm-symbolizer
    sudo ln -s /usr/bin/llvm-config-8 /usr/bin/llvm-config-8.0

    sudo apt-get install -y ${GLOW_DEPS}
fi

# Since we are using llvm-7 in these two branches, we cannot use pip install cmake
if [ "${CIRCLE_JOB}" != "PYTORCH" ] && [ "${CIRCLE_JOB}" != "CHECK_CLANG_AND_PEP8_FORMAT" ]; then
	sudo pip install cmake
else
	sudo apt-get install cmake
fi

# Install ninja, (newest version of) autopep8 through pip
sudo pip install ninja autopep8
hash cmake ninja

# Build glow
GLOW_DIR=$PWD
cd ${GLOW_DIR}
mkdir build && cd build

CMAKE_ARGS=()
# PYTORCH build are directly using the sccache wrappers
if [[ "${CIRCLE_JOB}" != "PYTORCH" ]]; then
    CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=/usr/bin/clang++-8")
    CMAKE_ARGS+=("-DCMAKE_C_COMPILER=/usr/bin/clang-8")
    CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER_LAUNCHER=sccache")
    CMAKE_ARGS+=("-DCMAKE_C_COMPILER_LAUNCHER=sccache")
fi

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
elif [[ "$CIRCLE_JOB" == "CHECK_CLANG_AND_PEP8_FORMAT" ]]; then
    sudo apt-get install -y clang-format-7
elif [[ "$CIRCLE_JOB" == "PYTORCH" ]]; then
    # Build PyTorch
    cd /tmp
    python3.6 -m virtualenv venv
    source venv/bin/activate
    git clone https://github.com/pytorch/pytorch.git --recursive --depth 1
    cd pytorch
    pip install -r requirements.txt
    BUILD_BINARY=OFF BUILD_TEST=0 BUILD_CAFFE2_OPS=0 USE_FBGEMM=ON python setup.py install
    cd ${GLOW_DIR}
    cd build
elif [[ "$CIRCLE_JOB" == "OPENCL" ]]; then
    install_pocl
    CMAKE_ARGS+=("-DGLOW_WITH_OPENCL=ON")
elif [[ "$CIRCLE_JOB" == "32B_DIM_T" ]]; then
    install_pocl
    CMAKE_ARGS+=("-DTENSOR_DIMS_32_BITS=ON -DGLOW_WITH_OPENCL=ON")
else
    CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Debug")
    if [[ "${CIRCLE_JOB}" == "SHARED" ]]; then
        CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=ON")
    fi
fi

if [ "${CIRCLE_JOB}" != "COVERAGE" ] && [ "${CIRCLE_JOB}" != "CHECK_CLANG_AND_PEP8_FORMAT" ] && [ "${CIRCLE_JOB}" != "PYTORCH" ]; then
    cmake -GNinja ${CMAKE_ARGS[*]} ../
    ninja
fi

# Report sccache hit/miss stats
if hash sccache 2>/dev/null; then
    sccache --show-stats
fi
