# Loading PyTorch models in Glow
**Warning:** PyTorch integration is still under development and does not yet have as much support as Caffe2 model loading.

## About
Import PyTorch models to Glow via the [PyTorch JIT IR](https://pytorch.org/docs/master/jit.html).

See `glow/torch_glow/examples` and `glow/torch_glow/tests` for illustrative examples.


## Setup
*See below for more detailed instructions for running on Linux*
* Follow directions in Building.md to make sure Glow can be built
* Build PyTorch source. torch_glow may work with PyTorch nightly build but could be out of sync so it's better to build from source.
* cd to `glow/torch_glow`
* python setup.py test --run_cmake


## Building on Linux
These instructions detail roughly how to setup and run torch_glow similar to how the
Glow/PyTorch CI is setup.
### Install Ubuntu 16.04
  * download [Ubuntu 16.04](https://releases.ubuntu.com/16.04/)
  * If running in a VM, ensure 30GB of disk are available
### Install [base dependencies](https://github.com/pytorch/pytorch/blob/master/.circleci/docker/common/install_base.sh#L23)
```
apt-get update

apt-get install -y --no-install-recommends \
  asciidoc \
  docbook-xml \
  docbook-xsl \
  xsltproc \
  gfortran \
  cmake=3.5* \
  apt-transport-https \
  autoconf \
  automake \
  build-essential \
  ca-certificates \
  curl \
  git \
  libatlas-base-dev \
  libc6-dbg \
  libiomp-dev \
  libyaml-dev \
  libz-dev \
  libjpeg-dev \
  libasound2-dev \
  libsndfile-dev \
  python \
  python-dev \
  python-setuptools \
  python-wheel \
  software-properties-common \
  sudo \
  wget \
  vim
```

### Install [clang-7](https://github.com/pytorch/pytorch/blob/master/.circleci/docker/common/install_clang.sh#L7)
```
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main"

apt-get update
apt-get install -y clang-7
apt-get install -y llvm-7

update-alternatives --install /usr/bin/clang clang /usr/bin/clang-7 50
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-7 50

clang_lib=("/usr/lib/llvm-7/lib/clang/"*"/lib/linux")
echo "$clang_lib" > /etc/ld.so.conf.d/clang.conf
ldconfig
```

### Install [protobuf](https://github.com/pytorch/pytorch/blob/master/.circleci/docker/common/install_protobuf.sh#L22)
```
apt-get update
apt-get install -y --no-install-recommends libprotobuf-dev protobuf-compiler
```

### Install [anaconda python 3.6](https://github.com/pytorch/pytorch/blob/master/.circleci/docker/common/install_conda.sh) and packages
```
sudo bash
BASE_URL="https://repo.anaconda.com/miniconda"
CONDA_FILE="Miniconda3-latest-Linux-x86_64.sh"
mkdir /opt/conda
pushd /tmp
wget -q "${BASE_URL}/${CONDA_FILE}"
chmod +x "${CONDA_FILE}"
sudo ./"${CONDA_FILE}" -b -f -p "/opt/conda"
popd
sudo sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
export PATH="/opt/conda/bin:$PATH"
pushd /opt/conda
conda update -n base conda
conda install python=3.6
conda install -q -y python=3.6 numpy pyyaml mkl mkl-include setuptools cffi typing future six
conda install -q -y python=3.6 nnpack -c killeent
pip install --progress-bar off pytest scipy==1.1.0 scikit-learn==0.20.3 scikit-image librosa>=0.6.2 psutil numba==0.46.0 llvmlite==0.30.0
popd
```

### Install [vision deps](https://github.com/pytorch/pytorch/blob/master/.circleci/docker/common/install_vision.sh#L22)
```
  apt-get update
  apt-get install -y --no-install-recommends libopencv-dev libavcodec-dev
```

### Clone Glow repo
```
git clone https://github.com/pytorch/glow.git
```

### Walk through steps of Glow's [.circleci/build.sh](https://github.com/pytorch/glow/blob/master/.circleci/build.sh#L65) including building PyTorch
```
# Install Glow dependencies, some of these are redundant
sudo apt-get update
sudo apt-get install -y llvm-7

# Redirect clang
sudo ln -s /usr/bin/clang-7 /usr/bin/clang
sudo ln -s /usr/bin/clang++-7 /usr/bin/clang++
sudo ln -s /usr/bin/llvm-symbolizer-7 /usr/bin/llvm-symbolizer
sudo ln -s /usr/bin/llvm-config-7 /usr/bin/llvm-config-7.0

GLOW_DEPS="libpng-dev libgoogle-glog-dev libboost-all-dev libdouble-conversion-dev libgflags-dev libjemalloc-dev libpthread-stubs0-dev libevent-dev libssl-dev"

apt-get install -y ${GLOW_DEPS}
install_fmt

# Build and Install fmt
git clone https://github.com/fmtlib/fmt
pushd fmt
mkdir build
cd build
cmake ..
make -j`nproc`
sudo make install
popd

# install other deps
which python # should be using conda python 3.6
pip install ninja autopep8 pytest-xdist

# Build and Install PyTorch
pushd /tmp
pip install virtualenv
python -m virtualenv venv
source venv/bin/activate
git clone https://github.com/pytorch/pytorch.git --recursive --depth 1
pushd pytorch
pip install -r requirements.txt
BUILD_BINARY=OFF BUILD_TEST=0 BUILD_CAFFE2_OPS=0 BUILD_CAFFE2=ON USE_FBGEMM=ON python setup.py install
popd
popd
```

### Walk through steps of Glow's [.circleci/test.sh](https://github.com/pytorch/glow/blob/master/.circleci/test.sh#L49) including building PyTorch
```
cd glow/torch_glow
python setup.py test --run_cmake
```


## torch_glow usage
### Run tests
* `python setup.py test`
### Temporarily install while developing on Glow
* `python setup.py develop`
  * verify with installation worked with `import torch_glow` in Python
### Install
* `python setup.py install`
  * verify with installation worked with `import torch_glow` in Python

## Tips
* Use the `--run_cmake` flag to force rerun cmake
* Use the `--cmake_prefix_path` flag to specify an llvm install location just like when building glow
* To disable capturing test outputs and print a lot more test details, add `addopts = -s` to `[tool:pytest]` in setup.cfg
