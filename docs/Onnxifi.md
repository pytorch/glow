# Getting Started with ONNXIFI

[ONNXIFI](https://github.com/onnx/onnx/blob/master/docs/ONNXIFI.md) is an
interface that allows PyTorch/Caffe2 to use Glow as an execution backend.
Right now, [FOXI](https://github.com/houseroad/foxi) (i.e., ONNXIFI with
Facebook Extension) is used to support more features.

Setting up a Caffe2 environment to use Glow via ONNXIFI can be tricky the first
time. The steps in this walkthrough show how to build all the required
components from source.  They have been tested using Ubuntu 16.04 and Python
3.6.

**(TODO: You'll need to apt-get install some packages to get everything to
build, but I don't remember all of them.)**

While not required, we recommend using a Python virtual environment, to avoid
quirks associated with the system Python environment.

```bash
cd $SRC
mkdir virtualenv && cd virtualenv && virtualenv . && source bin/activate
pip install pyyaml pytest future
```

Get PyTorch, ONNX and Glow.

```bash
cd $SRC
git clone https://github.com/pytorch/pytorch.git
git clone https://github.com/onnx/onnx.git
git clone https://github.com/pytorch/glow.git
```

Build and install ONNX.

```bash
cd $SRC/onnx
git submodule update --init --recursive
python setup.py install
```

Build Glow and copy/rename its ONNXIFI library.

```bash
cd $SRC/glow
git submodule update --init --recursive
mkdir build && cd build && cmake -G Ninja -DGLOW_WITH_OPENCL=OFF .. && ninja all
mkdir -p $HOME/lib
cp lib/Onnxifi/libonnxifi-glow.so $HOME/lib/libonnxifi.so
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
```

Build and install PyTorch.

```bash
cd $SRC/pytorch
git submodule update --init --recursive
python setup.py install
```

Remove the dummy libonnxifi.so that PyTorch installs. **(TODO: Needing to do
this is weird.  Can we avoid it?)**

```bash
find $SRC/pytorch/build -name libonnxifi.so -delete
find $SRC/virtualenv -name libonnxifi.so -delete
```

Test your installation.

```bash
cd $SRC/pytorch/build
sed -i.orig 's/@unittest.skip/#@unittest.skip/' caffe2/python/onnx/test_onnxifi.py
python -m pytest -s caffe2/python/onnx/test_onnxifi.py
```
