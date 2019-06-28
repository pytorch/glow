# Loading PyTorch models in Glow
**Warning:** PyTorch integration is still under development and does not yet have as much support as Caffe2 model loading.

## About
Import PyTorch models to Glow via the [PyTorch JIT IR](https://pytorch.org/docs/master/jit.html).

See `glow/torch_glow/examples` for illustrative examples.


## Setup
* Follow directions in Building.md to make sure Glow can be built
* install [PyTorch](https://pytorch.org/) nightly 
* install torchvision: `pip install torchvision`
* cd to `glow/torch_glow`

## Usage
### Run tests
* `python setup.py test`
  * use the `--cmake_prefix_path` flag to specify an llvm install location just like when building glow
  * to disable capturing test outputs, add `addopts = -s` to `[tool:pytest]` in setup.cfg
### Temporarily install while developing on Glow
* `python setup.py develop` 
  * verify with installation worked with `import torch_glow` in Python
### Install
* `python setup.py install` 
  * verify with installation worked with `import torch_glow` in Python