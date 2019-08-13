# Loading PyTorch models in Glow
**Warning:** PyTorch integration is still under development and does not yet have as much support as Caffe2 model loading.

## About
Import PyTorch models to Glow via the [PyTorch JIT IR](https://pytorch.org/docs/master/jit.html).

See `glow/torch_glow/examples` for illustrative examples.


## Setup
* Follow directions in Building.md to make sure Glow can be built
* install [PyTorch](https://pytorch.org/) nightly 
* cd to `glow/torch_glow`

## Usage
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