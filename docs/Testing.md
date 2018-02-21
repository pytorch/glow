## Testing the Glow compiler

The Glow test suite contains four major categories: unit tests, regression
tests, example programs, and the model loader.  Unit tests are the small tests
that stress specific parts of the compiler.  These tests are added to the
compiler when developing a feature. For example, we train a number of small
network and perform a gradient check on the operators.  We also compile networks
to IR and look for specific patterns.  Regression tests are tests that are added
when we fix bugs.  Both regression tests and feature tests are found under the
"test/" directory. To run the feature and regression tests run "ninja test".

## Example test suites.

We rely on external test suites to test the compiler. We use the data sets
CIFAR10 and MNIST (located in the "example/" directory) to test the correctness
of the whole system.  The script under 'utils/' download and extract the data
set.

## Model Loader

We test the correctness of the Glow implementation by loading Caffe2 models and
executing them end-to-end. The program 'loader' loads a model, a png file, and
runs a single pass of inference. If everything goes right the output of the
program is identical to the output of the Caffe2 model. Unfortunately, the caffe
model does not describe what the input format should be. Should the pixels be
between zero and one, or negative 128 to positive 128? The user needs to be
aware of these things when running the models. The script in the directory
'utils/' downloads a number of pre-trained networks that we can use for testing.

The Glow build scripts copy a few sample images and a run script that tests the
loader program. The script can be executed with the command:

  ```
  build$./tests/images/run.sh
  ```

## Caffe2 Models

The `loader` program loads pre-trained models from Caffe2. These pre-trained
models are downloaded via the `download_caffe2_models.sh` script located in
`utils/`. We also provide other scripts in `utils/` for working with Caffe2
models.

### Train and Save Caffe2 Models

The `caffe2_train_and_dump_pb.py` script in `utils/` allows the user to define
their own models and input training set in Caffe2, and then dumps the network
and weights to protobuf files (the network structure in `predict_net.pb/pbtxt`
and the pre-trained weights in `init_net.pb`). Right now it trains either LeNet
on MNIST; an MLP is also available and can be used by setting `USE_LENET_MODEL =
False`. This script is heavily based on the MNIST.py tutorial from Caffe2.

### Run the pre-trained Caffe2 Models using Caffe2

The `caffe2_pb_runner.py` script in `utils/` loads and runs a pre-trained model
using the protobuf files saved using `caffe2_train_and_dump_pb.py`. This can be
used to compare the output from Glow to Caffe2. Its usage is similar to running
the `loader`, which is found in the `run.sh` script in `tests/images/`. For
example, the following command will run the pre-trained resnet50 model using
Caffe2:

```
python caffe2_pb_runner.py -i [location_of_image] -d resnet50
```
