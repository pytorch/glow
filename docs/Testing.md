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

We test the correctness of the Glow implementation by loading Caffe2 and ONNX
models and executing them end-to-end.

### Image Classification

The program `image-classifier` loads a model, a png file, and runs a single pass
of inference. If everything goes right the output of the program is identical to
the output of the original (Caffe2 or ONNX) model. Unfortunately, the models do
not usually describe what the input format should be. Should the pixels be
between zero and one, or negative 128 to positive 128? The user needs to be
aware of these things when running the models. The script in the directory
'utils/' downloads a number of pre-trained networks that we can use for testing.

The Glow build scripts copy a few sample images and a run script that tests the
`image-classifier` program. The script can be executed with the command:

  ```
  build$./tests/images/run.sh
  ```

### Text Translation

The program `text-translator` loads a text translation model, reads a line from
stdin in a source language, and then prints the translation to the command line
in the destination language. The text translation model should be specified by a
directory via `-m`, containing the source and destination dictionaries
(`src_dictionary.txt` and `dst_dictionary.txt`), as well as the protobuf files
for the model. A backend can be optionally specified, just like for the
`image-classifier`.

```
$ ./bin/text-translator -m en2gr -cpu

Enter a sentence in English to translate to German: My favorite sport is basketball .
mein Lieblingssport ist Basketball .
```

This program expects a sequence-to-sequence model with beam search. Because Glow
currently does not support models that contain control flow (e.g. the
[RecurrentNetwork operator from
Caffe2](https://caffe2.ai/docs/operators-catalogue.html#recurrentnetwork)), the
input model must be unrolled to some maximum input and output length. These can
be specified on the command line via `-min_output_len` and
`-max_output_len`. Additionally, the beam search size can be specified via
`-beam_size`. The default options for the `text-translator` match those for the
en2gr model currently downloaded via `utils/download_caffe2_models.sh`
(`-max_input_len=10`, `-max_output_len=14`, `-beam_size=6`).

## Caffe2 and ONNX Models

Model loader programs (e.g. `image-classifier` and `text-translator`) load
pre-trained models from protobuf file (either Caffe2 or ONNX). These pre-trained
models are downloaded via `download_caffe2_models.sh` and
`download_onnx_models.sh` scripts located in `utils/`.

There is a more general way to run a pre-trained model, not related to images.
The `model-runner` program loads and runs a self-contained model, i.e. a model,
which has all its inputs initialized inside itself and does not ask for user's
input.

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
the `image-classifier`, which is found in the `run.sh` script in `tests/images/`. For
example, the following command will run the pre-trained resnet50 model using
Caffe2:

```
python caffe2_pb_runner.py -i [location_of_image] -d resnet50
```
