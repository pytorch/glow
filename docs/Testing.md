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

#### Calculating Top-1 and Top-5 Accuracy

The script `imagenet_topk_accuracy_driver.py` located in the `utils/` directory
can be used to calculate Top-1 and Top-5 accuracy. It can be run via a command
like the following:

```
python utils/imagenet_topk_accuracy_driver.py --batch-size=10 --validation-images-dir=${PATH_TO_IMAGES} --image-classifier-cmd="${PATH_TO_IMAGE_CLASSIFIER_BINARY} -image-mode=0to1 -m=${PATH_TO_RESNET50_PROTOS_DIR} -model-input-name=gpu_0/data -backend=CPU -topk=5 -"
```

Note that the `--image-classifier-cmd` must include `-topk=5` for printing the
Top-5 labels, and `-` to run in streaming mode.

The script expects the directory passed in via `--validation-images-dir` to
contain subdirectories alphabetically ordered in order of increasing label. For
example, for Imagenet with 1000 labels, subdirectories could be listed as
`label000/, label001/, ... , label999/`, where `label000/` contains all images
that should be classified with label 0.

The script can be used to resize and center crop images to 224x224 via
`--resize-input-images`. This resize and center cropping can be done by itself
via `--only-resize-and-save`, improving execution time of calculating Top-k
accuracy more than once (this saves the processed images to
`validation_images_dir/processed/`).

### Text Translation

The program `text-translator` loads a text translation model, reads a line from
stdin in a source language, and then prints the translation to the command line
in the destination language. The text translation model should be specified by a
directory via `-m`, containing the source and destination dictionaries
(`src_dictionary.txt` and `dst_dictionary.txt`), as well as the protobuf files
for the model. A backend can be optionally specified, just like for the
`image-classifier`.

```
$ ./bin/text-translator -m en2gr -backend=CPU

Enter a sentence in English to translate to German: My favorite sport is basketball .
mein Lieblingssport ist Basketball .
```

This program expects a sequence-to-sequence model with beam search. Because Glow
currently does not support models that contain control flow (e.g. the
[RecurrentNetwork operator from
Caffe2](https://caffe2.ai/docs/operators-catalogue.html#recurrentnetwork)), the
input model must be unrolled to some maximum input and output length. These can
be specified on the command line via `-max-input-len` and
`-max-output-len`. Additionally, the beam search size can be specified via
`-beam-size`. The default options for the `text-translator` match those for the
en2gr model currently downloaded via `utils/download_datasets_and_models.py`
(`-max-input-len=10`, `-max-output-len=14`, `-beam-size=6`).

## Caffe2 and ONNX Models

Model loader programs (e.g. `image-classifier` and `text-translator`) load
pre-trained models from protobuf file (either Caffe2 or ONNX). These pre-trained
models are downloaded via `download_datasets_and_models.py` script located in `utils/`.

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

## Integrated Testing

Glow also comes with tests integrated with the build environment for our command
line tools. We run those tests as part of our continuous integration (CI).

Run them as part of your local build using the following
```bash
cmake -G Ninja <glow_src>  -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/usr/local/opt/llvm         \
      -DGLOW_MODELS_DIR=<downloaded_c2_models>
```
Followed by
```bash
ninja check_expensive
```

Note: `ninja check_expensive` runs all of the tests that `ninja check` runs plus
any tests that have been marked as EXPENSIVE using add_glow_test(EXPENSIVE ...)
such as the integration tests.

Note: The difference between `ninja test` and `ninja check` is that
`ninja check` makes sure the build dependencies are current before
running the tests.
