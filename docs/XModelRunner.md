## XModelRunner

Glow supplies a number of out-of-the-box model builders/runners, which include
the `ImageClassifier` and the `ModelLoader`. The former is tailored towards specific 
networks built for image classification, while the latter is a generic model loader (see
the corresponding documentation). 

`XModelRunner` is a generic model builder/runner that is able to consume any model --
either in the ONNX format, or as a Caffe2 protobuf -- and
either compile it (producing a bundle if requested, along with profiling when requested), 
or run inference on it. `XModelRunner`
is built along with the other models using the standard build instructions (see the 
building documentation). Below is a description of the command line options for `XModelRunner`.

### Command Line Options

In addition to all of the "standard" command line options (i.e. those that are common to
all model builders/runners, and can be obtained with the `-help` option), a few
options are supported by `XModelRunner`, some of which are mandatory and some are optional.

#### Required Options

| Option | Expected Values | Description | Required |
| :------ | :------ | :------ | :------ |
|`input-tensor-dims` | Comma-separated list of ints | Input tensor dimensions | Yes |
|`output-tensor-names` | Comma-separated list of strings (no spaces) | Output tensor names (can be more than one) | Yes |
|`write-output` | Boolean flag | Whether to write output to output files (only applicable when not building a bundle) | No (default is `False`) |

#### Expected Input

`XModelRunner` expects either a list of file names containing input (one input tensor per file), which are positional arguments, or a file name containing the list of file names that contain input (one file name per line), specified with the `input-file-list` option. Input specification is required when either profiling the network, or running inference (i.e. not building a bundle). Otherwise, input can be omitted.

#### Produced Output

When input files are specified, output is produced (saved into binary files) only when the `write-output` option is specified. In this case, the input file name acts as a base name for the output file name. The output file name is composed as `[input file name].out.dat`. When input is not specified and bundles are requested, the runner produces bundles in the specified output directory. Also, `write-output` may be omitted when profiling the network. 

### Technical Notes

1. The runner currently supports only one (named) input tensor, but can work with multiple (named) output tensors.
2. The runner does not currently support streaming input, but this feature is planned.

### A Note on the Initial Contribution

The initial contribution of this runner (as well as the corresponding documentation) was made as part of the open source contribution initiative by the XPERI Corporation (xperi.com). 