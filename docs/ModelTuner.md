## ModelTuner

This front end tool is used for tuning (calibrating) the quantization parameters of a model.
During the quantization flow, the model is first **profiled** by gathering the dynamic range (min/max)
and the histogram for each tensor in the graph. When the model is **compiled**, the quantization
parameters are chosen in such a way that, for the given profile, no saturation occurs. Although this
makes sense at first glance, there is actually a tradeoff when choosing the quantization parameters
for a given tensor: it might be beneficial overall if the quantization parameters are chosen such
that to provide a smaller quantization step (e.g. smaller **scale** parameter) which means a better
representation of most of the tensor values (the bulk of the histogram) at the expense of actually
saturating the extreme values (outliers).

The **model-tuner** front end tool is currently designed to work only with image classification models,
that is models which are provided with an input image and which compute an output vector of scores. The
requirements for this tool are:
- The model was already profiled using the **image-classifier** or **model-profiler** tool such that
a profile YAML file is available for the model. A best practice when profiling is to use a set of images
which includes one representative image for each class.
- A labeled tuning dataset is available which is required by the tool to optimize the accuracy. A best
practice is to use a significant dataset which should include at least 10's (or even 100's) of images
for each class. With more images the duration of the tuning will increase but with more images used the
result of the tuning procedure becomes more statistically relevant.

The **model-tuner** tool uses a brute force approach and has the following logic:
- It initially computes and displays the accuracy for the floating-point model and the quantized model
without tuning:
  - The accuracy of the floating-point model provides an approximate upper bound for the accuracy of the quantized model.
  - The accuracy of the quantized model provides the starting point of the accuracy before the tuning procedure.
- Iterates over all the tensors from the graph.
- For each tensor in the graph multiple iterations are run:
  - For each iteration a different quantization range is tried (tested) for that tensor.
  - For each iteration the accuracy of the quantized model is computed using the tuning dataset.
  - From all the tested quantization ranges, the range which maximizes the accuracy is kept.
- The expected behavior is that the accuracy of the quantized model will rise progressively because for
each tensor the quantization parameters which maximize the accuracy are kept and used in further iterations.
- There are situations when the tuning for a tensor stops prematurely in order to save time and speed up the tuning:
  - If iteration N+1 obtains exactly the same accuracy as iteration N then the tuning of that tensor
    is stopped and a message `accuracy not improved` is displayed in the console.
  - If during subsequent iterations for the same tensor the accuracy drops with at least a given delta (default is 5%)
    then the tuning is stopped for that tensor and a message `accuracy dropped more than "acc-drop-skip"` is displayed
    in the console. As the message suggests the value of the delta can be chosen with the `acc-drop-skip` argument.
  - If the tensor being tuned has all its values equal (minimum value equals the maximum value) then the
    tuning is skipped and a message `not required` is displayed in the console.
- At the end of the tuning procedure a tuned YAML profile file will be dumped by the tool. This new profile
file can be used further to compile and quantize the model (e.g. using the **model-compiler** tool) and
should provide a better numerical behavior (in terms of accuracy) than the initial non-tuned profile.

### Command line options

The specific command line options for running this tool are presented below. Apart from the specific
options, some generic options are used which are also used for the other front end tools (see the 
image-classifier documentation):
- options for specifying the model, the quantization options (schema, precision), the backend,
the image preprocessing options (layout, channel order, normalization).

```
model-tuner -backend=CPU -model=<model-path> <image-options> <quantization-options> -dataset-path=<dataset-folder>
-dataset-file=<dataset-file> -load-profile=<input-profile> -dump-tuned-profile=<tuned-profile>
```

where:
- `-backend=CPU` - This option specifies the backend used to run the model in order to compute the accuracy.
                   It is recommended to use the CPU backend which runs faster than other backends.
- `<image-options>` - The options used to pre-process the images before running the inference. These options
                      should be the same as the options used to obtain the initial profile (for example when
                      using the **image-classifier** tool).
- `<quantization-options>` - The quantization options used to quantize the model. These quantization options
                             should match the desired quantization parameters used when the model is finally
                             compiled/quantized (for example when using the **model-compiler** tool).
- `dataset-path` - The folder where the dataset files are located. The assumption is that all the dataset files
                   are located in the same directory.
- `dataset-file` - The path to the dataset description file which contains on each line a data path and integer
                   label separated by space (" ") or comma (","). The integer labels start with 0 (0,1,...).
  An example might look like this:
  ```
  image0.png 0 
  image1.png 13
  .............
  ```
  Another example might look like this:
  
  ```
  image0.png,0, 
  image1.png,13,
  ..............
  ```
- `load-profile` - The path of the input profile obtained initially which is loaded and tuned.
- `dump-tuned-profile` - The path where the final tuned profile is dumped.

More information can be acquired by typing the following command:
```
model-tuner -help
```

### Extra command line options

There are a couple of extra command line parameters which can be used to tweak the algorithm behavior:
- `max-iter-per-node` - The maximum number of tuning iterations per node/tensor (default is 3).
- `acc-drop-skip` - The accuracy drop for which the tuning of any node/tensor is skipped. The default value is 0.05 (5%).
- `target-accuracy` - The tuning procedure is stopped when the accuracy has reached or surpassed the given
                      value. A float value between 0.0 and 1.0 is expected. If not specified, the tuning will
                      run until completion.

### Command line output

When running this tool the console output might look like this:

```
Computing initial accuracy ... 
Initial accuracy: 75.3333 % (FLOAT)
Initial accuracy: 74.3333 % (QUANTIZED)
Target  accuracy: 80.0000 % (QUANTIZED)
Number of tensors: 35

[1/35] Tuning quantization for tensor "Conv_0__1:0"
  [1/3] Testing range = [-0.5000, 0.5000]
  Accuracy = 75.3333 %
  [2/3] Testing range = [-0.2500, 0.2500]
  Accuracy = 66.6667 %
  Tuning stopped for this tensor: accuracy dropped more than "acc-drop-skip"
Best accuracy : 75.3333 %
Iteration time: 5 seconds
Remaining time: 0 hours 2 minutes

[2/35] Tuning quantization for tensor "Conv_0__2:0"
  [1/3] Testing range = [-4.0598, 4.0598]
  Accuracy = 75.3333 %
  Tuning stopped for this tensor: accuracy not improved
Best accuracy : 75.3333 %
Iteration time: 3 seconds
Remaining time: 0 hours 1 minutes

..................................
..................................

[35/35] Tuning quantization for tensor "zero__3:0"
  Tuning skipped for this tensor: not required
Best accuracy : 75.6667 %
Iteration time: 3 seconds
Remaining time: 0 hours 0 minutes


Final accuracy: 75.6667 % (QUANTIZED)

Total time: 0 hours 2 minutes
```

Notes:
- The quantization tuning procedure is a long procedure: the order of magnitude of the time
required to run is similar to training. For this reason the tool also prints an estimated
remaining time for running the tuning (the estimation gets better after calibrating more nodes).
- When the estimated time for the tuning is too high, one might use a smaller tuning dataset.
