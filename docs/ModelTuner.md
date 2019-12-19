## ModelTuner

This front end tool is used for tuning (calibrating) the quantization parameters of a model.
During the quantization flow, the model is first profiled by gathering the dynamic range (min/max)
for each tensor in the graph. Next, the quantization parameters are chosen in such a way that, for
the given profile, no saturation occurs. Although this makes sense at first glance, there
is actually a tradeoff when choosing the quantization parameters for a given tensor: it might be
be beneficial overall if the quantization parameters are chosen such that to provide a smaller
quantization step (e.g. smaller **scale** parameter) which means a better representation of most
of the tensor values (the bulk of the histogram) at the expense of actually saturating the extreme
values (outliers).

This tool is basically tuning the quantization parameters by using the following simple algorithm:
- For each node in the graph, try different quantization parameters in the vicinity of the initially
chosen values (right after the profiling). For example, this is done by successively dividing the
**scale** parameter by 2 for a maximum of 3 iterations.
- For each tested quantization parameters, keep the ones which provide the best accuracy with respect
to a given dataset.

### Command line options

The specific command line options for running this tool are presented below. Apart from the specific
options, some generic options are used which are also used for the other front end tools (see the 
image-classifier documentation):
- options for specifying the model, the quantization options (schema, precision), the backend,
the image preprocessing options (layout, channel order, normalization).

```
model-tuner -model=<model-path> <image-options> <quantization-options> -dataset-path=<dataset-folder>
-dataset-file=<dataset-file> -load-profile=<input-profile> -dump-tuned-profile=<tuned-profile>
```

where:
- *dataset-path* - the folder where the dataset files are located. The assumption is that all the dataset files
                    are located in the same directory.
- *dataset-file* - the path to the dataset description file which contains on each line a data path and integer
                   label separated by space (" ") or comma (","). The integer labels start with 0 (0,1,...).
                   An example might look like this:
                     image0.png 0 
                     image1.png 13
                     .............
                   Another example might look like this:
                     image0.png,0, 
                     image1.png,13,
                     ..............
- *load-profile* - the path of the input profile which is loaded and tuned.
- *dump-tuned-profile* - the path where the tuned profile is written.

More information can be acquired by typing the following command:
```
model-tuner -help
```

### Extra command line options

There are a couple of extra command line parameters which can be used to tweak the algorithm behavior:
- *target-accuracy* - The tuning procedure is stopped when the accuracy has reached or surpassed the given
                      value. A float value between 0.0 and 1.0 is expected. If not specified, the tuning will
                      run until completion.
- *max-iter-per-node* - The maximum number of tuning iterations per node (default is 3).
- *acc-drop-skip* - The accuracy drop for which the tuning of any node is skipped. The default value is 0.05 (5%).

### Command line output

When running this tool the console output will might look like this:

```
Computing initial accuracy ... 
Initial accuracy: 81.0180 %
Number of nodes: 277
Target accuracy: 100.0000 %

[1/277] Tuning node "broadcast_B_tile0_save__1:0"
  [1/3] Testing scale = 0.00195
  Accuracy = 81.0180 %
  Tunning stopped for this node (no effect)
Best accuracy : 81.0180 %
Iteration time: 34 seconds
Remaining time: 2 hours 36 minutes

[2/277] Tuning node "W52__1:0"
  [1/3] Testing scale = 0.06250
  Accuracy = 81.4422 %
  [2/3] Testing scale = 0.03125
  Accuracy = 79.0032 %
  [3/3] Testing scale = 0.01562
  Accuracy = 67.1262 %
Best accuracy : 81.4422 %
Iteration time: 68 seconds
Remaining time: 5 hours 11 minutes

..................................
..................................

[277/277] Tuning node "W42__1:0"
  [1/3] Testing scale = 0.01562
  Accuracy = 90.2439 %
  Tunning stopped for this node
Best accuracy : 97.9852 %
Iteration time: 66 seconds
Remaining time: 0 hours 0 minutes


Final accuracy: 97.9852 %

Total time: 5 hours 6 minutes
```

Notes:
- The quantization tuning procedure is a long procedure: the order of magnitude of the time
required to run is similar to training. For example, the model used for tuning in the above example
is a medium size model (e.g. similar to a Mobile Net with a scale factor of 0.5). For this reason
the tool also prints an estimated remaining time for running the tuning (the estimation gets
better after calibrating more nodes).
- When the estimated time for the tuning is too much, one might use a smaller tuning dataset.
