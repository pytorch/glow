## Overview of Network Debugger tool and classes

### Introduction

This document describes the motivation behind the Network Debugger tool and
how it can be used.

Network Debugger was created to provide a way to find an exact broken layer
when running a network on a backend. This is useful when
developing a new backend or compiling a new network that was never compiled before
on a particular backend.

### Usage:
  * Logic for detection of a broken layer is factored into Comparator classes (see section below) which allows for different usage modes:

  * Use the network debugger tool:
    * This tool loads a model from a protobuf file, runs conversions  to fp16, if needed, and optimizations
      on the network then passes it to the comparator class to find a broken layer:
      ```
      Module mod;
       // Load Module to mod.
       IntermediateLayerComparator netCompare(mod, "Interpreter", testBackend,numericCmpThreshold);
      netCompare.verify();
      ```
      A sample run line for the tool:
      ```
      bin/network-debugger --model onnxifi_function_0.zip --inputs input_0.onnx -glow_global_fp16    -backend=Interpreter
      ```
      ```
      --model specifies the protobuf file for the network.
      --input is a sample input ot the network in ONNX format.
      ```
    * Output.
      * The tool will print out what layer being verified; when it
        starts and when it finishes verifying the layer:

    ```
    [NetworkComparator.cpp:137] Verifying layer: layer_1	Type: SparseLengthsSum
    [NetworkComparator.cpp:151] DONE Verifying layer: layer_1

    [NetworkComparator.cpp:137] Verifying layer: layer_2	Type: Add
    [NetworkComparator.cpp:151] DONE Verifying layer: layer_2
    ```
      * If the layer is detected to generates the wrong result the tool will print the following:
    ```
    [NetworkComparator.cpp:137] Verifying layer: layer_1	Type: SparseLengthsSum
                                Error at output index 4, got 0.779 expected 0.564
                                Results differ
                                dumping tensors.
    [NetworkComparator.cpp:151] DONE Verifying layer: layer_1

    ```
       To help the user later reproduce the failure, the input and output tensors
       of the failing layer are dumped out to separate files. The name of the dumped file
       is "Input_#InputName_#LayerName". For example, a SparseLengthsSum
       node  has inputs called "Data", "Indices" and "Lengths", if the layer is called
       "layer_1" like in the above example the dumped files will be called:

       ```
       intput_Data_layer_1
       intput_Indices_layer_1
       intput_Lengths_layer_1
       ```

       The tool will also dump a reference output that is provided by the reference backend:
       If the output is called "Result" like in SparseLengthsSum the dumped file:
       ```
       ref_Result_layer1
       ```
       The user can later create a test case and feed these inputs to the test and compare the results against the reference.

  * Using the classes standalone fashion

    Instead of using the network-debugger tool the user can instantiate an object of the comparator while working on
    their application and just pass the module. The sample
    code above shows an example of that.

### How broken layers are found

The core logic for finding a broken layer is provided in the classes in `lib/Graph/NetworkComparator.cpp`. The comparator class receives a module and a test backend,
the provided module gets compiled and run on a reference backend (Interpreter) and compares the results to these of running on the provided test backend. 

 These classes provide different modes of operation:

#### `IntermediateLayerComparator`
* This class first tests layer all at once to find suspicious
layers that are different between the reference and test backend runs. It works as follows:
   1) Instrument the input network with Save nodes to capture the results at run time.
   2) Compile and run the instrumented network on the reference backend and the test backend.
   3) Compare the intermediate results from between the reference backend and the test backend. If no errors are found the comparator just returns success.
   4) If errors were detected; Start a single layer test debug.
      1) For every node in the network, create a new network that only has that one node.
      2) For the inputs of the network, feed in placeholders resulting from the reference run. This will guarantee errors that might be caused by previous layers don't propagate to the layer being tested and allows detection of only the broken layers.
      3) Compare the output of the one layer network with the outputs from the reference backend run. If the results don't match the mark the layer as broken.

#### `RecursiveLayerComparator`
  * This class tests layers by creating a subnet of the original
    graph for every layer instead of instrumenting all the layers
    at once.
    It works as follows:
    *  **For every layer in the network**:
    1) Visit the inputs of the layer recursively until reaching the input
   placeholders; add the visited notes to create a subnet.
    1) Add Save node(s) to the layer output(s).
    2) Run the network on the the test backend and reference backend.
    3) If the results differ mark the layer as broken.

  * This tester is slower than the Intermediate tester. However, it
     has a smaller memory footprint since it doesn't need to save
    intermediate results of all layers.


#### Future work
  * Add binary logging of tensors, e.g ONNX. This is best
    added to the ```Tensor::dump() ``` method overrides and called later in  ```NetworkComparatorBase::dumpTensors()```. If we add that the user can later
    use ``` glow::loadTensor()``` to load it in their test case.
  * Make testing more autonomous; instead of just dumping tensors for errant layer
   provide a way to dump the network (made of one layer) to a file to later be loaded and tested.
