## The Glow Lexicon

### Definitions

### A

* ANN  
  Artificial Neural Network  
  A computational framework based on a synthetic construction of the
  biological neural network in brains.

* AOT  
  Ahead Of Time (Compilation)  
  A technique used with JIT where some code is compiled during the build and
  is executed directly rather than compiled when needed.

### B

* BC  
  Byte Code  
  A representation of the instruction stream encoded to be efficiently
  interpreted by an executor.

### C

* CNN  
  Convolutional Neural Network  
  A subtype of DNNs, using feed-forwarding.  They are most commonly applied to
  analysis of images.

* CVP  
  Constant Value Propagation  

### D

* DAG  
  Directed Acyclic Graph  

* DMA  
  Direct Memory Access  
  Copy data to/from memory without occupying CPU time

* DNM  
  Do Not Merge  

* DNN  
  Deep Neural Networks  
  A neural network with multiple layers between the input and output layers.
  These work well with linear and non-linear relationships.

### G

* GEMM  
  General Matrix Multiply  

* GRU  
  Gated Recurrent Unit  
  A gating mechanism for neural networks.  They are similar to LSTM but exhibit
  better performance characteristics on smaller data sets.

### I

* IR  
  Intermediate Representation  
  A representation of source code while being converted from the source language
  to the target language.

### J

* JIT  
  Just In Time (Compilation)  

### L

* LGTM  
  Looks Good To Me  
  Indicates that the reviewer believes your code to be correct and a positive
  change.  Used to indicate approval for merging the changes.

* LSTM  
  Long Short Term Memory  
  Units of RNNs consisting of a cell, input gate, output gate, and a forget
  gate.  It is useful to model memory, making it useful for classifying,
  processing, and predicting over temporal data.

### M

* MLP  
  Multi-Layer Perceptron  
  A class of feed-forward ANN, consisting of an input, hidden, and output
  layers.  Excluding the input, the nodes constitute a neuron in the ANN.  Using
  non-linear activation functions and back-propagation allows these networks to
  distinguish between data with non-linear relationships.

### N

* NFC  
  No Functional Change  

* NFCI  
   No Functional Change Intended  

### R

* RAUW  
  Replace All Uses With  
  A term commonly used in LLVM referring to a method that replaces all the uses
  of one value with another.  In Glow, see `NodeValue::replaceAllUsesOfWith`.

* ReLU  
  Rectified Linear Unit  
  A unit with a linear activation function in the context of a DNN.  These are
  common in computer vision and speech recognition applications.

* RNN  
  Recurrent Neural Network  
  A class of neural networks where the nodes form a DAG.  It is useful to model
  temporal dynamic behaviour, making it useful for speech and handwriting
  recognition.

### W

* WIP  
  Work In Progress  
  Used as a tag to commits that are not ready to be merged.  This is often used
  to mark patches that are being uploaded to test with CI or to get some initial
  feedback on the changes.
