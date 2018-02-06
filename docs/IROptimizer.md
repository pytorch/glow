## IR Optimizer

This document describes the optimizations performed by the IR optimizer and some
implementation details.

### Overview

The IR optimizer performs a number of optimizations on the IR representation of a
neural network model.

The IR optimizations have two major objectives. One is to improve the performance
of training and inference steps. The other one is to reduce the memory consumption
during the execution of neural network models. 

It is worth mentioning that performing optimizations reducing memory consumption
is easier at the IR level, because memory allocations and deallocations are
explicitly represented in the IR, whereas they are not explicit in the graph
representation.

### Set of supported optimizations

Below you can see the list of currently supported optimizations:

  * peephole optimizations

    These are small, local optimizations that look for specific sequences of
    instructions and replace them with more efficient sequences of instructions.

  * dead store elimination

    This optimization removes stores into weights or allocations if it can
    prove that the results of these stores are never going to be used.

  * deallocations hoisting

    This optimization tries to place the buffer deallocation instructions right
    after the last use of a buffer. Doing so reduces the lifetime of the buffer
    and makes the freed memory available for the allocation of other buffers.
    It improves the memory consumption.

  * allocations sinking

    This optimization tries to place the buffer allocation instructions right
    before the first use of a buffer. Doing so reduces the lifetime of the
    buffer and makes the unused memory available for the allocation of other
    buffers. It improves the memory consumption.

  * dead allocations removal

    This optimization finds and removes allocations that are just allocated and
    deallocated, but are never used. Such situations may happen e.g. after
    performing other allocations. Performing this optimization improves the
    memory consumption.

  * making weights constant

    This optimization marks weights that are never mutated as constant. This may
    allow for placing such weights in a read only memory segments and share it
    between simultaneous executions of the same neural network model.

  * sharing of buffers

    The purpose of this optimization is to reduce the memory usage by reusing
    the memory buffers as much as possible. The overall idea is that it is fine
    to combine storage for two live intervals if they do not overlap. Typically,
    two live intervals are considered as candidates for sharing if they occur
    in the same instruction.
