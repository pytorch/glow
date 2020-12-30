/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _GLOW_GRAPH_CUSTOM_OP_INTERPRETER_INTERFACE_H
#define _GLOW_GRAPH_CUSTOM_OP_INTERPRETER_INTERFACE_H

#include <stddef.h>
#include <stdint.h>

#include "CustomOpTypes.h"

extern "C" {
// Signature for Interpreter execution kernels.
//
// Arguments:
//  - inputs: Array of CustomOpIOTensors containing the type, dimensions and
//            pointer to data of each input.
//            Input values can be read from the `data` pointer. Max size which
//            can be read is the size of input decided by `dims` and `dtype`.
//            Order of inputs is same as what was registered in the config.
//  - numInputs: number of inputs passed in `inputs` array.
//  - outputs: Array of CustomOpIOTensors containing the type, dimensions and
//             pointer to data of each output. Output values must be written to
//             the `data` pointer. Memory is already allocated according to size
//             of output decided by `dims` and `dtype`. Order of outputs is same
//             as what was registered in the config.
//  - numOutputs: number of outputs passed in `outputs` array.
//  - params: Array of CustomOpParams containing type and data for each param
//            for this op.
//            Order of params is same as what was registered in the config.
//  - numParams: number of params passed in `params` array.
typedef void (*customOpInterpreterKernel_t)(
    CustomOpIOTensor *inputs, int32_t numInputs, CustomOpIOTensor *outputs,
    int32_t numOutputs, CustomOpParam *params, int32_t numParams);

} // extern "C"
#endif
