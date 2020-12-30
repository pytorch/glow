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

#ifndef _GLOW_GRAPH_CUSTOM_OP_FUNCTIONS_H
#define _GLOW_GRAPH_CUSTOM_OP_FUNCTIONS_H

#include <stddef.h>
#include <stdint.h>

#include "CustomOpTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Signature for the verification function.
// Arguments are same as execution kernel function.
// The `data` field in CustomOpIOTensor is kept as nullptr.
// Verification logic should not rely on the data of inputs, and should also
// not write any data to outputs.
bool customOpVerify(CustomOpIOTensor *inputs, int32_t numInputs,
                    CustomOpIOTensor *outputs, int32_t numOutputs,
                    CustomOpParam *params, int32_t numParams);

typedef bool (*customOpVerify_t)(CustomOpIOTensor *inputs, int32_t numInputs,
                                 CustomOpIOTensor *outputs, int32_t numOutputs,
                                 CustomOpParam *params, int32_t numParams);

// Signature for the selection function.
// Arguments are same as verify function, only `backend` is passed extra.
// The `data` field in CustomOpIOTensor is kept as nullptr.
// Selection logic should not rely on the data of inputs, and should also
// not write any data to outputs.
// 'backend` contains the name of the backend for which flavour selection needs
// to be done.
const char *customOpSelectImpl(CustomOpIOTensor *inputs, int32_t numInputs,
                               CustomOpIOTensor *outputs, int32_t numOutputs,
                               CustomOpParam *params, int32_t numParams,
                               const char *backend);

typedef const char *(*customOpSelectImpl_t)(
    CustomOpIOTensor *inputs, int32_t numInputs, CustomOpIOTensor *outputs,
    int32_t numOutputs, CustomOpParam *params, int32_t numParams,
    const char *backend);

// Signature for Shape inference function.
// Arguments:
//  - outputs: Array of CustomOpIOTensor initialized as follows:
//    1. rank -> maxDims as registered for the output
//    2. dims -> array of maxDims initialized to all 0s
//    3. dtype -> DTFloat32
//    4. data -> nullptr
//    The inferred data type, rank and dims for outputs are assigned by this
//    function by modifying \p outputs.
//    If the inferred rank of the output is less than maxDims, then the first
//    $rank number of dims should be updated with actual values. If the inferred
//    rank of the output is greater than maxDims, then false should be returned
//    to indicate error. The `data` field in CustomOpIOTensor should not be
//    updated.
//  - numOutputs: number of outputs registered by the user and passed in \p
//    outputs.
//  - inputs: Array of CustomOpIOTensors containing the type, rank and
//    dimensions of each input.
//  - numInputs: number of inputs passed in \p inputs array.
//  - params: Array of CustomOpParams containing type and data for each param
//    for this op.
//  - numParams: number of params passed in \p params array.
// Order of inputs, params and outputs is same as what was registered.
// Returns: true, to indicate success. false, otherwise.
bool customOpInferShape(CustomOpIOTensor *outputs, const int32_t numOutputs,
                        const CustomOpIOTensor *inputs, const int32_t numInputs,
                        const CustomOpParam *params, const int32_t numParams);

typedef bool (*customOpInferShape_t)(CustomOpIOTensor *outputs,
                                     const int32_t numOutputs,
                                     const CustomOpIOTensor *inputs,
                                     const int32_t numInputs,
                                     const CustomOpParam *params,
                                     const int32_t numParams);

} // extern "C"

#endif // _GLOW_GRAPH_CUSTOM_OP_FUNCTIONS_H
