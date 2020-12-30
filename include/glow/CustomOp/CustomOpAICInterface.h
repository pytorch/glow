
/**
 * Copyright (c) 2019-present, Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 *
 */

#ifndef GLOW_BACKENDS_AIC_AICCUSTOMOPINTERFACE_H
#define GLOW_BACKENDS_AIC_AICCUSTOMOPINTERFACE_H

// CustomOp specific header(s).
#include "CustomOpTypes.h"

extern "C" {
typedef int32_t *CustomOpParamHandle;
typedef int32_t *CustomOpIOHandle;

// Arguments:
//  - outH: Array of CustomOpIOTensors containing the type, dimensions and
//          pointer to data of each output. Output values must be written to
//          the `data` pointer. Memory is already allocated according to size
//          of output decided by `dims` and `dtype`. Order of outputs is same
//          as what was registered in the config.
//  - numOutputs: number of outputs passed in `outputs` array.
//  - inH: Array of CustomOpIOTensors containing the type, dimensions and
//         pointer to data of each input.
//         Input values can be read from the `data` pointer. Max size which
//         can be read is the size of input decided by `dims` and `dtype`.
//         Order of inputs is same as what was registered in the config.
//  - numInputs: number of inputs passed in `inputs` array.
//  - paramHandle: Array of CustomOpParams containing type and data for each
//                 param for this op.
//                 Order of params is same as what was registered in the config.
//  - numParams: number of params passed in `params` array.

// Signature for AIC execution kernels which use common API (useCommonAPI).
typedef void (*customOpAICKernel_t)(CustomOpIOHandle outH, int32_t numOutputs,
                                    CustomOpIOHandle inH, int32_t numInputs,
                                    CustomOpParamHandle paramHandle,
                                    int32_t numParams);

} // extern "C"
#endif