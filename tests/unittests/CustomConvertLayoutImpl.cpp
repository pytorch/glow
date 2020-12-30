/*
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
 *
 *
 * CustomOp implementation source
 *
 * This op takes in shape of a tensor and
 * returns shape after converting layout.
 * Layout can be of types : NHWC, NCHW.
 * It has 1 input and 2 outputs.
 * It has 2 params.
 *  1. inLayout (Scalar): can be "NHWC" , "NCHW"
 *  2. outLayout (Scalar): can be "NHWC" , "NCHW"
 */

#include <string.h>

#include "CustomOpFunctions.h"
#include "CustomOpInterpreterInterface.h"
#include "CustomOpTypes.h"

static bool isValidLayout(const char *layout) {
  if (strcmp(layout, "NHWC") && strcmp(layout, "NCHW")) {
    return false;
  }
  return true;
}

bool customOpVerify(CustomOpIOTensor *inputs, int32_t numInputs,
                    CustomOpIOTensor *outputs, int32_t numOutputs,
                    CustomOpParam *params, int32_t numParams) {

  // Op must have 1 input and  1 output.
  if (numInputs != 1 || numOutputs != 1)
    return false;

  // input and output rank must be 4.
  if (inputs[0].rank != 1 || outputs[0].rank != 1)
    return false;

  // Must have 2 params.
  if (numParams != 2)
    return false;

  CustomOpParam &param0 = params[0];
  CustomOpParam &param1 = params[1];

  // Check params names are valid.
  if (strcmp(param0.name, "inLayout") || strcmp(param1.name, "outLayout")) {
    return false;
  }

  // param0 and param1 are scalars.
  if (param0.size != 0 || param1.size != 0)
    return false;

  char *inLayVal = getCharParam(param0);
  char *outLayVal = getCharParam(param1);

  // Check valid layout values.
  if (!isValidLayout(inLayVal) || !isValidLayout(outLayVal)) {
    return false;
  }

  return true;
}

const char *customOpSelectImpl(CustomOpIOTensor *inputs, int32_t numInputs,
                               CustomOpIOTensor *outputs, int32_t numOutputs,
                               CustomOpParam *params, int32_t numParams,
                               const char *backend) {
  // Same function name is used for all backends.
  return "customConvertLayout";
}

// Infer Output Shape.
bool customOpInferShape(CustomOpIOTensor *outputs, const int32_t numOutputs,
                        const CustomOpIOTensor *inputs, const int32_t numInputs,
                        const CustomOpParam *params, const int32_t numParams) {
  if (numInputs != 1 || numOutputs != 1)
    return false;

  // output has the same shape as input.
  CustomOpIOTensor &out = outputs[0];
  out.rank = inputs[0].rank;
  out.dtype = inputs[0].dtype;
  for (int j = 0; j < inputs[0].rank; j++) {
    out.dims[j] = inputs[0].dims[j];
  }

  return true;
}

/// Kernel implementations must be under extern "C".
extern "C" {
/// Interpreter backend implementation of Custom Copy.
void customConvertLayout(CustomOpIOTensor *inTensors, int32_t numInputs,
                         CustomOpIOTensor *outTensors, int32_t numOutputs,
                         CustomOpParam *params, int32_t numParams) {

  (void)numInputs;
  (void)numParams;

  CustomOpParam &param0 = params[0];
  CustomOpParam &param1 = params[1];
  char *inLay = getCharParam(param0);
  char *outLay = getCharParam(param1);

  // Assign same values to output as input.
  CustomOpIOTensor &out0 = outTensors[0];
  int32_t numElements = inTensors[0].dims[0];

  int32_t *inTensor = (int32_t *)inTensors[0].data;
  int32_t *outTensor = (int32_t *)out0.data;
  for (int i = 0; i < numElements; i++) {
    outTensor[i] = inTensor[i];
  }

  if (strcmp(inLay, outLay) == 0)
    return;

  if (strcmp(inLay, "NHWC") == 0) {
    // convert to NCHW
    outTensor[0] = inTensor[0];
    outTensor[1] = inTensor[3];
    outTensor[2] = inTensor[1];
    outTensor[3] = inTensor[2];
  } else {
    // convert to NHWC
    outTensor[0] = inTensor[0];
    outTensor[1] = inTensor[2];
    outTensor[2] = inTensor[3];
    outTensor[3] = inTensor[1];
  }

  return;
}

} // extern "C"