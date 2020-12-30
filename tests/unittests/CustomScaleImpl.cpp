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
 * This op implements a Scale function: y[c] = x[c] * scale + bias[c]
 *
 * This file can be compiled separate from glow and can be loaded using dlopen
 * Compilation command: (tried with gcc 5.5)
 * g++ -shared -std=c++11 -fPIC -o custom_scale.so CustomScaleImpl.cpp \
 *   -Iglow/include
 */

#include "CustomOpFunctions.h"
#include "CustomOpInterpreterInterface.h"
#include "CustomOpTypes.h"

#include "stdio.h"
#include "string.h"

bool customOpVerify(CustomOpIOTensor *inputs, int32_t numInputs,
                    CustomOpIOTensor *outputs, int32_t numOutputs,
                    CustomOpParam *params, int32_t numParams) {
  // Must have two params.
  if (numParams != 2)
    return false;

  // Op must have only 1 input and 1 output.
  if (numInputs != 1 || numOutputs != 1)
    return false;

  // Input and output must have the same data type.
  if (inputs[0].dtype != outputs[0].dtype)
    return false;

  // Input and output must have the same dimensions.
  if (inputs[0].rank != outputs[0].rank)
    return false;

  // Inout and output must have same dimensions.
  for (int i = 0; i < inputs[0].rank; i++) {
    if (inputs[0].dims[i] != outputs[0].dims[i])
      return false;
  }

  // Bias should match last dimension.
  if (params[1].size != outputs[0].dims[outputs[0].rank - 1])
    return false;

  return true;
}

const char *customOpSelectImpl(CustomOpIOTensor *inputs, int32_t numInputs,
                               CustomOpIOTensor *outputs, int32_t numOutputs,
                               CustomOpParam *params, int32_t numParams,
                               const char *backend) {
  if (strcmp(backend, "Interpreter") == 0)
    return "customScaleExecute";

  // This example supports interpreter only.
  // Developer can add other backends as it wishes
  return nullptr;
}

// Infer Output Shape.
bool customOpInferShape(CustomOpIOTensor *outputs, const int32_t numOutputs,
                        const CustomOpIOTensor *inputs, const int32_t numInputs,
                        const CustomOpParam *params, const int32_t numParams) {
  if (numInputs != 1 || numOutputs != 1)
    return false;

  // There is only 1 output.
  // Output has the same type as input.
  CustomOpIOTensor &out = outputs[0];
  out.rank = inputs[0].rank;
  for (int i = 0; i < inputs[0].rank; i++) {
    out.dims[i] = inputs[0].dims[i];
  }
  out.dtype = inputs[0].dtype;
  return true;
}

/// Kernel implementations must be under extern "C".
extern "C" {
/// Interpreter backend implementation of Custom ScaleOp.
void customScaleExecute(CustomOpIOTensor *inputs, int32_t numInputs,
                        CustomOpIOTensor *outputs, int32_t numOutputs,
                        CustomOpParam *params, int32_t numParams) {
  auto dimOut = outputs[0].dims;

  float scale = getFloatParam(params[0]);
  float *bias = getFloatVectorParam(params[1]);
  float *inT = (float *)inputs[0].data;
  float *outT = (float *)outputs[0].data;

  // Assuming CHW layout, compute scale, y[c] = x[c] * scale + bias[c].
  int x, y, z;
  for (z = 0; z < dimOut[2]; z++) {
    for (x = 0; x < dimOut[1]; x++) {
      for (y = 0; y < dimOut[0]; y++) {
        int loc = z * dimOut[1] * dimOut[0] + x * dimOut[0] + y;
        *(outT + loc) = (*(inT + loc)) * scale + bias[z];
      }
    }
  }
}
}