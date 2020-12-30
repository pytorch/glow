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
 * This op implements a HardSigmoid function: y = max(0, min(1, alpha*x + beta))
 *
 * This file can be compiled separate from glow and can be loaded using dlopen
 * Compilation command: (tried with gcc 5.5)
 * g++ -shared -std=c++11 -fPIC -o custom_relu.so CustomReluImpl.cpp \
 *   -Iglow/include
 */

#include "CustomOpFunctions.h"
#include "CustomOpInterpreterInterface.h"
#include "CustomOpTypes.h"

#include <algorithm>
#include <string.h>

bool customOpVerify(CustomOpIOTensor *inputs, int32_t numInputs,
                    CustomOpIOTensor *outputs, int32_t numOutputs,
                    CustomOpParam *params, int32_t numParams) {

  // Must have two params.
  if (numParams < 2)
    return false;

  CustomOpParam &param0 = params[0];
  CustomOpParam &param1 = params[1];

  // Check params names are valid.
  if (strcmp(param0.name, "alpha") || strcmp(param1.name, "beta"))
    return false;

  // Op must have only 1 input and 1 output.
  if (numInputs != 1 || numOutputs != 1)
    return false;

  // Input and Output must have the same data type.
  if (inputs[0].dtype != outputs[0].dtype)
    return false;

  // Input and Output must have the same dimensions.
  if (inputs[0].rank != outputs[0].rank)
    return false;

  for (int i = 0; i < inputs[0].rank; i++) {
    if (inputs[0].dims[i] != outputs[0].dims[i])
      return false;
  }

  return true;
}

const char *customOpSelectImpl(CustomOpIOTensor *inputs, int32_t numInputs,
                               CustomOpIOTensor *outputs, int32_t numOutputs,
                               CustomOpParam *params, int32_t numParams,
                               const char *backend) {

  // Interpreter implementation is in the same file (see below).
  if (strcmp(backend, "Interpreter") == 0)
    return "customReluExecute";

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
void customReluExecute(CustomOpIOTensor *inputs, int32_t numInputs,
                       CustomOpIOTensor *outputs, int32_t numOutputs,
                       CustomOpParam *params, int32_t numParams) {
  (void)numInputs;
  (void)numOutputs;
  (void)numParams;

  int32_t *inDims = inputs[0].dims;
  int32_t inRank = inputs[0].rank;
  // output dims and rank should be verified to be same.

  float alpha = getFloatParam(params[0]);
  float beta = getFloatParam(params[1]);

  int numElements = 1;
  for (int i = 0; i < inRank; i++) {
    numElements *= inDims[i];
  }

  float *inT = (float *)inputs[0].data;
  float *outT = (float *)outputs[0].data;

  // Relu implementation using std::max.
  for (int i = 0; i < numElements; i++) {
    outT[i] = std::max<float>(0, std::min(1.0f, alpha * inT[i] + beta));
  }
}
} // extern "C"
