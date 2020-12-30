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
 * This op implements a scaled Tanh function: y = amplitude * tanh( scale * x)
 */

#include "CustomOpFunctions.h"
#include "CustomOpInterpreterInterface.h"
#include "CustomOpTypes.h"

#include "math.h"
#include "string.h"

bool customOpVerify(CustomOpIOTensor *inputs, const int numInputs,
                    CustomOpIOTensor *outputs, const int numOutputs,
                    CustomOpParam *params, const int numParams) {
  if (numInputs != 1 || numInputs != numOutputs)
    return false;

  if (numParams != 2)
    return false;

  if (inputs[0].dtype != outputs[0].dtype)
    return false;

  if (inputs[0].rank != outputs[0].rank)
    return false;

  for (int i = 0; i < inputs[0].rank; i++)
    if (inputs[0].dims[i] != outputs[0].dims[i])
      return false;

  if (params[0].size != 0 || params[0].dtype != CustomOpDataType::DTFloat32 ||
      strcmp(params[0].name, "amplitude"))
    return false;

  if (params[1].size != 0 || params[1].dtype != CustomOpDataType::DTFloat32 ||
      strcmp(params[1].name, "scale"))
    return false;

  return true;
}

const char *customOpSelectImpl(CustomOpIOTensor *inputs, int32_t numInputs,
                               CustomOpIOTensor *outputs, int32_t numOutputs,
                               CustomOpParam *params, int32_t numParams,
                               const char *backend) {
  return "scaledTanhExecute";
}

bool customOpInferShape(CustomOpIOTensor *outputs, const int32_t numOutputs,
                        const CustomOpIOTensor *inputs, const int32_t numInputs,
                        const CustomOpParam *params, const int32_t numParams) {
  outputs[0].rank = inputs[0].rank;
  for (int i = 0; i < inputs[0].rank; i++)
    outputs[0].dims[i] = inputs[0].dims[i];

  return true;
}

extern "C" {

void scaledTanhExecute(CustomOpIOTensor *inputs, const int numInputs,
                       CustomOpIOTensor *outputs, const int numOutputs,
                       CustomOpParam *params, const int numParams) {
  float *inputData = (float *)inputs[0].data;
  float *outputData = (float *)outputs[0].data;

  float amplitude = *((float *)params[0].data);
  float scale = *((float *)params[1].data);

  int numElements = 1;
  for (int i = 0; i < inputs[0].rank; i++)
    numElements *= inputs[0].dims[i];

  for (int i = 0; i < numElements; i++)
    outputData[i] = amplitude * tanh(scale * inputData[i]);
}
}
