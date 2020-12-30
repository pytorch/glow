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
 * This op copies nth input to nth output.
 */

#include "CustomOpFunctions.h"
#include "CustomOpInterpreterInterface.h"
#include "CustomOpTypes.h"

bool customOpVerify(CustomOpIOTensor *inputs, int32_t numInputs,
                    CustomOpIOTensor *outputs, int32_t numOutputs,
                    CustomOpParam *params, int32_t numParams) {
  // Must have 4 params.
  if (numParams != 4)
    return false;

  // Op must have same number of inputs and outputs.
  if (numInputs != numOutputs)
    return false;

  // Op must have 6 inputs and 6 outputs.
  if (numInputs != 6)
    return false;

  for (int32_t i = 0; i < numInputs; i++) {
    // Input and output must have the same data type.
    if (inputs[i].dtype != outputs[i].dtype)
      return false;

    // Input and output must have the same rank.
    if (inputs[i].rank != outputs[i].rank)
      return false;

    // Input and output must have same dimensions.
    for (int d = 0; d < inputs[i].rank; d++)
      if (inputs[i].dims[d] != outputs[i].dims[d])
        return false;
  }
  return true;
}

const char *customOpSelectImpl(CustomOpIOTensor *inputs, int32_t numInputs,
                               CustomOpIOTensor *outputs, int32_t numOutputs,
                               CustomOpParam *params, int32_t numParams,
                               const char *backend) {
  // Same function name is used for all backends.
  return "TestCustomOp";
}

// Infer Output Shape.
bool customOpInferShape(CustomOpIOTensor *outputs, const int32_t numOutputs,
                        const CustomOpIOTensor *inputs, const int32_t numInputs,
                        const CustomOpParam *params, const int32_t numParams) {
  if (numInputs != numOutputs)
    return false;

  for (int inId = 0; inId < numInputs; inId++) {
    // Output has the same type as input.
    CustomOpIOTensor &out = outputs[inId];
    out.rank = inputs[inId].rank;
    for (int i = 0; i < inputs[inId].rank; i++) {
      out.dims[i] = inputs[inId].dims[i];
    }
    out.dtype = inputs[inId].dtype;
  }
  return true;
}

/// Kernel implementations must be under extern "C".
extern "C" {
/// Interpreter backend implementation of Custom Copy.
void TestCustomOp(CustomOpIOTensor *inTensors, int32_t numInputs,
                  CustomOpIOTensor *outTensors, int32_t numOutputs,
                  CustomOpParam *params, int32_t numParams) {

  (void)numOutputs;
  (void)numParams;
  (void)params;

  for (int i = 0; i < numInputs; i++) {
    uint64_t size = inTensors[i].dims[0];
    switch (inTensors[i].dtype) {
    case CustomOpDataType::DTFloat32: {
      for (uint64_t j = 0; j < size; j++) {
        float *pIn = (float *)inTensors[i].data;
        float *pOut = (float *)outTensors[i].data;
        pOut[j] = pIn[j];
      }
      break;
    }
    case CustomOpDataType::DTFloat16: {
      for (uint64_t j = 0; j < size; j++) {
        float16_ty *pIn = (float16_ty *)inTensors[i].data;
        float16_ty *pOut = (float16_ty *)outTensors[i].data;
        pOut[j] = pIn[j];
      }
      break;
    }
    case CustomOpDataType::DTQInt8: {
      for (uint64_t j = 0; j < size; j++) {
        int8_t *pIn = (int8_t *)inTensors[i].data;
        int8_t *pOut = (int8_t *)outTensors[i].data;
        pOut[j] = pIn[j];
      }
      break;
    }
    case CustomOpDataType::DTQUInt8: {
      for (uint64_t j = 0; j < size; j++) {
        uint8_t *pIn = (uint8_t *)inTensors[i].data;
        uint8_t *pOut = (uint8_t *)outTensors[i].data;
        pOut[j] = pIn[j];
      }
      break;
    }
    case CustomOpDataType::DTIInt32: {
      for (uint64_t j = 0; j < size; j++) {
        int32_t *pIn = (int32_t *)inTensors[i].data;
        int32_t *pOut = (int32_t *)outTensors[i].data;
        pOut[j] = pIn[j];
      }
      break;
    }
    case CustomOpDataType::DTIInt64: {
      for (uint64_t j = 0; j < size; j++) {
        int64_t *pIn = (int64_t *)inTensors[i].data;
        int64_t *pOut = (int64_t *)outTensors[i].data;
        pOut[j] = pIn[j];
      }
      break;
    }
    default:
      break;
    }
  }
  return;
}
}