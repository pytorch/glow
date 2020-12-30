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
 * This op implements a pooling operator using L2 norm aggregation.
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

  if (numParams != 1)
    return false;

  if (inputs[0].dtype != outputs[0].dtype)
    return false;

  if (inputs[0].rank != 4 || inputs[0].rank != outputs[0].rank)
    return false;

  if (params[0].size != 2 || params[0].dtype != CustomOpDataType::DTIInt32 ||
      strcmp(params[0].name, "kernel_size"))
    return false;

  int32_t kernel_h = ((int32_t *)params[0].data)[0];
  int32_t kernel_w = ((int32_t *)params[0].data)[1];

  if (inputs[0].dims[2] % kernel_w != 0)
    return false;
  if (inputs[0].dims[3] % kernel_h != 0)
    return false;

  return true;
}

const char *customOpSelectImpl(CustomOpIOTensor *inputs, int32_t numInputs,
                               CustomOpIOTensor *outputs, int32_t numOutputs,
                               CustomOpParam *params, int32_t numParams,
                               const char *backend) {
  return "l2poolExecute";
}

bool customOpInferShape(CustomOpIOTensor *outputs, const int32_t numOutputs,
                        const CustomOpIOTensor *inputs, const int32_t numInputs,
                        const CustomOpParam *params, const int32_t numParams) {
  outputs[0].rank = inputs[0].rank;
  outputs[0].dims[0] = inputs[0].dims[0];
  outputs[0].dims[1] = inputs[0].dims[1];

  int32_t kernel_h = ((int32_t *)params[0].data)[0];
  int32_t kernel_w = ((int32_t *)params[0].data)[1];

  outputs[0].dims[2] = inputs[0].dims[2] / kernel_w;
  outputs[0].dims[3] = inputs[0].dims[3] / kernel_h;

  return true;
}

inline float getXYZW(CustomOpIOTensor tensor, int32_t x, int32_t y, int32_t z,
                     int32_t w) {
  int32_t *dims = tensor.dims;
  float *data = (float *)tensor.data;

  return data[(x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
              (z * dims[3]) + w];
}

inline void setXYZW(CustomOpIOTensor tensor, float val, int32_t x, int32_t y,
                    int32_t z, int32_t w) {
  int32_t *dims = tensor.dims;
  float *data = (float *)tensor.data;

  data[(x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
       (z * dims[3]) + w] = val;
}

extern "C" {

void l2poolExecute(CustomOpIOTensor *inputs, const int numInputs,
                   CustomOpIOTensor *outputs, const int numOutputs,
                   CustomOpParam *params, const int numParams) {
  int32_t kernel_h = ((int32_t *)params[0].data)[0];
  int32_t kernel_w = ((int32_t *)params[0].data)[1];

  int32_t *inDims = inputs[0].dims;
  int32_t *outDims = outputs[0].dims;

  for (int n = 0; n < inDims[0]; n++) {
    for (int c = 0; c < inDims[1]; c++) {
      for (int ow = 0; ow < outDims[2]; ow++) {
        for (int oh = 0; oh < outDims[2]; oh++) {
          float accum = 0;
          for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
              float inp = getXYZW(inputs[0], n, c, oh * kernel_h + kh,
                                  ow * kernel_w + kw);
              accum += inp * inp;
            }
          }

          setXYZW(outputs[0], sqrtf(accum), n, c, oh, ow);
        }
      }
    }
  }
}
}
