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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "model.h"

#define FLATBUFFERS_LOCALE_INDEPENDENT 0
#include "flatbuffers/flexbuffers.h"

//===----------------------------------------------------------------------===//
//                                   Bundle
//===----------------------------------------------------------------------===//
/// Statically allocate memory for constant weights and initialize.
GLOW_MEM_ALIGN(MODEL_MEM_ALIGN)
uint8_t constantWeight[MODEL_CONSTANT_MEM_SIZE] = {
#include "model.weights.txt"
};

/// Statically allocate memory for mutable weights (input/output data).
GLOW_MEM_ALIGN(MODEL_MEM_ALIGN)
uint8_t mutableWeight[MODEL_MUTABLE_MEM_SIZE];

/// Statically allocate memory for activations (intermediate data).
GLOW_MEM_ALIGN(MODEL_MEM_ALIGN)
uint8_t activations[MODEL_ACTIVATIONS_MEM_SIZE];

/// Bundle input data absolute address.
uint8_t *inputAddr = GLOW_GET_ADDR(mutableWeight, MODEL_input);

/// Bundle output data absolute address.
uint8_t *outputAddr = GLOW_GET_ADDR(mutableWeight, MODEL_output);

/// TFLite custom operator handler.
extern "C" {
void glow_tflite_custom_operator(const char *type, const uint8_t *optsAddr,
                                 int optsSize, int opInp, int opOut,
                                 uint8_t **opAddr, int *opSize) {
  printf("TFLite Custom Operator\n");
  printf("Type: %s\n", type);
  // Print raw options.
  printf("Options (raw):\n");
  printf("  Size (uint8) = %d\n", optsSize);
  printf("  Vals (uint8) = ");
  for (int idx = 0; idx < optsSize; idx++) {
    printf("%d, ", optsAddr[idx]);
  }
  printf("\n");
  // Print flexbuffer options.
  flexbuffers::Map map = flexbuffers::GetRoot(optsAddr, optsSize).AsMap();
  printf("Options (flexbuffer):\n");
  printf("  int_field = %d\n", map["int_field"].AsInt32());
  printf("  string_field = %s\n", map["string_field"].AsString().c_str());
  // Print input operands assuming all are float.
  for (int inpIdx = 0; inpIdx < opInp; inpIdx++) {
    printf("Input[%d]:\n", inpIdx);
    int opIdx = inpIdx;
    int inpSize = opSize[opIdx] / sizeof(float);
    float *inpAddr = (float *)(opAddr[opIdx]);
    printf("  Size (f32) = %d\n", inpSize);
    printf("  Vals (f32) = ");
    for (int idx = 0; idx < inpSize; idx++) {
      printf("%f, ", inpAddr[idx]);
    }
    printf("\n");
  }
  // Print output operands assuming all are float.
  for (int outIdx = 0; outIdx < opOut; outIdx++) {
    printf("Output[%d]:\n", outIdx);
    int opIdx = opInp + outIdx;
    int outSize = opSize[opIdx] / sizeof(float);
    float *outAddr = (float *)(opAddr[opIdx]);
    printf("  Size (f32) = %d\n", outSize);
    printf("  Vals (f32) = ");
    for (int idx = 0; idx < outSize; idx++) {
      printf("%f, ", outAddr[idx]);
    }
    printf("\n");
  }
}
}

int main() {

  // Initialize input and output to check proper data access in the handler.
  float *modelInput = (float *)inputAddr;
  modelInput[0] = 1;
  float *modelOutput = (float *)outputAddr;
  modelOutput[0] = 1;
  modelOutput[1] = 2;
  modelOutput[2] = 3;

  // Run the model.
  int errCode = model(constantWeight, mutableWeight, activations);
  if (errCode != GLOW_SUCCESS) {
    printf("Error running bundle: error code %d\n", errCode);
  }

  return 0;
}
