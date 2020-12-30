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

#ifndef _GLOW_GRAPH_CUSTOM_OP_TYPES_H
#define _GLOW_GRAPH_CUSTOM_OP_TYPES_H

#include <stddef.h>
#include <stdint.h>

/// Half precision float can be used with clang++ compiler.
#if defined(__clang__) && (__clang__ == 1)
typedef __fp16 float16_ty;
#endif

// Datatype for Custom Ops Params and Tensors.
enum CustomOpDataType {
  // 32-bit float type (float)
  DTFloat32,
  // 16-bit float type (half, fp16)
  DTFloat16,
  // 16-bit float type (bfloat16)
  DTBFloat16,
  // 8-bit quantized type (int8_t)
  DTQInt8,
  // unsigned 8-bit quantized type (uint8_t)
  DTQUInt8,
  // 16-bit quantized type (int16_t)
  DTQInt16,
  // 32-bit quantized type (int32_t)
  DTQInt32,
  // 32-bit index type (int32_t)
  DTIInt32,
  // 64-bit index type (int64_t)
  DTIInt64,
  // Bool type (bool)
  DTBool,
  // String type (string)
  DTString,
};

// Holds data and type information of Tensor.
typedef struct {
  // Pointer to the data of this tensor.
  // Set to nullptr for APIs which don't require data (eg. verification
  // function).
  void *data;

  // Array containing dimensions.
  int32_t *dims;
  // Number of dimensions.
  int32_t rank;

  // Data type of a single element in this tensor.
  CustomOpDataType dtype;

  // Quantization params.
  float scale = 1.0f;
  float offset = 0.0f;
} CustomOpIOTensor;

// Holds data and type information of Param.
typedef struct {
  // Contains the data of this param. Points to single value for scalar params,
  // and array for vector params. Should not be modified.
  void *data;

  // Data type of single element.
  CustomOpDataType dtype;

  // String containing name registered by config.
  const char *name;

  // Number of elements in data.
  // Scalars will have size = 0.
  int32_t size;
} CustomOpParam;

inline float getFloatParam(CustomOpParam p) { return *((float *)p.data); }
inline float *getFloatVectorParam(CustomOpParam p) { return ((float *)p.data); }
inline int getIntParam(CustomOpParam p) { return *((int *)p.data); }
inline int *getIntVectorParam(CustomOpParam p) { return ((int *)p.data); }
inline char *getCharParam(CustomOpParam p) { return ((char *)p.data); }

#endif // _GLOW_GRAPH_CUSTOM_OP_TYPES_H
