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

#ifndef GLOW_GRAPH_CUSTOMOPUTILS_H
#define GLOW_GRAPH_CUSTOMOPUTILS_H

#include "glow/Base/Tensor.h"
#include "glow/Base/Type.h"
#include "glow/CustomOp/CustomOpTypes.h"
#include "glow/Graph/CustomOpData.h"

#include <vector>

namespace glow {

// TODO merge these two converter functions into single overloaded
// createIOTensors functions. Also change to accept and return vectors.
// QRANIUMSW-3876.

// Populates CustomOpIOTensor struct with dims and datatype information.
//
// Note: The returned struct should only be used for the duration of external
// API call. At the end, freeCustomOpIOTensor() must be called to avoid memory
// leaks. Copying and storing the struct elsewhere may lead to double free
// related crashes.
CustomOpIOTensor glowTypeToCustomOpIOTensor(const TypeRef type);

CustomOpIOTensor glowTensorToCustomOpIOTensor(Tensor *tensor);

// Creates and returns a CustomOpIOTensor with \p rank.
// Allocates memory for dims of size \p rank and initializes all
// dims to 0. Initializes dtype to \p dtype and data to \p data.
CustomOpIOTensor
initializeCustomOpIOTensor(const int32_t &rank = 0,
                           const CustomOpDataType &dtype = DTFloat32,
                           void *data = nullptr);

// Returns glow Type given a \p CustomOpIOTensor.
Type customOpIOTensorToglowType(const CustomOpIOTensor &iotensor);

// Calls free() on malloced iot.dims pointer.
void freeCustomOpIOTensor(CustomOpIOTensor &iot);

// Populates CustomOpParam struct with param data and returns a vector.
//
// Note: The returned vector should only be used for the duration of external
// API call. At the end, freeCustomOpParams() must be called to avoid memory
// leaks. Copying and storing the vector elsewhere may lead to double free
// related crashes.
std::vector<CustomOpParam> getCustomOpParams(CustomOpData &metadata);

// Calls free() on malloced param.data pointer.
void freeCustomOpParams(std::vector<CustomOpParam> &params);

// Returns string name of CustomOpDataType.
std::string toString(CustomOpDataType dtype);

} // end namespace glow

#endif
