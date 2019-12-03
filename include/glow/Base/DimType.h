/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#ifndef GLOW_DIMENSION_TYPE_H
#define GLOW_DIMENSION_TYPE_H

#include <cinttypes>
#include <cstddef>
#include <cstdint>

namespace glow {

#ifdef DIM_T_32
// The dimensions of Tensors are stored with this type. Note: The same
// fixed width type is used both in the host and the possible co-processors
// handling tensor data. The bit width should be chosen carefully for maximum
// data level parallel execution.
using dim_t = uint32_t;
using sdim_t = int32_t;

#define PRIdDIM PRId32
#define PRIuDIM PRIu32

#else // DIM_T_32
using dim_t = uint64_t;
using sdim_t = int64_t;

#define PRIdDIM PRId64
#define PRIuDIM PRIu64

#endif // DIM_T_32

constexpr unsigned DIM_T_BITWIDTH = sizeof(dim_t) * 8;
constexpr unsigned SDIM_T_BITWIDTH = sizeof(sdim_t) * 8;

} // namespace glow

#endif
