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
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {
    "GatherWithInt32PartialTensors/0",
    "GatherWithInt64PartialTensors/0",
    "LayerNorm_Int8/0",
    "RepeatedSLSWithPartialTensors/0",
    "SigmoidSweep_Float16/0",
    "TanHSweep_Float16/0",
    "ConvertFrom_BoolTy_To_FloatTy/0",
    "ConvertFrom_BoolTy_To_Float16Ty/0",
    "EmbeddingBag_1D_Float_End_Offset_Partial/0",
    "EmbeddingBag_2D_Float_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset_Partial/0",
    "BBoxTransform_Float16/0",
};
