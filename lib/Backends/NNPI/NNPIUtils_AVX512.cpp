/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#include "NNPIUtils.h"
#include <immintrin.h>

void convertI64toI32_AVX512(int64_t const *i64Data, int32_t *i32Data,
                            uint32_t elements) {
  const __mmask8 masks[9] = {
      0b0, 0b1, 0b11, 0b111, 0b1111, 0b11111, 0b111111, 0b1111111, 0b11111111,
  };
  constexpr uint32_t vecSize = (sizeof(__m512i) / sizeof(int64_t));
  const uint32_t fullIterations = (elements / vecSize);
  const uint32_t tailElements = (elements % vecSize);

  for (uint32_t i = 0; i < fullIterations; i++) {
    __m512i i64vec = _mm512_maskz_loadu_epi64(masks[vecSize], i64Data);
    _mm512_mask_cvtepi64_storeu_epi32(i32Data, masks[vecSize], i64vec);
    i64Data += vecSize;
    i32Data += vecSize;
  }
  if (tailElements > 0) {
    __m512i i64vec = _mm512_maskz_loadu_epi64(masks[tailElements], i64Data);
    _mm512_mask_cvtepi64_storeu_epi32(i32Data, masks[tailElements], i64vec);
  }
}
