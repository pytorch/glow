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

#include "Shaders.h"

const unsigned char SHADER_CODE[] = {
  #include "glow/kernels_cl.inc"
  0x00
};

const unsigned char FWD_CONV_CODE[] = {
  #include "glow/kernels_fwd_conv_cl.inc"
  0x00
};

const unsigned char FWD_CONV_QUANTIZED_CODE[] = {
  #include "glow/kernels_fwd_quantized_conv_cl.inc"
  0x00
};
