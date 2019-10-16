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
#ifndef GLOW_CONVERTER_FLOAT16CONVERTER_H
#define GLOW_CONVERTER_FLOAT16CONVERTER_H

namespace glow {

class Function;
struct PrecisionConfiguration;

/// Converts all inputs and outputs of a function \p F from Float to Float16,
/// and from UInt8FusedQTy to UInt8FusedFP16QTy, based on \p precConfig.
void convertFunctionToFloat16(Function *F,
                              const PrecisionConfiguration &precConfig);

} // namespace glow

#endif /* GLOW_CONVERTER_FLOAT16CONVERTER_H */
