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
#ifndef GLOW_BASE_TENSOR_SERIALIZATION_H
#define GLOW_BASE_TENSOR_SERIALIZATION_H

#include "glow/Base/Tensor.h"

#include "llvm/ADT/StringRef.h"

namespace glow {

/// Dump the content of \p tensor to a binary file \p filename. You can choose
/// to dump also the tensor type in the file with the flag \p dumpType. The
/// binary representation of data is guaranteed to preserve data precision
/// (bit exactness) upon round-trips (dump/load).
void dumpTensorToBinaryFile(Tensor &tensor, llvm::StringRef filename,
                            bool dumpType = true);

/// Load the content of \p tensor from a binary file \p filename. You must
/// specify whether the file data contains also the tensor type with the flag
/// \p loadType such that the tensor type is set properly before the data is
/// loaded. The binary representation of data is guaranteed to preserve data
/// precision (bit exactness) upon round-trips (dump/load).
void loadTensorFromBinaryFile(Tensor &tensor, llvm::StringRef filename,
                              bool loadType = true);

/// Dump the content of \p tensor to a text file \p filename. You can choose to
/// dump also the tensor type in the file with the flag \p dumpType. The data
/// will be listed as a 1D array of values separated by comma (",") without
/// other formatting. The text representation of data is NOT guaranteed to
/// preserve data precision (bit exactness) upon round-trips (dump/load) and is
/// used mainly for human readability.
void dumpTensorToTextFile(Tensor &tensor, llvm::StringRef filename,
                          bool dumpType = true);

/// Load the content of \p tensor from a text file \p filename. You must
/// specify whether the file data contains also the tensor type with the flag
/// \p loadType such that the tensor type is set properly before the data is
/// loaded. The values in the text file are expected to be listed as a 1D array
/// of values, separated by comma (",") without other formatting. The text
/// representation of data is NOT guaranteed to preserve data precision (bit
/// exactness) upon round-trips (dump/load) but is used for human readability.
void loadTensorFromTextFile(Tensor &tensor, llvm::StringRef filename,
                            bool loadType = true);

} // namespace glow

#endif // GLOW_BASE_TENSOR_SERIALIZATION_H
