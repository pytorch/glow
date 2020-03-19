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

/// Dump the data content of \p tensor to a raw-binary file without any
/// other meta information (data type and shape). The binary representation
/// of data is guaranteed to preserve data precision (bit exactness) upon
/// round-trips (dump/load).
void dumpToRawBinaryFile(Tensor &tensor, llvm::StringRef filename);

/// Load the data content of \p tensor from a raw-binary file. The tensor
/// configuration (data type and shape) must be set before loading the data
/// since the raw binary file is assumed to contain only the data. The binary
/// representation of data is guaranteed to preserve data precision (bit
/// exactness) upon round-trips (dump/load).
void loadFromRawBinaryFile(Tensor &tensor, llvm::StringRef filename);

/// Dump the data content of \p tensor to a raw-text file without any other
/// meta information (data type and shape). The data will be listed as a 1D
/// array of values separated by comma (",") without other formatting. The
/// text representation of data is NOT guaranteed to preserve data precision
/// (bit exactness) upon round-trips (dump/load) but is used for human
/// readability.
void dumpToRawTextFile(Tensor &tensor, llvm::StringRef filename);

/// Load the data content of \p tensor from a raw-text file. The tensor
/// configuration (data type and shape) must be set before loading the data
/// since the raw text file is assumed to contain only the data. The values
/// in the text file are expected to be listed as a 1D array of values,
/// separated by comma (",") without other formatting. The text representation
/// of data is NOT guaranteed to preserve data precision (bit exactness) upon
/// round-trips (dump/load) but is used for human readability.
void loadFromRawTextFile(Tensor &tensor, llvm::StringRef filename);

} // namespace glow

#endif // GLOW_BASE_TENSOR_SERIALIZATION_H
