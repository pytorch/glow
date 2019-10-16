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
#ifndef GLOW_BASE_IO_H
#define GLOW_BASE_IO_H

#include "glow/Base/Tensor.h"

#include "llvm/ADT/StringRef.h"

namespace glow {

/// Write Tensor data \p T to a file named \p filename.
///
/// NB: This function is primarily a debugging aid, not a serialization format.
/// It stores only the binary data of the tensor (not the dimensions), and
/// writes tensors assuming host endianness.
void writeToFile(const Tensor &T, llvm::StringRef filename);

/// Read Tensor data \p T from a file named \p filename.
///
/// NB: This function is primarily a debugging aid, not a serialization format.
/// It reads only the binary data of the tensor (the dimensions must be known
/// through other means), and reads tensors assuming host endianness.
void readFromFile(Tensor &T, llvm::StringRef filename);

} // namespace glow

#endif // GLOW_BASE_IO_H
