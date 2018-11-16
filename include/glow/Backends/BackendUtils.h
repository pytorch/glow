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
#ifndef GLOW_BACKENDS_BACKENDUTILS_H
#define GLOW_BACKENDS_BACKENDUTILS_H

#include "glow/Backends/CompiledFunction.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/IR/IR.h"

namespace glow {
/// At compile time condense constants to a single block of memory.
/// This allows the graph to go away after compile time.
/// Allocates a block of memory of size \p constantMaxSize then walks the given
/// function \p F and and copies weights to their address as specified by
/// offsets contained in \p symbolTable.
uint8_t *collectConstants(
    const IRFunction *F, uint64_t constantMaxSize,
    const std::unordered_map<std::string, runtime::RuntimeSymbolInfo>
        &symbolTable);
/// Helper function to retrieve offset for Value: \p v from \p symbolTable.
size_t
getValueOffset(Value *v,
               const std::unordered_map<std::string, runtime::RuntimeSymbolInfo>
                   &symbolTable);
/// Computes offsets and total allocation for Constants, Placeholders, and
/// Activations to build runtime symbol table. Returns RuntimeBundle.
runtime::RuntimeBundle
generateRuntimeBundle(const IRFunction &F, MemoryAllocator &constantAllocator,
                      MemoryAllocator &placeholderAllocator,
                      MemoryAllocator &activationsAllocator);
} // end namespace glow

#endif // GLOW_BACKENDS_BACKENDUTILS_H
