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
#ifndef GLOW_OPTIMIZER_OPTIMIZER_H
#define GLOW_OPTIMIZER_OPTIMIZER_H

#include "glow/Quantization/Quantization.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace glow {

class IRFunction;
class Function;
class Backend;
class Module;
class Context;
class Placeholder;

enum class CompilationMode {
  Train, /// Compile the graph in preperation for training.
  Infer, /// Compile the graph for inference. Notice that this operation
         /// changes the graph in a way that is not reversible.
};

/// Perform optimizations on the IR representation.
void optimize(IRFunction &M, bool shouldShareBuffers);
/// Perform optimizations on the graph representation.
void optimize(Function *F, CompilationMode mode);

/// Lower the high-level neural network nodes found in \p F into low-level
/// linear algebra operators. \p B can prevent lowering of a node via \ref
/// Backend::shouldLower(). If \p loweredMap is not a nullptr, then \p
/// loweredMap will contain a mapping from output names of the nodes found and
/// lowered in \p F to the output names of the nodes they were lowered from
/// along with the NodeKind.
void lower(Function *F, const Backend &B, LoweredInfoMap *loweredMap = nullptr);

/// Lower the high-level neural network nodes found in \p F into low-level
/// linear algebra operators. All nodes are fully lowered. If \p loweredMap is
/// not a nullptr, then \p loweredMap will contain a mapping from output names
/// of the nodes found and lowered in \p F to the output names of the nodes they
/// were lowered from along with the NodeKind.
void lowerEverything(Function *F, LoweredInfoMap *loweredMap = nullptr);

/// Dead code elimination.
void DCE(Function *F);

/// Convert placeholders in Module \p M to constants based on the values in \p
/// ctx.  Do not convert any placeholders explicitly listed in \p vars.
void convertPlaceholdersToConstants(Function *F, const Context &ctx,
                                    llvm::ArrayRef<Placeholder *> vars);

/// Instrument function \p F by inserting quantization profile nodes for
/// capturing stats for quantization. The nodes will refer to tensors allocate
/// in in context \p ctx. The new quantized function is called \p
/// newFuncName. If no name is given the method will generate a name.  \returns
/// a new function with the added quantization nodes.
Function *profileQuantization(Context &ctx, Function *F,
                              llvm::StringRef newFuncName = "");

/// Helper to generate and optimize IR from given Function \p F. \p
/// shouldShareBuffers signifies whether to use the share buffers optimization.
/// Backend /p B is used to allow for custom lowering from Node to
/// Instruction IR.
std::unique_ptr<IRFunction> generateAndOptimizeIR(Function *F, const Backend &B,
                                                  bool shouldShareBuffers);

} // namespace glow

#endif // GLOW_OPTIMIZER_OPTIMIZER_H
