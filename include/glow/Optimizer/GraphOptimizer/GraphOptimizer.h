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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_H

#include "glow/Optimizer/GraphOptimizer/CompilationContext.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPassManager.h"
#include "glow/Support/Error.h"

#include "llvm/ADT/ArrayRef.h"

namespace glow {

class Function;
class Backend;
class Module;
class PlaceholderBindings;
class Placeholder;

namespace runtime {
struct DeviceInfo;
}

/// Perform optimizations on the graph representation.
void optimize(Function *F, CompilationContext &cctx);
void optimize(Function *F, CompilationMode mode);
void optimize(Function *F, CompilationContext &cctx, const Backend &B);

/// Fold nodes that were expressed lowered in the input model.
void fold(Function *F, CompilationContext &cctx, const Backend *B = nullptr);

/// Performs the actual constant quantization in function \p F.
void convertQuantizedConstants(Function *F, CompilationContext &cctx);

/// Lower the high-level neural network nodes found in \p F into low-level
/// linear algebra operators. If \p B is not a nullptr then it can prevent
/// lowering of a node via \ref Backend::shouldLower(); otherwise everything
/// will be lowered. \p cctx will contain a mapping of loweredMap from output
/// names of the nodes found and lowered in \p F to the output names of the
/// nodes they were lowered from along with the NodeKind. \p doNotLowerKinds is
/// a set of NodeKinds which represents all Nodes that should not be lowered.
void lower(Function *F, CompilationContext &cctx, const Backend *B = nullptr,
           const KindSet &doNotLowerKinds = {});

/// Convert placeholders in Module \p M to constants based on the values in \p
/// bindings.  Do not convert any placeholders explicitly listed in \p vars.
void convertPlaceholdersToConstants(Function *F,
                                    const PlaceholderBindings &bindings,
                                    llvm::ArrayRef<Placeholder *> vars);

/// Instrument function \p F by inserting quantization profile nodes for
/// capturing stats for quantization. The nodes will refer to tensors allocate
/// in context \p bindings. The instrumentation for profiling will be performed
/// according to the profiling configuration \p profConfig.
void profileQuantization(
    PlaceholderBindings &bindings, Function *F,
    const quantization::ProfilingConfiguration &profConfig);

/// Optimize the Function \p F given compilation options \p cctx for Backend \B.
/// \returns success if all nodes in the final resulting optimized Function are
/// supported by \p B; if not, this represents a compiler error.
Error optimizeFunction(Function *F, const Backend &B, CompilationContext &cctx,
                       const glow::runtime::DeviceInfo *devInfo = nullptr);

/// Optimize the Function \p F given compilation options \p cctx performing
/// backend-independent optimizations that can be done before lowering.
/// \returns success if there were no compiler errors; if not, this represents a
/// compiler error.
Error optimizeFunctionBeforeLowering(Function *F, CompilationContext &cctx);

/// Helper function that may transform \p F given preferences of \p cctx and
/// \p B. The specific transformations are done based on the
/// PrecisionConfiguration found in \p cctx. This could include quantization,
/// profiling, and FP16 conversion.
void transformForPrecisionMode(const Backend &B, Function *F,
                               CompilationContext &cctx);

/// Perform a compile-time constant folding of the node \p N.
/// \returns list of constants which are the result of the constant-folding.
/// These constants correspond to results of the node. If no constant folding
/// was possible an empty vector will be returned.
std::vector<Constant *> constantFold(Node *N);

/// Execute function \p F by the \p backend using the provided \p bindings and
/// the compilation context \p cctx.
/// \returns error if function is not a constant function.
Error executeConstantFunction(Backend &backend, Function &F,
                              PlaceholderBindings &bindings,
                              CompilationContext &cctx);

/// Perform vertical split of FC weights in a given function.
/// Optimization could facilitate parallel execution of FCs on multiple device
/// cores.
/// \returns true in case split took place.
/// \param[in,out] F           function to optimize.
/// \param[in]     numOfChunks number of chunks to split weights and bias into.
/// \param[in]     minKToSplit minimum size of the second dimension of weights
///                            when the split is applied.
bool executeVerticalFCWeightsSplit(Function *F, unsigned numOfChunks,
                                   unsigned minKToSplit);

/// Represents what kind of parallelization transformation should be performed
/// by \ref parallelizeOps().
enum class ParallelTransformKind { None, Data, Model };

/// Perform data or model parallel transformation of supported Nodes in \p F.
/// \p numOfChunksMap maps Nodes to how many chunks they should be split into;
/// if not listed this falls back to \p numOfChunks. \p parOpts represents what
/// kind of parallelism to use. \returns an expected map of Nodes from \p F to
/// the ConcatNode that they were replaced with.
Expected<std::unordered_map<Node *, ConcatNode *>>
parallelizeOps(Function *F,
               const llvm::DenseMap<Node *, size_t> &numOfChunksMap,
               const llvm::DenseMap<Node *, ParallelTransformKind> &parOpts,
               size_t numOfChunks = 1);

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_H
