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

/// Use to keep a record what happened during constant folding -- key is the
/// Constant created during constant folding, and associated value is the
/// SaveNode from the constant folding partition that saved that Constant when
/// it was run.
using ConstantFoldingRecordMap = llvm::DenseMap<Constant *, SaveNode *>;

/// Perform optimizations on the graph representation.
void optimize(Function *F, CompilationContext &cctx);
void optimize(Function *F, CompilationMode mode);
void optimize(Function *F, CompilationContext &cctx, const Backend &B);

/// Delete unused Constants from \p mod.
void deleteUnusedConstants(Module &mod);

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
/// was possible an empty vector will be returned. If \p foldSingleSplats then
/// single splat subgraphs will be forced to fold.
std::vector<Constant *> constantFold(Node *N, bool foldSingleSplats = false);

/// Perform constant folding for all Nodes in \p F given \p cctx. \returns a
/// record of what Constants are created by what SaveNodes pointing to
/// Placeholders that replace them. The Functions and output Placeholders
/// created for running the constant folding subgraph are not deleted from the
/// module for this API.
ConstantFoldingRecordMap constantFoldAndRecord(Function *F,
                                               const CompilationContext &cctx);

/// Given \p record, remove the constant folding Functions and their associated
/// output Placeholder from \p mod and \p bindings.
void cleanupConstantFolding(Module &mod, const ConstantFoldingRecordMap &record,
                            PlaceholderBindings *bindings = nullptr);

/// Execute function \p F by the \p backend using the provided \p bindings and
/// the compilation context \p cctx. If \p enableQuantizeConstFolding then
/// QuantizeNodes can be folded as part of a constant chain.
/// \returns error if function is not a constant function.
Error executeConstantFunction(Backend &backend, Function &F,
                              PlaceholderBindings &bindings,
                              CompilationContext &cctx,
                              bool enableQuantizeConstFolding = false);

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

/// A specialized ScopeGuard which prevents constant modification from occuring
/// by swappiing in temporary Placeholders in place of Constants during the
/// scope of the ConstantModificationPreventer. Automatically replaces the
/// Constants back once the scope ends.
class ConstantModificationPreventer : protected ScopeGuard {
  /// Module which contains Constants we want to prevent modification of.
  Module &mod_;

  /// CompilationContext under which we're compiling.
  CompilationContext &cctx_;

  /// Original setting in \ref cctx_ for if constant folding was enabled.
  bool origEnableConstantFolding_;

  /// Map from temporary Placeholders to the Constants they replaced.
  std::unordered_map<Placeholder *, Constant *> tmpPHToConstMap_;

public:
  /// Ctor.
  ConstantModificationPreventer(Module &mod, CompilationContext &cctx);

  /// Make not copyable.
  ConstantModificationPreventer(const ConstantModificationPreventer &) = delete;

  /// Make not assignable.
  ConstantModificationPreventer &
  operator=(const ConstantModificationPreventer &) = delete;

  /// \returns the mapping of tmp PH to Constants.
  const std::unordered_map<Placeholder *, Constant *> &getMapping() const {
    return tmpPHToConstMap_;
  }

  /// Activate the preventer. By default it is deactivated when constructed.
  void activate();

  /// Deactivate the preventer and cleanup. This just forwards to
  /// ScopeGuard::runAndDismiss(), which would have otherwise occurred when
  /// falling out of scope.
  void deactivateAndCleanup() { runAndDismiss(); }
};

/// Perform data or model parallel transformation of supported Nodes in \p F.
/// \p numOfChunksMap maps Nodes to how many chunks they should be split into;
/// if not listed this falls back to \p numOfChunks. \p parOpts represents what
/// kind of parallelism to use. \p modelParallelSplitAlignment optionally can
/// increase the size of model parallel splits to multiple of the given value.
/// \returns an expected map of Nodes from \p F to the ConcatNode that they were
/// replaced with.
Expected<std::unordered_map<Node *, ConcatNode *>>
parallelizeOps(Function *F,
               const llvm::DenseMap<Node *, size_t> &numOfChunksMap,
               const llvm::DenseMap<Node *, ParallelTransformKind> &parOpts,
               size_t numOfChunks = 1, size_t modelParallelSplitAlignment = 1);

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_GRAPHOPTIMIZER_H
