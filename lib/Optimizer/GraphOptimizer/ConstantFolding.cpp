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

#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "glow/Backend/Backend.h"
#include "glow/Backends/Interpreter/Interpreter.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Log.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Graph/TensorLayout.h"
#include "glow/Graph/Utils.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace {
/// The name of the temporary function to be used to perform constant folding.
constexpr const char *constEvaluationFunctionName =
    "__constEvaluationFunction__";

/// \returns true if a node \p N is a constant operation, i.e. it is a trivial
/// constant like Constant or Splat or all of its inputs are recursively
/// constant operations, and it has no side-effects and supported by the \p
/// backend. If \p enableQuantizeConstFolding then QuantizeNodes are considered
/// a valid constant operation to fold.
bool isConstantOperation(const Node *N, const Backend &backend,
                         bool enableQuantizeConstFolding) {
  // An operation with side-effects cannot be computed at compile-time.
  if (N->hasSideEffects()) {
    return false;
  }
  // Quantize nodes are not handled by ConstantFolding but by a specific
  // quantization specific optimization.
  if (!enableQuantizeConstFolding && isa<QuantizeNode>(N)) {
    return false;
  }
  // Constant and splat nodes are trivially constant operations.
  if (isa<Constant>(N) || isa<SplatNode>(N)) {
    return true;
  }
  // If the node is backend specific, we cannot safely do constant folding using
  // the interpreter.
  if (!N->isCanonical()) {
    return false;
  }
  // If the operation is not supported by the backend, it cannot be computed at
  // compile-time.
  if (!backend.shouldLower(N) && !backend.isOpSupported(NodeInfo(*N))) {
    return false;
  }
  if (isa<Placeholder>(N)) {
    return false;
  }
  for (size_t idx = 0, e = N->getNumInputs(); idx < e; ++idx) {
    auto input = N->getNthInput(idx);
    if (!isConstantOperation(input.getNode(), backend,
                             enableQuantizeConstFolding)) {
      return false;
    }
  }
  return true;
}

/// \returns true if node \p N has at least one non-constant operation user.
/// \p backend and \p enableQuantizeConstFolding are used to determine what is
/// valid for folding.
bool hasNonConstantOperationUser(const Node *N, const Backend &backend,
                                 bool enableQuantizeConstFolding) {
  assert(isConstantOperation(N, backend, enableQuantizeConstFolding) &&
         "Expected constant operation");
  for (auto &use : N->getUsers()) {
    auto *user = use.getUser();
    // Only consider users in the current function.
    if (user->getParent() != N->getParent()) {
      continue;
    }
    if (!isConstantOperation(user, backend, enableQuantizeConstFolding)) {
      return true;
    }
  }
  return false;
}

/// Compile the function \p F for the provided \p backend using the compilation
/// context \p cctx.
/// \returns compiled function.
Expected<std::unique_ptr<CompiledFunction>>
compile(Backend &backend, Function &F, CompilationContext &cctx) {
  RETURN_IF_ERR(::glow::optimizeFunction(&F, backend, cctx));
  return backend.compile(&F, cctx.backendOpts);
}

/// Runs the compiled function \p compiledF on the \p backend using provided \p
/// bindings.
Error run(Backend &backend, CompiledFunction &compiledF,
          PlaceholderBindings &bindings) {
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  // TODO: Add only constants used by F to the compiled function. This should
  // reduce the amount of data that needs to be copied.
  auto executeErr = compiledF.execute(&context);
  RETURN_IF_ERR(std::move(executeErr));
  // Don't delete bindings.
  context.movePlaceholderBindings().release();
  return Error::success();
}

static bool isCanonicalLayout(const NodeValue &RN, Backend &backend,
                              Node *clonedC, size_t idx) {
  auto resultLayoutStr =
      backend.getTensorLayoutRequirements().getNthResultLayoutRequirements(
          clonedC, idx);
  auto resultLayout = TensorLayoutDescription(resultLayoutStr);
  auto &canInstance = CanonicalTensorLayout::getInstance();
  auto default4DStr = canInstance.getDefaultNDLayout(4);
  auto default4D = TensorLayoutDescription(default4DStr);
  if (resultLayout.getDims().size() == 4 &&
      !canInstance.isSatisfiedBy(RN.getType(), default4D, &resultLayout)) {
    return false;
  }
  return true;
}

// Bail on constant folding post-lowering for backends that break assumptions.
static void bailOnNonCanonicalLayout(
    Function *constEvaluationF, Module &mod,
    const llvm::SmallVectorImpl<SaveNode *> &savedResults) {
  // Some results may be in a non-canonical format post-lowering.
  // For example, if we are trying to constant fold an OpenCL 'Reshape' that
  // has NCHW layout. We cannot transpose it back to canonical layout for
  // two reasons: 1) Need to add a solver that supports weird non-NCHW2NHWC
  // backends. 2) Even if we get a constant tensor as a new "save" of the
  // transpose, the new constant tensor will have the wrong shape. We'd
  // actually need to transpose it back to its pre-modification shape. These
  // issues may be solved in the future (TODO), for now bail on such corner
  // cases. Clean-up before bailing:
  for (auto *SN : savedResults) {
    // Now erase the Placeholder that we created for the SaveNode.
    auto &vars = mod.getPlaceholders();
    mod.erasePlaceholder(
        std::find(vars.begin(), vars.end(), SN->getPlaceholder()));
  }
  mod.eraseFunction(constEvaluationF);
}

/// \returns whether \p N should be folded based on \p cctx's
/// optimizationOpts.materializeSplatsUsedBySet.count, where the backend may
/// specify what Splats should be materialized into Constants based on if
/// they're used by other op kinds.
static bool isSplatToFold(SplatNode *N, const CompilationContext &cctx) {
  for (const auto &U : N->getUsers()) {
    if (cctx.optimizationOpts.materializeSplatsUsedBySet.count(
            U.getUser()->getKind())) {
      return true;
    }
  }
  return false;
}

/// Use to make sure we don't reuse the same name for const fold Functions.
static uint64_t numFolds = 0;

/// Evaluates a provided constant operation \p C using the provided \p backend
/// and using the compilation context \p cctx. If \p record is not a nullptr
/// then the Constant created is added to the map, pointing to the SaveNode that
/// generated that Constant. Additionally if \p record is not a nullptr then the
/// constEvaluationF its associated Placeholders for saving results will not be
/// deleted, and the caller is responsble for deleting them if necessary.
/// \returns constant results. If \p foldSingleSplats then single splat
/// subgraphs will be forced to fold.
bool evaluateConstantOperation(Backend &backend, CompilationContext &cctx,
                               Node *C, std::vector<Constant *> &constResults,
                               ConstantFoldingRecordMap *record,
                               bool foldSingleSplats) {
  // Allow for quantize folding when we have a const folding record.
  const bool enableQuantizeConstFolding = record != nullptr;
  PlaceholderBindings bindings;
  assert(isConstantOperation(C, backend, enableQuantizeConstFolding) &&
         "Expected a constant expression");
  // Constants and splats do not need to be constant evaluated.
  if (isa<Constant>(C) || (isa<SplatNode>(C) && !foldSingleSplats &&
                           !isSplatToFold(cast<SplatNode>(C), cctx))) {
    return true;
  }
  Module &mod = *C->getParent()->getParent();
  const std::string funName = std::string(constEvaluationFunctionName) +
                              std::to_string(numFolds++) + "__" +
                              C->getName().data();
  // Create a temporary function to perform the constant operation.
  Function *constEvaluationF = mod.createFunction(funName);
  // Mapping from existing nodes to the new ones.
  NodeMap currToNew;
  // Clone the constant operation and some of its inputs if necessary.
  auto *clonedC = recursiveClone(constEvaluationF, C, currToNew);
  // Create save nodes for each of the results.
  llvm::SmallVector<SaveNode *, 16> savedResults;

  // If we're recording constant folding, only lower the const fold subgraph
  // (i.e. do not run optimizations when compiling the const fold subgraph).
  // Otherwise some graph optimizations may do folding themselves, meaning that
  // the the subgraph will not contain all folding that occurs.
  if (record) {
    cctx.optimizationOpts.onlyLowerFuns.insert(constEvaluationF);
  }
  ScopeGuard cleanupOnlyLowerFuns([&]() {
    if (record) {
      cctx.optimizationOpts.onlyLowerFuns.erase(constEvaluationF);
    }
  });

  for (size_t idx = 0, e = clonedC->getNumResults(); idx < e; ++idx) {
    auto RN = clonedC->getNthResult(idx);
    auto *SN = constEvaluationF->createSave(clonedC->getName(), RN);
    if (!isCanonicalLayout(RN, backend, clonedC, idx)) {
      bailOnNonCanonicalLayout(constEvaluationF, mod, savedResults);
      return false;
    }
    savedResults.emplace_back(SN);
    bindings.allocate(SN->getPlaceholder());
  }
  // Run the temporary backend to perform this constant operation
  // evaluation.
  if (ERR_TO_BOOL(executeConstantFunction(backend, *constEvaluationF, bindings,
                                          cctx, enableQuantizeConstFolding))) {
    mod.eraseFunction(constEvaluationF);
    return false;
  }

  // Get the results of the constant operation compile-time computation and
  // create new constants from it.
  constResults.reserve(savedResults.size());
  for (auto *SN : savedResults) {
    Tensor *outputTensor = bindings.get(SN->getPlaceholder());
    auto *constResult = mod.createConstant(
        SN->getInput().getNode()->getName().str() + ".constfold",
        std::move(*outputTensor));
    constResults.emplace_back(constResult);

    if (record) {
      // Note: we skip erasing the Placeholders during recording (someone else's
      // responsibility to delete them).
      (*record)[constResult] = SN;
    } else {
      // Now erase the Placeholder that we created for the SaveNode.
      auto &vars = mod.getPlaceholders();
      mod.erasePlaceholder(
          std::find(vars.begin(), vars.end(), SN->getPlaceholder()));
    }
  }
  // Remove the temporary function, unless we're recording the changes (someone
  // else's responsibility to delete them).
  if (!record) {
    mod.eraseFunction(constEvaluationF);
  }
  return true;
}

/// Check if function \p F consists of constant operations only.
LLVM_ATTRIBUTE_USED
Error verifyConstantFunction(Backend &backend, Function &F,
                             bool enableQuantizeConstFolding) {
  // Perform the checks in DEBUG builds only.
  for (auto &N : F.getNodes()) {
    // Saving results is fine.
    if (isa<SaveNode>(&N)) {
      continue;
    }
    // Placeholders can be used just to save results.
    if (!isa<Placeholder>(&N)) {
      RETURN_ERR_IF_NOT(
          isConstantOperation(&N, backend, enableQuantizeConstFolding),
          "Expected constant operation");
      continue;
    }
    if (!N.hasOneUse()) {
      return MAKE_ERR("Expected constant operation");
    }
    auto *SN = dyn_cast<SaveNode>(N.getUsers().begin()->getUser());
    if (SN && SN->getPlaceholder() == &N) {
      continue;
    }
    return MAKE_ERR("Expected constant operation");
  }
  return Error::success();
}

/// Perform a compile-time constant folding of the node \p N using the provided
/// \p backend. If \p record is not a nullptr then the Constant created is added
/// to the map, pointing to the SaveNode that generated that Constant.
/// \returns list of constants which are the result of the
/// constant-folding. These constants correspond to results of the node. If no
/// constant folding was possible an empty vector will be returned. If
/// \p foldSingleSplats then single splat subgraphs will be forced to fold.
bool constantFoldNodeImpl(
    Backend &backend, Node *N, std::vector<Constant *> &constResults,
    ConstantFoldingRecordMap *record = nullptr,
    const CompilationContext &origCctx = CompilationContext(),
    bool foldSingleSplats = false) {
  CompilationContext cctx;
  // Do not recursively call constant folding.
  cctx.optimizationOpts.enableConstantFolding = false;
  cctx.optimizationOpts.enableConstantDeduplication = false;
  cctx.backendOpts.collectConstants = true;
  // Do not print out compilation errors encountered, as constant folding is a
  // best effort; simply silently give up and continue with compilation.
  cctx.verboseCompile = false;
  // Signal to the graph optimizer that it should not be deleting unused
  // Constants in the module.
  cctx.optimizationOpts.delayAndRecordConstantModification = true;
  // Copy over the splats to materialize from the original cctx.
  cctx.optimizationOpts.materializeSplatsUsedBySet =
      origCctx.optimizationOpts.materializeSplatsUsedBySet;
  assert(!ERR_TO_BOOL(cctx.verify()) && "cctx for const folding must be valid");
  return evaluateConstantOperation(backend, cctx, N, constResults, record,
                                   foldSingleSplats);
}

} // namespace

Error glow::executeConstantFunction(Backend &backend, Function &F,
                                    PlaceholderBindings &bindings,
                                    CompilationContext &cctx,
                                    bool enableQuantizeConstFolding) {
// Perform the checks in DEBUG builds only.
#ifndef NDEBUG
  RETURN_IF_ERR(verifyConstantFunction(backend, F, enableQuantizeConstFolding));
#endif
  std::unique_ptr<CompiledFunction> compiledF;
  ASSIGN_VALUE_OR_RETURN_ERR(compiledF, compile(backend, F, cctx));
  return run(backend, *compiledF, bindings);
}

/// Perform constant folding in the function \p F . Any non-trivial node (i.e.
/// not a constant or a splat) that can be computed at compile-time is going to
/// be computed at compile-time. \returns true if any foldings were performed.
/// If \p record is not a nullptr then the Constants created for any constant
/// chain of Nodes is added to the map, pointing to the SaveNode that generated
/// that Constant.
static bool constantFoldFun(Function *F, const CompilationContext &cctx,
                            ConstantFoldingRecordMap *record = nullptr) {
  // Skip if specified in the cctx.
  if (!cctx.optimizationOpts.enableConstantFolding) {
    return false;
  }

  // Allow for quantize folding when we have a const folding record.
  const bool enableQuantizeConstFolding = record != nullptr;

  LOG_SCOPE(F->getLogContext(), "glow::constantFold")
  bool changed = false;
  // Backend to be used for compile-time computations.
  std::unique_ptr<Backend> backend(new Interpreter());
  // Traverse nodes in post-order, so that children are seen before parents.
  GraphPostOrderVisitor postOrderVisitor(*F);
  auto nodes = postOrderVisitor.getPostOrder();
  // Collect all non-trivial constant operations.
  for (auto *N : nodes) {
    // Skip trivial nodes/operations that do not require any constant
    // computations.
    if (isa<Storage>(N) || isa<Constant>(N) ||
        (isa<SplatNode>(N) && !isSplatToFold(cast<SplatNode>(N), cctx)) ||
        isa<TouchNode>(N)) {
      continue;
    }

    // Skip nodes that are not constant operations.
    if (!isConstantOperation(N, *backend, enableQuantizeConstFolding)) {
      continue;
    }

    // Add only a constant operation node whose value is used by at least
    // one non constant-operation node, because no other bigger constant
    // operation containing the current node can completely replace the result
    // of its computation. Doing this check allows for performing a smaller
    // number of evaluateConstantOperation calls later and thus reduces the
    // overhead.
    if (!hasNonConstantOperationUser(N, *backend, enableQuantizeConstFolding)) {
      continue;
    }

    // Compute the constant value of the node.
    std::vector<Constant *> constResults;
    if (!constantFoldNodeImpl(*backend, N, constResults, record, cctx)) {
      continue;
    }
    // Replace all results of the original operation by the computed
    // compile-time results of this operation.
    for (size_t idx = 0, e = constResults.size(); idx < e; ++idx) {
      auto constResult = constResults[idx];
      assert(N->getNthResult(idx).getType() ==
                 constResult->getOutput().getType() &&
             "Constant replacement type must match.");
      // Replace the old result by the new constant result.
      N->getNthResult(idx).replaceAllUsesOfWith(constResult);
    }
    // Perform Dead Code Elimination.
    runDCEPass(F, cctx);
    changed = true;
  }
  return changed;
}

/// Perform constant folding in the function \p F . Any non-trivial node (i.e.
/// not a constant or a splat) that can be computed at compile-time is going to
/// be computed at compile-time. \returns true if any foldings were performed.
bool glow::ConstantFold::run(Function *F, const CompilationContext &cctx) {
  return constantFoldFun(F, cctx);
}

ConstantFoldingRecordMap
glow::constantFoldAndRecord(Function *F, const CompilationContext &cctx) {
  ConstantFoldingRecordMap record;
  constantFoldFun(F, cctx, &record);
  return record;
}

std::vector<Constant *> glow::constantFold(Node *N, bool foldSingleSplats) {
  LOG_SCOPE(N->getParent()->getLogContext(), "glow::constantFold")
  std::unique_ptr<Backend> backend(new Interpreter());
  if (!isConstantOperation(N, *backend,
                           /* enableQuantizeConstFolding */ false)) {
    return {};
  }
  std::vector<Constant *> constResults;
  if (!constantFoldNodeImpl(*backend, N, constResults, nullptr,
                            CompilationContext(), foldSingleSplats)) {
    return {};
  }
  return constResults;
}

void glow::cleanupConstantFolding(Module &mod,
                                  const ConstantFoldingRecordMap &record,
                                  PlaceholderBindings *bindings) {
  auto &PHs = mod.getPlaceholders();
  std::unordered_set<Function *> funsToErase;
  for (auto &r : record) {
    SaveNode *SN = r.second;
    if (bindings && bindings->count(SN->getPlaceholder())) {
      bindings->erase(SN->getPlaceholder());
    }
    mod.erasePlaceholder(
        std::find(PHs.begin(), PHs.end(), SN->getPlaceholder()));
    funsToErase.insert(SN->getParent());
  }
  for (Function *F : funsToErase) {
    mod.eraseFunction(F);
  }
}
