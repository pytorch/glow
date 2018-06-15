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
#define DEBUG_TYPE "ir-function-specializer"

#include "CPUBackend.h"
#include "CommandLine.h"

#include "glow/IR/Instrs.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace glow;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::StringRef;

namespace {
/// Perform function specialization with constant arguments taking into account
/// only dimensions, but not the buffer addresses. This allows for faster JIT
/// compilation and the does degrade performance.
static llvm::cl::opt<bool>
    jitSpecializeDims("jit-specialize",
                      llvm::cl::desc("Create specialized functions for "
                                     "operations with constant dimensions"),
                      llvm::cl::init(true), llvm::cl::cat(CPUBackendCat));

STATISTIC(NumSpecializations, "Number of created specializations");
STATISTIC(NumSharedSpecializations, "Number of shared specializations");

/// Check if the value \p Value is a constant for the purposes of the function
/// specialization, i.e. it is an LLVM constant or it is a global constant
/// variable, which is initialized by an LLVM constant. These variables are
/// produced by IRGen e.g. for arrays of dimensions.
///
/// \returns the constant value if \p Value is a constant or nullptr.
llvm::Value *getConstantValue(llvm::Value *v) {
  // Check if it is a global variable which are constants and initialized by a
  // const. This pattern is produced by the IRGen for the const arrays
  // containing dimensions.
  if (auto *GV = dyn_cast<llvm::GlobalVariable>(v)) {
    auto *init = GV->getInitializer();
    if (!GV->isConstant() || !init)
      return nullptr;
    return v;
  }
  if (isa<llvm::Constant>(v))
    return v;
  // This is an unknown pattern. Be conservative and assume it is not a
  // constant.
  return nullptr;
}

/// Specialize functions for constant arguments. Such specialized functions are
/// marked as noinline and simply invoke the original function with constant
/// arguments. This call later gets inlined and optimized.
class FunctionSpecializer {
  /// Create a unique name for each specialization.
  std::string createUniqueName(llvm::StringRef name) {
    return llvm::Twine(name)
        .concat("_")
        .concat(llvm::Twine(uniqueIdx_++))
        .concat("_specialized")
        .str();
  }

  /// \returns True if the argument \p arg needs to be specialized in the
  /// function.
  bool shouldSpecializeParameter(llvm::Value *arg) {
    // This flag force-specializes all arguments.
    if (jitSpecializeAllArguments_) {
      return true;
    }

    // We don't specialize pointers to floating point buffers.
    if (arg->getType()->isPointerTy()) {
      auto elemTy = cast<llvm::PointerType>(arg->getType())->getElementType();
      if (elemTy->isFloatTy()) {
        return false;
      }
    }

    // We specialize all other arguments.
    return true;
  }

  /// Find an existing specialization or create a new one.
  /// \param CI the call that is being specialized.
  /// \param F the function being specialized.
  /// \param ArgsToBeSpecialized the set of arguments that should be
  /// specialized. See SpecializationKey docs for the explanation of how this
  /// information is encoded.
  /// \returns a specialized version of the function for
  /// provided parameters.
  llvm::Function *getOrCreateSpecializedFunction(llvm::CallInst *call,
                                                 llvm::Function *F,
                                                 uint64_t argsToBeSpecialized) {
    // Bail if there is nothing to do
    if (!jitSpecializeAllArguments_ && !jitSpecializeDims)
      return F;

    SpecializationKey key{call, argsToBeSpecialized};
    auto &specializedF = specializations_[key];
    // Is there any specialization for this hash code already?
    if (specializedF) {
      auto specializedFTy = specializedF->getFunctionType();
      auto FTy = F->getFunctionType();
      (void)specializedFTy;
      (void)FTy;
      assert(
          specializedFTy->getReturnType() == FTy->getReturnType() &&
          "A function and its specialization should have the same return type");
      // The specialized function only takes non-constant parameters from the
      // original function call. Check that the types of these parameter are the
      // same for the original and the specialized function.
      for (size_t argIdx = 0, e = F->arg_size(); argIdx < e; ++argIdx) {
        if (!(argsToBeSpecialized & (((uint64_t)1) << argIdx))) {
          assert(specializedFTy->getParamType(argIdx) ==
                     FTy->getParamType(argIdx) &&
                 "A function and its specialization should have the same "
                 "parameter type");
        }
      }
      NumSharedSpecializations++;
      return specializedF;
    }

    std::string specializedName = createUniqueName(F->getName());

    // We are going to clone the body of the original function and substitute
    // constant values for the (constant) arguments that are going to be
    // specialized. The LLVM's cloning function requires a map for its
    // operation. All arguments mapped by this map are removed from the argument
    // list of the specialized function.
    llvm::ValueToValueMapTy VMap;
    size_t argIdx = 0;
    for (auto &arg : F->args()) {
      // If this argument needs to be specialized, use its constant
      // value from the call instruction.
      if (argsToBeSpecialized & (((uint64_t)1) << argIdx)) {
        auto *argValue = call->getArgOperand(argIdx);
        // Map the argument to a constant value.
        VMap[&arg] = argValue;
      }
      argIdx++;
    }

    // Create a specialized function by cloning the body of the original
    // function and substituting the values of constant arguments. The
    // specialized function should be marked as noinline, to avoid code bloat.
    specializedF = llvm::CloneFunction(F, VMap);
    specializedF->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
    assert(specializedF && "Could not create a specialized function");
    // Specializations should not be inlined.
    specializedF->addFnAttr(llvm::Attribute::AttrKind::NoInline);
    specializedF->setName(specializedName);
    DEBUG(llvm::dbgs() << "\n\nCreated specialized function " << specializedName
                       << "\n";
          specializedF->print(llvm::errs(), nullptr));
    NumSpecializations++;
    return specializedF;
  }

  /// \returns true if a function is eligible for specialization.
  bool isEligibleForSpecialization(const llvm::CallInst *call) {
    // For now, specialize all functions invoked from "main". In the future, we
    // may introduce more complex logic for making this decision. It could be
    // based in the number of invocations of a function, number of its
    // arguments, its code size, etc.
    const auto *caller = call->getFunction();
    const auto *callee = call->getCalledFunction();
    // Specialized only calls inside main.
    assert(caller == entryF_ &&
           "Only calls inside the entry function are specialized");
    (void)caller;
    // Do not specialize any LLVM internal functions.
    if (callee && callee->getName().startswith("llvm."))
      return false;
    // Do not specialize noinline functions, because it does not improve
    // anything.
    return callee != nullptr &&
           !callee->hasFnAttribute(llvm::Attribute::AttrKind::NoInline);
  }

public:
  FunctionSpecializer(llvm::Function *entryF) : entryF_(entryF) {}

  /// Specialize a single call.
  /// \returns the specialized Call instruction if it was possible to specialize
  /// the call or nullptr otherwise.
  llvm::CallInst *specializeCall(llvm::CallInst *call) {
    llvm::IRBuilder<> builder(call->getParent());
    auto *callee = call->getCalledFunction();
    // Args to be used for calling the specialized function.
    llvm::SmallVector<llvm::Value *, 16> argsForSpecialized;
    // Set of arguments that need to be specialized. See SpecializationKey
    // documentation for more information about the encoding of this set.
    uint64_t argsToBeSpecialized = 0;

    // Go over all call arguments.
    // Check that all arguments are constants.
    // Form the set of arguments to be specialized.
    unsigned argIdx = 0;
    for (auto &arg : call->arg_operands()) {
      auto curArgIdx = argIdx++;

      if (!shouldSpecializeParameter(arg)) {
        argsForSpecialized.push_back(arg);
        continue;
      }

      argsToBeSpecialized |= (((uint64_t)1) << curArgIdx);

      // Bail if the values of arguments are not constants.
      if (!getConstantValue(arg)) {
        DEBUG(llvm::dbgs() << "Could not specialize call:\n";
              call->print(llvm::dbgs()));
        return nullptr;
      }
    }

    auto *specializedF =
        getOrCreateSpecializedFunction(call, callee, argsToBeSpecialized);
    // Generate a call of the specialized function before the current call
    // instruction.
    builder.SetInsertPoint(call);
    return createCall(builder, specializedF, argsForSpecialized);
  }

  void run() {
    // Bail if there is nothing to be specialized.
    if (!jitSpecializeDims && !jitSpecializeAllArguments_)
      return;
    // Collect calls that were replaced by specialized calls and can be erased.
    // The removal should happen after all specializations are done, because
    // these call instructions are used by the keys in Specializations_ map.
    llvm::SmallVector<llvm::Instruction *, 32> erasedInstructions;
    auto *F = entryF_;
    // Collect all eligable calls in the current function.
    llvm::SmallVector<llvm::CallInst *, 64> calls;
    for (auto &BB : *F) {
      for (auto &I : BB) {
        auto *CI = dyn_cast<llvm::CallInst>(&I);
        if (!CI)
          continue;
        if (!isEligibleForSpecialization(CI))
          continue;
        calls.push_back(CI);
      }
    }
    // Try to specialize all the collected calls.
    for (auto *call : calls) {
      if (specializeCall(call))
        erasedInstructions.push_back(call);
    }

    // Remove those calls that were successfully replaced by calls of
    // specialized functions. This needs to be done after all specializations,
    // because keys of Specializations_ use these Call instructions for the
    // duration of the whole specialization pass.
    for (auto *I : erasedInstructions) {
      I->eraseFromParent();
    }
    DEBUG(llvm::dbgs() << "Number of specializations: " << NumSpecializations
                       << "\n";
          llvm::dbgs() << "Number of shared specializations: "
                       << NumSharedSpecializations << "\n");
  }

private:
  /// This is a key into the specialization table. It consists of the call
  /// instruction and an integer encoding which arguments of this call should be
  /// used for the hash computation. If the Nth bit is set, then the Nth
  /// argument of the call should participate in the hash computation.
  ///
  /// This encoding heavily relies on the fact that LLVM constants are uniqued
  /// internally and their equality can be checked by means of a simple
  /// pointer comparison.
  struct SpecializationKey {
    SpecializationKey(llvm::CallInst *CI, uint64_t Args)
        : call_(CI), argsToBeSpecialized_(Args) {}

    /// The first call instruction that was used to create this specialization.
    llvm::CallInst *call_{nullptr};
    /// The set of argument numbers that need to be specialized.
    uint64_t argsToBeSpecialized_{0};
  };

  /// A helper class providing a hash function for FunctionSpecializer.
  struct SpecializationKeyHasher {
    size_t operator()(const SpecializationKey &key) const {
      // Take the name of the callee into account.
      llvm::hash_code hash =
          llvm::hash_value(key.call_->getCalledFunction()->getName());
      // Hash over all arguments required by the \p ArgsToBeSpecialized_.
      // We can compute the hash this way, because these arguments are LLVM
      // constants which are uniqued. Therefore, the address of a constant is
      // its unique representation.
      for (unsigned idx = 0, e = key.call_->getNumArgOperands(); idx < e;
           ++idx) {
        if ((((uint64_t)1) << idx) & key.argsToBeSpecialized_)
          hash = llvm::hash_combine(
              hash, getConstantValue(key.call_->getArgOperand(idx)));
      }
      return hash;
    }
  };

  /// A helper class providing the equality function for FunctionSpecializer.
  struct SpecializationKeyEq {
    bool operator()(const SpecializationKey &lhs,
                    const SpecializationKey &rhs) const {
      if (lhs.call_->getCalledFunction() != rhs.call_->getCalledFunction())
        return false;
      if (lhs.argsToBeSpecialized_ != rhs.argsToBeSpecialized_)
        return false;
      for (unsigned idx = 0, e = lhs.call_->getNumArgOperands(); idx < e;
           ++idx) {
        if ((((uint64_t)1) << idx) & lhs.argsToBeSpecialized_)
          if (getConstantValue(lhs.call_->getArgOperand(idx)) !=
              getConstantValue(rhs.call_->getArgOperand(idx)))
            return false;
      }
      return true;
    }
  };

  /// The entry function of the module.
  llvm::Function *entryF_;
  /// Mapping from specialization keys to the specialized functions.
  std::unordered_map<SpecializationKey, llvm::Function *,
                     SpecializationKeyHasher, SpecializationKeyEq>
      specializations_;

  /// An index to create unique specialization names.
  unsigned uniqueIdx_{0};

  /// If set, specialize taking into account the whole set of arguments,
  /// including buffer addresses.
  bool jitSpecializeAllArguments_{false};
};

} // namespace

void LLVMIRGen::performSpecialization() {
  FunctionSpecializer FuncSpecializer(llmodule_->getFunction("main"));
  FuncSpecializer.run();
  // Add debug info to all the newly created functions, i.e. to the created
  // specialized functions.
  for (auto &FF : getModule()) {
    if (FF.isDeclaration())
      continue;
    if (FF.getName().find("_specialized", 0) == llvm::StringRef::npos)
      continue;
    generateFunctionDebugInfo(&FF);
  }
}
