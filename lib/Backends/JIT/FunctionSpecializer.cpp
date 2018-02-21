// Copyright 2017 Facebook Inc.  All Rights Reserved.
#define DEBUG_TYPE "ir-function-specializer"

#include "JIT.h"
#include "glow/IR/Instrs.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace glow;

using llvm::StringRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace {
/// Perform function specialization with constant arguments taking into account
/// only dimensions, but not the buffer addresses. This allows for faster JIT
/// compilation and the does degrade performance.
static llvm::cl::opt<bool>
    jitSpecializeDims("jit-specialize",
                      llvm::cl::desc("Create specialized functions for "
                                     "operations with constant dimensions"),
                      llvm::cl::init(true));

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
  llvm_unreachable("Unknown type of const parameter");
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
                                                 unsigned argsToBeSpecialized) {
    // Bail if there is nothing to do
    if (!jitSpecializeAllArguments_ && !jitSpecializeDims)
      return F;
    llvm::LLVMContext &ctx = F->getContext();
    llvm::Module *M = F->getParent();

    SpecializationKey key{call, argsToBeSpecialized};
    auto &specializedF = specializations_[key];
    // Is there any specialization for this hash code already?
    if (specializedF) {
      assert(specializedF->getFunctionType() == F->getFunctionType() &&
             "A function and its specialization should have the same type");
      NumSharedSpecializations++;
      return specializedF;
    }

    std::string specializedName = createUniqueName(F->getName());
    // Create a specialized function.
    // This function should have exactly the same type as the original function.
    // It should forward all its non-constant arguments and use constant values
    // for all other arguments. The specialized function should be marked as
    // noinline, to avoid code bloat.
    specializedF = dyn_cast<llvm::Function>(
        M->getOrInsertFunction(specializedName, F->getFunctionType()));
    specializedF->setLinkage(llvm::GlobalValue::LinkageTypes::PrivateLinkage);
    assert(specializedF && "Could not create a specialized function");
    // Specialization thunks should not be inlined.
    specializedF->addFnAttr(llvm::Attribute::AttrKind::NoInline);
    F->removeFnAttr(llvm::Attribute::AttrKind::NoInline);
    // Make sure that the original function will be inlined into the specialized
    // function.
    F->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);

    // Generate the code invoke the original function with a new set of
    // arguments.

    // Setup the entry basic block and initialize the IR builder.
    llvm::BasicBlock *entryBB =
        llvm::BasicBlock::Create(ctx, "entry", specializedF);
    llvm::IRBuilder<> builder(entryBB);

    // Arguments to be used for the invocation of the original function.
    llvm::SmallVector<llvm::Value *, 16> forwardedArgs;
    int argIdx = 0;
    for (auto &arg : specializedF->args()) {
      llvm::Value *argValue = &arg;
      // If this argument needs to be specialized, use its constant
      // value from the call instruction.
      if (argsToBeSpecialized & (1 << argIdx)) {
        argValue = call->getArgOperand(argIdx);
      }
      forwardedArgs.push_back(argValue);
      argIdx++;
    }
    // Create the invocation of the original function.
    builder.CreateCall(F, forwardedArgs);
    builder.CreateRetVoid();
    DEBUG(llvm::dbgs() << "\n\nCreated specialized function " << specializedName
                       << "\n";
          specializedF->print(llvm::errs(), nullptr));
    NumSpecializations++;
    return specializedF;
  }

  /// Specialize a single call.
  /// \returns true if it was possible to specialize the call.
  bool specializeCall(llvm::CallInst *call) {
    llvm::IRBuilder<> builder(call->getParent());
    auto *callee = call->getCalledFunction();
    // Args to be used for calling the specialized function.
    llvm::SmallVector<llvm::Value *, 16> argsForSpecialized(call->arg_begin(),
                                                            call->arg_end());
    // Set of arguments that need to be specialized. See SpecializationKey
    // documentation for more information about the encoding of this set.
    unsigned argsToBeSpecialized = 0;

    // Go over all call arguments.
    // Check that all arguments are constants.
    // Form the set of arguments to be specialized.
    unsigned argIdx = 0;
    for (auto &arg : call->arg_operands()) {
      auto curArgIdx = argIdx++;
      // Do not take data buffer addresses into account, unless asked to do so.
      if (!jitSpecializeAllArguments_ && arg->getType()->isPointerTy() &&
          cast<llvm::PointerType>(arg->getType())
              ->getElementType()
              ->isFloatTy()) {
        continue;
      }

      argsToBeSpecialized |= (1 << curArgIdx);

      // Bail if the values of arguments are not constants.
      if (!getConstantValue(arg)) {
        DEBUG(llvm::dbgs() << "Could not specialize call:\n";
              call->print(llvm::dbgs()));
        return false;
      }
    }

    auto *specializedF =
        getOrCreateSpecializedFunction(call, callee, argsToBeSpecialized);
    // Generate a call of the specialized function before the current call
    // instruction.
    builder.SetInsertPoint(call);
    builder.CreateCall(specializedF, argsForSpecialized);
    return true;
  }

  /// \returns true if a function is eligable for specialization.
  bool isEligableForSpecialization(const llvm::CallInst *call) {
    // For now, specialize all functions invoked from "main". In the future, we
    // may introduce more complex logic for making this decision. It could be
    // based in the number of invocations of a function, number of its
    // arguments, its code size, etc.
    // TODO: May be make this list configurable? E.g. different backends may set
    // it according to their needs?
    const auto *caller = call->getFunction();
    const auto *callee = call->getCalledFunction();
    // Specialized only calls inside main.
    assert(caller->getName().equals("main") &&
           "Only calls inside main are specialized");
    (void)caller;
    // Do not specialize noinline functions, because it does not improve
    // anything.
    return callee != nullptr &&
           !callee->hasFnAttribute(llvm::Attribute::AttrKind::NoInline);
  }

public:
  void run(llvm::Module *M) {
    // Collect calls that were replaced by specialized calls and can be erased.
    // The removal should happen after all specializations are done, because
    // these call instructions are used by the keys in Specializations_ map.
    llvm::SmallVector<llvm::Instruction *, 32> erasedInstructions;
    auto *F = M->getFunction("main");
    // Collect all eligable calls in the current function.
    llvm::SmallVector<llvm::CallInst *, 64> calls;
    for (auto &BB : *F) {
      for (auto &I : BB) {
        auto *CI = dyn_cast<llvm::CallInst>(&I);
        if (!CI)
          continue;
        if (!isEligableForSpecialization(CI))
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
    /// The first call instruction that was used to create this specialization.
    llvm::CallInst *call_{nullptr};
    /// The set of argument numbers that need to be specialized.
    unsigned argsToBeSpecialized_{0};
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
        if ((1 << idx) & key.argsToBeSpecialized_)
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
        if ((1 << idx) & lhs.argsToBeSpecialized_)
          if (getConstantValue(lhs.call_->getArgOperand(idx)) !=
              getConstantValue(rhs.call_->getArgOperand(idx)))
            return false;
      }
      return true;
    }
  };

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
  FunctionSpecializer FuncSpecializer;
  FuncSpecializer.run(llmodule_.get());
}
