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
#ifndef GLOW_BACKENDS_BACKENDUTILS_H
#define GLOW_BACKENDS_BACKENDUTILS_H

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/IR/IR.h"

#include <map>

namespace glow {
namespace runtime {

/// An enum to indicate what type each symbol in the bundle is.
enum class SymbolCategory {
  Activation,
  Placeholder,
  Constant,
  PlaceholderTensorView,
  ConstantTensorView
};

/// Contains information for initialization and handling of symbol at runtime.
struct RuntimeSymbolInfo {
  /// The size in bytes.
  size_t size{0};
  /// Offset in bytes from the base address.
  size_t offset{0};
  /// Type of symbol.
  Type type;
  /// Is the symbol an input for the function.
  bool input{true};
  /// Is the symbol an output for the function.
  bool output{true};
  /// Indicates what category the symbol is.
  SymbolCategory symbolCategory;
  /// Logical id assigned during the codegen (-1 if unused).
  /// This id is used during the codegen to order the tensor
  /// information in memory (e.g. is an index in the offsets array),
  /// it is also used in runtime to generate tensor metadata,
  /// when eager mode is enabled.
  int index{-1};
};

using SymbolTableTy = std::map<std::string, RuntimeSymbolInfo>;

/// Contains the information needed to be passed forward from compile time to
/// runtime. In order to allocate and initialize memory.
class RuntimeBundle {
  /// Map from symbol name to a RuntimeSymbolInfo.
  SymbolTableTy symbolTable_;
  /// Pointer to memory containing the weights for execution.
  uint8_t *constants_{nullptr};
  /// Amount of memory needed for weights.
  size_t constantWeightVarsMemSize_{0};
  /// Amount of memory needed for mutable vars.
  size_t mutableWeightVarsMemSize_{0};
  /// Amount of memory needed for activations.
  size_t activationsMemSize_{0};
  /// True if the RuntimeBundle is valid, false if not.
  bool isValid_{false};

public:
  /// Get Constant Weights memory size.
  size_t getConstantWeightSize() const { return constantWeightVarsMemSize_; }
  /// Get Mutable Weights memory size.
  size_t getMutableWeightSize() const { return mutableWeightVarsMemSize_; }
  /// Get Activations Weights memory size.
  size_t getActivationsSize() const { return activationsMemSize_; }
  /// Get pointer to memory block of constants.
  uint8_t *getConstants() const { return constants_; }
  /// Set pointer to memory block of constants.
  void setConstants(uint8_t *constants) { constants_ = constants; }
  /// Helper function, gets offset of \p v.
  size_t getValueOffset(const Named *v) const;
  /// Helper function, gets symbol info for \p v.
  const RuntimeSymbolInfo &getSymbolInfo(const Named *v) const;
  /// Get a const reference to the symbol table.
  const SymbolTableTy &getSymbolTable() const { return symbolTable_; }
  void updateSymbolTable(const SymbolTableTy &symbolTable) {
    symbolTable_ = symbolTable;
  }
  /// At compile time condense constants to a single block of memory.
  /// This allows the graph to go away after compile time.
  /// Allocates a block of memory of size \p constantMaxSize then walks the
  /// given function \p F and and copies weights to their address as specified
  /// by offsets contained in symbolTable_.
  void collectConstants(const IRFunction *F);
  void collectConstants(const Module *M);
#if FACEBOOK_INTERNAL
  void collectConstants(const FXIRWrapper *F);
#endif
  /// Free constants.
  void freeConstants();

  /// Sets the input and output flags for each symbol in the symbolBundle.
  void setInputsandOutputs();

  /// Computes offsets and total allocation for Constants, Placeholders, and
  /// Activations to build runtime symbol table. Returns RuntimeBundle.
  static runtime::RuntimeBundle create(const IRFunction &F,
                                       MemoryAllocator &constantAllocator,
                                       MemoryAllocator &placeholderAllocator,
                                       MemoryAllocator &activationsAllocator);

  /// Computes offsets and total allocation for Constants, Placeholders, and
  /// Activations to build runtime symbol table. \returns RuntimeBundle.
  /// Constants and Placeholders are taken from \p F, and all Activations
  /// required by each function in \p funcs are placed into the same
  /// RuntimeBundle.
  static runtime::RuntimeBundle
  create(const Function &F, const std::vector<const IRFunction *> &funcs);

  /// Computes offsets and total allocations for Constants, Placeholders, and
  /// Activations to build runtime symbol table. \returns RuntimeBundle. Uses a
  /// single allocator \p allocator and allocates all buffers contiguously in
  /// the same block.
  static runtime::RuntimeBundle create(const IRFunction &F,
                                       MemoryAllocator &allocator);

  /// Build a runtime symbol table from a Function.  Computes Constant and
  /// Placeholder sizes, but not Activations, since Functions are unserialized.
  /// Only use this method to generate bundles for backends that do not use
  /// Glow's IR.
  static runtime::RuntimeBundle create(const Function &F);

  /// Deleted default constructor.  A properly constructed RuntimeBundle is
  /// necessary for correct execution using the HostManager.
  RuntimeBundle() = delete;

  // Constructor.
  RuntimeBundle(SymbolTableTy &symbolTable, size_t constWeight,
                size_t mutableWeight, size_t activations)
      : symbolTable_(std::move(symbolTable)), constants_(nullptr),
        constantWeightVarsMemSize_(constWeight),
        mutableWeightVarsMemSize_(mutableWeight),
        activationsMemSize_(activations), isValid_(true) {}

  // Explicit copy constructor and deleted assignment operator. A RuntimeBundle
  // should be moved. It should only be copied if absolutely necessary and never
  // implicitly.
  explicit RuntimeBundle(const RuntimeBundle &) = default;
  RuntimeBundle &operator=(const RuntimeBundle &) = delete;

  // Move constructor and assignment operator.
  RuntimeBundle(RuntimeBundle &&rhs);
  RuntimeBundle &operator=(RuntimeBundle &&rhs);
};
} // namespace runtime

/// Generates a struct named has_\p METHOD_NAME that looks for a method called
/// \p METHOD_NAME inside of ClassName with return type ReturnType.
#define CLASS_CONTAINS_METHOD(METHOD_NAME)                                     \
  template <typename ClassName, typename ReturnType>                           \
  struct has_##METHOD_NAME {                                                   \
  private:                                                                     \
    template <typename T>                                                      \
    static constexpr auto check(T *) ->                                        \
        typename std::is_same<decltype(std::declval<T>().METHOD_NAME()),       \
                              ReturnType>::type;                               \
    template <typename> static constexpr std::false_type check(...);           \
    typedef decltype(check<ClassName>(0)) type;                                \
                                                                               \
  public:                                                                      \
    static constexpr bool value = type::value;                                 \
  };

/// Use template meta-programming to check if typename ClassName contains
/// getFusedActivation() method. Below generates a struct named
/// has_getFusedActivation that looks for said method.
CLASS_CONTAINS_METHOD(getFusedActivation)

/// If \p W is a weight that is read from \returns true.
bool isInput(const Value *W);

/// If \p W is an output weight \returns true. This is determined by checking if
/// the weight has a user which uses it as a write output.
bool isOutput(const Value *W);

/// If \p PH is an output placeholder in the IRFunction \p F,
/// \returns true.
/// This is determined by checking if the PH has weights which are referenced by
/// other Instructions as OperandKind::InOut or OperandKind::Out.
bool isOutput(const Placeholder *PH, const IRFunction &F);

/// \returns true if \p PH is an output Placeholder for any function in \p
/// funcs.
bool isOutput(const Placeholder *PH,
              const std::vector<const Function *> &funcs);

/// If \p PH is an input placeholder in the IRFunction \p F,
/// \returns true.
/// This is determined by checking if the PH is always used as an @in parameter
/// by the current function.
bool isInput(const Placeholder *PH, const IRFunction &F);

/// If \p N does not have fused activation \returns true.
template <typename T,
          std::enable_if_t<!has_getFusedActivation<T, FusedActivation>::value,
                           int> = 0>
bool checkNoFusion(const T &N) {
  (void)N;
  return true;
}

/// If \p N does not have fused activation \returns true.
template <typename T,
          std::enable_if_t<has_getFusedActivation<T, FusedActivation>::value,
                           int> = 0>
bool checkNoFusion(const T &N) {
  if (N.getFusedActivation() != FusedActivation::NONE) {
    report("Glow backend does not support fused Activations for: " +
           std::string(N.getKindName()));
    return false;
  }
  return true;
}

/// If \p N does not have fused activation \returns true.
bool checkNoFusionForNode(const Node &N);

/// If \p I does not have fused activation \returns true.
bool checkNoFusionForInstr(const Instruction &I);

/// Contains information for placeholder during allocation.
struct PlaceholderInputOutputInfo {
  /// The placeholder address.
  const Placeholder *addr;
  /// Is the placeholder an input for the function.
  bool isInput;
  /// Is the placeholder an onput for the function.
  bool isOutput;
};

using ContiguousPlaceholders = std::vector<PlaceholderInputOutputInfo>;

/// Convert placeholders to be ordered as input|inputOutput|output|neither.
/// Packed into {Placeholder *, isInput, isOutput} as
/// PlaceholderInputOutputInfo. FUN could be Function or IRFunction. ARR could
/// be std::list<Placeholder *> or std::vector<const Placeholder *>
template <typename FUN, typename ARR>
ContiguousPlaceholders getContiguousPlaceHolder(const ARR &holders,
                                                const FUN &F);

/// Allocate \p placeholders using the provided \p allocator and store the
/// allocation results into a \p symbolTable.
void allocatePlaceholders(const ContiguousPlaceholders &placeholders,
                          MemoryAllocator &allocator,
                          glow::runtime::SymbolTableTy &symbolTable);

/// Allocate \p constants using the provided \p allocator and store the
/// allocation results into a \p symbolTable.
void allocateConstants(const ConstList &constants, MemoryAllocator &allocator,
                       glow::runtime::SymbolTableTy &symbolTable);

/// Allocate \p constants using the provided \p allocator and store the
/// allocation results into a \p symbolTable.
void allocateConstants(const std::vector<const glow::Constant *> &constants,
                       MemoryAllocator &allocator,
                       glow::runtime::SymbolTableTy &symbolTable);

/// Allocate activations from the instruction stream \p instrs using the
/// provided \p allocator and store the allocation results into a \p
/// symbolTable.
void allocateActivations(const glow::IRFunction::InstListTy &instrs,
                         MemoryAllocator &allocator,
                         glow::runtime::SymbolTableTy &symbolTable);

/// \returns true if \p V is capable of handling a partial tensor as input.
bool allowsPartialInput(const Placeholder *V, const Function *F);

/// \returns true if \p V requires last-element padding
bool requiresPadding(const Placeholder *V, const Function *F);

/// \returns true if \p V is used in \p F; false otherwise.
bool usedInFunction(const Placeholder *V, const Function *F);

} // end namespace glow
#endif // GLOW_BACKENDS_BACKENDUTILS_H
