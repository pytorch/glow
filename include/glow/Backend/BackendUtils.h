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

#include "glow/Base/Traits.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/IR/IR.h"

#include "llvm/Support/CommandLine.h"

#include <map>

extern llvm::cl::opt<bool> reuseActivationsMemory;

namespace glow {
namespace runtime {

class MemoryRegion;

/// An enum to indicate what type each symbol in the bundle is.
enum class SymbolCategory {
  Unknown,
  Activation,
  Placeholder,
  Constant,
  PlaceholderTensorView,
  ConstantTensorView
};

/// Id of a memory region.
using MemoryRegionId = uint64_t;

/// Pre-defined memory region ids.
enum MemoryRegions {
  UnknownMemoryRegion,
  ConstantWeight,
  MutableWeight,
  Activation,
  // Has to be the last one.
  LastStdRegion,
};

/// Contains information for initialization and handling of symbol at runtime.
struct RuntimeSymbolInfo {
  /// Symbol name.
  std::string name;
  /// The size in bytes.
  size_t size{0};
  /// Offset in bytes from the base address of the memory region.
  size_t offset{0};
  /// Memory region for a given symbol.
  MemoryRegion *memRegion{nullptr};
  /// Type of symbol.
  Type type;
  /// Graph-level or IR level value associated with this symbol.
  const Kinded *val{nullptr};
  /// Indicates what category the symbol is.
  SymbolCategory symbolCategory{SymbolCategory::Unknown};
  /// Logical id assigned during the codegen (-1 if unused).
  /// This id is used during the codegen to order the tensor
  /// information in memory (e.g. is an index in the offsets array),
  /// it is also used in runtime to generate tensor metadata,
  /// when eager mode is enabled.
  int index{-1};
  /// Is the symbol an input for the function.
  bool input{true};
  /// Is the symbol an output for the function.
  bool output{true};
  /// \returns memory region for a given symbol.
  MemoryRegion *getMemRegion() const { return memRegion; }
  /// \returns memory region id for a given symbol.
  MemoryRegionId getMemRegionId() const;
  /// Methods for dumping the information for debug purposes.
  void dump(llvm::raw_ostream &out) const;
  void dump() const;
  std::string toString() const;
  /// Get the key to be used when looking up this symbol.
  const std::string &getSymbolKey() const;
  /// \returns true if the symbol was allocated and got an address assigned.
  bool isAllocated() const { return size == type.getSizeInBytes(); }
};

/// Set of memory region attributes. Their semantics can be backend-specific.
/// E.g. they may control memory allocation policy inside the region, placement
/// of the region in the target's memory hierarchy and so on.
using MemoryRegionAttributes = std::unordered_map<std::string, std::string>;

/// Mapping from the symbol name to the symbol information.
using SymbolTableTy = std::map<std::string, std::shared_ptr<RuntimeSymbolInfo>>;

/// Description of a memory region to be used for creating an instance of a
/// memory region. It describes the properties of a memory region and also rules
/// defining which logical buffers belong to this memory region.
class MemoryRegionDescription {
  /// Id of the memory region.
  MemoryRegionId id_;
  /// Name of the memory region.
  std::string name_;
  /// Attributes of the memory region.
  MemoryRegionAttributes attrs_;

public:
  MemoryRegionDescription() = default;
  virtual ~MemoryRegionDescription() = default;
  /// \returns true if this memory region should contain a given value.
  virtual bool contains(const glow::Kinded *val) const { return false; }
  /// \returns true if this description results in producing per buffer memory
  /// regions.
  virtual bool isPerBuffer() const;
  /// \returns true if memory regions should preserve the allocation order.
  virtual bool isPreserveAllocationOrder() const;
  /// \returns true if allocations inside this region can reuse memory.
  virtual bool canReuseMemory() const;
  /// \returns name of the memory region description.
  std::string getName() const { return name_; }
  void setName(const std::string &name) { name_ = name; }
  // \returns memory region id.
  MemoryRegionId getMemRegionId() const { return id_; }
  /// Set memory region \p id.
  void setMemRegionId(MemoryRegionId id) { id_ = id; }
  /// Set attributes.
  void setAttributes(const MemoryRegionAttributes &memRegionAttrs) {
    attrs_ = memRegionAttrs;
  }
  /// \returns true if an attribute with a given \p name is defined.
  bool hasAttribute(const std::string &name) const {
    return attrs_.count(name) != 0;
  }
  /// \returns a value of an attribute with a given \p name.
  const std::string &getAttribute(const std::string &name) const {
    return attrs_.at(name);
  }
  /// Add a new attribute with provided \p name and \p value. Assert of an
  /// attribute with this \p name exists already.
  MemoryRegionDescription &addAttribute(const std::string &name,
                                        const std::string &value) {
    assert(!attrs_.count(name) && "Attribute exists already");
    attrs_[name] = value;
    return *this;
  }
  /// Set the \p value of the attribute with provided \p name .
  MemoryRegionDescription &setAttribute(const std::string &name,
                                        const std::string &value) {
    attrs_[name] = value;
    return *this;
  }
  /// \returns attributes of the memory region description.
  MemoryRegionAttributes &getAttributes() { return attrs_; }
  /// \returns attributes of the memory region description.
  const MemoryRegionAttributes &getAttributes() const { return attrs_; }
  /// Methods for dumping the information for debug purposes.
  void dump(llvm::raw_ostream &out) const;
  void dump() const;
  std::string toString() const;
};

/// Description of a memory region for consant weights.
class ConstantWeightMemoryRegionDescription : public MemoryRegionDescription {
public:
  ConstantWeightMemoryRegionDescription() = default;
  ~ConstantWeightMemoryRegionDescription() override = default;
  bool contains(const glow::Kinded *val) const override;
};

/// Description of a memory region for mutable weights.
class MutableWeightMemoryRegionDescription : public MemoryRegionDescription {
public:
  MutableWeightMemoryRegionDescription() = default;
  ~MutableWeightMemoryRegionDescription() override = default;
  bool contains(const glow::Kinded *val) const override;
};

/// Description of a memory region for mutable input weights.
struct MutableInputWeightMemoryRegionDescription
    : public MutableWeightMemoryRegionDescription {
public:
  MutableInputWeightMemoryRegionDescription() = default;
  ~MutableInputWeightMemoryRegionDescription() override = default;
  bool contains(const glow::Kinded *val) const override;
};

/// Description of a memory region for mutable output weights.
class MutableOutputWeightMemoryRegionDescription
    : public MutableWeightMemoryRegionDescription {
public:
  MutableOutputWeightMemoryRegionDescription() = default;
  ~MutableOutputWeightMemoryRegionDescription() override = default;
  bool contains(const glow::Kinded *val) const override;
};

/// Description of a memory region for activations.
struct ActivationMemoryRegionDescription : public MemoryRegionDescription {
public:
  ActivationMemoryRegionDescription() = default;
  ~ActivationMemoryRegionDescription() override = default;
  bool contains(const glow::Kinded *val) const override;
};

/// Memory region used for allocating objects like weights, activations,
/// placeholders, etc. Multiple objects can be allocated in the same memory
/// region if a backend allows for it, depending e.g. on the type and kind of
/// those objects and some backend specifics.
/// Memory region cannot contain symbols that have totally different semantics.
/// E.g. constant weights cannot be combined with per-run activations.
class MemoryRegion {
  /// Id of the memory region.
  MemoryRegionId id_{MemoryRegions::UnknownMemoryRegion};
  /// Name of the memory region.
  std::string name_;
  /// Description of a memory region.
  std::shared_ptr<const MemoryRegionDescription> desc_{nullptr};
  /// Attributes of the memory region.
  MemoryRegionAttributes attrs_;
  /// Ordered set of symbols in this memory region.
  SymbolTableTy symbolTable_;
  /// Allocation/deallocation sequence.
  std::list<Allocation> allocList_;
  /// Memory allocator to be used for this region.
  MemoryAllocator *memAllocator_{nullptr};
  /// Memory size for this region.
  size_t memSize_{0};

public:
  explicit MemoryRegion() = default;
  /// Explicit copy constructor and deleted assignment operator. A MemoryRegion
  /// should be moved. It should only be copied if absolutely necessary and
  /// never implicitly.
  explicit MemoryRegion(const MemoryRegion &) = default;
  virtual ~MemoryRegion() = default;
  MemoryRegion &operator=(const MemoryRegion &) = delete;
  /// \returns name of the region.
  const std::string &getName() const { return name_; }
  /// Sets the \p name of the region.
  void setName(const std::string &name) { name_ = name; }
  /// Sets a memory region id.
  void setId(MemoryRegionId id) { id_ = id; }
  /// \returns memory region id.
  MemoryRegionId getId() const { return id_; }
  /// \returns a symbol table for symbols in this memory region.
  SymbolTableTy &getSymbolTable() { return symbolTable_; }
  /// \returns attributes of this memory region.
  MemoryRegionAttributes getAttributes() { return attrs_; }
  /// Sets the attributes of this region.
  void setAttributes(const MemoryRegionAttributes &attrs) { attrs_ = attrs; }
  /// \returns the memory region descriptions corresponding to this memory
  /// region.
  std::shared_ptr<const MemoryRegionDescription>
  getMemoryRegionDescription() const {
    return desc_;
  }
  /// Set the memory region description.
  void setMemoryRegionDescription(
      std::shared_ptr<const MemoryRegionDescription> memRegionDesc) {
    desc_ = memRegionDesc;
  }
  /// \returs the allocations list for this memory region.
  std::list<Allocation> &getAllocationList() { return allocList_; }
  /// \returs the allocations list for this memory region.
  const std::list<Allocation> &getAllocationList() const { return allocList_; }
  /// \returns memory size of this region.
  size_t getMemSize() const { return memSize_; }
  /// Add a \p symbol to the memory region.
  void add(std::shared_ptr<RuntimeSymbolInfo> symbol);
  /// Perform memory allocation for the memory region. \p
  /// isExclusiveMemoryAllocator indicates if the memory allocator used for the
  /// region is exclusive and used only for this region.
  void allocate(bool isExclusiveMemoryAllocator);
  /// \returns true if this region contains a symbol with a given \p name.
  bool contains(llvm::StringRef name) const;
  /// \returns true if it is a per buffer memory region.
  bool isPerBuffer() const;
  /// \returns true if an attribute with a given \p name is defined.
  bool hasAttribute(const std::string &name) const {
    return attrs_.count(name) != 0;
  }
  /// \returns a value of an attribute with a given \p name.
  const std::string &getAttribute(const std::string &name) const {
    return attrs_.at(name);
  }
  /// Add a new attribute with provided \p name and \p value.
  MemoryRegion &addAttribute(const std::string &name,
                             const std::string &value) {
    attrs_[name] = value;
    return *this;
  }
  /// \returns memory allocator for this memory region.
  MemoryAllocator *getMemoryAllocator() { return memAllocator_; }
  /// Set \p memAlloc as a memory allocator for this memory region.
  void setMemoryAllocator(MemoryAllocator *memAlloc) {
    assert(!memAllocator_ && "Memory allocator can be set only once");
    memAllocator_ = memAlloc;
  }
  /// Methods for dumping the information for debug purposes.
  void dump(llvm::raw_ostream &out) const;
  void dump() const;
  std::string toString() const;
};

/// Descriptions of the memory regions.
class MemoryRegionDescriptions {
  /// Set of memory region descriptions.
  std::list<std::shared_ptr<MemoryRegionDescription>> descriptions;

public:
  virtual ~MemoryRegionDescriptions() = default;
  /// \returns set of memory region descriptions.
  const std::list<std::shared_ptr<MemoryRegionDescription>> &
  getDescriptions() const {
    return descriptions;
  }
  /// \returns true if descriptions are valid and there are no contradictions.
  bool verify() const;
  /// Append a memory region description \p desc at the end.
  MemoryRegionDescriptions &
  append(std::shared_ptr<MemoryRegionDescription> desc);
  /// Prepend a memory region description \p desc at the front.
  MemoryRegionDescriptions &
  prepend(std::shared_ptr<MemoryRegionDescription> desc);
  /// \returns memory region description for a provided \p id or
  /// nullptr if not present.
  std::shared_ptr<MemoryRegionDescription>
  getMemoryRegionDescription(MemoryRegionId id) const;
  /// \returns memory region description for a provided \p name or
  /// nullptr if not present.
  std::shared_ptr<MemoryRegionDescription>
  getMemoryRegionDescription(const std::string &name) const;
  /// Methods for dumping the information for debug purposes.
  void dump(llvm::raw_ostream &out) const;
  void dump() const;
  std::string toString() const;
};

/// Mapping from the region id to the memory region definition.
using MemoryRegionTableTy =
    std::map<MemoryRegionId, std::shared_ptr<MemoryRegion>>;

/// Dump the memory region table to the \p out stream.
void dumpMemoryRegionTable(llvm::raw_ostream &out,
                           const MemoryRegionTableTy &memRegionTable);
/// Dump the memory region table.
void dumpMemoryRegionTable(const MemoryRegionTableTy &memRegionTable);

/// Contains the information needed to be passed forward from compile time to
/// runtime. In order to allocate and initialize memory.
class RuntimeBundle {
  /// Map from symbol name to a RuntimeSymbolInfo.
  SymbolTableTy symbolTable_;
  /// Map from memory region id to a MemoryRegion.
  MemoryRegionTableTy memRegionTable_;
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

  /// Computes offsets and total allocation for Constants, Placeholders, and
  /// Activations to build runtime symbol table. Returns RuntimeBundle.
  void allocMemory(const IRContainer &F, MemoryAllocator &constantAllocator,
                   MemoryAllocator &placeholderAllocator,
                   MemoryAllocator &activationsAllocator);

public:
  /// Get max allocated memory size for a memory region with a provided \p
  /// regionId.
  size_t getMemoryRegionSize(MemoryRegionId regionId);
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
  /// Get the symbol table.
  const SymbolTableTy &getSymbolTable() const { return symbolTable_; }
  SymbolTableTy &getSymbolTable() { return symbolTable_; }
  /// Overwrite the symbol table by \p symbolTable.
  void updateSymbolTable(SymbolTableTy &symbolTable) {
    symbolTable_ = std::move(symbolTable);
  }
  /// Get the memory regions table.
  const MemoryRegionTableTy &getMemoryRegionTable() const {
    return memRegionTable_;
  }
  /// Overwrite the memory region table by \p memRegionTable.
  void updateMemoryRegionTable(MemoryRegionTableTy &memRegionTable) {
    memRegionTable_ = std::move(memRegionTable);
  }
  /// \returns memory region table for this RuntimeBundle.
  MemoryRegionTableTy &getMemoryRegionTable() { return memRegionTable_; }
  /// \returns a memory region with a given \p regionId.
  MemoryRegion &getMemoryRegion(MemoryRegionId regionId);
  /// \returns a memory region with a given \p regionId.
  const MemoryRegion &getMemoryRegion(MemoryRegionId regionId) const;
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
  /// Activations to build runtime symbol table. Returns RuntimeBundle. Uses
  /// provided \p memRegionDescriptions.
  static runtime::RuntimeBundle
  create(const IRFunction &F,
         const runtime::MemoryRegionDescriptions &memRegionDescriptions,
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

  /// Methods for dumping the information for debug purposes.
  void dump(llvm::raw_ostream &out) const;
  void dump() const;
  std::string toString() const;

  /// Deleted default constructor.  A properly constructed RuntimeBundle is
  /// necessary for correct execution using the HostManager.
  RuntimeBundle() = delete;
  // Constructor.
  RuntimeBundle(size_t constWeight, size_t mutableWeight, size_t activations)
      : constants_(nullptr), constantWeightVarsMemSize_(constWeight),
        mutableWeightVarsMemSize_(mutableWeight),
        activationsMemSize_(activations), isValid_(true) {}

  // Constructor.
  RuntimeBundle(SymbolTableTy &symbolTable, size_t constWeight,
                size_t mutableWeight, size_t activations)
      : symbolTable_(std::move(symbolTable)), constants_(nullptr),
        constantWeightVarsMemSize_(constWeight),
        mutableWeightVarsMemSize_(mutableWeight),
        activationsMemSize_(activations), isValid_(true) {}

  // Constructor.
  RuntimeBundle(SymbolTableTy &symbolTable, MemoryRegionTableTy &memRegionTable,
                size_t constWeight, size_t mutableWeight, size_t activations)
      : symbolTable_(std::move(symbolTable)),
        memRegionTable_(std::move(memRegionTable)), constants_(nullptr),
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
                          glow::runtime::RuntimeBundle &runtimeBundle);

/// Allocate \p constants using the provided \p allocator and store the
/// allocation results into a \p symbolTable.
void allocateConstants(const ConstList &constants, MemoryAllocator &allocator,
                       glow::runtime::RuntimeBundle &runtimeBundle);

/// Allocate \p constants using the provided \p allocator and store the
/// allocation results into a \p symbolTable.
void allocateConstants(const std::vector<const glow::Constant *> &constants,
                       MemoryAllocator &allocator,
                       glow::runtime::RuntimeBundle &runtimeBundle);

/// Allocate activations from the instruction stream \p instrs using the
/// provided \p allocator and store the allocation results into a \p
/// symbolTable.
void allocateActivations(const glow::IRFunction::InstListTy &instrs,
                         MemoryAllocator &allocator,
                         glow::runtime::RuntimeBundle &runtimeBundle);

/// \returns true if \p V is capable of handling a partial tensor as input.
bool allowsPartialInput(const Placeholder *V, const Function *F);

/// \returns true if \p V requires last-element padding
bool requiresPadding(const Placeholder *V, const Function *F);

/// \returns true if \p V is used in \p F; false otherwise.
bool usedInFunction(const Placeholder *V, const Function *F);

/// \return a Named object corresponding to the \p kinded object.
const Named *getNamed(const glow::Kinded *kinded);

/// Populate a memory region table for the function \p F, built based on
/// the \p memoryRegionDescriptions. Populate \p symbolTable accordingly.
void createMemoryRegionTable(
    const IRFunction &F,
    const runtime::MemoryRegionDescriptions &memoryRegionDescriptions,
    runtime::MemoryRegionTableTy &memRegionTable,
    runtime::SymbolTableTy &symbolTable);

/// Populate a memory region table for the constants and placeholders of the
/// function \p F, built based on the \p memoryRegionDescriptions. Populate \p
/// symbolTable accordingly. Use \p getValueForNode to map glow::Module level
/// objects like glow::Constants to e.g. glow::WeightVar. The idea is to map
/// Constants and Placeholders to WeightVars in case of processing the Glow
/// IRFunction \p F and to return the original Constants/Placeholders in case of
/// \p F being a glow::Function.
template <typename FUN>
void createMemoryRegionTableForConstantsAndPlaceholders(
    const FUN &F,
    const runtime::MemoryRegionDescriptions &memRegionDescriptions,
    runtime::MemoryRegionTableTy &memRegionTable,
    runtime::SymbolTableTy &symbolTable,
    std::function<const Kinded *(const Storage *)> getValueForNode);

/// \returns default memory regions descriptions. All constants go into one
/// region, all mutable weights into another region and all activations into a
/// third region.
std::shared_ptr<runtime::MemoryRegionDescriptions>
getDefaultMemoryRegionDescriptions();

/// Allocate memory for a function \p F, regions in \p memRegionTable and update
/// the \p symbolTable.
void allocateMemory(const IRContainer &F,
                    runtime::MemoryRegionTableTy &memRegionTable,
                    runtime::SymbolTableTy &symbolTable);
} // end namespace glow
#endif // GLOW_BACKENDS_BACKENDUTILS_H
