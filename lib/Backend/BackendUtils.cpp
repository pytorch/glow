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
#include "glow/Backend/BackendUtils.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/CommandLine.h"

#include "glow/Graph/FXIRWrapper.h"
#include <glog/logging.h>

#define DEBUG_TYPE "backend-utils"

using namespace glow;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

static llvm::cl::OptionCategory BackendUtilsCat("Glow Backend Utils Options");

static llvm::cl::opt<bool> reuseActivationsMemory(
    "reuse-activation-memory-allocations",
    llvm::cl::desc("Should activation memory allocations be reused"),
    llvm::cl::init(true), llvm::cl::cat(BackendUtilsCat));

glow::runtime::RuntimeBundle::RuntimeBundle(
    glow::runtime::RuntimeBundle &&rhs) {
  *this = std::move(rhs);
}

glow::runtime::RuntimeBundle &
glow::runtime::RuntimeBundle::operator=(glow::runtime::RuntimeBundle &&rhs) {
  if (this == &rhs) {
    // Do nothing if rhs is the same object as this.
    return *this;
  }

  std::swap(symbolTable_, rhs.symbolTable_);
  std::swap(constants_, rhs.constants_);
  std::swap(constantWeightVarsMemSize_, rhs.constantWeightVarsMemSize_);
  std::swap(mutableWeightVarsMemSize_, rhs.mutableWeightVarsMemSize_);
  std::swap(activationsMemSize_, rhs.activationsMemSize_);
  std::swap(isValid_, rhs.isValid_);
  // rhs is not valid now that all of its contents have been stolen.
  rhs.isValid_ = false;
  return *this;
}

void glow::runtime::RuntimeBundle::collectConstants(const IRFunction *F) {
  DCHECK(isValid_);
  collectConstants(F->getParent());
}

void glow::runtime::RuntimeBundle::freeConstants() {
  DCHECK(isValid_);

  if (constants_) {
    glow::alignedFree(constants_);
    constants_ = nullptr;
  }
}
void glow::runtime::RuntimeBundle::collectConstants(const Module *M) {
  DCHECK(isValid_);

  // At compile time condense constants to a single block of memory.
  // This allows the graph to go away after compile time.
  // If there are no constants return nullptr.
  if (constantWeightVarsMemSize_ == 0) {
    constants_ = nullptr;
    return;
  }

  assert(constants_ == nullptr && "constants already allocated");
  constants_ =
      (uint8_t *)alignedAlloc(constantWeightVarsMemSize_, TensorAlignment);

  for (const auto &symbol : symbolTable_) {
    llvm::StringRef name = symbol.first;
    const RuntimeSymbolInfo &info = symbol.second;

    Constant *c = M->getConstantByName(name);
    if (!c) {
      continue;
    }
    auto *payload = c->getPayload().getUnsafePtr();
    assert(info.size == c->getPayload().getSizeInBytes() &&
           "Mismatched constant size");

    // Copy weight to offset.
    memcpy(constants_ + info.offset, payload, info.size);
  }
}

#if FACEBOOK_INTERNAL
void glow::runtime::RuntimeBundle::collectConstants(const FXIRWrapper *F) {
  DCHECK(isValid_);

  // At compile time condense constants to a single block of memory.
  // This allows the graph to go away after compile time.
  // If there are no constants return nullptr.
  if (constantWeightVarsMemSize_ == 0) {
    constants_ = nullptr;
    return;
  }

  assert(constants_ == nullptr && "constants already allocated");
  constants_ =
      (uint8_t *)alignedAlloc(constantWeightVarsMemSize_, TensorAlignment);

  for (const auto &symbol : symbolTable_) {
    llvm::StringRef name = symbol.first;
    const RuntimeSymbolInfo &info = symbol.second;

    // Only work with constants/weights here.
    auto category = info.symbolCategory;
    if (category != glow::runtime::SymbolCategory::Constant) {
      continue;
    }

    auto mapToConstants = F->getMapNodeNameToStorage();
    assert(mapToConstants.find(name.str()) != mapToConstants.end());
    const auto *wt = mapToConstants[name.str()];
    const auto *c = llvm::dyn_cast<const Constant>(wt);
    if (!c) {
      continue;
    }
    auto *payload = c->getPayload().getUnsafePtr();
    assert(info.size == c->getPayload().getSizeInBytes() &&
           "Mismatched constant size");

    // Copy weight to offset.
    memcpy(constants_ + info.offset, payload, info.size);
  }
}
#endif

size_t glow::runtime::RuntimeBundle::getValueOffset(const Named *v) const {
  DCHECK(isValid_);
  auto it = symbolTable_.find(std::string(v->getName()));
  assert(it != symbolTable_.end() && "Symbol not found.");
  return it->second.offset;
}

const runtime::RuntimeSymbolInfo &
runtime::RuntimeBundle::getSymbolInfo(const Named *v) const {
  DCHECK(isValid_);
  auto it = symbolTable_.find(std::string(v->getName()));
  assert(it != symbolTable_.end() && "Symbol not found.");
  return it->second;
}

namespace glow {

/// If \p W is an output weight \returns true. This is determined by checking if
/// the weight has a user which uses it as a write output.
bool isOutput(const Value *W) {
  auto *weight = llvm::dyn_cast<WeightVar>(W);
  DCHECK(weight) << "Expected WeightVar";
  for (const auto &use : ValueUses(weight)) {
    Instruction *user = use.get();
    // Ignore deallocs.
    if (isa<DeallocActivationInst>(user)) {
      continue;
    }
    OperandKind kind = use.getOperand().second;
    if (kind == OperandKind::Out || kind == OperandKind::InOut) {
      return true;
    }
  }
  return false;
}

/// If \p PH is an output placeholder in the function \p F, \returns true.
/// This is determined by checking if the PH has a user which uses the PH as an
/// overwritten input.
bool isOutput(const Placeholder *PH, const IRFunction &F) {
  auto *weight = F.getWeightForNode(PH);
  DCHECK(weight) << "Weight for a node was not found";
  return isOutput(weight);
}

/// If \p W is a weight that is first read from \returns true.
bool isInput(const Value *W) {
  auto *weight = llvm::dyn_cast<WeightVar>(W);
  const glow::Instruction *firstUser = nullptr;
  bool hasReads = false;
  for (const auto &U : ValueUses(weight)) {
    const auto *user = U.get();
    // TensorView instruction doesn't read from a placeholder.
    if (isa<TensorViewInst>(user)) {
      continue;
    }
    // Remember the earliest use.
    if (!firstUser || firstUser->getIterator() > user->getIterator()) {
      firstUser = user;
    }
    // Ignore deallocs.
    if (isa<DeallocActivationInst>(user)) {
      continue;
    }
    OperandKind kind = U.getOperand().second;
    if (kind == OperandKind::In || kind == OperandKind::InOut) {
      hasReads = true;
    }
  }

  if (!hasReads) {
    return false;
  }

  // Check if the first use is a read.
  if (firstUser) {
    // If this instruction has reads, then the first use is an @in.
    auto *weightOrigin = getOrigin(weight);
    for (int idx = 0, e = firstUser->getNumOperands(); idx < e; ++idx) {
      const auto op = firstUser->getOperand(idx);
      auto *opOrigin = getOrigin(op.first);
      auto opKind = op.second;
      if (opOrigin == weightOrigin && opKind == OperandKind::In) {
        return true;
      }
    }
    // No reads were found, thus the first use is a write.
    return false;
  }
  // If there are no users, it is not an input.
  return false;
}

/// If \p PH is an input placeholder in the function \p F, \returns true.
bool isInput(const Placeholder *PH, const IRFunction &F) {
  // Check that the PH is always used as an @in parameter by the current
  // function.
  auto *weight = F.getWeightForNode(PH);
  DCHECK(weight) << "Weight for a node was not found";
  return isInput(weight);
}

bool isOutput(const Placeholder *PH,
              const std::vector<const Function *> &funcs) {
  for (const auto &f : funcs) {
    if (isOutput(PH, *f)) {
      return true;
    }
  }

  return false;
}

/// \returns true if \p PH is an input Placeholder for any function in \p funcs.
bool isInput(const Placeholder *PH,
             const std::vector<const Function *> &funcs) {
  for (const auto &f : funcs) {
    if (isInput(PH, *f)) {
      return true;
    }
  }

  return false;
}

/// If \p N does not have fused activation \returns true.
bool checkNoFusionForNode(const Node &N) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case Kinded::Kind::CLASS##Kind: {                                            \
    const CLASS *CI = llvm::cast<CLASS>(&N);                                   \
    return checkNoFusion(*CI);                                                 \
    break;                                                                     \
  }
  switch (N.getKind()) {
#include "glow/AutoGenNodes.def"
  default:
    llvm_unreachable("Invalid node.");
  }
  return true;
}

/// If \p I does not have fused activation \returns true.
bool checkNoFusionForInstr(const Instruction &I) {
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME)                                                 \
  case Kinded::Kind::CLASS##Kind: {                                            \
    const CLASS *CI = llvm::cast<CLASS>(&I);                                   \
    return checkNoFusion(*CI);                                                 \
    break;                                                                     \
  }
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME)                                \
  case Kinded::Kind::CLASS##Kind: {                                            \
    const CLASS *CI = llvm::cast<CLASS>(&I);                                   \
    return checkNoFusion(*CI);                                                 \
    break;                                                                     \
  }
  switch (I.getKind()) {
#include "glow/AutoGenInstr.def"
  default:
    llvm_unreachable("Invalid instruction.");
  }
  return true;
}

template <typename FUN, typename ARR>
ContiguousPlaceholders getContiguousPlaceHolder(const ARR &holders,
                                                const FUN &F) {
  // Pure input placeholders.
  std::vector<const Placeholder *> intputPlaceholders;
  // Pure output placeholders.
  std::vector<const Placeholder *> outputPlaceholders;
  // Input&output placeholders.
  std::vector<const Placeholder *> inputOutputPlaceholders;
  // Neither input nor output placeholders.
  std::vector<const Placeholder *> emptyPlaceholders;
  // Return value.
  ContiguousPlaceholders ret;

  for (auto &v : holders) {
    if (isInput(v, F)) {
      if (!isOutput(v, F)) {
        intputPlaceholders.push_back(v);
      } else {
        inputOutputPlaceholders.push_back(v);
      }
    } else {
      if (isOutput(v, F)) {
        outputPlaceholders.push_back(v);
      } else {
        emptyPlaceholders.push_back(v);
      }
    }
  }

  for (auto &v : intputPlaceholders) {
    PlaceholderInputOutputInfo holder;
    holder.addr = v;
    holder.isInput = true;
    holder.isOutput = false;
    ret.push_back(holder);
  }

  for (auto &v : inputOutputPlaceholders) {
    PlaceholderInputOutputInfo holder;
    holder.addr = v;
    holder.isInput = true;
    holder.isOutput = true;
    ret.push_back(holder);
  }

  for (auto &v : outputPlaceholders) {
    PlaceholderInputOutputInfo holder;
    holder.addr = v;
    holder.isInput = false;
    holder.isOutput = true;
    ret.push_back(holder);
  }

  for (auto &v : emptyPlaceholders) {
    PlaceholderInputOutputInfo holder;
    holder.addr = v;
    holder.isInput = false;
    holder.isOutput = false;
    ret.push_back(holder);
  }

  return ret;
}

/// \returns true if \p dst is capable of handling a partial tensor as input
/// from \p src.
static bool allowsPartialInput(const Node *src, const Node *dst) {
  // If N is used as the indices or weights of a sparse lookup, it is safe to
  // access a partial tensor.
  if (auto *SLS =
          llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(
              dst)) {
    return src == SLS->getIndices() || src == SLS->getWeights();
  } else if (auto *SLS =
                 llvm::dyn_cast<FusedRowwiseQuantizedSparseLengthsSumNode>(
                     dst)) {
    return src == SLS->getIndices();
  } else if (auto *SLS = llvm::dyn_cast<SparseLengthsWeightedSumNode>(dst)) {
    return src == SLS->getIndices() || src == SLS->getWeights();
  } else if (auto *SLS = llvm::dyn_cast<SparseLengthsSumNode>(dst)) {
    return src == SLS->getIndices();
  } else if (auto *EBB = llvm::dyn_cast<EmbeddingBagNode>(dst)) {
    return src == EBB->getIndices() || src == EBB->getWeights();
  } else if (auto *EBB =
                 llvm::dyn_cast<EmbeddingBagByteRowwiseOffsetsNode>(dst)) {
    return src == EBB->getIndices() || src == EBB->getWeights();
  }
  return false;
}

bool allowsPartialInput(const Placeholder *V, const Function *F) {
  for (auto const &U : V->getUsers()) {
    if (U.getUser()->getParent() != F) {
      continue;
    }
    if (!allowsPartialInput(*U.get(), U.getUser())) {
      return false;
    }
  }
  return true;
}

/// \returns true if \p dst requires last-element padding for \p src
/// It is assumed that \p src cannot be partial input
static bool requiresPadding(const Node *src, const Node *dst) {
  if (auto *EBB = llvm::dyn_cast<EmbeddingBagNode>(dst)) {
    return src == EBB->getOffsets();
  } else if (auto *EBB =
                 llvm::dyn_cast<EmbeddingBagByteRowwiseOffsetsNode>(dst)) {
    return src == EBB->getOffsets();
  }
  return false;
}

bool requiresPadding(const Placeholder *V, const Function *F) {
  // TODO: this function is largely duplicated with allowsPartialInput()
  // we should consider merging the two
  for (auto const &U : V->getUsers()) {
    if (U.getUser()->getParent() != F) {
      continue;
    }
    if (!requiresPadding(*U.get(), U.getUser())) {
      return false;
    }
  }
  return true;
}

bool usedInFunction(const Placeholder *V, const Function *F) {
  for (auto const &U : V->getUsers()) {
    if (U.getUser()->getParent() == F) {
      return true;
    }
  }
  return false;
}

/// Allocate space for the Constants in \p constants using \p allocator and
/// store the resultant symbols in \p symbolTable.
template <typename ConstantsTy>
static void allocateConstantsImpl(const ConstantsTy &constants,
                                  MemoryAllocator &allocator,
                                  glow::runtime::SymbolTableTy &symbolTable) {
  for (auto const *C : constants) {
    // Same constant may be used multiple times by different functions. But it
    // should be assigned an address only once.
    if (symbolTable.count(std::string(C->getName()))) {
      continue;
    }
    auto size = C->getType()->getSizeInBytes();
    auto offset = allocator.allocate(size, C);
    runtime::RuntimeSymbolInfo symbol;
    symbol.offset = offset;
    symbol.size = size;
    symbol.type = *C->getType();
    symbol.input = false;
    symbol.output = false;
    symbol.symbolCategory = glow::runtime::SymbolCategory::Constant;
    symbolTable.emplace(C->getName(), symbol);
    DEBUG_GLOW(LOG(INFO) << strFormat(
                   "Assigned address to constant %s: %zx (%zd bytes)\n",
                   C->getName().data(), symbol.offset, symbol.size));
  }
}

void allocateConstants(const ConstList &constants, MemoryAllocator &allocator,
                       glow::runtime::SymbolTableTy &symbolTable) {
  allocateConstantsImpl(constants, allocator, symbolTable);
}

void allocateConstants(const std::vector<const glow::Constant *> &constants,
                       MemoryAllocator &allocator,
                       glow::runtime::SymbolTableTy &symbolTable) {
  allocateConstantsImpl(constants, allocator, symbolTable);
}

/// Allocate space for the Placeholders in \p placeholders using \p allocator
/// and store the resultant symbols in \p symbolTable.
void allocatePlaceholders(const ContiguousPlaceholders &placeholders,
                          MemoryAllocator &allocator,
                          glow::runtime::SymbolTableTy &symbolTable) {
  for (const auto &p : placeholders) {
    auto &V = p.addr;
    assert(!symbolTable.count(std::string(V->getName())) &&
           "Allocation already made!");
    auto size = V->getType()->getSizeInBytes();
    auto offset = allocator.allocate(size, V);
    runtime::RuntimeSymbolInfo symbol;
    symbol.offset = offset;
    symbol.size = size;
    symbol.type = *V->getType();
    symbol.output = p.isOutput;
    symbol.input = p.isInput;
    symbol.symbolCategory = glow::runtime::SymbolCategory::Placeholder;
    symbolTable.emplace(std::string(V->getName()), symbol);
    DEBUG_GLOW(LOG(INFO) << strFormat(
                   "Assigned address to mutable weight %s: %zx (%zd bytes)\n",
                   V->getName().data(), symbol.offset, symbol.size));
  }
}

/// Allocate space for the activations of \p instrs using \p allocator and store
/// the resultant symbols in \p symbolTable.
void allocateActivations(const glow::IRFunction::InstListTy &instrs,
                         MemoryAllocator &allocator,
                         glow::runtime::SymbolTableTy &symbolTable) {

  // Gather allocation/deallocation sequence.
  std::list<Allocation> allocList;
  if (reuseActivationsMemory) {
    // When reusing memory we register allocs/deallocs in their original order.
    for (const auto &I : instrs) {
      if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
        auto numBytes = I.getSizeInBytes();
        allocList.emplace_back(A, /* alloc */ true, numBytes);
        continue;
      }
      if (auto *D = dyn_cast<DeallocActivationInst>(&I)) {
        auto *A = D->getAlloc();
        allocList.emplace_back(A, /* alloc */ false, 0);
        continue;
      }
    }
  } else {
    // When not reusing memory we register first the allocs then the deallocs.
    for (const auto &I : instrs) {
      if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
        auto numBytes = I.getSizeInBytes();
        allocList.emplace_back(A, /* alloc */ true, numBytes);
        continue;
      }
    }
    for (const auto &I : instrs) {
      if (auto *D = dyn_cast<DeallocActivationInst>(&I)) {
        auto *A = D->getAlloc();
        allocList.emplace_back(A, /* alloc */ false, 0);
        continue;
      }
    }
  }

  // Allocate all segments at once for better allocation efficiency.
  // We use a separate allocator object since the function "allocateAll()"
  // does not work together with the function "allocate()" which could have
  // been used with the original allocator.
  MemoryAllocator activationsAllocator("mem", 0, allocator.getAlignment());
  uint64_t activationsSize = activationsAllocator.allocateAll(allocList);

  // Allocate a contiguous segment for the activations of the current function.
  // The individual buffers within this segment are placed according to the
  // logic of allocateAll for better efficiency.
  uint64_t activationsBaseAddr = 0;
  if (activationsSize) {
    MemoryAllocator::Handle activationsHandle = &instrs;
    activationsBaseAddr =
        allocator.allocate(activationsSize, activationsHandle);
    if (reuseActivationsMemory) {
      allocator.deallocate(activationsHandle);
    }
  }

  // Map addresses of allocated segments.
  for (const auto &I : instrs) {
    if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
      auto numBytes = I.getSizeInBytes();
      size_t addr = activationsBaseAddr + activationsAllocator.getAddress(A);
      assert(!symbolTable.count(std::string(A->getName())) &&
             "Allocation already made!");
      runtime::RuntimeSymbolInfo symbol;
      symbol.offset = addr;
      symbol.size = numBytes;
      symbol.type = *A->getType();
      symbol.input = false;
      symbol.output = false;
      symbol.symbolCategory = glow::runtime::SymbolCategory::Activation;
      symbolTable.emplace(std::string(A->getName()), symbol);
      DEBUG_GLOW(LOG(INFO) << strFormat(
                     "Assigned address to activation %s: %zx (%zd bytes)\n",
                     A->getName().data(), symbol.offset, symbol.size));
      continue;
    }

    if (auto *TV = dyn_cast<TensorViewInst>(&I)) {
      // Calculate and store the length of the offset into the base, using the
      // source of the tensorview.
      assert(!symbolTable.count(std::string(TV->getName())) &&
             "Allocation already made!");
      auto *tvSource = getOrigin(TV);
      assert(symbolTable.count(std::string(tvSource->getName())) &&
             "Source allocation not found!");
      runtime::RuntimeSymbolInfo symbol;
      size_t originAddr = symbolTable[std::string(tvSource->getName())].offset;
      size_t offset = calculateTensorViewOffset(TV);

      symbol.offset = originAddr + offset;
      symbol.size = TV->getSizeInBytes();
      symbol.type = *TV->getType();
      symbol.input = false;
      symbol.output = false;
      auto parentCategory = symbolTable.find(std::string(tvSource->getName()))
                                ->second.symbolCategory;
      if (parentCategory == glow::runtime::SymbolCategory::Placeholder) {
        symbol.symbolCategory =
            glow::runtime::SymbolCategory::PlaceholderTensorView;
      } else {
        symbol.symbolCategory =
            glow::runtime::SymbolCategory::ConstantTensorView;
      }
      symbolTable.emplace(std::string(TV->getName()), symbol);
      DEBUG_GLOW(LOG(INFO) << strFormat(
                     "Assigned address to activation %s: %zx (%zd bytes)\n",
                     TV->getName().data(), symbol.offset, symbol.size));
      continue;
    }

    if (auto *D = dyn_cast<DeallocActivationInst>(&I)) {
      assert(symbolTable.count(std::string(D->getAlloc()->getName())) &&
             "Invalid deallocation!");
    }
  }
}

} // namespace glow

runtime::RuntimeBundle
runtime::RuntimeBundle::create(const Function &F,
                               const std::vector<const IRFunction *> &funcs) {
  std::map<std::string, runtime::RuntimeSymbolInfo> symbolTable;
  MemoryAllocator allocator("allocator", 0);
  uint64_t constantsMaxMem = 0, placeholdersMaxMem = 0, activationsMaxMem = 0;

  // Allocate constants.
  allocateConstants(F.getParent()->getConstants(), allocator, symbolTable);
  constantsMaxMem = allocator.getMaxMemoryUsage();

  // Allocate placeholders. Placeholders should be allocated in a order of
  // Input|InputOutput|Output.
  std::vector<const Function *> graphs;
  graphs.reserve(funcs.size());
  for (const auto &f : funcs) {
    graphs.emplace_back(f->getGraph());
  }

  auto contiguousPlaceholders =
      getContiguousPlaceHolder(F.getParent()->getPlaceholders(), graphs);
  allocatePlaceholders(contiguousPlaceholders, allocator, symbolTable);
  placeholdersMaxMem = allocator.getMaxMemoryUsage() - constantsMaxMem;

  // Allocate activations.
  for (const auto &f : funcs) {
    allocateActivations(f->getInstrs(), allocator, symbolTable);
  }

  activationsMaxMem =
      allocator.getMaxMemoryUsage() - constantsMaxMem - placeholdersMaxMem;

  return runtime::RuntimeBundle(symbolTable, constantsMaxMem,
                                placeholdersMaxMem, activationsMaxMem);
}

runtime::RuntimeBundle runtime::RuntimeBundle::create(const Function &F) {
  std::map<std::string, runtime::RuntimeSymbolInfo> symbolTable;

  MemoryAllocator constants("constants", 0);
  MemoryAllocator placeholders("placeholders", 0);

  // Allocate constants.
  allocateConstants(F.findConstants(), constants, symbolTable);

  // Allocate placeholders.
  // Placeholders should be allocated in a order of Input|InputOutput|Output.
  auto contiguousPlaceholders =
      getContiguousPlaceHolder(F.findPlaceholders(), F);

  // Compute the offsets for Placeholders.
  allocatePlaceholders(contiguousPlaceholders, placeholders, symbolTable);

  return runtime::RuntimeBundle(symbolTable, constants.getMaxMemoryUsage(),
                                placeholders.getMaxMemoryUsage(),
                                /*activationsMaxSize*/ 0);
}

runtime::RuntimeBundle
runtime::RuntimeBundle::create(const IRFunction &F,
                               MemoryAllocator &constantAllocator,
                               MemoryAllocator &placeholderAllocator,
                               MemoryAllocator &activationsAllocator) {

  // If all allocators refer to the same underlying allocator, Constants,
  // Placeholders and activations will be allocated contiguously. The maximum
  // memory usage reported by the allocator for each kind of storage will
  // include the memory usage of all previously allocated types of storage and
  // needs to be adjusted accordingly.
  bool contiguous = (&constantAllocator == &placeholderAllocator &&
                     &constantAllocator == &activationsAllocator);
  // Handle Constants, Placeholders, and Activations, in that order.
  // Symbol table mapping symbol name to offset for runtime.
  std::map<std::string, runtime::RuntimeSymbolInfo> symbolTable;

  allocateConstants(F.findConstants(), constantAllocator, symbolTable);
  auto constantMaxSize = constantAllocator.getMaxMemoryUsage();

  // Placeholders should be allocated in a order of Input|InputOutput|Output.
  auto contiguousPlaceholders =
      getContiguousPlaceHolder(F.findPlaceholders(), F);
  // Compute the offsets for Placeholders.
  allocatePlaceholders(contiguousPlaceholders, placeholderAllocator,
                       symbolTable);
  auto placeholderMaxSize = placeholderAllocator.getMaxMemoryUsage();
  if (contiguous) {
    placeholderMaxSize -= constantMaxSize;
  }

  // Compute the offsets for Activations.
  allocateActivations(F.getInstrs(), activationsAllocator, symbolTable);

  auto activationsMaxSize = activationsAllocator.getMaxMemoryUsage();
  if (contiguous) {
    activationsMaxSize -= constantMaxSize + placeholderMaxSize;
    DCHECK_EQ(constantAllocator.getMaxMemoryUsage(),
              constantMaxSize + placeholderMaxSize + activationsMaxSize);
  }

  return runtime::RuntimeBundle(symbolTable, constantMaxSize,
                                placeholderMaxSize, activationsMaxSize);
}

runtime::RuntimeBundle
runtime::RuntimeBundle::create(const IRFunction &F,
                               MemoryAllocator &allocator) {
  return create(F, allocator, allocator, allocator);
}
