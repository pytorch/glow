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
#include "glow/Base/Traits.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "glow/Support/Memory.h"
#include <glog/logging.h>
#include <memory>
#include <string>

#define DEBUG_TYPE "memory-regions"

using namespace glow;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

extern llvm::cl::opt<bool> reuseActivationsMemory;

namespace glow {
const char *getSymbolCategoryName(runtime::SymbolCategory category) {
  switch (category) {
  case runtime::SymbolCategory::Activation:
    return "activation";
  case runtime::SymbolCategory::Constant:
    return "constant";
  case runtime::SymbolCategory::Placeholder:
    return "placeholder";
  case runtime::SymbolCategory::ConstantTensorView:
    return "constant_tensor_view";
  case runtime::SymbolCategory::PlaceholderTensorView:
    return "placeholder_tensor_view";
  case runtime::SymbolCategory::Unknown:
    return "unknown";
  }
  llvm_unreachable("Unknown SymbolCategory");
  return nullptr;
}
} // namespace glow

/// \returns a Named object corresponding to the Kinded object whose address is
/// provided by \p handle.
const Named *glow::getNamed(const glow::Kinded *kinded) {
  if (auto storage = dyn_cast<glow::Storage>(kinded)) {
    return storage;
  }
  if (auto wv = dyn_cast<glow::WeightVar>(kinded)) {
    return wv;
  }
  if (auto val = dyn_cast<glow::AllocActivationInst>(kinded)) {
    return val;
  }
  if (auto val = dyn_cast<glow::TensorViewInst>(kinded)) {
    return val;
  }
  if (auto ph = dyn_cast<glow::Placeholder>(kinded)) {
    return ph;
  }
  LOG(FATAL) << "Unknown kind of a Kinded object: " << (int)kinded->getKind()
             << " for object: " << strFormat("0x%p", kinded);
}

runtime::RuntimeSymbolInfo &
getSymbolFromHandle(glow::Allocation::Handle handle) {
  const auto *symbolPtr =
      reinterpret_cast<const runtime::RuntimeSymbolInfo *>(handle);
  return *const_cast<runtime::RuntimeSymbolInfo *>(symbolPtr);
}

void assignAddressesToTensorViews(runtime::MemoryRegion &region) {
  auto &symbolTable = region.getSymbolTable();
  // Then process tensorview symbols.
  for (auto &pair : symbolTable) {
    auto &symbol = *pair.second;
    if (auto *TVI = llvm::dyn_cast<TensorViewInst>(symbol.val)) {
      auto *tvSource = getOrigin(TVI);
      size_t originAddr = symbolTable[std::string(tvSource->getName())]->offset;
      size_t offset = calculateTensorViewOffset(TVI);
      symbol.offset = originAddr + offset;
      DEBUG_GLOW(const auto *named = getNamed(symbol.val);
                 LOG(INFO) << strFormat(
                     "Assigned address to %s '%s': 0x%zx (%zd bytes)\n",
                     getSymbolCategoryName(symbol.symbolCategory),
                     named->getName().str().c_str(), symbol.offset,
                     symbol.size));
    }
  }
}

/// Allocate space for the activations of \p instrs using \p allocator and store
/// the resultant symbols in \p symbolTable.
void allocateWithoutPreservingAllocationOrder(runtime::MemoryRegion &region,
                                              MemoryAllocator &allocator) {
  auto &symbolTable = region.getSymbolTable();
  // Gather allocation/deallocation sequence.
  auto &memRegionAllocList = region.getAllocationList();
  // List of allocations to be processed during the current round of allocation.
  // Lazily initialize it only if there were previous rounds of allocation.
  std::list<Allocation> currentAllocList;
  if (region.getMemSize()) {
    for (auto &alloc : memRegionAllocList) {
      auto &symbol = getSymbolFromHandle(alloc.handle);
      // Ignore any symbols that got addresses assgigned already e.g. during the
      // previous rounds of allocation.
      if (symbol.isAllocated()) {
        continue;
      }
      currentAllocList.emplace_back(alloc);
    }
  }
  // Reference to the set of allocations to be processed.
  std::list<Allocation> &allocList =
      region.getMemSize() ? currentAllocList : memRegionAllocList;

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
    MemoryAllocator::Handle activationsHandle = &allocList;
    activationsBaseAddr =
        allocator.allocate(activationsSize, activationsHandle);
    if (reuseActivationsMemory) {
      allocator.deallocate(activationsHandle);
    }
  }

  // Map addresses of allocated symbols.
  for (auto &pair : symbolTable) {
    auto &symbol = *pair.second;
    // Skip symbols that were assigned addresses in previous rounds.
    if (symbol.isAllocated()) {
      continue;
    }
    // Process non-tensorview symbols.
    if (isa<TensorViewInst>(symbol.val)) {
      continue;
    }
    size_t addr =
        activationsBaseAddr + activationsAllocator.getAddress(&symbol);
    auto size = symbol.type.getSizeInBytes();
    symbol.offset = addr;
    symbol.size = size;
    DEBUG_GLOW(const auto *named = getNamed(symbol.val);
               LOG(INFO) << strFormat(
                   "Assigned address to %s '%s': 0x%zx (%zd bytes)\n",
                   getSymbolCategoryName(symbol.symbolCategory),
                   named->getName().str().c_str(), symbol.offset, symbol.size));
  }
}

void allocatePreservingAllocationOrder(runtime::MemoryRegion &region,
                                       MemoryAllocator &allocator) {
  bool canReuseMemory = region.getMemoryRegionDescription()->canReuseMemory();
  // Allocate memory based on the order of allocations in the allocList.
  for (auto &alloc : region.getAllocationList()) {
    auto &symbol = getSymbolFromHandle(alloc.handle);
    // Same symbol (e.g. a constant) may be used multiple times by different
    // functions. But it should be assigned an address only once.
    if (symbol.isAllocated()) {
      continue;
    }
    if (alloc.alloc) {
      // Perform allocation.
      auto size = symbol.type.getSizeInBytes();
      auto offset = allocator.allocate(size, symbol.val);
      symbol.offset = offset;
      symbol.size = size;
      DEBUG_GLOW(LOG(INFO) << strFormat(
                     "Assigned address to %s '%s': 0x%zx (%zd bytes)\n",
                     getSymbolCategoryName(symbol.symbolCategory),
                     symbol.name.c_str(), symbol.offset, symbol.size));
      continue;
    }
    // It is a deallocation.
    if (!canReuseMemory) {
      continue;
    }
    // Perform deallocation.
    allocator.deallocate(alloc.handle);
  }
}

void glow::runtime::MemoryRegion::allocate(bool isExclusiveMemoryAllocator) {
  MemoryAllocator *allocator{nullptr};
  if (memAllocator_) {
    allocator = memAllocator_;
  } else {
    // Create a memory region local allocator if none was provided.
    allocator = new MemoryAllocator(name_, 0);
  }
  uint64_t initialMemSize = allocator->getMaxMemoryUsage();
  DEBUG_GLOW(llvm::dbgs() << strFormat(
                 "MemoryRegion::alloc for region %s. Region size at entry: "
                 "%lu. Memory allocator max mem usage at entry: %lu. Memory "
                 "allocator is %s.\n",
                 getName().c_str(), getMemSize(), initialMemSize,
                 (memAllocator_ ? "reused" : "new")));
  bool canReuseMemory = getMemoryRegionDescription()->canReuseMemory();
  // Reserve some memory at the beginning if required.
  // If there is a user-provided allocator, assume that it has reserved this
  // amount of the memory at the front.
  uint64_t reservedAtFrontMemSize = 0;
  if (isExclusiveMemoryAllocator && !getMemSize() &&
      hasAttribute("alloc.reserve_at_front")) {
    reservedAtFrontMemSize = std::stol(getAttribute("alloc.reserve_at_front"));
    allocator->allocate(reservedAtFrontMemSize, this);
  }
  // If this region was allocated before, e.g. when it contains some constants
  // that are used by multiple IR functions, then old allocations should not be
  // changed. To enforce this, reserve the current size of the region at
  // the beginning of a freshly created allocator.
  if (isExclusiveMemoryAllocator && getMemSize() && !canReuseMemory) {
    allocator->allocate(getMemSize(), this);
  }
  if (!desc_->isPreserveAllocationOrder()) {
    // Allocate memory ignoring the order of allocations in the allocList.
    allocateWithoutPreservingAllocationOrder(*this, *allocator);
  } else {
    // Allocate memory preserving the order of allocations in the allocList.
    allocatePreservingAllocationOrder(*this, *allocator);
  }
  // Then process tensorview symbols and assign addresses to them.
  assignAddressesToTensorViews(*this);
  // Reserve some memory at the end if required.
  uint64_t reservedAtBackMemSize = 0;
  if (hasAttribute("alloc.reserve_at_back")) {
    reservedAtBackMemSize = std::stol(getAttribute("alloc.reserve_at_back"));
    allocator->allocate(reservedAtBackMemSize, this);
  }
  uint64_t finalMemSize = allocator->getMaxMemoryUsage();
  CHECK_GE(finalMemSize, initialMemSize) << "Region memory size is wrong";
  if (isExclusiveMemoryAllocator && initialMemSize > 0 && !memSize_) {
    // It is a first round of allocation for this region. Thus the
    // initialMemSize should be considered a part of this region.
    initialMemSize = 0;
  }
  memSize_ = finalMemSize - initialMemSize + reservedAtFrontMemSize +
             reservedAtBackMemSize;
  if (!memAllocator_) {
    delete allocator;
  }
}

bool runtime::MemoryRegionDescriptions::verify() const {
  std::unordered_map<runtime::MemoryRegionId,
                     const runtime::MemoryRegionDescription *>
      ids;
  for (auto &memRegionDesc : descriptions) {
    // Each description should have a region id.
    if (memRegionDesc->getMemRegionId() != MemoryRegions::UnknownMemoryRegion) {
      CHECK_EQ(ids.count(memRegionDesc->getMemRegionId()), 0)
          << "Memory region id " << memRegionDesc->getMemRegionId()
          << " for region " << memRegionDesc->getName()
          << " was used for another region: "
          << ids[memRegionDesc->getMemRegionId()]->getName();
      // Remember that this id was used.
      ids[memRegionDesc->getMemRegionId()] = memRegionDesc.get();
    }
  }
  return true;
}

/// \returns new memory region created based on a memory region description \p
/// desc.
static runtime::MemoryRegion *
createMemoryRegion(std::shared_ptr<runtime::MemoryRegionDescription> desc,
                   runtime::MemoryRegionId nextRegionId) {
  auto memRegion = new runtime::MemoryRegion;

  memRegion->setId(desc->getMemRegionId());
  memRegion->setAttributes(desc->getAttributes());
  if (memRegion->getId() == runtime::MemoryRegions::UnknownMemoryRegion) {
    // Assign a unique id.
    memRegion->setName(desc->getName() + "_" + std::to_string(nextRegionId));
    memRegion->setId(nextRegionId);
  } else {
    memRegion->setName(desc->getName());
  }
  memRegion->setMemoryRegionDescription(desc);
  return memRegion;
}

/// \returns memory region which should contain \p T based on the descriptions
/// form \p memRegionDescriptions and memory region table \p memRegionTable. It
/// can also consult \p symbolTable if needed.
///
/// This code supports use-cases where multiple memory regions can
/// be produced from the same description, e.g. if each  placeholder should live
/// in its own memory region.
template <class T>
static std::shared_ptr<runtime::MemoryRegion> getOrCreateMatchingMemRegion(
    runtime::MemoryRegionTableTy &memRegionTable,
    const runtime::MemoryRegionDescriptions &memRegionDescriptions,
    const runtime::SymbolTableTy &symbolTable, T val) {
  const auto valName = getNamed(val)->getName().str();
  // If the symbol with a given name is in the symbol table already, retun its
  // memory region.
  if (symbolTable.count(valName)) {
    return memRegionTable[symbolTable.at(valName)->getMemRegionId()];
  }
  // Check which memroy region description matches val.
  for (auto &memRegionDesc : memRegionDescriptions.getDescriptions()) {
    // Is it a matching memory region description?
    if (memRegionDesc->contains(val)) {
      // Find an existing matching memory region for this description.
      std::shared_ptr<runtime::MemoryRegion> memRegion;
      if (memRegionDesc->getMemRegionId() !=
          runtime::MemoryRegions::UnknownMemoryRegion) {
        // There can be only one memory region with a specific region id.
        auto memRegionIt = memRegionTable.find(memRegionDesc->getMemRegionId());
        if (memRegionIt != memRegionTable.end()) {
          memRegion = memRegionIt->second;
        }
      } else {
        // Try to find a first region that was created from this description.
        // Multiple memory regions corresponding to a given description may
        // exist.
        for (auto &pair : memRegionTable) {
          if (pair.second->getMemoryRegionDescription() == memRegionDesc) {
            memRegion = pair.second;
          }
        }
      }
      if (memRegion) {
        // If it is not a per-buffer region, it should contain this \p val.
        if (!memRegion->isPerBuffer()) {
          return memRegion;
        }
        // Check if this region contains val already.
        if (memRegion->contains(valName)) {
          return memRegion;
        }
        // If region is a per instance region and it is empty, then it can be
        // used for \p val.
        if (memRegion->isPerBuffer() && memRegion->getSymbolTable().empty()) {
          return memRegion;
        }
      }
      // Add a new memory region if needed.
      auto maxRegionId = memRegionTable.empty()
                             ? 0
                             : (std::prev(memRegionTable.end())->first + 1);
      memRegion = std::shared_ptr<runtime::MemoryRegion>(
          createMemoryRegion(memRegionDesc, maxRegionId));
      CHECK_EQ(memRegionTable.count(memRegion->getId()), 0)
          << "Memory region with id " << memRegion->getId()
          << " exists already";
      memRegionTable[memRegion->getId()] = memRegion;
      return memRegion;
    }
  }
  return nullptr;
}

template <typename FUN>
void glow::createMemoryRegionTableForConstantsAndPlaceholders(
    const FUN &F,
    const runtime::MemoryRegionDescriptions &memRegionDescriptions,
    runtime::MemoryRegionTableTy &memRegionTable,
    runtime::SymbolTableTy &symbolTable,
    std::function<const Kinded *(const Storage *)> getValueForNode) {
  memRegionDescriptions.verify();

  /// Try to always create default regions to be backwards compatible.
  std::array<runtime::MemoryRegionId, 3> defaultRegionIds{
      runtime::MemoryRegions::Activation,
      runtime::MemoryRegions::ConstantWeight,
      runtime::MemoryRegions::MutableWeight};
  for (auto defaultRegionId : defaultRegionIds) {
    // Do not add a region if it is defined already. This can happen if an IR
    // module contains multiple IR functions.
    if (memRegionTable.count(defaultRegionId)) {
      continue;
    }
    const auto desc =
        memRegionDescriptions.getMemoryRegionDescription(defaultRegionId);
    if (desc) {
      auto maxRegionId = memRegionTable.empty()
                             ? 0
                             : (std::prev(memRegionTable.end())->first + 1);
      auto memRegion = std::shared_ptr<runtime::MemoryRegion>(
          createMemoryRegion(desc, maxRegionId));
      memRegionTable[memRegion->getId()] = memRegion;
    }
  }

  auto constants = F.findConstants();
  auto contiguousPlaceholders =
      getContiguousPlaceHolder(F.findPlaceholders(), F);

  // Assign each buffer to a memory region. Each tensor should belong to a
  // memory region.

  // Add constants to memory regions.
  for (auto &C : constants) {
    auto *c = getValueForNode(C);
    if (auto memRegion = getOrCreateMatchingMemRegion(
            memRegionTable, memRegionDescriptions, symbolTable, c)) {
      auto symbolPtr = std::make_shared<runtime::RuntimeSymbolInfo>();
      auto &symbol = *symbolPtr;
      symbol.name = C->getName().str();
      symbol.symbolCategory = runtime::SymbolCategory::Constant;
      symbol.type = *C->getType();
      symbol.input = false;
      symbol.output = false;
      symbol.val = c;
      // If a module contains multiple functions, constants will be processed
      // multiple times. Do not insert them into the symbol table and region
      // twice.
      if (memRegion->contains(symbol.name)) {
        auto &memRegionSymbolTable = memRegion->getSymbolTable();
        auto &memRegionSymbol = *memRegionSymbolTable[symbol.name];
        if (symbol.type == memRegionSymbol.type &&
            symbol.input == memRegionSymbol.input &&
            symbol.output == memRegionSymbol.output) {
          continue;
        }
        LOG(FATAL) << strFormat(
            "Symbol %s is registered multiple times in the memory region: %s",
            symbol.name.c_str(), memRegion->getName().c_str());
      }
      CHECK_EQ(memRegion->contains(symbol.name), 0)
          << "Symbol is already defined with different settings: "
          << symbol.name;
      memRegion->add(symbolPtr);
      memRegion->getAllocationList().emplace_back(&symbol, true,
                                                  symbol.type.getSizeInBytes());
      CHECK_EQ(symbolTable.count(symbol.name), 0)
          << "Symbol is already defined: " << symbol.name;
      symbolTable[symbol.name] = symbolPtr;
      continue;
    }
    LOG(FATAL) << "Constant " << C->getName().str()
               << " does not belong to any memory region";
  }

  // Add placeholders to memory regions.
  for (auto &PH : contiguousPlaceholders) {
    auto *wv = getValueForNode(PH.addr);
    auto *wvNamed = getNamed(wv);
    // The following invariant is always true for IRs produced by the compiler,
    // but can be broken by hand-written tests which do no obey these rules.
    // CHECK_EQ(wv->getName().str(), PH.addr->getName().str())
    //     << "Placeholder and WeightVar should have the same name";
    if (auto memRegion = getOrCreateMatchingMemRegion(
            memRegionTable, memRegionDescriptions, symbolTable, wv)) {
      auto symbolPtr = std::make_shared<runtime::RuntimeSymbolInfo>();
      auto &symbol = *symbolPtr;
      symbol.name = wvNamed->getName().str();
      symbol.symbolCategory = runtime::SymbolCategory::Placeholder;
      symbol.type = *PH.addr->getType();
      symbol.input = PH.isInput;
      symbol.output = PH.isOutput;
      symbol.val = wv;
      CHECK_EQ(memRegion->contains(symbol.name), 0)
          << "Symbol is already defined: " << symbol.name;
      memRegion->add(symbolPtr);
      memRegion->getAllocationList().emplace_back(&symbol, true,
                                                  symbol.type.getSizeInBytes());
      CHECK_EQ(symbolTable.count(symbol.name), 0)
          << "Symbol is already defined: " << symbol.name;
      symbolTable[PH.addr->getName().str()] = symbolPtr;
      continue;
    }
    LOG(FATAL) << "Placeholder " << PH.addr->getName().str()
               << " does not belong to any memory region";
  }

  if (F.getIRKind() == IRKind::GlowGraphIRKind) {
    // Dump memory region table
    DEBUG_GLOW(dumpMemoryRegionTable(llvm::dbgs(), memRegionTable));
  }
}

/// Explicitly instantiate a function for specific argument types.
template void glow::createMemoryRegionTableForConstantsAndPlaceholders(
    const Function &F,
    const runtime::MemoryRegionDescriptions &memRegionDescriptions,
    runtime::MemoryRegionTableTy &memRegionTable,
    runtime::SymbolTableTy &symbolTable,
    std::function<const Kinded *(const Storage *)> getValueForNode);

template void glow::createMemoryRegionTableForConstantsAndPlaceholders(
    const IRFunction &F,
    const runtime::MemoryRegionDescriptions &memRegionDescriptions,
    runtime::MemoryRegionTableTy &memRegionTable,
    runtime::SymbolTableTy &symbolTable,
    std::function<const Kinded *(const Storage *)> getValueForNode);

void glow::createMemoryRegionTable(
    const IRFunction &F,
    const runtime::MemoryRegionDescriptions &memRegionDescriptions,
    runtime::MemoryRegionTableTy &memRegionTable,
    runtime::SymbolTableTy &symbolTable) {
  auto getValueForNode = [&](const Storage *v) -> const Kinded * {
    return F.getWeightForNode(v);
  };
  createMemoryRegionTableForConstantsAndPlaceholders<>(
      F, memRegionDescriptions, memRegionTable, symbolTable, getValueForNode);
  // Add activations to memory regions.
  auto &instrs = F.getInstrs();
  for (const auto &I : instrs) {
    // Allocation.
    const glow::Value *A{nullptr};
    // Origin of the allocation.
    const glow::Value *origin{nullptr};
    bool isAlloc{false};
    bool isNewSymbol{false};
    bool isTensorView{false};
    // Look for allocations, deallocations and tensorviews.
    switch (I.getKind()) {
    case glow::Kinded::Kind::AllocActivationInstKind: {
      A = llvm::cast<AllocActivationInst>(&I);
      origin = A;
      isAlloc = true;
      isNewSymbol = true;
      break;
    }
    case glow::Kinded::Kind::DeallocActivationInstKind: {
      const auto *D = llvm::cast<DeallocActivationInst>(&I);
      A = D->getAlloc();
      origin = A;
      break;
    }
    case glow::Kinded::Kind::TensorViewInstKind: {
      const auto *TVI = llvm::cast<TensorViewInst>(&I);
      A = TVI;
      origin = getOrigin(TVI);
      isNewSymbol = true;
      isTensorView = true;
      break;
    }
    default:
      continue;
    }
    if (auto memRegion = getOrCreateMatchingMemRegion(
            memRegionTable, memRegionDescriptions, symbolTable, origin)) {
      if (isNewSymbol) {
        auto symbolPtr = std::make_shared<runtime::RuntimeSymbolInfo>();
        auto &symbol = *symbolPtr;
        symbol.name = A->getName().str();
        symbol.symbolCategory = runtime::SymbolCategory::Activation;
        symbol.type = *A->getType();
        symbol.input = false;
        symbol.output = false;
        symbol.val = A;
        // Logic to figure out the category if a TensorView.
        if (isTensorView) {
          auto parentCategory =
              symbolTable.find(std::string(origin->getName()))
                  ->second->symbolCategory;
          if (parentCategory == glow::runtime::SymbolCategory::Placeholder) {
            symbol.symbolCategory =
                glow::runtime::SymbolCategory::PlaceholderTensorView;
          } else {
            symbol.symbolCategory =
                glow::runtime::SymbolCategory::ConstantTensorView;
          }
        }
        CHECK_EQ(memRegion->contains(symbol.name), 0)
            << "Symbol is already defined: " << symbol.name;
        memRegion->add(symbolPtr);
        CHECK_EQ(symbolTable.count(symbol.name), 0)
            << "Symbol is already defined: " << symbol.name;
        symbolTable[symbol.name] = symbolPtr;
      }
      // Remember if it is allocation or deallocation.
      if (!isTensorView) {
        auto &symbol = *symbolTable[A->getName().str()];
        memRegion->getAllocationList().emplace_back(&symbol, isAlloc,
                                                    A->getSizeInBytes());
      }
      continue;
    }
    LOG(FATAL) << "Activation " << A->getName().str()
               << " does not belong to any memory region";
  }
  // Set of observed deallocations.
  std::set<const void *> seenDeallocs;
  // Iterate over all memory regions, add missing deallocations and sort the
  // order of allocations if required.
  for (auto &pair : memRegionTable) {
    auto &memRegion = pair.second;
    seenDeallocs.clear();
    // Collect info about deallocs that are present in the current region.
    for (auto &alloc : memRegion->getAllocationList()) {
      if (!alloc.alloc) {
        seenDeallocs.insert(alloc.handle);
      }
    }
    // Add missing deallocs at the end if needed.
    for (auto &symbolPair : memRegion->getSymbolTable()) {
      auto &symbol = *symbolPair.second;
      if (!isa<TensorViewInst>(symbol.val) && !seenDeallocs.count(&symbol)) {
        memRegion->getAllocationList().emplace_back(&symbol, false, 0);
      }
    }
    // Sort the order of allocations if needed.
    if (!memRegion->getMemoryRegionDescription()->canReuseMemory()) {
      // Move deallocations to the end.
      std::vector<Allocation> allocs(memRegion->getAllocationList().size());
      allocs.assign(memRegion->getAllocationList().begin(),
                    memRegion->getAllocationList().end());
      std::stable_sort(
          allocs.begin(), allocs.end(),
          [](const Allocation &lhs, const Allocation &rhs) -> bool {
            return (lhs.alloc && !rhs.alloc);
          });
      memRegion->getAllocationList().assign(allocs.begin(), allocs.end());
    }
  }

  // Dump memory region table
  DEBUG_GLOW(dumpMemoryRegionTable(llvm::dbgs(), memRegionTable));
}

std::shared_ptr<runtime::MemoryRegionDescriptions>
glow::getDefaultMemoryRegionDescriptions() {
  auto defaultMemRegionDescriptions =
      std::make_shared<runtime::MemoryRegionDescriptions>();
  auto defaultConstantWeightRegionDescription =
      std::make_shared<runtime::ConstantWeightMemoryRegionDescription>();
  auto defaultMutableWeightRegionDescription =
      std::make_shared<runtime::MutableWeightMemoryRegionDescription>();
  auto defaultActivationRegionDescription =
      std::make_shared<runtime::ActivationMemoryRegionDescription>();
  defaultConstantWeightRegionDescription->setMemRegionId(
      runtime::MemoryRegions::ConstantWeight);
  defaultConstantWeightRegionDescription->setName("ConstantWeights");
  // constant weight region cannot reuse memory.
  defaultConstantWeightRegionDescription->setAttribute("no_memory_reuse",
                                                       "true");
  defaultConstantWeightRegionDescription->setAttribute("buffer_kind",
                                                       "constants");

  defaultMutableWeightRegionDescription->setMemRegionId(
      runtime::MemoryRegions::MutableWeight);
  defaultMutableWeightRegionDescription->setName("MutableWeights");
  // Mutable weight region cannot reuse memory.
  defaultMutableWeightRegionDescription->setAttribute("no_memory_reuse",
                                                      "true");
  defaultMutableWeightRegionDescription->setAttribute("buffer_kind",
                                                      "inputs_outputs");

  defaultActivationRegionDescription->setMemRegionId(
      runtime::MemoryRegions::Activation);
  defaultActivationRegionDescription->setName("Activations");
  // Activations can reuse memory.
  defaultActivationRegionDescription->setAttribute(
      "no_memory_reuse", reuseActivationsMemory ? "false" : "true");
  // Activation region doesn't need to preserve the allocation order.
  defaultActivationRegionDescription->setAttribute("preserve_allocation_order",
                                                   "false");
  defaultActivationRegionDescription->setAttribute("buffer_kind", "temps");

  defaultMemRegionDescriptions->append(
      std::move(defaultConstantWeightRegionDescription));
  defaultMemRegionDescriptions->append(
      std::move(defaultMutableWeightRegionDescription));
  defaultMemRegionDescriptions->append(
      std::move(defaultActivationRegionDescription));
  return defaultMemRegionDescriptions;
}

runtime::MemoryRegionDescriptions &runtime::MemoryRegionDescriptions::append(
    std::shared_ptr<MemoryRegionDescription> desc) {
  descriptions.emplace_back(desc);
  return *this;
}

runtime::MemoryRegionDescriptions &runtime::MemoryRegionDescriptions::prepend(
    std::shared_ptr<MemoryRegionDescription> desc) {
  descriptions.emplace_front(desc);
  return *this;
}

std::shared_ptr<runtime::MemoryRegionDescription>
runtime::MemoryRegionDescriptions::getMemoryRegionDescription(
    MemoryRegionId id) const {
  CHECK_NE(id, MemoryRegions::UnknownMemoryRegion)
      << "Unknown memory region id should not be used";
  for (auto &desc : descriptions) {
    if (desc->getMemRegionId() == id) {
      return desc;
    }
  }
  return nullptr;
}

std::shared_ptr<runtime::MemoryRegionDescription>
runtime::MemoryRegionDescriptions::getMemoryRegionDescription(
    const std::string &name) const {
  std::shared_ptr<runtime::MemoryRegionDescription> resultDesc;
  for (auto &desc : descriptions) {
    if (desc->getName() == name) {
      CHECK(!resultDesc)
          << "Multiple memory region descriptions with the same name";
      resultDesc = desc;
    }
  }
  return resultDesc;
}

bool runtime::ConstantWeightMemoryRegionDescription::contains(
    const glow::Kinded *val) const {
  if (llvm::isa<Constant>(val)) {
    return true;
  }
  if (auto *WV = llvm::dyn_cast<WeightVar>(val)) {
    return WV->isConstant();
  }
  return false;
}

bool runtime::MutableWeightMemoryRegionDescription::contains(
    const glow::Kinded *val) const {
  if (llvm::isa<Placeholder>(val)) {
    return true;
  }
  if (auto *WV = llvm::dyn_cast<WeightVar>(val)) {
    return !WV->isConstant();
  }
  return false;
}

bool runtime::MutableInputWeightMemoryRegionDescription::contains(
    const glow::Kinded *val) const {
  if (auto *WV = llvm::dyn_cast<WeightVar>(val)) {
    return !WV->isConstant() && isInput(WV);
  }
  return false;
}

bool runtime::MutableOutputWeightMemoryRegionDescription::contains(
    const glow::Kinded *val) const {
  if (auto *WV = llvm::dyn_cast<WeightVar>(val)) {
    return !WV->isConstant() && isOutput(WV);
  }
  return false;
}

bool runtime::ActivationMemoryRegionDescription::contains(
    const glow::Kinded *val) const {
  return llvm::isa<AllocActivationInst>(val);
}

const Kinded *getKindedOrigin(const Kinded *v) {
  if (auto *TVI = dyn_cast<TensorViewInst>(v)) {
    return getOrigin(TVI);
  }
  return v;
}

void runtime::MemoryRegion::add(std::shared_ptr<RuntimeSymbolInfo> symbol) {
  CHECK(!symbol->memRegion) << "Symbol cannot belong to multiple regions";
  CHECK(desc_->contains(getKindedOrigin(symbol->val)))
      << "Region should contain the symbol";
  // Add symbol to the set of symbols.
  symbolTable_[symbol->name] = symbol;
  // Remember inside the symbol information which region it belongs to.
  symbol->memRegion = this;
}

bool runtime::MemoryRegionDescription::isPerBuffer() const {
  auto attr = attrs_.find("region_per_buffer");
  if (attr == attrs_.end()) {
    return false;
  }
  return attr->second == "true";
}

bool runtime::MemoryRegionDescription::isPreserveAllocationOrder() const {
  auto attr = attrs_.find("preserve_allocation_order");
  if (attr == attrs_.end()) {
    return true;
  }
  return attr->second == "true";
}

bool runtime::MemoryRegionDescription::canReuseMemory() const {
  auto attr = attrs_.find("no_memory_reuse");
  if (attr == attrs_.end()) {
    return false;
  }
  return attr->second == "false";
}

bool runtime::MemoryRegion::isPerBuffer() const { return desc_->isPerBuffer(); }

bool runtime::MemoryRegion::contains(llvm::StringRef name) const {
  return symbolTable_.count(name.str()) != 0;
}

void runtime::MemoryRegion::dump(llvm::raw_ostream &out) const {
  out << "MemoryRegion {\n";
  out << "name: " << name_ << "\n";
  out << "id: " << id_ << "\n";
  out << "attributes {\n";
  for (auto &pair : attrs_) {
    out << pair.first << " = " << pair.second << "\n";
  }
  out << "}\n";
  out << "symbols {\n";
  for (auto &pair : symbolTable_) {
    pair.second->dump(out);
  }
  out << "}\n";
  out << "allocations order {\n";
  for (auto &alloc : allocList_) {
    auto &name = getSymbolFromHandle(alloc.handle).name;
    out << ((alloc.alloc) ? "allocate" : "deallocate") << " " << name
        << " size: " << alloc.size << "\n";
  }
  out << "}\n";
  out << "Memory size: " << getMemSize() << "\n";
  out << "}\n";
}

void runtime::MemoryRegion::dump() const { dump(llvm::outs()); }

std::string runtime::MemoryRegion::toString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dump(os);
  return os.str();
}

void runtime::MemoryRegionDescription::dump(llvm::raw_ostream &out) const {
  out << "MemoryRegionDescription {\n";
  out << "name: " << name_ << "\n";
  out << "id: " << id_ << "\n";
  out << "attributes {\n";
  for (auto &pair : attrs_) {
    out << pair.first << " = " << pair.second << "\n";
  }
  out << "}\n";
  // TODO: Dump rules.
  out << "}\n";
}

void runtime::MemoryRegionDescription::dump() const { dump(llvm::outs()); }

std::string runtime::MemoryRegionDescription::toString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dump(os);
  return os.str();
}

void runtime::MemoryRegionDescriptions::dump(llvm::raw_ostream &out) const {
  out << "MemoryRegionDescriptions {\n";
  for (auto &memRegionDesc : descriptions) {
    memRegionDesc->dump(out);
  }
  out << "}\n";
}

void runtime::MemoryRegionDescriptions::dump() const { dump(llvm::outs()); }

std::string runtime::MemoryRegionDescriptions::toString() const {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  dump(os);
  return os.str();
}
