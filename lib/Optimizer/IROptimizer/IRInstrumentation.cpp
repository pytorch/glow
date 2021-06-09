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

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/IROptimizer/CommandLine.h"
#include "glow/Optimizer/IROptimizer/IRFunctionPassManager.h"
#include "glow/Optimizer/IROptimizer/IRFunctionPasses.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <fstream>

using namespace glow;

using llvm::isa;

static bool performDebugInstrumentation(IRFunction &M) {

  // Make debug directory path absolute.
  std::string debugDir = instrumentDebugDir;
  if (!llvm::sys::path::is_absolute(debugDir)) {
    llvm::SmallVector<char, 128> path(debugDir.begin(), debugDir.end());
    CHECK(!llvm::sys::fs::make_absolute(path))
        << "Cannot create debug absolute path for '" << debugDir << "'!";
    debugDir = llvm::Twine(path).str();
  }

  // Make debug directory if not exists.
  if (!llvm::sys::fs::is_directory(debugDir)) {
    CHECK(!llvm::sys::fs::create_directory(debugDir))
        << "Cannot create debug directory '" << debugDir << "'!";
  }

  // Debug format.
  std::string debugFormat = instrumentDebugFormat;
  CHECK(debugFormat == "console" || debugFormat == "bin" ||
        debugFormat == "txt" || debugFormat == "rawbin" ||
        debugFormat == "rawtxt")
      << "Invalid debug IR instrumentation format! Only the following formats "
         "are supported: 'console', 'bin', 'txt', 'rawbin' and 'rawtxt'!";
  std::string debugExt =
      ((debugFormat == "bin") || (debugFormat == "rawbin")) ? "bin" : "txt";
  bool debugInfoWrite = (debugFormat != "console");

  // Open debug info file.
  std::string debugInfoPath = debugDir +
                              llvm::sys::path::get_separator().str() +
                              "instrument-debug.info";
  unsigned fileDumpIdx = 0;
  std::ofstream debugInfoFile;
  if (debugInfoWrite) {
    debugInfoFile.open(debugInfoPath, std::ios::out | std::ios::trunc);
    debugInfoFile << "Format: " << debugFormat << "\n";
  }

  // Instrument debug instructions.
  bool changed = false;
  auto &instrs = M.getInstrs();
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto *I = &*it;
    auto next = std::next(it);
    // If current instruction is not one of the provided in the list, skip it.
    if (!instrumentDebugOnly.empty()) {
      if (std::find_if(instrumentDebugOnly.begin(), instrumentDebugOnly.end(),
                       [&](const std::string &name) -> bool {
                         return I->getName().equals(name);
                       }) == instrumentDebugOnly.end()) {
        it = next;
        continue;
      }
    }
    if (isa<DebugPrintInst>(I) || isa<InstrumentInst>(I) ||
        isa<AllocActivationInst>(I) || isa<DeallocActivationInst>(I)) {
      it = next;
      continue;
    }
    // Don't instrument tensorview since it can lead to liveness verification
    // failures if the tensorview happens before any writes to the tensor.
    if (isa<TensorViewInst>(I)) {
      it = next;
      continue;
    }

    // Print instruction info.
    std::string instrName = I->getName().str();
    if (debugInfoWrite) {
      debugInfoFile << "\n";
      debugInfoFile << "Kind: " << I->getKindName() << "\n";
      debugInfoFile << "Name: " << instrName << "\n";
    }

    // Get maximum operand name length for pretty print.
    size_t opNameLenMax = 0;
    for (unsigned opIdx = 0; opIdx < I->getNumOperands(); ++opIdx) {
      auto opNameLen = I->getOperandName(opIdx).size();
      opNameLenMax = std::max(opNameLen, opNameLenMax);
    }

    // Instrument debug operands for current instruction.
    for (unsigned opIdx = 0; opIdx < I->getNumOperands(); ++opIdx) {
      const auto &op = I->getOperand(opIdx);
      const std::string opName = I->getOperandName(opIdx).str();
      const std::string opTypeName = op.first->getType()->toString();

      // DebugPrint instruction name for this operand.
      std::string name = instrName;
      name += ".";
      name += opName;
      name += ".";
      name += I->getKindName();

      // DebugPrint filename. When dumping files we do not use the name of the
      // debug instruction since it is not safe: most of the filesystems allow
      // a maximum length for a given file name in the order of 255 characters
      // and the debug instruction name (with the above format) is very likely
      // to be very long. We use a simple name format for dumping files and we
      // generate a separate meta file with additional information about every
      // dump.
      std::string filename = strFormat("data%04d.", fileDumpIdx++) + debugExt;
      std::string filepath =
          debugDir + llvm::sys::path::get_separator().str() + filename;
      if (debugInfoWrite) {
        unsigned spacing = std::max(opNameLenMax + 2, size_t(10));
        std::string format = "[%d] ";
        format += "%-" + std::to_string(spacing) + "s ";
        format += "%s    ";
        format += "%s\n";
        debugInfoFile << strFormat(format.c_str(), opIdx,
                                   (opName + ":").c_str(), filename.c_str(),
                                   opTypeName.c_str());
      }

      // Dump inputs of the current instruction before the instruction.
      if (op.second != OperandKind::Out) {
        name = "debug_print.before." + name;
        auto *dumpInstr =
            new DebugPrintInst(name, op.first, debugFormat, filepath);
        M.insertInstruction(I, dumpInstr);
        changed = true;
      }

      // Dump outputs of the current instruction after the instruction.
      if (op.second != OperandKind::In) {
        name = "debug_print.after." + name;
        auto *dumpInstr =
            new DebugPrintInst(name, op.first, debugFormat, filepath);
        if (next == e) {
          M.insertInstruction(dumpInstr);
        } else {
          M.insertInstruction(&*next, dumpInstr);
        }
        changed = true;
      }
    }
    it = next;
  }

  // Close debug info file.
  if (debugInfoWrite) {
    debugInfoFile.close();
  }

  return changed;
}

static bool performIRInstrumentation(IRFunction &M) {

  // Open instrument info file.
  std::string instrumentInfoPath = "instrument-ir.info";
  std::ofstream instrumentInfoFile;
  instrumentInfoFile.open(instrumentInfoPath, std::ios::out | std::ios::trunc);
  unsigned instructionID = 0;

  // Instrument instructions.
  bool changed = false;
  auto &instrs = M.getInstrs();
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto *I = &*it;
    auto next = std::next(it);
    std::string instrName = I->getName().str();

    // If current instruction is not one of the provided in the list, skip it.
    if (!instrumentIROnly.empty()) {
      if (std::find_if(instrumentIROnly.begin(), instrumentIROnly.end(),
                       [&](const std::string &name) -> bool {
                         return I->getName().equals(name);
                       }) == instrumentIROnly.end()) {
        it = next;
        continue;
      }
    }

    // Do not instrument other debug or memory related instructions.
    if (isa<DebugPrintInst>(I) || isa<InstrumentInst>(I) ||
        isa<AllocActivationInst>(I) || isa<DeallocActivationInst>(I) ||
        isa<TensorViewInst>(I)) {
      it = next;
      continue;
    }

    // Print instruction info.
    instrumentInfoFile << "ID   : " << instructionID << "\n";
    instrumentInfoFile << "Kind : " << (unsigned)(I->getKind()) << " ("
                       << I->getKindName() << ")\n";
    instrumentInfoFile << "Name : " << instrName << "\n";

    // Get maximum operand name length for pretty print.
    size_t opNameLenMax = 0;
    for (unsigned opIdx = 0; opIdx < I->getNumOperands(); ++opIdx) {
      auto opNameLen = I->getOperandName(opIdx).size();
      opNameLenMax = std::max(opNameLen, opNameLenMax);
    }

    // Print input operands.
    unsigned inputNum = 0;
    for (unsigned opIdx = 0; opIdx < I->getNumOperands(); ++opIdx) {
      const auto &op = I->getOperand(opIdx);
      if (op.second == OperandKind::Out) {
        continue;
      }
      const std::string opName = I->getOperandName(opIdx).str();
      const std::string opTypeName = op.first->getType()->toString();
      unsigned spacing = std::max(opNameLenMax + 2, size_t(10));
      std::string format = "Inp[%d] ";
      format += "%-" + std::to_string(spacing) + "s ";
      format += "%s\n";
      instrumentInfoFile << strFormat(
          format.c_str(), inputNum, (opName + ":").c_str(), opTypeName.c_str());
      inputNum++;
    }

    // Print output operands.
    unsigned outputNum = 0;
    for (unsigned opIdx = 0; opIdx < I->getNumOperands(); ++opIdx) {
      const auto &op = I->getOperand(opIdx);
      if (op.second == OperandKind::In) {
        continue;
      }
      const std::string opName = I->getOperandName(opIdx).str();
      const std::string opTypeName = op.first->getType()->toString();
      unsigned spacing = std::max(opNameLenMax + 2, size_t(10));
      std::string format = "Out[%d] ";
      format += "%-" + std::to_string(spacing) + "s ";
      format += "%s\n";
      instrumentInfoFile << strFormat(format.c_str(), outputNum,
                                      (opName + ":").c_str(),
                                      opTypeName.c_str());
      outputNum++;
    }
    instrumentInfoFile << "\n";

    // Allocation size for the instrumentation. We allocate one buffer to hold
    // the addresses and the sizes for all the input and output operands. We
    // allocate 8 bytes (int64) for each address and size to make sure it fits
    // any target architecture. Allocation size must be strictly positive.
    unsigned allocSize = (inputNum + outputNum) * 2 * sizeof(int64_t);
    allocSize = std::max(1u, allocSize);

    // Add instrumentation before instruction.
    auto *allocTy =
        M.getParent()->uniqueType(ElemKind::Int8QTy, {allocSize}, 0.0, 0);
    auto *instrAlloc =
        new AllocActivationInst("instrument.alloc." + instrName, allocTy);
    auto *instrBefore =
        new InstrumentInst("instrument.before." + instrName, instrAlloc, I,
                           instructionID, InstrumentKind::Before);
    M.insertInstruction(I, instrBefore);
    M.insertInstruction(instrBefore, instrAlloc);

    // Add instrumentation after instruction.
    auto *instrAfter =
        new InstrumentInst("instrument.after." + instrName, instrAlloc, I,
                           instructionID, InstrumentKind::After);
    auto *instrDealloc = new DeallocActivationInst(
        "instrument.dealloc." + instrName, instrAlloc);
    if (next == e) {
      M.insertInstruction(instrAfter);
      M.insertInstruction(instrDealloc);
    } else {
      M.insertInstruction(&*next, instrDealloc);
      M.insertInstruction(instrDealloc, instrAfter);
    }

    instructionID++;
    changed = true;
    it = next;
  }

  // Close instrumentation info file.
  instrumentInfoFile.close();

  return changed;
}

namespace glow {
namespace ir {
bool DebugInstrument::run(IRFunction *M, const CompilationContext &cctx) {
  return performDebugInstrumentation(*M);
}
bool IRInstrument::run(IRFunction *M, const CompilationContext &cctx) {
  return performIRInstrumentation(*M);
}
} // namespace ir
} // namespace glow
