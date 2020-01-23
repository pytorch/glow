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

#include "glow/Backend/Backend.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Hook.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/Utils.h"

#include "../../lib/Backends/CPU/CPUBackend.h"
#include "../../lib/Backends/CPU/CPULLVMIRGen.h"
#include "glow/IR/Instrs.h"

#include "gtest/gtest.h"

using namespace glow;

//==============================================================
// Test main classes
//==============================================================

/// We compile the standard library (libjit) to LLVM bitcode, and then convert
/// that binary data to an include file using an external utility (include-bin).
/// The resulting file is included here to compile the bitcode image into our
/// library.
static const unsigned char libjit_bc[] = {
#include "glow/CPU/test_libjit_bc.inc"
};
static const size_t libjit_bc_size = sizeof(libjit_bc);

class MockLLVMIRGen : public CPULLVMIRGen {

  // JIT test specific codegen method.
  bool generateJITTESTIRForInstr(llvm::IRBuilder<> &builder,
                                 const glow::Instruction *I) {
    if (I->getKind() == Kinded::Kind::ElementLogInstKind) {
      auto *AN = llvm::cast<ElementLogInst>(I);
      auto *src = AN->getSrc();
      auto *dest = AN->getDest();
      auto *srcPtr = emitValueAddress(builder, src);
      auto *destPtr = emitValueAddress(builder, dest);
      auto *F = getFunction("JITTestDispatch");
      createCall(builder, F, {srcPtr, destPtr});
      return true;
    }

    return false;
  }

  virtual void generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                      const glow::Instruction *I) override {
    // Try to generate test version of instruction.
    if (generateJITTESTIRForInstr(builder, I)) {
      return;
    }

    // Fall back to LLVMIRGen.
    LLVMIRGen::generateLLVMIRForInstr(builder, I);
  }

  bool
  canBePartOfDataParallelKernel(const glow::Instruction *I) const override {
    return false;
  }

public:
  MockLLVMIRGen(const IRFunction *F, AllocationsInfo &allocationsInfo,
                std::string mainEntryName, llvm::StringRef libjitBC)
      : CPULLVMIRGen(F, allocationsInfo, mainEntryName, libjitBC) {}
};

class MockCPUBackend : public CPUBackend {
  virtual std::unique_ptr<LLVMIRGen>
  createIRGen(const IRFunction *IR,
              AllocationsInfo &allocationsInfo) const override {
    MockLLVMIRGen *irgen =
        new MockLLVMIRGen(IR, allocationsInfo, "", getLibjitBitcode());
    return std::unique_ptr<MockLLVMIRGen>(irgen);
  }
  virtual std::string getBackendName() const override {
    return "MockCPUBackend";
  }

public:
  virtual llvm::StringRef getLibjitBitcode() const override {
    return llvm::StringRef(reinterpret_cast<const char *>(libjit_bc),
                           libjit_bc_size);
  }
  static std::string getName() { return "MockCPUBackend"; }
};

REGISTER_GLOW_BACKEND_FACTORY(MockCPUFactory, MockCPUBackend);

//==============================================================
// Actual tests
//
// For JIT tests, we take the following approach:
// - override the code generation for a particular node (log)
// - the log input value determines the index of libjit function
//   that must be called
// - in case there are later change in the Glow that may result
//   in a IR/code generation change, we check that the expected
//   JIT function was called by checking that the log output
//   value is JIT_MAGIC_VALUE + <log input>
//==============================================================

#define JIT_MAGIC_VALUE 555

// Test 0: test that the libjit code can use global C++ objects.
TEST(CPUJITTest, testCppConstructors) {
  glow::ExecutionEngine EE("MockCPUBackend");
  Module &M = EE.getModule();
  Function *F = M.createFunction("F");

  // Create a simple graph.
  auto *inputPH = M.createPlaceholder(ElemKind::FloatTy, {1}, "input", false);
  Node *addNode = F->createLog("testCppConstructors", inputPH);
  auto *saveNode = F->createSave("output", addNode);

  PlaceholderBindings bindings;
  auto *inputT = bindings.allocate(inputPH);
  inputT->getHandle().clear(0);
  Tensor expectedT(inputT->getType());
  expectedT.getHandle().clear(JIT_MAGIC_VALUE + 0);
  auto *outputT = bindings.allocate(saveNode->getPlaceholder());
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  EXPECT_TRUE(outputT->isEqual(expectedT));
}
