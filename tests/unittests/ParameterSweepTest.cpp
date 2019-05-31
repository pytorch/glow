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

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"

#include <functional>

using namespace glow;
using llvm::cast;

class ConvSweepTest : public BackendStatelessTest {};
INSTANTIATE_TEST_CASE_P_FOR_BACKEND_TEST(ConvSweepTest, ConvSweepTest);

class MatMulSweepTest : public BackendStatelessTest {};
INSTANTIATE_TEST_CASE_P_FOR_BACKEND_TEST(MatMulSweepTest, MatMulSweepTest);

/// Create a simple network that has a single fp convolution.
static FunctionTensorPair
createAndInitConvNet(glow::PlaceholderBindings &bindings,
                     glow::ExecutionEngine &EE, size_t size, size_t convDepth,
                     size_t kernel, size_t stride, size_t pad) {
  PseudoRNG PRNG;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createPlaceholder(ElemKind::FloatTy,
                                    {1, size, size, convDepth}, "var", false);
  bindings.allocate(var)->getHandle().initXavier(1, PRNG);

  auto *conv =
      F->createConv(bindings, "conv", var, convDepth, kernel, stride, pad, 1);
  bindings.get(cast<Placeholder>(conv->getFilter()))->getHandle().clear(0.1);
  bindings.get(cast<Placeholder>(conv->getBias()))->getHandle().clear(0.1);
  auto *result = F->createSave("ret", conv);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());
  convertPlaceholdersToConstants(F, bindings, {var, result->getPlaceholder()});

  return std::make_pair(F, resultTensor);
}

/// Helper to test sweeping across a variety of configurations of a convolution
/// by comparing the results to the Interpreter given some \p allowedError. \p b
/// is the backend to compare the Interpreter against. \p interpK and
/// \p backendK are the element kinds to use for the Interpreter and backend,
/// respectively.
static void testParamSweepConv(BackendKind b, ElemKind interpK,
                               ElemKind backendK, float allowedError) {
  for (size_t depth : {8, 64}) {
    for (size_t kernel : {1, 3}) {
      for (size_t size : {7, 5, 15}) {
        auto boundF = std::bind(createAndInitConvNet, std::placeholders::_1,
                                std::placeholders::_2, size, depth, kernel,
                                /* stride */ 1, /* pad */ 0);
        compareAgainstInterpreter(b, boundF, interpK, backendK, allowedError,
                                  parCloneCountOpt);
      }
    }
  }
}

/// Compare backend against the interpreter in Float.
TEST_P(ConvSweepTest, convTest_Float) {
  ENABLED_BACKENDS(CPU, OpenCL);
  testParamSweepConv(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy, 0.001f);
}

/// Compare backend against the interpreter in Int8.
TEST_P(ConvSweepTest, convTest_Int8) {
  ENABLED_BACKENDS(Interpreter, CPU, OpenCL);
  testParamSweepConv(GetParam(), ElemKind::FloatTy, ElemKind::Int8QTy, 0.045f);
}

/// Compare backend against the interpreter in FP16.
TEST_P(ConvSweepTest, convTest_Float16) {
  ENABLED_BACKENDS(Interpreter);
  testParamSweepConv(GetParam(), ElemKind::FloatTy, ElemKind::Float16Ty, 0.01f);
}

/// Create a simple network that has a single fp batch mat mul.
static FunctionTensorPair
createAndInitMatMulNet(glow::PlaceholderBindings &bindings,
                       glow::ExecutionEngine &EE, size_t N, size_t A, size_t Z,
                       size_t B) {
  PseudoRNG PRNG;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *LHS = mod.createPlaceholder(ElemKind::FloatTy, {N, A, Z}, "LHS", false);
  auto *RHS = mod.createPlaceholder(ElemKind::FloatTy, {N, Z, B}, "RHS", false);
  bindings.allocate(LHS)->getHandle().initXavier(10, PRNG);
  bindings.allocate(RHS)->getHandle().initXavier(10, PRNG);

  auto *R = F->createBatchMatMul("BMM", LHS, RHS);

  auto *save = F->createSave("save", R);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Helper to test sweeping across a variety of configurations of a MatMul by
/// comparing the results to the Interpreter given some \p allowedError. \p b is
/// the backend to compare the Interpreter against. \p interpK and \p backendK
/// are the element kinds to use for the Interpreter and backend, respectively.
static void testParamSweepMatMul(BackendKind b, ElemKind interpK,
                                 ElemKind backendK, float allowedError) {
  // Multiplying LHS {N, A, Z} by RHS {N, Z, B} to get result {N, A, B}.
  for (size_t N : {1, 4, 16, 24}) {
    for (size_t A = 10; A <= 16; A++) {
      for (size_t Z : {32, 64, 128, 256}) {
        size_t B = A;
        auto boundF = std::bind(createAndInitMatMulNet, std::placeholders::_1,
                                std::placeholders::_2, N, A, Z, B);
        compareAgainstInterpreter(b, boundF, interpK, backendK, allowedError,
                                  parCloneCountOpt);
      }
    }
  }
}

/// Compare backend against the interpreter in Float.
TEST_P(MatMulSweepTest, matMulTest_Float) {
  ENABLED_BACKENDS(CPU, OpenCL);
  testParamSweepMatMul(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy,
                       0.0001f);
}

/// Compare backend against the interpreter in Int8.
TEST_P(MatMulSweepTest, matMulTest_Int8) {
  ENABLED_BACKENDS(Interpreter, CPU, OpenCL);
  testParamSweepMatMul(GetParam(), ElemKind::FloatTy, ElemKind::Int8QTy, 0.06f);
}

/// Compare backend against the interpreter in FP16.
TEST_P(MatMulSweepTest, matMulTest_Float16) {
  ENABLED_BACKENDS(Interpreter);
  testParamSweepMatMul(GetParam(), ElemKind::FloatTy, ElemKind::Float16Ty,
                       0.005f);
}
