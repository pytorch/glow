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

class ConvCorrectnessTest : public BackendStatelessTest {};

INSTANTIATE_TEST_CASE_P_FOR_BACKEND_TEST(ConvCorrectnessTest,
                                         ConvCorrectnessTest);

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
TEST_P(ConvCorrectnessTest, convTest_Float) {
  ENABLED_BACKENDS(CPU, OpenCL);
  testParamSweepConv(GetParam(), ElemKind::FloatTy, ElemKind::FloatTy, 0.001f);
}

/// Compare backend against the interpreter in Int8.
TEST_P(ConvCorrectnessTest, convTest_Int8) {
  ENABLED_BACKENDS(Interpreter, CPU, OpenCL);
  testParamSweepConv(GetParam(), ElemKind::FloatTy, ElemKind::Int8QTy, 0.045f);
}

/// Compare backend against the interpreter in FP16.
TEST_P(ConvCorrectnessTest, convTest_Float16) {
  ENABLED_BACKENDS(Interpreter);
  testParamSweepConv(GetParam(), ElemKind::FloatTy, ElemKind::Float16Ty, 0.01f);
}
