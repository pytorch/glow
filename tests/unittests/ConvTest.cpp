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

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"

using namespace glow;
using llvm::cast;

class ConvCorrectnessTest : public ::testing::TestWithParam<BackendKind> {
protected:
  BackendKind backendKind_{GetParam()};
};

/// Create a simple network that has a single fp convolution.
void singleConvNet(Tensor *input, Tensor *out, BackendKind kind,
                   size_t convDepth, size_t kernel, size_t stride, size_t pad) {
  PlaceholderBindings bindings;
  ExecutionEngine EE(kind);
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *var = mod.createPlaceholder(input->getElementType(), input->dims(),
                                    "var", false);
  bindings.allocate(var)->assign(input);

  auto *conv =
      F->createConv(bindings, "conv", var, convDepth, kernel, stride, pad, 1);
  bindings.get(cast<Placeholder>(conv->getFilter()))->getHandle().clear(0.1);
  bindings.get(cast<Placeholder>(conv->getBias()))->getHandle().clear(0.1);
  auto *result = F->createSave("ret", conv);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());
  convertPlaceholdersToConstants(F, bindings, {var, result->getPlaceholder()});

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {var}, {input});
  EE.run(bindings);
  out->assign(resultTensor);
}

/// Test a few configurations of the fp convolution by comparing the results to
/// the interpreter.
TEST_P(ConvCorrectnessTest, convTest) {
  PseudoRNG PRNG;
  for (size_t depth : {8, 64}) {
    for (size_t kernel : {1, 3}) {
      for (size_t size : {7, 5, 15}) {
        Tensor input(ElemKind::FloatTy, {1, size, size, depth});
        input.getHandle().initXavier(1, PRNG);
        Tensor out1;
        Tensor out2;
        singleConvNet(&input, &out1, backendKind_, depth, kernel, 1, 0);
        singleConvNet(&input, &out2, BackendKind::Interpreter, depth, kernel, 1,
                      0);
        EXPECT_TRUE(out1.isEqual(out2, 0.001));
      }
    }
  }
}

#ifdef GLOW_WITH_CPU
INSTANTIATE_TEST_CASE_P(CPU, ConvCorrectnessTest,
                        ::testing::Values(BackendKind::CPU));
#endif // GLOW_WITH_CPU

#ifdef GLOW_WITH_OPENCL
INSTANTIATE_TEST_CASE_P(OpenCL, ConvCorrectnessTest,
                        ::testing::Values(BackendKind::OpenCL));
#endif // GLOW_WITH_OPENCL
