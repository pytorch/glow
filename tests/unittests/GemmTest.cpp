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
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Random.h"

#include "gtest/gtest.h"

#include <cassert>
#include <string>

using namespace glow;
using llvm::cast;

extern "C" {
// Forward declare functions from libjit.
extern void libjit_matmul_f(float *c, const float *a, const float *b,
                            const size_t *cDims, const size_t *aDims,
                            const size_t *bDims);
}

void infer(Tensor *out, Tensor *lhs, Tensor *rhs) {
  ExecutionEngine EE;
  PlaceholderBindings bindings;

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  auto *lhsVar =
      mod.createPlaceholder(lhs->getElementType(), lhs->dims(), "lhs", false);
  bindings.allocate(lhsVar);
  auto *rhsVar =
      mod.createPlaceholder(rhs->getElementType(), rhs->dims(), "rhs", false);
  bindings.allocate(rhsVar);
  auto OT = F->getParent()->uniqueType(out->getElementType(), out->dims());
  auto *matmul = F->createMatMul("matmul", OT, lhsVar, rhsVar);
  auto *save = F->createSave("ret", matmul);
  auto *res = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer, F);

  updateInputPlaceholders(bindings, {lhsVar, rhsVar}, {lhs, rhs});
  EE.run(bindings);

  out->assign(res);
}

TEST(Gemm, jitTest) {
  PseudoRNG PRNG;

  for (size_t m : {1, 4, 5, 8}) {
    for (size_t n : {1, 16, 17, 1024}) {
      for (size_t k : {1, 3}) {
        Tensor lhs(ElemKind::FloatTy, {m, k});
        Tensor rhs(ElemKind::FloatTy, {k, n});
        lhs.getHandle().randomize(-7.2, 8.3, PRNG);
        rhs.getHandle().randomize(-6.3, 10.1, PRNG);
        Tensor out1(ElemKind::FloatTy, {m, n});
        Tensor out2(ElemKind::FloatTy, {m, n});

        libjit_matmul_f((float *)out1.getUnsafePtr(),
                        (float *)lhs.getUnsafePtr(),
                        (float *)rhs.getUnsafePtr(), out1.dims().data(),
                        lhs.dims().data(), rhs.dims().data());

        infer(&out2, &lhs, &rhs);

        EXPECT_TRUE(out1.isEqual(out2, 0.001));
      }
    }
  }
}
