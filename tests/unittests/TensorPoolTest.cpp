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
#include "glow/Support/TensorPool.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"

#include <future>
#include <vector>

using namespace glow;

/// Can get Tensor from the pool without allocation.
TEST(TensorPool, BasicTest) {
  TensorPool pool;
  Type ty(ElemKind::FloatTy, {1, 2, 3});
  pool.reserve(&ty, 1);

  Tensor T = std::move(pool.get(&ty).value());
  EXPECT_TRUE(T.getType().isEqual(ty));
  EXPECT_EQ(T.dims(), ty.dims());

  const auto &stats = pool.getStats();
  EXPECT_EQ(stats.totalTypes, 1);
  EXPECT_EQ(stats.currentBuffers, 0);
  EXPECT_EQ(stats.totalAllocs, 1);
  EXPECT_EQ(stats.inlineAllocs, 0);
  EXPECT_EQ(stats.totalGets, 1);
  EXPECT_EQ(stats.totalReclaims, 0);

  pool.reclaim(std::move(T));
}

/// Can get a tensor, return it and get it again without allocation.
TEST(TensorPool, ReclaimAndGet) {
  TensorPool pool;
  Type ty(ElemKind::FloatTy, {1, 2, 3});
  pool.reserve(&ty, 1);

  Tensor T = std::move(pool.get(&ty).value());
  auto *backingPtr = T.getUnsafePtr();

  pool.reclaim(std::move(T));

  Tensor T2 = std::move(pool.get(&ty).value());
  // They are the same buffer.
  EXPECT_EQ(T2.getUnsafePtr(), backingPtr);

  const auto &stats = pool.getStats();
  EXPECT_EQ(stats.totalTypes, 1);
  EXPECT_EQ(stats.currentBuffers, 0);
  EXPECT_EQ(stats.totalAllocs, 1);
  EXPECT_EQ(stats.inlineAllocs, 0);
  EXPECT_EQ(stats.totalGets, 2);
  EXPECT_EQ(stats.totalReclaims, 1);

  pool.reclaim(std::move(T2));
}

/// The pool auto resizes when it's empty.
TEST(TensorPool, Extends) {
  TensorPool pool;
  Type ty(ElemKind::FloatTy, {1, 2, 3});
  pool.reserve(&ty, 1);

  Tensor T = std::move(pool.get(&ty).value());
  Tensor T2 = std::move(pool.get(&ty).value());
  EXPECT_TRUE(T.getType().isEqual(T2.getType()));
  EXPECT_TRUE(T.getType().isEqual(ty));
  EXPECT_TRUE(T2.getType().isEqual(ty));

  // They are not the same buffer.
  EXPECT_NE(T.getUnsafePtr(), T2.getUnsafePtr());

  const auto &stats = pool.getStats();
  EXPECT_EQ(stats.totalTypes, 1);
  EXPECT_EQ(stats.currentBuffers, 0);
  EXPECT_EQ(stats.totalAllocs, 2);
  EXPECT_EQ(stats.inlineAllocs, 1);
  EXPECT_EQ(stats.totalGets, 2);
  EXPECT_EQ(stats.totalReclaims, 0);

  pool.reclaim(std::move(T));
  pool.reclaim(std::move(T2));
}

/// The pool doesn't resize when you tell it not to.
TEST(TensorPool, DoesntExtend) {
  TensorPool pool(true);
  Type ty(ElemKind::FloatTy, {1, 2, 3});
  pool.reserve(&ty, 1);

  Tensor T = std::move(pool.get(&ty).value());
  Type Tt = T.getType();

  auto T2opt = pool.get(&ty);
  EXPECT_FALSE(T2opt.has_value());

  pool.reclaim(std::move(T));

  T = std::move(pool.get(&ty).value());
  EXPECT_EQ(Tt, T.getType());

  const auto &stats = pool.getStats();
  EXPECT_EQ(stats.totalTypes, 1);
  EXPECT_EQ(stats.currentBuffers, 0);
  EXPECT_EQ(stats.totalAllocs, 1);
  EXPECT_EQ(stats.inlineAllocs, 0);
  EXPECT_EQ(stats.totalGets, 3);
  EXPECT_EQ(stats.totalReclaims, 1);

  pool.reclaim(std::move(T));
}

/// Still works if you don't reserve it.
TEST(TensorPool, Noreserve) {
  TensorPool pool;
  Type ty(ElemKind::FloatTy, {1, 2, 3});

  Tensor T = std::move(pool.get(&ty).value());
  Tensor T2 = std::move(pool.get(&ty).value());

  EXPECT_TRUE(T.getType().isEqual(T2.getType()));

  const auto &stats = pool.getStats();
  EXPECT_EQ(stats.totalTypes, 1);
  EXPECT_EQ(stats.currentBuffers, 0);
  EXPECT_EQ(stats.totalAllocs, 2);
  EXPECT_EQ(stats.inlineAllocs, 2);
  EXPECT_EQ(stats.totalGets, 2);
  EXPECT_EQ(stats.totalReclaims, 0);

  pool.reclaim(std::move(T));
  pool.reclaim(std::move(T2));
}

/// Can handle multiple types of Tensors.
TEST(TensorPool, MultipleTypes) {
  TensorPool pool;
  Type ty(ElemKind::FloatTy, {1, 2, 3});
  Type ty2(ElemKind::Int8QTy, {3, 2, 1}, 1.0, 4);

  // Six total buffers.
  pool.reserve(&ty, 1);
  pool.reserve(&ty2, 5);

  std::vector<Tensor> tensors;
  // Ten total allocs.
  for (int i = 0; i < 5; ++i) {
    Tensor T = std::move(pool.get(&ty).value());
    Tensor T2 = std::move(pool.get(&ty2).value());
    EXPECT_FALSE(T.getType().isEqual(T2.getType()));
    EXPECT_TRUE(T.getType().isEqual(ty));
    EXPECT_TRUE(T2.getType().isEqual(ty2));
    EXPECT_NE(T.dims(), T2.dims());
    EXPECT_NE(T.getUnsafePtr(), T2.getUnsafePtr());

    tensors.emplace_back(std::move(T));
    tensors.emplace_back(std::move(T2));
  }

  const auto &stats = pool.getStats();
  EXPECT_EQ(stats.totalTypes, 2);
  EXPECT_EQ(stats.currentBuffers, 0);
  EXPECT_EQ(stats.totalAllocs, 10);
  EXPECT_EQ(stats.inlineAllocs, 4); // Four allocs inline.
  EXPECT_EQ(stats.totalGets, 10);
  EXPECT_EQ(stats.totalReclaims, 0);

  for (auto &t : tensors) {
    pool.reclaim(std::move(t));
  }

  const auto &stats2 = pool.getStats();
  EXPECT_EQ(stats2.totalTypes, 2);
  EXPECT_EQ(stats2.currentBuffers, 10);
  EXPECT_EQ(stats.totalReclaims, 10);
}

/// Reclaims still work with multiple types of Tensors.
TEST(TensorPool, MultipleTypesReclaim) {
  TensorPool pool;
  Type ty(ElemKind::FloatTy, {1, 2, 3});
  Type ty2(ElemKind::Int8QTy, {3, 2, 1}, 1.0, 4);
  pool.reserve(&ty, 1);
  pool.reserve(&ty2, 1);

  Tensor T = std::move(pool.get(&ty).value());
  Tensor T2 = std::move(pool.get(&ty2).value());

  pool.reclaim(std::move(T));
  pool.reclaim(std::move(T2));

  T = std::move(pool.get(&ty).value());
  T2 = std::move(pool.get(&ty2).value());

  pool.reclaim(std::move(T));
  pool.reclaim(std::move(T2));

  const auto &stats = pool.getStats();
  EXPECT_EQ(stats.totalTypes, 2);
  EXPECT_EQ(stats.currentBuffers, 2);
  EXPECT_EQ(stats.totalAllocs, 2);
  EXPECT_EQ(stats.inlineAllocs, 0);
  EXPECT_EQ(stats.totalGets, 4);
  EXPECT_EQ(stats.totalReclaims, 4);
}

/// Inserting a managed Tensor into the PlaceholderBindings does reclaim when
/// the bindings are cleared or destroyed.
TEST(TensorPool, PlaceholderBindingsReclaim) {
  TensorPool pool;
  Type ty(ElemKind::FloatTy, {1, 2, 3});

  PlaceholderBindings bindings;
  Module mod;

  auto *PH = mod.createPlaceholder(&ty, "test", false);
  bindings.insert(PH, std::move(pool.get(&ty).value()));

  /// Insert a non managed tensor.
  auto *PH2 = mod.createPlaceholder(&ty, "test2", false);
  Tensor T2(ty);
  bindings.insert(PH2, std::move(T2));

  bindings.clear();

  /// Bindings had two Tensors but only the first was reclaimed.
  const auto &stats = pool.getStats();
  EXPECT_EQ(stats.totalTypes, 1);
  EXPECT_EQ(stats.currentBuffers, 1);
  EXPECT_EQ(stats.totalAllocs, 1);
  EXPECT_EQ(stats.inlineAllocs, 1);
  EXPECT_EQ(stats.totalGets, 1);
  EXPECT_EQ(stats.totalReclaims, 1);

  bindings.insert(PH, std::move(pool.get(&ty).value()));

  bindings.erase(PH);
  const auto &stats2 = pool.getStats();
  EXPECT_EQ(stats.currentBuffers, 1);
  EXPECT_EQ(stats.totalGets, 2);
  EXPECT_EQ(stats2.totalReclaims, 2);
}

/// Clearing the Tensor pool removes contents but the pool still works.
TEST(TensorPool, Clear) {
  TensorPool pool;
  Type ty(ElemKind::FloatTy, {1, 2, 3});

  Tensor T = std::move(pool.get(&ty).value());
  pool.reclaim(std::move(T));

  const auto &stats = pool.getStats();
  EXPECT_EQ(stats.totalTypes, 1);
  EXPECT_EQ(stats.currentBuffers, 1);
  EXPECT_EQ(stats.totalAllocs, 1);
  EXPECT_EQ(stats.inlineAllocs, 1);
  EXPECT_EQ(stats.totalGets, 1);
  EXPECT_EQ(stats.totalReclaims, 1);
  EXPECT_EQ(stats.totalFrees, 0);

  pool.clear();

  T = std::move(pool.get(&ty).value());
  pool.reclaim(std::move(T));

  const auto &stats2 = pool.getStats();
  EXPECT_EQ(stats2.totalTypes, 1);
  EXPECT_EQ(stats2.currentBuffers, 1);
  EXPECT_EQ(stats2.totalAllocs, 2);
  EXPECT_EQ(stats2.inlineAllocs, 2);
  EXPECT_EQ(stats2.totalGets, 2);
  EXPECT_EQ(stats2.totalReclaims, 2);
  EXPECT_EQ(stats2.totalFrees, 1);
}
