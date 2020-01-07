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

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Base/Base.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <numeric>

using namespace glow;

class OperatorStatelessTest : public BackendStatelessTest {};

class OperatorTest : public BackendTest {
protected:
  PlaceholderBindings bindings_;
};

/// Helper to create a Placeholder; if \p T is quantized, then it will include a
/// dummy scale and offset, otherwise it will not.
static Placeholder *createPlaceholderConditionallyQuantized(
    Module &mod, ElemKind T, llvm::ArrayRef<dim_t> dims, llvm::StringRef name,
    bool isTrainable, llvm::StringRef layout = ANY_LAYOUT) {
  return isQuantizedElemKind(T)
             ? mod.createPlaceholder(T, dims, 1.0, 0, name, isTrainable, layout)
             : mod.createPlaceholder(T, dims, name, isTrainable, layout);
}

/// Helper to get a unique Type; if \p T is quantized, then it will include a
/// dummy scale and offset, otherwise it will not.
static TypeRef uniqueTypeConditionallyQuantized(Module &mod, ElemKind T,
                                                llvm::ArrayRef<dim_t> dims) {
  return isQuantizedElemKind(T) ? mod.uniqueType(T, dims, 1.0, 0)
                                : mod.uniqueType(T, dims);
}

/// Helper to create a Tensor; if \p T is quantized, then it will include a
/// dummy scale and offset, otherwise it will not.
static Tensor createTensorConditionallyQuantized(ElemKind T,
                                                 llvm::ArrayRef<dim_t> dims) {
  return isQuantizedElemKind(T) ? Tensor(T, dims, 1.0, 0) : Tensor(T, dims);
}

template <typename DataType>
glow::Handle<bool>
lessHelper(glow::PlaceholderBindings &bindings, glow::Module &mod,
           glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
           llvm::ArrayRef<DataType> xValues, llvm::ArrayRef<DataType> yValues,
           llvm::ArrayRef<dim_t> xDims, llvm::ArrayRef<dim_t> yDims) {
  auto *X = createPlaceholderConditionallyQuantized(mod, DTy, xDims, "X",
                                                    /* isTrainable */ false);

  auto *Y = createPlaceholderConditionallyQuantized(mod, DTy, yDims, "Y",
                                                    /* isTrainable */ false);

  bindings.allocate(llvm::dyn_cast<Placeholder>(X))->getHandle<DataType>() =
      xValues;

  bindings.allocate(llvm::dyn_cast<Placeholder>(Y))->getHandle<DataType>() =
      yValues;

  auto *cmpr =
      F->createNodeWithBroadcast<CmpLTNode>("cmpLT", /* axis */ -1, X, Y);

  auto *save = F->createSave("save", cmpr);
  auto *saveAlloc = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  return saveAlloc->getHandle<bool>();
}

TEST_P(OperatorTest, less_int8) {
  CHECK_IF_ENABLED();

  int8_t xValues[] = {3, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

                      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

                      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

                      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  int8_t yValues[] = {3, 4, 5, 7, 2, 5, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

                      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

                      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

                      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  dim_t xDims[] = {2, 2, 4, 4};
  dim_t yDims[] = {2, 2, 4, 4};

  Handle<bool> saveH =
      lessHelper<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy, xValues,
                         yValues, xDims, yDims);

  bool refResults[] = {
      false, true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,
  };

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, less_floatCases) {
  CHECK_IF_ENABLED();

  float xValues[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  float yValues[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  dim_t xDims[] = {5};
  dim_t yDims[] = {5};

  Handle<bool> saveH =
      lessHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                        yValues, xDims, yDims);

  bool refResults[] = {true, true, false, false, false};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i}));
  }
}

TEST_P(OperatorTest, less_float16Cases) {
  CHECK_IF_ENABLED();

  float16 xValues[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  float16 yValues[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  dim_t xDims[] = {5};
  dim_t yDims[] = {5};

  Handle<bool> saveH =
      lessHelper<float16>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                          xValues, yValues, xDims, yDims);

  bool refResults[] = {true, true, false, false, false};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i}));
  }
}

TEST_P(OperatorTest, less_int64Cases) {
  CHECK_IF_ENABLED();

  int64_t xValues[] = {1, 2, 3, 4, 5};

  int64_t yValues[] = {5, 4, 3, 2, 1};

  dim_t xDims[] = {5};
  dim_t yDims[] = {5};

  Handle<bool> saveH =
      lessHelper<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy, xValues,
                          yValues, xDims, yDims);

  bool refResults[] = {true, true, false, false, false};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i}));
  }
}

TEST_P(OperatorTest, less_float) {
  CHECK_IF_ENABLED();

  float xValues[] = {1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f};

  float yValues[] = {3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

                     3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

                     3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

                     3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  dim_t xDims[] = {2, 2, 4, 4};
  dim_t yDims[] = {2, 2, 4, 4};

  Handle<bool> saveH =
      lessHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                        yValues, xDims, yDims);

  bool refResults[] = {
      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,
  };

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, less_broadcast_float) {
  CHECK_IF_ENABLED();

  float xValues[] = {1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f};

  float yValues[] = {3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

                     3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  dim_t xDims[] = {2, 2, 4, 4};
  dim_t yDims[] = {1, 2, 4, 4};

  Handle<bool> saveH =
      lessHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                        yValues, xDims, yDims);

  bool refResults[] = {true,  true,  true,  true, false, false, false, true,
                       false, false, false, true, true,  true,  false, true,

                       true,  true,  true,  true, false, false, false, true,
                       false, false, false, true, true,  true,  false, true,

                       true,  true,  true,  true, false, false, false, true,
                       false, false, false, true, true,  true,  false, true,

                       true,  true,  true,  true, false, false, false, true,
                       false, false, false, true, true,  true,  false, true};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, less_int32Cases) {
  CHECK_IF_ENABLED();

  int32_t xValues[] = {1, 2, 3, 4, 5};
  int32_t yValues[] = {5, 4, 3, 2, 1};

  dim_t xDims[] = {1, 1, 1, 5};
  dim_t yDims[] = {1, 1, 1, 5};

  Handle<bool> saveH =
      lessHelper<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy, xValues,
                          yValues, xDims, yDims);

  bool refResults[] = {true, true, false, false, false};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

template <typename DataType>
glow::Handle<DataType>
whereHelper(glow::PlaceholderBindings &bindings, glow::Module &mod,
            glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
            llvm::ArrayRef<DataType> xValues, llvm::ArrayRef<DataType> yValues,
            llvm::ArrayRef<bool> cValues, llvm::ArrayRef<dim_t> xDims,
            llvm::ArrayRef<dim_t> yDims, llvm::ArrayRef<dim_t> cDims) {
  auto *cond = createPlaceholderConditionallyQuantized(mod, ElemKind::BoolTy,
                                                       cDims, "cond", false);
  auto *X = createPlaceholderConditionallyQuantized(mod, DTy, xDims, "X",
                                                    DTy != ElemKind::FloatTy);

  auto *Y = createPlaceholderConditionallyQuantized(mod, DTy, yDims, "Y",
                                                    DTy != ElemKind::FloatTy);

  bindings.allocate(llvm::dyn_cast<Placeholder>(cond))->getHandle<bool>() =
      cValues;

  bindings.allocate(llvm::dyn_cast<Placeholder>(X))->getHandle<DataType>() =
      xValues;

  bindings.allocate(llvm::dyn_cast<Placeholder>(Y))->getHandle<DataType>() =
      yValues;

  auto *whr = F->createNodeWithBroadcast<SelectNode>("Select", /* axis */ -1,
                                                     cond, X, Y);

  auto *save = F->createSave("save", whr);
  auto *saveAlloc = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  return saveAlloc->getHandle<DataType>();
}

TEST_P(OperatorTest, where_2d_broadcast_x_y_i8) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<int8_t, 16> xValues = {3, 5, 7};

  llvm::SmallVector<int8_t, 16> yValues = {2, 4, 6};

  llvm::SmallVector<bool, 4> cValues = {1, 0, 1};

  llvm::SmallVector<dim_t, 4> condDims = {3, 1, 1};

  llvm::SmallVector<dim_t, 4> xDims = {1, 3, 1};
  llvm::SmallVector<dim_t, 4> yDims = {3, 1, 1};

  Handle<int8_t> saveH =
      whereHelper<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy, xValues,
                          yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<int8_t, 16> refResults = {3, 5, 7, 4, 4, 4, 3, 5, 7};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        EXPECT_EQ(refResults[counter++], saveH.at({i, j, k}));
      }
    }
  }
}

TEST_P(OperatorTest, where_2d_wise_i8) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<int8_t, 16> xValues = {
      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  llvm::SmallVector<int8_t, 16> yValues = {
      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  llvm::SmallVector<bool, 4> cValues = {1, 0, 1, 0};

  llvm::SmallVector<dim_t, 4> condDims = {2, 2, 1, 1};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<int8_t> saveH =
      whereHelper<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy, xValues,
                          yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<int8_t, 16> refResults = {
      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, where_2d_wise_float) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<float, 16> xValues = {
      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f};

  llvm::SmallVector<float, 16> yValues = {
      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  llvm::SmallVector<bool, 4> cValues = {1, 0, 1, 0};

  llvm::SmallVector<dim_t, 4> condDims = {2, 2, 1, 1};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<float> saveH =
      whereHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                         yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<float, 16> refResults = {
      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, where_row_wise_float) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<bool, 4> cValues = {1, 1, 1, 0, 0, 1, 0, 0};

  llvm::SmallVector<dim_t, 4> condDims = {2, 4, 1};

  llvm::SmallVector<dim_t, 4> xDims = {2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 4, 4};

  llvm::SmallVector<float, 16> xValues = {
      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f};

  llvm::SmallVector<float, 16> yValues = {
      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  Handle<float> saveH =
      whereHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                         yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<float, 16> refResults = {
      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,
  };

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k}));
      }
    }
  }
}

TEST_P(OperatorTest, where_element_wise_float) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<dim_t, 4> condDims = {1, 4, 4};

  llvm::SmallVector<dim_t, 4> xDims = {1, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {1, 4, 4};

  llvm::SmallVector<bool, 4> cValues = {1, 1, 1, 0, 0, 1, 0, 0,
                                        0, 1, 0, 1, 1, 0, 1, 0};

  llvm::SmallVector<float, 16> xValues = {1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f,
                                          6.0f, 3.0f, 7.0f, 8.0f, 9.0f, 2.0f,
                                          3.0f, 5.0f, 7.0f, 1.0f};

  llvm::SmallVector<float, 16> yValues = {3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f,
                                          0.0f, 6.0f, 4.0f, 2.0f, 1.0f, 8.0f,
                                          5.0f, 9.0f, 2.0f, 6.0f};

  Handle<float> saveH =
      whereHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                         yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<float, 16> refResults = {1.0f, 2.0f, 3.0f, 7.0f, 2.0f, 5.0f,
                                             0.0f, 6.0f, 4.0f, 8.0f, 1.0f, 2.0f,
                                             3.0f, 9.0f, 7.0f, 6.0f};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k}));
      }
    }
  }
}

// Helper to test SpaceToDepth using \p DTy.
template <typename DataType>
static void testSpaceToDepthBlock3(glow::PlaceholderBindings &bindings,
                                   glow::Module &mod, glow::Function *F,
                                   glow::ExecutionEngine &EE, ElemKind DTy) {
  unsigned blockSize = 3;
  auto *in = createPlaceholderConditionallyQuantized(mod, DTy, {1, 2, 6, 6},
                                                     "in", false, "NHWC");
  auto *tri = F->createTranspose("sptdTransposeIn", in, {0, 2, 3, 1}, "NHWC");
  auto *stdn = F->createSpaceToDepth("spacetodepth", tri, blockSize);
  auto *tro =
      F->createTranspose("sptdTransposeOut", stdn, {0, 3, 1, 2}, "NCHW");
  auto *save = F->createSave("save", tro);
  auto *result = bindings.allocate(save->getPlaceholder());

  /*
    Example for first batch.
  FROM:
  C0:             C1:
  [0  1  2  3  16 17]    [ 0  -1  -2  -3  -16 -17]
  [4  5  6  7  18 19]    [-4  -5  -6  -7  -18 -19]
  [8  9  10 11 20 21]    [-8  -9  -10 -11 -20 -21]
  [12 13 14 15 22 23]    [-12 -13 -14 -15 -22 -23]
  [24 25 26 27 28 29]    [-24 -25 -26 -27 -28 -29]
  [30 31 32 33 34 35]    [-30 -31 -32 -33 -34 -35]

  TO:
  C = 0
  [0,3]
  [12,15]

  C = 1
  [0,-3]
  [-12,-15]

  C = 2
  [1,16]
  [13,22]

  C = 3
  [-1,-16]
  [-13,-22]

  C = 4
  [2,17]
  [14,23]

  C = 5
  [-2,-17]
  [-14,-23]

  C = 6
  [4,7]
  [24,27]

  C = 7
  [-4,-7]
  [-24,-27]

  C = 8
  [5,18]
  [25,28]

  C = 9
  [-5,-18]
  [-25,-28]

  C = 10
  [6,19]
  [26,29]

  C = 11
  [-6,-19]
  [-26,-29]

  C = 12
  [8,11]
  [30,33]

  C = 13
  [-8,-11]
  [-30,-33]

  C = 14
  [9,20]
  [31,34]

  C = 15
  [-9,-20]
  [-31,-34]

  C = 16
  [10,21]
  [32,35]

  C = 17
  [-10,-21]
  [-32,-35]
  */

  bindings.allocate(in)->getHandle<DataType>() = {
      0,   1,   2,   3,   16,  17,  4,   5,   6,   7,   18,  19,  8,   9,   10,
      11,  20,  21,  12,  13,  14,  15,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  0,   -1,  -2,  -3,  -16, -17, -4,  -5,  -6,
      -7,  -18, -19, -8,  -9,  -10, -11, -20, -21, -12, -13, -14, -15, -22, -23,
      -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35};

  std::vector<DataType> refResult = {
      0,   3,   12,  15,  0,  -3, -12, -15, 1,   16,  13,  22, -1, -16, -13,
      -22, 2,   17,  14,  23, -2, -17, -14, -23, 4,   7,   24, 27, -4,  -7,
      -24, -27, 5,   18,  25, 28, -5,  -18, -25, -28, 6,   19, 26, 29,  -6,
      -19, -26, -29, 8,   11, 30, 33,  -8,  -11, -30, -33, 9,  20, 31,  34,
      -9,  -20, -31, -34, 10, 21, 32,  35,  -10, -21, -32, -35};

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Handle<DataType> resultH = result->getHandle<DataType>();

  auto iDims = in->dims();
  auto oDims = resultH.dims();
  EXPECT_EQ(iDims[0], oDims[0]);
  EXPECT_EQ(iDims[1] * blockSize * blockSize, oDims[1]);
  EXPECT_EQ(iDims[2], oDims[2] * blockSize);
  EXPECT_EQ(iDims[3], oDims[3] * blockSize);

  // NCHW format
  dim_t resIndex = 0;
  for (dim_t on = 0; on < oDims[0]; ++on) {
    for (dim_t oc = 0; oc < oDims[1]; ++oc) {
      for (dim_t oh = 0; oh < oDims[2]; ++oh) {
        for (dim_t ow = 0; ow < oDims[3]; ++ow) {
          DataType resultVal = resultH.at({on, oc, oh, ow});
          DataType refVal = refResult[resIndex++];
          EXPECT_EQ(resultVal, refVal);
        }
      }
    }
  }
}

/// Verify that the SpaceToDepth operator works correctly for int8. Block
/// Size 3.
TEST_P(OperatorTest, spaceToDepth_block3_int8) {
  CHECK_IF_ENABLED();
  testSpaceToDepthBlock3<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Verify that the SpaceToDepth operator works correctly for Float. Block
/// Size 3.
TEST_P(OperatorTest, spaceToDepth_block3_Float) {
  CHECK_IF_ENABLED();
  testSpaceToDepthBlock3<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

// Helper to test SpaceToDepth using \p DTy.
template <typename DataType>
static void testSpaceToDepth(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  unsigned blockSize = 2;
  auto *in = createPlaceholderConditionallyQuantized(mod, DTy, {2, 2, 4, 4},
                                                     "in", false, "NHWC");
  auto *tri = F->createTranspose("sptdTransposeIn", in, {0, 2, 3, 1}, "NHWC");
  auto *stdn = F->createSpaceToDepth("spacetodepth", tri, blockSize);
  auto *tro =
      F->createTranspose("sptdTransposeOut", stdn, {0, 3, 1, 2}, "NCHW");
  auto *save = F->createSave("save", tro);
  auto *result = bindings.allocate(save->getPlaceholder());

  /*
    Example for first batch.
  FROM:
  C0:             C1:
  [0  1  2  3]    [ 0  -1  -2  -3]
  [4  5  6  7]    [-4  -5  -6  -7]
  [8  9  10 11]   [-8  -9  -10 -11]
  [12 13 14 15]   [-12 -13 -14 -15]

  TO:
  C0:
  [0,  2]
  [8,  10]

  C1:
  [ 0,  -2]
  [-8, -10]

  C2:
  [1, 3]
  [9, 11]

  C3:
  [-1, -3]
  [-9, -11]

  C4:
  [4,  6]
  [12, 14]

  C5:
  [-4,  -6]
  [-12, -14]

  C6:
  [5, 7]
  [13, 15]

  C7:
  [-5,  -7]
  [-13, -15]
  */

  bindings.allocate(in)->getHandle<DataType>() = {
      0, 1,   2,   3,   4,  5,  6,   7,   8,  9,  10,  11,  12,  13,  14,  15,
      0, -1,  -2,  -3,  -4, -5, -6,  -7,  -8, -9, -10, -11, -12, -13, -14, -15,
      0, 7,   9,   23,  24, 25, 26,  27,  8,  9,  10,  33,  12,  13,  14,  15,
      0, -21, -22, -23, -4, -5, -26, -27, -8, -9, -10, -11, -12, -13, -14, -15};

  std::vector<DataType> refResult = {
      0,  2,  8,  10, 0,  -2,  -8,  -10, 1,  3,  9,  11, -1,  -3,  -9,  -11,
      4,  6,  12, 14, -4, -6,  -12, -14, 5,  7,  13, 15, -5,  -7,  -13, -15,
      0,  9,  8,  10, 0,  -22, -8,  -10, 7,  23, 9,  33, -21, -23, -9,  -11,
      24, 26, 12, 14, -4, -26, -12, -14, 25, 27, 13, 15, -5,  -27, -13, -15};

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Handle<DataType> resultH = result->getHandle<DataType>();

  auto iDims = in->dims();
  auto oDims = resultH.dims();
  EXPECT_EQ(iDims[0], oDims[0]);
  EXPECT_EQ(iDims[1] * blockSize * blockSize, oDims[1]);
  EXPECT_EQ(iDims[2], oDims[2] * blockSize);
  EXPECT_EQ(iDims[3], oDims[3] * blockSize);

  // NCHW format
  dim_t resIndex = 0;
  for (dim_t on = 0; on < oDims[0]; ++on) {
    for (dim_t oc = 0; oc < oDims[1]; ++oc) {
      for (dim_t oh = 0; oh < oDims[2]; ++oh) {
        for (dim_t ow = 0; ow < oDims[3]; ++ow) {
          DataType resultVal = resultH.at({on, oc, oh, ow});
          DataType refVal = refResult[resIndex++];
          EXPECT_EQ(resultVal, refVal);
        }
      }
    }
  }
}

/// Verify that the SpaceToDepth operator works correctly for int8. Block
/// Size 2.
TEST_P(OperatorTest, spaceToDepth_block2_int8) {
  CHECK_IF_ENABLED();
  testSpaceToDepth<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Verify that the SpaceToDepth operator works correctly for Float. Block
/// Size 2.
TEST_P(OperatorTest, spaceToDepth_block2_Float) {
  CHECK_IF_ENABLED();
  testSpaceToDepth<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Helper to test ResizeNearest using \p DTy.
template <typename DataType>
static void testResizeNearest(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {1, 2, 2, 1},
                                                        "input", false, "NHWC");
  bindings.allocate(input)->getHandle<DataType>() = {2, 4, 8, 16};

  auto heightScaleUp = 2.0f;
  auto widthScaleUp = 1.5f;

  auto *resizeNearestUp = F->createResizeNearest("resizeNearestUp", input,
                                                 heightScaleUp, widthScaleUp);
  auto *saveUp = F->createSave("saveUp", resizeNearestUp);
  auto *resultUp = bindings.allocate(saveUp->getPlaceholder());

  auto heightScaleDown = 0.9f;
  auto widthScaleDown = 0.6;

  auto *resizeNearestDown = F->createResizeNearest(
      "resizeNearestDown", input, heightScaleDown, widthScaleDown);
  auto *saveDown = F->createSave("saveDown", resizeNearestDown);
  auto *resultDown = bindings.allocate(saveDown->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(
      F, bindings,
      {input, saveUp->getPlaceholder(), saveDown->getPlaceholder()});

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultUpH = resultUp->getHandle<DataType>();
  std::vector<dim_t> expectedDimsUp = {1, 4, 3, 1};
  ASSERT_TRUE(resultUpH.dims().vec() == expectedDimsUp);

  EXPECT_EQ(resultUpH.at({0, 0, 0, 0}), static_cast<DataType>(2));
  EXPECT_EQ(resultUpH.at({0, 0, 1, 0}), static_cast<DataType>(2));
  EXPECT_EQ(resultUpH.at({0, 0, 2, 0}), static_cast<DataType>(4));

  EXPECT_EQ(resultUpH.at({0, 1, 0, 0}), static_cast<DataType>(2));
  EXPECT_EQ(resultUpH.at({0, 1, 1, 0}), static_cast<DataType>(2));
  EXPECT_EQ(resultUpH.at({0, 1, 2, 0}), static_cast<DataType>(4));

  EXPECT_EQ(resultUpH.at({0, 2, 0, 0}), static_cast<DataType>(8));
  EXPECT_EQ(resultUpH.at({0, 2, 1, 0}), static_cast<DataType>(8));
  EXPECT_EQ(resultUpH.at({0, 2, 2, 0}), static_cast<DataType>(16));

  EXPECT_EQ(resultUpH.at({0, 3, 0, 0}), static_cast<DataType>(8));
  EXPECT_EQ(resultUpH.at({0, 3, 1, 0}), static_cast<DataType>(8));
  EXPECT_EQ(resultUpH.at({0, 3, 2, 0}), static_cast<DataType>(16));

  auto resultDownH = resultDown->getHandle<DataType>();
  std::vector<dim_t> expectedDimsDown = {1, 1, 1, 1};
  ASSERT_TRUE(resultDownH.dims().vec() == expectedDimsDown);
  EXPECT_EQ(resultDownH.at({0, 0, 0, 0}), static_cast<DataType>(2));
}

/// Verify that the ResizeNearest operator works correctly for Float.
TEST_P(OperatorTest, ResizeNearest_Float) {
  CHECK_IF_ENABLED();
  testResizeNearest<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Verify that the ResizeNearest operator works correctly for Float16.
TEST_P(OperatorTest, ResizeNearest_Float16) {
  CHECK_IF_ENABLED();
  testResizeNearest<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Verify that the ResizeNearest operator works correctly for Int8Q.
TEST_P(OperatorTest, ResizeNearest_Int8) {
  CHECK_IF_ENABLED();
  testResizeNearest<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Verify that the ResizeNearest operator works correctly for Int16Q.
TEST_P(OperatorTest, ResizeNearest_Int16) {
  CHECK_IF_ENABLED();
  testResizeNearest<int16_t>(bindings_, mod_, F_, EE_, ElemKind::Int16QTy);
}

/// Verify that the ResizeNearest operator works correctly for Int32Q.
TEST_P(OperatorTest, ResizeNearest_Int32) {
  CHECK_IF_ENABLED();
  testResizeNearest<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32QTy);
}

TEST_P(OperatorTest, pow) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 3}, "X", false);
  auto *Y = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "Y", false);
  auto *Exp = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "Exp", false);

  bindings_.allocate(X)->getHandle() = {5, 0.1f, -3};
  bindings_.allocate(Y)->getHandle() = {2, 100};
  bindings_.allocate(Exp)->getHandle() = {2, -1};

  auto *Pow1 = F_->createPow("Pow1", X, 2.0);
  auto *Pow2 = F_->createPow("Pow2", Y, 0.5);
  auto *Pow3 = F_->createPow("Pow3", Y, Exp);

  auto *save1 = F_->createSave("save", Pow1);
  auto *savePlaceholder1 = save1->getPlaceholder();

  auto *save2 = F_->createSave("save", Pow2);
  auto *savePlaceholder2 = save2->getPlaceholder();

  auto *save3 = F_->createSave("save", Pow3);
  auto *savePlaceholder3 = save3->getPlaceholder();

  bindings_.allocate(savePlaceholder1);
  bindings_.allocate(savePlaceholder2);
  bindings_.allocate(savePlaceholder3);

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto HX = bindings_.get(savePlaceholder1)->getHandle();
  EXPECT_NEAR(HX.at({0, 0, 0}), 25, 1E-5);
  EXPECT_NEAR(HX.at({0, 0, 1}), 0.01, 1E-5);
  EXPECT_NEAR(HX.at({0, 0, 2}), 9, 1E-5);

  auto HY = bindings_.get(savePlaceholder2)->getHandle();
  EXPECT_NEAR(HY.at({0}), sqrt(2.0), 1E-5);
  EXPECT_NEAR(HY.at({1}), 10, 1E-5);

  auto HZ = bindings_.get(savePlaceholder3)->getHandle();
  EXPECT_NEAR(HZ.at({0}), 4, 1E-5);
  EXPECT_NEAR(HZ.at({1}), 0.01, 1E-5);
}

/// Helper to test ReplaceNaN using \p DTy.
template <typename DataType>
static void testReplaceNaN(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind DTy) {
  auto value = 1.0f;
  auto *X = mod.createPlaceholder(DTy, {6}, "X", false);
  auto XH = bindings.allocate(X)->getHandle<DataType>();
  XH = {1, NAN, 2, NAN, 3, NAN};

  auto *RNN = F->createReplaceNaN("replaceNaN", X, value);

  auto *save = F->createSave("save", RNN);
  auto *saveTensor = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  auto saveH = saveTensor->getHandle<DataType>();

  for (size_t i = 0; i < 6; i++) {
    if (std::isnan((float)XH.raw(i))) {
      EXPECT_EQ(saveH.raw(i), (DataType)value);
    } else {
      EXPECT_EQ(XH.raw(i), saveH.raw(i));
    }
  }
}

/// Test that ReplaceNaN is correctly supported in FloatTy.
TEST_P(OperatorTest, replaceNaN_Float) {
  CHECK_IF_ENABLED();
  testReplaceNaN<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that ReplaceNaN is correctly supported in Float16Ty.
TEST_P(OperatorTest, replaceNaN_Float16) {
  CHECK_IF_ENABLED();
  testReplaceNaN<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

TEST_P(OperatorTest, log) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto XH = bindings_.allocate(X)->getHandle();
  XH = {210030, 600, 4, 0.7f, .005f, 0.000829f};

  auto *LN = F_->createLog("log", X);

  auto *save = F_->createSave("save", LN);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = saveTensor->getHandle();

  for (dim_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), log(XH.at({i})), 1E-5);
  }
}

/// Helper to test Logit using \p DTy.
template <typename DataType>
static void testLogit(glow::PlaceholderBindings &bindings, glow::Module &mod,
                      glow::Function *F, glow::ExecutionEngine &EE,
                      ElemKind DTy, float allowedError) {
  constexpr auto eps = 1E-6f;      // the default in Caffe2
  constexpr std::size_t size = 10; // sample size for randomized tests

  auto *input = mod.createPlaceholder(DTy, {size}, "input", false);
  // generate the input data in (0.0f, 1.0f) (probabilites including degenerate
  // cases) and test that afterward the input data is clamped in
  // (eps, 1 - eps) as in Caffe2.
  bindings.allocate(input)->getHandle<DataType>().randomize(0.0f, 1.0f,
                                                            mod.getPRNG());

  auto *logitDiff = F->createLogit("logitDiff", input, eps);
  auto *saveDiff = F->createSave("saveDiff", logitDiff);
  bindings.allocate(saveDiff->getPlaceholder());

  // property: zero-sum for the log-odds for complementary events probabilities
  // i.e., logit(p) + logit(1 - p) == 0
  Node *const1 = F->createSplat("const1", input->getType(), 1.0);
  Node *complInput = F->createSub("sub", const1, input);
  Node *logitCompl = F->createLogit("logitCompl", complInput, eps);
  auto *saveCompl = F->createSave("saveCompl", logitCompl);
  bindings.allocate(saveCompl->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // results: differential test against the oracle
  auto resultDiffH =
      bindings.get(saveDiff->getPlaceholder())->getHandle<DataType>();
  auto inputH = bindings.get(input)->getHandle<DataType>();

  // results: zero-sum property
  auto resultComplH =
      bindings.get(saveCompl->getPlaceholder())->getHandle<DataType>();

  // differential test:
  // ensure we match an oracle `logit_test` (a C++ reimplementation test)
  auto clamp_test = [](float v, float lo, float hi) {
    return std::max(std::min(v, hi), lo);
  };
  auto logit_test = [clamp_test](float x, float eps = 1E-6f) {
    float p = clamp_test(x, eps, 1.0f - eps);
    return std::log(p / (1.0f - p));
  };

  // property: the logit function is the right-inverse of the logistic function
  // i.e., logistic(logit(p)) == p
  auto logistic_test = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

  for (dim_t i = 0; i != size; ++i) {
    // differential test against the oracle
    EXPECT_NEAR(resultDiffH.at({i}), logit_test(inputH.at({i})), allowedError);
    // zero-sum property
    EXPECT_NEAR(resultComplH.at({i}) + resultDiffH.at({i}), 0.0f, allowedError);
    // right-inverse property
    EXPECT_NEAR(logistic_test(resultDiffH.at({i})),
                clamp_test(inputH.at({i}), eps, 1.0f - eps), allowedError);
  }
}

/// Test the Logit operator using FloatTy.
TEST_P(OperatorTest, Logit_Float) {
  CHECK_IF_ENABLED();
  testLogit<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 1E-5);
}

/// Test the Logit operator using Float16Ty.
TEST_P(OperatorTest, Logit_Float16) {
  CHECK_IF_ENABLED();
  testLogit<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.002);
}

TEST_P(OperatorTest, CmpEQ) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::Int64ITy, {2, 7}, "X", false);
  bindings_.allocate(X)->getHandle<int64_t>() = {
      0, 1, 17, 876, 1000, 44444, 9999999, 0, 1, 17, 876, 1000, 44444, 9999999};
  auto *Y = mod_.createPlaceholder(ElemKind::Int64ITy, {2, 7}, "Y", false);
  bindings_.allocate(Y)->getHandle<int64_t>() = {
      1, 2, 16, 900, 1111, 44544, 1999999, 0, 1, 17, 876, 1000, 44444, 9999999};

  auto *cmpEQ = F_->createCmpEQ("cmpEQ", X, Y);
  auto *save = F_->createSave("save", cmpEQ);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = saveTensor->getHandle<bool>();
  for (dim_t i = 0; i < 7; ++i) {
    EXPECT_FALSE(saveH.at({0, i}));
  }
  for (dim_t i = 0; i < 7; ++i) {
    EXPECT_TRUE(saveH.at({1, i}));
  }
}

/// Check that the add operator works properly with FP16.
TEST_P(OperatorTest, FP16Add) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "A", false);
  bindings_.allocate(inputA)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "B", false);
  bindings_.allocate(inputB)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *Pool = F_->createAdd("pool", inputA, inputB);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float16_t>();
  auto handleA = bindings_.get(inputA)->getHandle<float16_t>();
  auto handleB = bindings_.get(inputB)->getHandle<float16_t>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), handleA.raw(idx) + handleB.raw(idx));
  }
}

TEST_P(OperatorTest, matmul) {
  CHECK_IF_ENABLED();

  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(rhs)->getHandle() = {7, 10};

  auto *R = F_->createMatMul("MM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = saveTensor->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
}

/// Test that cloneFunInsideFun works correctly with matmuls.
TEST_P(OperatorTest, matmul_ParCloneTest10) {
  CHECK_IF_ENABLED();

  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(rhs)->getHandle() = {7, 10};

  auto *R = F_->createMatMul("MM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  CompilationContext cctx;
  const unsigned parallelCount = 10;
  auto resultTensors = cloneFunInsideFun(std::make_pair(F_, saveTensor),
                                         &bindings_, cctx, parallelCount);

  EXPECT_EQ(resultTensors.size(), parallelCount);

  EE_.compile(cctx);
  EE_.run(bindings_);

  for (Tensor *T : resultTensors) {
    auto H = T->getHandle();
    EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
    EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
    EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
  }
}

/// Test that compareAgainstInterpreter works correctly along with quantization
/// and parallel cloning.
TEST_P(OperatorTest, matmulQuantized_InterpCompareParClone) {
  CHECK_IF_ENABLED();

  constexpr unsigned parallelCount = 10;
  compareAgainstInterpreter(
      getBackendName(),
      [](PlaceholderBindings &bindings, ExecutionEngine &EE) {
        Module &mod = EE.getModule();
        Function *F = mod.createFunction("main");
        Placeholder *lhs =
            mod.createPlaceholder(ElemKind::FloatTy, {3, 2}, "lhs", false);
        Placeholder *rhs =
            mod.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
        bindings.allocate(lhs)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
        bindings.allocate(rhs)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());

        MatMulNode *R = F->createMatMul("MM", lhs, rhs);

        SaveNode *save = F->createSave("save", R);
        Tensor *saveTensor = bindings.allocate(save->getPlaceholder());
        return std::make_pair(F, saveTensor);
      },
      ElemKind::FloatTy, ElemKind::Int8QTy, 0.006, parallelCount);
}

/// Check that the matmul operator behaves correctly with FP16.
TEST_P(OperatorTest, FP16Matmul) {
  CHECK_IF_ENABLED();

  auto *lhs = mod_.createPlaceholder(ElemKind::Float16Ty, {3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::Float16Ty, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle<float16_t>() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(rhs)->getHandle<float16_t>() = {7, 10};

  auto *R = F_->createMatMul("MM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = saveTensor->getHandle<float16_t>();
  EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
}

/// Test that the broadcasted batch mat mul operator works as expected.
TEST_P(OperatorTest, BroadcastedBatchMatMul) {
  CHECK_IF_ENABLED();

  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1,  2,  3,  4,  5,  6,
                                          -1, -2, -3, -4, -5, -6};
  bindings_.allocate(rhs)->getHandle() = {7, 10};

  auto *R = F_->createBatchMatMul("BMM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
  EXPECT_NEAR(H.at({1, 0, 0}), -27, 0.001);
  EXPECT_NEAR(H.at({1, 1, 0}), -61, 0.001);
  EXPECT_NEAR(H.at({1, 2, 0}), -95, 0.001);
}

/// Test that the broadcasted batch mat mul operator works as expected when the
/// RHS does not have to be tiled.
TEST_P(OperatorTest, NonBroadcastedBatchMatMul) {
  CHECK_IF_ENABLED();
  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(rhs)->getHandle() = {7, 10};

  auto *R = F_->createBatchMatMul("BMM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
}

TEST_P(OperatorTest, ParallelBatchMatMul) {
  CHECK_IF_ENABLED();

  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "lhs", false);
  auto *rhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1,  2,  3,  4,  5,  6,
                                          -1, -2, -3, -4, -5, -6};
  bindings_.allocate(rhs)->getHandle() = {7, 10, 12, -1};

  auto *R = F_->createBatchMatMul("BMM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
  EXPECT_NEAR(H.at({1, 0, 0}), -10, 0.001);
  EXPECT_NEAR(H.at({1, 1, 0}), -32, 0.001);
  EXPECT_NEAR(H.at({1, 2, 0}), -54, 0.001);
}

static FunctionTensorPair
createAndInitParallelBatchMatMulTest(glow::PlaceholderBindings &bindings,
                                     glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *lhs =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 50, 100}, "lhs", false);
  auto *rhs =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 100, 80}, "rhs", false);
  bindings.allocate(lhs)->getHandle().randomize(-0.1, 0.1, mod.getPRNG());
  bindings.allocate(rhs)->getHandle().randomize(-0.1, 0.1, mod.getPRNG());

  auto *R = F->createBatchMatMul("BMM", lhs, rhs);

  auto *save = F->createSave("save", R);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, ParallelBatchMatMul_Float16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitParallelBatchMatMulTest, ElemKind::FloatTy,
      ElemKind::Float16Ty, 0.0005f, parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, ParallelBatchMatMul_Int8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitParallelBatchMatMulTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.002f, parCloneCountOpt);
}

/// Helper to test BatchedReduceAdd using \p DTy.
template <typename DataType>
static void testBatchedReduceAdd(glow::PlaceholderBindings &bindings,
                                 glow::Module &mod, glow::Function *F,
                                 glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *batch = mod.createPlaceholder(DTy, {2, 4}, "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {10, 20, 30, 40,
                                                     1,  2,  3,  4};

  auto *R = F->createBatchedReduceAdd("reduce.add", batch, /* axis */ 0);

  auto *save = F->createSave("save", R);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected(DTy, {4});
  expected.getHandle<DataType>() = {11, 22, 33, 44};
  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchedReduceAdd is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceAdd_Float) {
  CHECK_IF_ENABLED();

  testBatchedReduceAdd<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that BatchedReduceAdd is correctly supported in Float16Ty.
TEST_P(OperatorTest, batchedReduceAdd_Float16) {
  CHECK_IF_ENABLED();
  testBatchedReduceAdd<float16_t>(bindings_, mod_, F_, EE_,
                                  ElemKind::Float16Ty);
}

/// Test that BatchedReduceAdd works correctly reducing the outermost axis.
TEST_P(OperatorTest, batchedReduceAdd_outerAxis) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {10, 20, 30, 40, 1, 2, 3, 4,
                                                   10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 4});
  expected.getHandle<float>() = {20, 40, 60, 80, 2, 4, 6, 8};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchedReduceAdd works correctly reducing an internal axis.
TEST_P(OperatorTest, batchedReduceAdd_innerAxis) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {10, 20, 30, 40, 1, 2, 3, 4,
                                                   10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 1);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 4});
  expected.getHandle<float>() = {11, 22, 33, 44, 11, 22, 33, 44};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchedReduceAdd works correctly reducing the innermost axis.
TEST_P(OperatorTest, batchedReduceAdd_lastAxis) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {10, 20, 30, 40, 1, 2, 3, 4,
                                                   10, 20, 30, 40, 1, 2, 3, 4};
  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 2);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 2});
  expected.getHandle<float>() = {100, 10, 100, 10};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchReducedAdd works on a 4D input.
TEST_P(OperatorTest, batchedReduceAdd_4Dinput) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4,
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 2, 4});
  expected.getHandle<float>() = {20, 40, 60, 80, 2, 4, 6, 8,
                                 20, 40, 60, 80, 2, 4, 6, 8};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchReducedAdd works on a 5D input.
TEST_P(OperatorTest, batchedReduceAdd_5Dinput) {
  CHECK_IF_ENABLED();
  auto *batch = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 2, 4},
                                       "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4,
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4,
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4,
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 2);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 2, 2, 4});
  expected.getHandle<float>() = {20, 40, 60, 80, 2,  4,  6,  8,  20, 40, 60,
                                 80, 2,  4,  6,  8,  20, 40, 60, 80, 2,  4,
                                 6,  8,  20, 40, 60, 80, 2,  4,  6,  8};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Helper to test BatchedReduceMin using \p DTy.
template <typename DataType>
static void testBatchedReduceMin(glow::PlaceholderBindings &bindings,
                                 glow::Module &mod, glow::Function *F,
                                 glow::ExecutionEngine &EE, ElemKind DTy) {

  auto *batch = mod.createPlaceholder(DTy, {2, 4}, "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {10, 20, 30, 40,
                                                     1,  2,  3,  4};
  auto *R = F->createBatchedReduceMin("reduce.min", batch, /* axis */ 0);

  auto *save = F->createSave("save", R);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected(DTy, {4});
  expected.getHandle<DataType>() = {1, 2, 3, 4};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Helper to test BatchedReduceMin using \p DTy.
template <typename DataType>
static void testBatchedReduceMinMultiAxis(glow::PlaceholderBindings &bindings,
                                          glow::Module &mod, glow::Function *F,
                                          glow::ExecutionEngine &EE,
                                          ElemKind DTy) {
  auto *batch = mod.createPlaceholder(DTy, {2, 2, 2, 2}, "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {
      1, -2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  auto *R = F->createBatchedReduceMin("reduce.min", batch, /* axis */ {1, 3});
  auto *save = F->createSave("save", R);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected(DTy, {2, 2});
  expected.getHandle<DataType>() = {-2, -4, 9, 11};
  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchedReduceMin is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceMin_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceMin<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that BatchedReduceMin is correctly supported in Int32Ty.
TEST_P(OperatorTest, batchedReduceMin_Int32) {
  CHECK_IF_ENABLED();
  testBatchedReduceMin<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Test that BatchedReduceMin is correctly supported in Int64Ty.
TEST_P(OperatorTest, batchedReduceMin_Int64) {
  CHECK_IF_ENABLED();
  testBatchedReduceMin<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test that BatchedReduceMin is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceMinMultiAxis_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceMinMultiAxis<float>(bindings_, mod_, F_, EE_,
                                       ElemKind::FloatTy);
}

/// Test that BatchedReduceMin is correctly supported in Int32Ty.
TEST_P(OperatorTest, batchedReduceMinMultiAxis_Int32) {
  CHECK_IF_ENABLED();
  testBatchedReduceMinMultiAxis<int32_t>(bindings_, mod_, F_, EE_,
                                         ElemKind::Int32ITy);
}

/// Test that BatchedReduceMin is correctly supported in Int64Ty.
TEST_P(OperatorTest, batchedReduceMinMultiAxis_Int64) {
  CHECK_IF_ENABLED();
  testBatchedReduceMinMultiAxis<int64_t>(bindings_, mod_, F_, EE_,
                                         ElemKind::Int64ITy);
}

/// Helper to test BatchedReduceZeroDimResult using \p DTy.
template <typename DataType>
static void testBatchedReduceZeroDimResult(glow::PlaceholderBindings &bindings,
                                           glow::Module &mod, glow::Function *F,
                                           glow::ExecutionEngine &EE,
                                           ElemKind DTy) {
  auto *batch = createPlaceholderConditionallyQuantized(
      mod, DTy, {4}, "batch", /* isTrainable */ false, "N");
  bindings.allocate(batch)->getHandle<DataType>() = {2, 4, 6, 8};

  auto OT = uniqueTypeConditionallyQuantized(mod, DTy, {});
  auto *RA = F->createBatchedReduceAdd("reduce.add", OT, batch, /* axis */ 0);
  auto *RM = F->createBatchedReduceMean("reduce.mean", OT, batch, /* axis */ 0);
  auto *saveRA = F->createSave("saveRA", RA);
  auto *saveRM = F->createSave("saveRM", RM);
  auto *resultRA = bindings.allocate(saveRA->getPlaceholder());
  auto *resultRM = bindings.allocate(saveRM->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto RAH = resultRA->getHandle<DataType>();
  auto RMH = resultRM->getHandle<DataType>();
  if (isQuantizedElemKind(DTy)) {
    EXPECT_EQ(RAH.at({}), static_cast<DataType>(20));
    EXPECT_EQ(RMH.at({}), static_cast<DataType>(5));
  } else {
    EXPECT_NEAR(RAH.at({}), 20, 0.001);
    EXPECT_NEAR(RMH.at({}), 5, 0.001);
  }
}

/// Test reduction down to a zero-dim tensor on FloatTy.
TEST_P(OperatorTest, batchedReduceZeroDimResult_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceZeroDimResult<float>(bindings_, mod_, F_, EE_,
                                        ElemKind::FloatTy);
}

/// Test reduction down to a zero-dim tensor on Float16Ty.
TEST_P(OperatorTest, batchedReduceZeroDimResult_Float16) {
  CHECK_IF_ENABLED();
  testBatchedReduceZeroDimResult<float16_t>(bindings_, mod_, F_, EE_,
                                            ElemKind::Float16Ty);
}

/// Test reduction down to a zero-dim tensor on Int8QTy.
TEST_P(OperatorTest, batchedReduceZeroDimResult_Int8) {
  CHECK_IF_ENABLED();
  testBatchedReduceZeroDimResult<int8_t>(bindings_, mod_, F_, EE_,
                                         ElemKind::Int8QTy);
}

/// Helper to test BatchedReduceAddWithAxis using \p DTy.
template <typename DataType>
static void testBatchedReduceAddWithAxis(glow::PlaceholderBindings &bindings,
                                         glow::Module &mod, glow::Function *F,
                                         glow::ExecutionEngine &EE,
                                         ElemKind DTy) {
  auto *batch = createPlaceholderConditionallyQuantized(mod, DTy, {2, 3, 2},
                                                        "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {0, 1, 2, 3, 4,  5,
                                                     6, 7, 8, 9, 10, 11};

  auto OT1 = uniqueTypeConditionallyQuantized(mod, DTy, {2, 2});
  auto *R1 =
      F->createBatchedReduceAdd("reduce.add.axis.1", OT1, batch, /* axis */ 1);
  auto OT2 = uniqueTypeConditionallyQuantized(mod, DTy, {2, 3});
  auto *R2 =
      F->createBatchedReduceAdd("reduce.add.axis.2", OT2, batch, /* axis */ 2);
  auto *save1 = F->createSave("save1", R1);
  auto *save2 = F->createSave("save2", R2);

  auto *result1 = bindings.allocate(save1->getPlaceholder());
  auto *result2 = bindings.allocate(save2->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto expected1 = createTensorConditionallyQuantized(DTy, {2, 2});
  expected1.getHandle<DataType>() = {6, 9, 24, 27};
  EXPECT_TRUE(result1->isEqual(expected1));

  auto expected2 = createTensorConditionallyQuantized(DTy, {2, 3});
  expected2.getHandle<DataType>() = {1, 5, 9, 13, 17, 21};
  EXPECT_TRUE(result2->isEqual(expected2));
}

/// Test that batchedReduceAddWithAxis is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceAddWithAxis_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceAddWithAxis<float>(bindings_, mod_, F_, EE_,
                                      ElemKind::FloatTy);
}

/// Test that batchedReduceAddWithAxis is correctly supported in Float16Ty.
TEST_P(OperatorTest, batchedReduceAddWithAxis_Float16) {
  CHECK_IF_ENABLED();
  testBatchedReduceAddWithAxis<float16_t>(bindings_, mod_, F_, EE_,
                                          ElemKind::Float16Ty);
}

/// Test that batchedReduceAddWithAxis is correctly supported in Int8QTy.
TEST_P(OperatorTest, batchedReduceAddWithAxis_Int8Q) {
  CHECK_IF_ENABLED();
  testBatchedReduceAddWithAxis<int8_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::Int8QTy);
}

TEST_P(OperatorTest, batchedReduceAddQuantized) {
  CHECK_IF_ENABLED();

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {3, 8}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {8}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3, 8}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  bindings_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = bindings_.get(batch)->getHandle<int8_t>();

  auto *R =
      F_->createBatchedReduceAdd("batched.reduce.add", OT, batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < 8; i++) {
    std::array<int32_t, 3> b{{BH.at({0, i}), BH.at({1, i}), BH.at({2, i})}};
    float s = BT->getScale() / OT->getScale();
    int32_t o = BT->getOffset();
    float result = (b[0] - o) + (b[1] - o) + (b[2] - o);
    result = s * result + OT->getOffset();

    EXPECT_NEAR(std::round(result), OH.at({i}), 1.0);
  }
}

TEST_P(OperatorTest, batchedReduceAddQuantizedWithAxis) {
  CHECK_IF_ENABLED();

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {2, 3, 4}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {2, 4}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 3, 4}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  bindings_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = bindings_.get(batch)->getHandle<int8_t>();

  auto *R =
      F_->createBatchedReduceAdd("batched.reduce.add", OT, batch, /* axis */ 1);
  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < 2; i++) {
    for (dim_t j = 0; j < 4; j++) {
      std::array<int32_t, 3> b{
          {BH.at({i, 0, j}), BH.at({i, 1, j}), BH.at({i, 2, j})}};
      float s = BT->getScale() / OT->getScale();
      int32_t o = BT->getOffset();
      float result = (b[0] - o) + (b[1] - o) + (b[2] - o);
      result = s * result + OT->getOffset();

      EXPECT_NEAR(std::round(result), OH.at({i, j}), 1.0);
    }
  }
}

TEST_P(OperatorTest, batchedReduceMean) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceMean("reduce.add", batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0}), 5.5, 0.001);
  EXPECT_NEAR(H.at({1}), 11.0, 0.001);
  EXPECT_NEAR(H.at({2}), 16.5, 0.001);
  EXPECT_NEAR(H.at({3}), 22.0, 0.001);
}

TEST_P(OperatorTest, batchedReduceMeanWithAxis) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "batch", false);
  bindings_.allocate(batch)->getHandle() = {0, 1, 2, 3, 4,  5,
                                            6, 7, 8, 9, 10, 11};

  auto *R = F_->createBatchedReduceMean("reduce.add", batch, /* axis */ 1);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 2.0, 0.001);
  EXPECT_NEAR(H.at({0, 1}), 3.0, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 8.0, 0.001);
  EXPECT_NEAR(H.at({1, 1}), 9.0, 0.001);
}

TEST_P(OperatorTest, batchedReduceMeanQuantized) {
  CHECK_IF_ENABLED();

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {3, 8}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {8}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3, 8}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  bindings_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = bindings_.get(batch)->getHandle<int8_t>();

  auto *R = F_->createBatchedReduceMean("batched.reduce.add", OT, batch,
                                        /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < 8; i++) {
    std::array<int32_t, 3> b{{BH.at({0, i}), BH.at({1, i}), BH.at({2, i})}};
    float s = BT->getScale() / OT->getScale();
    int32_t o = BT->getOffset();
    float result = ((b[0] - o) + (b[1] - o) + (b[2] - o)) / 3;
    result = s * result + OT->getOffset();

    EXPECT_NEAR(std::round(result), OH.at({i}), 1.0);
  }
}

TEST_P(OperatorTest, batchedReduceMeanQuantizedWithAxis) {
  CHECK_IF_ENABLED();

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {2, 3, 4}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {2, 4}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 3, 4}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  bindings_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = bindings_.get(batch)->getHandle<int8_t>();

  auto *R = F_->createBatchedReduceMean("batched.reduce.add", OT, batch,
                                        /* axis */ 1);
  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < 2; i++) {
    for (dim_t j = 0; j < 4; j++) {
      std::array<int32_t, 3> b{
          {BH.at({i, 0, j}), BH.at({i, 1, j}), BH.at({i, 2, j})}};
      float s = BT->getScale() / OT->getScale();
      int32_t o = BT->getOffset();
      float result = ((b[0] - o) + (b[1] - o) + (b[2] - o)) / 3;
      result = s * result + OT->getOffset();

      EXPECT_NEAR(std::round(result), OH.at({i, j}), 1.0);
    }
  }
}

/// Verify that batchedReduceMean optimization using AvgPool works correctly.
TEST_P(OperatorTest, batchedReduceMeanUsingAvgPool) {
  CHECK_IF_ENABLED();

  std::vector<dim_t> dims = {3, 20, 4, 8};

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, dims, "batch", false, "NHWC");

  auto IH = bindings_.allocate(batch)->getHandle();
  IH.randomize(1.0, 100.0, mod_.getPRNG());

  auto *R = F_->createBatchedReduceMean("reduce.mean", batch, {2, 3});

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);
  auto H = result->getHandle();

  std::array<std::array<float, 20>, 3> results{};
  for (dim_t i = 0; i < dims[0]; i++) {
    for (dim_t j = 0; j < dims[1]; j++) {
      for (dim_t k = 0; k < dims[2]; k++) {
        for (dim_t l = 0; l < dims[3]; l++) {
          results[i][j] += IH.at({i, j, k, l});
        }
      }
      results[i][j] /= (dims[2] * dims[3]);
      EXPECT_NEAR(H.at({i, j}), results[i][j], 0.001);
    }
  }
}

/// Verify that quantized batchedReduceMean optimization using AvgPool works
/// correctly.
TEST_P(OperatorTest, batchedReduceMeanUsingAvgPoolQuantized) {
  CHECK_IF_ENABLED();

  std::vector<dim_t> dims = {2, 3, 3, 4};

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, dims, 1, 0);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {dims[0], dims[1]}, 1, 0);
  auto *batch = mod_.createPlaceholder(ElemKind::Int8QTy, dims, BT->getScale(),
                                       BT->getOffset(), "batch", false);

  auto IH = bindings_.allocate(batch)->getHandle<int8_t>();
  IH.randomize(1, 100, mod_.getPRNG());

  auto *R = F_->createBatchedReduceMean("reduce.mean", OT, batch, {2, 3});

  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::array<std::array<float, 3>, 2> results{};
  float s = BT->getScale() / OT->getScale();
  for (dim_t i = 0; i < dims[0]; i++) {
    for (dim_t j = 0; j < dims[1]; j++) {
      for (dim_t k = 0; k < dims[2]; k++) {
        int32_t o = BT->getOffset();
        for (dim_t l = 0; l < dims[3]; l++) {
          results[i][j] += IH.at({i, j, k, l}) - o;
        }
      }
      results[i][j] = s * results[i][j] + OT->getOffset();
      results[i][j] /= (dims[2] * dims[3]);
      EXPECT_NEAR(std::round(results[i][j]), OH.at({i, j}), 1.0);
    }
  }
}

/// Test that the BatchedAdd operator works.
TEST_P(OperatorTest, BatchedAdd) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 3}, "batch", false);
  auto *added =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "added", false);

  bindings_.allocate(batch)->getHandle() = {9, 8, 7, 6, 5,  4,  3,  4,  5,
                                            6, 7, 8, 9, 10, 11, 12, 13, 14};
  bindings_.allocate(added)->getHandle().clear(1.0);

  auto *R = F_->createBatchedAdd("batch.add", batch, added);
  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto BH = bindings_.get(batch)->getHandle();
  auto RH = result->getHandle();
  for (dim_t i = 0; i < 2; i++) {
    for (dim_t j = 0; j < 3; j++) {
      for (dim_t k = 0; k < 3; k++) {
        EXPECT_NEAR(RH.at({i, j, k}), BH.at({i, j, k}) + 1.0, 0.001);
      }
    }
  }
}

/// Broadcast Tensor of shape (2,1,1) to (2,4,2) with axis 0.
TEST_P(OperatorTest, broadcastSimple) {
  CHECK_IF_ENABLED();

  const dim_t numDims_A = 3;
  const dim_t dimY_A = 2;
  const dim_t dimZ_A = 4;
  const dim_t dimW_A = 2;
  const dim_t dims_A[numDims_A] = {dimY_A, dimZ_A, dimW_A};

  const dim_t numDims_B = 3;
  const dim_t dimY_B = 2;
  const dim_t dimZ_B = 1;
  const dim_t dimW_B = 1;
  const dim_t dims_B[numDims_B] = {dimY_B, dimZ_B, dimW_B};

  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, dims_B, "B", false);
  auto *QB =
      mod_.createPlaceholder(ElemKind::Int8QTy, dims_B, 1.1, -2, "QB", false);
  auto H_B = bindings_.allocate(B)->getHandle();
  auto H_QB = bindings_.allocate(QB)->getHandle<int8_t>();
  H_B = {20, 10};
  H_QB = {35, -18};

  const unsigned axis = 0;

  auto *R = F_->createBroadcast("broadcasted", B, dims_A, axis);
  auto *QR = F_->createBroadcast("broadcastedQ", QB, dims_A, axis);

  auto *save = F_->createSave("save", R);
  auto *broadcasted = bindings_.allocate(save->getPlaceholder());

  auto *saveQ = F_->createSave("saveQ", QR);
  auto *broadcastedQ = bindings_.allocate(saveQ->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto broadcastedBHandle = broadcasted->getHandle();
  auto broadcastedQBHandle = broadcastedQ->getHandle<int8_t>();
  // Verify broadcasted B has same shape.
  EXPECT_EQ(broadcastedBHandle.dims().size(), numDims_A);
  EXPECT_EQ(broadcastedQBHandle.dims().size(), numDims_A);
  for (size_t i = 0; i < broadcastedBHandle.dims().size(); i++) {
    EXPECT_EQ(broadcastedBHandle.dims()[i], dims_A[i]);
    EXPECT_EQ(broadcastedQBHandle.dims()[i], dims_A[i]);
  }

  // Look at the two values in X_B and verify in the three dimensions it was
  // broadcasted that the values were correctly broadcasted.
  const dim_t k_B = 0;
  const dim_t l_B = 0;
  for (dim_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({j_B, k_B, l_B});
    const int8_t origValQ = H_QB.at({j_B, k_B, l_B});
    const dim_t j_A = j_B; // This dim was not broadcasted (dims were equal).
    for (dim_t k_A = 0; k_A < dimZ_A; k_A++) {
      for (dim_t l_A = 0; l_A < dimW_A; l_A++) {
        EXPECT_EQ(broadcastedBHandle.at({j_A, k_A, l_A}), origVal);
        EXPECT_EQ(broadcastedQBHandle.at({j_A, k_A, l_A}), origValQ);
      }
    }
  }
}

/// Broadcast a Tensor of shape (2,1) to (3,2,4,2) with axis 1.
TEST_P(OperatorTest, broadcast) {
  CHECK_IF_ENABLED();

  const dim_t numDims_A = 4;
  const dim_t dimX_A = 3;
  const dim_t dimY_A = 2;
  const dim_t dimZ_A = 4;
  const dim_t dimW_A = 2;
  const dim_t dims_A[numDims_A] = {dimX_A, dimY_A, dimZ_A, dimW_A};

  const dim_t numDims_B = 2;
  const dim_t dimY_B = 2;
  const dim_t dimZ_B = 1;
  const dim_t dims_B[numDims_B] = {dimY_B, dimZ_B};

  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, dims_B, "B", false);
  auto *QB =
      mod_.createPlaceholder(ElemKind::Int8QTy, dims_B, 0.8, 3, "QB", false);

  auto H_B = bindings_.allocate(B)->getHandle();
  auto H_QB = bindings_.allocate(QB)->getHandle<int8_t>();
  H_B = {20, 10};
  H_QB = {-8, 41};

  const unsigned axis = 1;

  auto *R = F_->createBroadcast("broadcasted", B, dims_A, axis);
  auto *QR = F_->createBroadcast("broadcastedQ", QB, dims_A, axis);

  auto *save = F_->createSave("save", R);
  auto *broadcasted = bindings_.allocate(save->getPlaceholder());

  auto *saveQ = F_->createSave("saveQ", QR);
  auto *broadcastedQ = bindings_.allocate(saveQ->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto broadcastedBHandle = broadcasted->getHandle();
  auto broadcastedQBHandle = broadcastedQ->getHandle<int8_t>();
  // Verify broadcasted B has same shape.
  EXPECT_EQ(broadcastedBHandle.dims().size(), numDims_A);
  EXPECT_EQ(broadcastedQBHandle.dims().size(), numDims_A);
  for (size_t i = 0; i < broadcastedBHandle.dims().size(); i++) {
    EXPECT_EQ(broadcastedBHandle.dims()[i], dims_A[i]);
    EXPECT_EQ(broadcastedQBHandle.dims()[i], dims_A[i]);
  }

  // Look at the two values in X_B and verify in the three dimensions it was
  // broadcasted that the values were correctly broadcasted.
  const dim_t k_B = 0;
  for (dim_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({j_B, k_B});
    const int8_t origValQ = H_QB.at({j_B, k_B});
    const dim_t j_A = j_B; // This dim was not broadcasted (dims were equal).
    for (dim_t i_A = 0; i_A < dimX_A; i_A++) {
      for (dim_t k_A = 0; k_A < dimZ_A; k_A++) {
        for (dim_t l_A = 0; l_A < dimW_A; l_A++) {
          EXPECT_EQ(broadcastedBHandle.at({i_A, j_A, k_A, l_A}), origVal);
          EXPECT_EQ(broadcastedQBHandle.at({i_A, j_A, k_A, l_A}), origValQ);
        }
      }
    }
  }
}

/// Perform a simple weighted sum.
TEST_P(OperatorTest, weightedSum) {
  CHECK_IF_ENABLED();

  // Create the data.
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "A", false);
  bindings_.allocate(A)->getHandle() = {1.0, 2.0, 3.0, 4.0};

  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "B", false);
  bindings_.allocate(B)->getHandle() = {5.0, 6.0, 7.0, 8.0};

  // Create the weights.
  auto *AW = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "AW", false);
  bindings_.allocate(AW)->getHandle() = {0.1f};

  auto *BW = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "BW", false);
  bindings_.allocate(BW)->getHandle() = {10.0f};

  // Create the weighted sum with the data and weights, and save it.
  auto *WS = F_->createWeightedSum("ws", {A, B}, {AW, BW});
  auto *save = F_->createSave("save", WS);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  // Verify the weighted sum was correctly calculated.
  auto resultH = saveTensor->getHandle();
  EXPECT_NEAR(resultH.at({0, 0}), 50.1, 1E-5);
  EXPECT_NEAR(resultH.at({0, 1}), 60.2, 1E-5);
  EXPECT_NEAR(resultH.at({1, 0}), 70.3, 1E-5);
  EXPECT_NEAR(resultH.at({1, 1}), 80.4, 1E-5);
}

/// Helper to test ReluSimple using \p DTy.
template <typename DataType>
static void testReluSimple(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *in = mod.createPlaceholder(DTy, {7}, "in", false);
  auto *relu = F->createRELU("relu", in);
  auto *save = F->createSave("relu", relu);
  auto *result = bindings.allocate(save->getPlaceholder());

  bindings.allocate(in)->getHandle<DataType>() = {0, -1, -2, -3, 4, 5, 6};

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultH = result->getHandle<DataType>();

  for (size_t i = 0; i < 7; i++) {
    if (i < 4) {
      EXPECT_EQ(resultH.raw(i), static_cast<DataType>(0));
    } else {
      EXPECT_EQ(resultH.raw(i), static_cast<DataType>(i));
    }
  }
}

/// Verify that the RELU operator works correctly for Float.
TEST_P(OperatorTest, ReluSimple_Float) {
  CHECK_IF_ENABLED();

  testReluSimple<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Verify that the RELU operator works correctly for Float16.
TEST_P(OperatorTest, ReluSimple_Float16) {
  CHECK_IF_ENABLED();
  testReluSimple<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Helper to test PReluSimple using \p DTy.
template <typename DataType>
static void testPReluSimple(glow::PlaceholderBindings &bindings,
                            glow::Module &mod, glow::Function *F,
                            glow::ExecutionEngine &EE, ElemKind DTy,
                            double allowedError) {
  auto *in = mod.createPlaceholder(DTy, {7}, "in", false);
  auto *slope = mod.createPlaceholder(DTy, {7}, "slope", false);
  auto *prelu = F->createPRELU("prelu", in, slope);
  auto *save = F->createSave("prelu", prelu);
  auto *result = bindings.allocate(save->getPlaceholder());

  bindings.allocate(in)->getHandle<DataType>() = {0, -1, -2, -3, 4, 5, 6};
  bindings.allocate(slope)->getHandle<DataType>().randomize(0.1, 3.0,
                                                            mod.getPRNG());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultH = result->getHandle<DataType>();
  auto inH = bindings.get(in)->getHandle<DataType>();
  auto slopeH = bindings.get(slope)->getHandle<DataType>();

  for (size_t i = 0; i < 7; i++) {
    DataType expectedResult =
        slopeH.raw(i) * std::min<DataType>(0, inH.raw(i)) +
        std::max<DataType>(0, inH.raw(i));
    EXPECT_NEAR(resultH.at(i), expectedResult, allowedError);
  }
}

/// Verify that the PRELU operator works correctly for Float.
TEST_P(OperatorTest, PReluSimple_Float) {
  CHECK_IF_ENABLED();
  testPReluSimple<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 1E-32);
}

/// Verify that the PRELU operator works correctly for Float16.
TEST_P(OperatorTest, PReluSimple_Float16) {
  CHECK_IF_ENABLED();
  testPReluSimple<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                             1E-16);
}

TEST_P(OperatorTest, TopK) {
  CHECK_IF_ENABLED();

  auto *inp =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 5}, "input", false);
  auto *values =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 3}, "values", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {3, 1, 3}, "indices", false);

  bindings_.allocate(inp)->getHandle() = {
      28, 4, 411, 19, 42, 0.4f, 0.4f, 0.4f, -0.4f, 0.45f, 7, 5, 9, 8, 100,
  };
  bindings_.allocate(values);
  bindings_.allocate(indices);

  auto *R = F_->createTopK("TopK", inp, 3);

  F_->createSave("save.values", {R, 0}, values);
  F_->createSave("save.indices", {R, 1}, indices);

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto V = bindings_.get(values)->getHandle();
  auto I = bindings_.get(indices)->getHandle<sdim_t>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 411);
  EXPECT_EQ(I.at({0, 0, 0}), 2);
  EXPECT_FLOAT_EQ(V.at({0, 0, 1}), 42);
  EXPECT_EQ(I.at({0, 0, 1}), 4);
  EXPECT_FLOAT_EQ(V.at({0, 0, 2}), 28);
  EXPECT_EQ(I.at({0, 0, 2}), 0);

  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 0.45);
  EXPECT_EQ(I.at({1, 0, 0}), 4);
  EXPECT_FLOAT_EQ(V.at({1, 0, 1}), 0.4);
  EXPECT_EQ(I.at({1, 0, 1}), 0);
  EXPECT_FLOAT_EQ(V.at({1, 0, 2}), 0.4);
  EXPECT_EQ(I.at({1, 0, 2}), 1);

  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 100);
  EXPECT_EQ(I.at({2, 0, 0}), 4);
  EXPECT_FLOAT_EQ(V.at({2, 0, 1}), 9);
  EXPECT_EQ(I.at({2, 0, 1}), 2);
  EXPECT_FLOAT_EQ(V.at({2, 0, 2}), 8);
  EXPECT_EQ(I.at({2, 0, 2}), 3);
}

template <typename DataType>
static void testArgMaxKeepDim(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {2, 3, 2, 2},
                                                        "input", false, "NHWC");
  auto *argmax = mod.createPlaceholder(IndexElemKind, {1, 3, 2, 2}, "argmax",
                                       false, "NHWC");

  bindings.allocate(input)->getHandle<DataType>() = {
      11, 24, 33, 41, 15, 26, 37, 48, 12, 28, 31, 42,
      13, 24, 35, 46, 12, 28, 39, 40, 11, 22, 33, 47};
  bindings.allocate(argmax);

  auto *AM = F->createArgMax("argmax", input, 0, true);
  F->createSave("save.argmax", AM, argmax);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto I = bindings.get(argmax)->getHandle<sdim_t>();
  EXPECT_EQ(I.raw(0), 1);
  EXPECT_EQ(I.raw(1), 0);
  EXPECT_EQ(I.raw(2), 1);
  EXPECT_EQ(I.raw(3), 1);
  EXPECT_EQ(I.raw(4), 0);
  EXPECT_EQ(I.raw(5), 1);
  EXPECT_EQ(I.raw(6), 1);
  EXPECT_EQ(I.raw(7), 0);
  EXPECT_EQ(I.raw(8), 0);
  EXPECT_EQ(I.raw(9), 0);
  EXPECT_EQ(I.raw(10), 1);
  EXPECT_EQ(I.raw(11), 1);
}

TEST_P(OperatorTest, FloatArgMaxKeepDim) {
  CHECK_IF_ENABLED();
  testArgMaxKeepDim<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedArgMaxKeepDim) {
  CHECK_IF_ENABLED();
  testArgMaxKeepDim<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

template <typename DataType>
static void testArgMaxNoKeepDim(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {2, 3, 2, 2},
                                                        "input", false, "NHWC");
  auto *argmax =
      mod.createPlaceholder(IndexElemKind, {2, 2, 2}, "argmax", false);

  bindings.allocate(input)->getHandle<DataType>() = {
      11, 24, 33, 41, 15, 26, 37, 48, 12, 28, 31, 42,
      13, 24, 35, 46, 12, 28, 39, 40, 11, 22, 33, 47};
  bindings.allocate(argmax);

  auto *AM = F->createArgMax("argmax", input, 1, false);
  F->createSave("save.argmax", AM, argmax);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto I = bindings.get(argmax)->getHandle<sdim_t>();
  EXPECT_EQ(I.raw(0), 1);
  EXPECT_EQ(I.raw(1), 2);
  EXPECT_EQ(I.raw(2), 1);
  EXPECT_EQ(I.raw(3), 1);
  EXPECT_EQ(I.raw(4), 0);
  EXPECT_EQ(I.raw(5), 1);
  EXPECT_EQ(I.raw(6), 1);
  EXPECT_EQ(I.raw(7), 2);
}

TEST_P(OperatorTest, FloatArgMaxNoKeepDim) {
  CHECK_IF_ENABLED();
  testArgMaxNoKeepDim<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedArgMaxNoKeepDim) {
  CHECK_IF_ENABLED();
  testArgMaxNoKeepDim<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

// Check that concatenating Nodes with multiple outputs works correctly.
TEST_P(OperatorTest, ConcatTopK) {
  CHECK_IF_ENABLED();

  auto *inp1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  auto *inp2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {4, 1, 2}, "indices", false);

  bindings_.allocate(inp1)->getHandle() = {1, 2, 3, 17.4f, -0.1f, -10.1f};
  bindings_.allocate(inp2)->getHandle() = {1, 2, -3, -17.4f, -0.1f, -10.1f};

  auto *R1 = F_->createTopK("TopK1", inp1, 2);
  auto *R2 = F_->createTopK("TopK2", inp2, 2);

  // Concat the values and indices separately, both on the 0th dimension,
  // matching the shapes of the values and indices variables above.
  auto *CV =
      F_->createConcat("Concat.Values", {R1->getValues(), R2->getValues()}, 0);
  auto *CI = F_->createConcat("Concat.Indices",
                              {R1->getIndices(), R2->getIndices()}, 0);

  auto *saveValues = F_->createSave("Save.Values", CV);
  auto *saveValuesTensor = bindings_.allocate(saveValues->getPlaceholder());

  auto *saveIndices = F_->createSave("Save.Indices", CI, indices);
  auto *saveIndicesTensor = bindings_.allocate(saveIndices->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto V = saveValuesTensor->getHandle();
  auto I = saveIndicesTensor->getHandle<sdim_t>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 3);
  EXPECT_FLOAT_EQ(I.at({0, 0, 0}), 2);
  EXPECT_FLOAT_EQ(V.at({0, 0, 1}), 2);
  EXPECT_FLOAT_EQ(I.at({0, 0, 1}), 1);

  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 17.4f);
  EXPECT_FLOAT_EQ(I.at({1, 0, 0}), 0);
  EXPECT_FLOAT_EQ(V.at({1, 0, 1}), -0.1f);
  EXPECT_FLOAT_EQ(I.at({1, 0, 1}), 1);

  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 2);
  EXPECT_FLOAT_EQ(I.at({2, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({2, 0, 1}), 1);
  EXPECT_FLOAT_EQ(I.at({2, 0, 1}), 0);

  EXPECT_FLOAT_EQ(V.at({3, 0, 0}), -0.1f);
  EXPECT_FLOAT_EQ(I.at({3, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({3, 0, 1}), -10.1f);
  EXPECT_FLOAT_EQ(I.at({3, 0, 1}), 2);
}

// Check that matrix multiplication works well on some predefined values.
TEST_P(OperatorTest, matmul2) {
  CHECK_IF_ENABLED();

  auto *inp0 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input0", false);
  auto *inp1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input1", false);
  auto *inp2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input2", false);
  auto *rot = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "rot", false);

  float deg = 45.0 / 180.0 * 3.1415926;
  // Use the rotation matrix to manipulate some values.
  // https://en.wikipedia.org/wiki/Rotation_matrix
  bindings_.allocate(rot)->getHandle() = {
      cosf(deg),
      -sinf(deg),
      sinf(deg),
      cosf(deg),
  };

  // Some test vectors.
  bindings_.allocate(inp0)->getHandle() = {1, 4};
  bindings_.allocate(inp1)->getHandle() = {14, 2};
  bindings_.allocate(inp2)->getHandle() = {5, 2};

  auto *A0 = F_->createMatMul("m0", inp0, rot);
  auto *A1 = F_->createMatMul("m1", inp1, rot);
  auto *A2 = F_->createMatMul("m2", inp2, rot);

  auto *res0 = F_->createSave("save.values", A0);
  bindings_.allocate(res0->getPlaceholder());
  auto *res1 = F_->createSave("save.values", A1);
  bindings_.allocate(res1->getPlaceholder());
  auto *res2 = F_->createSave("save.values", A2);
  bindings_.allocate(res2->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto R0 = bindings_.get(res0->getPlaceholder())->getHandle();
  auto R1 = bindings_.get(res1->getPlaceholder())->getHandle();
  auto R2 = bindings_.get(res2->getPlaceholder())->getHandle();

  EXPECT_FLOAT_EQ(R0.at({0, 0}), 3.5355339);
  EXPECT_FLOAT_EQ(R0.at({0, 1}), 2.1213205);
  EXPECT_FLOAT_EQ(R1.at({0, 0}), 11.313709);
  EXPECT_FLOAT_EQ(R1.at({0, 1}), -8.485281);
  EXPECT_FLOAT_EQ(R2.at({0, 0}), 4.9497476);
  EXPECT_FLOAT_EQ(R2.at({0, 1}), -2.1213202);
}

// Check the TopK operator for the special case of K=1.
TEST_P(OperatorTest, TopK1) {
  CHECK_IF_ENABLED();

  auto *inp =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 5}, "input", false);

  bindings_.allocate(inp)->getHandle() = {
      0, 18, 7, 16, 5, 14, 33, 2, 41, 0, 1, -23, 34, 4, -5,
  };

  auto *R = F_->createTopK("TopK", inp, 1);

  auto *values = F_->createSave("save.values", {R, 0});
  bindings_.allocate(values->getPlaceholder());

  auto *indices = F_->createSave("save.indices", {R, 1});
  bindings_.allocate(indices->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto V = bindings_.get(values->getPlaceholder())->getHandle();
  auto I = bindings_.get(indices->getPlaceholder())->getHandle<sdim_t>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 18);
  EXPECT_EQ(I.at({0, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 41);
  EXPECT_EQ(I.at({1, 0, 0}), 3);
  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 34);
  EXPECT_EQ(I.at({2, 0, 0}), 2);
}

TEST_P(OperatorTest, QuantizedTopK) {
  CHECK_IF_ENABLED();

  auto *INV = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 1, 5}, 1.2, 5,
                                     "input", false);
  bindings_.allocate(INV)->getHandle<int8_t>() = {
      -12, -28, -7, 8, -93, 0, 10, 3, -1, 10, -2, 3, -2, 3, 3,
  };

  auto *TK = F_->createTopK("TopK", INV, 3);

  auto *values = F_->createSave("save.values", TK->getValues());
  bindings_.allocate(values->getPlaceholder());
  auto *indices = F_->createSave("save.indices", TK->getIndices());
  bindings_.allocate(indices->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto VH = bindings_.get(values->getPlaceholder())->getHandle<int8_t>();
  auto IH = bindings_.get(indices->getPlaceholder())->getHandle<sdim_t>();

  EXPECT_EQ(VH.at({0, 0, 0}), 8);
  EXPECT_EQ(IH.at({0, 0, 0}), 3);
  EXPECT_EQ(VH.at({0, 0, 1}), -7);
  EXPECT_EQ(IH.at({0, 0, 1}), 2);
  EXPECT_EQ(VH.at({0, 0, 2}), -12);
  EXPECT_EQ(IH.at({0, 0, 2}), 0);

  EXPECT_EQ(VH.at({1, 0, 0}), 10);
  EXPECT_EQ(IH.at({1, 0, 0}), 1);
  EXPECT_EQ(VH.at({1, 0, 1}), 10);
  EXPECT_EQ(IH.at({1, 0, 1}), 4);
  EXPECT_EQ(VH.at({1, 0, 2}), 3);
  EXPECT_EQ(IH.at({1, 0, 2}), 2);

  EXPECT_EQ(VH.at({2, 0, 0}), 3);
  EXPECT_EQ(IH.at({2, 0, 0}), 1);
  EXPECT_EQ(VH.at({2, 0, 1}), 3);
  EXPECT_EQ(IH.at({2, 0, 1}), 3);
  EXPECT_EQ(VH.at({2, 0, 2}), 3);
  EXPECT_EQ(IH.at({2, 0, 2}), 4);
}

/// Helper for testing Gather with different \p ITy / \p IndexType.
template <typename DataType, typename IndexType>
static void gatherFloatInputTest(glow::PlaceholderBindings &bindings,
                                 glow::Module &mod, glow::Function *F,
                                 glow::ExecutionEngine &EE, ElemKind DTy,
                                 ElemKind ITy) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [
        [0, 1, 0, 1],
        [1, 2, 2, 0],
    ]
    OUTPUT = [
        [
            [1.0, 1.2],
            [2.3, 3.4],
            [1.0, 1.2],
            [2.3, 3.4],
        ],
        [
            [2.3, 3.4],
            [4.5, 5.7],
            [4.5, 5.7],
            [1.0, 1.2],
        ],
    ]
  */
  auto *data = mod.createPlaceholder(DTy, {3, 2}, "data", false);
  auto *indices = mod.createPlaceholder(ITy, {2, 4}, "indices", false);

  bindings.allocate(data)->getHandle<DataType>() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };
  bindings.allocate(indices)->getHandle<IndexType>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto *R = F->createGather("gather", data, indices);

  auto *result = F->createSave("save", R);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor *resultT = bindings.get(result->getPlaceholder());
  Tensor expectedT(DTy, {2, 4, 2});
  expectedT.getHandle<DataType>() = {1.0, 1.2, 2.3, 3.4, 1.0, 1.2, 2.3, 3.4,
                                     2.3, 3.4, 4.5, 5.7, 4.5, 5.7, 1.0, 1.2};

  EXPECT_TRUE(resultT->isEqual(expectedT));
}

/// Test that Gather works with Float data and Int32 indices.
TEST_P(OperatorTest, GatherDataFloatIdxInt32) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<float, int32_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::FloatTy, ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test that Gather works with Float data and Int64 indices.
TEST_P(OperatorTest, GatherDataFloatIdxInt64) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<float, int64_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::FloatTy, ElemKind::Int64ITy);
}
#endif

/// Test that Gather works with Float16 data and Int32 indices.
TEST_P(OperatorTest, GatherDataFloat16IdxInt32) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<float16_t, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, ElemKind::Int32ITy);
}

/// Test that Gather works with Float16 data and Int64 indices.
TEST_P(OperatorTest, GatherDataFloat16IdxInt64) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<float16_t, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, ElemKind::Int64ITy);
}

/// Helper for testing Gather with different \p ITy / \p IndexType.
template <typename IndexType>
static void gatherInt8InputTest(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind ITy) {
  /*
    DATA  = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    INDICES = [
        [0, 1, 0, 1],
        [1, 2, 2, 0],
    ]
    OUTPUT = [
        [
            [1, 2],
            [3, 4],
            [1, 2],
            [3, 4],
        ],
        [
            [3, 4],
            [5, 6],
            [5, 6],
            [1, 2],
        ],
    ]
  */
  auto *data =
      mod.createPlaceholder(ElemKind::Int8QTy, {3, 2}, 1.0, 0, "data", false);
  auto *indices = mod.createPlaceholder(ITy, {2, 4}, "indices", false);

  bindings.allocate(data)->getHandle<int8_t>() = {
      1, 2, 3, 4, 5, 6,
  };
  bindings.allocate(indices)->getHandle<IndexType>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto *R = F->createGather("gather", data, indices);

  auto *result = F->createSave("save", R);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor *resultT = bindings.get(result->getPlaceholder());
  Tensor expectedT(ElemKind::Int8QTy, {2, 4, 2}, 1.0, 0);
  expectedT.getHandle<int8_t>() = {1, 2, 3, 4, 1, 2, 3, 4,
                                   3, 4, 5, 6, 5, 6, 1, 2};

  EXPECT_TRUE(resultT->isEqual(expectedT));
}

/// Test that Gather works with Int8 data and Int32 indices.
TEST_P(OperatorTest, GatherDataInt8IdxInt32) {
  CHECK_IF_ENABLED();
  gatherInt8InputTest<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test that Gather works with Int8 data and Int64 indices.
TEST_P(OperatorTest, GatherDataInt8IdxInt64) {
  CHECK_IF_ENABLED();
  gatherInt8InputTest<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}
#endif

/// Helper for testing GatherRanges with different \p ITy / \p IndexType.
template <typename DataType, typename IndexType>
void gatherRangesTest(glow::PlaceholderBindings &bindings_, glow::Module &mod_,
                      glow::Function *F_, glow::ExecutionEngine &EE_,
                      ElemKind DTy, ElemKind ITy) {
  /*
    DATA  = [1, 2, 3, 4, 5, 6]
    RANGES = [
      [
        [0, 1],
        [2, 2],
      ],
      [
        [4, 1],
        [5, 1],
      ]
    ]
    OUTPUT = [1, 3, 4, 5, 6]
    LENGTHS = [3, 2]
  */
  auto *data = createPlaceholderConditionallyQuantized(mod_, DTy, {6}, "data",
                                                       false, "N");
  auto *ranges = mod_.createPlaceholder(ITy, {2, 2, 2}, "ranges", false);

  bindings_.allocate(data)->getHandle<DataType>() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(ranges)->getHandle<IndexType>() = {0, 1, 2, 2, 4, 1, 5, 1};

  auto *R =
      F_->createGatherRanges("gatherranges", data, ranges, /*maxOutputSize=*/5);

  auto *output = F_->createSave("output", R->getOutput());
  auto *lengths = F_->createSave("lengths", R->getLengths());

  Tensor *outputT = bindings_.allocate(output->getPlaceholder());
  Tensor *lengthsT = bindings_.allocate(lengths->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto expectedOutputT = createTensorConditionallyQuantized(DTy, {5});
  expectedOutputT.getHandle<DataType>() = {1, 3, 4, 5, 6};
  EXPECT_TRUE(outputT->isEqual(expectedOutputT));

  Tensor expectedLengthsT(ITy, {2});
  expectedLengthsT.getHandle<IndexType>() = {3, 2};
  EXPECT_TRUE(lengthsT->isEqual(expectedLengthsT));
}

/// Test GatherRanges with Int64 data and Int32 indices.
TEST_P(OperatorTest, GatherRangesDataInt64IdxInt32) {
  CHECK_IF_ENABLED();
  gatherRangesTest<int64_t, int32_t>(bindings_, mod_, F_, EE_,
                                     ElemKind::Int64ITy, ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test GatherRanges with Int64 data and Int64 indices.
TEST_P(OperatorTest, GatherRangesDataInt64IdxInt64) {
  CHECK_IF_ENABLED();
  gatherRangesTest<int64_t, int64_t>(bindings_, mod_, F_, EE_,
                                     ElemKind::Int64ITy, ElemKind::Int64ITy);
}
#endif

/// Test GatherRanges with Float data and Int32 indices.
TEST_P(OperatorTest, GatherRangesDataFloatIdxInt32) {
  CHECK_IF_ENABLED();
  gatherRangesTest<float, int32_t>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                                   ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test GatherRanges with Float data and Int64 indices.
TEST_P(OperatorTest, GatherRangesDataFloatIdxInt64) {
  CHECK_IF_ENABLED();
  gatherRangesTest<float, int64_t>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                                   ElemKind::Int64ITy);
}
#endif

/// Test GatherRanges with Float16 data and Int32 indices.
TEST_P(OperatorTest, GatherRangesDataFloat16IdxInt32) {
  CHECK_IF_ENABLED();
  gatherRangesTest<float16_t, int32_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::Float16Ty, ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test GatherRanges with Float16 data and Int64 indices.
TEST_P(OperatorTest, GatherRangesDataFloat16IdxInt64) {
  CHECK_IF_ENABLED();
  gatherRangesTest<float16_t, int64_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::Float16Ty, ElemKind::Int64ITy);
}
#endif

/// Test GatherRanges with Int8Q data and Int32 indices.
TEST_P(OperatorTest, GatherRangesDataInt8QIdxInt32) {
  CHECK_IF_ENABLED();
  gatherRangesTest<int8_t, int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy,
                                    ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test GatherRanges with Int8Q data and Int64 indices.
TEST_P(OperatorTest, GatherRangesDataInt8QIdxInt64) {
  CHECK_IF_ENABLED();
  gatherRangesTest<int8_t, int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy,
                                    ElemKind::Int64ITy);
}
#endif

/// Check if the code generation of transposes
/// is correct for tensors with 2 dimensions.
/// Note: This assumes that Tensor::transpose is correct.
TEST_P(OperatorTest, Transpose2Dims) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {20, 13}, "A", false);
  bindings_.allocate(A)->getHandle().randomize(-3.0, 3.0, mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor dest(ElemKind::FloatTy, {13, 20});
  bindings_.get(A)->transpose(&dest, {1, 0});
  EXPECT_TRUE(bindings_.get(result->getPlaceholder())->isEqual(dest));
}

/// Check that transpose is supported for FP16.
TEST_P(OperatorTest, FP16Transpose2Dims) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::Float16Ty, {20, 13}, "A", false);
  bindings_.allocate(A)->getHandle<float16_t>().randomize(-3.0, 3.0,
                                                          mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor dest(ElemKind::Float16Ty, {13, 20});
  bindings_.get(A)->transpose(&dest, {1, 0});
  EXPECT_TRUE(bindings_.get(result->getPlaceholder())->isEqual(dest));
}

/// Check that transpose is supported for BoolTy.
TEST_P(OperatorTest, BoolTranspose2Dims) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::BoolTy, {20, 13}, "A", false);
  bindings_.allocate(A)->getHandle<bool>().randomize(0, 1, mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor dest(ElemKind::BoolTy, {13, 20});
  bindings_.get(A)->transpose(&dest, {1, 0});
  EXPECT_TRUE(bindings_.get(result->getPlaceholder())->isEqual(dest));
}

/// Helper to check if the code generation of transposes
/// is correct for tensors with 3 dimensions using \p DTy.
/// Note: This assumes that Tensor::transpose is correct.
template <typename DataType>
static void testTranspose3Dims(glow::PlaceholderBindings &bindings,
                               glow::Module &mod, glow::Function *F,
                               glow::ExecutionEngine &EE, ElemKind DTy) {
  constexpr dim_t dims[] = {20, 13, 7};
  auto *A = createPlaceholderConditionallyQuantized(mod, DTy, dims, "A", false);
  bindings.allocate(A)->getHandle<DataType>().randomize(-3.0, 3.0,
                                                        mod.getPRNG());

  int nbOfShuffle = 0;
  SaveNode *savedTransposes[6];
  unsigned_t shuffles[6][3];

  for (unsigned_t i = 0; i < 3; ++i) {
    for (unsigned_t j = 0; j < 3; ++j) {
      if (j == i) {
        continue;
      }
      for (unsigned_t k = 0; k < 3; ++k) {
        if (k == j || k == i) {
          continue;
        }
        shuffles[nbOfShuffle][0] = i;
        shuffles[nbOfShuffle][1] = j;
        shuffles[nbOfShuffle][2] = k;
        auto *tr = F->createTranspose("tr", A, shuffles[nbOfShuffle]);
        savedTransposes[nbOfShuffle] = F->createSave("saveTranspose", tr);
        bindings.allocate(savedTransposes[nbOfShuffle]->getPlaceholder());
        ++nbOfShuffle;
      }
    }
  }

  // We should have exactly 6 possible permutations for 3 dimensions.
  EXPECT_EQ(6, nbOfShuffle);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  for (int i = 0; i < 6; ++i) {
    auto dest = createTensorConditionallyQuantized(
        DTy,
        {dims[shuffles[i][0]], dims[shuffles[i][1]], dims[shuffles[i][2]]});
    bindings.get(A)->transpose(&dest, shuffles[i]);
    EXPECT_TRUE(
        bindings.get(savedTransposes[i]->getPlaceholder())->isEqual(dest));
  }
}

/// Test Transpose3Dims with Float.
TEST_P(OperatorTest, Transpose3Dims_Float) {
  CHECK_IF_ENABLED();
  testTranspose3Dims<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test Transpose3Dims with Float16.
TEST_P(OperatorTest, Transpose3Dims_Float16) {
  CHECK_IF_ENABLED();
  testTranspose3Dims<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test Transpose3Dims with Int8.
TEST_P(OperatorTest, Transpose3Dims_Int8) {
  CHECK_IF_ENABLED();
  testTranspose3Dims<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Test that Transpose optimization into Reshape yields expected results.
TEST_P(OperatorTest, TransposeIntoReshapeOptim) {
  CHECK_IF_ENABLED();
  auto *batch = mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 2, 4}, "batch",
                                       false, "NHWC");
  auto IH = bindings_.allocate(batch)->getHandle();
  for (size_t i = 0; i < 24; i++) {
    IH.raw(i) = i + 1;
  }

  Node *T = F_->createTranspose("transpose", batch, {1, 2, 0, 3}, "HWNC");
  Node *R = F_->createBatchedReduceMean("reduce.mean", T, {2, 3});
  SaveNode *O = F_->createSave("ret", R);
  bindings_.allocate(mod_.getPlaceholders());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(O->getPlaceholder())->getHandle();
  std::vector<dim_t> expectedDims = {3, 2};
  EXPECT_TRUE(result.dims().vec() == expectedDims);

  std::vector<float> expectedValues = {2.5f, 6.5f, 10.5f, 14.5f, 18.5f, 22.5f};
  for (size_t i = 0; i < 3 * 2; i++) {
    EXPECT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Check that gather on Int64ITy/size_t works.
TEST_P(OperatorTest, GatherSizeT) {
  CHECK_IF_ENABLED();

  /*
    DATA  = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    INDICES = [
        [0, 1, 0, 1],
        [1, 2, 2, 0],
    ]
    OUTPUT = [
        [
            [1, 2],
            [3, 4],
            [1, 2],
            [3, 4],
        ],
        [
            [3, 4],
            [5, 6],
            [5, 6],
            [1, 2],
        ],
    ]
  */
  auto *data =
      mod_.createPlaceholder(ElemKind::Int64ITy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {2, 4}, "indices", false);

  bindings_.allocate(data)->getHandle<int64_t>() = {
      1, 2, 3, 4, 5, 6,
  };
  bindings_.allocate(indices)->getHandle<sdim_t>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto *R = F_->createGather("gather", data, indices);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle<int64_t>();

  EXPECT_EQ(H.at({0, 0, 0}), 1);
  EXPECT_EQ(H.at({0, 0, 1}), 2);
  EXPECT_EQ(H.at({0, 1, 0}), 3);
  EXPECT_EQ(H.at({0, 1, 1}), 4);
  EXPECT_EQ(H.at({0, 2, 0}), 1);
  EXPECT_EQ(H.at({0, 2, 1}), 2);
  EXPECT_EQ(H.at({0, 3, 0}), 3);
  EXPECT_EQ(H.at({0, 3, 1}), 4);

  EXPECT_EQ(H.at({1, 0, 0}), 3);
  EXPECT_EQ(H.at({1, 0, 1}), 4);
  EXPECT_EQ(H.at({1, 1, 0}), 5);
  EXPECT_EQ(H.at({1, 1, 1}), 6);
  EXPECT_EQ(H.at({1, 2, 0}), 5);
  EXPECT_EQ(H.at({1, 2, 1}), 6);
  EXPECT_EQ(H.at({1, 3, 0}), 1);
  EXPECT_EQ(H.at({1, 3, 1}), 2);
}

TEST_P(OperatorTest, BatchedGather) {
  CHECK_IF_ENABLED();

  /*
   DATA  = [
    [1.0, 1.2, 2.4, 4.5],
    [2.3, 3.4, 3.6, 2.3],
    [4.5, 5.7, 1.2, 4.5],
   ]

   INDICES = [0, 2],

   OUTPUT = [
    [1.0, 2.4],
    [2.3, 3.6],
    [4.5, 1.2],
   ]
   */
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 4}, "data", false);
  auto *indices = mod_.createPlaceholder(IndexElemKind, {2}, "indices", false);

  bindings_.allocate(data)->getHandle() = {
      1.0f, 1.2f, 2.4f, 4.5f, 2.3f, 3.4f, 3.6f, 2.3f, 4.5f, 5.7f, 1.2f, 4.5f,
  };
  bindings_.allocate(indices)->getHandle<sdim_t>() = {
      0,
      2,
  };

  // Create a batched gather (a single batch dimension).
  auto *R = F_->createGather("gather", data, indices, 1);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_FLOAT_EQ(H.at({0, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 1}), 2.4);
  EXPECT_FLOAT_EQ(H.at({1, 0}), 2.3);
  EXPECT_FLOAT_EQ(H.at({1, 1}), 3.6);
  EXPECT_FLOAT_EQ(H.at({2, 0}), 4.5);
  EXPECT_FLOAT_EQ(H.at({2, 1}), 1.2);
}

TEST_P(OperatorTest, ScatterData) {
  CHECK_IF_ENABLED();

  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {5, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {2, 1}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "slices", false);

  bindings_.allocate(data)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  bindings_.allocate(indices)->getHandle<sdim_t>() = {1, 3};
  bindings_.allocate(slices)->getHandle() = {-3, -4, -7, -8};

  auto *R = F_->createScatterData("scatterdata", data, indices, slices);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle();

  EXPECT_FLOAT_EQ(H.at({0, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 1}), 2.0);
  EXPECT_FLOAT_EQ(H.at({1, 0}), -3.0);
  EXPECT_FLOAT_EQ(H.at({1, 1}), -4.0);
  EXPECT_FLOAT_EQ(H.at({2, 0}), 5.0);
  EXPECT_FLOAT_EQ(H.at({2, 1}), 6.0);
  EXPECT_FLOAT_EQ(H.at({3, 0}), -7.0);
  EXPECT_FLOAT_EQ(H.at({3, 1}), -8.0);
  EXPECT_FLOAT_EQ(H.at({4, 0}), 9.0);
  EXPECT_FLOAT_EQ(H.at({4, 1}), 10.0);
}

TEST_P(OperatorTest, ScatterDataQuantized) {
  CHECK_IF_ENABLED();

  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {5, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {2, 1}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "slices", false);

  bindings_.allocate(data)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  bindings_.allocate(indices)->getHandle<sdim_t>() = {1, 3};
  bindings_.allocate(slices)->getHandle() = {-3, -4, -7, -8};

  auto qParams = glow::quantization::chooseQuantizationParams(-11, 11);
  auto dataTy =
      mod_.uniqueType(ElemKind::Int8QTy, {5, 2}, qParams.scale, qParams.offset);
  auto slicesTy =
      mod_.uniqueType(ElemKind::Int8QTy, {2, 2}, qParams.scale, qParams.offset);

  auto *dataQ = F_->createQuantize("quantizeQ", data, dataTy);
  auto *slicesQ = F_->createQuantize("quantizeS", slices, slicesTy);
  auto *SA = F_->createScatterData("scatterdata", dataQ, indices, slicesQ);
  auto *DQ = F_->createDequantize("dequantize", SA);

  auto *result = F_->createSave("save", DQ);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle();

  EXPECT_NEAR(H.at({0, 0}), 1.0, 0.05);
  EXPECT_NEAR(H.at({0, 1}), 2.0, 0.05);
  EXPECT_NEAR(H.at({1, 0}), -3.0, 0.05);
  EXPECT_NEAR(H.at({1, 1}), -4.0, 0.05);
  EXPECT_NEAR(H.at({2, 0}), 5.0, 0.05);
  EXPECT_NEAR(H.at({2, 1}), 6.0, 0.05);
  EXPECT_NEAR(H.at({3, 0}), -7.0, 0.05);
  EXPECT_NEAR(H.at({3, 1}), -8.0, 0.05);
  EXPECT_NEAR(H.at({4, 0}), 9.0, 0.05);
  EXPECT_NEAR(H.at({4, 1}), 10.0, 0.05);
}

TEST_P(OperatorTest, ScatterDataNDimensionalSimple) {
  CHECK_IF_ENABLED();

  // Data = {{1,2},{3,4},{5,6}}
  // Slices = {-3,-4}
  // Indices = {{1,0},{1,1}}
  // Result = {{1,2},{-3,-4},{5,6}}
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {2, 2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "slices", false);

  // Fill tensor with consecutive data.
  std::vector<float> init(6);
  std::iota(init.begin(), init.end(), 1);
  bindings_.allocate(data)->getHandle() = init;
  bindings_.allocate(indices)->getHandle<sdim_t>() = {1, 0, 1, 1};
  bindings_.allocate(slices)->getHandle() = {-3., -4.};
  auto *R = F_->createScatterData("scatterdata", data, indices, slices);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<dim_t> expectedDims = {3, 2};
  std::vector<float> expectedValues = {1., 2., -3., -4., 5., 6.};
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_TRUE(H.dims().vec() == expectedDims);
  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(expectedValues[i], H.raw(i));
  }
}

TEST_P(OperatorTest, ScatterDataNDimensional) {
  CHECK_IF_ENABLED();

  // In tensor 2x4x4x3, make two updates with 2-dimensional slices by
  // 2-dimensional indices:
  // 1. By index [0, 3], set [[-1.,  -2.,  -3.]
  //                          [-4.,  -5.,  -6.]
  //                          [-7.,  -8.,  -9.]
  //                          [-10., -11., -12.]];
  //
  // 2. By index [1, 1], set [[-13., -14., -15.]
  //                          [-16., -17., -18.]
  //                          [-19., -20., -21.]
  //                          [-22., -23., -24.]];
  //
  auto *data =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4, 4, 3}, "data", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {2, 2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4, 3}, "slices", false);

  // Fill tensor with consecutive data.
  std::vector<float> init(2 * 4 * 4 * 3);
  std::iota(init.begin(), init.end(), 0);
  bindings_.allocate(data)->getHandle() = init;
  bindings_.allocate(indices)->getHandle<sdim_t>() = {0, 3, 1, 1};
  std::vector<float> initUpdates;
  for (int32_t i = -1; i > -25; i--) {
    initUpdates.push_back(static_cast<float>(i));
  }
  bindings_.allocate(slices)->getHandle() = initUpdates;

  auto *R = F_->createScatterData("scatterdata", data, indices, slices);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<dim_t> expectedDims = {2, 4, 4, 3};
  std::vector<float> expectedValues = {
      0.0f,   1.0f,   2.0f,   3.0f,   4.0f,   5.0f,
      6.0f,   7.0f,   8.0f,   9.0f,   10.0f,  11.0f,

      12.0f,  13.0f,  14.0f,  15.0f,  16.0f,  17.0f,
      18.0f,  19.0f,  20.0f,  21.0f,  22.0f,  23.0f,

      24.0f,  25.0f,  26.0f,  27.0f,  28.0f,  29.0f,
      30.0f,  31.0f,  32.0f,  33.0f,  34.0f,  35.0f,

      -1.0f,  -2.0f,  -3.0f,  -4.0f,  -5.0f,  -6.0f,
      -7.0f,  -8.0f,  -9.0f,  -10.0f, -11.0f, -12.0f,

      48.0f,  49.0f,  50.0f,  51.0f,  52.0f,  53.0f,
      54.0f,  55.0f,  56.0f,  57.0f,  58.0f,  59.0f,

      -13.0f, -14.0f, -15.0f, -16.0f, -17.0f, -18.0f,
      -19.0f, -20.0f, -21.0f, -22.0f, -23.0f, -24.0f,

      72.0f,  73.0f,  74.0f,  75.0f,  76.0f,  77.0f,
      78.0f,  79.0f,  80.0f,  81.0f,  82.0f,  83.0f,

      84.0f,  85.0f,  86.0f,  87.0f,  88.0f,  89.0f,
      90.0f,  91.0f,  92.0f,  93.0f,  94.0f,  95.0f};
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_TRUE(H.dims().vec() == expectedDims);
  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(expectedValues[i], H.raw(i));
  }
}

TEST_P(OperatorTest, ScatterAddQuantized) {
  CHECK_IF_ENABLED();

  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {5, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {2, 1}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "slices", false);

  bindings_.allocate(data)->getHandle() = {1, 2, -3, -8, 5, 6, 7, 8, 9, 10};
  bindings_.allocate(indices)->getHandle<sdim_t>() = {1, 3};
  bindings_.allocate(slices)->getHandle() = {3, -8, -7, 8};

  auto qParams = glow::quantization::chooseQuantizationParams(-11, 11);
  auto dataTy =
      mod_.uniqueType(ElemKind::Int8QTy, {5, 2}, qParams.scale, qParams.offset);
  auto slicesTy =
      mod_.uniqueType(ElemKind::Int8QTy, {2, 2}, qParams.scale, qParams.offset);

  auto *dataQ = F_->createQuantize("quantizeQ", data, dataTy);
  auto *slicesQ = F_->createQuantize("quantizeS", slices, slicesTy);
  auto *SA = F_->createScatterData("scatteradd", dataQ, indices, slicesQ,
                                   /*Cumulative*/ true);
  auto *DQ = F_->createDequantize("dequantize", SA);

  auto *result = F_->createSave("save", DQ);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle();

  EXPECT_NEAR(H.at({0, 0}), 1.0, 0.05);
  EXPECT_NEAR(H.at({0, 1}), 2.0, 0.05);
  EXPECT_NEAR(H.at({1, 0}), 0.0, 0.05);
  EXPECT_NEAR(H.at({1, 1}), -11.0, 0.05);
  EXPECT_NEAR(H.at({2, 0}), 5.0, 0.05);
  EXPECT_NEAR(H.at({2, 1}), 6.0, 0.05);
  EXPECT_NEAR(H.at({3, 0}), 0.0, 0.05);
  EXPECT_NEAR(H.at({3, 1}), 11.0, 0.05);
  EXPECT_NEAR(H.at({4, 0}), 9.0, 0.05);
  EXPECT_NEAR(H.at({4, 1}), 10.0, 0.05);
}

TEST_P(OperatorTest, ScatterAddNDimensionalSimple) {
  CHECK_IF_ENABLED();
  // Test that scatter addition works.
  // Data = {{1,2},{3,4},{5,6}}
  // Slices = {-3,-4}
  // Indices = {{1,0},{1,1}}
  // Result = {{1,2},{0,0},{5,6}}
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {2, 2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "slices", false);

  // Fill tensor with consecutive data.
  std::vector<float> init;
  for (int32_t i = 1; i < 7; i++) {
    init.push_back(static_cast<float>(i));
  }
  bindings_.allocate(data)->getHandle() = init;
  bindings_.allocate(indices)->getHandle<sdim_t>() = {1, 0, 1, 1};
  bindings_.allocate(slices)->getHandle() = {-3., -4.};
  auto *R = F_->createScatterData("scatteradd", data, indices, slices,
                                  /*Cumulative*/ true);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<dim_t> expectedDims = {3, 2};
  std::vector<float> expectedValues = {1., 2., 0., 0., 5., 6.};
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_TRUE(H.dims().vec() == expectedDims);
  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(expectedValues[i], H.raw(i));
  }
}

TEST_P(OperatorTest, ScatterAddNDimensionalDuplicatingIndices) {
  CHECK_IF_ENABLED();
  // Test that scatter addition with duplicating indices works.
  // Data = {{1,2},{3,4},{5,6}}
  // Slices = {-3,-4,-3,-4}
  // Indices = {{1,0},{1,1}{1,0},{1,1}}
  // Result = {{1,2},{-3,-4},{5,6}}
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {4, 2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {4}, "slices", false);

  // Fill tensor with consecutive data.
  std::vector<float> init;
  for (int32_t i = 1; i < 7; i++) {
    init.push_back(static_cast<float>(i));
  }
  bindings_.allocate(data)->getHandle() = init;
  bindings_.allocate(indices)->getHandle<sdim_t>() = {1, 0, 1, 1, 1, 0, 1, 1};
  bindings_.allocate(slices)->getHandle() = {-3., -4., -3., -4.};
  auto *R = F_->createScatterData("scatteradd", data, indices, slices,
                                  /*Cumulative*/ true);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<dim_t> expectedDims = {3, 2};
  std::vector<float> expectedValues = {1., 2., -3., -4., 5., 6.};
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_TRUE(H.dims().vec() == expectedDims);
  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(expectedValues[i], H.raw(i));
  }
}

#define COMPARE_ARITH_FUN(_OP_NAME_)                                           \
  static FunctionTensorPair createAndInitBasic##_OP_NAME_##Test(               \
      glow::PlaceholderBindings &bindings, glow::ExecutionEngine &EE) {        \
    auto &mod = EE.getModule();                                                \
    Function *F = mod.createFunction("main");                                  \
                                                                               \
    auto *A = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "A", false);    \
    auto *B = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "B", false);    \
    bindings.allocate(A)->getHandle() = {1.0f, -1.2f, 0.5f, -1.3f};            \
    bindings.allocate(B)->getHandle() = {1.8f, -0.2f, -2.4f, 2.7f};            \
                                                                               \
    auto *add = F->create##_OP_NAME_("arith", A, B);                           \
    auto *result = F->createSave("save", add);                                 \
    auto *resultTensor = bindings.allocate(result->getPlaceholder());          \
                                                                               \
    return std::make_pair(F, resultTensor);                                    \
  }
COMPARE_ARITH_FUN(Add)
COMPARE_ARITH_FUN(Sub)
COMPARE_ARITH_FUN(Mul)
COMPARE_ARITH_FUN(Div)
COMPARE_ARITH_FUN(Max)
COMPARE_ARITH_FUN(Min)
#undef COMPARE_ARITH_FUN

#define COMPARE_ARITH_FLOAT_VS_INT8(_OP_NAME_)                                 \
  TEST_P(OperatorStatelessTest, Basic##_OP_NAME_##NetFloatVsInt8) {            \
    CHECK_IF_ENABLED();                                                        \
    compareAgainstInterpreter(                                                 \
        getBackendName(), createAndInitBasic##_OP_NAME_##Test,                 \
        ElemKind::FloatTy, ElemKind::Int8QTy, 0.035f, parCloneCountOpt);       \
  }
COMPARE_ARITH_FLOAT_VS_INT8(Add)
COMPARE_ARITH_FLOAT_VS_INT8(Sub)
COMPARE_ARITH_FLOAT_VS_INT8(Mul)
COMPARE_ARITH_FLOAT_VS_INT8(Div)
COMPARE_ARITH_FLOAT_VS_INT8(Max)
COMPARE_ARITH_FLOAT_VS_INT8(Min)
#undef COMPARE_ARITH_FLOAT_VS_INT8

#define COMPARE_ARITH_FLOAT_VS_FLOAT16(_OP_NAME_)                              \
  TEST_P(OperatorStatelessTest, Basic##_OP_NAME_##NetFloatVsFloat16) {         \
    CHECK_IF_ENABLED();                                                        \
    compareAgainstInterpreter(                                                 \
        getBackendName(), createAndInitBasic##_OP_NAME_##Test,                 \
        ElemKind::FloatTy, ElemKind::Float16Ty, 0.01f, parCloneCountOpt);      \
  }
COMPARE_ARITH_FLOAT_VS_FLOAT16(Add)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Sub)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Mul)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Div)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Max)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Min)
#undef COMPARE_ARITH_FLOAT_VS_FLOAT16

#define ARITH_FUN_IMPL(_OP_NAME_, _REFERENCE_FUNCTION_, _PARENTHESES_)         \
  template <typename DataType>                                                 \
  static void testArithmetic##_OP_NAME_##Impl(                                 \
      glow::PlaceholderBindings &bindings, glow::Module &mod,                  \
      glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy) {            \
    std::vector<DataType> data1 = {3, 17, 7, 23};                              \
    std::vector<DataType> data2 = {13, 5, 19, 11};                             \
    auto *A = mod.createPlaceholder(DTy, {1, 4}, "A", false);                  \
    auto *B = mod.createPlaceholder(DTy, {1, 4}, "B", false);                  \
    bindings.allocate(A)->getHandle<DataType>() = data1;                       \
    bindings.allocate(B)->getHandle<DataType>() = data2;                       \
                                                                               \
    auto *add = F->create##_OP_NAME_("arith", A, B);                           \
    auto *result = F->createSave("save", add);                                 \
    auto *resultTensor = bindings.allocate(result->getPlaceholder());          \
                                                                               \
    EE.compile(CompilationMode::Infer);                                        \
    EE.run(bindings);                                                          \
    std::vector<DataType> reference;                                           \
    assert(data1.size() == data2.size() && "Size mismatch!");                  \
    for (size_t i = 0; i < data1.size(); i++) {                                \
      reference.push_back(                                                     \
          _REFERENCE_FUNCTION_<DataType> _PARENTHESES_(data1[i], data2[i]));   \
    }                                                                          \
    auto RH = resultTensor->getHandle<DataType>();                             \
    EXPECT_EQ(reference.size(), RH.size());                                    \
    for (size_t i = 0; i < reference.size(); i++) {                            \
      EXPECT_EQ(reference[i], RH.raw(i));                                      \
    }                                                                          \
  }

#define ARITH_FUNC_TEST_TYPED(_OP_NAME_, _DATA_TYPE_, _ELEM_KIND_)             \
  TEST_P(OperatorTest, Arith##_OP_NAME_##_##_DATA_TYPE_) {                     \
    CHECK_IF_ENABLED();                                                        \
    testArithmetic##_OP_NAME_##Impl<_DATA_TYPE_>(bindings_, mod_, F_, EE_,     \
                                                 _ELEM_KIND_);                 \
  }

#define ARITH_FUNC_TEST(_OP_NAME_, _REFERENCE_FUNCTION_, _PARENTHESES_)        \
  ARITH_FUN_IMPL(_OP_NAME_, _REFERENCE_FUNCTION_, _PARENTHESES_)               \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, int32_t, ElemKind::Int32ITy)                \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, int64_t, ElemKind::Int64ITy)                \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, float, ElemKind::FloatTy)                   \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, float16_t, ElemKind::Float16Ty)

ARITH_FUNC_TEST(Add, std::plus, ())
ARITH_FUNC_TEST(Sub, std::minus, ())
ARITH_FUNC_TEST(Mul, std::multiplies, ())
ARITH_FUNC_TEST(Max, std::max, )
ARITH_FUNC_TEST(Min, std::min, )
#undef ARITH_FUN_IMPL
#undef ARITH_FUNC_TEST_TYPED
#undef ARITH_FUNC_TEST

TEST_P(OperatorTest, IntMatMul) {
  CHECK_IF_ENABLED();

  // The scaling factor 1.4x was carefully selected to make sure we don't
  // overflow or underflow the calculation.
  TypeRef resTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.60, 4);
  TypeRef lhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, -2);
  TypeRef rhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, 2);

  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", false);

  bindings_.allocate(lhs)->getHandle() = {
      1.0, 2.0, 3.0, 4.0, 5.0, -5.0, -4.0, -3.0, 9.0,
  };

  bindings_.allocate(rhs)->getHandle() = {
      0.1f, -0.2f, 0.3f, 9.0f, -8.0f, 7.0f, 6.0f, 5.0f, 9.0f,
  };

  auto *lhsq = F_->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F_->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F_->createMatMul("matmul.q", resTy, lhsq, rhsq);

  auto *rq = F_->createDequantize("dequant", matmulq);

  auto *result = F_->createSave("save", rq);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  /*
   Test the following matrix multiplication:
   A = [[1.0, 2.0, 3.0], [4.0, 5.0, -5.0], [-4.0, -3.0, 9.0]]
   B = [[0.1, -0.2, 0.3], [9.0, -8.0, 7.0], [6.0, 5.0, 9.0]]
   A x B = [36.1,  -1.2,  41.3], [15.4, -65.8, -8.8], [26.6, 69.8,  58.8]]
   */

  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 36.1, 1.0);
  EXPECT_NEAR(H.at({0, 1}), -1.2, 1.0);
  EXPECT_NEAR(H.at({0, 2}), 41.3, 1.0);
  EXPECT_NEAR(H.at({1, 0}), 15.4, 1.0);
  EXPECT_NEAR(H.at({1, 1}), -65.8, 1.0);
  EXPECT_NEAR(H.at({1, 2}), -8.8, 1.0);
  EXPECT_NEAR(H.at({2, 0}), 26.6, 1.0);
  EXPECT_NEAR(H.at({2, 1}), 69.8, 1.0);
  EXPECT_NEAR(H.at({2, 2}), 58.8, 1.0);
}

TEST_P(OperatorTest, IntBatchedArith) {
  CHECK_IF_ENABLED();

  TypeRef resTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.10, 1.0);
  TypeRef lhsTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.11, 4.0);
  TypeRef rhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.14, -2.0);

  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3}, "lhs", false);
  bindings_.allocate(lhs);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", false);
  bindings_.allocate(rhs);

  bindings_.get(lhs)->getHandle() = {
      8.7f, 6.5f, 4.3f, 2.1f, 1.0f, -5.1f, -4.0f, -12.0f, 0.2f,
  };

  bindings_.get(rhs)->getHandle() = {
      -9.1f, -0.4f, 1.3f, 2.2f, -8.1f, 7.6f, -6.4f, 10.0f, 9.1f,
  };

  auto *lhsq = F_->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F_->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F_->createBatchedAdd("add", resTy, lhsq, rhsq);

  auto *rq = F_->createDequantize("dequant", matmulq);

  auto *result = F_->createSave("save", rq);
  bindings_.allocate(result->getPlaceholder());
  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  // A = [8.7, 6.5, 4.3, 2.1, 1.0, -5.1, -4.0, -12.0, 0.2]
  // B = [-9.1, -0.4, 1.3, 2.2, -8.1, 7.6, -6.4, 10.0, 9.1]
  // A + B = [-0.4, 6.1, 5.6, 4.3, -7.1, 2.5, -10.4, -2. , 9.3]
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  constexpr float allowedError = 0.105;
  EXPECT_NEAR(H.at({0, 0, 0}), -0.4, allowedError);
  EXPECT_NEAR(H.at({0, 0, 1}), 6.1, allowedError);
  EXPECT_NEAR(H.at({0, 0, 2}), 5.6, allowedError);
  EXPECT_NEAR(H.at({0, 1, 0}), 4.3, allowedError);
  EXPECT_NEAR(H.at({0, 1, 1}), -7.1, allowedError);
  EXPECT_NEAR(H.at({0, 1, 2}), 2.5, allowedError);
  EXPECT_NEAR(H.at({0, 2, 0}), -10.4, allowedError);
  EXPECT_NEAR(H.at({0, 2, 1}), -2, allowedError);
  EXPECT_NEAR(H.at({0, 2, 2}), 9.3, allowedError);
}

TEST_P(OperatorTest, convTest) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  IH = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  FH = {0, 0, 0, 1, 1, 1, 0, 0, 0};

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 3, 1});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 3, 1, 1, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());

  Tensor expected(outTy);
  expected.getHandle() = {2, 3, 2, 2, 3, 2, 2, 3, 2};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, convTest_Float16) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle<float16_t>();
  IH = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};

  auto filter = mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1},
                                       "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle<float16_t>();
  FH = {0.25, 0.5, 0.25, 1, 1, 1, 0.25, 0.5, 0.25};

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::Float16Ty, {1, 3, 3, 1});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 3, 1, 1, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float16_t>();

  Tensor expected(outTy);
  auto expectedH = expected.getHandle<float16_t>();
  expectedH = {3.375, 5.102, 3.676, 5.051, 7.5, 5.449, 4.574, 6.898, 4.875};

  for (dim_t x = 0; x < 3; x++) {
    for (dim_t y = 0; y < 3; y++) {
      EXPECT_NEAR(result.at({0, x, y, 0}), expectedH.at({0, x, y, 0}), 0.001);
    }
  }
}

template <size_t convDepth>
static FunctionTensorPair
createAndInitConvDepthTest(glow::PlaceholderBindings &bindings,
                           glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);
  auto *conv = F->createConv(bindings, "conv", input, convDepth, 5, 1, 0, 1);
  auto *bias = llvm::cast<Placeholder>(conv->getBias().getNode());

  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.get(bias)->getHandle().randomize(-2.0, 2.0, mod.getPRNG());

  auto *res = F->createSave("save", conv);
  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, Int8ConvolutionDepth10) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<10>,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.045f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, Int16ConvolutionDepth10) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<10>,
                            ElemKind::FloatTy, ElemKind::Int16QTy, 0.03f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, Int8ConvolutionDepth8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<8>,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.03f,
                            parCloneCountOpt);
}
TEST_P(OperatorStatelessTest, Int16ConvolutionDepth8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<8>,
                            ElemKind::FloatTy, ElemKind::Int16QTy, 0.03f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, FP16ConvolutionDepth10) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<10>,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.015f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, FP16ConvolutionDepth8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<8>,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.015f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, ConvolutionDepth10_Int8_BiasInt8) {
  ENABLED_BACKENDS(Interpreter, CPU);
  compareAgainstInterpreter(
      getBackendName(), createAndInitConvDepthTest<10>, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.03f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int8QTy);
}

TEST_P(OperatorStatelessTest, ConvolutionDepth10_Int8_BiasInt32) {
  ENABLED_BACKENDS(Interpreter, CPU);
  compareAgainstInterpreter(
      getBackendName(), createAndInitConvDepthTest<10>, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.03f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int32QTy);
}

TEST_P(OperatorStatelessTest, ConvolutionDepth10_Int16_BiasInt16) {
  ENABLED_BACKENDS(Interpreter);
  compareAgainstInterpreter(
      getBackendName(), createAndInitConvDepthTest<10>, ElemKind::FloatTy,
      ElemKind::Int16QTy, 0.0003f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int16QTy);
}

TEST_P(OperatorStatelessTest, ConvolutionDepth10_Int16_BiasInt32) {
  ENABLED_BACKENDS(Interpreter);
  compareAgainstInterpreter(
      getBackendName(), createAndInitConvDepthTest<10>, ElemKind::FloatTy,
      ElemKind::Int16QTy, 0.0003f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int32QTy);
}

static FunctionTensorPair
createAndInitBasicConcatTest(glow::PlaceholderBindings &bindings,
                             glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "A", false);
  auto *B = mod.createPlaceholder(ElemKind::FloatTy, {2, 3}, "B", false);
  bindings.allocate(A)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(B)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());

  auto *C = F->createConcat("concat", {A, B}, 0);
  auto *res = F->createSave("save", C);
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {A, B, res->getPlaceholder()});

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, IntConcat) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicConcatTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.05f,
                            parCloneCountOpt);
}

TEST_P(OperatorTest, FCWithFlatten) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  Constant *weights = mod_.createConstant(ElemKind::FloatTy, {3, 4}, "weights");
  Constant *bias = mod_.createConstant(ElemKind::FloatTy, {4}, "bias");

  bindings_.allocate(input)->getHandle() = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  weights->getPayloadMutable().getHandle() = {1.0f, 4.0f, 7.0f, 10.0f, //
                                              2.0f, 5.0f, 8.0f, 11.0f, //
                                              3.0f, 6.0f, 9.0f, 12.0f};
  bias->getPayloadMutable().getHandle() = {0.1f, 0.2f, 0.3f, 0.4f};

  auto *FC = F_->createFullyConnected("fc", input, weights, bias);
  auto *S = F_->createSave("save", FC);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();
  std::vector<dim_t> expectedDimensions = {2, 4};
  std::vector<float> expectedValues = {14.1f, 32.2f, 50.3f,  68.4f,
                                       32.1f, 77.2f, 122.3f, 167.4f};
  EXPECT_TRUE(result.dims().vec() == expectedDimensions);
  for (size_t i = 0; i < 2 * 4; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

static FunctionTensorPair
createAndInitBasicFCTest(glow::PlaceholderBindings &bindings,
                         glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);
  auto *fc = F->createFullyConnected(bindings, "FC", input, 30);

  auto *weights = llvm::cast<Placeholder>(fc->getWeights());
  auto *bias = llvm::cast<Placeholder>(fc->getBias());

  bindings.allocate(input)->getHandle().randomize(-0.5, 0.5, mod.getPRNG());
  bindings.get(bias)->getHandle().randomize(0, 0.00001, mod.getPRNG());
  bindings.get(weights)->getHandle().randomize(-0.7, 0.7, mod.getPRNG());

  auto *res = F->createSave("save", fc);
  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, IntFC) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicFCTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.05f,
                            parCloneCountOpt);
}

/// Test FC with Float16.
TEST_P(OperatorStatelessTest, FC_Float16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicFCTest,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.02f,
                            parCloneCountOpt);
}

/// Test Int8 FullyConnected with Int8 bias.
TEST_P(OperatorStatelessTest, FullyConnected_Int8_BiasInt8) {
  ENABLED_BACKENDS(Interpreter, CPU);
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.05f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int8QTy);
}

/// Test Int8 FullyConnected with Int32 bias.
TEST_P(OperatorStatelessTest, FullyConnected_Int8_BiasInt32) {
  ENABLED_BACKENDS(Interpreter, CPU);
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.05f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int32QTy);
}

/// Test Int16 FullyConnected with Int16 bias.
TEST_P(OperatorStatelessTest, FullyConnected_Int16_BiasInt16) {
  ENABLED_BACKENDS(Interpreter);
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicFCTest, ElemKind::FloatTy,
      ElemKind::Int16QTy, 0.0005f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int16QTy);
}

/// Test Int16 FullyConnected with Int32 bias.
TEST_P(OperatorStatelessTest, FullyConnected_Int16_BiasInt32) {
  ENABLED_BACKENDS(Interpreter);
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicFCTest, ElemKind::FloatTy,
      ElemKind::Int16QTy, 0.0005f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int32QTy);
}

TEST_P(OperatorTest, EntropyLossTest) {
  CHECK_IF_ENABLED();

  auto *P = mod_.createPlaceholder(ElemKind::FloatTy, {2, 3}, "P", false);
  auto *Y = mod_.createPlaceholder(IndexElemKind, {2}, "Y", false);

  bindings_.allocate(P)->getHandle() = {0.2f, 0.5f, 0.3f, 0.4f, 0.3f, 0.3f};
  bindings_.allocate(Y)->getHandle<sdim_t>() = {1, 2};
  auto *ceLoss = F_->createCrossEntropyLoss("CELoss", P, Y);
  auto *L = F_->createSave("save", ceLoss);
  bindings_.allocate(L->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto R = bindings_.get(L->getPlaceholder())->getHandle();
  EXPECT_NEAR(R.at({0}), -log(0.5) - log(0.3), 0.1);
}

/// Check that the max operator works properly with FP16.
TEST_P(OperatorTest, FP16Max) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "A", false);
  bindings_.allocate(inputA)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "B", false);
  bindings_.allocate(inputB)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *Max = F_->createMax("max", inputA, inputB);
  auto *S = F_->createSave("save", Max);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float16_t>();
  auto handleA = bindings_.get(inputA)->getHandle<float16_t>();
  auto handleB = bindings_.get(inputB)->getHandle<float16_t>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), std::max(handleA.raw(idx), handleB.raw(idx)));
  }
}

TEST_P(OperatorTest, RescaleNode) {
  CHECK_IF_ENABLED();

  // Check the outputs of the RescaleQuantized operation.
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.4, -3,
                                       "input", false);
  bindings_.allocate(input)->init(Tensor::InitKind::Broadcast, 40,
                                  mod_.getPRNG());

  auto T1 = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.7, 5);
  auto T2 = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.3, -4);
  auto resTy = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.4, -4);

  // Test a sequence of rescale operations that the optimizer may try to
  // optimize at some point.
  auto *X = F_->createRescaleQuantized("R1", input, T1);
  auto *Y = F_->createRescaleQuantized("R2", X, T2);
  auto *Z = F_->createRescaleQuantized("R3", Y, resTy);

  auto *output = F_->createSave("save", Z);
  bindings_.allocate(output->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto RI = bindings_.get(input)->getHandle<int8_t>();
  auto RO = bindings_.get(output->getPlaceholder())->getHandle<int8_t>();

  EXPECT_EQ(RI.raw(0), 40);
  EXPECT_NEAR(RO.raw(0), 40, 1);
}

TEST_P(OperatorTest, QuantizedArithmeticRescaled) {
  CHECK_IF_ENABLED();

  const size_t len = 100;

  // In this test we check the correctness of the quantized Max, Min, Add,
  // Sub, Mul, and Div nodes as well as how they interact with the rescaling
  // node.
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "A", false);
  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "B", false);
  auto *C = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "C", false);

  auto AH = bindings_.allocate(A)->getHandle();
  auto BH = bindings_.allocate(B)->getHandle();
  auto CH = bindings_.allocate(C)->getHandle();

  AH.randomize(-10, 10, mod_.getPRNG());
  BH.randomize(-10, 10, mod_.getPRNG());
  // Below, randomize between 1 and 10 to avoid division by 0 later.
  CH.randomize(1, 10, mod_.getPRNG());

  auto TA = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.2, 0);
  auto TB = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.1, 0);
  auto TC = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.3, 0);

  auto TI1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);
  auto TI2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.8, 0);
  auto TI3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 0);
  auto TI4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TI5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 0);
  auto TI6 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.7, 0);

  auto TO1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TO2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 0);
  auto TO3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);
  auto TO4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 0);
  auto TO5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TO6 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);

  // Quantize input vars and apply max/min/add/sub/mul/div quantized.
  auto *QA = F_->createQuantize("QA", A, TA);
  auto *QB = F_->createQuantize("QB", B, TB);
  auto *QC = F_->createQuantize("QC", C, TC);

  Node *max = F_->createMax("max", TI1, QA, QB);
  Node *min = F_->createMin("min", TI2, QA, QB);
  Node *add = F_->createAdd("add", TI3, QA, QB);
  Node *sub = F_->createSub("sub", TI4, QA, QB);
  Node *mul = F_->createMul("mul", TI5, QA, QB);
  Node *div = F_->createDiv("div", TI6, QB, QC);

  // Rescale quantized results.
  max = F_->createRescaleQuantized("rescaleMax", max, TO1);
  min = F_->createRescaleQuantized("rescaleMin", min, TO2);
  add = F_->createRescaleQuantized("rescaleAdd", add, TO3);
  sub = F_->createRescaleQuantized("rescaleSub", sub, TO4);
  mul = F_->createRescaleQuantized("rescaleMul", mul, TO5);
  div = F_->createRescaleQuantized("rescaleDiv", div, TO6);

  // Dequantize results back to floating-point.
  max = F_->createDequantize("maxDQ", max);
  min = F_->createDequantize("minDQ", min);
  add = F_->createDequantize("addDQ", add);
  sub = F_->createDequantize("subDQ", sub);
  mul = F_->createDequantize("mulDQ", mul);
  div = F_->createDequantize("divDQ", div);

  // Save results of the operations.
  auto *O1 = F_->createSave("saveMax", max);
  auto *O2 = F_->createSave("saveMin", min);
  auto *O3 = F_->createSave("saveAdd", add);
  auto *O4 = F_->createSave("saveSub", sub);
  auto *O5 = F_->createSave("saveMul", mul);
  auto *O6 = F_->createSave("saveDiv", div);

  bindings_.allocate(O1->getPlaceholder());
  bindings_.allocate(O2->getPlaceholder());
  bindings_.allocate(O3->getPlaceholder());
  bindings_.allocate(O4->getPlaceholder());
  bindings_.allocate(O5->getPlaceholder());
  bindings_.allocate(O6->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < len; i++) {
    auto max = std::max(AH.at({i}), BH.at({i}));
    auto min = std::min(AH.at({i}), BH.at({i}));
    auto add = AH.at({i}) + BH.at({i});
    auto sub = AH.at({i}) - BH.at({i});
    auto mul = AH.at({i}) * BH.at({i});
    auto div = BH.at({i}) / CH.at({i});

    // We generate numbers up to 110, so a difference of 2 (~2%) is
    // reasonable.
    EXPECT_NEAR(max, bindings_.get(O1->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(min, bindings_.get(O2->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(add, bindings_.get(O3->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(sub, bindings_.get(O4->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(mul, bindings_.get(O5->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(div, bindings_.get(O6->getPlaceholder())->getHandle().at({i}),
                2.0);
  }
}

static FunctionTensorPair
createAndInitTransposeNet(glow::PlaceholderBindings &bindings,
                          glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {2, 3}, "A", false);
  bindings.allocate(A)->getHandle() = {1, 1.2f, 0.5f, 1.3f, 2.7f, 3.1f};
  auto *tr = F->createTranspose("Tr", A, {1, 0});
  auto *result = F->createSave("Ret", tr);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, QuantizedTranspose) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitTransposeNet,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.0045f,
                            parCloneCountOpt);
}

TEST_P(OperatorTest, QuantizedArithmeticUnrescaled) {
  CHECK_IF_ENABLED();

  const size_t len = 1000;

  // In this test we check the correctness of the quantized Max, Min, Add,
  // Sub, Mul, and Div operations.
  auto TQA = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, -1);
  auto TQB = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 2);
  // For TQC, set offset to -11 to avoid division by 0 later.
  auto TQC = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, -11);
  auto TO1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.4, 3);
  auto TO2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.8, 2);
  auto TO3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.7, 5);
  auto TO4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.3, -7);
  auto TO5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 3);
  auto TO6 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, -2);

  auto *QA = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQA->getScale(),
                                    TQA->getOffset(), "QA", false);
  auto *QB = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQB->getScale(),
                                    TQB->getOffset(), "QB", false);
  auto *QC = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQC->getScale(),
                                    TQC->getOffset(), "QC", false);

  bindings_.allocate(QA)->getHandle<int8_t>().randomize(-10, 10,
                                                        mod_.getPRNG());
  bindings_.allocate(QB)->getHandle<int8_t>().randomize(-10, 10,
                                                        mod_.getPRNG());
  bindings_.allocate(QC)->getHandle<int8_t>().randomize(-10, 10,
                                                        mod_.getPRNG());

  // Apply max/min/add/sub/mul/div quantized.
  Node *max = F_->createMax("max", TO1, QA, QB);
  Node *min = F_->createMin("min", TO2, QA, QB);
  Node *add = F_->createAdd("add", TO3, QA, QB);
  Node *sub = F_->createSub("sub", TO4, QA, QB);
  Node *mul = F_->createMul("mul", TO5, QA, QB);
  Node *div = F_->createDiv("div", TO6, QB, QC);

  // Save results of the operations.
  auto *O1 = F_->createSave("saveMax", max);
  auto *O2 = F_->createSave("saveMin", min);
  auto *O3 = F_->createSave("saveAdd", add);
  auto *O4 = F_->createSave("saveSub", sub);
  auto *O5 = F_->createSave("saveMul", mul);
  auto *O6 = F_->createSave("saveDiv", div);

  bindings_.allocate(O1->getPlaceholder());
  bindings_.allocate(O2->getPlaceholder());
  bindings_.allocate(O3->getPlaceholder());
  bindings_.allocate(O4->getPlaceholder());
  bindings_.allocate(O5->getPlaceholder());
  bindings_.allocate(O6->getPlaceholder());

  auto QAH = bindings_.get(QA)->getHandle<int8_t>();
  auto QBH = bindings_.get(QB)->getHandle<int8_t>();
  auto QCH = bindings_.get(QC)->getHandle<int8_t>();
  auto O1H = bindings_.get(O1->getPlaceholder())->getHandle<int8_t>();
  auto O2H = bindings_.get(O2->getPlaceholder())->getHandle<int8_t>();
  auto O3H = bindings_.get(O3->getPlaceholder())->getHandle<int8_t>();
  auto O4H = bindings_.get(O4->getPlaceholder())->getHandle<int8_t>();
  auto O5H = bindings_.get(O5->getPlaceholder())->getHandle<int8_t>();
  auto O6H = bindings_.get(O6->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < len; i++) {
    float a = TQA->getScale() * (QAH.at({i}) - TQA->getOffset());
    float b = TQB->getScale() * (QBH.at({i}) - TQB->getOffset());
    float c = TQC->getScale() * (QCH.at({i}) - TQC->getOffset());
    float max = std::max(a, b) / TO1->getScale() + TO1->getOffset();
    float min = std::min(a, b) / TO2->getScale() + TO2->getOffset();
    float add = (a + b) / TO3->getScale() + TO3->getOffset();
    float sub = (a - b) / TO4->getScale() + TO4->getOffset();
    float mul = (a * b) / TO5->getScale() + TO5->getOffset();
    float div = (b / c) / TO6->getScale() + TO6->getOffset();

    EXPECT_NEAR(std::round(max), O1H.at({i}), 1.0);
    EXPECT_NEAR(std::round(min), O2H.at({i}), 1.0);
    EXPECT_NEAR(std::round(add), O3H.at({i}), 1.0);
    EXPECT_NEAR(std::round(sub), O4H.at({i}), 1.0);
    EXPECT_NEAR(std::round(mul), O5H.at({i}), 1.0);
    EXPECT_NEAR(std::round(div), O6H.at({i}), 1.0);
  }
}

TEST_P(OperatorTest, QuantizedCmpLTEAndSelect) {
  CHECK_IF_ENABLED();

  // In this test we check the correctness of the quantized
  // less-than-or-equal-to comparison operator.
  const size_t len = 1000;
  auto TQA = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, -3);
  auto TQB = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 5);
  auto TQC = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.8, 3);
  auto TQD = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, -4);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.5, -2);

  auto *QA = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQA->getScale(),
                                    TQA->getOffset(), "QA", false);
  auto *QB = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQB->getScale(),
                                    TQB->getOffset(), "QB", false);
  auto *QC = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQC->getScale(),
                                    TQC->getOffset(), "QC", false);
  auto *QD = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQD->getScale(),
                                    TQD->getOffset(), "QD", false);

  auto QAH = bindings_.allocate(QA)->getHandle<int8_t>();
  auto QBH = bindings_.allocate(QB)->getHandle<int8_t>();
  auto QCH = bindings_.allocate(QC)->getHandle<int8_t>();
  auto QDH = bindings_.allocate(QD)->getHandle<int8_t>();

  QAH.randomize(-128, 127, mod_.getPRNG());
  QBH.randomize(-128, 127, mod_.getPRNG());
  QCH.randomize(-128, 127, mod_.getPRNG());
  QDH.randomize(-128, 127, mod_.getPRNG());

  // Apply comparison and selection quantized.
  Node *cmpLTE = F_->createCmpLTE("cmpLTE", QA, QB);
  Node *select = F_->createSelect("select", OT, cmpLTE, QC, QD);

  // Save result of the operation.
  auto *out = F_->createSave("save", select);
  auto OH = bindings_.allocate(out->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  int count_strict = 0;
  int count = 0;
  for (dim_t i = 0; i < len; i++) {
    float a = TQA->getScale() * (QAH.at({i}) - TQA->getOffset());
    float b = TQB->getScale() * (QBH.at({i}) - TQB->getOffset());
    float c = TQC->getScale() * (QCH.at({i}) - TQC->getOffset());
    float d = TQD->getScale() * (QDH.at({i}) - TQD->getOffset());
    float tmp = (a <= b) ? c : d;
    int32_t q = std::round(tmp / 1.5 - 2);
    int8_t select = quantization::clip<int32_t, int8_t>(q);

    if (OH.at({i}) != select) {
      count_strict++;
      if (std::abs(OH.at({i}) - select) > 1) {
        count++;
      }
    }
  }
  // Require that the number of off-by-1 errors be at most 0.6%.
  EXPECT_LE(count_strict, 6);
  EXPECT_LE(count, 4);
}

TEST_P(OperatorTest, TestQuantizedRescaleSequence) {
  CHECK_IF_ENABLED();

  const size_t len = 100;

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "A", false);

  auto AH = bindings_.allocate(A)->getHandle();

  // Notice that the range below is the an approximation of the scale factors
  // in T3 and T4. If we increase the size of the range we may start losing
  // some values.
  AH.randomize(-12, 12, mod_.getPRNG());

  auto T1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto T2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 2);
  auto T3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.1, -3);
  auto T4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.1, 7);
  auto T5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.3, -3);

  Node *R = F_->createQuantize("R", A, T1);
  // Check that a sequence of type conversions does not change the result.
  R = F_->createRescaleQuantized("R", R, T1);
  R = F_->createRescaleQuantized("R", R, T2);
  R = F_->createRescaleQuantized("R", R, T3);
  // Check that adding the quantized zero does not change the result.
  auto *G = F_->createSplat("splatZero", T3, 0.0);
  R = F_->createAdd("addZero", G, R);
  R = F_->createRescaleQuantized("R", R, T4);
  R = F_->createRescaleQuantized("R", R, T5);
  R = F_->createRescaleQuantized("R", R, T1);
  auto *DQ = F_->createDequantize("DQ", R);

  // Test a sequence of rescale operations t
  auto *result = F_->createSave("save", DQ);
  auto OH = bindings_.allocate(result->getPlaceholder())->getHandle();
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < len; i++) {
    EXPECT_NEAR(AH.at({i}), OH.at({i}), 1.0);
  }
}

TEST_P(OperatorTest, FCGradientCheck) {
  CHECK_IF_ENABLED();

  // Create net representing A*X+Y=B, where X and Y are trainable, while
  // A and B are fixed. Record gradients for X and Y after 3 steps and compare
  // with reference values.
  TrainingConfig TC;

  // This variable records the number of the next sample to be used for
  // training.
  size_t sampleCounter = 0;

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "A", false);
  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "B", false);
  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1}, "X", true);
  auto *Y = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "Y", true);

  bindings_.allocate(A);
  bindings_.allocate(B);
  bindings_.allocate(X)->init(Tensor::InitKind::Broadcast, -1.26274,
                              mod_.getPRNG());
  bindings_.allocate(Y)->init(Tensor::InitKind::Broadcast, 0.1, mod_.getPRNG());

  auto *FC = F_->createFullyConnected("fc", A, X, Y);
  auto *S = F_->createRegression("reg", FC, B);
  auto *save = F_->createSave("ret", S);
  bindings_.allocate(save->getPlaceholder());

  Tensor initA(ElemKind::FloatTy, {2, 1});
  Tensor initB(ElemKind::FloatTy, {2, 1});
  initA.getHandle() = {4.2f, 9.875f};
  initB.getHandle() = {-13.1f, 3.14f};

  Function *DF = glow::differentiate(F_, TC, "d_main");
  auto dfName = DF->getName();
  EE_.compile(CompilationMode::Train);
  runBatch(EE_, bindings_, 3, sampleCounter, {A, B}, {&initA, &initB}, dfName);

  EXPECT_NEAR(bindings_.get(X)->getHandle().raw(0), -0.21294, 1E-5);
  EXPECT_NEAR(bindings_.get(Y)->getHandle().raw(0), 0.01656, 1E-5);
}

/// Helper to test concatVectors using \p DTy.
template <typename DataType>
static void testConcatVectors(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind DTy) {
  F->setName("concatVectors");

  auto *V1 =
      createPlaceholderConditionallyQuantized(mod, DTy, {10}, "V1", false);
  auto *V2 =
      createPlaceholderConditionallyQuantized(mod, DTy, {20}, "V2", false);
  auto *V3 =
      createPlaceholderConditionallyQuantized(mod, DTy, {30}, "V3", false);

  bindings.allocate(V1);
  bindings.allocate(V2);
  bindings.allocate(V3);

  Node *L = F->createConcat("concat", {V1, V2, V3}, 0);
  auto *result = F->createSave("ret", L);
  bindings.allocate(result->getPlaceholder());

  auto I1 = createTensorConditionallyQuantized(DTy, {10});
  auto I2 = createTensorConditionallyQuantized(DTy, {20});
  auto I3 = createTensorConditionallyQuantized(DTy, {30});

  for (dim_t i = 0; i < 10; i++) {
    I1.getHandle<DataType>().at({i}) = i;

    I2.getHandle<DataType>().at({i}) = i + 10;
    I2.getHandle<DataType>().at({i + 10}) = i + 20;
    I3.getHandle<DataType>().at({i}) = i + 30;
    I3.getHandle<DataType>().at({i + 10}) = i + 40;
    I3.getHandle<DataType>().at({i + 20}) = i + 50;
  }

  EE.compile(CompilationMode::Infer);

  // Testing the output vector.
  updateInputPlaceholders(bindings, {V1, V2, V3}, {&I1, &I2, &I3});
  EE.run(bindings);

  auto RNWH = bindings.get(result->getPlaceholder())->getHandle<DataType>();
  (void)RNWH;

  for (dim_t i = 0; i < 60; i++) {
    EXPECT_NEAR(RNWH.at({i}), static_cast<DataType>(i), 0.001);
  }
}

/// Test concatenating vectors that are Int64ITy.
TEST_P(OperatorTest, concatVectors_Int64) {
  CHECK_IF_ENABLED();
  testConcatVectors<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test concatenating vectors that are Int32ITy.
TEST_P(OperatorTest, concatVectors_Int32) {
  CHECK_IF_ENABLED();
  testConcatVectors<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Test concatenating vectors that are Int8Qty.
TEST_P(OperatorTest, concatVectors_Int8) {
  CHECK_IF_ENABLED();
  testConcatVectors<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Test concatenating vectors that are BoolTy.
TEST_P(OperatorTest, concatVectors_Bool) {
  CHECK_IF_ENABLED();
  testConcatVectors<bool>(bindings_, mod_, F_, EE_, ElemKind::BoolTy);
}

/// Test concatenating vectors that are FloatTy.
TEST_P(OperatorTest, concatVectors_Float) {
  CHECK_IF_ENABLED();
  testConcatVectors<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test concatenating vectors that are Float16Ty.
TEST_P(OperatorTest, concatVectors_Float16) {
  CHECK_IF_ENABLED();
  testConcatVectors<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Helper to test ConcatVectorsRepeated using \p DTy.
template <typename DataType>
static void testConcatVectorsRepeated(glow::PlaceholderBindings &bindings,
                                      glow::Module &mod, glow::Function *F,
                                      glow::ExecutionEngine &EE, ElemKind DTy) {
  F->setName("concatVectors");

  auto *V1 =
      createPlaceholderConditionallyQuantized(mod, DTy, {10}, "V1", false);
  auto *V2 =
      createPlaceholderConditionallyQuantized(mod, DTy, {20}, "V2", false);
  bindings.allocate(V1);
  bindings.allocate(V2);

  // Alternate adding sequences of V1 and V2, so that the IRGen'd
  // InsertTensors have different counts.
  Node *L = F->createConcat("concat", {V2, V1, V1, V1, V2, V2, V1, V1, V2}, 0);
  auto *result = F->createSave("ret", L);
  bindings.allocate(result->getPlaceholder());

  auto I1 = createTensorConditionallyQuantized(DTy, {10});
  auto I2 = createTensorConditionallyQuantized(DTy, {20});
  auto I1H = I1.getHandle<DataType>();
  auto I2H = I2.getHandle<DataType>();
  for (dim_t i = 0; i < 10; i++) {
    I1H.at({i}) = 1;

    I2H.at({i}) = 2;
    I2H.at({i + 10}) = 2;
  }

  EE.compile(CompilationMode::Infer);

  // Testing the output vector.
  updateInputPlaceholders(bindings, {V1, V2}, {&I1, &I2});
  EE.run(bindings);

  auto outH = bindings.get(result->getPlaceholder())->getHandle<DataType>();

  // Simply verify here that the values are in their correct places, based on
  // the number of times/order V1 and V2 are concatenated and their sizes.
  for (dim_t i = 0; i < 130; i++) {
    if ((i < 20) || (i >= 50 && i < 90) || (i >= 110)) {
      EXPECT_EQ(outH.at({i}), static_cast<DataType>(2));
    } else {
      EXPECT_EQ(outH.at({i}), static_cast<DataType>(1));
    }
  }
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing Int64ITy data.
TEST_P(OperatorTest, concatVectorsRepeated_Int64) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<int64_t>(bindings_, mod_, F_, EE_,
                                     ElemKind::Int64ITy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing Int32ITy data.
TEST_P(OperatorTest, concatVectorsRepeated_Int32) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<int32_t>(bindings_, mod_, F_, EE_,
                                     ElemKind::Int32ITy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing Int8QTy data.
TEST_P(OperatorTest, concatVectorsRepeated_Int8) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<int8_t>(bindings_, mod_, F_, EE_,
                                    ElemKind::Int8QTy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing BoolTy data.
TEST_P(OperatorTest, concatVectorsRepeated_Bool) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<bool>(bindings_, mod_, F_, EE_, ElemKind::BoolTy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing FloatTy data.
TEST_P(OperatorTest, concatVectorsRepeated_Float) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing Float16Ty data.
TEST_P(OperatorTest, concatVectorsRepeated_Float16) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<float16_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::Float16Ty);
}

/// Helper to test SliceVectors using \p DTy.
template <typename DataType>
static void testSliceVectors(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  F->setName("sliceVectors");

  auto *V =
      createPlaceholderConditionallyQuantized(mod, DTy, {3, 30}, "V", false);
  bindings.allocate(V);

  Node *S1 = F->createSlice("slice1", V, {0, 10}, {3, 13});
  Node *S2 = F->createSlice("slice2", V, {1, 0}, {2, 30});
  Node *S3 = F->createSlice("slice3", V, {2, 10}, {3, 12});

  auto *result1 = F->createSave("ret1", S1);
  auto *result2 = F->createSave("ret2", S2);
  auto *result3 = F->createSave("ret3", S3);

  bindings.allocate(result1->getPlaceholder());
  bindings.allocate(result2->getPlaceholder());
  bindings.allocate(result3->getPlaceholder());

  auto I = createTensorConditionallyQuantized(DTy, {3, 30});
  auto IH = I.getHandle<DataType>();
  for (dim_t j = 0; j < 30; j++) {
    IH.at({0, j}) = j;
    IH.at({1, j}) = j + 30;
    IH.at({2, j}) = j + 60;
  }

  EE.compile(CompilationMode::Infer);

  // Testing the output slices.
  updateInputPlaceholders(bindings, {V}, {&I});
  EE.run(bindings);

  auto RNWH1 = bindings.get(result1->getPlaceholder())->getHandle<DataType>();
  auto RNWH2 = bindings.get(result2->getPlaceholder())->getHandle<DataType>();
  auto RNWH3 = bindings.get(result3->getPlaceholder())->getHandle<DataType>();

  EXPECT_EQ(3, RNWH1.dims()[0]);
  EXPECT_EQ(3, RNWH1.dims()[1]);
  for (dim_t i = 0; i < 3; i++) {
    for (dim_t j = 10; j < 13; j++) {
      EXPECT_NEAR(RNWH1.at({i, j - 10}), j + i * 30, 0.001);
    }
  }
  EXPECT_EQ(1, RNWH2.dims()[0]);
  EXPECT_EQ(30, RNWH2.dims()[1]);
  for (dim_t j = 0; j < 30; j++) {
    EXPECT_NEAR(RNWH2.at({0, j}), j + 30, 0.001);
  }
  EXPECT_EQ(1, RNWH3.dims()[0]);
  EXPECT_EQ(2, RNWH3.dims()[1]);
  for (dim_t j = 10; j < 12; j++) {
    EXPECT_NEAR(RNWH3.at({0, j - 10}), j + 60, 0.001);
  }
}

/// Test slicing with Int64ITy.
TEST_P(OperatorTest, sliceVectors_Int64) {
  CHECK_IF_ENABLED();
  testSliceVectors<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test slicing with FloatTy.
TEST_P(OperatorTest, sliceVectors_Float) {
  CHECK_IF_ENABLED();
  testSliceVectors<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test slicing with Float16Ty.
TEST_P(OperatorTest, sliceVectors_Float16) {
  CHECK_IF_ENABLED();
  testSliceVectors<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test slicing with Int8QTy.
TEST_P(OperatorTest, sliceVectors_Int8) {
  CHECK_IF_ENABLED();
  testSliceVectors<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Test slicing with Int32QTy.
TEST_P(OperatorTest, sliceVectors_Int32) {
  CHECK_IF_ENABLED();
  testSliceVectors<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32QTy);
}

/// Helper to test SliceConcatVectors using \p DTy.
template <typename DataType>
static void testSliceConcatVectors(glow::PlaceholderBindings &bindings,
                                   glow::Module &mod, glow::Function *F,
                                   glow::ExecutionEngine &EE, ElemKind DTy) {
  F->setName("sliceConcatVectors");

  auto *V =
      createPlaceholderConditionallyQuantized(mod, DTy, {5, 4}, "V", false);
  bindings.allocate(V);

  auto I = createTensorConditionallyQuantized(DTy, {5, 4});
  auto IH = I.getHandle<DataType>();
  for (dim_t i = 0; i < 5; i++) {
    for (dim_t j = 0; j < 4; j++) {
      IH.at({i, j}) = i * 10 + j;
    }
  }

  Node *S0 = F->createSlice("slice0", V, {1, 0}, {5, 4});
  Node *S1 = F->createSlice("slice1", S0, {0, 0}, {2, 4});
  Node *S2 = F->createSlice("slice2", S0, {2, 0}, {4, 4});
  Node *S3 = F->createSlice("slice3", S0, {0, 0}, {2, 2});
  Node *S4 = F->createSlice("slice4", S0, {2, 2}, {4, 4});
  Node *S5 = F->createSlice("slice5", V, {0, 0}, {1, 4});

  Node *C0 = F->createConcat("concat0", {S5, S1}, 0);
  Node *C1 = F->createConcat("concat1", {S3, S4}, 1);
  Node *C2 = F->createConcat("concat2", {S2, C1, C0}, 0);

  auto *result = F->createSave("ret", C2);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {V}, {&I});
  EE.run(bindings);

  const DataType expected[7][4] = {
      {30, 31, 32, 33}, {40, 41, 42, 43}, {10, 11, 32, 33}, {20, 21, 42, 43},
      {0, 1, 2, 3},     {10, 11, 12, 13}, {20, 21, 22, 23}};

  auto resultH = bindings.get(result->getPlaceholder())->getHandle<DataType>();
  EXPECT_EQ(7, resultH.dims()[0]);
  EXPECT_EQ(4, resultH.dims()[1]);
  for (dim_t i = 0; i < 7; i++) {
    for (dim_t j = 0; j < 4; j++) {
      EXPECT_EQ(resultH.at({i, j}), expected[i][j]);
    }
  }
}

/// Test a combination of slicing and concating, in Int64ITy.
TEST_P(OperatorTest, sliceConcatVectors_Int64) {
  CHECK_IF_ENABLED();
  testSliceConcatVectors<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test a combination of slicing and concating, in Int8QTy.
TEST_P(OperatorTest, sliceConcatVectors_Int8) {
  CHECK_IF_ENABLED();
  testSliceConcatVectors<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Test a combination of slicing and concating, in FloatTy.
TEST_P(OperatorTest, sliceConcatVectors_Float) {
  CHECK_IF_ENABLED();
  testSliceConcatVectors<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test a combination of slicing and concating, in Float16Ty.
TEST_P(OperatorTest, sliceConcatVectors_Float16) {
  CHECK_IF_ENABLED();
  testSliceConcatVectors<float16_t>(bindings_, mod_, F_, EE_,
                                    ElemKind::Float16Ty);
}

TEST_P(OperatorTest, Tile) {
  CHECK_IF_ENABLED();

  F_->setName("concatVectors");

  auto *V = mod_.createPlaceholder(ElemKind::FloatTy, {4, 5}, "V", false);
  bindings_.allocate(V);

  Node *T0 = F_->createTile("tile0", V, /* tiles */ 3, /* axis */ 0);
  auto *result0 = F_->createSave("res0", T0);
  bindings_.allocate(result0->getPlaceholder());

  Node *T1 = F_->createTile("tile1", V, /* tiles */ 3, /* axis */ 1);
  auto *result1 = F_->createSave("res1", T1);
  bindings_.allocate(result1->getPlaceholder());

  Tensor VT(ElemKind::FloatTy, {4, 5});

  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 5; j++) {
      VT.getHandle<float>().at({i, j}) = i * 5 + j;
    }
  }

  EE_.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings_, {V}, {&VT});
  EE_.run(bindings_);

  // Testing the output vector with axis 0.
  auto res0 = bindings_.get(result0->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < res0.dims()[0]; i++) {
    for (dim_t j = 0; j < res0.dims()[1]; j++) {
      EXPECT_EQ(res0.at({i, j}), (i % 4) * 5 + j);
    }
  }

  // Testing the output vector with axis 1.
  auto res1 = bindings_.get(result1->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < res1.dims()[0]; i++) {
    for (dim_t j = 0; j < res1.dims()[1]; j++) {
      EXPECT_EQ(res1.at({i, j}), i * 5 + (j % 5));
    }
  }
}

TEST_P(OperatorTest, QuantizedTile) {
  CHECK_IF_ENABLED();

  F_->setName("concatVectors");

  auto *V = mod_.createPlaceholder(ElemKind::FloatTy, {4, 5}, "V", false);
  bindings_.allocate(V);

  auto quantizationParams = glow::quantization::chooseQuantizationParams(0, 20);
  auto quantizeTy =
      mod_.uniqueType(ElemKind::Int8QTy, {4, 5}, quantizationParams.scale,
                      quantizationParams.offset);
  auto *Q = F_->createQuantize("quantize", V, quantizeTy);

  Node *T0 = F_->createTile("tile0", Q, /* tiles */ 3, /* axis */ 0);
  auto *DQ0 = F_->createDequantize("dequantize0", T0);
  auto *result0 = F_->createSave("res0", DQ0);
  bindings_.allocate(result0->getPlaceholder());

  Node *T1 = F_->createTile("tile1", Q, /* tiles */ 3, /* axis */ 1);
  auto *DQ1 = F_->createDequantize("dequantize1", T1);
  auto *result1 = F_->createSave("res1", DQ1);
  bindings_.allocate(result1->getPlaceholder());

  Tensor VT(ElemKind::FloatTy, {4, 5});

  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 5; j++) {
      VT.getHandle<float>().at({i, j}) = i * 5 + j;
    }
  }

  EE_.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings_, {V}, {&VT});
  EE_.run(bindings_);

  // Testing the output vector with axis 0.
  auto res0 = bindings_.get(result0->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < res0.dims()[0]; i++) {
    for (dim_t j = 0; j < res0.dims()[1]; j++) {
      EXPECT_NEAR(res0.at({i, j}), (i % 4) * 5 + j, 0.05);
    }
  }

  // Testing the output vector with axis 1.
  auto res1 = bindings_.get(result1->getPlaceholder())->getHandle<float>();
  (void)res1;
  for (dim_t i = 0; i < res1.dims()[0]; i++) {
    for (dim_t j = 0; j < res1.dims()[1]; j++) {
      EXPECT_NEAR(res1.at({i, j}), i * 5 + (j % 5), 0.05);
    }
  }
}

TEST_P(OperatorTest, Clip) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {5, 5}, "X", false);
  auto xHandle = bindings_.allocate(X)->getHandle();
  xHandle = {45.0, 16.0, 59.0, 99.0, 48.0, 12.0, 44.0, 46.0, 82.0,
             28.0, 1.0,  91.0, 18.0, 9.0,  71.0, 24.0, 37.0, 61.0,
             12.0, 81.0, 36.0, 38.0, 30.0, 84.0, 40.0};

  float min = 20.0;
  float max = 60.0;
  auto *node = F_->createClip("clip", X, min, max);
  auto *save = F_->createSave("save", node);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = saveTensor->getHandle();
  std::vector<dim_t> expectedDims = {5, 5};
  std::vector<float> expectedValues = {45.0, 20.0, 59.0, 60.0, 48.0, 20.0, 44.0,
                                       46.0, 60.0, 28.0, 20.0, 60.0, 20.0, 20.0,
                                       60.0, 24.0, 37.0, 60.0, 20.0, 60.0, 36.0,
                                       38.0, 30.0, 60.0, 40.0};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 5 * 5; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

TEST_P(OperatorTest, simpleCmpSelectPredication) {
  CHECK_IF_ENABLED();

  // A simple test that checks predication of some values using the
  // compare-select pair of instructions. Keep doubling some values
  // until some condition is met.

  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {10}, "inputs", false);
  auto *counters =
      mod_.createPlaceholder(ElemKind::FloatTy, {10}, "counters", false);

  bindings_.allocate(counters)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  bindings_.allocate(inputs)->getHandle().clear(1);

  Node *cnt = counters;
  NodeValue data = inputs;
  Node *const1 = F_->createSplat("const1", counters->getType(), 1.0);
  Node *const0 = F_->createSplat("const0", counters->getType(), 0.0);

  for (int i = 0; i < 10; i++) {
    cnt = F_->createSub("sub1", cnt, const1);
    Node *pred = F_->createCmpLTE("cmp", const0, cnt);

    Node *const2 = F_->createSplat("const2", data.getType(), 2.0);
    Node *newData = F_->createMul("mul2x", data, const2);

    data = F_->createSelect("select", pred, newData, data);
  }

  auto *SN = F_->createSave("ret", data);
  bindings_.allocate(SN->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(SN->getPlaceholder())->getHandle();
  ASSERT_NEAR(H.at(0), 1, 0.001);
  ASSERT_NEAR(H.at(1), 2, 0.001);
  ASSERT_NEAR(H.at(2), 4, 0.001);
  ASSERT_NEAR(H.at(3), 8, 0.001);
  ASSERT_NEAR(H.at(4), 16, 0.001);
  ASSERT_NEAR(H.at(5), 32, 0.001);
  ASSERT_NEAR(H.at(6), 64, 0.001);
  ASSERT_NEAR(H.at(7), 128, 0.001);
  ASSERT_NEAR(H.at(8), 256, 0.001);
  ASSERT_NEAR(H.at(9), 512, 0.001);
}

TEST_P(OperatorTest, simplePredication) {
  CHECK_IF_ENABLED();

  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10, 10}, "inputs", false);
  auto *counters =
      mod_.createPlaceholder(ElemKind::FloatTy, {10}, "counters", false);

  bindings_.allocate(counters)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  bindings_.allocate(inputs)->getHandle().randomize(-10, 10, mod_.getPRNG());

  Node *C5 = F_->createSplat("C5", counters->getType(), 5.0);
  Node *pred = F_->createCmpLTE("cmp", C5, counters);

  auto *FC0 = F_->createFullyConnected(bindings_, "FC0", inputs, 128);
  auto *RL0 = F_->createRELU("RL0", FC0);
  auto *FC1 = F_->createFullyConnected(bindings_, "FC1", RL0, 64);
  auto *RL1 = F_->createRELU("RL1", FC1);
  auto *FC2 = F_->createFullyConnected(bindings_, "FC2", RL1, 32);
  auto *RL2 = F_->createRELU("RL2", FC2);

  auto *save = F_->createSave("ret", RL2);
  bindings_.allocate(save->getPlaceholder());

  FC0->setPredicate(pred);
  FC1->setPredicate(pred);
  FC2->setPredicate(pred);

  ::glow::convertPlaceholdersToConstants(
      F_, bindings_, {inputs, counters, save->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
}

TEST_P(OperatorTest, ChannelShuffle) {
  CHECK_IF_ENABLED();

  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 12, 1, 1}, "inputs", false);
  bindings_.allocate(inputs)->getHandle() = {1, 2, 3, 4,  5,  6,
                                             7, 8, 9, 10, 11, 12};

  Node *CS = F_->createChannelShuffle("CS", inputs, 3, 1);
  SaveNode *S = F_->createSave("save", CS);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto results = bindings_.get(S->getPlaceholder())->getHandle();

  EXPECT_EQ(results.size(), 12);
  std::vector<float> expected = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  for (dim_t i = 0; i < expected.size(); i++)
    EXPECT_FLOAT_EQ(results.at({0, i, 0, 0}), expected[i]);
}

TEST_P(OperatorTest, Squeeze) {
  CHECK_IF_ENABLED();

  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 5}, "inputs", false);
  bindings_.allocate(inputs)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<float> expectedValues = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Test 1:
  {
    std::vector<dim_t> axes = {0};
    Node *SQZ = F_->createSqueeze("SQZ", inputs, axes);
    SaveNode *S = F_->createSave("save", SQZ);
    bindings_.allocate(S->getPlaceholder());

    EE_.compile(CompilationMode::Infer);
    EE_.run(bindings_);

    auto results = bindings_.get(S->getPlaceholder())->getHandle();
    std::vector<dim_t> expectedDims = {2, 1, 5};
    EXPECT_TRUE(results.dims().vec() == expectedDims);
    for (size_t i = 0; i < 10; i++)
      EXPECT_FLOAT_EQ(results.raw(i), expectedValues[i]);
  }
  bindings_.clear();
  EE_.setBackendName(getBackendName());
  // Test 2:
  {
    auto mod = &EE_.getModule();
    F_ = mod->createFunction("main");
    inputs = mod->createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 5}, "inputs",
                                    false);
    bindings_.allocate(inputs)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<dim_t> axes = {0, 2, 2};
    Node *SQZ = F_->createSqueeze("SQZ", inputs, axes);
    SaveNode *S = F_->createSave("save", SQZ);
    bindings_.allocate(S->getPlaceholder());

    EE_.compile(CompilationMode::Infer);
    EE_.run(bindings_);

    auto results = bindings_.get(S->getPlaceholder())->getHandle();
    std::vector<dim_t> expectedDims = {2, 5};
    EXPECT_TRUE(results.dims().vec() == expectedDims);
    for (size_t i = 0; i < 10; i++)
      EXPECT_FLOAT_EQ(results.raw(i), expectedValues[i]);
  }
  bindings_.clear();
  EE_.setBackendName(getBackendName());
  // Test 3: 0-dimensional Tensor
  {
    auto mod = &EE_.getModule();
    F_ = mod->createFunction("main");
    inputs = mod->createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 5}, "inputs",
                                    false);
    bindings_.allocate(inputs)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto *emptyInput =
        mod->createPlaceholder(ElemKind::FloatTy, {1}, "emptyInput", false);
    bindings_.allocate(emptyInput)->getHandle() = {42.0};

    std::vector<dim_t> axes = {0};
    Node *SQZ = F_->createSqueeze("SQZ", emptyInput, axes);
    SaveNode *S1 = F_->createSave("save", SQZ);
    Node *UnSQZ = F_->createExpandDims("UnSQZ", SQZ, axes);
    SaveNode *S2 = F_->createSave("save", UnSQZ);

    bindings_.allocate(S1->getPlaceholder());
    bindings_.allocate(S2->getPlaceholder());

    EE_.compile(CompilationMode::Infer);
    EE_.run(bindings_);

    auto res1 = bindings_.get(S1->getPlaceholder())->getHandle();
    EXPECT_TRUE(res1.dims().vec() == std::vector<dim_t>());
    EXPECT_FLOAT_EQ(res1.raw(0), 42.0);
    auto res2 = bindings_.get(S2->getPlaceholder())->getHandle();
    EXPECT_TRUE(res2.dims().vec() == std::vector<dim_t>(1, 1));
    EXPECT_FLOAT_EQ(res2.raw(0), 42.0);
  }
}

/// Helper to test ExpandDims using \p DTy.
template <typename DataType>
static void testExpandDims(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *inputs = createPlaceholderConditionallyQuantized(mod, DTy, {2, 2},
                                                         "inputs", false);
  auto IH = bindings.allocate(inputs)->getHandle<DataType>();
  IH = {1, 2, 3, 4};

  // This should be uniqued and sorted, so should become {0, 1, 3, 5}.
  std::vector<dim_t> axes = {3, 0, 5, 1, 3};
  Node *EDN = F->createExpandDims("expand", inputs, axes);
  SaveNode *S = F->createSave("save", EDN);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Expected dims based on the axes above; inserted new dimensions of 1 in
  // every unique axes location, based on the output tensor shape.
  std::vector<dim_t> expectedDims = {1, 1, 2, 1, 2, 1};
  auto results = bindings.get(S->getPlaceholder())->getHandle<DataType>();
  EXPECT_TRUE(results.dims().vec() == expectedDims);

  // The data should be the same, as this was just a reshape.
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(results.raw(i), IH.raw(i));
  }
}

/// Check that the expand dims operator works, which is implemented with a
/// reshape, in FloatTy.
TEST_P(OperatorTest, ExpandDims_Float) {
  CHECK_IF_ENABLED();
  testExpandDims<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Check that the expand dims operator works, which is implemented with a
/// reshape, in Float16Ty.
TEST_P(OperatorTest, ExpandDims_Float16) {
  CHECK_IF_ENABLED();
  testExpandDims<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Check that the expand dims operator works, which is implemented with a
/// reshape, in Int8QTy.
TEST_P(OperatorTest, ExpandDims_Int8) {
  CHECK_IF_ENABLED();
  testExpandDims<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Helper to test Split using \p DTy.
template <typename DataType>
static void testSplit(glow::PlaceholderBindings &bindings, glow::Module &mod,
                      glow::Function *F, glow::ExecutionEngine &EE,
                      ElemKind DTy) {
  auto *inputs = createPlaceholderConditionallyQuantized(mod, DTy, {1, 2, 6},
                                                         "inputs", false);
  bindings.allocate(inputs)->getHandle<DataType>() = {1, 2, 3, 4,  5,  6,
                                                      7, 8, 9, 10, 11, 12};

  std::vector<SliceNode *> outputs1;
  F->createSplit("Split1", inputs, /*outputNum = */ 2, /*axis = */ 2,
                 /*split = */ {}, outputs1);
  std::vector<SliceNode *> outputs2;
  F->createSplit("Split2", inputs, /*outputNum = */ 2, /*axis = */ 2,
                 /*split = */ {2, 4}, outputs2);
  auto *S1 = F->createSave("save1", outputs1[0]);
  auto *S2 = F->createSave("save2", outputs1[1]);
  auto *S3 = F->createSave("save3", outputs2[0]);
  auto *S4 = F->createSave("save4", outputs2[1]);

  auto *result1 = bindings.allocate(S1->getPlaceholder());
  auto *result2 = bindings.allocate(S2->getPlaceholder());
  auto *result3 = bindings.allocate(S3->getPlaceholder());
  auto *result4 = bindings.allocate(S4->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected1 = createTensorConditionallyQuantized(DTy, {1, 2, 3});
  expected1.getHandle<DataType>() = {1, 2, 3, 7, 8, 9};
  EXPECT_TRUE(result1->isEqual(expected1));

  Tensor expected2 = createTensorConditionallyQuantized(DTy, {1, 2, 3});
  expected2.getHandle<DataType>() = {4, 5, 6, 10, 11, 12};
  EXPECT_TRUE(result2->isEqual(expected2));

  Tensor expected3 = createTensorConditionallyQuantized(DTy, {1, 2, 2});
  expected3.getHandle<DataType>() = {1, 2, 7, 8};
  EXPECT_TRUE(result3->isEqual(expected3));

  Tensor expected4 = createTensorConditionallyQuantized(DTy, {1, 2, 4});
  expected4.getHandle<DataType>() = {3, 4, 5, 6, 9, 10, 11, 12};
  EXPECT_TRUE(result4->isEqual(expected4));
}

/// Test that Split is correctly supported in FloatTy.
TEST_P(OperatorTest, Split_Float) {
  CHECK_IF_ENABLED();
  testSplit<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that Split is correctly supported in Float16Ty.
TEST_P(OperatorTest, Split_Float16) {
  CHECK_IF_ENABLED();
  testSplit<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test that Split is correctly supported in Int8QTy.
TEST_P(OperatorTest, Split_Int8) {
  CHECK_IF_ENABLED();
  testSplit<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

TEST_P(OperatorTest, IntRelu) {
  CHECK_IF_ENABLED();

  const float splatValue = 10;
  const float scale = 1.0;
  const float rescaleScale = 2.0;
  const int32_t reluOffset = -128;
  const int32_t offset = 5;
  const size_t size = 5;

  auto splatTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, scale, offset);
  auto rescaleTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, rescaleScale, offset);

  auto *splat = F_->createSplat("splat", splatTy, splatValue);
  auto *rescale = F_->createRescaleQuantized("rescale", splat, rescaleTy);
  auto *reluOutTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, rescaleScale, reluOffset);
  auto *relu = F_->createRELU("relu", rescale, reluOutTy);
  auto *dequantize = F_->createDequantize("dequantize", relu);

  auto *save = F_->createSave("save", dequantize);
  bindings_.allocate(mod_.getPlaceholders());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(save->getPlaceholder())->getHandle();
  float expectedValue = std::max(0.0f, splatValue);
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(expectedValue, result.raw(i));
  }
}

TEST_P(OperatorTest, IntSplat) {
  CHECK_IF_ENABLED();

  const float splatValue = 10;
  const float scale = 1.0;
  const int32_t offset = 5;
  const size_t size = 3;

  auto splatTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, scale, offset);
  auto *splat = F_->createSplat("splat", splatTy, splatValue);
  auto *dequantize = F_->createDequantize("dequantize", splat);

  auto *save = F_->createSave("save", dequantize);
  bindings_.allocate(mod_.getPlaceholders());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(save->getPlaceholder())->getHandle();
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(splatValue, result.raw(i));
  }
}

TEST_P(OperatorTest, Fp16Splat) {
  CHECK_IF_ENABLED();

  const float splatValue = 10;
  const size_t size = 3;

  auto splatTy = mod_.uniqueType(ElemKind::Float16Ty, {size});
  auto *splat = F_->createSplat("splat", splatTy, splatValue);

  auto *save = F_->createSave("save", splat);
  bindings_.allocate(mod_.getPlaceholders());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(save->getPlaceholder())->getHandle<float16_t>();
  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_EQ(float16_t(splatValue), result.raw(i));
  }
}

TEST_P(OperatorTest, GroupConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 8}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 2 * 8; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {6, 1, 1, 4}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 6; i++)
    for (dim_t j = 0; j < 4; j++) {
      FH.at({i, 0, 0, j}) = pow(10.0, i);
    }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {6}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 1, 6});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 1, 1, 0, 2);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 2, 1, 6};
  ASSERT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 1 + 2 + 3 + 4);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), (1 + 2 + 3 + 4) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 2}), (1 + 2 + 3 + 4) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 3}), (5 + 6 + 7 + 8) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 4}), (5 + 6 + 7 + 8) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 5}), (5 + 6 + 7 + 8) * 100000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0}), 9 + 10 + 11 + 12);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1}), (9 + 10 + 11 + 12) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 2}), (9 + 10 + 11 + 12) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 3}), (13 + 14 + 15 + 16) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 4}), (13 + 14 + 15 + 16) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 5}), (13 + 14 + 15 + 16) * 100000);
}

/// Test the functionality of channelwise quantized group convolution using
/// ChannelwiseQuantizedConvNode.
TEST_P(OperatorTest, ChannelwiseQuantizedGroupConvolution) {
  CHECK_IF_ENABLED();

  constexpr size_t groups = 2;
  constexpr size_t output_channel = 4;

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 2}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle<float>();
  for (size_t i = 0; i < 2 * 3 * 2; i++) {
    IH.raw(i) = i + 1;
  }

  auto *qInTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 2, 3, 2}, 1.0, 0);
  auto *qInput = F_->createQuantize("qInput", input, qInTy);

  auto filterT = Tensor(ElemKind::Int8QTy, {4, 2, 1, 1}, 1.0, 0);
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 1; k++) {
        for (dim_t l = 0; l < 1; l++) {
          filterT.getHandle<int8_t>().at({i, j, k, l}) = j + 1;
        }
      }
    }
  }
  auto *filter = mod_.createConstant("filter", std::move(filterT));

  auto biasT = Tensor(ElemKind::FloatTy, {4});
  biasT.zero();
  auto *bias = mod_.createConstant("bias", std::move(biasT));

  auto scalesT = Tensor(ElemKind::FloatTy, {output_channel});
  for (size_t i = 0; i < scalesT.size(); i++) {
    scalesT.getHandle<float>().raw(i) = 1;
  }
  auto *scales = mod_.createConstant("scales", std::move(scalesT));

  auto offsetsT = Tensor(ElemKind::Int32ITy, {output_channel});
  offsetsT.zero();
  auto *offsets = mod_.createConstant("offsets", std::move(offsetsT));

  auto *outTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 1, 3, 4}, 1.0, 0);
  ChannelwiseQuantizedConvolutionNode *CQC = F_->createChannelwiseQuantizedConv(
      "channelwiseQuantizedConv", qInput, filter, bias, scales, offsets, outTy,
      {2, 1}, {1, 1}, {0, 0, 0, 0}, groups);

  DequantizeNode *dq = F_->createDequantize("dequantize", CQC);
  SaveNode *S = F_->createSave("save", dq);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 1, 3, 4};
  ASSERT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 15);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), 15);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 2}), 18);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 3}), 18);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 0}), 21);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 1}), 21);

  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 2}), 24);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 3}), 24);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 0}), 27);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 1}), 27);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 2}), 30);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 3}), 30);
}

TEST_P(OperatorTest, DilatedConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 1, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 3; i++)
    for (dim_t j = 0; j < 3; j++) {
      FH.at({0, i, j, 0}) = 1;
    }
  FH.at({0, 1, 1, 0}) = 0;

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 1, 1});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 3, 1, 2, 1, 2);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 4, 1, 1};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 3);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0}), 4);
  EXPECT_FLOAT_EQ(result.at({0, 2, 0, 0}), 1);
  EXPECT_FLOAT_EQ(result.at({0, 3, 0, 0}), 2);
}

TEST_P(OperatorTest, GroupDilatedConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 2}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (dim_t i = 0; i < 4 * 4 * 2; i++) {
    IH.raw(i) = i;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 2; i++)
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 2; k++) {
        FH.at({i, j, k, 0}) = 1;
      }
    }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 4, 2});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 2, 1, 1, 2, 2);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 4, 4, 2};
  ASSERT_TRUE(result.dims().vec() == expectedDims);

  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 10);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), 11);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 0}), 20);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 1}), 22);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 0}), 24);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 1}), 26);
  EXPECT_FLOAT_EQ(result.at({0, 0, 3, 0}), 12);
  EXPECT_FLOAT_EQ(result.at({0, 0, 3, 1}), 13);

  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0}), 20);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1}), 22);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1, 0}), 40);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1, 1}), 44);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2, 0}), 48);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2, 1}), 52);
  EXPECT_FLOAT_EQ(result.at({0, 1, 3, 0}), 24);
  EXPECT_FLOAT_EQ(result.at({0, 1, 3, 1}), 26);

  EXPECT_FLOAT_EQ(result.at({0, 2, 0, 0}), 36);
  EXPECT_FLOAT_EQ(result.at({0, 2, 0, 1}), 38);
  EXPECT_FLOAT_EQ(result.at({0, 2, 1, 0}), 72);
  EXPECT_FLOAT_EQ(result.at({0, 2, 1, 1}), 76);
  EXPECT_FLOAT_EQ(result.at({0, 2, 2, 0}), 80);
  EXPECT_FLOAT_EQ(result.at({0, 2, 2, 1}), 84);
  EXPECT_FLOAT_EQ(result.at({0, 2, 3, 0}), 40);
  EXPECT_FLOAT_EQ(result.at({0, 2, 3, 1}), 42);

  EXPECT_FLOAT_EQ(result.at({0, 3, 0, 0}), 18);
  EXPECT_FLOAT_EQ(result.at({0, 3, 0, 1}), 19);
  EXPECT_FLOAT_EQ(result.at({0, 3, 1, 0}), 36);
  EXPECT_FLOAT_EQ(result.at({0, 3, 1, 1}), 38);
  EXPECT_FLOAT_EQ(result.at({0, 3, 2, 0}), 40);
  EXPECT_FLOAT_EQ(result.at({0, 3, 2, 1}), 42);
  EXPECT_FLOAT_EQ(result.at({0, 3, 3, 0}), 20);
  EXPECT_FLOAT_EQ(result.at({0, 3, 3, 1}), 21);
}

/// Test Conv3D with group size of 2 to make sure that group 3d convolution
/// works as expected.
TEST_P(OperatorTest, GroupConv3D) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 2, 8},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < input->getType()->size(); i++) {
    IH.raw(i) = i + 1;
  }

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {6, 1, 1, 1, 4},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 6; i++)
    for (dim_t j = 0; j < 4; j++) {
      FH.at({i, 0, 0, 0, j}) = pow(10.0, i);
    }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {6}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 1, 2, 6});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, zeroBias, outTy, 1, 1, 0, 2);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 2, 1, 2, 6};
  ASSERT_TRUE(result.dims().vec() == expectedDims);

  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 0}), 1 + 2 + 3 + 4);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 1}), (1 + 2 + 3 + 4) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 2}), (1 + 2 + 3 + 4) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 3}), (5 + 6 + 7 + 8) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 4}), (5 + 6 + 7 + 8) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 5}), (5 + 6 + 7 + 8) * 100000);

  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 0}), 9 + 10 + 11 + 12);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 1}), (9 + 10 + 11 + 12) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 2}), (9 + 10 + 11 + 12) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 3}), (13 + 14 + 15 + 16) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 4}), (13 + 14 + 15 + 16) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 5}), (13 + 14 + 15 + 16) * 100000);

  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 0}), 17 + 18 + 19 + 20);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 1}), (17 + 18 + 19 + 20) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 2}), (17 + 18 + 19 + 20) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 3}), (21 + 22 + 23 + 24) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 4}), (21 + 22 + 23 + 24) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 5}), (21 + 22 + 23 + 24) * 100000);

  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 0}), 25 + 26 + 27 + 28);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 1}), (25 + 26 + 27 + 28) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 2}), (25 + 26 + 27 + 28) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 3}), (29 + 30 + 31 + 32) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 4}), (29 + 30 + 31 + 32) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 5}), (29 + 30 + 31 + 32) * 100000);
}

/// Check non-square padding for convolution. The first conv has non-square
/// padding, while the second one has zero padding. The second conv's input is
/// the same as the first one's after-padding input. All other parameters of
/// the two convs are the same.
TEST_P(OperatorTest, NonSquarePaddingConvolution) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input",
                                       false, "NHWC");
  auto IH = bindings_.allocate(input)->getHandle();
  for (dim_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 1},
                                       "filter", false, "NHWC");
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 2 * 2 * 2; i++) {
    FH.raw(i) = pow(2.0, i);
  }
  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 8, 2});

  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 2}, {1, 1}, {0, 2, 1, 3}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});

  Tensor &result = *bindings_.get(S->getPlaceholder());

  // Create the reference conv operator whose input is the same as the
  // after-padding-input above.
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 1},
                                        "input1", false, "NHWC");
  bindings_.allocate(input1)->zero();
  auto IH1 = bindings_.get(input1)->getHandle();
  for (dim_t i = 0; i < 4; i++)
    for (dim_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  CN = refF->createConv("Conv1", input1, filter, zeroBias, outTy, {2, 2},
                        {1, 1}, {0, 0, 0, 0}, 1);
  S = refF->createSave("save1", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(refF, bindings_,
                                         {input, input1, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_, "main");
  EE_.run(bindings_, "mainRef");
  Tensor &result1 = *bindings_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

/// Check non-cubic padding for conv3D. The first conv3D has non-cubic
/// padding, while the second one has zero padding. The second conv3D's input is
/// the same as the first one's after-padding input. All other parameters of
/// the two conv3Ds are the same.
TEST_P(OperatorTest, NonCubicPaddingConv3D) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  int nextVal = 1;
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 0; k < 4; k++) {
        IH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 2, 1},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (size_t i = 0; i < filter->getType()->size(); i++) {
    FH.raw(i) = pow(2.0, i);
  }
  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 8, 12, 2});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, zeroBias, outTy, {2, 2, 2},
                       {1, 1, 1}, {0, 2, 5, 1, 3, 4}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});

  Tensor &result = *bindings_.get(S->getPlaceholder());

  // Create the reference conv3D operator whose input is the same as the
  // after-padding-input above.
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 13, 1},
                                        "input1", false);
  bindings_.allocate(input1)->zero();
  auto IH1 = bindings_.get(input1)->getHandle();
  nextVal = 1;
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 2; j < 6; j++) {
      for (dim_t k = 5; k < 9; k++) {
        IH1.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  Function *refF = mod_.createFunction("mainRef");
  CN = refF->createConv3D("Conv3D_1", input1, filter, zeroBias, outTy,
                          {2, 2, 2}, {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  S = refF->createSave("save1", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(refF, bindings_,
                                         {input, input1, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_, "main");
  EE_.run(bindings_, "mainRef");
  Tensor &result1 = *bindings_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

/// Check non-square padding for AveragePool. The first pool op has non-square
/// padding, while the second one has zero padding. The second pool op's input
/// is the same as the first one's after-padding input. All other parameters
/// of the two convs are the same.
TEST_P(OperatorTest, NonSquarePaddingAveragePool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 2, 1, 3});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  Tensor &result = *bindings_.get(S->getPlaceholder());

  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 1}, "input1", false);
  bindings_.allocate(input1)->zero();
  auto IH1 = bindings_.get(input1)->getHandle();
  for (dim_t i = 0; i < 4; i++)
    for (dim_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  Pool = refF->createAvgPool("pool1", input1, 2, 1, 0);
  S = refF->createSave("save1", Pool);
  bindings_.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_, "main");
  EE_.run(bindings_, "mainRef");
  Tensor &result1 = *bindings_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

/// Check non-square padding for MaxPool. The first pool op has non-square
/// padding, while the second one has zero padding. The second pool-op's input
/// is the same as the first one's after-padding input. All other parameters
/// of the two convs are the same.
TEST_P(OperatorTest, NonSquarePaddingMaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 2, 1, 3});
  auto *S = F_->createSave("save", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  Tensor &result = *bindings_.get(S->getPlaceholder());

  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 1}, "input1", false);
  bindings_.allocate(input1)->zero();
  auto IH1 = bindings_.get(input1)->getHandle();
  for (dim_t i = 0; i < 4; i++)
    for (dim_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  Pool = refF->createMaxPool("pool1", input1, 2, 1, 0);
  S = refF->createSave("save1", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_, "main");
  EE_.run(bindings_, "mainRef");

  Tensor &result1 = *bindings_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

TEST_P(OperatorTest, FP16AvgPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {0., 1., 2., 3., 4.,
                                                       5., 6., 7., 8.};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 2, 2, 1});
  out.getHandle<float16_t>() = {2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AvgPool operator works correctly.
TEST_P(OperatorTest, AvgPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {0., 1., 2., 3., 4., 5., 6., 7., 8.};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 2, 2, 1});
  out.getHandle() = {2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, Int8AvgPool) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0,
                                       "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {2, 2}, 1, 0);
  out.getHandle<int8_t>() = {2, 3, 5, 6};
  for (size_t i = 0; i < 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

/// Verify that the AdaptiveAvgPool operator works correctly.
TEST_P(OperatorTest, AdaptiveAvgPool) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {
      0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.};

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 3, 1});
  auto *pool = F_->createAdaptiveAvgPool("pool", input, outTy);
  auto *S = F_->createSave("save", pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 3, 3, 1});
  out.getHandle() = {2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AdaptiveAvgPool operator works correctly with fp16.
TEST_P(OperatorTest, FP16AdaptiveAvgPool) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 4, 4, 1}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {
      0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.};
  auto outTy = mod_.uniqueType(ElemKind::Float16Ty, {1, 3, 3, 1});
  auto *pool = F_->createAdaptiveAvgPool("pool", input, outTy);
  auto *S = F_->createSave("save", pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 3, 3, 1});
  out.getHandle<float16_t>() = {2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AdaptiveAvgPool operator works correctly with int8.
TEST_P(OperatorTest, Int8AdaptiveAvgPool) {
  CHECK_IF_ENABLED();
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 4, 4, 1}, 1, 0,
                                       "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0);
  auto *pool = F_->createAdaptiveAvgPool("pool", input, outTy);
  auto *S = F_->createSave("save", pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0);
  out.getHandle<int8_t>() = {3, 4, 5, 7, 8, 9, 11, 12, 13};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AdaptiveAvgPool operator works correctly with non-square
/// inputs and outputs.
TEST_P(OperatorTest, AdaptiveAvgPoolNonSquare) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {0., 1., 2.,  3.,  4.,  5.,  6., 7.,
                                            8., 9., 10., 11., 12., 13., 14.};

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 2, 1});
  auto *pool = F_->createAdaptiveAvgPool("pool", input, outTy);
  auto *S = F_->createSave("save", pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 3, 2, 1});
  out.getHandle() = {2, 3, 6.5, 7.5, 11, 12};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, MaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {0., 1., 2., 3., 4., 5., 6., 7., 8.};
  auto *pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 2, 2, 1});
  out.getHandle() = {4., 5., 7., 8.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, FP16MaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {0., 1., 2., 3., 4.,
                                                       5., 6., 7., 8.};
  auto *pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 2, 2, 1});
  out.getHandle<float16_t>() = {4., 5., 7., 8.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, Int8MaxPool) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0,
                                       "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {2, 2}, 1, 0);
  out.getHandle<int8_t>() = {4, 5, 7, 8};
  for (size_t i = 0; i < 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

#define COMPARE_UNARY_OP_FUN(_OP_NAME_, LEN, LOW, HIGH)                        \
  static FunctionTensorPair createAndInitBasic##_OP_NAME_##Test(               \
      glow::PlaceholderBindings &bindings, glow::ExecutionEngine &EE) {        \
    auto &mod = EE.getModule();                                                \
    Function *F = mod.createFunction("main");                                  \
                                                                               \
    auto *input =                                                              \
        mod.createPlaceholder(ElemKind::FloatTy, {LEN}, "input", false);       \
    bindings.allocate(input)->getHandle().randomize(LOW, HIGH, mod.getPRNG()); \
    auto *tanh = F->create##_OP_NAME_(#_OP_NAME_, input);                      \
    auto *save = F->createSave("Save", tanh);                                  \
    auto *resultTensor = bindings.allocate(save->getPlaceholder());            \
    return std::make_pair(F, resultTensor);                                    \
  }
COMPARE_UNARY_OP_FUN(Exp, 10, -1.0F, 1.0F)
COMPARE_UNARY_OP_FUN(Tanh, 10, -10.0F, 10.0F)
COMPARE_UNARY_OP_FUN(Log, 1000, 1.0F, 100.0F)
COMPARE_UNARY_OP_FUN(Sigmoid, 10, -10.0F, 10.0F)
#undef COMPARE_UNARY_OP_FUN

/// Reference ideal sigmoid implementation. Computes an fp32 sigmoid
/// and casts the result to FP16.
static float16_t refSigmoidFp16(float x) {
  float res = 1 / (1 + exp(-x));

  return (float16_t)res;
}

/// Test to verify that the sigmoid implementation is equal to the
/// Mirrored LUT implementation
/// Does a sweep of -15,15 and prints the outputs of the NNPI implementation
/// compared to the LUT one, the ideal sigmoid in fp16 is also provided as
/// a visual sanity check, but nothing is enforced against that last one.
static void testSigmoidFp16Sweep(glow::PlaceholderBindings &bindings,
                                 glow::Module &mod, glow::Function *F,
                                 glow::ExecutionEngine &EE) {
  constexpr dim_t N = 100;
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {N}, "input", false);
  auto inputH = bindings.allocate(input)->getHandle();

  constexpr float rangeStart = -20;
  constexpr float rangeEnd = 20;
  constexpr float delta = (rangeEnd - rangeStart) / N;

  for (dim_t i = 0; i < N; i++) {
    inputH.raw(i) = rangeStart + i * delta;
  }

  auto *sigmoid = F->createSigmoid("Sigmoid", input);
  auto *save = F->createSave("Save", sigmoid);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;

  EE.compile(cctx);
  EE.run(bindings);

  auto resultH = resultTensor->getHandle();
  int numDiffs = 0;

  for (dim_t i = 0; i < N; i++) {
    float inputV = inputH.at({i});
    float refIdeal = refSigmoidFp16(inputV);
    float output = resultH.at({i});
    float absDiff = fabs(output - refIdeal);
    float relDiff = fabs(absDiff / (refIdeal + 1e-8));

    bool failed = false;
    // Relative error should be 2^-11 but we are relaxing this constraint
    // due to linear interpolation
    // Absolute error can remain 1e-5 for now
    if (absDiff > 1e-5 && relDiff > 2e-3) {
      numDiffs++;
      failed = true;
    }

    llvm::outs() << "Sigmoid " << i << " " << inputV << " Backend:" << output
                 << " ref_ideal:" << refIdeal << " relDiff:" << relDiff
                 << " absDiff:" << absDiff << " failed:" << failed << "\n";
  }
  llvm::outs() << "Number of diffs: " << numDiffs << "\n";
  llvm::outs().flush();

  EXPECT_EQ(numDiffs, 0);
}

TEST_P(OperatorTest, SigmoidSweep_Float16) {
  CHECK_IF_ENABLED();

  testSigmoidFp16Sweep(bindings_, mod_, F_, EE_);
}

/// Reference ideal tanh implementation. Computes an fp32 tanh
/// and casts the result to FP16, no denorms
static float16_t refTanHFp16(float x) {
  float res = (exp(2 * x) - 1) / (exp(2 * x) + 1);
  if (fabs(res) < 6e-5) {
    res = 0.0;
  }
  return (float16_t)res;
}

/// Test to verify that the tanh implementation is close to the ideal one
/// Does a sweep of -15,15 and prints the outputs of the NNPI implementation
/// compared to the ideal tanh in fp16.
static void testTanHFp16Sweep(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE) {
  constexpr size_t N = 100;
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {N}, "input", false);
  auto inputH = bindings.allocate(input)->getHandle();

  constexpr float rangeStart = -15;
  constexpr float rangeEnd = 15;
  constexpr float delta = (rangeEnd - rangeStart) / N;

  for (size_t i = 0; i < N; i++) {
    inputH.raw(i) = rangeStart + i * delta;
  }

  auto *sigmoid = F->createTanh("TanH", input);
  auto *save = F->createSave("Save", sigmoid);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;

  EE.compile(cctx);
  EE.run(bindings);

  auto resultH = resultTensor->getHandle();
  int count = 0;

  for (dim_t i = 0; i < N; i++) {
    float inputV = inputH.at({i});
    float refIdeal = refTanHFp16(inputV);
    float output = resultH.at({i});
    float diff = fabs(output - refIdeal);

    if (diff > 1e-6) {
      count++;
    }

    llvm::outs() << "TanH " << i << " " << inputV << " Backend:" << output
                 << " ref_ideal:" << refIdeal << " diff:" << diff << "\n";
  }
  llvm::outs().flush();

  EXPECT_EQ(count, 0);
}

TEST_P(OperatorTest, TanHSweep_Float16) {
  CHECK_IF_ENABLED();

  testTanHFp16Sweep(bindings_, mod_, F_, EE_);
}

template <typename DataType>
static void testMaxPoolWithArgmax(glow::PlaceholderBindings &bindings,
                                  glow::Module &mod, glow::Function *F,
                                  glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {1, 3, 3, 1},
                                                        "input", false, "NHWC");
  bindings.allocate(input)->getHandle<DataType>() = {0, 3, 7, 6, 5, 1, 2, 8, 4};
  auto *pool = F->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *SResult = F->createSave("save_result", pool->getResult());
  auto *SArgmax = F->createSave("save_argmax", pool->getArgmax());
  bindings.allocate(SResult->getPlaceholder());
  bindings.allocate(SArgmax->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = bindings.get(SResult->getPlaceholder());
  auto argmax = bindings.get(SArgmax->getPlaceholder());
  Tensor out1 = createTensorConditionallyQuantized(DTy, {1, 2, 2, 1});
  out1.getHandle<DataType>() = {6, 7, 8, 8};
  EXPECT_TRUE(out1.isEqual(*result));

  Tensor out2(IndexElemKind, {1, 2, 2, 1});
  out2.getHandle<sdim_t>() = {3, 2, 7, 7};
  EXPECT_TRUE(out2.isEqual(*argmax));
}

TEST_P(OperatorTest, FloatMaxPoolWithArgmax) {
  CHECK_IF_ENABLED();
  testMaxPoolWithArgmax<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedMaxPoolWithArgmax) {
  CHECK_IF_ENABLED();
  testMaxPoolWithArgmax<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

template <typename DataType>
static void
testMaxPoolWithArgmaxTransposed(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind DTy) {
  // Show that sequence Tensor(NCHW) -> Transpose(NCHWtoNHWC) ->
  // MaxPoolWithArgmax -> Transpose(NHWCtoNCHW) produces correct linearization.
  auto *inputNCHW = createPlaceholderConditionallyQuantized(
      mod, DTy, {1, 3, 4, 4}, "input", false, "NCHW");
  auto inHandle = bindings.allocate(inputNCHW)->getHandle<DataType>();
  inHandle.clear(0.);
  inHandle.at({0, 0, 2, 2}) = 11;
  inHandle.at({0, 1, 2, 2}) = 22;
  inHandle.at({0, 2, 2, 2}) = 33;

  // Input NCHW to NHWC conversion.
  auto *inputNHWC =
      F->createTranspose("transposeInput", inputNCHW, {0, 2, 3, 1}, "NHWC");
  auto *pool =
      F->createMaxPool("pool", inputNHWC, {4, 4}, {4, 4}, {0, 0, 0, 0});

  // NHWC to NCHW conversion.
  auto *resultNCHW = F->createTranspose("transposeRes", pool->getResult(),
                                        {0, 3, 1, 2}, "NCHW");
  auto *argmaxNCHW = F->createTranspose("transposeArgmax", pool->getArgmax(),
                                        {0, 3, 1, 2}, "NCHW");

  auto *SResult = F->createSave("save_result", resultNCHW);
  auto *SArgmax = F->createSave("save_argmax", argmaxNCHW);
  bindings.allocate(SResult->getPlaceholder());
  bindings.allocate(SArgmax->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = bindings.get(SResult->getPlaceholder());
  auto argmax = bindings.get(SArgmax->getPlaceholder());
  Tensor out1 = createTensorConditionallyQuantized(DTy, {1, 3, 1, 1});
  out1.getHandle<DataType>() = {11, 22, 33};
  EXPECT_TRUE(out1.isEqual(*result));

  Tensor out2(IndexElemKind, {1, 3, 1, 1});
  out2.getHandle<sdim_t>() = {0 + 2 * 3 + 2 * 12, 1 + 2 * 3 + 2 * 12,
                              2 + 2 * 3 + 2 * 12};
  EXPECT_TRUE(out2.isEqual(*argmax));
}

TEST_P(OperatorTest, FloatMaxPoolWithArgmaxTransposed) {
  CHECK_IF_ENABLED();
  testMaxPoolWithArgmaxTransposed<float>(bindings_, mod_, F_, EE_,
                                         ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedMaxPoolWithArgmaxTransposed) {
  CHECK_IF_ENABLED();
  testMaxPoolWithArgmaxTransposed<int8_t>(bindings_, mod_, F_, EE_,
                                          ElemKind::Int8QTy);
}

TEST_P(OperatorStatelessTest, Int8Tanh) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicTanhTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.005f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, Tanh_Float16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicTanhTest,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.001f,
                            parCloneCountOpt);
}

/// Verify that the Tanh operator works correctly.
TEST_P(OperatorTest, Tanh) {
  CHECK_IF_ENABLED();

  constexpr size_t size = 10;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  bindings_.allocate(input)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *tanh = F_->createTanh("Tanh", input);
  auto *save = F_->createSave("Save", tanh);
  bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(save->getPlaceholder())->getHandle();
  auto inputH = bindings_.get(input)->getHandle();

  for (dim_t i = 0; i < size; i++) {
    EXPECT_NEAR(resultH.at({i}), std::tanh(inputH.at({i})), 0.001);
  }
}

TEST_P(OperatorStatelessTest, Exp_Float16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicExpTest,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.005f,
                            parCloneCountOpt);
}

/// Verify that the Exp operator works correctly.
TEST_P(OperatorTest, Exp) {
  CHECK_IF_ENABLED();
  constexpr dim_t size = 10;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  bindings_.allocate(input)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *expn = F_->createExp("Exp", input);
  auto *save = F_->createSave("Save", expn);
  bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(save->getPlaceholder())->getHandle();
  auto inputH = bindings_.get(input)->getHandle();

  for (dim_t i = 0; i < size; i++) {
    EXPECT_NEAR(resultH.at({i}), std::exp(inputH.at({i})), 0.001);
  }
}

/// Verify that a quantized Log works correctly.
TEST_P(OperatorStatelessTest, Int8Log) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicLogTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.1f,
                            parCloneCountOpt);
}

/// Check Non-square kernel for conv.
TEST_P(OperatorTest, NonSquareKernelConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (size_t i = 0; i < 1 * 2 * 3; i++) {
    FH.raw(i) = i + 1;
  }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 2, 1});
  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 3}, {1, 1}, {0, 0, 0, 0}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {106, 127, 190, 211, 274, 295};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-cubic kernel for conv3D.
TEST_P(OperatorTest, NonCubicKernelConv3D) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  int nextVal = 1;
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 0; k < 4; k++) {
        IH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 2, 3, 1},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  nextVal = 1;
  for (dim_t i = 0; i < 1; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 3; k++) {
        FH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 3, 2, 1});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, zeroBias, outTy, {1, 2, 3},
                       {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {106, 127, 190,  211,  274,  295,  442,  463,
                              526, 547, 610,  631,  778,  799,  862,  883,
                              946, 967, 1114, 1135, 1198, 1219, 1282, 1303};
  for (size_t i = 0; i < 4 * 3 * 2; i++) {
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
  }
}

/// Check Non-cubic kernel for conv3D with quantized input, filters, and bias.
TEST_P(OperatorTest, NonCubicKernelConv3DQuantized) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  int nextVal = 1;
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 0; k < 4; k++) {
        IH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto qInType = mod_.uniqueType(ElemKind::Int16QTy, {1, 4, 4, 4, 1}, 0.1, 0);
  QuantizeNode *qInput = F_->createQuantize("q_input", input, qInType);

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 2, 3, 1},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  nextVal = 1;
  for (dim_t i = 0; i < 1; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 3; k++) {
        FH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto qFilterType =
      mod_.uniqueType(ElemKind::Int16QTy, {1, 1, 2, 3, 1}, 0.1, 0);
  QuantizeNode *qFilter = F_->createQuantize("q_filter", filter, qFilterType);

  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(bias)->zero();

  auto qBiasType = mod_.uniqueType(ElemKind::Int32QTy, {1}, 0.1, 0);
  QuantizeNode *qBias = F_->createQuantize("q_bias", bias, qBiasType);

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 3, 2, 1});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, bias, outTy, {1, 2, 3},
                       {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);

  auto qOutTy = mod_.uniqueType(ElemKind::Int16QTy, {1, 4, 3, 2, 1}, 0.1, 0);

  Convolution3DNode *qCN =
      F_->createConv3D("q_Conv3D", qInput, qFilter, qBias, qOutTy, {1, 2, 3},
                       {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);

  SaveNode *S = F_->createSave("save", CN);

  DequantizeNode *deQ = F_->createDequantize("deQ_result", qCN);
  SaveNode *qS = F_->createSave("save", deQ);

  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  bindings_.allocate(mod_.getPlaceholders());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor &qResult = *bindings_.get(qS->getPlaceholder());

  for (size_t i = 0; i < 4 * 3 * 2; i++) {
    EXPECT_NEAR(qResult.getHandle().raw(i), result.getHandle().raw(i), 0.5);
  }
}

/// Test for quantized Convolution3D.
static void Conv3DQuantizedTest(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind elemKind,
                                ElemKind biaselemKind) {
  // Create floating-point network.
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1}, "input", false);
  auto *filter = mod.createPlaceholder(ElemKind::FloatTy, {1, 1, 2, 3, 1},
                                       "filter", false);
  auto *bias = mod.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 4, 3, 2, 1});
  Convolution3DNode *conv3d =
      F->createConv3D("Conv3D", input, filter, bias, outTy, {1, 2, 3},
                      {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  SaveNode *save = F->createSave("save", conv3d);

  // Quantized types.
  auto inputTQP = quantization::chooseQuantizationParams(
      -1.0, 1.0, quantization::Schema::Asymmetric, elemKind);
  auto filterTQP = quantization::chooseQuantizationParams(
      -1.0, 1.0, quantization::Schema::Asymmetric, elemKind);
  auto biasTQP = quantization::chooseQuantizationParams(
      -1.0, 1.0, quantization::Schema::Asymmetric, biaselemKind);
  auto outputTQP = quantization::chooseQuantizationParams(
      -4.0, 4.0, quantization::Schema::Asymmetric, elemKind);

  // Create quantized network.
  auto inputQTy = mod.uniqueType(elemKind, {1, 4, 4, 4, 1}, inputTQP.scale,
                                 inputTQP.offset);
  auto filterQTy = mod.uniqueType(elemKind, {1, 1, 2, 3, 1}, filterTQP.scale,
                                  filterTQP.offset);
  auto biasQTy =
      mod.uniqueType(biaselemKind, {1}, biasTQP.scale, biasTQP.offset);
  auto outQTy = mod.uniqueType(elemKind, {1, 4, 3, 2, 1}, outputTQP.scale,
                               outputTQP.offset);
  QuantizeNode *inputQ = F->createQuantize("inputQ", input, inputQTy);
  QuantizeNode *filterQ = F->createQuantize("filterQ", filter, filterQTy);
  QuantizeNode *biasQ = F->createQuantize("biasQ", bias, biasQTy);
  Convolution3DNode *conv3dQ =
      F->createConv3D("Conv3DQ", inputQ, filterQ, biasQ, outQTy, {1, 2, 3},
                      {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  DequantizeNode *deQ = F->createDequantize("deQ", conv3dQ);
  SaveNode *saveQ = F->createSave("saveQ", deQ);

  // Allocate placeholders.
  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(filter)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(bias)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(save->getPlaceholder());
  bindings.allocate(saveQ->getPlaceholder());

  // Run network.
  ::glow::convertPlaceholdersToConstants(
      F, bindings, {input, save->getPlaceholder(), saveQ->getPlaceholder()});
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compare.
  Tensor &res = *bindings.get(save->getPlaceholder());
  Tensor &resQ = *bindings.get(saveQ->getPlaceholder());
  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_NEAR(res.getHandle().raw(i), resQ.getHandle().raw(i), 0.03);
  }
}

/// Test Int8 Conv3D with Int8 bias.
TEST_P(OperatorTest, Conv3DQuantizedTest_Int8_BiasInt8) {
  ENABLED_BACKENDS(Interpreter);
  Conv3DQuantizedTest(bindings_, mod_, F_, EE_, ElemKind::Int8QTy,
                      ElemKind::Int8QTy);
}

/// Test Int8 Conv3D with Int32 bias.
TEST_P(OperatorTest, Conv3DQuantizedTest_Int8_BiasInt32) {
  ENABLED_BACKENDS(Interpreter);
  Conv3DQuantizedTest(bindings_, mod_, F_, EE_, ElemKind::Int8QTy,
                      ElemKind::Int32QTy);
}

/// Test Int16 Conv3D with Int16 bias.
TEST_P(OperatorTest, Conv3DQuantizedTest_Int16_BiasInt16) {
  ENABLED_BACKENDS(Interpreter);
  Conv3DQuantizedTest(bindings_, mod_, F_, EE_, ElemKind::Int16QTy,
                      ElemKind::Int16QTy);
}

/// Test Int16 Conv3D with Int32 bias.
TEST_P(OperatorTest, Conv3DQuantizedTest_Int16_BiasInt32) {
  ENABLED_BACKENDS(Interpreter);
  Conv3DQuantizedTest(bindings_, mod_, F_, EE_, ElemKind::Int16QTy,
                      ElemKind::Int32QTy);
}

/// Check Non-square kernel for AveragePool.
TEST_P(OperatorTest, NonSquareKernelAveragePool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 3}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {4, 5, 8, 9, 12, 13};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square kernel for MaxPool.
TEST_P(OperatorTest, NonSquareKernelMaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 3}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {7, 8, 11, 12, 15, 16};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for conv.
TEST_P(OperatorTest, NonSquareStrideConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 2, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (size_t i = 0; i < 1 * 2 * 2; i++) {
    FH.raw(i) = i + 1;
  }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 2, 1});
  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 2}, {3, 2}, {0, 0, 1, 1}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {44, 64, 41, 47};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-cubic stride for conv3D.
TEST_P(OperatorTest, NonCubicStrideConv3D) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  int nextVal = 1;
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 0; k < 4; k++) {
        IH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 2, 2, 1},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  nextVal = 1;
  for (dim_t i = 0; i < 2; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 2; k++) {
        FH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 2, 2, 1});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, zeroBias, outTy, {2, 2, 2},
                       {3, 2, 3}, {0, 0, 0, 1, 1, 1}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {560, 296, 848, 424, 524, 220, 604, 252};
  for (size_t i = 0; i < 8; i++) {
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
  }
}

/// Check Non-square stride for AveragePool.
TEST_P(OperatorTest, NonSquareStrideAveragePool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {3, 2}, {0, 0, 1, 1});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {3.5, 5.5, 6.75, 7.75};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for MaxPool.
TEST_P(OperatorTest, NonSquareStrideMaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {3, 2}, {0, 0, 1, 1});
  auto *S = F_->createSave("save", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {6, 8, 14, 16};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

TEST_P(OperatorTest, SigmoidOverflow) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  IH.raw(0) = 1000;
  IH.raw(1) = -1000;

  auto *fpSigmoid = F_->createSigmoid("fpSigmoid", input);
  auto *S = F_->createSave("fpSave", fpSigmoid);
  bindings_.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());
  static const float ref[] = {1, 0};
  for (size_t i = 0; i < 2; i++) {
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
  }
}

TEST_P(OperatorStatelessTest, Int8Sigmoid) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicSigmoidTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.005f,
                            parCloneCountOpt);
}

/// Check that the batch add operator works properly.
TEST_P(OperatorTest, BatchAdd) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {13, 3, 3}, "A", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *slice =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "slice", false);
  bindings_.allocate(slice)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *batchAdd = F_->createBatchedAdd("batchAdd", input, slice);
  auto *S = F_->createSave("save", batchAdd);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float>();
  auto handleInput = bindings_.get(input)->getHandle<float>();
  auto handleSlice = bindings_.get(slice)->getHandle<float>();
  ASSERT_EQ(result.size(), handleInput.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx),
              handleInput.raw(idx) + handleSlice.raw(idx % handleSlice.size()));
  }
}

/// Check that the batch add operator works properly for FP16.
TEST_P(OperatorTest, FP16BatchAdd) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {13, 3, 3}, "A", false);
  bindings_.allocate(input)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *slice =
      mod_.createPlaceholder(ElemKind::Float16Ty, {3, 3}, "slice", false);
  bindings_.allocate(slice)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *batchAdd = F_->createBatchedAdd("batchAdd", input, slice);
  auto *S = F_->createSave("save", batchAdd);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float16_t>();
  auto handleInput = bindings_.get(input)->getHandle<float16_t>();
  auto handleSlice = bindings_.get(slice)->getHandle<float16_t>();
  ASSERT_EQ(result.size(), handleInput.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx),
              handleInput.raw(idx) + handleSlice.raw(idx % handleSlice.size()));
  }
}

TEST_P(OperatorTest, BroadcastAdd2x) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 1}, "input", false);
  auto *bias = mod_.createConstant(ElemKind::FloatTy, {1, 1}, "bias");
  bias->getPayloadMutable().getHandle() = {42};
  auto *tile = F_->createTile("tile", bias, 10, 0);
  auto *add = F_->createAdd("add", input, tile);
  auto *save = F_->createSave("save", add);
  auto *output = save->getPlaceholder();
  bindings_.allocate(input)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  bindings_.allocate(output);
  EE_.compile(CompilationMode::Infer);
  for (int i = 0; i < 2; i++) {
    Tensor expected(ElemKind::FloatTy, {10, 1});
    expected.getHandle() = {42, 43, 44, 45, 46, 47, 48, 49, 50, 51};
    EE_.run(bindings_);
    EXPECT_TRUE(bindings_.get(output)->isEqual(expected));
  }
}

/// Helper to test Sigmoid using \p DTy.
template <typename DataType>
static void testSigmoid(glow::PlaceholderBindings &bindings, glow::Module &mod,
                        glow::Function *F, glow::ExecutionEngine &EE,
                        ElemKind DTy) {
  constexpr size_t size = 10;
  auto *input = mod.createPlaceholder(DTy, {size}, "input", false);
  bindings.allocate(input)->getHandle<DataType>().randomize(-10.0, 10.0,
                                                            mod.getPRNG());

  auto *sigmoid = F->createSigmoid("sigmoid", input);
  auto *save = F->createSave("Save", sigmoid);
  bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto RH = bindings.get(save->getPlaceholder())->getHandle<DataType>();
  auto inH = bindings.get(input)->getHandle<DataType>();

  for (dim_t i = 0; i < size; i++) {
    float val = 1 / (1 + std::exp(-(float)inH.at({i})));
    EXPECT_NEAR(RH.at({i}), val, 0.001);
  }
}

/// Verify that the Sigmoid operator works correctly with FloatTy.
TEST_P(OperatorTest, Sigmoid_Float) {
  CHECK_IF_ENABLED();
  testSigmoid<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Verify that the Sigmoid operator works correctly with Float16Ty.
TEST_P(OperatorTest, Sigmoid_Float16) {
  CHECK_IF_ENABLED();
  testSigmoid<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

TEST_P(OperatorTest, IntLookupTable) {
  CHECK_IF_ENABLED();

  constexpr size_t size = 6;
  auto *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {size}, 1, 0, "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5};

  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, 3, 3);

  // Mapping i -> i.
  std::vector<int8_t> initValues(256);
  for (size_t i = 0; i < 256; ++i) {
    initValues[i] = i - 128;
  }

  auto *lookupTable =
      F_->createIntLookupTable("lookupTable", input, initValues, outTy);
  auto *save = F_->createSave("save", lookupTable);
  bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(save->getPlaceholder())->getHandle<int8_t>();
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(result.raw(i), i);
  }
}

/// Helper to test BatchAdd using \p DTy.
template <typename DataType>
static void testBatchAdd(glow::PlaceholderBindings &bindings, glow::Module &mod,
                         glow::Function *F, glow::ExecutionEngine &EE,
                         ElemKind DTy) {
  unsigned numSlices = 10;
  auto *input = mod.createPlaceholder(DTy, {numSlices, 10, 10}, "input", false);
  auto *slice = mod.createPlaceholder(DTy, {10, 10}, "slice", false);

  bindings.allocate(input)->getHandle<DataType>().randomize(-10.0, 10.0,
                                                            mod.getPRNG());
  bindings.allocate(slice)->getHandle<DataType>().randomize(-10.0, 10.0,
                                                            mod.getPRNG());

  std::vector<NodeValue> adds;
  for (dim_t i = 0; i < numSlices; i++) {
    auto *ex = F->createSlice("slice", input, {i, 0, 0}, {i + 1, 10, 10});
    auto *ba = F->createBatchedAdd("add", ex, slice);
    adds.push_back(ba);
  }

  auto *cc = F->createConcat("concat", adds, 0);

  // Remove the reference to the graph nodes to allow DCE to remove them.
  adds.clear();

  auto *result = F->createSave("save", cc);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto RH = bindings.get(result->getPlaceholder())->getHandle<DataType>();
  auto IH = bindings.get(input)->getHandle<DataType>();
  auto SH = bindings.get(slice)->getHandle<DataType>();

  // Check that batched add works as expected.
  for (dim_t i = 0; i < numSlices; i++) {
    for (dim_t j = 0; j < 10; j++) {
      for (dim_t k = 0; k < 10; k++) {
        EXPECT_NEAR(IH.at({i, j, k}) + SH.at({j, k}), RH.at({i, j, k}),
                    0.00001);
      }
    }
  }
}

/// Check that the sequence of extract-batchedadd-concat works.
TEST_P(OperatorTest, testBatchAdd_Float) {
  CHECK_IF_ENABLED();
  testBatchAdd<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Check that the sequence of extract-batchedadd-concat works.
TEST_P(OperatorTest, testBatchAdd_Float16) {
  CHECK_IF_ENABLED();
  testBatchAdd<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

static void quantizedBatchAdd(ExecutionEngine &EE,
                              PlaceholderBindings &bindings, ElemKind Ty) {
  EE.setBackendName(EE.getBackendName());
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  bindings.clear();
  unsigned numSlices = 10;
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {numSlices, 10, 10},
                                      "input", false);
  auto *slice =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 10}, "slice", false);

  bindings.allocate(input)->getHandle().randomize(-5.0, 5.0, mod.getPRNG());
  bindings.allocate(slice)->getHandle().randomize(-5.0, 5.0, mod.getPRNG());

  // Scale the numbers in the range (-5. .. 5.) to (-50 .. 50).
  auto qInType = mod.uniqueType(ElemKind::Int8QTy, {numSlices, 10, 10}, .1, 0);
  auto qSliceType2 = mod.uniqueType(Ty, {10, 10}, .1, 0);
  auto qSliceType3 = mod.uniqueType(ElemKind::Int8QTy, {1, 10, 10}, .1, 0);

  auto *intInput = F->createQuantize("qinput", input, qInType);
  auto *intSlice = F->createQuantize("qslice", slice, qSliceType2);

  std::vector<NodeValue> adds;
  for (dim_t i = 0; i < numSlices; i++) {
    auto *ex = F->createSlice("slice", intInput, {i, 0, 0}, qSliceType3);
    auto *ba = F->createBatchedAdd("add", ex, intSlice);
    adds.push_back(ba);
  }

  Node *cc = F->createConcat("concat", adds, 0, qInType);
  cc = F->createDequantize("dq", cc);
  auto *result = F->createSave("save", cc);
  bindings.allocate(result->getPlaceholder());

  // Remove the reference to the graph nodes to allow DCE to remove them.
  adds.clear();

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto RH = bindings.get(result->getPlaceholder())->getHandle();
  auto IH = bindings.get(input)->getHandle();
  auto SH = bindings.get(slice)->getHandle();

  // Check that batched add works as expected.
  for (dim_t i = 0; i < numSlices; i++) {
    for (dim_t j = 0; j < 10; j++) {
      for (dim_t k = 0; k < 10; k++) {
        EXPECT_NEAR(IH.at({i, j, k}) + SH.at({j, k}), RH.at({i, j, k}), 0.1);
      }
    }
  }
}

/// Tests quantized batched-add arithmetic.
TEST_P(OperatorTest, testQuantizedBatchAdd) {
  CHECK_IF_ENABLED();
  // Test Int8QTy Slice.
  quantizedBatchAdd(EE_, bindings_, ElemKind::Int8QTy);
  // Test Int32QTy Slice.
  quantizedBatchAdd(EE_, bindings_, ElemKind::Int32QTy);
}

TEST_P(OperatorTest, LengthsSum) {
  CHECK_IF_ENABLED();

  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 3.7],
        [3.0, 2.9],
        [1.1, 1.4],
        [2.8, 8.4],
    ]
    LENGTHS = [2, 0, 3, 1]
    OUTPUT = [
        [3.3, 4.6],
        [0.0, 0.0],
        [8.6, 8.0],
        [2.8, 8.4],
    ]
  */
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {6, 2}, "data", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings_.allocate(data)->getHandle() = {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 3.7f,
                                           3.0f, 2.9f, 1.1f, 1.4f, 2.8f, 8.4f};
  bindings_.allocate(lengths)->getHandle<int32_t>() = {2, 0, 3, 1};

  auto *R = F_->createLengthsSum("LS", data, lengths);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {4, 2});
  expected.getHandle() = {3.3f, 4.6f, 0.0f, 0.0f, 8.6f, 8.0f, 2.8f, 8.4f};

  EXPECT_TRUE(expected.isEqual(result));
}

/// Helper to test SLS using \p DTy.
template <typename DataType>
static void testSLS(glow::PlaceholderBindings &bindings, glow::Module &mod,
                    glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
                    float allowedError) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    OUTPUT = [
        [5.5, 6.9],
        [0.0, 0.0],
        [6.8, 9.1],
        [1.0, 1.2],
        [3.0, 3.6],
    ]
  */
  auto *data = mod.createPlaceholder(DTy, {3, 2}, "data", false);
  auto *indices = mod.createPlaceholder(IndexElemKind, {8}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {5}, "lengths", false);

  bindings.allocate(data)->getHandle<DataType>() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };
  bindings.allocate(indices)->getHandle<sdim_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  auto *R = F->createSparseLengthsSum("SLS", data, indices, lengths);

  auto *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {5, 2});
  expected.getHandle<DataType>() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test that SLS is correctly supported in FloatTy.
TEST_P(OperatorTest, SparseLengthsSum_Float) {
  CHECK_IF_ENABLED();
  testSLS<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001);
}

/// Test that SLS is correctly supported in Float16Ty.
TEST_P(OperatorTest, SparseLengthsSum_Float16) {
  CHECK_IF_ENABLED();
  testSLS<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.002);
}

TEST_P(OperatorTest, SparseLengthsSumI8) {
  CHECK_IF_ENABLED();

  /*
    DATA  = [
        [11, 13],
        [24, 35],
        [46, 58],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    OUTPUT = [
        [56, 70],
        [ 1,  1],
        [69, 92],
        [11, 13],
        [31, 37],
    ]
  */
  auto *data =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3, 2}, 0.1f, 1, "data", false);
  auto *indices = mod_.createPlaceholder(IndexElemKind, {8}, "indices", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {5}, "lengths", false);

  bindings_.allocate(data)->getHandle<int8_t>() = {
      11, 13, 24, 35, 46, 58,
  };
  bindings_.allocate(indices)->getHandle<sdim_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings_.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  auto *R = F_->createSparseLengthsSum("SLS", data, indices, lengths);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::Int8QTy, {5, 2}, 0.1f, 1);
  expected.getHandle<int8_t>() = {
      56, 70, 1, 1, 69, 92, 11, 13, 31, 37,
  };
  EXPECT_TRUE(expected.isEqual(result));
}

/// Test SparseLengthsWeightedSum with an N-dimension embedding table.
template <typename DataType>
static void testSLWS(glow::PlaceholderBindings &bindings, glow::Module &mod,
                     glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
                     float allowedError, size_t ndims) {
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  ShapeVector idims(ndims, 1);
  ShapeVector odims(ndims, 1);
  idims[0] = 3;
  odims[0] = 4;

  auto *data = mod.createPlaceholder(DTy, idims, "data", false);
  auto *weights = mod.createPlaceholder(DTy, {8}, "weights", false);
  auto *indices = mod.createPlaceholder(IndexElemKind, {8}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings.allocate(data)->getHandle<DataType>() = {
      2.0,
      -0.5,
      13,
  };
  bindings.allocate(weights)->getHandle<DataType>() = {
      3, 1, 0, 0, 0, 0, 2, -0.5,
  };
  bindings.allocate(indices)->getHandle<sdim_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                              lengths);
  auto *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, odims);
  expected.getHandle<DataType>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

/// Test that SLWS is correctly supported in FloatTy in 1D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_1D_Float) {
  CHECK_IF_ENABLED();
  testSLWS<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                  /* ndims */ 1);
}

/// Test that SLWS is correctly supported in FloatTy in 2D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_2D_Float) {
  CHECK_IF_ENABLED();
  testSLWS<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                  /* ndims */ 2);
}

/// Test that SLWS is correctly supported in Float16Ty in 1D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_1D_Float16) {
  CHECK_IF_ENABLED();
  testSLWS<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.0001,
                      /* ndims */ 1);
}

/// Test that SLWS is correctly supported in Float16Ty in 2D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_2D_Float16) {
  CHECK_IF_ENABLED();
  testSLWS<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.0001,
                      /* ndims */ 2);
}

TEST_P(OperatorTest, SparseLengthsWeightedSumI8) {
  CHECK_IF_ENABLED();

  /*
    DATA  =   [4, -1, 26]
    WEIGHTS = [6, 2, 0, 0, 0, 0, 4, -1]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [1, 0, 0, 50]
  */
  auto *data =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3}, 0.5, 0, "data", false);
  auto *weights =
      mod_.createPlaceholder(ElemKind::Int8QTy, {8}, 0.5, 0, "weights", false);
  auto *indices = mod_.createPlaceholder(IndexElemKind, {8}, "indices", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings_.allocate(data)->getHandle<int8_t>() = {
      4,
      -1,
      26,
  };
  bindings_.allocate(weights)->getHandle<int8_t>() = {
      6, 2, 0, 0, 0, 0, 4, -1,
  };
  bindings_.allocate(indices)->getHandle<sdim_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings_.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F_->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                               lengths);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::Int8QTy, {4}, 0.5, 0);
  expected.getHandle<int8_t>() = {
      1,
      0,
      0,
      50,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

/// Test EmbeddingBag with an N-dimension embedding table.
template <typename DataType>
static void
testEmbeddingBag(glow::PlaceholderBindings &bindings, glow::Module &mod,
                 glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
                 float allowedError, dim_t ndims, bool hasEndOffset) {
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    OFFSETS = [0, 3, 3, 6]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  ShapeVector idims(ndims, 1);
  ShapeVector odims(ndims, 1);
  idims[0] = 3;
  odims[0] = 4;

  auto *data = mod.createPlaceholder(DTy, idims, "data", false);

  bindings.allocate(data)->getHandle<DataType>() = {
      2.0,
      -0.5,
      13,
  };

  // If hasEndOffset then add some additional junk to the end of indices and
  // weights and an extra offset to offsets.
  Placeholder *weights;
  Placeholder *indices;
  Placeholder *offsets;
  if (hasEndOffset) {
    weights = mod.createPlaceholder(DTy, {10}, "weights", false);
    indices = mod.createPlaceholder(IndexElemKind, {10}, "indices", false);
    offsets = mod.createPlaceholder(IndexElemKind, {5}, "offsets", false);

    bindings.allocate(weights)->getHandle<DataType>() = {
        3, 1, 0, 0, 0, 0, 2, -0.5, 42.0, 42.0,
    };
    bindings.allocate(indices)->getHandle<sdim_t>() = {
        1, 0, 2, 0, 1, 2, 2, 0, 13, 10,
    };
    bindings.allocate(offsets)->getHandle<sdim_t>() = {
        0, 3, 3, 6,
        8, // extra end offset
    };

  } else {
    weights = mod.createPlaceholder(DTy, {8}, "weights", false);
    indices = mod.createPlaceholder(IndexElemKind, {8}, "indices", false);
    offsets = mod.createPlaceholder(IndexElemKind, {4}, "offsets", false);

    bindings.allocate(weights)->getHandle<DataType>() = {
        3, 1, 0, 0, 0, 0, 2, -0.5,
    };
    bindings.allocate(indices)->getHandle<sdim_t>() = {
        1, 0, 2, 0, 1, 2, 2, 0,
    };
    bindings.allocate(offsets)->getHandle<sdim_t>() = {
        0,
        3,
        3,
        6,
    };
  }

  auto *R = F->createEmbeddingBag("EB", data, weights, indices, offsets,
                                  hasEndOffset);
  auto *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, odims);
  expected.getHandle<DataType>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test that EB is correctly supported in FloatTy in 1D.
TEST_P(OperatorTest, EmbeddingBag_1D_Float) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 1, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in FloatTy in 1D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_1D_Float_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 1, /* hasEndOffset */ true);
}

/// Test that EB is correctly supported in FloatTy in 2D.
TEST_P(OperatorTest, EmbeddingBag_2D_Float) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 2, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in FloatTy in 2D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_2D_Float_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 2, /* hasEndOffset */ true);
}

/// Test that EB is correctly supported in Float16Ty in 1D.
TEST_P(OperatorTest, EmbeddingBag_1D_Float16) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              0.0001,
                              /* ndims */ 1, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in Float16Ty in 1D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_1D_Float16_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              0.0001,
                              /* ndims */ 1, /* hasEndOffset */ true);
}

/// Test that EB is correctly supported in Float16Ty in 2D.
TEST_P(OperatorTest, EmbeddingBag_2D_Float16) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              0.0001,
                              /* ndims */ 2, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in Float16Ty in 2D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_2D_Float16_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              0.0001,
                              /* ndims */ 2, /* hasEndOffset */ true);
}

/// Helper to test EmbeddingBagByteRowwiseOffsets using \p DTy.
template <typename DataType>
static void testEmbeddingBagByteRowwiseOffsets(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind fusedDTy, float allowedError,
    bool useFP16Accumulation, bool hasEndOffset) {
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    OFFSETS = [0, 3, 3, 6]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  const bool fusedData = isFusedQuantizedElemKind(fusedDTy);
  const ElemKind DTy =
      fusedData ? getScaleOffsetElemKindFromFused(fusedDTy) : fusedDTy;
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  // If hasEndOffset then add some additional junk to the end of indices and
  // weights and an extra offset to offsets.
  Constant *weights;
  Placeholder *indices;
  Placeholder *offsets;
  if (hasEndOffset) {
    weights = mod.createConstant(DTy, {10}, "weights");
    weights->getPayloadMutable().getHandle<DataType>() = {
        3., 1., 0., 0., 0., 0., 2., -0.5, 42.0, 35.0,
    };

    indices = mod.createPlaceholder(IndexElemKind, {10}, "indices",
                                    /* isTrainable */ false);
    offsets = mod.createPlaceholder(IndexElemKind, {5}, "offsets",
                                    /* isTrainable */ false);

    bindings.allocate(indices)->getHandle<sdim_t>() = {1, 0, 2, 0, 1,
                                                       2, 2, 0, 1, 5};
    bindings.allocate(offsets)->getHandle<sdim_t>() = {
        0, 3, 3, 6,
        8, // extra end offset
    };
  } else {
    weights = mod.createConstant(DTy, {8}, "weights");
    weights->getPayloadMutable().getHandle<DataType>() = {
        3., 1., 0., 0., 0., 0., 2., -0.5,
    };

    indices = mod.createPlaceholder(IndexElemKind, {8}, "indices",
                                    /* isTrainable */ false);
    offsets = mod.createPlaceholder(IndexElemKind, {4}, "offsets",
                                    /* isTrainable */ false);

    bindings.allocate(indices)->getHandle<sdim_t>() = {
        1, 0, 2, 0, 1, 2, 2, 0,
    };
    bindings.allocate(offsets)->getHandle<sdim_t>() = {
        0,
        3,
        3,
        6,
    };
  }

  auto *R = F->createEmbeddingBagByteRowwiseOffsets(
      "EBBRO", data, weights, indices, offsets, fusedDTy, useFP16Accumulation,
      hasEndOffset);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {4, 1});
  expected.getHandle<DataType>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test EmbeddingBagByteRowwiseOffsets in Float.
TEST_P(OperatorTest, EmbeddingBagByteRowwiseOffsets_Float) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ false);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float with end offset.
TEST_P(OperatorTest, EmbeddingBagByteRowwiseOffsets_Float_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ true);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float accumulation.
TEST_P(OperatorTest, EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ false);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float accumulation. Has
/// end offset.
TEST_P(OperatorTest,
       EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ true);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float16 accumulation.
TEST_P(OperatorTest, EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ true, /* hasEndOffset */ false);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float16 accumulation.
/// Has end offset.
TEST_P(OperatorTest,
       EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ true, /* hasEndOffset */ true);
}

/// Helper to test RowwiseQuantizedSparseLengthsWeightedSum using \p DTy.
template <typename DataType>
static void testRowwiseQuantizedSparseLengthsWeightedSum(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind DTy, float allowedError,
    bool useFP16Accumulation = false) {
  /*
    DATA  =   [2.0, -0.5, 13]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  Tensor data(ElemKind::FloatTy, {3});
  data.getHandle<float>() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(DTy, {8}, "weights");
  weights->getPayloadMutable().getHandle<DataType>() = {
      3., 1., 0., 0., 0., 0., 2., -0.5,
  };

  Placeholder *indices = mod.createPlaceholder(IndexElemKind, {8}, "indices",
                                               /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F->createRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths,
      quantization::Schema::Asymmetric, DTy, useFP16Accumulation);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {4});
  expected.getHandle<DataType>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test RWQ-SLWS with Float Weights, Scales, Offsets, and Output.
TEST_P(OperatorTest, RowwiseQuantizedSparseLengthsWeightedSum_Float) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsWeightedSum<float>(
      bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001);
}

/// Test RWQ-SLWS with Float16 Weights, Scales, Offsets, and Output. Uses Float
/// accumulation.
TEST_P(OperatorTest,
       RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsWeightedSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.0001,
      /* useFP16Accumulation */ false);
}

/// Test RWQ-SLWS with Float16 Weights, Scales, Offsets, and Output. Uses
/// Float16 accumulation.
TEST_P(OperatorTest,
       RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsWeightedSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.0001,
      /* useFP16Accumulation */ true);
}

static FunctionTensorPair
createAndInitRWQSLWSAllSame(glow::PlaceholderBindings &bindings,
                            glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Tensor data(ElemKind::FloatTy, {20, 2});
  data.getHandle<float>() = {
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
  };

  Constant *weights = mod.createConstant(ElemKind::FloatTy, {21}, "weights");
  weights->getPayloadMutable().getHandle<float>() = {
      0.44419134, 0.3419154,  0.28775468, 0.47224975, 0.05422213, 0.14346851,
      0.05846643, 0.3750175,  0.09190885, 0.3335992,  0.09665264, 0.4560224,
      0.2244578,  0.44881952, 0.42696562, 0.33007848, 0.4511249,  0.11568925,
      0.02629679, 0.33864713, 0.42614424};

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {21}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {2}, "lengths",
                            /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<int64_t>() = {
      11, 8, 19, 8, 4, 11, 4, 19, 6, 18, 2, 6, 15, 5, 14, 14, 15, 13, 4, 6, 5,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {15, 6};

  auto *R = F->createRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths,
      quantization::Schema::Asymmetric, ElemKind::FloatTy,
      /* useFP16Accumulation */ false);
  SaveNode *S = F->createSave("save", R);
  Tensor *resultT = bindings.allocate(S->getPlaceholder());

  return std::make_pair(F, resultT);
}

TEST_P(OperatorTest, RWQSLWSAllSame_Float16_AccumFP16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitRWQSLWSAllSame, ElemKind::Float16Ty,
      ElemKind::Float16Ty, 1e-6, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      /*schema */ quantization::Schema::Asymmetric,
      /* biasElemKind */ ElemKind::Int32QTy, /* forceFP16AccumSLS */ true);
}

TEST_P(OperatorTest, RWQSLWSAllSame_Float16_AccumFP32) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitRWQSLWSAllSame, ElemKind::Float16Ty,
      ElemKind::Float16Ty, 1e-6, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      /*schema */ quantization::Schema::Asymmetric,
      /* biasElemKind */ ElemKind::Int32QTy, /* forceFP16AccumSLS */ false);
}

/// Helper to test RowwiseQuantizedSparseLengthsWeightedSum using \p DTy.
template <typename DataType>
static void testRowwiseQuantizedSparseLengthsSum(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind DTy, float allowedError,
    bool useFP16Accumulation = false) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    OUTPUT = [
        [5.5, 6.9],
        [0.0, 0.0],
        [6.8, 9.1],
        [1.0, 1.2],
        [3.0, 3.6],
    ]
  */
  Tensor data(ElemKind::FloatTy, {3, 2});
  data.getHandle() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };

  Placeholder *indices = mod.createPlaceholder(IndexElemKind, {8}, "indices",
                                               /* isTrainable */ false);
  Placeholder *lengths = mod.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengths", /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  auto *R = F->createRowwiseQuantizedSparseLengthsSum(
      "RQSLWS", data, indices, lengths, quantization::Schema::Asymmetric, DTy,
      useFP16Accumulation);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {5, 2});
  expected.getHandle<DataType>() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test RWQ-SLS with Float Weights, Scales, Offsets, and Output.
TEST_P(OperatorTest, RowwiseQuantizedSparseLengthsSum_Float) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsSum<float>(bindings_, mod_, F_, EE_,
                                              ElemKind::FloatTy, 0.015);
}

/// Test RWQ-SLS with Float16 Weights, Scales, Offsets, and Output. Uses
/// Float accumulation.
TEST_P(OperatorTest, RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.02,
      /* useFP16Accumulation */ false);
}

/// Test RWQ-SLS with Float16 Weights, Scales, Offsets, and Output. Uses
/// Float16 accumulation.
TEST_P(OperatorTest, RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.02,
      /* useFP16Accumulation */ true);
}

TEST_P(OperatorTest, RepeatedSLSWithPartialTensors) {
  CHECK_IF_ENABLED();

  // This test is only meaningful if the backend supports partial tensors.
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());

  constexpr size_t embeddingRows = 1275;
  constexpr size_t numLengths = 20;
  constexpr size_t maxIndices = 20000;
  constexpr size_t numIndices = 20; // Must be less than sum(lengths).
  constexpr size_t iterations = 33;

  auto *data =
      mod_.createConstant(ElemKind::FloatTy, {embeddingRows, 1}, "data");
  data->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                         mod_.getPRNG());
  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {maxIndices}, "indices", false);
  auto *lengths = mod_.createPlaceholder(ElemKind::Int32ITy, {numLengths},
                                         "lengths", false);
  auto *SLS = F_->createSparseLengthsSum("SLS", data, indices, lengths);
  auto *save = F_->createSave("save", SLS);
  auto *outPH = save->getPlaceholder();
  EE_.compile(CompilationMode::Infer);

  Tensor indicesReal(IndexElemKind, {numIndices});
  indicesReal.getHandle<sdim_t>().randomize(0, embeddingRows - 1,
                                            mod_.getPRNG());
  Tensor indicesPartial(indicesReal.getUnsafePtr(), indices->getType(),
                        indicesReal.getSizeInBytes());
  Tensor indicesPadded(indices->getType());
  indicesPadded.zero();
  memcpy(indicesPadded.getUnsafePtr(), indicesReal.getUnsafePtr(),
         numIndices * sizeof(int64_t));

  Tensor lengthsReal(ElemKind::Int32ITy, {numLengths});
  lengthsReal.getHandle<int32_t>().clear(1);
  Tensor lengthsPartial(lengthsReal.getUnsafePtr(), lengths->getType(),
                        lengthsReal.getSizeInBytes());
  Tensor lengthsPadded(ElemKind::Int32ITy, {numLengths});
  lengthsPadded.assign(&lengthsReal);

  bindings_.insert(indices, std::move(indicesPartial));
  bindings_.insert(lengths, std::move(lengthsPartial));
  bindings_.allocate(outPH);

  PlaceholderBindings paddedBindings;
  paddedBindings.insert(indices, std::move(indicesPadded));
  paddedBindings.insert(lengths, std::move(lengthsPadded));
  paddedBindings.allocate(outPH);

  for (size_t i = 0; i < iterations; i++) {
    EE_.run(bindings_);
    EE_.run(paddedBindings);
    ASSERT_TRUE(bindings_.get(outPH)->isEqual(*paddedBindings.get(outPH)));
  }
}

/// Helper to test gathers using partial inputs using \p ITy.
template <typename IndicesType>
static void testPartialGather(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind ITy) {
  /*
    The acutal input we care about has the following shape/result:

    DATA  = [1.0, 2.3, 4.5]
    INDICES = [0, 1, 0, 1, 2, 0]
    OUTPUT = [1.0, 2.3, 1.0, 2.3, 4.5, 1.0]

    However, we are going to create a larger INDICES input that is only
    partially filled, and expect a larger OUTPUT that we expect will have data
    we do not care about.
  */

  Placeholder *data = mod.createPlaceholder(ElemKind::FloatTy, {3}, "data",
                                            /* isTrainable */ false);
  Placeholder *indices =
      mod.createPlaceholder(ITy, {10000}, "indices", /* isTrainable */ false);

  bindings.allocate(data)->getHandle<float>() = {1.0f, 2.3f, 4.5f};

  Tensor indicesReal(ITy, {6});
  indicesReal.getHandle<IndicesType>() = {0, 1, 0, 1, 2, 0};
  Tensor indicesPartial(indicesReal.getUnsafePtr(), indices->getType(),
                        indicesReal.getSizeInBytes());
  bindings.insert(indices, std::move(indicesPartial));

  auto *R = F->createGather("gather", data, indices);

  auto *result = F->createSave("save", R);
  Tensor *resultT = bindings.allocate(result->getPlaceholder());

  // Result should be 10000, even though we only care about the first 6 results.
  EXPECT_EQ(resultT->getType().dims().size(), 1);
  EXPECT_EQ(resultT->getType().dims()[0], 10000);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expectedT(ElemKind::FloatTy, {6});
  auto expectedH = expectedT.getHandle<float>();
  expectedH = {1.0, 2.3, 1.0, 2.3, 4.5, 1.0};
  auto resultH = resultT->getHandle<float>();

  for (dim_t i = 0; i < 6; ++i) {
    EXPECT_EQ(expectedH.at({i}), resultH.at({i}));
  }
}

TEST_P(OperatorTest, GatherWithInt64PartialTensors) {
  CHECK_IF_ENABLED();
  // This test is only meaningful if the backend supports partial tensors.
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());
  testPartialGather<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

TEST_P(OperatorTest, GatherWithInt32PartialTensors) {
  CHECK_IF_ENABLED();
  // This test is only meaningful if the backend supports partial tensors.
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());
  testPartialGather<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Helper to test FusedRowwiseQuantizedSparseLengthsWeightedSum using \p DTy.
template <typename DataType>
static void testFusedRowwiseQuantizedSparseLengthsWeightedSum(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind fusedDTy, float allowedError,
    bool useFP16Accumulation = false) {
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [[0.5, 0, 0, 25]]
  */
  const bool fusedData = isFusedQuantizedElemKind(fusedDTy);
  const ElemKind DTy =
      fusedData ? getScaleOffsetElemKindFromFused(fusedDTy) : fusedDTy;
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(DTy, {8}, "weights");
  weights->getPayloadMutable().getHandle<DataType>() = {
      3., 1., 0., 0., 0., 0., 2., -0.5,
  };

  Placeholder *indices = mod.createPlaceholder(IndexElemKind, {8}, "indices",
                                               /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths, fusedDTy, useFP16Accumulation);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {4, 1});
  expected.getHandle<DataType>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test Fused-RWQ-SLWS in Float.
TEST_P(OperatorTest, FusedRowwiseQuantizedSparseLengthsWeightedSum_Float) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsWeightedSum<float>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, 0.0001);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsWeightedSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ false);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsWeightedSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ true);
}

static void testRowwiseQuantizedSparseLengthsSum_ConvertedFloat16(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, float allowedError, bool convertFusedToFP16,
    bool useFP16AccumSLS) {
  CHECK_IF_ENABLED();
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [[0.5, 0, 0, 25]]
  */
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(ElemKind::FloatTy, {8}, "weights");
  weights->getPayloadMutable().getHandle<float>() = {
      3., 1., 0., 0., 0., 0., 2., -0.5,
  };

  Placeholder *indices = mod.createPlaceholder(IndexElemKind, {8}, "indices",
                                               /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = convertFusedToFP16;
  cctx.precisionConfig.forceFP16AccumSLS = useFP16AccumSLS;
  EE.compile(cctx);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {4, 1});
  expected.getHandle<float>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test Fused-RWQ-SLWS in where the weights are in Fp16, data
/// inputs are UInt8FusedQTy.
TEST_P(
    OperatorTest,
    FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_NoFusedConvert) {
  CHECK_IF_ENABLED();
  return testRowwiseQuantizedSparseLengthsSum_ConvertedFloat16(
      bindings_, mod_, F_, EE_, 0.02,
      /* convertFusedToFP16*/ false, /* useFP16AccumSLS */ true);
}

TEST_P(
    OperatorTest,
    FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_NoFusedConvert_FP32Accum) {
  CHECK_IF_ENABLED();
  return testRowwiseQuantizedSparseLengthsSum_ConvertedFloat16(
      bindings_, mod_, F_, EE_, 0.02,
      /* convertFusedToFP16*/ false, /* useFP16AccumSLS */ false);
}

TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16) {
  CHECK_IF_ENABLED();
  return testRowwiseQuantizedSparseLengthsSum_ConvertedFloat16(
      bindings_, mod_, F_, EE_, 0.02,
      /* convertFusedToFP16*/ true, /* useFP16AccumSLS */ true);
}

TEST_P(
    OperatorTest,
    FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_back) {
  CHECK_IF_ENABLED();
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [1]
    INDICES = [0]
    LENGTHS = [0, 0, 0, 1] and then [1, 0, 0, 0]
    OUTPUT =  [[0, 0, 0, 0.2]] and then [[2.0, 0, 0, 0]]
  */
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod_.createConstant(ElemKind::FloatTy, {1}, "weights");
  weights->getPayloadMutable().getHandle<float>() = {1.};

  Placeholder *indices = mod_.createPlaceholder(IndexElemKind, {1}, "indices",
                                                /* isTrainable */ false);
  Placeholder *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                             /* isTrainable */ false);

  bindings_.allocate(indices)->getHandle<sdim_t>() = {
      0,
  };
  bindings_.allocate(lengths)->getHandle<int32_t>() = {
      0,
      0,
      0,
      1,
  };

  auto *R = F_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths);
  SaveNode *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;
  EE_.compile(cctx);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {4, 1});
  expected.getHandle<float>() = {
      0,
      0,
      0,
      2.0,
  };

  EXPECT_TRUE(expected.isEqual(result, 0.02));

  // Send another inference
  bindings_.get(lengths)->getHandle<int32_t>() = {
      1,
      0,
      0,
      0,
  };
  EE_.run(bindings_);

  Tensor &result1 = *bindings_.get(S->getPlaceholder());
  Tensor expected1(ElemKind::FloatTy, {4, 1});
  expected1.getHandle<float>() = {
      2.0,
      0,
      0,
      0,
  };
  EXPECT_TRUE(expected1.isEqual(result1, 0.02));
}

TEST_P(
    OperatorTest,
    FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_back2) {
  CHECK_IF_ENABLED();

  Tensor data(ElemKind::FloatTy, {10000, 64});
  data.getHandle().randomize(-1, 1, mod_.getPRNG());

  Placeholder *weights =
      mod_.createPlaceholder(ElemKind::FloatTy, {10000}, "weights",
                             /* isTrainable */ false);

  Placeholder *indices =
      mod_.createPlaceholder(IndexElemKind, {10000}, "indices",
                             /* isTrainable */ false);
  Placeholder *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {32}, "lengths",
                             /* isTrainable */ false);

  Tensor *wT = bindings_.allocate(weights);
  wT->zero();
  wT->getHandle<float>().at({0}) = 4.18067;

  Tensor *iT = bindings_.allocate(indices);
  iT->zero();
  iT->getHandle<sdim_t>().at({0}) = 4124;

  bindings_.allocate(lengths)->getHandle<int32_t>() = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};

  auto *R = F_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths);
  SaveNode *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;
  EE_.compile(cctx);
  EE_.run(bindings_);

  // This is the result for the first inference. We expect the result in the
  // second last row or raw location 30 * 64 to 31 * 64 -1. The rest of the rows
  // should be all 0.
  Tensor &result = *bindings_.get(S->getPlaceholder());

  // Send another inference
  result.zero();
  // set new indices.
  iT = bindings_.get(indices);
  iT->zero();
  iT->getHandle<sdim_t>().at({0}) = 1256;
  // set new lengths.
  bindings_.get(lengths)->getHandle<int32_t>() = {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

  };
  EE_.run(bindings_);

  // We now expect the second to last row to be all 0.
  Tensor &result1 = *bindings_.get(S->getPlaceholder());
  float *d = reinterpret_cast<float *>(result1.getUnsafePtr());
  for (size_t i = 30 * 64; i < 31 * 64; ++i) {
    EXPECT_EQ(0, d[i]);
  }
}

/// Helper to test FusedRowwiseQuantizedSparseLengthsSum using \p fusedDTy.
template <typename DataType>
static void testFusedRowwiseQuantizedSparseLengthsSum(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind fusedDTy, float allowedError,
    bool useFP16Accumulation = false) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    OUTPUT = [
        [5.5, 6.9],
        [0.0, 0.0],
        [6.8, 9.1],
        [1.0, 1.2],
        [3.0, 3.6],
    ]
  */
  const bool fusedData = isFusedQuantizedElemKind(fusedDTy);
  const ElemKind DTy =
      fusedData ? getScaleOffsetElemKindFromFused(fusedDTy) : fusedDTy;

  Tensor data(ElemKind::FloatTy, {3, 2});
  data.getHandle() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };

  Placeholder *indices = mod.createPlaceholder(IndexElemKind, {8}, "indices",
                                               /* isTrainable */ false);
  Placeholder *lengths = mod.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengths", /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  auto *R = F->createFusedRowwiseQuantizedSparseLengthsSum(
      "RQSLWS", data, indices, lengths, fusedDTy, useFP16Accumulation);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {5, 2});
  expected.getHandle<DataType>() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test Fused-RWQ-SLS in Float.
TEST_P(OperatorTest, FusedRowwiseQuantizedSparseLengthsSum_Float) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsSum<float>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, 0.015);
}

/// Test Fused-RWQ-SLS in Float16. Uses Float accumulation.
TEST_P(OperatorTest, FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.02,
      /* useFP16Accumulation */ false);
}

/// Test Fused-RWQ-SLS in Float16. Uses Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.02,
      /* useFP16Accumulation */ true);
}

/// Test Fused-RWQ-SLS in Float16 wth 4-bit quantization for the embedding. Uses
/// Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsSum_Fused4Bit_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt4FusedFP16QTy, 0.15,
      /* useFP16Accumulation */ true);
}

/// Helper to test SLWS with more columns in data input, with precision \p DTy,
/// and precision for data \p dataDTy.
template <typename DataType>
static void testSLWSTwoColumn(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind dataDTy,
                              float allowedError,
                              bool useFP16Accumulation = false) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    WEIGHTS = [1, -1, 1.5, 0.5, -1.5, 2, -2, -0.5]
    OUTPUT = [
        [3.5, 4.5],
        [0.0, 0.0],
        [5.7, 7.95],
        [-1.5, -1.8],
        [-0.5, -0.6],
    ]
  */
  const bool fusedData = isFusedQuantizedElemKind(dataDTy);
  const ElemKind DTy =
      fusedData ? getScaleOffsetElemKindFromFused(dataDTy) : dataDTy;

  Tensor data(fusedData ? ElemKind::FloatTy : DTy, {3, 2});
#define floatData                                                              \
  { 1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f, }
  if (fusedData) {
    data.getHandle<float>() = floatData;
  } else {
    data.getHandle<DataType>() = floatData;
  }

  Placeholder *indices = mod.createPlaceholder(IndexElemKind, {8}, "indices",
                                               /* isTrainable */ false);
  Placeholder *lengths = mod.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengths", /* isTrainable */ false);
  Placeholder *weights =
      mod.createPlaceholder(DTy, {8}, "weights", /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<sdim_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };
  bindings.allocate(weights)->getHandle<DataType>() = {
      1, -1, 1.5, 0.5, -1.5, 2, -2, -0.5,
  };

  Node *SLWS = nullptr;
  if (fusedData) {
    SLWS = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "RQSLWS", data, weights, indices, lengths, dataDTy,
        useFP16Accumulation);
  } else {
    Placeholder *dataP =
        mod.createPlaceholder(&data.getType(), "data", /* isTrainable */ false);
    bindings.insert(dataP, std::move(data));
    SLWS = F->createSparseLengthsWeightedSum("SLWS", dataP, weights, indices,
                                             lengths);
  }
  SaveNode *S = F->createSave("save", SLWS);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {5, 2});
  expected.getHandle<DataType>() = {
      3.5, 4.5, 0.0, 0.0, 5.7, 7.95, -1.5, -1.8, -0.5, -0.6,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test SLWS in Float.
TEST_P(OperatorTest, SLWSTwoColumn_Float) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001);
}

/// Test SLWS in Float16.
TEST_P(OperatorTest, SLWSTwoColumn_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                               0.005,
                               /* useFP16Accumulation */ false);
}

/// Test Fused-RWQ-SLWS in Float.
TEST_P(OperatorTest, FusedRowwiseQuantizedSLWSTwoColumn_Float) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float>(bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy,
                           0.015);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float accumulation.
TEST_P(OperatorTest, FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float16_t>(bindings_, mod_, F_, EE_,
                               ElemKind::UInt8FusedFP16QTy, 0.015,
                               /* useFP16Accumulation */ false);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float16 accumulation.
TEST_P(OperatorTest, FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float16_t>(bindings_, mod_, F_, EE_,
                               ElemKind::UInt8FusedFP16QTy, 0.015,
                               /* useFP16Accumulation */ true);
}

/// Test Fused-RWQ-SLWS in Float16 wth 4-bit quantization for the embedding.
/// Uses Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSLWSTwoColumn_Fused4Bit_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float16_t>(bindings_, mod_, F_, EE_,
                               ElemKind::UInt4FusedFP16QTy, 0.1,
                               /* useFP16Accumulation */ true);
}

/// Test SLS when some input tensors are constants.
TEST_P(OperatorTest, ConstantSLS) {
  CHECK_IF_ENABLED();

  auto *data = mod_.createConstant(ElemKind::FloatTy, {1024, 32}, "data");
  auto *indices = mod_.createConstant(IndexElemKind, {314}, "indices");
  auto *lengths = mod_.createConstant(ElemKind::Int32ITy, {20}, "lengths");

  // data
  auto DH = data->getPayload().getHandle();
  for (dim_t i = 0; i < 1024; i++) {
    for (dim_t j = 0; j < 32; j++) {
      DH.at({i, j}) = (float)i;
    }
  }

  // indices
  auto IH = indices->getHandle<sdim_t>();
  std::iota(IH.begin(), IH.end(), 0);

  // lengths
  auto LH = lengths->getHandle<int32_t>();
  LH.clear(16);
  for (dim_t ldx : {1, 2, 6, 13, 14, 19}) {
    LH.at({ldx}) = 15;
  }

  auto *R = F_->createSparseLengthsSum("SLS", data, indices, lengths);
  auto *S = F_->createSave("save", R);
  auto *out = bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<float> expected = {120,  345,  570,  856,  1112, 1368, 1515,
                                 1864, 2120, 2376, 2632, 2888, 3144, 3180,
                                 3405, 3880, 4136, 4392, 4648, 4590};
  auto OH = out->getHandle();
  for (dim_t i = 0; i < 20; i++) {
    for (dim_t j = 0; j < 32; j++) {
      EXPECT_EQ(OH.at({i, j}), expected[i]);
    }
  }
}

/// Test SLS when some "lengths" inputs are zero.
TEST_P(OperatorTest, SLSWithZeroLengths) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(
      getBackendName(),
      [](PlaceholderBindings &bindings, ExecutionEngine &EE) {
        auto &mod = EE.getModule();
        auto *F = mod.createFunction("main");
        constexpr size_t embedWidth = 1000;
        Tensor data(ElemKind::FloatTy, {embedWidth, 8});
        data.getHandle().randomize(-1, 1, mod.getPRNG());
        Constant *weights =
            mod.createConstant(ElemKind::FloatTy, {3000}, "weights");
        weights->getPayloadMutable().getHandle().clear(1.0f);
        auto *indices =
            mod.createPlaceholder(IndexElemKind, {3000}, "indices", false);
        auto *lengths =
            mod.createPlaceholder(ElemKind::Int32ITy, {1000}, "lengths", false);
        bindings.allocate(indices)->getHandle<sdim_t>().randomize(
            0, embedWidth - 1, mod.getPRNG());
        auto LH = bindings.allocate(lengths)->getHandle<int32_t>();
        LH.clear(0);
        auto it = LH.begin();
        for (int i = 0; i < 13; ++i, ++it) {
          *it = 20;
        }

        auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
            "RQSLWS", data, weights, indices, lengths);
        auto *S = F->createSave("save", R);
        auto *res = bindings.allocate(S->getPlaceholder());
        return std::make_pair(F, res);
      },
      ElemKind::FloatTy, ElemKind::FloatTy);
}

/// Helper to create an SLS test with all zero lengths, with and without fused
/// rowwise quantization based on \p convertToRowwiseQuantization.
static FunctionTensorPair
createAndInitZeroLengthsSLSTest(glow::PlaceholderBindings &bindings,
                                glow::ExecutionEngine &EE,
                                bool convertToRowwiseQuantization) {
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  constexpr dim_t embedWidth = 1000;
  auto dataTy = mod.uniqueType(ElemKind::FloatTy, {embedWidth, 8});
  Tensor data(dataTy);
  data.getHandle().randomize(-1, 1, mod.getPRNG());
  Constant *weights = mod.createConstant(ElemKind::FloatTy, {3000}, "weights");
  weights->getPayloadMutable().getHandle().clear(1.0f);
  auto *indices =
      mod.createPlaceholder(IndexElemKind, {3000}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {1000}, "lengths", false);
  bindings.allocate(indices)->getHandle<sdim_t>().randomize(0, embedWidth - 1,
                                                            mod.getPRNG());
  auto LH = bindings.allocate(lengths)->getHandle<int32_t>();
  LH.clear(0);

  Node *R = nullptr;
  if (convertToRowwiseQuantization) {
    R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "RQSLWS", data, weights, indices, lengths);
  } else {
    Placeholder *dataP =
        mod.createPlaceholder(dataTy, "data", /* isTrainable */ false);
    bindings.insert(dataP, std::move(data));
    R = F->createSparseLengthsWeightedSum("SLWS", dataP, weights, indices,
                                          lengths);
  }
  auto *S = F->createSave("save", R);
  auto *res = bindings.allocate(S->getPlaceholder());
  return std::make_pair(F, res);
}

/// Test Fused RWQ-SLS when all "lengths" inputs are zero in FloatTy.
TEST_P(OperatorTest, FusedRWQSLSAllZeroLengths_Float) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(
      getBackendName(),
      std::bind(createAndInitZeroLengthsSLSTest, std::placeholders::_1,
                std::placeholders::_2, /* convertToRowwiseQuantization */ true),
      ElemKind::FloatTy, ElemKind::FloatTy);
}

/// Test Fused RWQ-SLS when all "lengths" inputs are zero in Float16Ty.
TEST_P(OperatorTest, FusedRWQSLSAllZeroLengths_Float16) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(
      getBackendName(),
      std::bind(createAndInitZeroLengthsSLSTest, std::placeholders::_1,
                std::placeholders::_2, /* convertToRowwiseQuantization */ true),

      ElemKind::Float16Ty, ElemKind::Float16Ty);
}

/// Test SLS when all "lengths" inputs are zero in FloatTy.
TEST_P(OperatorTest, SLSAllZeroLengths_Float) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(getBackendName(),
                            std::bind(createAndInitZeroLengthsSLSTest,
                                      std::placeholders::_1,
                                      std::placeholders::_2,
                                      /* convertToRowwiseQuantization */ false),
                            ElemKind::FloatTy, ElemKind::FloatTy);
}

/// Test SLS when all "lengths" inputs are zero in Float16Ty.
TEST_P(OperatorTest, SLSAllZeroLengths_Float16) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(getBackendName(),
                            std::bind(createAndInitZeroLengthsSLSTest,
                                      std::placeholders::_1,
                                      std::placeholders::_2,
                                      /* convertToRowwiseQuantization */ false),

                            ElemKind::Float16Ty, ElemKind::Float16Ty);
}

TEST_P(OperatorTest, SparseToDense) {
  CHECK_IF_ENABLED();

  // Create and initialize inputs. Make input 3D to make sure
  // multidimensional values are handled properly.
  constexpr dim_t kNumIndices = 4;
  constexpr dim_t kRows = 10;
  constexpr dim_t kCols = 5;
  constexpr dim_t kMaxIndex = 10;

  auto *indices =
      mod_.createPlaceholder(IndexElemKind, {kNumIndices}, "indices", false);
  auto *values = mod_.createPlaceholder(
      ElemKind::FloatTy, {kNumIndices, kRows, kCols}, "data", false);
  auto *dataToInferDim = mod_.createPlaceholder(ElemKind::FloatTy, {kMaxIndex},
                                                "dataToInferDim", false);

  auto IH = bindings_.allocate(indices)->getHandle<sdim_t>();
  auto VH = bindings_.allocate(values)->getHandle();

  // Duplicate one index to test that the corresponding values are added.
  IH = {1, 3, 1, 9};
  VH.randomize(-3.0, 3.0, mod_.getPRNG());

  auto *STDN = F_->createSparseToDense("STDN", indices, values, dataToInferDim);
  auto *S = F_->createSave("save", STDN);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());

  // Compute expected output.
  Tensor expected(ElemKind::FloatTy, {kMaxIndex, kRows, kCols});
  auto EH = expected.getHandle();

  expected.zero();
  for (dim_t i = 0; i < kNumIndices; ++i) {
    dim_t idx = IH.at({i});
    for (dim_t j = 0; j < kRows; ++j) {
      for (dim_t k = 0; k < kCols; ++k) {
        EH.at({idx, j, k}) += VH.at({i, j, k});
      }
    }
  }

  EXPECT_TRUE(expected.isEqual(result));
}

TEST_P(OperatorTest, SparseToDenseMask1) {
  CHECK_IF_ENABLED();

  /*
    INDICES = [4, 42, 13, 0, 100, 13]
    VALUES = [-5.5, 0.7, 11, 1e6, 2, 3.5]
    DEFAULTVALUE = 1.1
    LENGTHS = [4, 2]
    MASK = [2, 1, 0, 13, 42, 43]
    OUTPUT =  [[1.1, 1.1, 1e6, 11, 0.7, 1.1], [1.1, 1.1, 1.1, 3.5, 1.1, 1.1]]
  */
  auto *indices = mod_.createPlaceholder(IndexElemKind, {6}, "indices", false);
  auto *values =
      mod_.createPlaceholder(ElemKind::FloatTy, {6}, "values", false);
  auto *defaultValue =
      mod_.createPlaceholder(ElemKind::FloatTy, {}, "default_value", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {2}, "lengths", false);
  std::vector<dim_t> mask{2, 1, 0, 13, 42, 43};

  bindings_.allocate(indices)->getHandle<sdim_t>() = {4, 42, 13, 0, 100, 13};
  bindings_.allocate(values)->getHandle<float>() = {-5.5, 0.7, 11, 1e6, 2, 3.5};
  bindings_.allocate(defaultValue)->getHandle<float>().raw(0) = 1.1;
  bindings_.allocate(lengths)->getHandle<int32_t>() = {4, 2};

  auto *R = F_->createSparseToDenseMask("STDM", indices, values, defaultValue,
                                        lengths, mask);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {2, 6});
  expected.getHandle<float>() = {
      1.1, 1.1, 1e6, 11, 0.7, 1.1, 1.1, 1.1, 1.1, 3.5, 1.1, 1.1,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

TEST_P(OperatorTest, SparseToDenseMask2) {
  CHECK_IF_ENABLED();

  /*
    INDICES = [300, 100, 101, 299]
    VALUES = [[[-0.1, -0.2], [-0.3, -0.4]], [[2, -2], [2, 9]],
              [[15, 4.2], [10.3, 30.4]], [[0, 2], [3, 4.4]]]
    DEFAULTVALUE = [[0.1, 0.2], [0.3, 0.4]]
    LENGTHS = []
    MASK = [100, 300, 1]
    OUTPUT =  [[[2, -2], [2, 9]], [[-0.1, -0.2], [-0.3, -0.4]],
               [[0.1, 0.2], [0.3, 0.4]]]
  */
  auto *indices = mod_.createPlaceholder(IndexElemKind, {4}, "indices", false);
  auto *values =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 2, 2}, "values", false);
  auto *defaultValue =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "default_value", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {}, "lengths", false);
  std::vector<dim_t> mask{100, 300, 1};

  bindings_.allocate(indices)->getHandle<sdim_t>() = {300, 100, 101, 299};
  bindings_.allocate(values)->getHandle<float>() = {
      -0.1, -0.2, -0.3, -0.4, 2, -2, 2, 9, 15, 4.2, 10.3, 30.4, 0, 2, 3, 4.4};
  bindings_.allocate(defaultValue)->getHandle<float>() = {0.1, 0.2, 0.3, 0.4};
  bindings_.allocate(lengths)->getHandle<int32_t>() = {4};

  auto *R = F_->createSparseToDenseMask("STDM", indices, values, defaultValue,
                                        lengths, mask);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {3, 2, 2});
  expected.getHandle<float>() = {
      2, -2, 2, 9, -0.1, -0.2, -0.3, -0.4, 0.1, 0.2, 0.3, 0.4,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

TEST_P(OperatorTest, FP16Reshape) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::Float16Ty, {20, 13}, "A", false);
  auto inputHandle = bindings_.allocate(A)->getHandle<float16_t>();
  inputHandle.randomize(-3.0, 3.0, mod_.getPRNG());

  auto *tr = F_->createReshape("tr", A, {13, 20, 1});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputHandle =
      bindings_.get(result->getPlaceholder())->getHandle<float16_t>();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

/// Verify that the Reshape operator works correctly.
TEST_P(OperatorTest, Reshape) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {5, 7}, "A", false);
  auto inputHandle = bindings_.allocate(A)->getHandle();
  inputHandle.randomize(-3.0, 3.0, mod_.getPRNG());

  auto *RN = F_->createReshape("reshape", A, {7, 5, 1});
  auto *result = F_->createSave("saveReshape", RN);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputHandle = bindings_.get(result->getPlaceholder())->getHandle();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  ASSERT_EQ(outputHandle.dims().size(), 3);
  EXPECT_EQ(outputHandle.dims()[0], 7);
  EXPECT_EQ(outputHandle.dims()[1], 5);
  EXPECT_EQ(outputHandle.dims()[2], 1);

  // Check values are still in the same order.
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

/// Verify that the Reshape operator works correctly with Int64ITy..
TEST_P(OperatorTest, ReshapeInt) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::Int64ITy, {5, 7}, "A", false);
  auto inputHandle = bindings_.allocate(A)->getHandle<int64_t>();
  inputHandle.randomize<int64_t>(0, 100, mod_.getPRNG());

  auto *RN = F_->createReshape("reshape", A, {7, 5, 1});
  auto *result = F_->createSave("saveReshape", RN);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputHandle =
      bindings_.get(result->getPlaceholder())->getHandle<int64_t>();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  ASSERT_EQ(outputHandle.dims().size(), 3);
  EXPECT_EQ(outputHandle.dims()[0], 7);
  EXPECT_EQ(outputHandle.dims()[1], 5);
  EXPECT_EQ(outputHandle.dims()[2], 1);

  // Check values are still in the same order.
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

/// Verify that the Select operator works correctly.
TEST_P(OperatorTest, Select) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::BoolTy, {5}, "A", false);
  bindings_.allocate(A)->getHandle<bool>() = {false, true, true, false, false};

  auto SNTy = mod_.uniqueType(ElemKind::FloatTy, {5});
  SplatNode *SN10 = F_->createSplat("zero", SNTy, 10.0);
  SplatNode *SN20 = F_->createSplat("zero", SNTy, 20.0);

  auto *SN = F_->createSelect("select", A, SN10, SN20);
  auto *result = F_->createSave("saveSelect", SN);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resH = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_EQ(resH.at({0}), 20.0);
  EXPECT_EQ(resH.at({1}), 10.0);
  EXPECT_EQ(resH.at({2}), 10.0);
  EXPECT_EQ(resH.at({3}), 20.0);
  EXPECT_EQ(resH.at({4}), 20.0);
}

/// Verify that the CmpLTE operator works correctly.
TEST_P(OperatorTest, CmpLTE) {
  CHECK_IF_ENABLED();

  Constant *A = mod_.createConstant(ElemKind::FloatTy, {5}, "A");
  Constant *B = mod_.createConstant(ElemKind::FloatTy, {5}, "B");
  A->getPayloadMutable().getHandle<float>() = {0.0, 1.0, 2.0, 3.0, 4.0};
  B->getPayloadMutable().getHandle<float>() = {0.0, 1.1, 1.5, 10.1, -1.0};

  auto *CMPLTE = F_->createCmpLTE("select", A, B);
  auto *result = F_->createSave("saveCMPLTE", CMPLTE);
  Tensor *resultT = bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resH = resultT->getHandle<bool>();
  EXPECT_TRUE(resH.at({0}));
  EXPECT_TRUE(resH.at({1}));
  EXPECT_FALSE(resH.at({2}));
  EXPECT_TRUE(resH.at({3}));
  EXPECT_FALSE(resH.at({4}));
}

/// Helper to test SliceReshape using \p DTy.
template <typename DataType>
static void testSliceReshape(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *X =
      createPlaceholderConditionallyQuantized(mod, DTy, {3, 3}, "X", false);

  auto XH = bindings.allocate(X)->getHandle<DataType>();
  for (dim_t i = 0; i < 3; i++) {
    for (dim_t j = 0; j < 3; j++) {
      XH.at({i, j}) = i * 3 + j;
    }
  }

  // Do an assortment of slices/reshapes stacked on top of each other.
  auto *SX = F->createSlice("sliceX", X, {2, 0}, {3, 3});
  auto *RSX = F->createReshape("reshapeSX", SX, {3});
  auto *SSX = F->createSlice("sliceSliceX", SX, {0, 2}, {1, 3});
  auto *RSSX = F->createReshape("reshapeSliceSliceX", SSX, {1});

  auto *resultSX = F->createSave("saveSX", SX);
  auto *resultRSX = F->createSave("saveRSX", RSX);
  auto *resultSSX = F->createSave("saveSSX", SSX);
  auto *resultRSSX = F->createSave("saveRSSX", RSSX);

  bindings.allocate(resultSX->getPlaceholder());
  bindings.allocate(resultRSX->getPlaceholder());
  bindings.allocate(resultSSX->getPlaceholder());
  bindings.allocate(resultRSSX->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  // Verify the slice has the same data as the original X.
  auto SXH = bindings.get(resultSX->getPlaceholder())->getHandle<DataType>();
  for (dim_t i = 0; i < 3; i++) {
    EXPECT_NEAR(SXH.at({0, i}), XH.at({2, i}), 1E-5);
  }

  // Verify the reshaped slice has the same data as the slice.
  auto RSXH = bindings.get(resultRSX->getPlaceholder())->getHandle<DataType>();
  for (dim_t i = 0; i < 3; i++) {
    EXPECT_NEAR(SXH.at({0, i}), RSXH.at({i}), 1E-5);
  }

  // Verify the slice of the slice has the same data as the slice.
  auto SSXH = bindings.get(resultSSX->getPlaceholder())->getHandle<DataType>();
  EXPECT_NEAR(SXH.at({0, 2}), SSXH.at({0, 0}), 1E-5);

  // Verify the reshape of the slice of the slice has the same data as the
  // slice of the slice.
  auto RSSXH =
      bindings.get(resultRSSX->getPlaceholder())->getHandle<DataType>();
  EXPECT_NEAR(RSSXH.at({0}), SSXH.at({0, 0}), 1E-5);
}

/// Stack many slices/reshapes together. Some of these may be turned into
/// tensor views stacked onto each other. Test in FloatTy.
TEST_P(OperatorTest, sliceReshape_Float) {
  CHECK_IF_ENABLED();

  testSliceReshape<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Stack many slices/reshapes together. Some of these may be turned into
/// tensor views stacked onto each other. Test in Float16Ty.
TEST_P(OperatorTest, sliceReshape_Float16) {
  CHECK_IF_ENABLED();
  testSliceReshape<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Stack many slices/reshapes together. Some of these may be turned into
/// tensor views stacked onto each other. Test in Int8QTy.
TEST_P(OperatorTest, sliceReshape_Int8) {
  CHECK_IF_ENABLED();
  testSliceReshape<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Stack many slices/reshapes together. Some of these may be turned into
/// tensor views stacked onto each other. Test in Int32QTy.
TEST_P(OperatorTest, sliceReshape_Int32) {
  CHECK_IF_ENABLED();
  testSliceReshape<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32QTy);
}

/// Helper to test Flatten using \p DTy.
template <typename DataType>
static void testFlatten(glow::PlaceholderBindings &bindings, glow::Module &mod,
                        glow::Function *F, glow::ExecutionEngine &EE,
                        ElemKind DTy) {
  auto *tensor4D = createPlaceholderConditionallyQuantized(
      mod, DTy, {3, 2, 4, 3}, "4D", false, "NHWC");
  bindings.allocate(tensor4D)->getHandle<DataType>().randomize(0, 100,
                                                               mod.getPRNG());

  NodeValue reshape4Dto2DAxis1 = F->createFlatten("flat4Dto2Da1", tensor4D, 1);
  EXPECT_EQ(reshape4Dto2DAxis1.dims().size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis1.dims()[0], 3);
  EXPECT_EQ(reshape4Dto2DAxis1.dims()[1], 24);

  NodeValue reshape4Dto2DAxis2 = F->createFlatten("flat4Dto2Da2", tensor4D, 2);
  EXPECT_EQ(reshape4Dto2DAxis2.dims().size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis2.dims()[0], 6);
  EXPECT_EQ(reshape4Dto2DAxis2.dims()[1], 12);

  NodeValue reshape4Dto2DAxis3 = F->createFlatten("flat4Dto2Da3", tensor4D, 3);
  EXPECT_EQ(reshape4Dto2DAxis3.dims().size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis3.dims()[0], 24);
  EXPECT_EQ(reshape4Dto2DAxis3.dims()[1], 3);

  // Now, let us do the fifth (4) axis.
  // This comes straight from caffe2 because flattening is
  // supported for every axis up and including the rank of a tensor.
  // The rank of this tensor is 4, so axis 4 is fine.
  NodeValue reshape4Dto2DAxis4 = F->createFlatten("flat4Dto2Da4", tensor4D, 4);
  EXPECT_EQ(reshape4Dto2DAxis4.dims().size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis4.dims()[0], 72);
  EXPECT_EQ(reshape4Dto2DAxis4.dims()[1], 1);

  // This one is weird because we flatten something that is already flat, but
  // again because flattening is supported for every axis up and including the
  // rank of a tensor, 1D vector means we can flatten it on axis 1.
  auto *tensor1D =
      createPlaceholderConditionallyQuantized(mod, DTy, {15}, "1D", false, "N");
  bindings.allocate(tensor1D)->getHandle<DataType>().randomize(0, 100,
                                                               mod.getPRNG());

  NodeValue reshape1Dto2DAxis1 = F->createFlatten("flat1Dto2D", tensor1D, 1);
  EXPECT_EQ(reshape1Dto2DAxis1.dims().size(), 2);
  EXPECT_EQ(reshape1Dto2DAxis1.dims()[0], 15);
  EXPECT_EQ(reshape1Dto2DAxis1.dims()[1], 1);

  // Save all the reshapes so that the optimizations won't kill the network.
  auto *save1Dto2D = F->createSave("save1Dto2D", reshape1Dto2DAxis1);
  auto *save4Dto2Da1 = F->createSave("save4Dto2Da1", reshape4Dto2DAxis1);
  auto *save4Dto2Da2 = F->createSave("save4Dto2Da2", reshape4Dto2DAxis2);
  auto *save4Dto2Da3 = F->createSave("save4Dto2Da3", reshape4Dto2DAxis3);
  auto *save4Dto2Da4 = F->createSave("save4Dto2Da4", reshape4Dto2DAxis4);

  bindings.allocate(save1Dto2D->getPlaceholder());
  bindings.allocate(save4Dto2Da1->getPlaceholder());
  bindings.allocate(save4Dto2Da2->getPlaceholder());
  bindings.allocate(save4Dto2Da3->getPlaceholder());
  bindings.allocate(save4Dto2Da4->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  // Verify the reshapes have the same data as the original value.
  auto tensor4DH = bindings.get(tensor4D)->getHandle<DataType>();
  auto save4Dto2Da1H =
      bindings.get(save4Dto2Da1->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da1H.raw(i), 1E-5);
  }

  auto save4Dto2Da2H =
      bindings.get(save4Dto2Da2->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da2H.raw(i), 1E-5);
  }

  auto save4Dto2Da3H =
      bindings.get(save4Dto2Da3->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da3H.raw(i), 1E-5);
  }

  auto save4Dto2Da4H =
      bindings.get(save4Dto2Da4->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da4H.raw(i), 1E-5);
  }

  auto tensor1DH = bindings.get(tensor1D)->getHandle<DataType>();
  auto save1Dto2DH =
      bindings.get(save1Dto2D->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 15; i++) {
    EXPECT_NEAR(tensor1DH.raw(i), save1Dto2DH.raw(i), 1E-5);
  }
}

/// Check that the flatten operator produces 2D tensors of the right
/// dimensions, using FloatTy.
TEST_P(OperatorTest, Flatten_FloatTy) {
  CHECK_IF_ENABLED();
  testFlatten<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Check that the flatten operator produces 2D tensors of the right
/// dimensions, using Float16Ty.
TEST_P(OperatorTest, Flatten_Float16Ty) {
  CHECK_IF_ENABLED();
  testFlatten<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Check that the flatten operator produces 2D tensors of the right
/// dimensions, using Int8QTy.
TEST_P(OperatorTest, Flatten_Int8) {
  CHECK_IF_ENABLED();
  testFlatten<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Check that div on Int64ITy/size_t works.
TEST_P(OperatorTest, DivSizeT) {
  CHECK_IF_ENABLED();

  auto *LHS = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 2}, "LHS", false);
  auto *RHS = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 2}, "RHS", false);
  auto LHSH = bindings_.allocate(LHS)->getHandle<int64_t>();
  auto RHSH = bindings_.allocate(RHS)->getHandle<int64_t>();

  LHSH = {10, 20, 30, 40, 50, 60};
  RHSH = {2, 20, 100, 41, 3, 59};

  auto *R = F_->createDiv("div", LHS, RHS);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle<int64_t>();

  for (dim_t i = 0; i < 3; i++) {
    for (dim_t j = 0; j < 2; j++) {
      EXPECT_EQ(LHSH.at({i, j}) / RHSH.at({i, j}), H.at({i, j}));
    }
  }
}

TEST_P(OperatorTest, SigmoidCrossEntropyWithLogits) {
  CHECK_IF_ENABLED();

  /*
    LOGITS  = [
      [
        [1.0, 1.2, -0.5],
        [0.1, 0.6, 0.5],
      ],
      [
        [-0.1, -2., 0.3],
        [1, 2, 3],
      ],
    ]
    TARGETS = [
      [
        [0.7, 0.7, 0.7],
        [-0.7, -0.99, 1.0],
      ],
      [
        [0, 0, 0],
        [1, 2, 3],
      ],
    ]
    OUTPUT = [
      [ 0.68687367,  0.97332054],
      [ 0.5418933,  -2.50374103],
    ]
  */
  auto *logits =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 3}, "logits", false);
  auto *targets =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 3}, "targets", false);

  bindings_.allocate(logits)->getHandle() = {
      1.0f, 1.2f, -0.5f, 0.1f, 0.6f, 0.5f, -0.1f, -2.f, 0.3f, 1.f, 2.f, 3.f};
  bindings_.allocate(targets)->getHandle() = {
      0.7f, 0.7f, 0.7f, -0.7f, -0.99f, 1.0f, 0.f, 0.f, 0.f, 1.f, 2.f, 3.f};

  auto *R = F_->createSigmoidCrossEntropyWithLogits("SCEL", logits, targets);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 2});
  expected.getHandle() = {
      0.68687367f,
      0.97332054f,
      0.5418933f,
      -2.50374103f,
  };

  EXPECT_TRUE(expected.isEqual(*bindings_.get(result->getPlaceholder())));
}

/// Test the InsertTensor node works correctly.
TEST_P(OperatorTest, insertTensorTest) {
  CHECK_IF_ENABLED();

  // 0 0 0 0 0 0
  // 0 0 0 0 0 0
  // 0 0 0 0 0 0
  // 0 0 0 0 0 0
  auto *SN0 = mod_.createPlaceholder(ElemKind::Int64ITy, {4, 6}, "SN0", false);
  bindings_.allocate(SN0)->init(Tensor::InitKind::Broadcast, 0, mod_.getPRNG());

  // 1 1
  // 1 1
  auto *SN1 = mod_.createPlaceholder(ElemKind::Int64ITy, {2, 2}, "SN1", false);
  bindings_.allocate(SN1)->init(Tensor::InitKind::Broadcast, 1, mod_.getPRNG());

  // 0 0 0 0 0 0
  // 0 1 1 1 1 0
  // 0 1 1 1 1 0
  // 0 0 0 0 0 0
  Node *IN = F_->createInsertTensor("insert", SN0, SN1, /* start */ {1, 1},
                                    /* count */ 2, /* axis */ 1);
  SaveNode *result = F_->createSave("result", IN);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  // Verify the output looks as expected (pictured above).
  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<int64_t>();
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 6; j++) {
      int64_t expected = 1;
      if (i == 0 || i == 3 || j == 0 || j == 5)
        expected = 0;
      EXPECT_EQ(resultH.at({i, j}), expected);
    }
  }
}

static FunctionTensorPair
createAndInitBasicRowwiseFCTest(glow::PlaceholderBindings &bindings,
                                glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // In this test we subtract the outputs of a row-wise quantized FC and a
  // floating-point FC and ensure that the error is below some low value.
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {2, 100}, "in", false);
  auto *fc = F->createFullyConnected(bindings, "FC", input, 5);

  auto *weights = llvm::cast<Placeholder>(fc->getWeights());
  auto *bias = llvm::cast<Placeholder>(fc->getBias());

  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.get(bias)->getHandle().randomize(0, 0.1, mod.getPRNG());
  bindings.get(weights)->getHandle().randomize(-1.1, 1.1, mod.getPRNG());

  auto *res = F->createSave("save", fc);
  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Test Int8 RowwiseQuantizedFullyConnected Node with Int8 bias.
TEST_P(OperatorStatelessTest, rowwiseQuantizedFCTest_Int8_BiasInt8) {
  ENABLED_BACKENDS(Interpreter, CPU);
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicRowwiseFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.06f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ true, quantization::Schema::Asymmetric,
      ElemKind::Int8QTy);
}

/// Test Int8 RowwiseQuantizedFullyConnected Node with Int32 bias.
TEST_P(OperatorStatelessTest, rowwiseQuantizedFCTest_Int8_BiasInt32) {
  ENABLED_BACKENDS(Interpreter, CPU);
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicRowwiseFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.06f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ true, quantization::Schema::Asymmetric,
      ElemKind::Int32QTy);
}

/// Test RowwiseQuantizedFullyConnected Node with Symmetric quantization.
TEST_P(OperatorStatelessTest, rowwiseQuantizedFCTestSymmetric) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicRowwiseFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.07f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ true, quantization::Schema::Symmetric);
}

static FunctionTensorPair
createAndInitBasicSLWSTest(glow::PlaceholderBindings &bindings,
                           glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  /*
    DATA  =   [2.0, -0.5, 13]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  auto *data = mod.createPlaceholder(ElemKind::FloatTy, {3}, "data", false);
  auto *weights =
      mod.createPlaceholder(ElemKind::FloatTy, {8}, "weights", false);
  auto *indices = mod.createPlaceholder(IndexElemKind, {8}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings.allocate(data)->getHandle() = {
      2.0,
      -0.5,
      13,
  };
  bindings.allocate(weights)->getHandle() = {
      3, 1, 0, 0, 0, 0, 2, -0.5,
  };
  bindings.allocate(indices)->getHandle<sdim_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *SLWS = F->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                                 lengths);
  auto *res = F->createSave("save", SLWS);
  ::glow::convertPlaceholdersToConstants(
      F, bindings, {indices, lengths, res->getPlaceholder()});
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Test RowwiseQuantizedSLWS Node.
TEST_P(OperatorStatelessTest, rowwiseQuantizedSLWSTest) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicSLWSTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.01f,
                            parCloneCountOpt,
                            /* convertToRowwiseQuantization */ true);
}

static SaveNode *setupBucketNode(Function *F, PlaceholderBindings &bindings,
                                 Placeholder *input,
                                 const std::string &suffix) {
  std::vector<float> boundaries = {0.1, 2.5};

  auto *bucketize =
      F->createBucketizeNode("bucketize" + suffix, input, boundaries);
  auto *save = F->createSave("save" + suffix, bucketize);
  bindings.allocate(save->getPlaceholder());
  return save;
}

/// Check the correctness of the bucketize operator.
TEST_P(OperatorTest, Bucketize) {
  CHECK_IF_ENABLED();

  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>() = {2.0, 4.0, 1.0};
  auto *save1 =
      setupBucketNode(F_, bindings_, input1, /* suffix */ std::to_string(1));

  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "input2", false);
  bindings_.allocate(input2)->getHandle<float>() = {2.0, 3.0, 4.0,
                                                    1.0, 2.0, 5.0};
  auto *save2 =
      setupBucketNode(F_, bindings_, input2, /* suffix */ std::to_string(2));

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  // Check the result of the first op:
  Tensor *result1 = bindings_.get(save1->getPlaceholder());
  Tensor expected1(ElemKind::Int32ITy, {3});
  expected1.getHandle<int32_t>() = {1, 2, 1};
  EXPECT_TRUE(expected1.isEqual(*result1));

  // Check the result of the second op:
  Tensor *result2 = bindings_.get(save2->getPlaceholder());
  Tensor expected2(ElemKind::Int32ITy, {3, 2});
  expected2.getHandle<int32_t>() = {1, 2, 2, 1, 1, 2};
  EXPECT_TRUE(expected2.isEqual(*result2));
}

/// Check the correctness of the SoftMax operator.
/// The semantic of SoftMax is
/// res_i = exp(input_i) / (exp(input_0) + ... + exp(input_N)).
TEST_P(OperatorTest, SoftMax) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 6}, "input", false);
  bindings_.allocate(input)->getHandle<float>() = {1., 3., 2.5, 5., 4., 2.};
  auto *selected =
      mod_.createPlaceholder(IndexElemKind, {1, 1}, "expected", false);
  auto *Pool = F_->createSoftMax("pool", input, selected);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 6});
  // Expected results are:
  // sum = exp(input_0) + ... + exp(input_N) = ~245.387
  // res_0 = exp(1) / sum = ~0.011
  // res_1 = exp(3) / sum = ~0.082
  // And so on.
  out.getHandle<float>() = {0.011f, 0.082f, 0.05f, 0.605f, 0.222f, 0.03f};
  EXPECT_TRUE(out.isEqual(*result, 0.001));
}

/// Check that the softmax operator works properly with FP16.
/// See the test that check the SoftMax operator for more details.
TEST_P(OperatorTest, FP16SoftMax) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 6}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {1., 3., 2.5, 5., 4., 2.};
  auto *selected =
      mod_.createPlaceholder(IndexElemKind, {1, 1}, "expected", false);
  auto *Pool = F_->createSoftMax("pool", input, selected);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 6});
  out.getHandle<float16_t>() = {0.011f, 0.082f, 0.05f, 0.605f, 0.222f, 0.03f};
  EXPECT_TRUE(out.isEqual(*result, 0.001));
}

/// Verify that Quantize, Rescale, Dequantize work correctly together.
static void quantizeSimpleTest(glow::PlaceholderBindings &bindings_,
                               glow::Module &mod_, glow::Function *F_,
                               glow::ExecutionEngine &EE_, ElemKind QTy) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1}, "input", true);
  bindings_.allocate(input)->init(Tensor::InitKind::Broadcast, 21,
                                  mod_.getPRNG());

  auto *Q =
      F_->createQuantize("quant", input, mod_.uniqueType(QTy, {1, 1}, 0.25, 4));
  auto *RS = F_->createRescaleQuantized("rescale", Q,
                                        mod_.uniqueType(QTy, {1, 1}, 0.5, 11));
  auto *D = F_->createDequantize("dequantize", RS);
  auto *save = F_->createSave("ret", D);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EXPECT_EQ(F_->getNodes().size(), 4);
  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);
  EXPECT_EQ(F_->getNodes().size(), 1);

  auto RH = result->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 21.0, 0.001);
}

TEST_P(OperatorTest, QuantizeSimpleInt8) {
  CHECK_IF_ENABLED();
  quantizeSimpleTest(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}
TEST_P(OperatorTest, QuantizeSimpleInt16) {
  CHECK_IF_ENABLED();
  quantizeSimpleTest(bindings_, mod_, F_, EE_, ElemKind::Int16QTy);
}
TEST_P(OperatorTest, QuantizeSimpleInt32) {
  CHECK_IF_ENABLED();
  quantizeSimpleTest(bindings_, mod_, F_, EE_, ElemKind::Int32QTy);
}

TEST_P(OperatorTest, LengthsToRanges) {
  CHECK_IF_ENABLED();

  /*
    LENGTHS = [1, 3, 0, 2]
    OUTPUT =  [[0, 1], [1, 3], [4, 0], [4, 2]]
  */
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings_.allocate(lengths)->getHandle<int32_t>() = {1, 3, 0, 2};

  auto *R = F_->createLengthsToRanges("LTR", lengths);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::Int32ITy, {4, 2});
  expected.getHandle<int32_t>() = {
      0, 1, 1, 3, 4, 0, 4, 2,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

/// Test that LengthsRangeFill works.
TEST_P(OperatorTest, LengthsRangeFill) {
  CHECK_IF_ENABLED();

  /*
    LENGTHS = [4, 3, 1]
    OUTPUT =  [0, 1, 2, 3, 0, 1, 2, 0]
  */
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {3}, "lengths", false);

  bindings_.allocate(lengths)->getHandle<int32_t>() = {4, 3, 1};

  auto *LRF = F_->createLengthsRangeFill("LRF", lengths, /* maxOutputSize */ 8);
  auto *S = F_->createSave("save", LRF);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::Int32ITy, {8});
  expected.getHandle<int32_t>() = {0, 1, 2, 3, 0, 1, 2, 0};

  EXPECT_TRUE(expected.isEqual(result));
}

/// Helper for testing BatchOneHot with different \p DTy.
template <typename DataType>
void batchOneHotTest(glow::PlaceholderBindings &bindings, glow::Module &mod,
                     glow::Function *F, glow::ExecutionEngine &EE,
                     ElemKind DTy) {
  /*
    DATA = [[5, 0], [11, 3], [0, 5]]
    LENGTHS = [4, 2]
    VALUES = [5, 0, 11, 0, 5, 0]
    OUTPUT =  [[1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0]]
  */
  auto *data =
      createPlaceholderConditionallyQuantized(mod, DTy, {3, 2}, "data", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {2}, "lengths", false, "N");
  auto *values = createPlaceholderConditionallyQuantized(mod, DTy, {6},
                                                         "values", false, "N");

  bindings.allocate(data)->getHandle<DataType>() = {5, 0, 11, 3, 0, 5};
  bindings.allocate(lengths)->getHandle<int32_t>() = {4, 2};
  bindings.allocate(values)->getHandle<DataType>() = {5, 0, 11, 0, 5, 0};

  auto *R = F->createBatchOneHot("BOH", data, lengths, values);
  auto *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  auto expected = createTensorConditionallyQuantized(DTy, {3, 6});
  expected.getHandle<DataType>() = {
      1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

/// Test BatchOneHot with Float data and Int32 Lengths.
TEST_P(OperatorTest, BatchOneHotDataFloat) {
  CHECK_IF_ENABLED();
  batchOneHotTest<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test BatchOneHot with Float16 data and Int32 Lengths
TEST_P(OperatorTest, BatchOneHotDataFloat16) {
  CHECK_IF_ENABLED();
  batchOneHotTest<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test BatchOneHot with Int64 data and Int32 Lengths.
TEST_P(OperatorTest, BatchOneHotDataInt64) {
  CHECK_IF_ENABLED();
  batchOneHotTest<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test BatchOneHot with Int32 data and Int32 Lengths.
TEST_P(OperatorTest, BatchOneHotDataInt32) {
  CHECK_IF_ENABLED();
  batchOneHotTest<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Test BatchOneHot with Int8 data and Int32 Lengths.
TEST_P(OperatorTest, BatchOneHotDataInt8) {
  CHECK_IF_ENABLED();
  batchOneHotTest<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Modulo with Int64 Tensors with SignFollowDivisor off.
TEST_P(OperatorTest, ModuloInt64NoSignFollow) {
  CHECK_IF_ENABLED();

  auto *src = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 5}, "src", false);
  auto srcH = bindings_.allocate(src)->getHandle<int64_t>();

  srcH = {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  int64_t divisor = 3;
  bool signFollowDivisor = false;

  auto *modulo = F_->createModulo("mod", src, divisor, signFollowDivisor);
  auto *result = F_->createSave("save", modulo);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<int64_t>();

  std::vector<int64_t> expectedResults = {-1, 0, -2, -1, 0, -2, -1, 0,
                                          1,  2, 0,  1,  2, 0,  1};
  ASSERT_EQ(expectedResults.size(), resultH.size());

  for (size_t i = 0, end = expectedResults.size(); i < end; ++i) {
    EXPECT_EQ(resultH.raw(i), expectedResults.at(i));
  }
}

/// Modulo with Int64 Tensors with SignFollowDivisor on.
TEST_P(OperatorTest, ModuloInt64SignFollow) {
  CHECK_IF_ENABLED();

  auto *src = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 5}, "src", false);
  auto srcH = bindings_.allocate(src)->getHandle<int64_t>();

  srcH = {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  int64_t divisor = 3;
  bool signFollowDivisor = true;

  auto *modulo = F_->createModulo("mod", src, divisor, signFollowDivisor);
  auto *result = F_->createSave("save", modulo);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<int64_t>();

  std::vector<int64_t> expectedResults = {2, 0, 1, 2, 0, 1, 2, 0,
                                          1, 2, 0, 1, 2, 0, 1};
  ASSERT_EQ(expectedResults.size(), resultH.size());

  for (size_t i = 0, end = expectedResults.size(); i < end; ++i) {
    EXPECT_EQ(resultH.raw(i), expectedResults.at(i));
  }
}

/// Modulo with Int32 Tensors with SignFollowDivisor off.
TEST_P(OperatorTest, ModuloInt32NoSignFollow) {
  CHECK_IF_ENABLED();
#define TENSORTYPE int32_t
  auto *src = mod_.createPlaceholder(ElemKind::Int32ITy, {3, 5}, "src", false);
  auto srcH = bindings_.allocate(src)->getHandle<int32_t>();

  srcH = {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  int64_t divisor = 3;
  bool signFollowDivisor = false;

  auto *modulo = F_->createModulo("mod", src, divisor, signFollowDivisor);
  auto *result = F_->createSave("save", modulo);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<int32_t>();

  std::vector<int32_t> expectedResults = {-1, 0, -2, -1, 0, -2, -1, 0,
                                          1,  2, 0,  1,  2, 0,  1};
  ASSERT_EQ(expectedResults.size(), resultH.size());

  for (size_t i = 0, end = expectedResults.size(); i < end; ++i) {
    EXPECT_EQ(resultH.raw(i), expectedResults.at(i));
  }
}

/// Modulo with Int32 Tensors with SignFollowDivisor off.
TEST_P(OperatorTest, ModuloInt32SignFollow) {
  CHECK_IF_ENABLED();

  auto *src = mod_.createPlaceholder(ElemKind::Int32ITy, {3, 5}, "src", false);
  auto srcH = bindings_.allocate(src)->getHandle<int32_t>();

  srcH = {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  int64_t divisor = 3;
  bool signFollowDivisor = true;

  auto *modulo = F_->createModulo("mod", src, divisor, signFollowDivisor);
  auto *result = F_->createSave("save", modulo);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<int32_t>();

  std::vector<int32_t> expectedResults = {2, 0, 1, 2, 0, 1, 2, 0,
                                          1, 2, 0, 1, 2, 0, 1};
  ASSERT_EQ(expectedResults.size(), resultH.size());

  for (size_t i = 0, end = expectedResults.size(); i < end; ++i) {
    EXPECT_EQ(resultH.raw(i), expectedResults.at(i));
  }
}

/// Helper to test DotProduct1D using \p DTy.
template <typename DataType>
static void testDotProduct1D(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  // Input tensors.
  constexpr std::size_t kDataSize = 10;
  auto *X = createPlaceholderConditionallyQuantized(mod, DTy, {kDataSize}, "X",
                                                    false, "N");
  auto *Y = createPlaceholderConditionallyQuantized(mod, DTy, {kDataSize}, "Y",
                                                    false, "N");
  auto XH = bindings.allocate(X)->getHandle<DataType>();
  auto YH = bindings.allocate(Y)->getHandle<DataType>();

  // Fill inputs with random values.
  XH.randomize(-10.0, 10.0, mod.getPRNG());
  YH.randomize(-10.0, 10.0, mod.getPRNG());

  // Compute expected output.
  auto expected = createTensorConditionallyQuantized(DTy, {kDataSize});
  auto expectedH = expected.getHandle<DataType>();

  for (dim_t i = 0; i < kDataSize; ++i) {
    expectedH.at({i}) = XH.at({i}) * YH.at({i});
  }

  // Compile and run the model.
  auto *dotProduct = F->createDotProduct("prod", X, Y);
  auto *result = F->createSave("save", dotProduct);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto actualH = bindings.get(result->getPlaceholder())->getHandle<DataType>();

  // Check that the output tensor is the same as the expected output.
  EXPECT_EQ(actualH.size(), expectedH.size());
  for (std::size_t i = 0; i < actualH.size(); ++i) {
    EXPECT_NEAR(actualH.raw(i), expectedH.raw(i), 0.00001);
  }
}

/// Test a DotProduct operator with 1D inputs, using FloatTy.
TEST_P(OperatorTest, dotProduct1D_Float) {
  CHECK_IF_ENABLED();
  testDotProduct1D<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test a DotProduct operator with 1D inputs, using Float16Ty.
TEST_P(OperatorTest, dotProduct1D_Float16) {
  CHECK_IF_ENABLED();
  testDotProduct1D<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test a DotProduct operator with 1D inputs, using Int8Ty.
TEST_P(OperatorTest, dotProduct1D_Int8) {
  CHECK_IF_ENABLED();
  testDotProduct1D<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

// Test an ElementwiseLinear operator with both axis = 0 and axis = 1
// arguments.
TEST_P(OperatorTest, elementwiseLinear) {
  CHECK_IF_ENABLED();

  constexpr std::size_t kRows = 10;
  constexpr std::size_t kCols = 20;

  // Create and allocate input placeholders.
  auto *X =
      mod_.createPlaceholder(ElemKind::FloatTy, {kCols, kRows}, "X", false);
  auto *w = mod_.createPlaceholder(ElemKind::FloatTy, {kCols}, "w", false);
  auto *b = mod_.createPlaceholder(ElemKind::FloatTy, {kCols}, "b", false);

  auto XH = bindings_.allocate(X)->getHandle();
  auto wH = bindings_.allocate(w)->getHandle();
  auto bH = bindings_.allocate(b)->getHandle();

  // Fill inputs with random values.
  XH.randomize(-3.0, 3.0, mod_.getPRNG());
  wH.randomize(-3.0, 3.0, mod_.getPRNG());
  bH.randomize(-3.0, 3.0, mod_.getPRNG());

  // Create two separate models to test behaviour when axis = 0 and axis = 1.
  // For the test with axis = 0, the 0th dimension of X, w, and b must match.
  auto *elementwiseLinearAxisZero =
      F_->createElementwiseLinear("elAxisZero", X, w, b, /*axis=*/0);
  auto *resultAxisZero =
      F_->createSave("saveAxisZero", elementwiseLinearAxisZero);
  bindings_.allocate(resultAxisZero->getPlaceholder());

  // For the test with axis = 1, the 1st dimension of X must match the 0th
  // dimension of w and b must match, so a transpose is needed.
  auto *XT = F_->createTranspose("XT", X, {1, 0});
  auto *elementwiseLinearAxisOne =
      F_->createElementwiseLinear("elAxisOne", XT, w, b, /*axis=*/1);
  auto *resultAxisOne = F_->createSave("saveAxisOne", elementwiseLinearAxisOne);
  bindings_.allocate(resultAxisOne->getPlaceholder());

  // Compile and run the model.
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resAxisZeroH =
      bindings_.get(resultAxisZero->getPlaceholder())->getHandle();
  auto resAxisOneH =
      bindings_.get(resultAxisOne->getPlaceholder())->getHandle();

  // Results should be the same shape as X/XT.
  ASSERT_EQ(resAxisZeroH.size(), XH.size());
  ASSERT_EQ(resAxisOneH.size(), (XT->getResult().getType())->size());

  // Compute the expected output and check that the model outputs match.
  for (dim_t i = 0; i < resAxisZeroH.dims()[0]; ++i) {
    for (dim_t j = 0; j < resAxisZeroH.dims()[1]; ++j) {
      float expected = (XH.at({i, j}) * wH.at({i})) + bH.at({i});
      EXPECT_NEAR(resAxisZeroH.at({i, j}), expected, 0.00001);
      EXPECT_NEAR(resAxisOneH.at({j, i}), expected, 0.00001);
    }
  }
}

/// Helper to test DotProduct2D using \p DTy.
template <typename DataType>
static void testDotProduct2D(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  // Input tensors.
  constexpr std::size_t kRows = 10;
  constexpr std::size_t kCols = 14;
  auto *X = createPlaceholderConditionallyQuantized(mod, DTy, {kRows, kCols},
                                                    "X", false);
  auto *Y = createPlaceholderConditionallyQuantized(mod, DTy, {kRows, kCols},
                                                    "Y", false);
  auto XH = bindings.allocate(X)->getHandle<DataType>();
  auto YH = bindings.allocate(Y)->getHandle<DataType>();

  // Fill inputs with random values.
  XH.randomize(-3.0, 3.0, mod.getPRNG());
  YH.randomize(-3.0, 3.0, mod.getPRNG());

  // Compute expected output.
  auto expected = createTensorConditionallyQuantized(DTy, {kRows});
  auto expectedH = expected.getHandle<DataType>();

  for (dim_t i = 0; i < kRows; ++i) {
    DataType dotProduct = 0.0f;

    // Compute dot product of the i-th row of X and Y.
    for (dim_t j = 0; j < kCols; ++j) {
      dotProduct += (XH.at({i, j}) * YH.at({i, j}));
    }

    expectedH.at({i}) = dotProduct;
  }

  // Compile and run the model.
  auto *dotProduct = F->createDotProduct("prod", X, Y);
  auto *result = F->createSave("save", dotProduct);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto actualH = bindings.get(result->getPlaceholder())->getHandle<DataType>();

  // Check that the output tensor is the same as the expected output.
  EXPECT_EQ(actualH.size(), expectedH.size());
  for (std::size_t i = 0; i < actualH.size(); ++i) {
    EXPECT_NEAR(actualH.raw(i), expectedH.raw(i), 0.00001);
  }
}

// Test a DotProduct operator with 2D inputs, using FloatTy.
TEST_P(OperatorTest, dotProduct2D_Float) {
  CHECK_IF_ENABLED();
  testDotProduct2D<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

// Test a DotProduct operator with 2D inputs, using Float16Ty.
TEST_P(OperatorTest, dotProduct2D_Float16) {
  CHECK_IF_ENABLED();
  testDotProduct2D<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

// Test a DotProduct operator with 2D inputs, using Int8QTy.
TEST_P(OperatorTest, dotProduct2D_Int8) {
  CHECK_IF_ENABLED();
  testDotProduct2D<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Helper to test BatchBoxCox using \p DTy.
template <typename DataType>
static void testBatchBoxCox(glow::PlaceholderBindings &bindings,
                            glow::Module &mod, glow::Function *F,
                            glow::ExecutionEngine &EE, ElemKind DTy,
                            float allowedError = 0.0001f) {
  // Input tensors.
  const size_t kRows = 10;
  const size_t kCols = 5;
  auto *data = mod.createPlaceholder(DTy, {kRows, kCols}, "data",
                                     /* isTrainable */ false);
  auto *lambda1 = mod.createPlaceholder(DTy, {kCols}, "lambda1",
                                        /* isTrainable */ false);
  auto *lambda2 = mod.createPlaceholder(DTy, {kCols}, "lambda2",
                                        /* isTrainable */ false);
  auto dataH = bindings.allocate(data)->getHandle<DataType>();
  auto lambda1H = bindings.allocate(lambda1)->getHandle<DataType>();
  auto lambda2H = bindings.allocate(lambda2)->getHandle<DataType>();

  // Fill inputs with random values.
  dataH.randomize(0.0, 5.0, mod.getPRNG());
  lambda1H.randomize(1.0, 2.0, mod.getPRNG());
  lambda2H.randomize(1.0, 2.0, mod.getPRNG());

  // Zero out every other element to lambda1 to test that case of the transform.
  for (dim_t i = 0; i < kCols; i += 2) {
    lambda1H.at({i}) = 0;
  }

  const float epsilon = std::is_same<float, DataType>::value
                            ? std::numeric_limits<float>::min()
                            : 1e-6f;

  // Construct the graph for the backend to run.
  auto *BBC = F->createBatchBoxCox("bbc", data, lambda1, lambda2, epsilon);
  auto *save = F->createSave("save", BBC);
  auto resultH =
      bindings.allocate(save->getPlaceholder())->getHandle<DataType>();

  // Compile and run the model, setting results in tensor backed by resultH.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compute expected output here on the host to compare results.
  Tensor expected(DTy, {kRows, kCols});
  auto expectedH = expected.getHandle<DataType>();

  for (dim_t i = 0; i < kRows; ++i) {
    for (dim_t j = 0; j < kCols; ++j) {
      float d = dataH.at({i, j});
      float l1 = lambda1H.at({j});
      float l2 = lambda2H.at({j});

      // Compute elementwise Box-Cox transform.
      float tmp = std::max(d + l2, 1e-6f);
      if (l1 == 0) {
        // Clip argument to log and pow at 1e-6 to avoid saturation.
        expectedH.at({i, j}) = std::log(tmp);
      } else {
        expectedH.at({i, j}) = (std::pow(tmp, l1) - 1) / l1;
      }
    }
  }

  // Check that the output tensor is the same as the expected output.
  for (size_t i = 0; i < resultH.size(); ++i) {
    EXPECT_NEAR(resultH.raw(i), expectedH.raw(i), allowedError);
  }
}

/// Test that the BatchBoxCox operator works as expected in FloatTy.
TEST_P(OperatorTest, BatchBoxCox_Float) {
  CHECK_IF_ENABLED();
  testBatchBoxCox<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.001f);
}

/// Test that the BatchBoxCox operator works as expected in Float16Ty.
TEST_P(OperatorTest, BatchBoxCox_Float16) {
  CHECK_IF_ENABLED();
  testBatchBoxCox<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                             0.01f);
}

/// Test that Arithmetic ops work.
#define TEST_ARITH_OP_FLOAT(OP_NAME_, OP_)                                     \
  TEST_P(OperatorTest, OP_NAME_##ArithFloatTest) {                             \
    CHECK_IF_ENABLED();                                                        \
    constexpr size_t size = 50;                                                \
    auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {size}, "A", false);   \
    auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {size}, "B", false);   \
    auto *AT = bindings_.allocate(A);                                          \
    auto *BT = bindings_.allocate(B);                                          \
    auto AH = AT->getHandle();                                                 \
    auto BH = BT->getHandle();                                                 \
    AH.randomize(-10.0f, 10.0f, mod_.getPRNG());                               \
    BH.randomize(0.01f, 10.0f, mod_.getPRNG());                                \
                                                                               \
    auto *N = F_->create##OP_NAME_("op", A, B);                                \
    auto *save = F_->createSave("save", N);                                    \
    auto resultH = bindings_.allocate(save->getPlaceholder())->getHandle();    \
                                                                               \
    EE_.compile(CompilationMode::Infer);                                       \
    EE_.run(bindings_);                                                        \
                                                                               \
    for (size_t i = 0; i < size; i++) {                                        \
      EXPECT_FLOAT_EQ(resultH.raw(i), OP_(AH.raw(i), BH.raw(i)));              \
    }                                                                          \
  }

TEST_ARITH_OP_FLOAT(Add, [](float a, float b) { return a + b; })
TEST_ARITH_OP_FLOAT(Sub, [](float a, float b) { return a - b; })
TEST_ARITH_OP_FLOAT(Mul, [](float a, float b) { return a * b; })
TEST_ARITH_OP_FLOAT(Div, [](float a, float b) { return a / b; })
TEST_ARITH_OP_FLOAT(Min, [](float a, float b) { return std::min(a, b); })
TEST_ARITH_OP_FLOAT(Max, [](float a, float b) { return std::max(a, b); })

/// Helper to test ConvertTo casting from \p STy to \p DTy.
template <typename SourceType, typename DestType>
static void testConvertTo(glow::PlaceholderBindings &bindings_,
                          glow::Module &mod_, glow::Function *F_,
                          glow::ExecutionEngine &EE_, ElemKind STy,
                          ElemKind DTy) {
  // Input tensor in source type.
  dim_t shape[] = {5, 3, 20};
  auto *data = mod_.createPlaceholder(STy, shape, "data",
                                      /* isTrainable */ false);
  auto dataH = bindings_.allocate(data)->getHandle<SourceType>();
  dataH.randomize(-1000, 1000, mod_.getPRNG());

  // Construct the graph for the backend to run, converting to dest type.
  auto OT = mod_.uniqueType(DTy, shape);
  auto *convert = F_->createConvertTo("convert", data, OT);
  auto *save = F_->createSave("save", convert);
  auto resultH =
      bindings_.allocate(save->getPlaceholder())->getHandle<DestType>();

  // Compile and run the model, setting results in tensor backed by resultH.
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  // Compute expected output here on the host to compare results.
  Tensor expected(DTy, shape);
  auto expectedH = expected.getHandle<DestType>();
  for (size_t i = 0, e = expectedH.size(); i < e; ++i) {
    expectedH.raw(i) = static_cast<DestType>(dataH.raw(i));
  }

  // Check that the output tensor is the same as the expected output.
  for (size_t i = 0, e = resultH.size(); i < e; i++) {
    const DestType exp = expectedH.raw(i);
    const DestType res = resultH.raw(i);
    if (DTy == ElemKind::FloatTy) {
      EXPECT_FLOAT_EQ(exp, res);
    } else {
      EXPECT_EQ(exp, res);
    }
  }
}

/// Test that ConvertTo operator casts correctly from one type to another.
#define TEST_CONVERT_TO(T_FROM, T_TO, DTY_FROM, DTY_TO)                        \
  TEST_P(OperatorTest, ConvertFrom_##DTY_FROM##_To_##DTY_TO) {                 \
    CHECK_IF_ENABLED();                                                        \
    testConvertTo<T_FROM, T_TO>(bindings_, mod_, F_, EE_, ElemKind::DTY_FROM,  \
                                ElemKind::DTY_TO);                             \
  }
TEST_CONVERT_TO(float, float, FloatTy, FloatTy)
TEST_CONVERT_TO(float, float16_t, FloatTy, Float16Ty)
TEST_CONVERT_TO(float, int32_t, FloatTy, Int32ITy)
TEST_CONVERT_TO(float, int64_t, FloatTy, Int64ITy)
TEST_CONVERT_TO(float16_t, float, Float16Ty, FloatTy)
TEST_CONVERT_TO(float16_t, float16_t, Float16Ty, Float16Ty)
TEST_CONVERT_TO(float16_t, int32_t, Float16Ty, Int32ITy)
TEST_CONVERT_TO(float16_t, int64_t, Float16Ty, Int64ITy)
TEST_CONVERT_TO(int32_t, float, Int32ITy, FloatTy)
TEST_CONVERT_TO(int32_t, float16_t, Int32ITy, Float16Ty)
TEST_CONVERT_TO(int32_t, int32_t, Int32ITy, Int32ITy)
TEST_CONVERT_TO(int32_t, int64_t, Int32ITy, Int64ITy)
TEST_CONVERT_TO(int64_t, float, Int64ITy, FloatTy)
TEST_CONVERT_TO(int64_t, float16_t, Int64ITy, Float16Ty)
TEST_CONVERT_TO(int64_t, int32_t, Int64ITy, Int32ITy)
TEST_CONVERT_TO(int64_t, int64_t, Int64ITy, Int64ITy)

#undef TEST_CONVERT_TO

/// Helper to test ConvertTo casting from \p STy to \p DTy and back.
template <typename SourceType, typename DestType>
static void testConvertToAndBack(glow::PlaceholderBindings &bindings_,
                                 glow::Module &mod_, glow::Function *F_,
                                 glow::ExecutionEngine &EE_, ElemKind STy,
                                 ElemKind DTy, bool castIsNoOp) {
  // Input tensor in source type.
  dim_t shape[] = {5, 3, 20};
  auto *data = mod_.createPlaceholder(STy, shape, "data",
                                      /* isTrainable */ false);
  auto dataH = bindings_.allocate(data)->getHandle<SourceType>();
  dataH.randomize(-1000, 1000, mod_.getPRNG());

  // Construct the graph for the backend to run, converting to dest type and
  // back.
  auto IT = mod_.uniqueType(STy, shape);
  auto OT = mod_.uniqueType(DTy, shape);
  auto *convert = F_->createConvertTo("convert_forth", data, OT);
  auto *convertBack = F_->createConvertTo("convert_back", convert, IT);
  auto *save = F_->createSave("save", convertBack);
  auto resultH =
      bindings_.allocate(save->getPlaceholder())->getHandle<SourceType>();

  // Compile and run the model, setting results in tensor backed by resultH.
  EXPECT_EQ(F_->getNodes().size(), 3);
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  EXPECT_EQ(F_->getNodes().size(), size_t(castIsNoOp ? 1 : 3));

  for (size_t i = 0, e = resultH.size(); i < e; i++) {
    const SourceType res = resultH.raw(i);
    const SourceType expected =
        static_cast<SourceType>(static_cast<DestType>(dataH.raw(i)));
    EXPECT_EQ(res, expected);
  }
}

/// Test that ConvertTo operator casts correctly from one type to another.
#define TEST_CAST_2WAYS(T_FROM, T_TO, DTY_FROM, DTY_TO, NOOP_CAST)             \
  TEST_P(OperatorTest, ConvertFrom_##DTY_FROM##_To_##DTY_TO##_AndBack) {       \
    CHECK_IF_ENABLED();                                                        \
    testConvertToAndBack<T_FROM, T_TO>(bindings_, mod_, F_, EE_,               \
                                       ElemKind::DTY_FROM, ElemKind::DTY_TO,   \
                                       NOOP_CAST);                             \
  }
TEST_CAST_2WAYS(float, float, FloatTy, FloatTy, /* castIsNoOp */ true)
TEST_CAST_2WAYS(float, float16_t, FloatTy, Float16Ty, /* castIsNoOp */ false)
TEST_CAST_2WAYS(float, int32_t, FloatTy, Int32ITy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(float, int64_t, FloatTy, Int64ITy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(float16_t, float, Float16Ty, FloatTy, /* castIsNoOp */ true)
TEST_CAST_2WAYS(float16_t, float16_t, Float16Ty, Float16Ty,
                /* castIsNoOp */ true)
TEST_CAST_2WAYS(float16_t, int32_t, Float16Ty, Int32ITy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(float16_t, int64_t, Float16Ty, Int64ITy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(int32_t, float, Int32ITy, FloatTy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(int32_t, float16_t, Int32ITy, Float16Ty, /* castIsNoOp */ false)
TEST_CAST_2WAYS(int32_t, int32_t, Int32ITy, Int32ITy, /* castIsNoOp */ true)
TEST_CAST_2WAYS(int32_t, int64_t, Int32ITy, Int64ITy, /* castIsNoOp */ true)
TEST_CAST_2WAYS(int64_t, float, Int64ITy, FloatTy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(int64_t, float16_t, Int64ITy, Float16Ty, /* castIsNoOp */ false)
TEST_CAST_2WAYS(int64_t, int32_t, Int64ITy, Int32ITy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(int64_t, int64_t, Int64ITy, Int64ITy, /* castIsNoOp */ true)

#undef TEST_CAST_2WAYS

INSTANTIATE_BACKEND_TEST(OperatorStatelessTest);
INSTANTIATE_BACKEND_TEST(OperatorTest);
